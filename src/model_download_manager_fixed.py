"""
Fixed Model Download Manager - Uses HuggingFace Cache Only
Eliminates duplicate storage by using cache_dir instead of local_dir
"""

import json
import os
import shutil
import signal
import sys
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    from huggingface_hub import (
        HfApi, 
        hf_hub_download, 
        repo_info, 
        snapshot_download,
        HfFolder
    )
    from huggingface_hub.utils import (
        RepositoryNotFoundError, 
        RevisionNotFoundError,
        HfHubHTTPError
    )
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DownloadProgress:
    """Progress tracking for downloads"""
    total_files: int = 0
    completed_files: int = 0
    total_size_bytes: int = 0
    downloaded_bytes: int = 0
    current_file: str = ""
    start_time: float = 0.0
    is_complete: bool = False
    error_message: str = ""


@dataclass
class ModelDownloadConfig:
    """Configuration for model downloads"""
    model_name: str
    cache_dir: Optional[str] = None  # Use cache_dir instead of local_dir
    resume_download: bool = True
    max_workers: int = 4
    chunk_size: int = 8192
    timeout: int = 300
    retry_attempts: int = 3
    verify_integrity: bool = True
    cleanup_on_failure: bool = True
    ignore_patterns: List[str] = None


class FixedModelDownloadManager:
    """
    Fixed model download manager that uses HuggingFace cache only
    Eliminates duplicate storage issues
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "huggingface_hub is required. Install with: pip install huggingface_hub"
            )
        
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
        self.api = HfApi()
        self.download_interrupted = False
        self._progress_callbacks: List[Callable[[DownloadProgress], None]] = []
        self._lock = threading.Lock()
        
        # Supported models with their configurations
        self.supported_models = {
            "Qwen/Qwen-Image": {
                "type": "text-to-image",
                "expected_size_gb": 8,
                "priority": "high",
                "pipeline_class": "AutoPipelineForText2Image",
                "architecture": "MMDiT",
                "ignore_patterns": ["*.md", "*.txt", ".gitattributes"]
            },
            "Qwen/Qwen-Image-Edit": {
                "type": "image-editing",
                "expected_size_gb": 54,
                "priority": "medium",
                "pipeline_class": "DiffusionPipeline",
                "architecture": "MMDiT",
                "ignore_patterns": ["*.md", "*.txt", ".gitattributes"]
            },
            "Qwen/Qwen2-VL-7B-Instruct": {
                "type": "multimodal-language",
                "expected_size_gb": 15,
                "priority": "high",
                "pipeline_class": "Qwen2VLForConditionalGeneration",
                "architecture": "Transformer",
                "capabilities": ["text_understanding", "image_analysis", "prompt_enhancement"],
                "ignore_patterns": ["*.md", "*.txt", ".gitattributes"]
            },
            "Qwen/Qwen2-VL-2B-Instruct": {
                "type": "multimodal-language", 
                "expected_size_gb": 4,
                "priority": "medium",
                "pipeline_class": "Qwen2VLForConditionalGeneration",
                "architecture": "Transformer",
                "capabilities": ["text_understanding", "image_analysis", "prompt_enhancement"],
                "ignore_patterns": ["*.md", "*.txt", ".gitattributes"]
            }
        }
        
        # Setup signal handlers for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interruption signals gracefully"""
        logger.warning(f"Download interrupted by signal {signum}")
        logger.info("Progress has been saved, you can resume later")
        self.download_interrupted = True
    
    def add_progress_callback(self, callback: Callable[[DownloadProgress], None]):
        """Add a progress callback function"""
        self._progress_callbacks.append(callback)
    
    def _notify_progress(self, progress: DownloadProgress):
        """Notify all progress callbacks"""
        for callback in self._progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    def check_model_availability(self, model_name: str) -> Dict[str, Any]:
        """Check if model is available in HuggingFace cache"""
        status = {
            "model_name": model_name,
            "remote_available": False,
            "cache_available": False,
            "cache_complete": False,
            "cache_path": None,
            "remote_size_bytes": 0,
            "cache_size_bytes": 0,
            "missing_files": [],
            "integrity_verified": False,
            "error": None
        }
        
        try:
            # Check remote availability
            try:
                repo_data = repo_info(model_name)
                status["remote_available"] = True
                status["remote_size_bytes"] = sum(
                    file.size for file in repo_data.siblings if file.size
                )
                logger.info(f"✅ Remote model accessible: {model_name}")
                logger.info(f"📊 Remote size: {self._format_size(status['remote_size_bytes'])}")
            except (RepositoryNotFoundError, RevisionNotFoundError) as e:
                status["error"] = f"Repository not found: {e}"
                logger.error(f"❌ Repository not accessible: {model_name}")
                return status
            except Exception as e:
                status["error"] = f"Remote check failed: {e}"
                logger.error(f"❌ Remote check failed: {e}")
                return status
            
            # Check HuggingFace cache
            cache_path = self._get_cache_model_path(model_name)
            if cache_path and cache_path.exists():
                status["cache_available"] = True
                status["cache_path"] = str(cache_path)
                
                # Calculate cache size
                status["cache_size_bytes"] = sum(
                    f.stat().st_size for f in cache_path.rglob("*") if f.is_file()
                )
                
                # Check completeness by comparing with remote files
                if status["remote_available"]:
                    missing_files = []
                    
                    # Find the actual model directory (in snapshots)
                    snapshots_dir = cache_path / "snapshots"
                    if snapshots_dir.exists():
                        snapshots = list(snapshots_dir.iterdir())
                        if snapshots:
                            # Use the latest snapshot
                            latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
                            
                            for file_info in repo_data.siblings:
                                cache_file = latest_snapshot / file_info.rfilename
                                if not cache_file.exists():
                                    missing_files.append(file_info.rfilename)
                                elif cache_file.stat().st_size != file_info.size:
                                    missing_files.append(f"{file_info.rfilename} (size mismatch)")
                    
                    status["missing_files"] = missing_files
                    status["cache_complete"] = len(missing_files) == 0
                    
                    if status["cache_complete"]:
                        logger.info(f"✅ Cache model complete: {model_name}")
                    else:
                        logger.warning(f"⚠️ Cache model incomplete: {len(missing_files)} files missing")
                
                logger.info(f"📁 Cache size: {self._format_size(status['cache_size_bytes'])}")
            else:
                logger.info(f"📁 No cached model found: {model_name}")
        
        except Exception as e:
            status["error"] = f"Availability check failed: {e}"
            logger.error(f"❌ Availability check failed: {e}")
        
        return status
    
    def download_model(
        self, 
        model_name: str, 
        config: Optional[ModelDownloadConfig] = None
    ) -> bool:
        """
        Download a model to HuggingFace cache only (no local directory)
        """
        if model_name not in self.supported_models:
            logger.error(f"❌ Unsupported model: {model_name}")
            return False
        
        # Use default config if none provided
        if config is None:
            config = ModelDownloadConfig(
                model_name=model_name,
                ignore_patterns=self.supported_models[model_name].get("ignore_patterns", [])
            )
        
        logger.info(f"🚀 Starting download to HuggingFace cache: {model_name}")
        logger.info("=" * 60)
        
        # Check current status
        status = self.check_model_availability(model_name)
        if status.get("error"):
            logger.error(f"❌ Cannot download: {status['error']}")
            return False
        
        if status["cache_complete"]:
            logger.info("✅ Model already fully cached!")
            return self.verify_model_integrity(model_name)
        
        # Initialize progress tracking
        progress = DownloadProgress(start_time=time.time())
        
        try:
            return self._download_to_cache(model_name, config, progress)
        
        except KeyboardInterrupt:
            logger.warning("⚠️ Download interrupted by user")
            logger.info("💾 Progress saved, run again to resume")
            return False
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False
    
    def _download_to_cache(
        self, 
        model_name: str, 
        config: ModelDownloadConfig, 
        progress: DownloadProgress
    ) -> bool:
        """Download model to HuggingFace cache only"""
        logger.info(f"📥 Downloading to cache: {model_name}")
        
        try:
            # Use snapshot_download with cache_dir (NOT local_dir)
            logger.info(f"📁 Cache directory: {self.cache_dir}")
            logger.info("⏳ This may take 10-60 minutes depending on model size...")
            logger.info("💡 Download can be resumed if interrupted")
            logger.info("🎯 Using HuggingFace cache - no duplicates!")
            
            downloaded_path = snapshot_download(
                repo_id=model_name,
                cache_dir=config.cache_dir or self.cache_dir,  # Use cache_dir instead of local_dir
                resume_download=config.resume_download,
                local_files_only=False,
                ignore_patterns=config.ignore_patterns,
                tqdm_class=tqdm if TQDM_AVAILABLE else None,
                etag_timeout=config.timeout,
                max_workers=config.max_workers
            )
            
            if self.download_interrupted:
                logger.warning("⚠️ Download was interrupted but can be resumed")
                return False
            
            progress.is_complete = True
            self._notify_progress(progress)
            
            logger.info(f"✅ Download completed!")
            logger.info(f"📁 Model cached at: {downloaded_path}")
            logger.info("🎯 No duplicate storage - using cache only!")
            
            # Verify integrity if requested
            if config.verify_integrity:
                return self.verify_model_integrity(model_name)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache download failed: {e}")
            progress.error_message = str(e)
            self._notify_progress(progress)
            return False
    
    def verify_model_integrity(self, model_name: str) -> bool:
        """Verify the integrity of a cached model"""
        logger.info(f"🔍 Verifying cached model integrity: {model_name}")
        
        try:
            cache_path = self._get_cache_model_path(model_name)
            
            if not cache_path or not cache_path.exists():
                logger.error("❌ Model cache path not found for verification")
                return False
            
            # Find the actual model directory in snapshots
            snapshots_dir = cache_path / "snapshots"
            if not snapshots_dir.exists():
                logger.error("❌ No snapshots directory found in cache")
                return False
            
            snapshots = list(snapshots_dir.iterdir())
            if not snapshots:
                logger.error("❌ No snapshots found in cache")
                return False
            
            # Use the latest snapshot
            latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
            
            # Check for essential files based on model type
            model_config = self.supported_models.get(model_name, {})
            model_type = model_config.get("type", "")
            
            essential_files = []
            if model_type == "multimodal-language":
                essential_files = [
                    "config.json",
                    "generation_config.json", 
                    "tokenizer.json",
                    "tokenizer_config.json"
                ]
            else:  # diffusion models
                essential_files = [
                    "model_index.json",
                    "scheduler/scheduler_config.json",
                    "text_encoder/config.json",
                    "transformer/config.json",
                    "vae/config.json"
                ]
            
            missing_files = []
            for file_name in essential_files:
                file_path = latest_snapshot / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
            
            if missing_files:
                logger.error(f"❌ Missing essential files: {missing_files}")
                return False
            
            # Check for model weight files
            has_weights = any(
                latest_snapshot.rglob("*.safetensors") or 
                latest_snapshot.rglob("*.bin")
            )
            
            if not has_weights:
                logger.error("❌ No model weight files found")
                return False
            
            logger.info("✅ Model integrity verification passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Integrity verification failed: {e}")
            return False
    
    def download_qwen_image(self, force_redownload: bool = False) -> bool:
        """Download the Qwen-Image model for text-to-image generation"""
        model_name = "Qwen/Qwen-Image"
        
        if not force_redownload:
            status = self.check_model_availability(model_name)
            if status["cache_complete"]:
                logger.info("✅ Qwen-Image model already cached")
                return True
        
        config = ModelDownloadConfig(
            model_name=model_name,
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
            verify_integrity=True
        )
        
        return self.download_model(model_name, config)
    
    def download_qwen2_vl(self, model_size: str = "7B", force_redownload: bool = False) -> bool:
        """Download Qwen2-VL model for enhanced multimodal capabilities"""
        if model_size == "7B":
            model_name = "Qwen/Qwen2-VL-7B-Instruct"
        elif model_size == "2B":
            model_name = "Qwen/Qwen2-VL-2B-Instruct"
        else:
            logger.error(f"❌ Unsupported Qwen2-VL model size: {model_size}")
            return False
        
        if not force_redownload:
            status = self.check_model_availability(model_name)
            if status["cache_complete"]:
                logger.info(f"✅ {model_name} already cached")
                return True
        
        config = ModelDownloadConfig(
            model_name=model_name,
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
            verify_integrity=True,
            max_workers=2  # Limit for large models
        )
        
        return self.download_model(model_name, config)
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """Get the path to a cached model"""
        cache_path = self._get_cache_model_path(model_name)
        
        if not cache_path or not cache_path.exists():
            return None
        
        # Find the actual model directory in snapshots
        snapshots_dir = cache_path / "snapshots"
        if snapshots_dir.exists():
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                # Return the latest snapshot path
                latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
                return str(latest_snapshot)
        
        return None
    
    def list_cached_models(self) -> Dict[str, Dict[str, Any]]:
        """List all cached models and their status"""
        models_status = {}
        
        for model_name in self.supported_models:
            status = self.check_model_availability(model_name)
            models_status[model_name] = {
                **self.supported_models[model_name],
                **status
            }
        
        return models_status
    
    def _get_cache_model_path(self, model_name: str) -> Optional[Path]:
        """Get the cache path for a model"""
        try:
            cache_path = Path(self.cache_dir) / f"models--{model_name.replace('/', '--')}"
            return cache_path if cache_path.exists() else None
        except:
            return None
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human readable format"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"


# Convenience functions for easy usage
def download_qwen_image_fixed(force_redownload: bool = False) -> bool:
    """Convenience function to download Qwen-Image model to cache only"""
    manager = FixedModelDownloadManager()
    return manager.download_qwen_image(force_redownload)


def download_qwen2_vl_fixed(model_size: str = "7B", force_redownload: bool = False) -> bool:
    """Convenience function to download Qwen2-VL model to cache only"""
    manager = FixedModelDownloadManager()
    return manager.download_qwen2_vl(model_size, force_redownload)


def get_cached_model_path(model_name: str) -> Optional[str]:
    """Convenience function to get cached model path"""
    manager = FixedModelDownloadManager()
    return manager.get_model_path(model_name)


if __name__ == "__main__":
    # Example usage and testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed Qwen Model Download Manager")
    parser.add_argument("--model", choices=["qwen-image", "qwen2-vl-7b", "qwen2-vl-2b"], 
                       help="Model to download")
    parser.add_argument("--check", type=str, help="Check status of a model")
    parser.add_argument("--list", action="store_true", help="List all cached models")
    parser.add_argument("--force", action="store_true", help="Force redownload")
    parser.add_argument("--path", type=str, help="Get path to cached model")
    
    args = parser.parse_args()
    
    manager = FixedModelDownloadManager()
    
    if args.list:
        models = manager.list_cached_models()
        print("\n📋 Cached Models:")
        print("=" * 60)
        for name, info in models.items():
            status = "✅ Complete" if info.get("cache_complete") else "❌ Missing"
            print(f"{name}: {status} ({info['expected_size_gb']}GB)")
    
    elif args.check:
        status = manager.check_model_availability(args.check)
        print(f"\n📊 Status for {args.check}:")
        print("=" * 40)
        for key, value in status.items():
            print(f"{key}: {value}")
    
    elif args.path:
        path = manager.get_model_path(args.path)
        if path:
            print(f"📁 Cached model path: {path}")
        else:
            print(f"❌ Model not found in cache: {args.path}")
    
    elif args.model:
        if args.model == "qwen-image":
            success = manager.download_qwen_image(args.force)
        elif args.model == "qwen2-vl-7b":
            success = manager.download_qwen2_vl("7B", args.force)
        elif args.model == "qwen2-vl-2b":
            success = manager.download_qwen2_vl("2B", args.force)
        
        if success:
            print("✅ Download completed successfully!")
        else:
            print("❌ Download failed!")
            sys.exit(1)
    
    else:
        print("Use --help to see available options")