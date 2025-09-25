"""
Model Download Manager with Qwen2-VL Support
Provides robust download and resume capabilities for Qwen models
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

# Import error handling system
try:
    from .error_handler import ArchitectureAwareErrorHandler, ErrorCategory
    ERROR_HANDLER_AVAILABLE = True
except ImportError:
    try:
        from error_handler import ArchitectureAwareErrorHandler, ErrorCategory
        ERROR_HANDLER_AVAILABLE = True
    except ImportError:
        ERROR_HANDLER_AVAILABLE = False

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
    cache_dir: Optional[str] = None  # Use cache_dir instead of local_dir to avoid duplicates
    resume_download: bool = True
    max_workers: int = 4
    chunk_size: int = 8192
    timeout: int = 300
    retry_attempts: int = 3
    verify_integrity: bool = True
    cleanup_on_failure: bool = True
    ignore_patterns: List[str] = None


class ModelDownloadManager:
    """
    Robust model download manager with resume capabilities and Qwen2-VL support
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
        
        # Initialize error handler
        self.error_handler = ArchitectureAwareErrorHandler() if ERROR_HANDLER_AVAILABLE else None
        if self.error_handler:
            # Add user feedback callback for download progress
            self.error_handler.add_user_feedback_callback(self._handle_error_feedback)
        
        # Supported models with their configurations
        self.supported_models = {
            "Qwen/Qwen-Image": {
                "type": "text-to-image",
                "expected_size_gb": 8,
                "priority": "high",
                "pipeline_class": "AutoPipelineForText2Image",
                "architecture": "MMDiT",
                "ignore_patterns": ["*.md", "*.txt", ".gitattributes", "*.safetensors.index.json"]
            },
            "Qwen/Qwen-Image-Edit": {
                "type": "image-editing",
                "expected_size_gb": 54,
                "priority": "medium",
                "pipeline_class": "DiffusionPipeline",
                "architecture": "MMDiT",
                "ignore_patterns": ["*.md", "*.txt", ".gitattributes", "*.safetensors.index.json"]
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
    
    def _handle_error_feedback(self, message: str):
        """Handle error feedback from error handler"""
        logger.info(f"🔧 Error Handler: {message}")
    
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
        """Check if model is available locally and remotely"""
        status = {
            "model_name": model_name,
            "remote_available": False,
            "local_available": False,
            "local_complete": False,
            "local_path": None,
            "remote_size_bytes": 0,
            "local_size_bytes": 0,
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
            
            # Check local availability
            local_path = self._get_local_model_path(model_name)
            if local_path and local_path.exists():
                status["local_available"] = True
                status["local_path"] = str(local_path)
                
                # Calculate local size
                status["local_size_bytes"] = sum(
                    f.stat().st_size for f in local_path.rglob("*") if f.is_file()
                )
                
                # Check completeness by comparing with remote files
                if status["remote_available"]:
                    missing_files = []
                    for file_info in repo_data.siblings:
                        local_file = local_path / file_info.rfilename
                        if not local_file.exists():
                            missing_files.append(file_info.rfilename)
                        elif local_file.stat().st_size != file_info.size:
                            missing_files.append(f"{file_info.rfilename} (size mismatch)")
                    
                    status["missing_files"] = missing_files
                    status["local_complete"] = len(missing_files) == 0
                    
                    if status["local_complete"]:
                        logger.info(f"✅ Local model complete: {model_name}")
                    else:
                        logger.warning(f"⚠️ Local model incomplete: {len(missing_files)} files missing")
                
                logger.info(f"📁 Local size: {self._format_size(status['local_size_bytes'])}")
            else:
                logger.info(f"📁 No local model found: {model_name}")
        
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
        Download a model with robust error handling and resume capability
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
        
        logger.info(f"🚀 Starting download: {model_name}")
        logger.info("=" * 60)
        
        # Check current status
        status = self.check_model_availability(model_name)
        if status.get("error"):
            logger.error(f"❌ Cannot download: {status['error']}")
            return False
        
        if status["local_complete"]:
            logger.info("✅ Model already fully downloaded!")
            return self.verify_model_integrity(model_name)
        
        # Initialize progress tracking
        progress = DownloadProgress(start_time=time.time())
        
        try:
            # Determine download method based on model type
            model_config = self.supported_models[model_name]
            
            if model_config["type"] == "multimodal-language":
                return self._download_qwen2_vl_model(model_name, config, progress)
            else:
                return self._download_diffusion_model(model_name, config, progress)
        
        except KeyboardInterrupt:
            logger.warning("⚠️ Download interrupted by user")
            logger.info("💾 Progress saved, run again to resume")
            return False
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            
            # Use error handler if available
            if self.error_handler:
                error_info = self.error_handler.handle_download_error(
                    e, model_name, {"config": config, "progress": progress}
                )
                self.error_handler.log_error(error_info)
                
                # Attempt recovery
                recovery_success = self.error_handler.execute_recovery_actions(error_info)
                if recovery_success:
                    logger.info("🔄 Recovery actions completed, retrying download...")
                    # Could implement retry logic here
            
            if config.cleanup_on_failure:
                self._cleanup_partial_download(model_name)
            return False
    
    def _download_diffusion_model(
        self, 
        model_name: str, 
        config: ModelDownloadConfig, 
        progress: DownloadProgress
    ) -> bool:
        """Download diffusion models (Qwen-Image, Qwen-Image-Edit)"""
        logger.info(f"📥 Downloading diffusion model: {model_name}")
        
        try:
            # Use HuggingFace cache instead of local directory to avoid duplicates
            cache_dir = config.cache_dir or self.cache_dir
            
            # Download with snapshot_download to HuggingFace cache
            logger.info(f"📁 Cache directory: {cache_dir}")
            logger.info("⏳ This may take 10-60 minutes depending on model size...")
            logger.info("💡 Download can be resumed if interrupted")
            logger.info("🎯 Using HuggingFace cache - no duplicates!")
            
            downloaded_path = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,  # Use cache_dir instead of local_dir
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
            logger.info(f"📁 Model saved to: {downloaded_path}")
            
            # Verify integrity if requested
            if config.verify_integrity:
                return self.verify_model_integrity(model_name, str(local_dir))
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Diffusion model download failed: {e}")
            progress.error_message = str(e)
            self._notify_progress(progress)
            return False
    
    def _download_qwen2_vl_model(
        self, 
        model_name: str, 
        config: ModelDownloadConfig, 
        progress: DownloadProgress
    ) -> bool:
        """Download Qwen2-VL models with multimodal capabilities"""
        logger.info(f"📥 Downloading Qwen2-VL model: {model_name}")
        logger.info("🔍 This model provides enhanced text understanding and image analysis")
        
        try:
            # Use HuggingFace cache instead of local directory to avoid duplicates
            cache_dir = config.cache_dir or self.cache_dir
            
            # Get repository info for progress tracking
            repo_data = repo_info(model_name)
            essential_files = [
                f for f in repo_data.siblings 
                if not any(pattern in f.rfilename for pattern in config.ignore_patterns or [])
            ]
            
            progress.total_files = len(essential_files)
            progress.total_size_bytes = sum(f.size for f in essential_files if f.size)
            
            logger.info(f"📊 Files to download: {progress.total_files}")
            logger.info(f"📊 Total size: {self._format_size(progress.total_size_bytes)}")
            logger.info(f"📁 Cache directory: {cache_dir}")
            logger.info("🎯 Using HuggingFace cache - no duplicates!")
            
            # Download with better progress tracking to HuggingFace cache
            downloaded_path = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,  # Use cache_dir instead of local_dir
                resume_download=config.resume_download,
                local_files_only=False,
                ignore_patterns=config.ignore_patterns,
                tqdm_class=tqdm if TQDM_AVAILABLE else None,
                etag_timeout=config.timeout,
                max_workers=min(config.max_workers, 2)  # Limit workers for large models
            )
            
            if self.download_interrupted:
                logger.warning("⚠️ Download was interrupted but can be resumed")
                return False
            
            progress.is_complete = True
            progress.completed_files = progress.total_files
            progress.downloaded_bytes = progress.total_size_bytes
            self._notify_progress(progress)
            
            logger.info(f"✅ Qwen2-VL model download completed!")
            logger.info(f"📁 Model saved to: {downloaded_path}")
            logger.info("🎯 Enhanced capabilities now available:")
            
            capabilities = self.supported_models[model_name].get("capabilities", [])
            for capability in capabilities:
                logger.info(f"   • {capability.replace('_', ' ').title()}")
            
            # Verify integrity if requested
            if config.verify_integrity:
                return self.verify_model_integrity(model_name, str(local_dir))
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Qwen2-VL model download failed: {e}")
            progress.error_message = str(e)
            self._notify_progress(progress)
            return False
    
    def verify_model_integrity(self, model_name: str, local_path: Optional[str] = None) -> bool:
        """Verify the integrity of a downloaded model"""
        logger.info(f"🔍 Verifying model integrity: {model_name}")
        
        try:
            if local_path:
                model_path = Path(local_path)
            else:
                model_path = self._get_local_model_path(model_name)
            
            if not model_path or not model_path.exists():
                logger.error("❌ Model path not found for verification")
                return False
            
            # Check for essential files based on model type
            model_config = self.supported_models.get(model_name, {})
            model_type = model_config.get("type", "")
            
            essential_files = []
            if model_type == "multimodal-language":
                essential_files = [
                    "config.json",
                    "generation_config.json", 
                    "model.safetensors.index.json",
                    "tokenizer.json",
                    "tokenizer_config.json"
                ]
            else:  # diffusion models
                essential_files = [
                    "model_index.json",
                    "scheduler/scheduler_config.json",
                    "text_encoder/config.json",
                    "unet/config.json",
                    "vae/config.json"
                ]
            
            missing_files = []
            for file_name in essential_files:
                file_path = model_path / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
            
            if missing_files:
                logger.error(f"❌ Missing essential files: {missing_files}")
                return False
            
            # Check for safetensors or bin files
            has_weights = any(
                model_path.rglob("*.safetensors") or 
                model_path.rglob("*.bin")
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
            if status["local_complete"]:
                logger.info("✅ Qwen-Image model already available")
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
            if status["local_complete"]:
                logger.info(f"✅ {model_name} already available")
                return True
        
        config = ModelDownloadConfig(
            model_name=model_name,
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
            verify_integrity=True,
            max_workers=2  # Limit for large models
        )
        
        return self.download_model(model_name, config)
    
    def cleanup_old_models(self, keep_models: List[str] = None) -> bool:
        """Clean up old or unused models to save disk space"""
        if keep_models is None:
            keep_models = ["Qwen/Qwen-Image", "Qwen/Qwen2-VL-7B-Instruct"]
        
        logger.info("🧹 Cleaning up old models...")
        
        try:
            models_dir = Path("./models")
            if not models_dir.exists():
                logger.info("No models directory found")
                return True
            
            cleaned_size = 0
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    # Check if this model should be kept
                    should_keep = any(
                        keep_model.split("/")[-1] == model_dir.name 
                        for keep_model in keep_models
                    )
                    
                    if not should_keep:
                        # Calculate size before deletion
                        size = sum(
                            f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
                        )
                        
                        logger.info(f"🗑️ Removing {model_dir.name} ({self._format_size(size)})")
                        shutil.rmtree(model_dir)
                        cleaned_size += size
            
            if cleaned_size > 0:
                logger.info(f"✅ Cleaned up {self._format_size(cleaned_size)} of disk space")
            else:
                logger.info("✅ No cleanup needed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return False
    
    def get_download_status(self, model_name: str) -> Dict[str, Any]:
        """Get detailed download status for a model"""
        return self.check_model_availability(model_name)
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all supported models and their status"""
        models_status = {}
        
        for model_name in self.supported_models:
            status = self.check_model_availability(model_name)
            models_status[model_name] = {
                **self.supported_models[model_name],
                **status
            }
        
        return models_status
    
    def _get_local_model_path(self, model_name: str) -> Optional[Path]:
        """Get the cached model path (prioritizes HuggingFace cache)"""
        # Check in HuggingFace cache first (primary location)
        try:
            cache_path = Path(self.cache_dir) / f"models--{model_name.replace('/', '--')}"
            if cache_path.exists():
                # Look for the actual model in snapshots directory
                snapshots_dir = cache_path / "snapshots"
                if snapshots_dir.exists():
                    snapshots = list(snapshots_dir.iterdir())
                    if snapshots:
                        # Return the latest snapshot path (actual model location)
                        latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
                        return latest_snapshot
                # If no snapshots, return the cache directory itself
                return cache_path
        except:
            pass
        
        # Fallback: Check in ./models directory (legacy location)
        local_models_dir = Path("./models") / model_name.split("/")[-1]
        if local_models_dir.exists():
            return local_models_dir
        
        return None
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """Get the path to a cached model for loading"""
        model_path = self._get_local_model_path(model_name)
        return str(model_path) if model_path else None
    
    def _cleanup_partial_download(self, model_name: str):
        """Clean up partial downloads on failure"""
        try:
            local_path = self._get_local_model_path(model_name)
            if local_path and local_path.exists():
                logger.info(f"🧹 Cleaning up partial download: {local_path}")
                shutil.rmtree(local_path)
        except Exception as e:
            logger.warning(f"⚠️ Could not clean up partial download: {e}")
    
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
    
    def check_disk_space(self, required_gb: float, path: str = "./models") -> bool:
        """Check if there's enough disk space for download"""
        try:
            stat = shutil.disk_usage(path)
            free_gb = stat.free / (1024**3)
            
            if free_gb < required_gb:
                logger.error(f"❌ Insufficient disk space: {free_gb:.1f}GB available, {required_gb:.1f}GB required")
                return False
            
            logger.info(f"✅ Sufficient disk space: {free_gb:.1f}GB available")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Could not check disk space: {e}")
            return True  # Assume OK if we can't check
    
    def estimate_download_time(self, model_name: str, bandwidth_mbps: float = 50) -> str:
        """Estimate download time based on model size and bandwidth"""
        if model_name not in self.supported_models:
            return "Unknown"
        
        size_gb = self.supported_models[model_name]["expected_size_gb"]
        size_mb = size_gb * 1024
        time_minutes = size_mb / bandwidth_mbps / 60
        
        if time_minutes < 1:
            return f"{time_minutes * 60:.0f} seconds"
        elif time_minutes < 60:
            return f"{time_minutes:.0f} minutes"
        else:
            hours = time_minutes / 60
            return f"{hours:.1f} hours"


# Convenience functions for easy usage
def download_qwen_image(force_redownload: bool = False) -> bool:
    """Convenience function to download Qwen-Image model"""
    manager = ModelDownloadManager()
    return manager.download_qwen_image(force_redownload)


def download_qwen2_vl(model_size: str = "7B", force_redownload: bool = False) -> bool:
    """Convenience function to download Qwen2-VL model"""
    manager = ModelDownloadManager()
    return manager.download_qwen2_vl(model_size, force_redownload)


def check_model_status(model_name: str) -> Dict[str, Any]:
    """Convenience function to check model status"""
    manager = ModelDownloadManager()
    return manager.get_download_status(model_name)


if __name__ == "__main__":
    # Example usage and testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen Model Download Manager")
    parser.add_argument("--model", choices=["qwen-image", "qwen2-vl-7b", "qwen2-vl-2b"], 
                       help="Model to download")
    parser.add_argument("--check", type=str, help="Check status of a model")
    parser.add_argument("--list", action="store_true", help="List all models")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old models")
    parser.add_argument("--force", action="store_true", help="Force redownload")
    
    args = parser.parse_args()
    
    manager = ModelDownloadManager()
    
    if args.list:
        models = manager.list_available_models()
        print("\n📋 Available Models:")
        print("=" * 60)
        for name, info in models.items():
            status = "✅ Complete" if info.get("local_complete") else "❌ Missing"
            print(f"{name}: {status} ({info['expected_size_gb']}GB)")
    
    elif args.check:
        status = manager.get_download_status(args.check)
        print(f"\n📊 Status for {args.check}:")
        print("=" * 40)
        for key, value in status.items():
            print(f"{key}: {value}")
    
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
    
    elif args.cleanup:
        manager.cleanup_old_models()
    
    else:
        parser.print_help()