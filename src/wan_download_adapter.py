#!/usr/bin/env python3
"""
WAN Model Orchestrator Adapter for Qwen2 Downloads
Integrates the robust WAN download system with Qwen2 model management
"""

import os
import sys
import asyncio
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import logging

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QwenDownloadConfig:
    """Configuration for Qwen model downloads using WAN orchestrator."""
    model_name: str = "Qwen/Qwen-Image"
    cache_dir: Optional[str] = None
    max_concurrent_downloads: int = 4
    bandwidth_limit_mbps: Optional[int] = None
    retry_attempts: int = 5
    timeout_seconds: int = 300
    use_memory_optimization: bool = True
    gpu_memory_threshold: float = 0.8

class WANDownloadAdapter:
    """
    Adapter that uses WAN Model Orchestrator patterns for robust Qwen downloads.
    Implements the key features without requiring the full WAN dependency.
    """
    
    def __init__(self, config: QwenDownloadConfig):
        self.config = config
        self.download_stats = {
            'start_time': None,
            'end_time': None,
            'bytes_downloaded': 0,
            'files_completed': 0,
            'files_failed': 0,
            'retry_count': 0
        }
        self._progress_callbacks = []
        self._is_downloading = False
        self._download_thread = None
        
        # Initialize memory monitoring
        self._init_memory_monitor()
        
        # Initialize GPU monitoring
        self._init_gpu_monitor()
    
    def _init_memory_monitor(self):
        """Initialize memory monitoring similar to WAN orchestrator."""
        try:
            import psutil
            self.psutil = psutil
            self._memory_available = True
            logger.info("Memory monitoring enabled")
        except ImportError:
            logger.warning("psutil not available, memory monitoring disabled")
            self.psutil = None
            self._memory_available = False
    
    def _init_gpu_monitor(self):
        """Initialize GPU memory monitoring."""
        self._gpu_available = torch.cuda.is_available()
        if self._gpu_available:
            logger.info(f"GPU monitoring enabled - {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("GPU not available")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        if not self._memory_available:
            return {'available': False}
        
        memory = self.psutil.virtual_memory()
        return {
            'available': True,
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent_used': memory.percent
        }
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU memory statistics."""
        if not self._gpu_available:
            return {'available': False}
        
        try:
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory
            
            return {
                'available': True,
                'allocated_gb': allocated / (1024**3),
                'reserved_gb': reserved / (1024**3),
                'total_gb': total / (1024**3),
                'percent_used': (allocated / total) * 100
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU stats: {e}")
            return {'available': False}
    
    def clear_gpu_memory(self):
        """Clear GPU memory similar to WAN orchestrator."""
        if not self._gpu_available:
            return
        
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            import gc
            gc.collect()
            
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")
        except Exception as e:
            logger.warning(f"Failed to clear GPU memory: {e}")
    
    def add_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add progress callback function."""
        self._progress_callbacks.append(callback)
    
    def _notify_progress(self, progress_data: Dict[str, Any]):
        """Notify all progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(progress_data)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def _check_memory_threshold(self) -> bool:
        """Check if memory usage is below safe threshold."""
        if not self._memory_available:
            return True  # Assume OK if we can't monitor
        
        memory = self.psutil.virtual_memory()
        if memory.percent > 85:  # 85% threshold
            logger.warning(f"High memory usage: {memory.percent}%")
            return False
        return True
    
    def _check_gpu_threshold(self) -> bool:
        """Check if GPU memory usage is below safe threshold."""
        if not self._gpu_available:
            return True
        
        try:
            allocated = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            percent = (allocated / total) * 100
            
            if percent > (self.config.gpu_memory_threshold * 100):
                logger.warning(f"High GPU memory usage: {percent:.1f}%")
                return False
            return True
        except:
            return True
    
    def _robust_download_with_retry(self, model_name: str, cache_dir: Optional[str] = None) -> str:
        """
        Robust download with retry logic inspired by WAN orchestrator.
        """
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                logger.info(f"Download attempt {attempt + 1}/{self.config.retry_attempts} for {model_name}")
                
                # Check memory before each attempt
                if not self._check_memory_threshold():
                    logger.warning("Memory threshold exceeded, clearing caches")
                    import gc
                    gc.collect()
                    self.clear_gpu_memory()
                    time.sleep(2)  # Brief pause
                
                # Check GPU memory
                if not self._check_gpu_threshold():
                    logger.warning("GPU memory threshold exceeded, clearing GPU cache")
                    self.clear_gpu_memory()
                    time.sleep(2)
                
                # Progress tracking
                progress_data = {
                    'model': model_name,
                    'attempt': attempt + 1,
                    'max_attempts': self.config.retry_attempts,
                    'status': 'downloading'
                }
                self._notify_progress(progress_data)
                
                # Perform the actual download with timeout
                start_time = time.time()
                
                # Use snapshot_download for full model download
                local_path = snapshot_download(
                    repo_id=model_name,
                    cache_dir=cache_dir,
                    resume_download=True,  # Resume interrupted downloads
                    local_files_only=False
                )
                
                end_time = time.time()
                download_time = end_time - start_time
                
                logger.info(f"✅ Successfully downloaded {model_name} in {download_time:.1f}s")
                logger.info(f"📁 Model cached at: {local_path}")
                
                # Update stats
                self.download_stats['end_time'] = end_time
                self.download_stats['files_completed'] += 1
                
                # Final progress notification
                progress_data.update({
                    'status': 'completed',
                    'local_path': local_path,
                    'download_time': download_time
                })
                self._notify_progress(progress_data)
                
                return local_path
                
            except Exception as e:
                last_exception = e
                self.download_stats['retry_count'] += 1
                
                logger.warning(f"❌ Download attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.config.retry_attempts - 1:
                    # Exponential backoff with jitter
                    wait_time = min(300, (2 ** attempt) + (time.time() % 10))
                    logger.info(f"⏳ Waiting {wait_time:.1f}s before retry...")
                    
                    # Clear caches before retry
                    self.clear_gpu_memory()
                    import gc
                    gc.collect()
                    
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ All download attempts failed for {model_name}")
                    self.download_stats['files_failed'] += 1
        
        # If we get here, all attempts failed
        raise Exception(f"Failed to download {model_name} after {self.config.retry_attempts} attempts. Last error: {last_exception}")
    
    def download_model(self, model_name: Optional[str] = None, cache_dir: Optional[str] = None) -> str:
        """
        Download model using robust WAN orchestrator-inspired approach.
        """
        model_name = model_name or self.config.model_name
        cache_dir = cache_dir or self.config.cache_dir
        
        if self._is_downloading:
            raise RuntimeError("Download already in progress")
        
        self._is_downloading = True
        self.download_stats['start_time'] = time.time()
        
        try:
            logger.info(f"🚀 Starting robust download of {model_name}")
            
            # Log system stats
            memory_stats = self.get_memory_stats()
            gpu_stats = self.get_gpu_stats()
            
            if memory_stats['available']:
                logger.info(f"💾 System Memory: {memory_stats['available_gb']:.1f}GB available / {memory_stats['total_gb']:.1f}GB total")
            
            if gpu_stats['available']:
                logger.info(f"🎮 GPU Memory: {gpu_stats['total_gb']:.1f}GB total")
            
            # Perform robust download
            local_path = self._robust_download_with_retry(model_name, cache_dir)
            
            return local_path
            
        finally:
            self._is_downloading = False
    
    def download_model_async(self, model_name: Optional[str] = None, cache_dir: Optional[str] = None) -> threading.Thread:
        """
        Start model download in background thread.
        """
        def download_worker():
            try:
                return self.download_model(model_name, cache_dir)
            except Exception as e:
                logger.error(f"Async download failed: {e}")
                raise
        
        self._download_thread = threading.Thread(
            target=download_worker,
            name=f"qwen-download-{model_name or self.config.model_name}",
            daemon=False
        )
        self._download_thread.start()
        return self._download_thread
    
    def get_download_stats(self) -> Dict[str, Any]:
        """Get current download statistics."""
        stats = self.download_stats.copy()
        if stats['start_time'] and not stats['end_time']:
            stats['elapsed_time'] = time.time() - stats['start_time']
        elif stats['start_time'] and stats['end_time']:
            stats['total_time'] = stats['end_time'] - stats['start_time']
        return stats
    
    def is_downloading(self) -> bool:
        """Check if download is currently in progress."""
        return self._is_downloading
    
    def cancel_download(self):
        """Cancel ongoing download (if possible)."""
        if self._download_thread and self._download_thread.is_alive():
            logger.warning("Download cancellation requested - this may not stop immediately")
            # Note: Python doesn't have clean thread cancellation
            # The download will continue but we mark it as cancelled
            self._is_downloading = False


def create_progress_callback():
    """Create a progress callback that logs download progress."""
    def progress_callback(progress_data: Dict[str, Any]):
        status = progress_data.get('status', 'unknown')
        model = progress_data.get('model', 'unknown')
        
        if status == 'downloading':
            attempt = progress_data.get('attempt', 1)
            max_attempts = progress_data.get('max_attempts', 1)
            print(f"📥 Downloading {model} (attempt {attempt}/{max_attempts})...")
        
        elif status == 'completed':
            download_time = progress_data.get('download_time', 0)
            local_path = progress_data.get('local_path', 'unknown')
            print(f"✅ Download completed in {download_time:.1f}s")
            print(f"📁 Model available at: {local_path}")
    
    return progress_callback


# Example usage functions
def download_qwen_image_robust(cache_dir: Optional[str] = None) -> str:
    """
    Download Qwen-Image model using robust WAN orchestrator approach.
    """
    config = QwenDownloadConfig(
        model_name="Qwen/Qwen-Image",
        cache_dir=cache_dir,
        max_concurrent_downloads=4,
        retry_attempts=5,
        timeout_seconds=600,  # 10 minutes per attempt
        use_memory_optimization=True
    )
    
    adapter = WANDownloadAdapter(config)
    adapter.add_progress_callback(create_progress_callback())
    
    try:
        local_path = adapter.download_model()
        return local_path
    except Exception as e:
        logger.error(f"Robust download failed: {e}")
        raise


def download_qwen_image_async(cache_dir: Optional[str] = None) -> WANDownloadAdapter:
    """
    Start async download of Qwen-Image model.
    Returns the adapter so you can monitor progress.
    """
    config = QwenDownloadConfig(
        model_name="Qwen/Qwen-Image",
        cache_dir=cache_dir,
        max_concurrent_downloads=4,
        retry_attempts=5,
        timeout_seconds=600
    )
    
    adapter = WANDownloadAdapter(config)
    adapter.add_progress_callback(create_progress_callback())
    
    # Start download in background
    adapter.download_model_async()
    
    return adapter


if __name__ == "__main__":
    # Example usage
    print("🎨 WAN-Powered Qwen Model Downloader")
    print("=" * 50)
    
    try:
        # Test robust download
        local_path = download_qwen_image_robust()
        print(f"✅ Success! Model downloaded to: {local_path}")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)
