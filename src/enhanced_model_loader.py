#!/usr/bin/env python3
"""
Enhanced model loader with optimized download strategy
"""

import os
import shutil
import time
from typing import Optional, Tuple
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
import requests

class EnhancedQwenLoader:
    def __init__(self, model_name: str = "Qwen/Qwen-Image"):
        self.model_name = model_name
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def check_disk_space(self, required_gb: float = 40) -> bool:
        """Check if sufficient disk space is available"""
        try:
            statvfs = os.statvfs(self.cache_dir)
            free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
            print(f"💾 Available disk space: {free_gb:.1f}GB (required: {required_gb}GB)")
            return free_gb >= required_gb
        except Exception as e:
            print(f"⚠️ Could not check disk space: {e}")
            return True  # Assume sufficient space
    
    def check_network_speed(self) -> float:
        """Test network speed with a small file"""
        try:
            print("🌐 Testing network speed...")
            start_time = time.time()
            response = requests.get(
                "https://huggingface.co/Qwen/Qwen-Image/resolve/main/model_index.json",
                timeout=10
            )
            elapsed = time.time() - start_time
            size_mb = len(response.content) / (1024 * 1024)
            speed_mbps = size_mb / elapsed
            print(f"📊 Network speed: ~{speed_mbps:.1f} MB/s")
            return speed_mbps
        except Exception as e:
            print(f"⚠️ Network speed test failed: {e}")
            return 1.0  # Assume slow connection
    
    def estimate_download_time(self, speed_mbps: float) -> str:
        """Estimate download time based on network speed"""
        model_size_gb = 36  # Approximate size
        model_size_mb = model_size_gb * 1024
        time_seconds = model_size_mb / speed_mbps
        
        if time_seconds < 60:
            return f"{time_seconds:.0f} seconds"
        elif time_seconds < 3600:
            return f"{time_seconds/60:.1f} minutes"
        else:
            return f"{time_seconds/3600:.1f} hours"
    
    def optimized_download(self) -> bool:
        """Download model with optimized strategy"""
        try:
            # Pre-flight checks
            if not self.check_disk_space():
                print("❌ Insufficient disk space!")
                return False
            
            speed = self.check_network_speed()
            eta = self.estimate_download_time(speed)
            print(f"⏱️ Estimated download time: {eta}")
            
            print("🚀 Starting optimized model download...")
            
            # Use snapshot_download for better control
            snapshot_download(
                repo_id=self.model_name,
                cache_dir=self.cache_dir,
                resume_download=True,
                max_workers=4,  # Parallel downloads
                repo_type="model",
                ignore_patterns=["*.bin"],  # Prefer safetensors
                local_files_only=False,
                force_download=False
            )
            
            print("✅ Model download completed!")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def load_with_optimizations(self) -> Optional[DiffusionPipeline]:
        """Load model with performance optimizations"""
        try:
            print("🔄 Loading model with performance optimizations...")
            
            # Try optimized loading first
            pipe = DiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                trust_remote_code=True,
                low_cpu_mem_usage=False,  # Use RAM for speed
                device_map=None,
                local_files_only=True,  # Use cached files
                variant="fp16"
            )
            
            print("✅ Model loaded with optimizations")
            return pipe
            
        except Exception as e:
            print(f"⚠️ Optimized loading failed: {e}")
            
            # Fallback to basic loading
            try:
                print("🔄 Trying fallback loading...")
                pipe = DiffusionPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    local_files_only=True
                )
                print("✅ Model loaded with fallback")
                return pipe
            except Exception as e2:
                print(f"❌ All loading attempts failed: {e2}")
                return None
    
    def download_and_load(self) -> Optional[DiffusionPipeline]:
        """Complete download and load process"""
        print("🎯 Enhanced Qwen-Image Download & Load Process")
        print("=" * 50)
        
        # Check if model is already cached
        model_path = os.path.join(self.cache_dir, f"models--{self.model_name.replace('/', '--')}")
        
        if os.path.exists(model_path):
            print("📁 Model found in cache, checking completeness...")
            try:
                # Try to load from cache first
                pipe = self.load_with_optimizations()
                if pipe is not None:
                    print("✅ Model loaded from cache successfully!")
                    return pipe
                else:
                    print("⚠️ Cached model incomplete, re-downloading...")
            except:
                print("⚠️ Cached model corrupted, re-downloading...")
        
        # Download model
        if not self.optimized_download():
            return None
        
        # Load model
        return self.load_with_optimizations()

def test_enhanced_loader():
    """Test the enhanced loader"""
    loader = EnhancedQwenLoader()
    pipe = loader.download_and_load()
    
    if pipe:
        print("🎉 Enhanced loader successful!")
        return pipe
    else:
        print("❌ Enhanced loader failed")
        return None

if __name__ == "__main__":
    test_enhanced_loader()