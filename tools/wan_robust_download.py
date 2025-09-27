#!/usr/bin/env python3
"""
WAN-Powered Robust Download Tool for Qwen Models
Replaces the hanging download with a robust, memory-safe approach
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from wan_download_adapter import WANDownloadAdapter, QwenDownloadConfig, create_progress_callback

def clear_gpu_memory():
    """Clear all GPU memory before starting"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear any remaining allocations
            torch.cuda.empty_cache()
            
            print("🧹 GPU memory cleared")
            return True
    except Exception as e:
        print(f"⚠️ Could not clear GPU memory: {e}")
        return False

def check_system_resources():
    """Check system resources before starting download"""
    print("🔍 Checking system resources...")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            print(f"🎮 GPU: {gpu_name}")
            print(f"📊 GPU Memory: {total_memory / (1024**3):.1f}GB")
        else:
            print("⚠️ No GPU detected")
    except ImportError:
        print("⚠️ PyTorch not available")
    
    # Check system memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"💾 System Memory: {memory.available / (1024**3):.1f}GB available / {memory.total / (1024**3):.1f}GB total")
        
        if memory.percent > 80:
            print(f"⚠️ High memory usage: {memory.percent}%")
            return False
    except ImportError:
        print("⚠️ psutil not available for memory monitoring")
    
    # Check disk space
    try:
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        if os.path.exists(cache_dir):
            statvfs = os.statvfs(cache_dir)
            free_space = statvfs.f_frsize * statvfs.f_bavail
            print(f"💿 Disk Space: {free_space / (1024**3):.1f}GB available in {cache_dir}")
            
            if free_space < 50 * (1024**3):  # Less than 50GB
                print("⚠️ Low disk space - Qwen models are large!")
                return False
    except Exception as e:
        print(f"⚠️ Could not check disk space: {e}")
    
    return True

def download_qwen_models(models=None, cache_dir=None, max_retries=5):
    """Download Qwen models using WAN orchestrator approach"""
    
    if models is None:
        models = ["Qwen/Qwen-Image"]
    
    print("🎨 WAN-Powered Qwen Model Downloader")
    print("=" * 50)
    
    # Check system resources
    if not check_system_resources():
        print("❌ System resources insufficient or at risk")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    success_count = 0
    total_models = len(models)
    
    for i, model_name in enumerate(models, 1):
        print(f"\n📦 Downloading model {i}/{total_models}: {model_name}")
        print("-" * 40)
        
        try:
            # Configure download
            config = QwenDownloadConfig(
                model_name=model_name,
                cache_dir=cache_dir,
                max_concurrent_downloads=4,
                retry_attempts=max_retries,
                timeout_seconds=900,  # 15 minutes per attempt
                use_memory_optimization=True,
                gpu_memory_threshold=0.8
            )
            
            # Create adapter
            adapter = WANDownloadAdapter(config)
            
            # Add progress callback
            def detailed_progress_callback(progress_data):
                status = progress_data.get('status', 'unknown')
                model = progress_data.get('model', 'unknown')
                
                if status == 'downloading':
                    attempt = progress_data.get('attempt', 1)
                    max_attempts = progress_data.get('max_attempts', 1)
                    print(f"📥 Downloading {model} (attempt {attempt}/{max_attempts})...")
                    
                    # Show system stats during download
                    memory_stats = adapter.get_memory_stats()
                    gpu_stats = adapter.get_gpu_stats()
                    
                    if memory_stats.get('available'):
                        print(f"   💾 Memory: {memory_stats['percent_used']:.1f}% used")
                    
                    if gpu_stats.get('available'):
                        print(f"   🎮 GPU Memory: {gpu_stats['percent_used']:.1f}% used")
                
                elif status == 'completed':
                    download_time = progress_data.get('download_time', 0)
                    local_path = progress_data.get('local_path', 'unknown')
                    print(f"✅ Download completed in {download_time:.1f}s")
                    print(f"📁 Model available at: {local_path}")
            
            adapter.add_progress_callback(detailed_progress_callback)
            
            # Start download
            start_time = time.time()
            local_path = adapter.download_model()
            end_time = time.time()
            
            print(f"🎉 Successfully downloaded {model_name}")
            print(f"⏱️ Total time: {end_time - start_time:.1f}s")
            print(f"📂 Location: {local_path}")
            
            success_count += 1
            
            # Show download stats
            stats = adapter.get_download_stats()
            if stats.get('retry_count', 0) > 0:
                print(f"🔄 Required {stats['retry_count']} retries")
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {str(e)}")
            print("🔧 This might be due to:")
            print("   • Network connectivity issues")
            print("   • Insufficient memory or disk space")
            print("   • HuggingFace Hub access issues")
            print("   • Model repository access restrictions")
            
            # Clear memory after failure
            clear_gpu_memory()
            
            continue
    
    print(f"\n📊 Download Summary")
    print("=" * 30)
    print(f"✅ Successful: {success_count}/{total_models}")
    print(f"❌ Failed: {total_models - success_count}/{total_models}")
    
    return success_count == total_models

def main():
    parser = argparse.ArgumentParser(description='WAN-Powered Robust Qwen Model Downloader')
    parser.add_argument('--models', nargs='+', 
                       default=['Qwen/Qwen-Image'],
                       help='Models to download (default: Qwen/Qwen-Image)')
    parser.add_argument('--cache-dir', type=str,
                       help='Cache directory for models')
    parser.add_argument('--max-retries', type=int, default=5,
                       help='Maximum retry attempts per model (default: 5)')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check system resources, don\'t download')
    
    args = parser.parse_args()
    
    if args.check_only:
        print("🔍 System Resource Check")
        print("=" * 30)
        resources_ok = check_system_resources()
        if resources_ok:
            print("✅ System resources look good!")
        else:
            print("⚠️ System resources may be insufficient")
        return
    
    try:
        success = download_qwen_models(
            models=args.models,
            cache_dir=args.cache_dir,
            max_retries=args.max_retries
        )
        
        if success:
            print("\n🎉 All downloads completed successfully!")
            print("\n🚀 You can now start the Qwen UI:")
            print("   ./scripts/safe_restart.sh")
        else:
            print("\n❌ Some downloads failed. Check the errors above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Download cancelled by user")
        clear_gpu_memory()
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        clear_gpu_memory()
        sys.exit(1)

if __name__ == "__main__":
    main()
