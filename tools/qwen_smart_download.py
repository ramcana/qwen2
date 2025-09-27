#!/usr/bin/env python3
"""
Smart Qwen-Image Downloader
Downloads files individually with progress tracking to avoid hanging
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import threading

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from huggingface_hub import hf_hub_download, list_repo_files
import torch

def clear_gpu_memory():
    """Clear GPU memory before starting"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print("🧹 GPU memory cleared")
    except Exception as e:
        print(f"⚠️ Could not clear GPU memory: {e}")

def get_file_size_mb(repo_id: str, filename: str) -> float:
    """Get file size in MB (estimated)"""
    # Estimate based on file type and name
    if 'safetensors' in filename:
        if 'text_encoder' in filename:
            return 1500  # ~1.5GB per text encoder file
        elif 'transformer' in filename:
            return 2000  # ~2GB per transformer file
        else:
            return 500   # ~500MB for other safetensors
    elif filename.endswith('.json'):
        return 0.1   # Small config files
    else:
        return 10    # Default estimate
    
def download_file_with_retry(repo_id: str, filename: str, cache_dir: str = None, max_retries: int = 3) -> str:
    """Download a single file with retry logic"""
    
    for attempt in range(max_retries):
        try:
            print(f"📥 Downloading {filename} (attempt {attempt + 1}/{max_retries})")
            
            start_time = time.time()
            
            # Download the file
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                resume_download=True,
                local_files_only=False
            )
            
            end_time = time.time()
            download_time = end_time - start_time
            file_size = get_file_size_mb(repo_id, filename)
            speed_mbps = file_size / max(download_time, 0.1)
            
            print(f"✅ Downloaded {filename} in {download_time:.1f}s (~{speed_mbps:.1f} MB/s)")
            return local_path
            
        except Exception as e:
            print(f"❌ Failed to download {filename} (attempt {attempt + 1}): {str(e)}")
            
            if attempt < max_retries - 1:
                wait_time = min(60, (2 ** attempt) * 5)  # Exponential backoff, max 60s
                print(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                
                # Clear memory before retry
                clear_gpu_memory()
            else:
                raise Exception(f"Failed to download {filename} after {max_retries} attempts: {str(e)}")

def download_qwen_image_smart(cache_dir: str = None, max_retries: int = 3) -> str:
    """Smart download of Qwen-Image model with individual file tracking"""
    
    repo_id = "Qwen/Qwen-Image"
    
    print("🎨 Smart Qwen-Image Downloader")
    print("=" * 50)
    
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    try:
        # Get list of all files in the repository
        print("📋 Getting file list from repository...")
        all_files = list_repo_files(repo_id)
        print(f"Found {len(all_files)} files to download")
        
        # Categorize files by priority (small files first, then large files)
        small_files = []
        large_files = []
        
        for filename in all_files:
            if filename.endswith('.json') or filename.endswith('.txt') or filename in ['README.md', 'LICENSE', '.gitattributes']:
                small_files.append(filename)
            else:
                large_files.append(filename)
        
        # Sort large files by estimated size (smaller first)
        large_files.sort(key=lambda f: get_file_size_mb(repo_id, f))
        
        # Download order: small files first, then large files
        download_order = small_files + large_files
        
        print(f"📦 Download plan:")
        print(f"   Small files: {len(small_files)}")
        print(f"   Large files: {len(large_files)}")
        
        # Track progress
        downloaded_files = []
        failed_files = []
        total_start_time = time.time()
        
        # Download files one by one
        for i, filename in enumerate(download_order, 1):
            try:
                file_size_mb = get_file_size_mb(repo_id, filename)
                print(f"\n[{i}/{len(download_order)}] {filename} (~{file_size_mb:.0f}MB)")
                
                local_path = download_file_with_retry(repo_id, filename, cache_dir, max_retries)
                downloaded_files.append((filename, local_path))
                
                # Show progress
                progress = (i / len(download_order)) * 100
                elapsed = time.time() - total_start_time
                print(f"📊 Progress: {progress:.1f}% ({i}/{len(download_order)}) - Elapsed: {elapsed:.1f}s")
                
                # Brief pause between downloads to avoid overwhelming the server
                if i < len(download_order):
                    time.sleep(1)
                    
            except Exception as e:
                print(f"❌ Failed to download {filename}: {str(e)}")
                failed_files.append((filename, str(e)))
                
                # Continue with other files
                continue
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        print(f"\n📊 Download Summary")
        print("=" * 30)
        print(f"✅ Successful: {len(downloaded_files)}/{len(download_order)}")
        print(f"❌ Failed: {len(failed_files)}/{len(download_order)}")
        print(f"⏱️ Total time: {total_time:.1f}s")
        
        if failed_files:
            print(f"\n❌ Failed files:")
            for filename, error in failed_files:
                print(f"   {filename}: {error}")
        
        if downloaded_files:
            # Get the cache directory from the first downloaded file
            first_file_path = downloaded_files[0][1]
            model_cache_dir = str(Path(first_file_path).parent.parent)
            print(f"\n📁 Model cached at: {model_cache_dir}")
            
            if len(failed_files) == 0:
                print("🎉 All files downloaded successfully!")
                return model_cache_dir
            elif len(downloaded_files) > len(failed_files):
                print("⚠️ Partial download completed - you may be able to use the model")
                return model_cache_dir
            else:
                raise Exception("Too many files failed to download")
        else:
            raise Exception("No files were downloaded successfully")
            
    except Exception as e:
        print(f"💥 Download failed: {str(e)}")
        raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart Qwen-Image Model Downloader')
    parser.add_argument('--cache-dir', type=str, help='Cache directory for models')
    parser.add_argument('--max-retries', type=int, default=3, help='Max retries per file')
    parser.add_argument('--check-existing', action='store_true', help='Check if model already exists')
    
    args = parser.parse_args()
    
    if args.check_existing:
        try:
            from diffusers import DiffusionPipeline
            print("🔍 Checking if Qwen-Image is already available...")
            
            # Try to load from cache
            pipe = DiffusionPipeline.from_pretrained(
                "Qwen/Qwen-Image",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                local_files_only=True  # Only check cache
            )
            print("✅ Qwen-Image model is already available in cache!")
            return
            
        except Exception as e:
            print(f"❌ Model not found in cache: {str(e)}")
            print("Proceeding with download...")
    
    try:
        model_path = download_qwen_image_smart(
            cache_dir=args.cache_dir,
            max_retries=args.max_retries
        )
        
        print(f"\n🎉 Download completed successfully!")
        print(f"📁 Model location: {model_path}")
        print(f"\n🚀 You can now use the model:")
        print(f"   python launch.py")
        print(f"   ./scripts/safe_restart.sh")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Download cancelled by user")
        clear_gpu_memory()
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Download failed: {e}")
        clear_gpu_memory()
        sys.exit(1)

if __name__ == "__main__":
    main()
