#!/usr/bin/env python3
"""
Enhanced Qwen-Image-Edit Model Downloader using HuggingFace Hub API
Provides better progress tracking, resume capability, and error handling
"""

import os
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
from huggingface_hub import HfApi, hf_hub_download, repo_info, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class QwenEditDownloader:
    """Enhanced downloader for Qwen-Image-Edit model using HF Hub API"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.repo_id = "Qwen/Qwen-Image-Edit"
        self.cache_dir = cache_dir or "./models/qwen-image-edit"
        self.api = HfApi()
        self.download_interrupted = False
        
        # Create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup signal handlers for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interruption signals gracefully"""
        print(f"\n⚠️ Download interrupted by signal {signum}")
        print("💾 Progress has been saved, you can resume later")
        self.download_interrupted = True
        
    def check_model_status(self) -> dict:
        """Check current model download status"""
        print("🔍 Checking Qwen-Image-Edit model status...")
        
        status = {
            "repo_accessible": False,
            "locally_available": False,
            "files_downloaded": [],
            "missing_files": [],
            "total_size": 0,
            "downloaded_size": 0
        }
        
        try:
            # Check if repository is accessible
            repo_data = repo_info(self.repo_id)
            status["repo_accessible"] = True
            status["total_size"] = sum(file.size for file in repo_data.siblings if file.size)
            
            print(f"✅ Repository accessible: {self.repo_id}")
            print(f"📊 Total size: {self._format_size(status['total_size'])}")
            
            # Check local files
            cache_path = Path(self.cache_dir)
            if cache_path.exists():
                for file_info in repo_data.siblings:
                    local_file = cache_path / file_info.rfilename
                    if local_file.exists() and local_file.stat().st_size == file_info.size:
                        status["files_downloaded"].append(file_info.rfilename)
                        status["downloaded_size"] += file_info.size
                    else:
                        status["missing_files"].append(file_info.rfilename)
                
                if status["files_downloaded"]:
                    status["locally_available"] = True
                    print(f"📁 Local files found: {len(status['files_downloaded'])}")
                    print(f"💾 Downloaded: {self._format_size(status['downloaded_size'])}")
            
            return status
            
        except (RepositoryNotFoundError, RevisionNotFoundError) as e:
            print(f"❌ Repository error: {e}")
            return status
        except Exception as e:
            print(f"❌ Status check failed: {e}")
            return status
    
    def download_with_progress(self, resume: bool = True) -> bool:
        """Download model with progress tracking and resume capability"""
        print("🚀 Starting enhanced Qwen-Image-Edit download...")
        print("=" * 60)
        
        status = self.check_model_status()
        
        if not status["repo_accessible"]:
            print("❌ Cannot access repository. Check internet connection.")
            return False
        
        if status["locally_available"] and not status["missing_files"]:
            print("✅ Model already fully downloaded!")
            return self._verify_model_loading()
        
        if resume and status["files_downloaded"]:
            print(f"🔄 Resuming download ({len(status['missing_files'])} files remaining)")
        
        try:
            print("📥 Downloading with HuggingFace Hub API...")
            print(f"📁 Cache directory: {self.cache_dir}")
            print("⏳ This may take 10-30 minutes for a 20GB model...")
            print("💡 Download can be resumed if interrupted")
            
            # Use snapshot_download for complete model download
            start_time = time.time()
            
            # Download with progress callback
            downloaded_path = snapshot_download(
                repo_id=self.repo_id,
                cache_dir=self.cache_dir,
                resume_download=resume,
                local_files_only=False,
                repo_type="model",
                ignore_patterns=["*.md", "*.txt", ".gitattributes"],  # Skip non-essential files
                tqdm_class=tqdm,
                etag_timeout=300,  # 5 minute timeout for etag check
                max_workers=4  # Parallel downloads
            )
            
            if self.download_interrupted:
                print("⚠️ Download was interrupted but can be resumed")
                return False
            
            download_time = time.time() - start_time
            print(f"✅ Download completed in {download_time:.1f} seconds!")
            print(f"📁 Model cached at: {downloaded_path}")
            
            return self._verify_model_loading()
            
        except KeyboardInterrupt:
            print("\n⚠️ Download interrupted by user")
            print("💾 Progress saved, run again to resume")
            return False
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("\n💡 Troubleshooting suggestions:")
            print("1. Check internet connection stability")
            print("2. Verify sufficient disk space (~25GB)")
            print("3. Try running with --resume flag")
            print("4. Check HuggingFace Hub status")
            return False
    
    def download_selective_files(self, file_patterns: List[str]) -> bool:
        """Download only specific files (useful for testing)"""
        print("🎯 Downloading selective files...")
        
        try:
            repo_data = repo_info(self.repo_id)
            
            for pattern in file_patterns:
                matching_files = [f for f in repo_data.siblings if pattern in f.rfilename]
                
                for file_info in matching_files:
                    print(f"📥 Downloading: {file_info.rfilename}")
                    
                    hf_hub_download(
                        repo_id=self.repo_id,
                        filename=file_info.rfilename,
                        cache_dir=self.cache_dir,
                        resume_download=True
                    )
                    
                    if self.download_interrupted:
                        return False
            
            print("✅ Selective download completed!")
            return True
            
        except Exception as e:
            print(f"❌ Selective download failed: {e}")
            return False
    
    def _verify_model_loading(self) -> bool:
        """Verify that the downloaded model can be loaded"""
        print("🔍 Verifying model loading...")
        
        try:
            from diffusers import QwenImageEditPipeline

            # Try to load from local cache
            pipeline = QwenImageEditPipeline.from_pretrained(
                self.repo_id,
                cache_dir=self.cache_dir,
                local_files_only=True,
                torch_dtype=torch.bfloat16
            )
            
            print("✅ Model loading verification successful!")
            
            # Test GPU loading if available
            if torch.cuda.is_available():
                print("🔄 Testing GPU loading...")
                try:
                    pipeline = pipeline.to("cuda")
                    print("✅ GPU loading successful!")
                except Exception as e:
                    print(f"⚠️ GPU loading failed: {e}")
                    print("Model will work on CPU")
            
            del pipeline  # Free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
            
        except ImportError:
            print("⚠️ QwenImageEditPipeline not available in diffusers")
            print("💡 Install latest diffusers: pip install git+https://github.com/huggingface/diffusers.git")
            return False
        except Exception as e:
            print(f"❌ Model verification failed: {e}")
            return False
    
    def cleanup_partial_downloads(self):
        """Clean up any partial or corrupted downloads"""
        print("🧹 Cleaning up partial downloads...")
        
        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            print("📁 No cache directory found")
            return
        
        # Remove .tmp files and incomplete downloads
        tmp_files = list(cache_path.glob("**/*.tmp"))
        incomplete_files = list(cache_path.glob("**/*.incomplete"))
        
        for tmp_file in tmp_files + incomplete_files:
            try:
                tmp_file.unlink()
                print(f"🗑️ Removed: {tmp_file.name}")
            except Exception as e:
                print(f"⚠️ Could not remove {tmp_file}: {e}")
        
        print("✅ Cleanup completed")
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format byte size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Qwen-Image-Edit Downloader")
    parser.add_argument("--cache-dir", default="./models/qwen-image-edit", 
                       help="Custom cache directory")
    parser.add_argument("--resume", action="store_true", default=True,
                       help="Resume interrupted downloads")
    parser.add_argument("--no-resume", action="store_true",
                       help="Start fresh download")
    parser.add_argument("--status-only", action="store_true",
                       help="Only check download status")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up partial downloads")
    parser.add_argument("--selective", nargs="+",
                       help="Download only files matching patterns")
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = QwenEditDownloader(cache_dir=args.cache_dir)
    
    print("🎨 Enhanced Qwen-Image-Edit Downloader")
    print("=" * 50)
    
    try:
        if args.cleanup:
            downloader.cleanup_partial_downloads()
            return
        
        if args.status_only:
            status = downloader.check_model_status()
            if status["locally_available"]:
                print("🎉 Model is ready for use!")
            else:
                print("📥 Model needs to be downloaded")
            return
        
        if args.selective:
            success = downloader.download_selective_files(args.selective)
        else:
            resume = args.resume and not args.no_resume
            success = downloader.download_with_progress(resume=resume)
        
        if success:
            print("\n🎉 Qwen-Image-Edit model is ready!")
            print("Enhanced features now available:")
            print("• Image-to-Image generation")
            print("• Inpainting capabilities")
            print("• Advanced image editing")
        else:
            print("\n❌ Download incomplete. Run again to resume.")
            
    except KeyboardInterrupt:
        print("\n👋 Download interrupted. Progress saved for resume.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the error details above.")

if __name__ == "__main__":
    main()