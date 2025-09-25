#!/usr/bin/env python3
"""
Robust model downloader with true resumable downloads.
"""
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


def check_existing_files(repo_id: str, dest: str) -> dict:
    """Check which files already exist and their sizes."""
    dest_path = Path(dest)
    existing_files = {}
    
    if dest_path.exists():
        for file_path in dest_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(dest_path)
                existing_files[str(rel_path)] = file_path.stat().st_size
    
    return existing_files


def download_with_retry(repo_id: str, dest: str, max_retries: int = 3) -> bool:
    """Download with retry logic and proper error handling."""
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            print(f"📥 Attempt {attempt + 1}/{max_retries}: Downloading {repo_id}")
            
            # Check existing files before download
            existing_files = check_existing_files(repo_id, dest)
            if existing_files:
                print(f"📂 Found {len(existing_files)} existing files, resuming download...")
                for filename, size in list(existing_files.items())[:5]:  # Show first 5
                    print(f"   ✓ {filename} ({size:,} bytes)")
                if len(existing_files) > 5:
                    print(f"   ... and {len(existing_files) - 5} more files")
            
            # Use snapshot_download with cache_dir to avoid duplicates
            snapshot_download(
                repo_id=repo_id,
                cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),  # Use HF cache
                resume_download=True,
                # Note: Model will be cached, create symlink to dest if needed
            )
            
            # Create symlink to requested destination if needed
            from huggingface_hub import snapshot_download
            cached_path = snapshot_download(repo_id=repo_id, local_files_only=True)
            if not os.path.exists(dest):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if os.name == 'nt':  # Windows
                    import shutil
                    shutil.copytree(cached_path, dest)
                else:  # Unix/Linux
                    os.symlink(cached_path, dest)
            
            print(f"✅ Successfully downloaded {repo_id}")
            return True
            
        except HfHubHTTPError as e:
            print(f"🌐 HTTP Error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed after {max_retries} attempts: {e}")
                return False
                
        except KeyboardInterrupt:
            print("\n⏹️ Download interrupted by user")
            print("💡 Run the command again to resume from where it left off")
            return False
            
        except Exception as e:
            print(f"💥 Unexpected error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed after {max_retries} attempts: {e}")
                return False
    
    return False


def verify_download(repo_id: str, dest: str) -> bool:
    """Verify the download is complete by checking key files."""
    dest_path = Path(dest)
    
    # Key files that should exist for Qwen-Image-Edit
    required_files = [
        "model_index.json",
        "scheduler/scheduler_config.json",  # Fixed path
        "text_encoder/config.json",
        "transformer/config.json",
    ]
    
    missing_files = []
    for required_file in required_files:
        file_path = dest_path / required_file
        if not file_path.exists():
            missing_files.append(required_file)
    
    if missing_files:
        print(f"⚠️ Missing required files: {missing_files}")
        return False
    
    # Check for any .tmp files (incomplete downloads)
    tmp_files = list(dest_path.rglob("*.tmp"))
    if tmp_files:
        print(f"⚠️ Found {len(tmp_files)} temporary files (incomplete downloads)")
        for tmp_file in tmp_files:
            print(f"   🗑️ Removing: {tmp_file}")
            tmp_file.unlink()
        return False
    
    print("✅ Download verification passed")
    return True


def get_download_stats(dest: str) -> dict:
    """Get statistics about the downloaded model."""
    dest_path = Path(dest)
    if not dest_path.exists():
        return {"files": 0, "size": 0}
    
    total_size = 0
    file_count = 0
    
    for file_path in dest_path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
            file_count += 1
    
    return {
        "files": file_count,
        "size": total_size,
        "size_gb": total_size / (1024**3)
    }


def main():
    """Main download function."""
    repo_id = "Qwen/Qwen-Image-Edit"
    dest = "./models/Qwen-Image-Edit"
    
    print("🚀 Qwen Image Edit Model Downloader")
    print("=" * 50)
    
    # Show initial stats
    initial_stats = get_download_stats(dest)
    if initial_stats["files"] > 0:
        print(f"📊 Current: {initial_stats['files']} files, {initial_stats['size_gb']:.2f} GB")
    
    # Attempt download
    success = download_with_retry(repo_id, dest, max_retries=3)
    
    if success:
        # Verify download
        if verify_download(repo_id, dest):
            final_stats = get_download_stats(dest)
            print("\n🎉 Download completed successfully!")
            print(f"📊 Final: {final_stats['files']} files, {final_stats['size_gb']:.2f} GB")
        else:
            print("\n⚠️ Download completed but verification failed")
            print("💡 Run the command again to complete missing files")
            sys.exit(1)
    else:
        print("\n❌ Download failed")
        print("💡 Run the command again to resume from where it left off")
        sys.exit(1)


if __name__ == "__main__":
    main()