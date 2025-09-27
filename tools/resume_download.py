#!/usr/bin/env python3
"""
Resume Qwen-Image download with proper verification
"""
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, repo_info

def resume_qwen_image_download():
    """Resume the Qwen-Image download"""
    print("📥 Resuming Qwen-Image download...")
    print("=" * 50)
    
    try:
        # Check current cache status
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        qwen_cache = Path(cache_dir) / "models--Qwen--Qwen-Image"
        
        if qwen_cache.exists():
            # Calculate current size
            current_size = sum(f.stat().st_size for f in qwen_cache.rglob('*') if f.is_file())
            print(f"📂 Found existing cache: {current_size / (1024**3):.2f} GB")
        
        # Resume download (this will automatically resume from where it left off)
        print("🔄 Resuming download (this may take 10-60 minutes)...")
        print("💡 You can safely interrupt with Ctrl+C and resume later")
        
        model_path = snapshot_download(
            'Qwen/Qwen-Image',
            resume_download=True,  # This is the key parameter
            local_files_only=False
        )
        
        # Calculate final size
        model_dir = Path(model_path)
        total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        
        print(f"\n✅ Download completed!")
        print(f"📁 Model path: {model_path}")
        print(f"📊 Total size: {total_size / (1024**3):.2f} GB")
        
        # Verify completeness
        print("\n🔍 Verifying download completeness...")
        repo_data = repo_info('Qwen/Qwen-Image')
        remote_files = [f.rfilename for f in repo_data.siblings]
        
        missing_files = []
        for remote_file in remote_files:
            local_file = model_dir / remote_file
            if not local_file.exists():
                missing_files.append(remote_file)
        
        if missing_files:
            print(f"⚠️ Warning: {len(missing_files)} files still missing")
            print("💡 Run this script again to complete the download")
            return False
        else:
            print("✅ All files verified - download complete!")
            
            # Create local symlink
            local_dir = Path("./models/Qwen-Image")
            if not local_dir.exists():
                try:
                    local_dir.parent.mkdir(exist_ok=True)
                    local_dir.symlink_to(model_path)
                    print(f"🔗 Created symlink: {local_dir}")
                except Exception as e:
                    print(f"⚠️ Could not create symlink: {e}")
            
            return True
        
    except KeyboardInterrupt:
        print("\n⏹️ Download interrupted by user")
        print("💾 Progress saved - run this script again to resume")
        return False
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("💡 Try running the script again")
        return False

def main():
    """Main function"""
    success = resume_qwen_image_download()
    
    if success:
        print("\n🎉 Ready to use!")
        print("💡 Next steps:")
        print("   1. Run: python tools/monitor_download.py  # Verify integrity")
        print("   2. Run: make smoke                        # Test the model")
    else:
        print("\n⚠️ Download incomplete")
        print("💡 Run this script again to continue")
        sys.exit(1)

if __name__ == "__main__":
    main()