#!/usr/bin/env python3
"""
Quick Fix for Qwen-Image-Edit Download Issues
Run this to fix the advanced model download problems
"""

import os
import subprocess
import sys
from pathlib import Path


def install_huggingface_hub():
    """Ensure huggingface_hub is installed"""
    try:
        import huggingface_hub

        print("✅ huggingface_hub already installed")
        return True
    except ImportError:
        print("📦 Installing huggingface_hub...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "huggingface_hub>=0.19.0"]
            )
            print("✅ huggingface_hub installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install huggingface_hub: {e}")
            return False


def fix_download_issues():
    """Apply fixes for download issues"""
    print("🔧 Applying fixes for Qwen-Image-Edit download issues...")

    # 1. Install missing dependency
    if not install_huggingface_hub():
        return False

    # 2. Clear problematic cache if exists
    cache_dirs = [
        "./models/qwen-image-edit",
        os.path.expanduser("~/.cache/huggingface/transformers"),
        os.path.expanduser("~/.cache/huggingface/hub"),
    ]

    for cache_dir in cache_dirs:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            # Look for incomplete downloads
            tmp_files = list(cache_path.glob("**/*.tmp"))
            incomplete_files = list(cache_path.glob("**/*.incomplete"))

            if tmp_files or incomplete_files:
                print(f"🧹 Cleaning incomplete downloads in {cache_dir}")
                for tmp_file in tmp_files + incomplete_files:
                    try:
                        tmp_file.unlink()
                        print(f"  Removed: {tmp_file.name}")
                    except Exception as e:
                        print(f"  Warning: Could not remove {tmp_file}: {e}")

    # 3. Test the enhanced downloader
    print("🧪 Testing enhanced download method...")
    try:
        from tools.download_qwen_edit_hub import QwenEditDownloader

        downloader = QwenEditDownloader()
        status = downloader.check_model_status()

        if status["repo_accessible"]:
            print("✅ Repository is accessible")
            if status["locally_available"]:
                print("✅ Model partially or fully downloaded")
            else:
                print("📥 Model needs to be downloaded")
            return True
        else:
            print("❌ Cannot access Qwen repository")
            return False

    except ImportError as e:
        print(f"⚠️ Enhanced downloader not available: {e}")
        return False


def run_enhanced_download():
    """Run the enhanced download process"""
    print("\n🚀 Starting enhanced download process...")

    try:
        # Run the enhanced downloader
        from tools.download_qwen_edit_hub import QwenEditDownloader

        downloader = QwenEditDownloader()
        success = downloader.download_with_progress(resume=True)

        if success:
            print("✅ Enhanced download completed successfully!")
            return True
        else:
            print("⚠️ Download incomplete but can be resumed")
            return False

    except Exception as e:
        print(f"❌ Enhanced download failed: {e}")
        print("\n💡 Fallback: Try manual download:")
        print("python tools/download_qwen_edit.py")
        return False


def main():
    print("🎯 Qwen-Image-Edit Download Fix Utility")
    print("=" * 50)

    print("This script will:")
    print("1. Install missing dependencies")
    print("2. Clean up incomplete downloads")
    print("3. Use enhanced HuggingFace Hub API")
    print("4. Provide resume capability")

    choice = input("\n🚀 Continue with fixes? (y/N): ").strip().lower()

    if choice not in ["y", "yes"]:
        print("👋 Exiting without changes")
        return

    # Apply fixes
    if not fix_download_issues():
        print("❌ Fix process failed")
        return

    print("✅ Fixes applied successfully!")

    # Ask about download
    download_choice = input("\n📥 Download Qwen-Image-Edit now? (y/N): ").strip().lower()

    if download_choice in ["y", "yes"]:
        run_enhanced_download()
    else:
        print("\n💡 To download later, run:")
        print("python tools/download_qwen_edit_hub.py")

    print("\n🎉 Setup complete! Enhanced features should now work.")


if __name__ == "__main__":
    main()
