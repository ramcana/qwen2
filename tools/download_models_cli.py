#!/usr/bin/env python3
"""
Alternative model downloader using huggingface-cli for better reliability.
"""
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, cwd: str = None) -> bool:
    """Run a command and return success status."""
    try:
        print(f"🔧 Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        return False


def check_cli_available() -> bool:
    """Check if huggingface-cli is available."""
    try:
        result = subprocess.run(
            ["huggingface-cli", "--help"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_with_cli(repo_id: str, dest: str) -> bool:
    """Download using huggingface-cli which handles resuming better."""
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Use huggingface-cli download command
    cmd = [
        "huggingface-cli",
        "download",
        repo_id,
        "--local-dir", dest,
        "--local-dir-use-symlinks", "False"
    ]
    
    return run_command(cmd)


def main():
    """Main download function using CLI."""
    repo_id = "Qwen/Qwen-Image-Edit"
    dest = "./models/Qwen-Image-Edit"
    
    print("🚀 Qwen Image Edit Model Downloader (CLI Method)")
    print("=" * 60)
    
    # Check if CLI is available
    if not check_cli_available():
        print("❌ huggingface-cli not found!")
        print("💡 Install with: pip install huggingface_hub[cli]")
        print("💡 Or use: python tools/download_models.py")
        sys.exit(1)
    
    print("✅ huggingface-cli found")
    
    # Show existing files
    dest_path = Path(dest)
    if dest_path.exists():
        existing_files = list(dest_path.rglob("*"))
        file_count = len([f for f in existing_files if f.is_file()])
        if file_count > 0:
            total_size = sum(f.stat().st_size for f in existing_files if f.is_file())
            print(f"📂 Found {file_count} existing files ({total_size / (1024**3):.2f} GB)")
            print("🔄 CLI will automatically resume incomplete downloads")
    
    # Download
    print(f"\n📥 Downloading {repo_id}...")
    success = download_with_cli(repo_id, dest)
    
    if success:
        print("\n🎉 Download completed successfully!")
        
        # Show final stats
        final_files = list(dest_path.rglob("*"))
        final_count = len([f for f in final_files if f.is_file()])
        final_size = sum(f.stat().st_size for f in final_files if f.is_file())
        print(f"📊 Final: {final_count} files, {final_size / (1024**3):.2f} GB")
    else:
        print("\n❌ Download failed")
        print("💡 Try running the command again - CLI automatically resumes")
        sys.exit(1)


if __name__ == "__main__":
    main()