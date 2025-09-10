#!/usr/bin/env python3
"""
Robust Qwen Model Downloader with Rust Accelerator Support
Implements fast, reliable downloads with resume capability and error handling
"""

import gc
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Enable Rust accelerator if available
HF_HOME = os.getenv("HF_HOME", "")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # Rust accel if installed


def robust_download(
    repo_id: str,
    out_dir: str,
    retries: int = 3,
    max_workers: int = 8,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Robust download function with retry mechanism and Rust accelerator support

    Args:
        repo_id: HuggingFace repository ID
        out_dir: Local directory to save the model
        retries: Number of retry attempts
        max_workers: Number of concurrent download workers
        cache_dir: Optional cache directory override

    Returns:
        str: Path to downloaded model

    Raises:
        RuntimeError: If download fails after all retries
    """
    from huggingface_hub import snapshot_download

    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Set cache directory if provided
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir

    print(f"🚀 Starting robust download of {repo_id}")
    print(f"📁 Output directory: {out_dir}")
    print(f"🔧 Max workers: {max_workers}")
    print(f"🔄 Retries: {retries}")

    # Check if Rust transfer is enabled
    if os.getenv("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        try:
            import hf_transfer

            print("⚡ Rust accelerator enabled for faster downloads")
        except ImportError:
            print("⚠️ Rust accelerator not available, using default downloader")
    else:
        print("🔧 Using default downloader")

    for attempt in range(1, retries + 1):
        try:
            print(f"📥 Download attempt {attempt}/{retries}")

            # Perform download with resume capability
            result = snapshot_download(
                repo_id,
                local_dir=out_dir,
                local_dir_use_symlinks=False,  # Windows-safe
                resume_download=True,
                max_workers=max_workers
                if attempt == 1
                else max_workers // 2,  # Reduce workers on retry
            )

            print(f"✅ Download completed successfully!")
            return result

        except Exception as e:
            print(f"[Attempt {attempt}] Download failed: {e}")

            # Clean up memory
            gc.collect()

            # Wait before retry with exponential backoff
            if attempt < retries:
                wait_time = 3 * attempt
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

                # Optional: Clean incomplete shards before next attempt
                _cleanup_incomplete_shards(out_dir)
            else:
                raise RuntimeError(
                    f"Failed to download {repo_id} after {retries} attempts"
                )


def _cleanup_incomplete_shards(download_dir: str):
    """
    Clean up incomplete shard files to prevent download corruption

    Args:
        download_dir: Directory to scan for incomplete files
    """
    try:
        download_path = Path(download_dir)
        if not download_path.exists():
            return

        # Find and remove incomplete files
        incomplete_files = list(download_path.glob("**/*.incomplete"))
        tmp_files = list(download_path.glob("**/*.tmp"))

        for file_path in incomplete_files + tmp_files:
            try:
                file_path.unlink()
                print(f"🗑️ Removed incomplete file: {file_path.name}")
            except Exception as e:
                print(f"⚠️ Could not remove {file_path}: {e}")

    except Exception as e:
        print(f"⚠️ Error during cleanup: {e}")


def main():
    """Main CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Robust Qwen Model Downloader")
    parser.add_argument("repo_id", help="HuggingFace repository ID")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries")
    parser.add_argument("--workers", type=int, default=8, help="Max download workers")
    parser.add_argument("--cache-dir", help="Cache directory")

    args = parser.parse_args()

    try:
        result = robust_download(
            repo_id=args.repo_id,
            out_dir=args.out_dir,
            retries=args.retries,
            max_workers=args.workers,
            cache_dir=args.cache_dir,
        )
        print(f"🎉 Model successfully downloaded to: {result}")
        return 0
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
