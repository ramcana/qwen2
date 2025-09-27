#!/usr/bin/env python3
"""
Clear stale HuggingFace download lock files.
"""
import os
import time
from pathlib import Path


def find_lock_files(base_path: str) -> list:
    """Find all .lock files in the download cache."""
    path = Path(base_path)
    if not path.exists():
        return []
    
    lock_files = list(path.rglob("*.lock"))
    return lock_files


def is_stale_lock(lock_file: Path, max_age_seconds: int = 300) -> bool:
    """Check if a lock file is stale (older than max_age_seconds)."""
    try:
        # Get file modification time
        mtime = lock_file.stat().st_mtime
        current_time = time.time()
        age = current_time - mtime
        
        return age > max_age_seconds
    except (OSError, FileNotFoundError):
        return True  # If we can't stat it, consider it stale


def clear_stale_locks(base_path: str, max_age_seconds: int = 300) -> int:
    """Clear stale lock files."""
    lock_files = find_lock_files(base_path)
    
    if not lock_files:
        print("✅ No lock files found")
        return 0
    
    print(f"🔍 Found {len(lock_files)} lock files")
    
    cleared_count = 0
    for lock_file in lock_files:
        try:
            if is_stale_lock(lock_file, max_age_seconds):
                age = time.time() - lock_file.stat().st_mtime
                print(f"🗑️ Removing stale lock: {lock_file.name} (age: {age:.1f}s)")
                lock_file.unlink()
                cleared_count += 1
            else:
                age = time.time() - lock_file.stat().st_mtime
                print(f"⏳ Keeping recent lock: {lock_file.name} (age: {age:.1f}s)")
        except Exception as e:
            print(f"❌ Failed to remove {lock_file}: {e}")
    
    return cleared_count


def main():
    """Main function."""
    cache_path = "./models/Qwen-Image-Edit/.cache"
    
    print("🔓 HuggingFace Lock File Cleaner")
    print("=" * 40)
    
    if not Path(cache_path).exists():
        print(f"📂 Cache directory not found: {cache_path}")
        print("💡 This is normal if no downloads have started yet")
        return
    
    # Clear locks older than 5 minutes (300 seconds)
    cleared = clear_stale_locks(cache_path, max_age_seconds=300)
    
    if cleared > 0:
        print(f"\n✅ Cleared {cleared} stale lock files")
        print("💡 You can now retry the download with: make models-cli")
    else:
        print("\n✅ No stale locks found")
        
        # Show active locks
        all_locks = find_lock_files(cache_path)
        if all_locks:
            print(f"⏳ {len(all_locks)} active locks (recent downloads in progress)")


if __name__ == "__main__":
    main()