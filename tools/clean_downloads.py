#!/usr/bin/env python3
"""
Clean up partial/corrupted downloads and restart fresh.
"""
import shutil
import sys
from pathlib import Path


def clean_model_directory(model_path: str) -> bool:
    """Clean up the model directory."""
    path = Path(model_path)
    
    if not path.exists():
        print(f"📂 Directory {model_path} doesn't exist - nothing to clean")
        return True
    
    try:
        # Get size before cleanup
        total_size = 0
        file_count = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        if file_count == 0:
            print(f"📂 Directory {model_path} is already empty")
            return True
        
        print(f"🗑️ Removing {file_count} files ({total_size / (1024**3):.2f} GB)")
        
        # Ask for confirmation
        response = input("❓ Are you sure you want to delete all downloaded files? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Cleanup cancelled")
            return False
        
        # Remove the directory
        shutil.rmtree(path)
        print(f"✅ Successfully cleaned {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to clean {model_path}: {e}")
        return False


def clean_temp_files(model_path: str) -> bool:
    """Clean up only temporary/partial files, keep complete ones."""
    path = Path(model_path)
    
    if not path.exists():
        print(f"📂 Directory {model_path} doesn't exist")
        return True
    
    # Find temporary files
    temp_patterns = ["*.tmp", "*.part", "*.download"]
    temp_files = []
    
    for pattern in temp_patterns:
        temp_files.extend(path.rglob(pattern))
    
    if not temp_files:
        print("✅ No temporary files found")
        return True
    
    print(f"🗑️ Found {len(temp_files)} temporary files:")
    for temp_file in temp_files:
        size = temp_file.stat().st_size
        print(f"   - {temp_file.name} ({size:,} bytes)")
    
    try:
        for temp_file in temp_files:
            temp_file.unlink()
        print(f"✅ Cleaned up {len(temp_files)} temporary files")
        return True
    except Exception as e:
        print(f"❌ Failed to clean temporary files: {e}")
        return False


def main():
    """Main cleanup function."""
    model_path = "./models/Qwen-Image-Edit"
    
    print("🧹 Model Download Cleanup Tool")
    print("=" * 40)
    
    print("\nOptions:")
    print("1. Clean temporary files only (recommended)")
    print("2. Clean everything and start fresh")
    print("3. Cancel")
    
    try:
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            print("\n🧹 Cleaning temporary files...")
            success = clean_temp_files(model_path)
            if success:
                print("\n💡 Now run: make models-cli")
        
        elif choice == "2":
            print("\n🧹 Cleaning everything...")
            success = clean_model_directory(model_path)
            if success:
                print("\n💡 Now run: make models-cli")
        
        elif choice == "3":
            print("❌ Cleanup cancelled")
            return
        
        else:
            print("❌ Invalid choice")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n❌ Cleanup cancelled")
        sys.exit(1)


if __name__ == "__main__":
    main()