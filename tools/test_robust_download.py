#!/usr/bin/env python3
"""
Test script for the robust download utility
Demonstrates usage of the Rust-accelerated downloader
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_robust_download():
    """Test the robust download functionality"""
    print("🧪 Testing Robust Download Utility")
    print("=" * 40)
    
    try:
        # Import our robust download utility
        from tools.robust_download import robust_download
        
        # Test with a small model for demonstration
        # In practice, you would use the full Qwen model
        repo_id = "Qwen/Qwen2-0.5B"  # Small test model
        out_dir = "./test_model"
        
        print(f"🚀 Testing download of {repo_id}")
        print(f"📁 Output directory: {out_dir}")
        
        # Perform the download
        result = robust_download(
            repo_id=repo_id,
            out_dir=out_dir,
            retries=2,
            max_workers=4
        )
        
        print(f"✅ Download completed successfully!")
        print(f"📁 Model saved to: {result}")
        
        # Verify the download
        result_path = Path(result)
        if result_path.exists():
            files = list(result_path.glob("*"))
            print(f"📋 Downloaded {len(files)} files")
            for file in files[:5]:  # Show first 5 files
                print(f"   • {file.name}")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more files")
                
            return True
        else:
            print("❌ Download directory not found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Robust Download Utility Test")
    print("=" * 40)
    
    success = test_robust_download()
    
    if success:
        print("\n🎉 All tests passed!")
        print("💡 You can now use the robust_download.py script for faster model downloads")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())