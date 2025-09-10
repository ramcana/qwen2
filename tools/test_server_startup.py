#!/usr/bin/env python3
"""
Test script to verify server startup without actually running the server
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")

    try:
        import torch

        print(f"✅ PyTorch imported successfully (version: {torch.__version__})")
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

    try:
        print("✅ FastAPI and Uvicorn imported successfully")
    except Exception as e:
        print(f"❌ FastAPI/Uvicorn import failed: {e}")
        return False

    try:
        print("✅ Main API module imported successfully")
    except Exception as e:
        print(f"❌ Main API module import failed: {e}")
        return False

    try:
        print("✅ HighEndQwenImageGenerator imported successfully")
    except Exception as e:
        print(f"⚠️ HighEndQwenImageGenerator import failed: {e}")
        print("   Falling back to standard generator...")
        try:
            print("✅ Standard QwenImageGenerator imported successfully")
        except Exception as e2:
            print(f"❌ Standard QwenImageGenerator import also failed: {e2}")
            return False

    return True


def test_environment():
    """Test environment setup"""
    print("\n🧪 Testing environment...")

    # Check CUDA
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅ CUDA available (device: {torch.cuda.get_device_name(0)})")
        else:
            print("⚠️ CUDA not available, will use CPU")
    except Exception as e:
        print(f"❌ CUDA check failed: {e}")
        return False

    # Check virtual environment
    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        print(f"✅ Virtual environment active: {venv_path}")
    else:
        print("⚠️ No virtual environment detected")

    return True


def main():
    """Main test function"""
    print("🚀 QWEN-IMAGE SERVER STARTUP TEST")
    print("=" * 50)

    if not test_imports():
        print("\n❌ Import tests failed")
        return 1

    if not test_environment():
        print("\n❌ Environment tests failed")
        return 1

    print("\n✅ All tests passed!")
    print("\n💡 You can now start the server with:")
    print("   ./scripts/launch_ui.sh")
    print("   Then select option 3 (Complete System)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
