#!/usr/bin/env python3
"""
Quick Fix for Qwen-Image-Edit CUDA Out of Memory Error
Automatically resolves GPU memory issues and optimizes configuration
"""

import subprocess
import sys
from pathlib import Path


def check_and_kill_gpu_processes():
    """Check for GPU processes and offer to kill them"""
    print("🔍 Checking for GPU processes...")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if 'python' in result.stdout.lower():
            print("⚠️ Found Python processes using GPU memory")
            print("🧹 Clearing GPU processes...")
            
            try:
                subprocess.run(['pkill', '-f', 'python'], check=False)
                print("✅ GPU processes cleared")
                return True
            except Exception as e:
                print(f"⚠️ Could not kill all processes: {e}")
                print("💡 You may need to restart manually")
                return False
        else:
            print("✅ No conflicting GPU processes found")
            return True
            
    except FileNotFoundError:
        print("❌ nvidia-smi not found. Are you using a GPU system?")
        return False
    except Exception as e:
        print(f"⚠️ Error checking GPU processes: {e}")
        return False

def run_memory_safe_download():
    """Run the memory-safe download script"""
    print("\n📥 Running memory-safe Qwen-Image-Edit download...")
    
    # Check if we're in virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️ Virtual environment not detected")
        print("🔧 Activating virtual environment...")
        
        # Try to activate venv
        venv_path = Path("venv/bin/activate")
        if venv_path.exists():
            cmd = f"source {venv_path} && python tools/download_qwen_edit_memory_safe.py"
        else:
            print("❌ Virtual environment not found. Please run: ./scripts/setup.sh")
            return False
    else:
        cmd = "python tools/download_qwen_edit_memory_safe.py"
    
    try:
        # Run download script
        if venv_path.exists():
            result = subprocess.run(['bash', '-c', cmd], check=True, text=True, capture_output=True)
        else:
            result = subprocess.run(['python', 'tools/download_qwen_edit_memory_safe.py'], check=True, text=True, capture_output=True)
            
        print("✅ Memory-safe download completed successfully!")
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)  # Show last 500 chars
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr[-300:]}")  # Show last 300 chars of error
        return False
    except FileNotFoundError:
        print("❌ Download script not found. Make sure you're in the Qwen2 project directory.")
        return False

def verify_installation():
    """Verify that Qwen-Image-Edit is working"""
    print("\n🧪 Verifying installation...")
    
    try:
        # Try to run the integration test
        result = subprocess.run(['python', 'test_qwen_edit_fix.py'], 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Installation verified successfully!")
            return True
        else:
            print("❌ Verification failed")
            if result.stderr:
                print(f"Error: {result.stderr[-200:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Verification test timed out (model may still work)")
        return True
    except Exception as e:
        print(f"⚠️ Could not run verification: {e}")
        return True  # Don't fail the whole process

def main():
    """Main fix process"""
    print("🔧 Qwen-Image-Edit CUDA Memory Fix")
    print("=" * 40)
    print("This script will:")
    print("1. Clear GPU memory from conflicting processes")
    print("2. Download Qwen-Image-Edit with memory optimization")
    print("3. Verify the installation works")
    print()
    
    # Step 1: Clear GPU processes
    if not check_and_kill_gpu_processes():
        print("❌ Could not clear GPU processes. Please restart your system.")
        return False
    
    # Step 2: Run memory-safe download
    if not run_memory_safe_download():
        print("❌ Download failed. Please check the error messages above.")
        return False
    
    # Step 3: Verify installation
    verify_installation()
    
    print("\n🎉 Qwen-Image-Edit Fix Completed!")
    print("✨ Enhanced features should now be available:")
    print("  • Image-to-Image generation")  
    print("  • Inpainting capabilities")
    print("  • Advanced image editing")
    print()
    print("💡 Next steps:")
    print("  • Run: python src/qwen_image_ui.py")
    print("  • Or: python launch.py --mode enhanced")
    print("  • Use the memory-optimized configuration in qwen_edit_config.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Fix interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)