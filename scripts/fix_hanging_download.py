#!/usr/bin/env python3
"""
Quick Fix for Hanging Qwen Model Download
Uses WAN orchestrator patterns to replace the problematic download
"""

import os
import sys
import subprocess
import time

def kill_hanging_processes():
    """Kill any hanging Python processes related to Qwen"""
    print("🛑 Stopping any hanging download processes...")
    
    try:
        # Kill any Python processes that might be hanging
        subprocess.run(['pkill', '-f', 'python.*qwen'], check=False)
        subprocess.run(['pkill', '-f', 'qwen_image_ui'], check=False)
        subprocess.run(['pkill', '-f', 'download.*qwen'], check=False)
        time.sleep(2)
        print("✅ Processes stopped")
    except Exception as e:
        print(f"⚠️ Could not stop processes: {e}")

def clear_gpu_memory():
    """Clear GPU memory"""
    print("🧹 Clearing GPU memory...")
    
    try:
        # Run a quick GPU memory clear
        result = subprocess.run([
            sys.executable, '-c', 
            '''
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared")
else:
    print("No GPU available")
'''
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ GPU memory cleared")
        else:
            print(f"⚠️ GPU clear warning: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⚠️ GPU clear timed out")
    except Exception as e:
        print(f"⚠️ Could not clear GPU: {e}")

def run_robust_download():
    """Run the WAN-powered robust download"""
    print("🚀 Starting WAN-powered robust download...")
    
    # Make sure the script is executable
    script_path = os.path.join(os.path.dirname(__file__), 'tools', 'wan_robust_download.py')
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        # Run the robust download script
        result = subprocess.run([
            sys.executable, script_path,
            '--models', 'Qwen/Qwen-Image',
            '--max-retries', '5'
        ], cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print("✅ Robust download completed successfully!")
            return True
        else:
            print(f"❌ Robust download failed with code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to run robust download: {e}")
        return False

def main():
    print("🔧 Qwen Download Fix Tool")
    print("=" * 30)
    print("This tool will:")
    print("1. Stop any hanging download processes")
    print("2. Clear GPU memory")
    print("3. Start a robust WAN-powered download")
    print()
    
    # Step 1: Kill hanging processes
    kill_hanging_processes()
    
    # Step 2: Clear GPU memory
    clear_gpu_memory()
    
    # Step 3: Run robust download
    success = run_robust_download()
    
    if success:
        print("\n🎉 Download fix completed successfully!")
        print("\n🚀 Next steps:")
        print("1. The model should now be downloaded properly")
        print("2. You can start the UI with: ./scripts/safe_restart.sh")
        print("3. Or use: python launch.py")
    else:
        print("\n❌ Download fix failed")
        print("\n🔧 Manual alternatives:")
        print("1. Try: python tools/wan_robust_download.py")
        print("2. Check your internet connection")
        print("3. Verify HuggingFace Hub access")
        print("4. Check available disk space (need ~50GB)")

if __name__ == "__main__":
    main()
