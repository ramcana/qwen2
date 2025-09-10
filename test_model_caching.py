#!/usr/bin/env python3
"""
Test script to verify model caching is working correctly
"""

import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_caching():
    """Test that models are properly cached and not redownloaded"""
    print("🔍 Testing model caching...")
    
    # Check QWEN_HOME environment variable
    qwen_home = os.environ.get('QWEN_HOME', '')
    print(f"QWEN_HOME: {qwen_home}")
    
    # Check model directories
    model_dirs = [
        "./models/Qwen-Image",
        "./models/qwen-image",
        os.path.join(qwen_home, "models", "qwen-image") if qwen_home else None,
        os.path.expanduser("~/.cache/huggingface/hub")
    ]
    
    model_dirs = [d for d in model_dirs if d]  # Remove None values
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            print(f"✅ Found model directory: {model_dir}")
            # Count files in directory
            try:
                file_count = sum([len(files) for r, d, files in os.walk(model_dir)])
                print(f"   Files in directory: {file_count}")
            except Exception as e:
                print(f"   Error counting files: {e}")
        else:
            print(f"❌ Model directory not found: {model_dir}")
    
    # Check if QWEN_HOME is properly set
    if not qwen_home:
        print("⚠️ QWEN_HOME is not set. This may cause model redownloading issues.")
        print("💡 Recommendation: Set QWEN_HOME to your project directory:")
        print("   export QWEN_HOME=/home/ramji_t/projects/Qwen2")
    else:
        print("✅ QWEN_HOME is properly set")
        
        # Check if cache directory matches expected structure
        cache_dir = os.path.join(qwen_home, "models", "qwen-image")
        if os.path.exists(cache_dir):
            print(f"✅ Cache directory exists: {cache_dir}")
            
            # Check for HuggingFace structure
            hf_structure = os.path.join(cache_dir, "models--Qwen--Qwen-Image")
            if os.path.exists(hf_structure):
                print(f"✅ HuggingFace model structure found: {hf_structure}")
            else:
                print(f"⚠️ HuggingFace model structure not found in: {cache_dir}")
        else:
            print(f"❌ Cache directory does not exist: {cache_dir}")
    
    print("\n📋 Recommendations:")
    print("1. Ensure QWEN_HOME is set to your project directory")
    print("2. Verify that model files are in the correct cache directory")
    print("3. If models are still downloading, check disk space and permissions")

if __name__ == "__main__":
    test_model_caching()