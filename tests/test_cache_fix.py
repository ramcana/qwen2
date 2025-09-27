"""
Test script to verify the cache fix works correctly
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from model_download_manager import ModelDownloadManager, ModelDownloadConfig


def test_cache_fix():
    """Test that the fixed ModelDownloadManager uses cache correctly"""
    
    print("🧪 Testing Cache Fix")
    print("=" * 50)
    
    # Create manager
    manager = ModelDownloadManager()
    
    # Test configuration
    config = ModelDownloadConfig(
        model_name="Qwen/Qwen-Image",
        verify_integrity=False  # Skip for test
    )
    
    print(f"📊 Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Cache dir: {config.cache_dir or manager.cache_dir}")
    print(f"  Uses cache_dir: {'cache_dir' in str(config.__dict__)}")
    print(f"  No local_dir: {'local_dir' not in str(config.__dict__)}")
    
    # Test model path resolution
    print(f"\n🔍 Testing model path resolution:")
    
    # Check if model exists in cache
    model_path = manager._get_local_model_path("Qwen/Qwen-Image")
    if model_path:
        print(f"✅ Found model at: {model_path}")
        print(f"📁 Path type: {'HF Cache' if '.cache/huggingface' in str(model_path) else 'Local'}")
    else:
        print(f"❌ Model not found in cache")
    
    # Test get_model_path method
    public_path = manager.get_model_path("Qwen/Qwen-Image")
    if public_path:
        print(f"✅ Public path: {public_path}")
    else:
        print(f"❌ No public path available")
    
    # Check for local models directory
    local_models = Path("./models")
    if local_models.exists():
        local_items = list(local_models.iterdir())
        local_sizes = []
        for item in local_items:
            if item.is_dir():
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (1024**3)
                local_sizes.append((item.name, size))
        
        print(f"\n📁 Local models directory:")
        for name, size in local_sizes:
            status = "⚠️ Large" if size > 1.0 else "✅ Small"
            print(f"  {name}: {size:.1f}GB {status}")
        
        total_local = sum(size for _, size in local_sizes)
        print(f"  Total local: {total_local:.1f}GB")
        
        if total_local < 1.0:
            print(f"✅ Local storage minimized successfully!")
        else:
            print(f"⚠️ Still has large local files")
    else:
        print(f"✅ No local models directory")
    
    print(f"\n🎯 Cache Fix Status:")
    
    # Check if the fix is working
    fixes_applied = []
    
    # Check 1: ModelDownloadConfig doesn't have local_dir
    if not hasattr(config, 'local_dir'):
        fixes_applied.append("✅ ModelDownloadConfig uses cache_dir")
    else:
        fixes_applied.append("❌ ModelDownloadConfig still has local_dir")
    
    # Check 2: Model path points to cache
    if model_path and '.cache/huggingface' in str(model_path):
        fixes_applied.append("✅ Model path points to HF cache")
    elif model_path:
        fixes_applied.append("⚠️ Model path points to local directory")
    else:
        fixes_applied.append("❓ No model path found")
    
    # Check 3: Local storage is minimal
    if local_models.exists():
        total_local = sum(
            sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (1024**3)
            for item in local_models.iterdir() if item.is_dir()
        )
        if total_local < 1.0:
            fixes_applied.append("✅ Local storage minimized")
        else:
            fixes_applied.append("⚠️ Local storage still large")
    else:
        fixes_applied.append("✅ No local models directory")
    
    for fix in fixes_applied:
        print(f"  {fix}")
    
    # Overall status
    success_count = sum(1 for fix in fixes_applied if fix.startswith("✅"))
    total_count = len(fixes_applied)
    
    print(f"\n📊 Overall Status: {success_count}/{total_count} checks passed")
    
    if success_count == total_count:
        print("🎉 Cache fix is working correctly!")
        return True
    else:
        print("⚠️ Some issues remain")
        return False


if __name__ == "__main__":
    success = test_cache_fix()
    sys.exit(0 if success else 1)