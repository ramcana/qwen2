#!/usr/bin/env python3
"""
Memory-Optimized Qwen-Image-Edit Downloader
Handles CUDA out of memory errors and provides safe model loading
"""

import os
import sys
from pathlib import Path

import torch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def clear_gpu_memory():
    """Clear all GPU memory before starting"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear any remaining allocations
        torch.cuda.empty_cache()
        
        print("🧹 GPU memory cleared")
        
        # Check current memory usage
        allocated = torch.cuda.memory_allocated(0)
        total = torch.cuda.get_device_properties(0).total_memory
        print(f"📊 VRAM: {allocated/1e9:.1f}GB used / {total/1e9:.1f}GB total")
        return total - allocated
    return 0

def download_with_memory_optimization():
    """Download Qwen-Image-Edit with memory optimization"""
    
    print("🎨 Memory-Optimized Qwen-Image-Edit Downloader")
    print("=" * 55)
    
    # Step 1: Clear GPU memory
    available_memory = clear_gpu_memory()
    
    if available_memory < 8e9:  # Less than 8GB available
        print("⚠️ WARNING: Less than 8GB VRAM available")
        print("💡 Consider closing other GPU processes first")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ Download cancelled")
            return False
    
    # Step 2: Set memory optimization environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("✅ Memory optimization enabled")
    
    # Step 3: Import after clearing memory
    try:
        print("📦 Importing HuggingFace libraries...")
        from diffusers import DiffusionPipeline
        from huggingface_hub import repo_info, snapshot_download
        print("✅ Libraries loaded successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install required packages: pip install diffusers huggingface_hub")
        return False
    
    # Step 4: Download model with memory constraints
    repo_id = "Qwen/Qwen-Image-Edit"
    cache_dir = "./models/qwen-image-edit"
    
    try:
        print(f"🔍 Checking {repo_id} accessibility...")
        
        # Check if repository is accessible
        repo_data = repo_info(repo_id)
        total_size = sum(file.size for file in repo_data.siblings if file.size)
        print(f"📊 Model size: {total_size/1e9:.1f}GB")
        
        # Check local cache
        cache_path = Path(cache_dir)
        if cache_path.exists():
            local_files = list(cache_path.glob("**/*"))
            if local_files:
                print(f"📁 Found {len(local_files)} cached files")
        
        print("📥 Starting download with memory optimization...")
        print("⏳ This may take 10-30 minutes...")
        
        # Download with minimal memory footprint
        model_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False,
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
            max_workers=2,  # Reduced workers to save memory
            repo_type="model"
        )
        
        print("✅ Download completed!")
        print(f"📁 Model cached at: {model_path}")
        
        return model_path
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        
        if "CUDA out of memory" in str(e):
            print("\n💡 CUDA Memory Error Solutions:")
            print("1. Close other GPU applications")
            print("2. Restart your system to clear GPU memory")
            print("3. Try downloading on CPU only")
            
        return False

def test_model_loading(model_path):
    """Test if the downloaded model can be loaded safely"""
    
    print("\n🔍 Testing model loading...")
    
    # Clear memory before testing
    clear_gpu_memory()
    
    try:
        from qwen_edit_config import (
            apply_memory_optimizations,
            get_memory_optimized_config,
        )

        # Get optimized configuration
        config = get_memory_optimized_config()
        print(f"🔧 Using config: {config}")
        
        # Try loading with memory optimization
        from diffusers import DiffusionPipeline
        
        print("📦 Loading pipeline with memory optimization...")
        
        pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=config["torch_dtype"],
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            low_cpu_mem_usage=config["low_cpu_mem_usage"],
            cache_dir=config["cache_dir"]
        )
        
        # Apply memory optimizations
        pipeline = apply_memory_optimizations(pipeline)
        
        # Device mapping is already handled during loading with device_map='balanced'
        # Don't use .to("cuda") when device_map is set
        if not config.get("device_map"):
            if torch.cuda.is_available():
                print("🚀 Moving to GPU...")
                pipeline = pipeline.to("cuda")
        else:
            print("✅ Model loaded with balanced device mapping")
        
        print("✅ Model loading test successful!")
        
        # Check final memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            print(f"📊 Final VRAM usage: {allocated/1e9:.1f}GB / {total/1e9:.1f}GB ({100*allocated/total:.1f}%)")
        
        # Clean up
        del pipeline
        clear_gpu_memory()
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        
        if "CUDA out of memory" in str(e):
            print("\n💡 Memory Error Solutions:")
            print("1. Use CPU offloading: enable_sequential_cpu_offload=True")
            print("2. Reduce max_memory setting in config")
            print("3. Use fp16 instead of bfloat16")
            
        return False

def main():
    """Main execution function"""
    
    try:
        # Step 1: Download model
        model_path = download_with_memory_optimization()
        
        if not model_path:
            print("\n❌ Download failed. Please check the error messages above.")
            return False
        
        # Step 2: Test loading
        success = test_model_loading(model_path)
        
        if success:
            print("\n🎉 Qwen-Image-Edit is ready!")
            print("✨ Enhanced features now available:")
            print("  • Image-to-Image generation")
            print("  • Inpainting capabilities") 
            print("  • Advanced image editing")
            print("\n💡 Use the optimized configuration from qwen_edit_config.py")
        else:
            print("\n⚠️ Model downloaded but loading failed")
            print("💡 Try the memory optimization suggestions above")
        
        return success
        
    except KeyboardInterrupt:
        print("\n👋 Download interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)