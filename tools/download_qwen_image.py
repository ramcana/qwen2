#!/usr/bin/env python3
"""
Download Qwen models for modern architecture support.
Supports both Qwen-Image (MMDiT text-to-image) and Qwen2-VL (multimodal) models.
"""
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

# Add src directory to path for model detection
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from model_detection_service import ModelDetectionService
    MODEL_DETECTION_AVAILABLE = True
except ImportError:
    MODEL_DETECTION_AVAILABLE = False
    print("⚠️ Model detection service not available, using basic functionality")


# Supported models with their configurations
SUPPORTED_MODELS = {
    "qwen-image": {
        "repo_id": "Qwen/Qwen-Image",
        "dest_dir": "Qwen-Image",
        "description": "MMDiT text-to-image model (optimized for T2I, ~60GB)",
        "architecture": "MMDiT",
        "use_case": "text-to-image generation",
        "expected_size_gb": 60,
        "performance": "2-5 seconds per step (when optimized)"
    },
    "qwen-image-edit": {
        "repo_id": "Qwen/Qwen-Image-Edit",
        "dest_dir": "Qwen-Image-Edit", 
        "description": "MMDiT image editing model (for editing tasks, ~60GB)",
        "architecture": "MMDiT",
        "use_case": "image editing and manipulation",
        "expected_size_gb": 60,
        "performance": "Slower for T2I (not optimized for text-to-image)"
    },
    "qwen2-vl-7b": {
        "repo_id": "Qwen/Qwen2-VL-7B-Instruct",
        "dest_dir": "Qwen2-VL-7B-Instruct",
        "description": "Multimodal language model (15GB)",
        "architecture": "Transformer",
        "use_case": "text understanding and image analysis",
        "expected_size_gb": 15,
        "performance": "Enhanced prompt understanding"
    },
    "qwen2-vl-2b": {
        "repo_id": "Qwen/Qwen2-VL-2B-Instruct", 
        "dest_dir": "Qwen2-VL-2B-Instruct",
        "description": "Compact multimodal model (4GB)",
        "architecture": "Transformer",
        "use_case": "lightweight text understanding",
        "expected_size_gb": 4,
        "performance": "Good prompt understanding, lower resource usage"
    }
}


def download_model(model_key: str, force: bool = False) -> bool:
    """Download a specific model."""
    if model_key not in SUPPORTED_MODELS:
        print(f"❌ Unknown model: {model_key}")
        print(f"Available models: {', '.join(SUPPORTED_MODELS.keys())}")
        return False
    
    model_config = SUPPORTED_MODELS[model_key]
    repo_id = model_config["repo_id"]
    dest = f"./models/{model_config['dest_dir']}"
    
    print(f"🚀 Downloading {model_config['description']}")
    print("=" * 70)
    print(f"📦 Model: {repo_id}")
    print(f"🏗️ Architecture: {model_config['architecture']}")
    print(f"🎯 Use Case: {model_config['use_case']}")
    print(f"📊 Expected Size: ~{model_config['expected_size_gb']}GB")
    print(f"⚡ Performance: {model_config['performance']}")
    print("")
    
    dest_path = Path(dest)
    
    # Check if model already exists
    if dest_path.exists() and not force:
        existing_files = list(dest_path.rglob("*"))
        if existing_files:
            print(f"⚠️ Model already exists at {dest}")
            print(f"📁 Found {len(existing_files)} files")
            
            if not force:
                response = input("Continue download anyway? (y/N): ").strip().lower()
                if response != 'y':
                    print("❌ Download cancelled")
                    return False
    
    dest_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"📥 Downloading {repo_id} to {dest}")
        print("⏳ This may take several minutes depending on your connection...")
        
        start_time = time.time()
        
        # Download to HuggingFace cache to avoid duplicates
        cached_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
            resume_download=True,
            local_files_only=False
        )
        
        # Create symlink to requested destination if needed
        if str(dest_path) != cached_path:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if dest_path.exists():
                if dest_path.is_symlink():
                    dest_path.unlink()
                else:
                    shutil.rmtree(dest_path)
            
            if os.name == 'nt':  # Windows
                shutil.copytree(cached_path, dest_path)
            else:  # Unix/Linux
                dest_path.symlink_to(cached_path, target_is_directory=True)
        
        download_time = time.time() - start_time
        
        print(f"✅ Successfully downloaded {repo_id}")
        print(f"📁 Cached at: {cached_path}")
        if str(dest_path) != cached_path:
            print(f"🔗 Linked to: {dest_path}")
        
        # Get final stats from cached location
        total_size = 0
        file_count = 0
        for file_path in Path(cached_path).rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        size_gb = total_size / (1024**3)
        print(f"📊 Downloaded: {file_count} files, {size_gb:.1f} GB")
        print(f"⏱️ Download time: {download_time:.1f} seconds")
        
        # Verify download completeness
        if size_gb < model_config['expected_size_gb'] * 0.8:
            print(f"⚠️ Warning: Downloaded size ({size_gb:.1f}GB) is smaller than expected ({model_config['expected_size_gb']}GB)")
            print("   The download might be incomplete. Consider running with --force to re-download.")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def detect_and_recommend():
    """Detect current models and provide recommendations."""
    if not MODEL_DETECTION_AVAILABLE:
        print("⚠️ Model detection not available")
        return
    
    print("🔍 Analyzing current model setup...")
    print("=" * 50)
    
    try:
        detector = ModelDetectionService()
        current_model = detector.detect_current_model()
        
        if current_model:
            print(f"📦 Current Model: {current_model.name}")
            print(f"📁 Path: {current_model.path}")
            print(f"📊 Size: {current_model.size_gb:.1f}GB")
            print(f"🏷️ Type: {current_model.model_type}")
            print(f"✅ Status: {current_model.download_status}")
            print(f"🎯 Optimal for T2I: {'Yes' if current_model.is_optimal else 'No'}")
            
            # Analyze architecture
            architecture = detector.detect_model_architecture(current_model)
            print(f"🏗️ Architecture: {architecture}")
            
            # Performance analysis
            perf_chars = detector.analyze_performance_characteristics(current_model)
            print(f"⚡ Expected Performance: {perf_chars['expected_generation_time']}")
            print(f"💾 Memory Usage: {perf_chars['memory_usage']}")
            
            if perf_chars['bottlenecks']:
                print(f"⚠️ Bottlenecks: {', '.join(perf_chars['bottlenecks'])}")
        else:
            print("❌ No Qwen models detected")
        
        # Check optimization needs
        optimization_needed = detector.is_optimization_needed()
        if optimization_needed:
            recommended = detector.get_recommended_model()
            print(f"\n💡 Optimization Recommended:")
            print(f"   Download: {recommended}")
            
            # Find the model key for the recommended model
            for key, config in SUPPORTED_MODELS.items():
                if config["repo_id"] == recommended:
                    print(f"   Command: python {sys.argv[0]} --model {key}")
                    break
        else:
            print(f"\n✅ Current setup is optimal for text-to-image generation")
        
        # Check Qwen2-VL capabilities
        qwen2_vl_info = detector.detect_qwen2_vl_capabilities()
        if qwen2_vl_info["available_models"]:
            print(f"\n🎭 Multimodal Capabilities Available:")
            for model in qwen2_vl_info["available_models"]:
                print(f"   • {model['name']} ({model['size_gb']:.1f}GB)")
        else:
            print(f"\n💡 Consider downloading Qwen2-VL for enhanced text understanding:")
            print(f"   Command: python {sys.argv[0]} --model qwen2-vl-7b")
        
    except Exception as e:
        print(f"❌ Detection failed: {e}")


def list_models():
    """List all supported models."""
    print("📦 Supported Models:")
    print("=" * 50)
    
    for key, config in SUPPORTED_MODELS.items():
        print(f"\n🔹 {key}")
        print(f"   Repository: {config['repo_id']}")
        print(f"   Description: {config['description']}")
        print(f"   Architecture: {config['architecture']}")
        print(f"   Use Case: {config['use_case']}")
        print(f"   Size: ~{config['expected_size_gb']}GB")
        print(f"   Performance: {config['performance']}")


def download_optimal_setup():
    """Download the optimal setup for text-to-image generation with multimodal support."""
    print("🎯 Downloading Optimal Setup for Modern Qwen Architecture")
    print("=" * 70)
    print("This will download:")
    print("• Qwen-Image (MMDiT text-to-image, 8GB)")
    print("• Qwen2-VL-7B-Instruct (multimodal understanding, 15GB)")
    print("• Total: ~23GB download")
    print("")
    
    response = input("Continue with optimal setup download? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Download cancelled")
        return False
    
    success = True
    
    # Download Qwen-Image first (primary model)
    print("\n" + "="*50)
    print("Step 1/2: Downloading Qwen-Image (Primary Model)")
    print("="*50)
    success &= download_model("qwen-image")
    
    # Download Qwen2-VL for multimodal support
    print("\n" + "="*50)
    print("Step 2/2: Downloading Qwen2-VL (Multimodal Support)")
    print("="*50)
    success &= download_model("qwen2-vl-7b")
    
    if success:
        print("\n🎉 Optimal Setup Download Complete!")
        print("=" * 50)
        print("✅ Qwen-Image: Fast text-to-image generation")
        print("✅ Qwen2-VL: Enhanced text understanding")
        print("")
        print("🚀 Next Steps:")
        print("• Try: python examples/optimized_text_to_image_demo.py")
        print("• Try: python examples/multimodal_integration_demo.py")
        print("• Run: python examples/performance_comparison_demo.py")
    else:
        print("\n❌ Some downloads failed")
        return False
    
    return True


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Download Qwen models for modern architecture support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model qwen-image          # Download fast text-to-image model
  %(prog)s --model qwen2-vl-7b         # Download multimodal model
  %(prog)s --optimal                   # Download optimal setup (both models)
  %(prog)s --detect                    # Analyze current setup
  %(prog)s --list                      # List all supported models
        """
    )
    
    parser.add_argument(
        "--model", 
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model to download"
    )
    parser.add_argument(
        "--optimal", 
        action="store_true",
        help="Download optimal setup (Qwen-Image + Qwen2-VL)"
    )
    parser.add_argument(
        "--detect", 
        action="store_true",
        help="Detect current models and provide recommendations"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all supported models"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force download even if model exists"
    )
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.list:
        list_models()
        return
    
    if args.detect:
        detect_and_recommend()
        return
    
    if args.optimal:
        success = download_optimal_setup()
        sys.exit(0 if success else 1)
    
    if args.model:
        success = download_model(args.model, args.force)
        if success:
            print(f"\n🎉 {args.model} download completed!")
            print("💡 Model ready for use with optimized pipeline")
            
            # Provide usage examples based on model type
            model_config = SUPPORTED_MODELS[args.model]
            if "qwen-image" in args.model:
                print("🚀 Try: python examples/optimized_text_to_image_demo.py")
            elif "qwen2-vl" in args.model:
                print("🚀 Try: python examples/multimodal_integration_demo.py")
        else:
            print(f"\n❌ {args.model} download failed")
        sys.exit(0 if success else 1)
    
    # No arguments provided - show help and detect current setup
    parser.print_help()
    print("\n" + "="*50)
    if MODEL_DETECTION_AVAILABLE:
        detect_and_recommend()
    else:
        print("💡 Use --model <model_name> to download a specific model")
        print("💡 Use --optimal to download the recommended setup")


if __name__ == "__main__":
    main()