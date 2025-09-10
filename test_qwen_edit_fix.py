#!/usr/bin/env python3
"""
Test Qwen-Image-Edit integration with memory optimization
"""

import os
import sys

import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def test_qwen_edit_integration():
    """Test Qwen-Image-Edit integration with the main system"""

    print("🧪 Testing Qwen-Image-Edit Integration")
    print("=" * 40)

    try:
        # Import configurations
        from qwen_edit_config import (
            apply_memory_optimizations,
            clear_gpu_cache,
            get_memory_optimized_config,
        )

        # Clear GPU cache first
        clear_gpu_cache()

        # Get optimized config
        config = get_memory_optimized_config()
        print(
            f"🔧 Using config: device_map={config.get('device_map')}, max_memory={config.get('max_memory')}"
        )

        # Import pipeline
        from diffusers import DiffusionPipeline

        print("📦 Loading Qwen-Image-Edit pipeline...")

        # Load with optimized settings
        pipeline = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit",
            torch_dtype=config["torch_dtype"],
            device_map=config.get("device_map"),
            max_memory=config.get("max_memory"),
            low_cpu_mem_usage=config["low_cpu_mem_usage"],
            cache_dir=config["cache_dir"],
        )

        # Apply optimizations
        pipeline = apply_memory_optimizations(pipeline)

        print("✅ Pipeline loaded successfully!")

        # Check pipeline components
        components = []
        if hasattr(pipeline, "vae"):
            components.append(f"VAE: {pipeline.vae.device}")
        if hasattr(pipeline, "text_encoder"):
            components.append(f"Text Encoder: {pipeline.text_encoder.device}")
        if hasattr(pipeline, "unet"):
            components.append(f"UNet: {pipeline.unet.device}")
        if hasattr(pipeline, "scheduler"):
            components.append("Scheduler: present")

        print("🔍 Pipeline components:")
        for component in components:
            print(f"  • {component}")

        # Check memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            print(
                f"📊 VRAM usage: {allocated/1e9:.1f}GB / {total/1e9:.1f}GB ({100*allocated/total:.1f}%)"
            )

        # Clean up
        del pipeline
        clear_gpu_cache()

        print("✅ Integration test passed!")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Main test function"""
    success = test_qwen_edit_integration()

    if success:
        print("\n🎉 Qwen-Image-Edit is ready for use!")
        print("💡 You can now use enhanced features in your UI")
    else:
        print("\n❌ Integration test failed")
        print("💡 Check the error messages above")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
