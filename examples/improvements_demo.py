#!/usr/bin/env python3
"""
Demonstration of the new improvements in the Qwen-Image generator.
"""

import os
import sys

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..", "src")
sys.path.insert(0, src_dir)


def demonstrate_improvements():
    """Demonstrate the key improvements"""
    print("🎨 Qwen-Image Generator Improvements Demo")
    print("=" * 50)

    # 1. Device Policy Helper
    print("\n1. 🖥️ Device Policy Helper")
    from utils.devices import (
        check_vram_availability,
        get_attention_implementation,
        get_device_config,
        perform_preflight_checks,
    )

    device_config = get_device_config()
    print(f"   Device config: {device_config}")

    attn_impl = get_attention_implementation()
    print(f"   Attention implementation: {attn_impl}")

    vram_info = check_vram_availability()
    print(f"   VRAM: {vram_info['total_gb']:.1f}GB total")

    # 2. Pre-flight Checks
    print("\n2. 🚦 Pre-flight Checks")
    checks = perform_preflight_checks()
    print(f"   CUDA version: {checks['cuda_version']}")
    print(f"   Disk space: {checks['disk']['free_gb']:.1f}GB free")

    # 3. Pixel Window Governor
    print("\n3. 🖼️ Pixel Window Governor")
    from utils.devices import clamp_image_size

    width, height = clamp_image_size(2048, 2048)
    print(f"   Clamped 2048x2048 → {width}x{height}")

    # 4. Model Loading with Retry
    print("\n4. 🔄 Model Loading with Retry")
    print("   Model loading now includes:")
    print("   • Automatic retry on failures")
    print("   • Exponential backoff")
    print("   • Intelligent fallback strategies")
    print("   • Detailed error logging")

    # 5. Graceful Model Switching
    print("\n5. 🔄 Graceful Model Switching")
    print("   Safe model switching includes:")
    print("   • Memory cleanup")
    print("   • Garbage collection")
    print("   • Proper resource management")

    # 6. Eager Processor, Lazy Weights
    print("\n6. ⚡ Eager Processor, Lazy Weights")
    print("   • Processor loaded immediately (fast)")
    print("   • Model weights loaded on demand")
    print("   • UI remains responsive")

    # 7. QWEN_HOME Support
    print("\n7. 📁 QWEN_HOME Environment Variable")
    qwen_home = os.environ.get("QWEN_HOME", "~/.cache/huggingface/hub")
    print(f"   Cache directory: {qwen_home}")

    print("\n" + "=" * 50)
    print("🎉 All improvements are now active!")
    print("💡 These enhancements provide better reliability,")
    print("   performance, and user experience.")


if __name__ == "__main__":
    demonstrate_improvements()
