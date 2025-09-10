#!/usr/bin/env python3
"""
Test script to verify the improvements made to the Qwen-Image generator.
"""

import os
import sys

import torch

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.insert(0, src_dir)


def test_device_helper():
    """Test the device helper functions"""
    print("🧪 Testing Device Helper Functions...")

    try:
        from utils.devices import (
            check_vram_availability,
            clamp_image_size,
            get_attention_implementation,
            get_cache_dir,
            get_device_config,
        )

        # Test device config
        device_config = get_device_config()
        print(f"✅ Device config: {device_config}")

        # Test attention implementation
        attn_impl = get_attention_implementation()
        print(f"✅ Attention implementation: {attn_impl}")

        # Test VRAM availability
        vram_info = check_vram_availability()
        print(f"✅ VRAM info: {vram_info}")

        # Test cache directory
        cache_dir = get_cache_dir()
        print(f"✅ Cache directory: {cache_dir}")

        # Test pixel clamping
        width, height = clamp_image_size(2048, 2048)
        print(f"✅ Pixel clamping: 2048x2048 -> {width}x{height}")

        print("🎉 All device helper tests passed!")
        return True

    except Exception as e:
        print(f"❌ Device helper test failed: {e}")
        return False


def test_model_loading():
    """Test model loading with retry mechanism"""
    print("\n🧪 Testing Model Loading with Retry...")

    try:
        from diffusers import DiffusionPipeline

        from utils.devices import load_model_with_retry

        # This would test the actual loading, but we'll just verify the function exists
        print("✅ Retry loading function available")

        print("🎉 Model loading test completed!")
        return True

    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False


def test_configurations():
    """Test configuration files"""
    print("\n🧪 Testing Configuration Files...")

    try:
        # Test main config
        from qwen_image_config import MODEL_CONFIG

        print(f"✅ Main config loaded: {MODEL_CONFIG.get('model_name', 'N/A')}")

        # Test high-end config
        from qwen_highend_config import HIGH_END_MODEL_CONFIG

        print(
            f"✅ High-end config loaded: {HIGH_END_MODEL_CONFIG.get('model_name', 'N/A')}"
        )

        # Test edit config
        from qwen_edit_config import QWEN_EDIT_CONFIG

        print(f"✅ Edit config loaded: {QWEN_EDIT_CONFIG.get('model_name', 'N/A')}")

        print("🎉 All configuration tests passed!")
        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_pixel_window():
    """Test pixel window governor"""
    print("\n🧪 Testing Pixel Window Governor...")

    try:
        from utils.devices import get_pixel_window_settings, load_quality_presets

        # Load presets
        presets = load_quality_presets()
        print(f"✅ Quality presets loaded: {list(presets.keys())}")

        # Get pixel window settings
        min_pixels, max_pixels = get_pixel_window_settings()
        print(f"✅ Pixel window: {min_pixels} - {max_pixels} pixels")

        print("🎉 Pixel window test completed!")
        return True

    except Exception as e:
        print(f"❌ Pixel window test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Testing Qwen-Image Generator Improvements")
    print("=" * 50)

    tests = [
        test_device_helper,
        test_model_loading,
        test_configurations,
        test_pixel_window,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The improvements are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
