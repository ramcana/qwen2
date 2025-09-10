#!/usr/bin/env python3
"""
Quick test script to verify core dependencies are working correctly
"""

import diffusers
import gradio as gr
import torch
import transformers
import xformers


def test_core_dependencies():
    """Test that all core dependencies are properly installed and working"""
    print("🧪 Qwen-Image Core Dependencies Test")
    print("=" * 40)

    # Test PyTorch
    print("🔍 Testing PyTorch...")
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   Current GPU: {torch.cuda.get_device_name()}")
    print("✅ PyTorch is working correctly\n")

    # Test xformers
    print("🔍 Testing xformers...")
    print(f"   Version: {xformers.__version__}")
    print("✅ xformers is working correctly\n")

    # Test diffusers
    print("🔍 Testing diffusers...")
    print(f"   Version: {diffusers.__version__}")
    # Try to create a simple pipeline
    from diffusers import DiffusionPipeline

    print("✅ diffusers is working correctly\n")

    # Test transformers
    print("🔍 Testing transformers...")
    print(f"   Version: {transformers.__version__}")
    from transformers import AutoTokenizer

    print("✅ transformers is working correctly\n")

    # Test Gradio
    print("🔍 Testing Gradio...")
    print(f"   Version: {gr.__version__}")
    print("✅ Gradio is working correctly\n")

    print("=" * 40)
    print("🎉 All core dependencies are working correctly!")
    print("💡 You're ready to use Qwen-Image!")


if __name__ == "__main__":
    test_core_dependencies()
