#!/usr/bin/env python3
"""
Official Qwen-Image Example Script
Based on the official Hugging Face documentation
Demonstrates advanced text rendering capabilities
"""

import os
import sys
from datetime import datetime

import torch
from diffusers import DiffusionPipeline

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def official_qwen_example():
    """Run the official Qwen-Image example from Hugging Face docs"""

    print("🎨 Official Qwen-Image Example")
    print("=" * 50)
    print("Based on: https://huggingface.co/Qwen/Qwen-Image")
    print()

    model_name = "Qwen/Qwen-Image"

    # Load the pipeline (matching official docs)
    print("📦 Loading Qwen-Image pipeline...")
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
        print("✅ Using CUDA with bfloat16")
    else:
        torch_dtype = torch.float32
        device = "cpu"
        print("⚠️ Using CPU with float32")

    try:
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
        pipe = pipe.to(device)
        print(f"✅ Pipeline loaded successfully on {device}")
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        return

    # Official positive magic strings
    positive_magic = {
        "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
        "zh": ", 超清，4K，电影级构图.",  # for chinese prompt
    }

    # Official example prompt (complex text rendering)
    prompt = """A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition"""

    negative_prompt = " "  # Empty string as per official docs

    # Official aspect ratios
    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }

    # Use 16:9 ratio as in official example
    width, height = aspect_ratios["16:9"]

    print("🎯 Generating image with official example:")
    print(f"   Resolution: {width}x{height}")
    print(f"   Prompt: {prompt[:80]}...")
    print(f"   Using positive magic: {positive_magic['en']}")
    print()

    try:
        # Generate with official parameters
        print("🎨 Starting generation...")
        image = pipe(
            prompt=prompt + positive_magic["en"],
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=torch.Generator(device=device).manual_seed(42),
        ).images[0]

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"official_qwen_example_{timestamp}.png"
        filepath = os.path.join("generated_images", filename)

        # Ensure directory exists
        os.makedirs("generated_images", exist_ok=True)

        image.save(filepath)

        print("✅ Image generated successfully!")
        print(f"📁 Saved as: {filepath}")
        print()
        print("🎯 This example demonstrates:")
        print("   • Complex text rendering (English + Chinese)")
        print("   • Mathematical symbols and equations")
        print("   • Emoji support")
        print("   • Mixed typography in realistic scenes")
        print("   • Official 'positive magic' enhancement")

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return

    # Demonstrate other capabilities
    print()
    print("🌟 Additional Qwen-Image Capabilities:")
    print("   • Text rendering in 20+ languages")
    print("   • Multiple artistic styles (photorealistic, anime, impressionist)")
    print("   • Image editing and manipulation")
    print("   • Object detection and segmentation")
    print("   • Novel view synthesis")
    print("   • Super-resolution")
    print()
    print("📖 For more examples, check the official documentation:")
    print("   https://huggingface.co/Qwen/Qwen-Image")


def quick_text_test():
    """Quick test with simpler text rendering"""

    print("🚀 Quick Text Rendering Test")
    print("=" * 50)

    try:
        from src.qwen_generator import QwenImageGenerator

        generator = QwenImageGenerator()
        if not generator.load_model():
            print("❌ Failed to load model")
            return

        # Test prompt with both English and Chinese
        test_prompt = 'A modern café with a sign reading "AI Coffee Shop 人工智能咖啡店" and a menu board showing "Latte $4 拿铁咖啡"'

        print(f"🎯 Testing: {test_prompt}")

        image, message = generator.generate_image(
            prompt=test_prompt,
            width=1664,
            height=928,
            num_inference_steps=30,  # Faster for testing
            cfg_scale=4.0,
            seed=42,
            language="en",
        )

        if image:
            print("✅ Quick test successful!")
            print(message)
        else:
            print(f"❌ Quick test failed: {message}")

    except Exception as e:
        print(f"❌ Quick test error: {e}")


if __name__ == "__main__":
    print("🎨 Qwen-Image Official Examples")
    print("=" * 60)
    print()

    choice = input(
        "Choose example:\n1. Official Hugging Face example (complex text)\n2. Quick text test with existing generator\n\nEnter choice (1-2): "
    ).strip()

    if choice == "1":
        official_qwen_example()
    elif choice == "2":
        quick_text_test()
    else:
        print("Running both examples...")
        print()
        official_qwen_example()
        print("\n" + "=" * 60 + "\n")
        quick_text_test()
