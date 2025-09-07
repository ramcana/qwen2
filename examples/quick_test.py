#!/usr/bin/env python3
"""
Quick Test Script for Qwen-Image Generator
Run this to verify your setup without launching the full UI
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.qwen_generator import QwenImageGenerator

def main():
    print("🧪 Testing Qwen-Image Generator Setup...")
    print("=" * 50)
    
    # Initialize generator
    generator = QwenImageGenerator()
    
    # Load model
    print("Loading model...")
    if not generator.load_model():
        print("❌ Failed to load model. Check your setup.")
        return False
    
    # Test generation
    test_prompt = "A beautiful coffee shop with a neon sign reading 'AI Café', modern interior, warm lighting"
    
    print(f"\n📝 Generating test image...")
    print(f"Prompt: {test_prompt}")
    
    try:
        image, message = generator.generate_image(
            prompt=test_prompt,
            width=1024,  # Smaller size for quick test
            height=1024,
            num_inference_steps=20,  # Fast generation
            seed=42  # Reproducible result
        )
        
        if image:
            print(f"\n✅ Success! {message}")
            print("\n🎯 Test completed successfully!")
            print("Your Qwen-Image setup is working properly.")
            return True
        else:
            print(f"\n❌ Generation failed: {message}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)