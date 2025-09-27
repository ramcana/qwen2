#!/usr/bin/env python3
"""
Quick test to verify the attention processor fix
"""

import os
import sys
import torch
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from qwen_generator import QwenImageGenerator
    print("✅ QwenImageGenerator imported successfully")
except ImportError as e:
    print(f"❌ Failed to import QwenImageGenerator: {e}")
    sys.exit(1)

def test_attention_fix():
    """Test that the attention processor fix works"""
    print("🧪 Testing Attention Processor Fix")
    print("=" * 50)
    
    # Initialize generator
    print("🔧 Initializing QwenImageGenerator...")
    generator = QwenImageGenerator()
    
    # Load model
    print("📥 Loading model...")
    if not generator.load_model():
        print("❌ Failed to load model")
        return False
    
    print("✅ Model loaded successfully")
    
    # Test simple generation
    print("🎨 Testing simple image generation...")
    try:
        result = generator.generate_image(
            prompt="A simple test image of a red apple",
            width=512,
            height=512,
            num_inference_steps=10,  # Fast test
            seed=42
        )
        
        if result and result[0]:  # Check if image was generated
            print("✅ Image generation successful!")
            print(f"📁 Image saved to: {result[1] if len(result) > 1 else 'generated_images/'}")
            return True
        else:
            print("❌ Image generation failed - no image returned")
            return False
            
    except Exception as e:
        print(f"❌ Image generation failed with error: {e}")
        return False

def main():
    """Main test function"""
    print("🔍 Attention Processor Fix Test")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("⚠️ No GPU available, using CPU")
    
    # Run test
    success = test_attention_fix()
    
    if success:
        print("\n🎉 Attention processor fix test PASSED!")
        print("✅ The 'not enough values to unpack' error has been resolved")
    else:
        print("\n❌ Attention processor fix test FAILED!")
        print("⚠️ The issue may still exist")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)