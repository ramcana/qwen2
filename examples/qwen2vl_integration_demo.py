#!/usr/bin/env python3
"""
Qwen2-VL Integration Demo
Demonstrates the multimodal capabilities without requiring model downloads
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from qwen2vl_integration import create_qwen2vl_integration
from PIL import Image, ImageDraw
import json

def create_demo_image():
    """Create a demo image for testing"""
    img = Image.new('RGB', (400, 300), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    draw.rectangle([50, 50, 150, 150], fill='red', outline='black')
    draw.ellipse([200, 100, 350, 200], fill='yellow', outline='black')
    draw.text((160, 250), "Demo Image", fill='black')
    
    return img

def demo_prompt_enhancement():
    """Demonstrate prompt enhancement capabilities"""
    print("🎨 Prompt Enhancement Demo")
    print("-" * 30)
    
    integration = create_qwen2vl_integration(fallback_enabled=True)
    
    test_prompts = [
        ("Basic prompt", "a cat", "general"),
        ("Artistic prompt", "a landscape painting", "artistic"),
        ("Technical prompt", "a product photo", "technical"),
        ("Creative prompt", "a fantasy scene", "creative")
    ]
    
    for description, prompt, enhancement_type in test_prompts:
        print(f"\n{description}:")
        print(f"  Original: {prompt}")
        
        result = integration.enhance_prompt(prompt, enhancement_type)
        print(f"  Enhanced: {result.enhanced_prompt}")
        print(f"  Type: {result.enhancement_type}")
        print(f"  Confidence: {result.confidence:.2f}")

def demo_image_analysis():
    """Demonstrate image analysis capabilities"""
    print("\n\n🔍 Image Analysis Demo")
    print("-" * 25)
    
    integration = create_qwen2vl_integration(fallback_enabled=True)
    demo_image = create_demo_image()
    
    analysis_types = ["comprehensive", "style", "composition", "elements"]
    
    for analysis_type in analysis_types:
        print(f"\n{analysis_type.title()} Analysis:")
        result = integration.analyze_image(demo_image, analysis_type)
        
        print(f"  Description: {result.description}")
        print(f"  Key Elements: {', '.join(result.key_elements[:3])}")
        print(f"  Style: {result.style_analysis.get('style', 'unknown')}")
        print(f"  Confidence: {result.confidence:.2f}")

def demo_context_aware_prompts():
    """Demonstrate context-aware prompt creation"""
    print("\n\n🔮 Context-Aware Prompts Demo")
    print("-" * 35)
    
    integration = create_qwen2vl_integration(fallback_enabled=True)
    demo_image = create_demo_image()
    
    base_prompts = [
        "a modern artwork",
        "a colorful composition",
        "an abstract design"
    ]
    
    for base_prompt in base_prompts:
        print(f"\nBase prompt: {base_prompt}")
        
        # Without image context
        enhanced = integration.enhance_prompt(base_prompt, "creative")
        print(f"  Enhanced: {enhanced.enhanced_prompt}")
        
        # With image context
        context_aware = integration.create_context_aware_prompt(base_prompt, demo_image)
        print(f"  Context-aware: {context_aware}")

def demo_integration_status():
    """Show integration status and capabilities"""
    print("\n\n📊 Integration Status")
    print("-" * 20)
    
    integration = create_qwen2vl_integration(fallback_enabled=True)
    status = integration.get_integration_status()
    
    print(f"Available: {status['available']}")
    print(f"Loaded: {status['loaded']}")
    print(f"Fallback enabled: {status['fallback_enabled']}")
    print(f"Cache enabled: {status['cache_enabled']}")
    
    print("\nCapabilities:")
    for capability, enabled in status['capabilities'].items():
        print(f"  {capability}: {'✅' if enabled else '❌'}")

def demo_generator_integration():
    """Demonstrate integration with QwenImageGenerator"""
    print("\n\n🤖 Generator Integration Demo")
    print("-" * 30)
    
    try:
        from qwen_generator import QwenImageGenerator
        
        # Create generator (this will initialize Qwen2-VL integration)
        print("Initializing generator with Qwen2-VL integration...")
        generator = QwenImageGenerator()
        
        # Show Qwen2-VL status
        status = generator.get_qwen2vl_status()
        print(f"\nQwen2-VL Status:")
        print(f"  Available: {status['available']}")
        print(f"  Multimodal enabled: {status.get('multimodal_enabled', False)}")
        
        # Test enhanced prompt generation
        test_prompt = "a futuristic cityscape"
        result = generator.enhance_prompt_with_qwen2vl(test_prompt, "creative")
        
        print(f"\nPrompt Enhancement:")
        print(f"  Original: {result['original_prompt']}")
        print(f"  Enhanced: {result['enhanced_prompt']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        
        # Test image analysis
        demo_image = create_demo_image()
        analysis = generator.analyze_image_with_qwen2vl(demo_image, "style")
        
        print(f"\nImage Analysis:")
        print(f"  Available: {analysis['available']}")
        if 'description' in analysis:
            print(f"  Description: {analysis['description']}")
        
        print("✅ Generator integration working correctly!")
        
    except Exception as e:
        print(f"⚠️ Generator integration test failed: {e}")

def main():
    """Run all demos"""
    print("🚀 Qwen2-VL Integration Demo")
    print("=" * 50)
    print("This demo shows Qwen2-VL integration capabilities")
    print("using fallback mode (no model download required)")
    print("=" * 50)
    
    try:
        demo_prompt_enhancement()
        demo_image_analysis()
        demo_context_aware_prompts()
        demo_integration_status()
        demo_generator_integration()
        
        print("\n" + "=" * 50)
        print("🎉 Demo completed successfully!")
        print("Qwen2-VL integration is ready for use.")
        print("\nTo enable full capabilities:")
        print("1. Ensure Qwen2-VL model is downloaded")
        print("2. Call generator.load_qwen2vl_model()")
        print("3. Enjoy enhanced multimodal features!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)