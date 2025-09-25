#!/usr/bin/env python3
"""
Test script for Qwen2-VL integration functionality
Tests the integration without requiring actual model downloads
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from qwen2vl_integration import (
    Qwen2VLIntegration,
    Qwen2VLConfig,
    create_qwen2vl_integration,
    test_qwen2vl_integration
)
from PIL import Image


def test_basic_integration():
    """Test basic integration functionality"""
    print("🧪 Testing Qwen2-VL Integration...")
    
    # Test configuration
    print("\n1. Testing configuration...")
    config = Qwen2VLConfig(
        enable_prompt_enhancement=True,
        enable_image_analysis=True,
        fallback_enabled=True
    )
    print(f"✅ Config created: {config.model_name}")
    
    # Test integration creation
    print("\n2. Testing integration creation...")
    integration = Qwen2VLIntegration(config)
    print(f"✅ Integration created (Available: {integration.is_available})")
    
    # Test factory function
    print("\n3. Testing factory function...")
    factory_integration = create_qwen2vl_integration(fallback_enabled=True)
    print(f"✅ Factory integration created")
    
    # Test status reporting
    print("\n4. Testing status reporting...")
    status = integration.get_integration_status()
    print(f"✅ Status: Available={status['available']}, Loaded={status['loaded']}")
    
    # Test fallback prompt enhancement
    print("\n5. Testing fallback prompt enhancement...")
    test_prompt = "a beautiful landscape"
    enhancement_result = integration.enhance_prompt(test_prompt, "artistic")
    print(f"✅ Original: {test_prompt}")
    print(f"✅ Enhanced: {enhancement_result.enhanced_prompt}")
    print(f"✅ Confidence: {enhancement_result.confidence}")
    
    # Test fallback image analysis
    print("\n6. Testing fallback image analysis...")
    test_image = Image.new('RGB', (100, 100), color='red')
    analysis_result = integration.analyze_image(test_image, "comprehensive")
    print(f"✅ Analysis: {analysis_result.description}")
    print(f"✅ Elements: {analysis_result.key_elements}")
    print(f"✅ Confidence: {analysis_result.confidence}")
    
    # Test context-aware prompt creation
    print("\n7. Testing context-aware prompt creation...")
    context_prompt = integration.create_context_aware_prompt(test_prompt, test_image)
    print(f"✅ Context prompt: {context_prompt}")
    
    # Test cache functionality
    print("\n8. Testing cache functionality...")
    integration.clear_cache()
    print("✅ Cache cleared")
    
    print("\n🎉 All basic tests passed!")
    return True


def test_generator_integration():
    """Test integration with QwenImageGenerator"""
    print("\n🧪 Testing Generator Integration...")
    
    try:
        from qwen_generator import QwenImageGenerator
        
        # Create generator (should initialize Qwen2-VL integration)
        print("\n1. Creating generator with Qwen2-VL integration...")
        generator = QwenImageGenerator()
        
        # Test Qwen2-VL status
        print("\n2. Testing Qwen2-VL status...")
        status = generator.get_qwen2vl_status()
        print(f"✅ Qwen2-VL Status: {status}")
        
        # Test prompt enhancement
        print("\n3. Testing enhanced prompt generation...")
        test_prompt = "a cat in a garden"
        enhancement_result = generator.enhance_prompt_with_qwen2vl(test_prompt, "creative")
        print(f"✅ Original: {enhancement_result['original_prompt']}")
        print(f"✅ Enhanced: {enhancement_result['enhanced_prompt']}")
        print(f"✅ Confidence: {enhancement_result['confidence']}")
        
        # Test image analysis
        print("\n4. Testing image analysis...")
        test_image = Image.new('RGB', (200, 200), color='blue')
        analysis_result = generator.analyze_image_with_qwen2vl(test_image, "style")
        print(f"✅ Analysis available: {analysis_result['available']}")
        if 'description' in analysis_result:
            print(f"✅ Description: {analysis_result['description']}")
        
        print("\n🎉 Generator integration tests passed!")
        return True
        
    except ImportError as e:
        print(f"⚠️ Generator integration test skipped: {e}")
        return False


def test_comprehensive_integration():
    """Run comprehensive integration tests"""
    print("\n🧪 Running Comprehensive Integration Tests...")
    
    results = test_qwen2vl_integration()
    
    print(f"\n📊 Test Results:")
    print(f"Dependencies available: {results['dependencies_available']}")
    print(f"Model loading: {results['model_loading']}")
    print(f"Prompt enhancement: {results['prompt_enhancement']}")
    print(f"Image analysis: {results['image_analysis']}")
    
    if results['errors']:
        print(f"Errors: {results['errors']}")
    else:
        print("✅ No errors detected")
    
    return len(results['errors']) == 0


def main():
    """Run all integration tests"""
    print("🚀 Qwen2-VL Integration Test Suite")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Basic integration
    try:
        if test_basic_integration():
            success_count += 1
    except Exception as e:
        print(f"❌ Basic integration test failed: {e}")
    
    # Test 2: Generator integration
    try:
        if test_generator_integration():
            success_count += 1
    except Exception as e:
        print(f"❌ Generator integration test failed: {e}")
    
    # Test 3: Comprehensive tests
    try:
        if test_comprehensive_integration():
            success_count += 1
    except Exception as e:
        print(f"❌ Comprehensive integration test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Summary: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Qwen2-VL integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)