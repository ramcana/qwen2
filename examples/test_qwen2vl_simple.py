#!/usr/bin/env python3
"""
Simple test for Qwen2-VL integration without model downloads
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all imports work correctly"""
    print("🧪 Testing imports...")
    
    try:
        from qwen2vl_integration import (
            Qwen2VLIntegration,
            Qwen2VLConfig,
            PromptEnhancementResult,
            ImageAnalysisResult,
            create_qwen2vl_integration
        )
        print("✅ Qwen2-VL integration imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_fallback_functionality():
    """Test fallback functionality without model loading"""
    print("\n🧪 Testing fallback functionality...")
    
    try:
        from qwen2vl_integration import create_qwen2vl_integration
        from PIL import Image
        
        # Create integration with fallback enabled
        integration = create_qwen2vl_integration(fallback_enabled=True)
        
        # Test prompt enhancement (should work with fallback)
        test_prompt = "a sunset over mountains"
        result = integration.enhance_prompt(test_prompt, "artistic")
        
        print(f"✅ Original prompt: {test_prompt}")
        print(f"✅ Enhanced prompt: {result.enhanced_prompt}")
        print(f"✅ Enhancement type: {result.enhancement_type}")
        print(f"✅ Confidence: {result.confidence}")
        
        # Test image analysis (should work with fallback)
        test_image = Image.new('RGB', (100, 100), color='green')
        analysis = integration.analyze_image(test_image)
        
        print(f"✅ Image analysis: {analysis.description}")
        print(f"✅ Confidence: {analysis.confidence}")
        
        # Test status
        status = integration.get_integration_status()
        print(f"✅ Integration available: {status['available']}")
        print(f"✅ Fallback enabled: {status['fallback_enabled']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False

def test_generator_integration():
    """Test integration with generator (without model loading)"""
    print("\n🧪 Testing generator integration...")
    
    try:
        from qwen_generator import QwenImageGenerator
        
        # Create generator
        generator = QwenImageGenerator()
        
        # Test Qwen2-VL status
        status = generator.get_qwen2vl_status()
        print(f"✅ Qwen2-VL available: {status.get('available', False)}")
        print(f"✅ Integration available: {status.get('integration_available', False)}")
        
        # Test prompt enhancement (should use fallback)
        test_prompt = "a robot in space"
        result = generator.enhance_prompt_with_qwen2vl(test_prompt, "technical")
        print(f"✅ Enhanced prompt: {result['enhanced_prompt']}")
        print(f"✅ Enhancement type: {result['enhancement_type']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generator integration test failed: {e}")
        return False

def main():
    """Run simple tests"""
    print("🚀 Qwen2-VL Simple Integration Test")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Fallback Functionality", test_fallback_functionality),
        ("Generator Integration", test_generator_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Qwen2-VL integration is working.")
        return True
    else:
        print("⚠️ Some tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)