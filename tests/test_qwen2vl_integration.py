"""
Unit tests for Qwen2-VL Integration
Tests multimodal integration and prompt enhancement functionality
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from PIL import Image
import torch

# Import the module under test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from qwen2vl_integration import (
    Qwen2VLIntegration,
    Qwen2VLConfig,
    PromptEnhancementResult,
    ImageAnalysisResult,
    create_qwen2vl_integration,
    test_qwen2vl_integration
)


class TestQwen2VLConfig(unittest.TestCase):
    """Test Qwen2VL configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = Qwen2VLConfig()
        
        self.assertEqual(config.model_name, "Qwen/Qwen2-VL-7B-Instruct")
        self.assertEqual(config.torch_dtype, torch.float16)
        self.assertEqual(config.device_map, "auto")
        self.assertTrue(config.trust_remote_code)
        self.assertTrue(config.enable_prompt_enhancement)
        self.assertTrue(config.enable_image_analysis)
        self.assertTrue(config.fallback_enabled)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = Qwen2VLConfig(
            model_name="Qwen/Qwen2-VL-2B-Instruct",
            enable_prompt_enhancement=False,
            max_new_tokens=256
        )
        
        self.assertEqual(config.model_name, "Qwen/Qwen2-VL-2B-Instruct")
        self.assertFalse(config.enable_prompt_enhancement)
        self.assertEqual(config.max_new_tokens, 256)


class TestQwen2VLIntegration(unittest.TestCase):
    """Test Qwen2VL integration functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Qwen2VLConfig(fallback_enabled=True)
        self.integration = Qwen2VLIntegration(self.config)
    
    def test_initialization(self):
        """Test integration initialization"""
        self.assertIsNotNone(self.integration.config)
        self.assertFalse(self.integration.is_loaded)
        self.assertEqual(self.integration.config.model_name, "Qwen/Qwen2-VL-7B-Instruct")
    
    def test_initialization_with_custom_config(self):
        """Test initialization with custom config"""
        custom_config = Qwen2VLConfig(
            model_name="Qwen/Qwen2-VL-2B-Instruct",
            enable_prompt_enhancement=False
        )
        integration = Qwen2VLIntegration(custom_config)
        
        self.assertEqual(integration.config.model_name, "Qwen/Qwen2-VL-2B-Instruct")
        self.assertFalse(integration.config.enable_prompt_enhancement)
    
    @patch('qwen2vl_integration.QWEN2VL_AVAILABLE', False)
    def test_unavailable_dependencies(self):
        """Test behavior when dependencies are unavailable"""
        integration = Qwen2VLIntegration()
        self.assertFalse(integration.is_available)
        self.assertFalse(integration.load_model())
    
    def test_fallback_prompt_enhancement(self):
        """Test fallback prompt enhancement"""
        original_prompt = "a cat"
        result = self.integration._fallback_prompt_enhancement(original_prompt, "general")
        
        self.assertIsInstance(result, PromptEnhancementResult)
        self.assertEqual(result.original_prompt, original_prompt)
        self.assertGreater(len(result.enhanced_prompt), len(original_prompt))
        self.assertTrue(result.metadata["fallback_used"])
        self.assertEqual(result.confidence, 0.3)
    
    def test_fallback_prompt_enhancement_types(self):
        """Test different types of fallback prompt enhancement"""
        original_prompt = "a landscape"
        
        # Test artistic enhancement
        result = self.integration._fallback_prompt_enhancement(original_prompt, "artistic")
        self.assertIn("artistic", result.enhanced_prompt.lower())
        
        # Test technical enhancement
        result = self.integration._fallback_prompt_enhancement(original_prompt, "technical")
        self.assertIn("professional", result.enhanced_prompt.lower())
        
        # Test creative enhancement
        result = self.integration._fallback_prompt_enhancement(original_prompt, "creative")
        self.assertIn("creative", result.enhanced_prompt.lower())
    
    def test_fallback_image_analysis(self):
        """Test fallback image analysis"""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        result = self.integration._fallback_image_analysis(test_image, "comprehensive")
        
        self.assertIsInstance(result, ImageAnalysisResult)
        self.assertIn("not available", result.description)
        self.assertEqual(result.confidence, 0.1)
        self.assertIn("Enable Qwen2-VL", result.suggested_improvements[0])
    
    def test_enhance_prompt_disabled(self):
        """Test prompt enhancement when disabled"""
        self.integration.config.enable_prompt_enhancement = False
        
        original_prompt = "a dog"
        result = self.integration.enhance_prompt(original_prompt)
        
        self.assertEqual(result.original_prompt, original_prompt)
        self.assertEqual(result.enhanced_prompt, original_prompt)
        self.assertEqual(result.enhancement_type, "disabled")
        self.assertEqual(result.confidence, 0.0)
    
    def test_analyze_image_disabled(self):
        """Test image analysis when disabled"""
        self.integration.config.enable_image_analysis = False
        
        test_image = Image.new('RGB', (100, 100), color='blue')
        result = self.integration.analyze_image(test_image)
        
        self.assertEqual(result.description, "Image analysis disabled")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(len(result.key_elements), 0)
    
    def test_extract_enhanced_prompt(self):
        """Test enhanced prompt extraction"""
        # Test with prefix removal
        generated_text = "Enhanced prompt: a beautiful sunset over mountains"
        original_prompt = "sunset"
        
        result = self.integration._extract_enhanced_prompt(generated_text, original_prompt)
        self.assertEqual(result, "a beautiful sunset over mountains")
        
        # Test with short result (should fallback)
        generated_text = "sun"
        result = self.integration._extract_enhanced_prompt(generated_text, original_prompt)
        self.assertIn("high quality", result)
    
    def test_calculate_enhancement_confidence(self):
        """Test enhancement confidence calculation"""
        original = "cat"
        
        # Test good enhancement
        enhanced = "a beautiful cat, high quality, detailed, masterpiece"
        confidence = self.integration._calculate_enhancement_confidence(original, enhanced)
        self.assertGreater(confidence, 0.5)
        
        # Test poor enhancement (shorter)
        enhanced = "ca"
        confidence = self.integration._calculate_enhancement_confidence(original, enhanced)
        self.assertEqual(confidence, 0.2)
    
    def test_parse_image_analysis(self):
        """Test image analysis parsing"""
        analysis_text = "This is a realistic photograph of a person standing in a landscape with trees and mountains. The composition is well-balanced with bright lighting."
        
        result = self.integration._parse_image_analysis(analysis_text, "comprehensive")
        
        self.assertIsInstance(result, ImageAnalysisResult)
        self.assertIn("person", result.key_elements)
        self.assertIn("landscape", result.key_elements)
        self.assertEqual(result.style_analysis["style"], "realistic")
        self.assertEqual(result.style_analysis["technique"], "photography")
        self.assertGreater(result.confidence, 0.3)
    
    def test_get_integration_status(self):
        """Test integration status reporting"""
        status = self.integration.get_integration_status()
        
        self.assertIn("available", status)
        self.assertIn("loaded", status)
        self.assertIn("capabilities", status)
        self.assertIn("fallback_enabled", status)
        
        self.assertEqual(status["loaded"], False)
        self.assertEqual(status["fallback_enabled"], True)
    
    def test_cache_functionality(self):
        """Test response caching"""
        # Enable caching
        self.integration.config.cache_responses = True
        self.integration.response_cache = {}
        
        # Test prompt enhancement caching
        original_prompt = "test prompt"
        result1 = self.integration.enhance_prompt(original_prompt)
        result2 = self.integration.enhance_prompt(original_prompt)  # Should use cache
        
        self.assertEqual(result1.enhanced_prompt, result2.enhanced_prompt)
        self.assertGreater(len(self.integration.response_cache), 0)
        
        # Test cache clearing
        self.integration.clear_cache()
        self.assertEqual(len(self.integration.response_cache), 0)
    
    def test_create_context_aware_prompt(self):
        """Test context-aware prompt creation"""
        base_prompt = "a flower"
        
        # Test without reference image
        result = self.integration.create_context_aware_prompt(base_prompt)
        self.assertGreater(len(result), len(base_prompt))
        
        # Test with reference image
        test_image = Image.new('RGB', (100, 100), color='green')
        result_with_image = self.integration.create_context_aware_prompt(base_prompt, test_image)
        self.assertGreater(len(result_with_image), len(result))
    
    @patch('qwen2vl_integration.ModelDetectionService')
    def test_model_availability_check(self, mock_detection_service):
        """Test model availability checking"""
        # Mock detection service
        mock_service = Mock()
        mock_service.detect_qwen2_vl_capabilities.return_value = {
            "integration_possible": True
        }
        mock_detection_service.return_value = mock_service
        
        integration = Qwen2VLIntegration()
        result = integration._check_model_availability()
        
        self.assertTrue(result)
        mock_service.detect_qwen2_vl_capabilities.assert_called_once()
    
    def test_unload_model(self):
        """Test model unloading"""
        # Simulate loaded model
        self.integration.is_loaded = True
        self.integration.model = Mock()
        self.integration.tokenizer = Mock()
        self.integration.processor = Mock()
        
        self.integration.unload_model()
        
        self.assertFalse(self.integration.is_loaded)
        self.assertIsNone(self.integration.model)
        self.assertIsNone(self.integration.tokenizer)
        self.assertIsNone(self.integration.processor)


class TestFactoryFunction(unittest.TestCase):
    """Test factory function for creating integrations"""
    
    def test_create_qwen2vl_integration_defaults(self):
        """Test factory function with defaults"""
        integration = create_qwen2vl_integration()
        
        self.assertEqual(integration.config.model_name, "Qwen/Qwen2-VL-7B-Instruct")
        self.assertTrue(integration.config.enable_prompt_enhancement)
        self.assertTrue(integration.config.enable_image_analysis)
        self.assertTrue(integration.config.fallback_enabled)
    
    def test_create_qwen2vl_integration_custom(self):
        """Test factory function with custom parameters"""
        integration = create_qwen2vl_integration(
            model_name="Qwen/Qwen2-VL-2B-Instruct",
            enable_prompt_enhancement=False,
            temperature=0.5
        )
        
        self.assertEqual(integration.config.model_name, "Qwen/Qwen2-VL-2B-Instruct")
        self.assertFalse(integration.config.enable_prompt_enhancement)
        self.assertEqual(integration.config.temperature, 0.5)


class TestIntegrationTesting(unittest.TestCase):
    """Test integration testing utilities"""
    
    @patch('qwen2vl_integration.QWEN2VL_AVAILABLE', True)
    @patch('qwen2vl_integration.create_qwen2vl_integration')
    def test_integration_test_function(self, mock_create_integration):
        """Test the integration test function"""
        # Mock integration
        mock_integration = Mock()
        mock_integration.is_available = True
        mock_integration.load_model.return_value = True
        mock_integration.config.enable_image_analysis = True
        
        # Mock enhancement result
        mock_enhancement = Mock()
        mock_enhancement.enhanced_prompt = "enhanced test prompt"
        mock_integration.enhance_prompt.return_value = mock_enhancement
        
        # Mock analysis result
        mock_analysis = Mock()
        mock_analysis.description = "test analysis"
        mock_integration.analyze_image.return_value = mock_analysis
        
        mock_create_integration.return_value = mock_integration
        
        results = test_qwen2vl_integration()
        
        self.assertTrue(results["dependencies_available"])
        self.assertTrue(results["model_loading"])
        self.assertTrue(results["prompt_enhancement"])
        self.assertTrue(results["image_analysis"])
        self.assertEqual(len(results["errors"]), 0)


class TestPromptEnhancementResult(unittest.TestCase):
    """Test PromptEnhancementResult dataclass"""
    
    def test_prompt_enhancement_result_creation(self):
        """Test creating PromptEnhancementResult"""
        result = PromptEnhancementResult(
            original_prompt="test",
            enhanced_prompt="enhanced test",
            enhancement_type="general",
            confidence=0.8,
            metadata={"test": True}
        )
        
        self.assertEqual(result.original_prompt, "test")
        self.assertEqual(result.enhanced_prompt, "enhanced test")
        self.assertEqual(result.enhancement_type, "general")
        self.assertEqual(result.confidence, 0.8)
        self.assertTrue(result.metadata["test"])


class TestImageAnalysisResult(unittest.TestCase):
    """Test ImageAnalysisResult dataclass"""
    
    def test_image_analysis_result_creation(self):
        """Test creating ImageAnalysisResult"""
        result = ImageAnalysisResult(
            description="test image",
            key_elements=["element1", "element2"],
            style_analysis={"style": "realistic"},
            composition_notes="well composed",
            suggested_improvements=["improve lighting"],
            confidence=0.7
        )
        
        self.assertEqual(result.description, "test image")
        self.assertEqual(len(result.key_elements), 2)
        self.assertEqual(result.style_analysis["style"], "realistic")
        self.assertEqual(result.composition_notes, "well composed")
        self.assertEqual(len(result.suggested_improvements), 1)
        self.assertEqual(result.confidence, 0.7)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.integration = Qwen2VLIntegration()
    
    def test_analyze_image_with_invalid_path(self):
        """Test image analysis with invalid file path"""
        result = self.integration.analyze_image("/nonexistent/path.jpg")
        
        # Should return fallback result
        self.assertIn("not available", result.description)
        self.assertEqual(result.confidence, 0.1)
    
    def test_analyze_image_with_invalid_type(self):
        """Test image analysis with invalid input type"""
        result = self.integration.analyze_image(123)  # Invalid type
        
        # Should return fallback result
        self.assertIn("not available", result.description)
        self.assertEqual(result.confidence, 0.1)
    
    def test_enhance_prompt_with_empty_string(self):
        """Test prompt enhancement with empty string"""
        result = self.integration.enhance_prompt("")
        
        # Should handle gracefully
        self.assertIsInstance(result, PromptEnhancementResult)
        self.assertGreater(len(result.enhanced_prompt), 0)
    
    def test_create_context_aware_prompt_with_invalid_image(self):
        """Test context-aware prompt with invalid image"""
        base_prompt = "test prompt"
        result = self.integration.create_context_aware_prompt(base_prompt, "/invalid/path.jpg")
        
        # Should fallback to enhanced prompt without image context
        self.assertGreater(len(result), len(base_prompt))


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)