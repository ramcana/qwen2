"""
Unit tests for DiffSynth Service Foundation
Tests service initialization, configuration, and basic functionality
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import time
from PIL import Image
import torch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from diffsynth_service import (
    DiffSynthService, DiffSynthConfig, DiffSynthServiceStatus,
    ResourceUsage, create_diffsynth_service
)
from diffsynth_models import InpaintRequest, OutpaintRequest, EditOperation
from resource_manager import ResourceManager, ServiceType, ResourcePriority


class TestDiffSynthConfig(unittest.TestCase):
    """Test DiffSynth configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = DiffSynthConfig()
        
        self.assertEqual(config.model_name, "Qwen/Qwen-Image-Edit")
        self.assertEqual(config.torch_dtype, torch.bfloat16)
        self.assertTrue(config.enable_vram_management)
        self.assertTrue(config.enable_cpu_offload)
        self.assertEqual(config.default_num_inference_steps, 20)
        self.assertEqual(config.default_guidance_scale, 7.5)
        self.assertEqual(config.max_memory_usage_gb, 4.0)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = DiffSynthConfig(
            model_name="custom/model",
            torch_dtype=torch.float16,
            device="cpu",
            enable_vram_management=False,
            default_num_inference_steps=30
        )
        
        self.assertEqual(config.model_name, "custom/model")
        self.assertEqual(config.torch_dtype, torch.float16)
        self.assertEqual(config.device, "cpu")
        self.assertFalse(config.enable_vram_management)
        self.assertEqual(config.default_num_inference_steps, 30)


class TestDiffSynthService(unittest.TestCase):
    """Test DiffSynth service functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = DiffSynthConfig(device="cpu")  # Use CPU for testing
        self.service = DiffSynthService(self.config)
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self.service, 'shutdown'):
            self.service.shutdown()
    
    def test_service_initialization(self):
        """Test service initialization"""
        self.assertEqual(self.service.status, DiffSynthServiceStatus.NOT_INITIALIZED)
        self.assertIsNone(self.service.pipeline)
        self.assertEqual(self.service.operation_count, 0)
        self.assertEqual(self.service.error_count, 0)
    
    def test_memory_optimizations_setup(self):
        """Test memory optimizations setup"""
        # This should not raise any exceptions
        self.service._setup_memory_optimizations()
        
        # Verify the service is still in correct state
        self.assertEqual(self.service.status, DiffSynthServiceStatus.NOT_INITIALIZED)
    
    def test_system_requirements_check(self):
        """Test system requirements checking"""
        result = self.service._check_system_requirements()
        
        # Should pass basic requirements
        self.assertTrue(result)
        
        # Device should be set correctly
        self.assertIn(self.service.config.device, ["cuda", "cpu"])
    
    @patch('diffsynth_service.torch.cuda.is_available')
    def test_system_requirements_no_cuda(self, mock_cuda_available):
        """Test system requirements when CUDA is not available"""
        mock_cuda_available.return_value = False
        
        config = DiffSynthConfig(device="cuda")
        service = DiffSynthService(config)
        
        result = service._check_system_requirements()
        self.assertTrue(result)
        self.assertEqual(service.config.device, "cpu")
    
    @patch('psutil.disk_usage')
    def test_system_requirements_low_disk_space(self, mock_disk_usage):
        """Test system requirements with low disk space"""
        # Mock low disk space
        mock_usage = Mock()
        mock_usage.free = 5 * 1e9  # 5GB
        mock_disk_usage.return_value = mock_usage
        
        result = self.service._check_system_requirements()
        self.assertFalse(result)
    
    def test_prepare_generation_params(self):
        """Test generation parameter preparation"""
        params = self.service._prepare_generation_params(
            prompt="test prompt",
            num_inference_steps=None,
            guidance_scale=None,
            height=None,
            width=None,
            seed=42
        )
        
        self.assertEqual(params["prompt"], "test prompt")
        self.assertEqual(params["num_inference_steps"], self.config.default_num_inference_steps)
        self.assertEqual(params["guidance_scale"], self.config.default_guidance_scale)
        self.assertEqual(params["height"], self.config.default_height)
        self.assertEqual(params["width"], self.config.default_width)
        self.assertEqual(params["seed"], 42)
    
    def test_prepare_generation_params_custom(self):
        """Test generation parameter preparation with custom values"""
        params = self.service._prepare_generation_params(
            prompt="custom prompt",
            num_inference_steps=50,
            guidance_scale=10.0,
            height=1024,
            width=1024,
            seed=123
        )
        
        self.assertEqual(params["prompt"], "custom prompt")
        self.assertEqual(params["num_inference_steps"], 50)
        self.assertEqual(params["guidance_scale"], 10.0)
        self.assertEqual(params["height"], 1024)
        self.assertEqual(params["width"], 1024)
        self.assertEqual(params["seed"], 123)
    
    def test_load_image_pil(self):
        """Test loading PIL Image"""
        # Create a test image
        test_image = Image.new("RGB", (100, 100), color="red")
        
        loaded_image = self.service._load_image(test_image)
        
        self.assertIsNotNone(loaded_image)
        self.assertEqual(loaded_image.mode, "RGB")
        self.assertEqual(loaded_image.size, (100, 100))
    
    def test_load_image_file_path(self):
        """Test loading image from file path"""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            test_image = Image.new("RGB", (50, 50), color="blue")
            test_image.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            loaded_image = self.service._load_image(tmp_path)
            
            self.assertIsNotNone(loaded_image)
            self.assertEqual(loaded_image.mode, "RGB")
            self.assertEqual(loaded_image.size, (50, 50))
        finally:
            os.unlink(tmp_path)
    
    def test_load_image_invalid_path(self):
        """Test loading image with invalid path"""
        loaded_image = self.service._load_image("/nonexistent/path.png")
        self.assertIsNone(loaded_image)
    
    def test_load_image_invalid_type(self):
        """Test loading image with invalid type"""
        loaded_image = self.service._load_image(123)
        self.assertIsNone(loaded_image)
    
    def test_update_resource_usage(self):
        """Test resource usage tracking"""
        initial_time = self.service.resource_usage.last_updated
        
        self.service._update_resource_usage()
        
        # Should update timestamp
        self.assertGreater(self.service.resource_usage.last_updated, initial_time)
        
        # Should have reasonable values
        self.assertGreaterEqual(self.service.resource_usage.cpu_memory_used, 0)
    
    def test_cleanup_resources(self):
        """Test resource cleanup"""
        # Should not raise exceptions
        self.service._cleanup_resources()
    
    def test_get_status(self):
        """Test status reporting"""
        status = self.service.get_status()
        
        self.assertIn("status", status)
        self.assertIn("initialized", status)
        self.assertIn("model_name", status)
        self.assertIn("device", status)
        self.assertIn("operation_count", status)
        self.assertIn("error_count", status)
        self.assertIn("resource_usage", status)
        self.assertIn("config", status)
        
        self.assertEqual(status["status"], DiffSynthServiceStatus.NOT_INITIALIZED.value)
        self.assertFalse(status["initialized"])
        self.assertEqual(status["model_name"], self.config.model_name)
        self.assertEqual(status["device"], self.config.device)
    
    def test_shutdown(self):
        """Test service shutdown"""
        self.service.shutdown()
        
        self.assertEqual(self.service.status, DiffSynthServiceStatus.OFFLINE)
        self.assertIsNone(self.service.pipeline)
    
    @patch('diffsynth_service.logger')
    def test_log_error(self, mock_logger):
        """Test error logging"""
        from error_handler import ErrorInfo, ErrorCategory
        
        error_info = ErrorInfo(
            category=ErrorCategory.PIPELINE,
            severity="HIGH",
            message="Test error",
            details="Test error details",
            suggested_fixes=["Fix 1", "Fix 2", "Fix 3", "Fix 4"]
        )
        
        self.service._log_error(error_info)
        
        # Should log error and details
        mock_logger.error.assert_called()
        mock_logger.info.assert_called()


class TestDiffSynthServiceInitialization(unittest.TestCase):
    """Test DiffSynth service initialization with mocked dependencies"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = DiffSynthConfig(device="cpu")
    
    @patch('diffsynth_service.DiffSynthService._check_system_requirements')
    def test_initialization_system_requirements_fail(self, mock_check_requirements):
        """Test initialization failure due to system requirements"""
        mock_check_requirements.return_value = False
        
        service = DiffSynthService(self.config)
        result = service.initialize()
        
        self.assertFalse(result)
        self.assertEqual(service.status, DiffSynthServiceStatus.ERROR)
    
    @patch('diffsynth_service.DiffSynthService._verify_initialization')
    @patch('diffsynth_service.DiffSynthService._create_optimized_pipeline')
    @patch('diffsynth_service.DiffSynthService._check_system_requirements')
    def test_initialization_pipeline_creation_fail(self, mock_check_requirements, 
                                                  mock_create_pipeline, mock_verify):
        """Test initialization failure due to pipeline creation"""
        mock_check_requirements.return_value = True
        mock_create_pipeline.return_value = None
        
        service = DiffSynthService(self.config)
        result = service.initialize()
        
        self.assertFalse(result)
        self.assertEqual(service.status, DiffSynthServiceStatus.ERROR)
    
    @patch('diffsynth_service.DiffSynthService._verify_initialization')
    @patch('diffsynth_service.DiffSynthService._create_optimized_pipeline')
    @patch('diffsynth_service.DiffSynthService._check_system_requirements')
    def test_initialization_verification_fail(self, mock_check_requirements, 
                                            mock_create_pipeline, mock_verify):
        """Test initialization failure due to verification"""
        mock_check_requirements.return_value = True
        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline
        mock_verify.return_value = False
        
        service = DiffSynthService(self.config)
        result = service.initialize()
        
        self.assertFalse(result)
        self.assertEqual(service.status, DiffSynthServiceStatus.ERROR)
    
    def test_initialization_import_error(self):
        """Test initialization failure due to import error"""
        with patch('builtins.__import__', side_effect=ImportError("DiffSynth not installed")):
            service = DiffSynthService(self.config)
            result = service.initialize()
        
        self.assertFalse(result)
        self.assertEqual(service.status, DiffSynthServiceStatus.ERROR)
    
    def test_verify_initialization_no_pipeline(self):
        """Test verification with no pipeline"""
        service = DiffSynthService(self.config)
        result = service._verify_initialization()
        
        self.assertFalse(result)
    
    def test_verify_initialization_missing_methods(self):
        """Test verification with pipeline missing methods"""
        service = DiffSynthService(self.config)
        service.pipeline = Mock()
        # Remove the __call__ method by setting it to None
        service.pipeline.__call__ = None
        
        # Mock hasattr to return False for __call__
        with patch('builtins.hasattr', side_effect=lambda obj, attr: attr != '__call__'):
            result = service._verify_initialization()
        
        self.assertFalse(result)


class TestDiffSynthServiceImageEditing(unittest.TestCase):
    """Test DiffSynth service image editing functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = DiffSynthConfig(device="cpu")
        self.service = DiffSynthService(self.config)
        
        # Mock pipeline for testing
        self.mock_pipeline = Mock()
        self.mock_pipeline.return_value = Image.new("RGB", (100, 100), color="green")
        self.service.pipeline = self.mock_pipeline
        self.service.status = DiffSynthServiceStatus.READY
    
    def test_edit_image_success(self):
        """Test successful image editing"""
        test_image = Image.new("RGB", (100, 100), color="red")
        
        result_image, message = self.service.edit_image(
            prompt="test prompt",
            image=test_image,
            num_inference_steps=10,
            guidance_scale=5.0
        )
        
        self.assertIsNotNone(result_image)
        self.assertIn("successfully", message)
        self.assertEqual(self.service.operation_count, 1)
        self.assertGreater(self.service.last_operation_time, 0)
        
        # Verify pipeline was called with correct parameters
        self.mock_pipeline.assert_called_once()
        call_kwargs = self.mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs["prompt"], "test prompt")
        self.assertEqual(call_kwargs["num_inference_steps"], 10)
        self.assertEqual(call_kwargs["guidance_scale"], 5.0)
    
    def test_edit_image_service_not_ready(self):
        """Test image editing when service is not ready"""
        self.service.status = DiffSynthServiceStatus.NOT_INITIALIZED
        
        with patch.object(self.service, 'initialize', return_value=False):
            result_image, message = self.service.edit_image(
                prompt="test prompt",
                image=Image.new("RGB", (100, 100))
            )
        
        self.assertIsNone(result_image)
        self.assertIn("not available", message)
    
    def test_edit_image_invalid_image(self):
        """Test image editing with invalid image"""
        result_image, message = self.service.edit_image(
            prompt="test prompt",
            image="/nonexistent/path.png"
        )
        
        self.assertIsNone(result_image)
        self.assertIn("Failed to load", message)
    
    def test_edit_image_pipeline_error(self):
        """Test image editing with pipeline error"""
        self.mock_pipeline.side_effect = Exception("Pipeline error")
        
        test_image = Image.new("RGB", (100, 100), color="red")
        
        result_image, message = self.service.edit_image(
            prompt="test prompt",
            image=test_image
        )
        
        self.assertIsNone(result_image)
        self.assertIn("failed", message)
        self.assertEqual(self.service.error_count, 1)
    
    def test_edit_image_with_fallback(self):
        """Test image editing with fallback enabled"""
        self.service.config.enable_fallback = True
        self.mock_pipeline.side_effect = Exception("Pipeline error")
        
        test_image = Image.new("RGB", (100, 100), color="red")
        
        with patch.object(self.service, '_fallback_edit_image', 
                         return_value=(None, "Fallback message")) as mock_fallback:
            result_image, message = self.service.edit_image(
                prompt="test prompt",
                image=test_image
            )
        
        mock_fallback.assert_called_once()
        self.assertIn("Fallback", message)


class TestDiffSynthServiceInpainting(unittest.TestCase):
    """Test DiffSynth service inpainting functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = DiffSynthConfig(device="cpu")
        self.service = DiffSynthService(self.config)
        
        # Mock pipeline for testing
        self.mock_pipeline = Mock()
        self.mock_pipeline.return_value = Image.new("RGB", (100, 100), color="green")
        self.service.pipeline = self.mock_pipeline
        self.service.status = DiffSynthServiceStatus.READY
        
        # Create test images
        self.test_image = Image.new("RGB", (100, 100), color="red")
        self.test_mask = Image.new("L", (100, 100), color=255)  # White mask
        
        # Create test InpaintRequest
        from diffsynth_models import InpaintRequest
        self.test_request = InpaintRequest(
            prompt="Fill with blue flowers",
            image_base64="test_image_data",
            mask_base64="test_mask_data",
            num_inference_steps=20,
            guidance_scale=7.5,
            strength=0.9
        )
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self.service, 'shutdown'):
            self.service.shutdown()
    
    @patch('diffsynth_service.DiffSynthService._load_image_from_request')
    @patch('diffsynth_service.DiffSynthService._load_mask_from_request')
    @patch('diffsynth_service.DiffSynthService._save_edited_image')
    def test_inpaint_success(self, mock_save, mock_load_mask, mock_load_image):
        """Test successful inpainting operation"""
        # Setup mocks
        mock_load_image.return_value = self.test_image
        mock_load_mask.return_value = self.test_mask
        mock_save.return_value = "output/inpainted.jpg"
        
        # Perform inpainting
        response = self.service.inpaint(self.test_request)
        
        # Verify response
        self.assertTrue(response.success)
        self.assertIn("successfully", response.message)
        self.assertEqual(response.operation, "inpaint")
        self.assertEqual(response.image_path, "output/inpainted.jpg")
        self.assertIsNotNone(response.processing_time)
        self.assertIsNotNone(response.parameters)
        
        # Verify service state
        self.assertEqual(self.service.operation_count, 1)
        self.assertGreater(self.service.last_operation_time, 0)
        
        # Verify pipeline was called
        self.mock_pipeline.assert_called_once()
    
    def test_inpaint_service_not_ready(self):
        """Test inpainting when service is not ready"""
        self.service.status = DiffSynthServiceStatus.NOT_INITIALIZED
        
        with patch.object(self.service, 'initialize', return_value=False):
            response = self.service.inpaint(self.test_request)
        
        self.assertFalse(response.success)
        self.assertIn("not available", response.message)
    
    @patch('diffsynth_service.DiffSynthService._load_image_from_request')
    def test_inpaint_invalid_image(self, mock_load_image):
        """Test inpainting with invalid image"""
        mock_load_image.return_value = None
        
        response = self.service.inpaint(self.test_request)
        
        self.assertFalse(response.success)
        self.assertIn("Failed to load input image", response.message)
    
    @patch('diffsynth_service.DiffSynthService._load_image_from_request')
    @patch('diffsynth_service.DiffSynthService._load_mask_from_request')
    def test_inpaint_invalid_mask(self, mock_load_mask, mock_load_image):
        """Test inpainting with invalid mask"""
        mock_load_image.return_value = self.test_image
        mock_load_mask.return_value = None
        
        response = self.service.inpaint(self.test_request)
        
        self.assertFalse(response.success)
        self.assertIn("Failed to load mask image", response.message)
    
    @patch('diffsynth_service.DiffSynthService._load_image_from_request')
    @patch('diffsynth_service.DiffSynthService._load_mask_from_request')
    @patch('diffsynth_service.DiffSynthService._validate_mask_compatibility')
    def test_inpaint_incompatible_mask(self, mock_validate, mock_load_mask, mock_load_image):
        """Test inpainting with incompatible mask"""
        mock_load_image.return_value = self.test_image
        mock_load_mask.return_value = self.test_mask
        mock_validate.return_value = False
        
        response = self.service.inpaint(self.test_request)
        
        self.assertFalse(response.success)
        self.assertIn("Mask incompatible", response.message)
    
    def test_load_mask_from_request_path(self):
        """Test loading mask from file path"""
        # Create temporary mask file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            self.test_mask.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Create request with mask path
            request = self.test_request.model_copy()
            request.mask_path = tmp_path
            request.mask_base64 = None
            
            mask = self.service._load_mask_from_request(request)
            
            self.assertIsNotNone(mask)
            self.assertEqual(mask.mode, "L")
            self.assertEqual(mask.size, (100, 100))
        finally:
            os.unlink(tmp_path)
    
    def test_load_mask_from_request_base64(self):
        """Test loading mask from base64"""
        with patch('src.diffsynth_models.decode_base64_to_image', return_value=self.test_mask) as mock_decode, \
             patch.object(self.service, '_load_and_validate_mask', return_value=self.test_mask) as mock_validate:
            mask = self.service._load_mask_from_request(self.test_request)
        
        self.assertIsNotNone(mask)
        mock_decode.assert_called_once_with("test_mask_data")
    
    def test_load_mask_from_request_no_input(self):
        """Test loading mask with no input provided"""
        request = self.test_request.model_copy()
        request.mask_path = None
        request.mask_base64 = None
        
        mask = self.service._load_mask_from_request(request)
        
        self.assertIsNone(mask)
    
    def test_load_and_validate_mask_pil_image(self):
        """Test loading and validating PIL Image mask"""
        mask = self.service._load_and_validate_mask(self.test_mask)
        
        self.assertIsNotNone(mask)
        self.assertEqual(mask.mode, "L")
        self.assertEqual(mask.size, (100, 100))
    
    def test_load_and_validate_mask_file_path(self):
        """Test loading and validating mask from file path"""
        # Create temporary mask file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            self.test_mask.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            mask = self.service._load_and_validate_mask(tmp_path)
            
            self.assertIsNotNone(mask)
            self.assertEqual(mask.mode, "L")
        finally:
            os.unlink(tmp_path)
    
    def test_load_and_validate_mask_nonexistent_file(self):
        """Test loading mask from nonexistent file"""
        mask = self.service._load_and_validate_mask("/nonexistent/mask.png")
        
        self.assertIsNone(mask)
    
    def test_load_and_validate_mask_invalid_type(self):
        """Test loading mask with invalid type"""
        mask = self.service._load_and_validate_mask(123)
        
        self.assertIsNone(mask)
    
    def test_validate_mask_properties_valid(self):
        """Test mask property validation with valid mask"""
        result = self.service._validate_mask_properties(self.test_mask)
        
        self.assertTrue(result)
    
    def test_validate_mask_properties_too_small(self):
        """Test mask property validation with too small mask"""
        small_mask = Image.new("L", (32, 32), color=255)
        
        result = self.service._validate_mask_properties(small_mask)
        
        self.assertFalse(result)
    
    def test_validate_mask_properties_too_large(self):
        """Test mask property validation with too large mask"""
        large_mask = Image.new("L", (5000, 5000), color=255)
        
        result = self.service._validate_mask_properties(large_mask)
        
        self.assertFalse(result)
    
    def test_validate_mask_compatibility_matching_size(self):
        """Test mask compatibility with matching dimensions"""
        result = self.service._validate_mask_compatibility(self.test_image, self.test_mask)
        
        self.assertTrue(result)
    
    def test_validate_mask_compatibility_different_size(self):
        """Test mask compatibility with different dimensions"""
        different_mask = Image.new("L", (200, 200), color=255)
        
        result = self.service._validate_mask_compatibility(self.test_image, different_mask)
        
        self.assertFalse(result)
    
    def test_validate_mask_compatibility_different_aspect_ratio(self):
        """Test mask compatibility with different aspect ratio"""
        different_mask = Image.new("L", (100, 200), color=255)
        
        result = self.service._validate_mask_compatibility(self.test_image, different_mask)
        
        self.assertFalse(result)
    
    def test_preprocess_mask_basic(self):
        """Test basic mask preprocessing"""
        processed_mask = self.service._preprocess_mask(
            self.test_mask, 
            target_size=(100, 100),
            blur_radius=0,
            invert=False
        )
        
        self.assertIsNotNone(processed_mask)
        self.assertEqual(processed_mask.mode, "L")
        self.assertEqual(processed_mask.size, (100, 100))
    
    def test_preprocess_mask_resize(self):
        """Test mask preprocessing with resizing"""
        processed_mask = self.service._preprocess_mask(
            self.test_mask, 
            target_size=(200, 200),
            blur_radius=0,
            invert=False
        )
        
        self.assertEqual(processed_mask.size, (200, 200))
    
    def test_preprocess_mask_blur(self):
        """Test mask preprocessing with blur"""
        processed_mask = self.service._preprocess_mask(
            self.test_mask, 
            target_size=(100, 100),
            blur_radius=4,
            invert=False
        )
        
        self.assertIsNotNone(processed_mask)
        # Blur should be applied (hard to test exact result)
    
    def test_preprocess_mask_invert(self):
        """Test mask preprocessing with inversion"""
        processed_mask = self.service._preprocess_mask(
            self.test_mask, 
            target_size=(100, 100),
            blur_radius=0,
            invert=True
        )
        
        self.assertIsNotNone(processed_mask)
        # Mask should be inverted (white becomes black)
    
    @patch('diffsynth_service.DiffSynthService._load_image_from_request')
    @patch('diffsynth_service.DiffSynthService._load_mask_from_request')
    @patch('diffsynth_service.DiffSynthService._save_edited_image')
    def test_inpaint_pipeline_error(self, mock_save, mock_load_mask, mock_load_image):
        """Test inpainting with pipeline error"""
        mock_load_image.return_value = self.test_image
        mock_load_mask.return_value = self.test_mask
        mock_save.return_value = "output/test.jpg"
        
        # Make the pipeline fail during inpainting
        self.mock_pipeline.side_effect = Exception("Pipeline error")
        
        response = self.service.inpaint(self.test_request)
        
        self.assertFalse(response.success)
        self.assertIn("failed", response.message)
        self.assertEqual(self.service.error_count, 1)
    
    @patch('diffsynth_service.DiffSynthService._load_image_from_request')
    @patch('diffsynth_service.DiffSynthService._load_mask_from_request')
    def test_inpaint_with_fallback(self, mock_load_mask, mock_load_image):
        """Test inpainting with fallback enabled"""
        self.service.config.enable_fallback = True
        mock_load_image.return_value = self.test_image
        mock_load_mask.return_value = self.test_mask
        self.mock_pipeline.side_effect = Exception("Pipeline error")
        
        response = self.service.inpaint(self.test_request)
        
        self.assertFalse(response.success)
        self.assertIn("failed", response.message.lower())
        self.assertIsNotNone(response.suggested_fixes)
    
    @patch('diffsynth_service.DiffSynthService._load_image_from_request')
    @patch('diffsynth_service.DiffSynthService._load_mask_from_request')
    @patch('diffsynth_service.DiffSynthService._save_edited_image')
    def test_inpaint_tiled_processing(self, mock_save, mock_load_mask, mock_load_image):
        """Test inpainting with tiled processing"""
        # Setup mocks
        mock_load_image.return_value = self.test_image
        mock_load_mask.return_value = self.test_mask
        mock_save.return_value = "output/inpainted.jpg"
        
        # Create request with tiled processing enabled
        request = self.test_request.model_copy()
        request.use_tiled_processing = True
        
        # Mock tiled processing methods
        with patch.object(self.service.tiled_processor, 'should_use_tiled_processing', return_value=True), \
             patch.object(self.service, '_inpaint_image_tiled', return_value=self.test_image) as mock_inpaint_tiled:
            response = self.service.inpaint(request)
        
        self.assertTrue(response.success)
        mock_inpaint_tiled.assert_called_once()
    
    def test_inpaint_single_pass(self):
        """Test single pass inpainting"""
        result = self.service._inpaint_image_single(self.test_request, self.test_image, self.test_mask)
        
        self.assertIsNotNone(result)
        self.mock_pipeline.assert_called_once()
        
        # Verify pipeline was called with correct parameters
        call_kwargs = self.mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs["prompt"], "Fill with blue flowers")
        self.assertIn("mask_image", call_kwargs)
        self.assertEqual(call_kwargs["num_inference_steps"], 20)
        self.assertEqual(call_kwargs["guidance_scale"], 7.5)
        self.assertEqual(call_kwargs["strength"], 0.9)
    
    def test_inpaint_single_pass_pipeline_error(self):
        """Test single pass inpainting with pipeline error"""
        self.mock_pipeline.side_effect = Exception("Pipeline error")
        
        result = self.service._inpaint_image_single(self.test_request, self.test_image, self.test_mask)
        
        self.assertIsNone(result)
    
    def test_inpaint_tiled_processing_implementation(self):
        """Test tiled inpainting implementation"""
        # Mock tile coordinates
        with patch.object(self.service.tiled_processor, 'calculate_tiles', return_value=[(0, 0, 50, 50), (50, 0, 100, 50), (0, 50, 50, 100), (50, 50, 100, 100)]) as mock_calc_tiles, \
             patch.object(self.service.tiled_processor, 'merge_tiles', return_value=self.test_image) as mock_merge_tiles, \
             patch.object(self.service, '_inpaint_image_single', return_value=self.test_image):
            result = self.service._inpaint_image_tiled(self.test_request, self.test_image, self.test_mask)
        
        self.assertIsNotNone(result)
        mock_calc_tiles.assert_called_once()
        mock_merge_tiles.assert_called_once()
    
    def test_fallback_inpaint_response(self):
        """Test fallback inpaint response"""
        response = self.service._fallback_inpaint_response(self.test_request)
        
        self.assertFalse(response.success)
        self.assertIn("failed", response.message.lower())
        self.assertIsNotNone(response.suggested_fixes)
        self.assertGreater(len(response.suggested_fixes), 0)


class TestCreateDiffSynthService(unittest.TestCase):
    """Test DiffSynth service factory function"""
    
    def test_create_service_defaults(self):
        """Test creating service with default parameters"""
        service = create_diffsynth_service()
        
        self.assertIsInstance(service, DiffSynthService)
        self.assertEqual(service.config.model_name, "Qwen/Qwen-Image-Edit")
        self.assertTrue(service.config.enable_vram_management)
    
    def test_create_service_custom_params(self):
        """Test creating service with custom parameters"""
        service = create_diffsynth_service(
            model_name="custom/model",
            device="cpu",
            enable_optimizations=False
        )
        
        self.assertIsInstance(service, DiffSynthService)
        self.assertEqual(service.config.model_name, "custom/model")
        self.assertEqual(service.config.device, "cpu")
        self.assertFalse(service.config.enable_vram_management)
    
    @patch('diffsynth_service.torch.cuda.is_available')
    def test_create_service_auto_device_detection(self, mock_cuda_available):
        """Test automatic device detection"""
        mock_cuda_available.return_value = True
        
        service = create_diffsynth_service()
        
        self.assertEqual(service.config.device, "cuda")


class TestResourceUsage(unittest.TestCase):
    """Test ResourceUsage dataclass"""
    
    def test_resource_usage_defaults(self):
        """Test ResourceUsage default values"""
        usage = ResourceUsage()
        
        self.assertEqual(usage.gpu_memory_allocated, 0.0)
        self.assertEqual(usage.gpu_memory_reserved, 0.0)
        self.assertEqual(usage.cpu_memory_used, 0.0)
        self.assertEqual(usage.processing_time, 0.0)
        self.assertEqual(usage.last_updated, 0.0)
    
    def test_resource_usage_custom_values(self):
        """Test ResourceUsage with custom values"""
        usage = ResourceUsage(
            gpu_memory_allocated=1.5,
            gpu_memory_reserved=2.0,
            cpu_memory_used=0.8,
            processing_time=5.2,
            last_updated=time.time()
        )
        
        self.assertEqual(usage.gpu_memory_allocated, 1.5)
        self.assertEqual(usage.gpu_memory_reserved, 2.0)
        self.assertEqual(usage.cpu_memory_used, 0.8)
        self.assertEqual(usage.processing_time, 5.2)
        self.assertGreater(usage.last_updated, 0)


class TestDiffSynthServiceOutpainting(unittest.TestCase):
    """Test DiffSynth service outpainting functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = DiffSynthConfig(device="cpu")
        self.service = DiffSynthService(self.config)
        
        # Mock pipeline for testing
        self.mock_pipeline = Mock()
        self.mock_pipeline.return_value = Image.new("RGB", (200, 200), color="blue")
        self.service.pipeline = self.mock_pipeline
        self.service.status = DiffSynthServiceStatus.READY
        
        # Create test image
        self.test_image = Image.new("RGB", (100, 100), color="red")
        
        # Create test OutpaintRequest
        from diffsynth_models import OutpaintRequest, OutpaintDirection
        self.test_request = OutpaintRequest(
            prompt="Extend this landscape with mountains",
            image_base64="test_image_data",
            direction=OutpaintDirection.ALL,
            pixels=64,  # Use minimum valid value
            num_inference_steps=20,
            guidance_scale=7.5,
            strength=0.8
        )
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self.service, 'shutdown'):
            self.service.shutdown()
    
    @patch('diffsynth_service.DiffSynthService._load_image_from_request')
    @patch('diffsynth_service.DiffSynthService._expand_canvas_for_outpainting')
    @patch('diffsynth_service.DiffSynthService._save_edited_image')
    def test_outpaint_success(self, mock_save, mock_expand, mock_load_image):
        """Test successful outpainting operation"""
        # Setup mocks
        mock_load_image.return_value = self.test_image
        expanded_image = Image.new("RGB", (200, 200), color="gray")
        expanded_mask = Image.new("L", (200, 200), color=255)
        mock_expand.return_value = (expanded_image, expanded_mask)
        mock_save.return_value = "output/outpainted.jpg"
        
        # Perform outpainting
        response = self.service.outpaint(self.test_request)
        
        # Verify response
        self.assertTrue(response.success)
        self.assertIn("successfully", response.message)
        self.assertEqual(response.operation, "outpaint")
        self.assertEqual(response.image_path, "output/outpainted.jpg")
        self.assertIsNotNone(response.processing_time)
        self.assertIsNotNone(response.parameters)
        
        # Verify service state
        self.assertEqual(self.service.operation_count, 1)
        self.assertGreater(self.service.last_operation_time, 0)
        
        # Verify pipeline was called
        self.mock_pipeline.assert_called_once()
    
    def test_outpaint_service_not_ready(self):
        """Test outpainting when service is not ready"""
        self.service.status = DiffSynthServiceStatus.NOT_INITIALIZED
        
        with patch.object(self.service, 'initialize', return_value=False):
            response = self.service.outpaint(self.test_request)
        
        self.assertFalse(response.success)
        self.assertIn("not available", response.message)
    
    @patch('diffsynth_service.DiffSynthService._load_image_from_request')
    def test_outpaint_invalid_image(self, mock_load_image):
        """Test outpainting with invalid image"""
        mock_load_image.return_value = None
        
        response = self.service.outpaint(self.test_request)
        
        self.assertFalse(response.success)
        self.assertIn("Failed to load input image", response.message)
    
    @patch('diffsynth_service.DiffSynthService._load_image_from_request')
    @patch('diffsynth_service.DiffSynthService._expand_canvas_for_outpainting')
    def test_outpaint_canvas_expansion_failure(self, mock_expand, mock_load_image):
        """Test outpainting when canvas expansion fails"""
        mock_load_image.return_value = self.test_image
        mock_expand.return_value = (None, None)
        
        response = self.service.outpaint(self.test_request)
        
        self.assertFalse(response.success)
        self.assertIn("Failed to expand canvas", response.message)
    
    def test_expand_canvas_all_directions(self):
        """Test canvas expansion in all directions"""
        from diffsynth_models import OutpaintDirection
        
        expanded_image, mask = self.service._expand_canvas_for_outpainting(
            self.test_image, OutpaintDirection.ALL, 64, "edge"
        )
        
        self.assertIsNotNone(expanded_image)
        self.assertIsNotNone(mask)
        self.assertEqual(expanded_image.size, (228, 228))  # 100 + 64*2
        self.assertEqual(mask.size, (228, 228))
    
    def test_expand_canvas_left_direction(self):
        """Test canvas expansion to the left"""
        from diffsynth_models import OutpaintDirection
        
        expanded_image, mask = self.service._expand_canvas_for_outpainting(
            self.test_image, OutpaintDirection.LEFT, 64, "edge"
        )
        
        self.assertIsNotNone(expanded_image)
        self.assertIsNotNone(mask)
        self.assertEqual(expanded_image.size, (164, 100))  # 100 + 64
        self.assertEqual(mask.size, (164, 100))
    
    def test_expand_canvas_right_direction(self):
        """Test canvas expansion to the right"""
        from diffsynth_models import OutpaintDirection
        
        expanded_image, mask = self.service._expand_canvas_for_outpainting(
            self.test_image, OutpaintDirection.RIGHT, 64, "edge"
        )
        
        self.assertIsNotNone(expanded_image)
        self.assertIsNotNone(mask)
        self.assertEqual(expanded_image.size, (164, 100))  # 100 + 64
        self.assertEqual(mask.size, (164, 100))
    
    def test_expand_canvas_top_direction(self):
        """Test canvas expansion to the top"""
        from diffsynth_models import OutpaintDirection
        
        expanded_image, mask = self.service._expand_canvas_for_outpainting(
            self.test_image, OutpaintDirection.TOP, 64, "edge"
        )
        
        self.assertIsNotNone(expanded_image)
        self.assertIsNotNone(mask)
        self.assertEqual(expanded_image.size, (100, 164))  # 100 + 64
        self.assertEqual(mask.size, (100, 164))
    
    def test_expand_canvas_bottom_direction(self):
        """Test canvas expansion to the bottom"""
        from diffsynth_models import OutpaintDirection
        
        expanded_image, mask = self.service._expand_canvas_for_outpainting(
            self.test_image, OutpaintDirection.BOTTOM, 64, "edge"
        )
        
        self.assertIsNotNone(expanded_image)
        self.assertIsNotNone(mask)
        self.assertEqual(expanded_image.size, (100, 164))  # 100 + 64
        self.assertEqual(mask.size, (100, 164))
    
    def test_expand_canvas_invalid_direction(self):
        """Test canvas expansion with invalid direction"""
        expanded_image, mask = self.service._expand_canvas_for_outpainting(
            self.test_image, "invalid_direction", 64, "edge"
        )
        
        self.assertIsNone(expanded_image)
        self.assertIsNone(mask)
    
    def test_create_outpainting_mask(self):
        """Test outpainting mask creation"""
        mask = self.service._create_outpainting_mask(100, 100, 200, 200, 50, 50)
        
        self.assertIsNotNone(mask)
        self.assertEqual(mask.size, (200, 200))
        self.assertEqual(mask.mode, "L")
        
        # Check that mask has white areas for outpainting regions
        import numpy as np
        mask_array = np.array(mask)
        
        # Check that there are white pixels (areas to fill)
        self.assertTrue(np.any(mask_array == 255))
        # Check that there are black pixels (areas to preserve)
        self.assertTrue(np.any(mask_array == 0))
    
    @patch('diffsynth_service.DiffSynthService._load_image_from_request')
    @patch('diffsynth_service.DiffSynthService._expand_canvas_for_outpainting')
    @patch('diffsynth_service.DiffSynthService._save_edited_image')
    def test_outpaint_with_tiled_processing(self, mock_save, mock_expand, mock_load_image):
        """Test outpainting with tiled processing"""
        # Setup mocks
        mock_load_image.return_value = self.test_image
        large_expanded_image = Image.new("RGB", (2000, 2000), color="gray")
        large_expanded_mask = Image.new("L", (2000, 2000), color=255)
        mock_expand.return_value = (large_expanded_image, large_expanded_mask)
        mock_save.return_value = "output/outpainted_tiled.jpg"
        
        # Force tiled processing
        self.test_request.use_tiled_processing = True
        
        # Perform outpainting
        response = self.service.outpaint(self.test_request)
        
        # Verify response
        self.assertTrue(response.success)
        self.assertIn("successfully", response.message)
        self.assertEqual(response.operation, "outpaint")
    
    @patch('diffsynth_service.DiffSynthService._load_image_from_request')
    @patch('diffsynth_service.DiffSynthService._expand_canvas_for_outpainting')
    def test_outpaint_pipeline_failure_with_fallback(self, mock_expand, mock_load_image):
        """Test outpainting pipeline failure with fallback enabled"""
        # Setup mocks
        mock_load_image.return_value = self.test_image
        expanded_image = Image.new("RGB", (200, 200), color="gray")
        expanded_mask = Image.new("L", (200, 200), color=255)
        mock_expand.return_value = (expanded_image, expanded_mask)
        
        # Make pipeline fail
        self.mock_pipeline.side_effect = Exception("Pipeline error")
        
        # Enable fallback
        self.service.config.enable_fallback = True
        
        # Perform outpainting
        response = self.service.outpaint(self.test_request)
        
        # Verify fallback response
        self.assertFalse(response.success)
        self.assertIn("outpainting failed", response.message.lower())
        self.assertIsNotNone(response.suggested_fixes)
    
    def test_outpaint_different_fill_modes(self):
        """Test outpainting with different fill modes"""
        from diffsynth_models import OutpaintDirection
        
        # Test edge fill mode
        expanded_image_edge, _ = self.service._expand_canvas_for_outpainting(
            self.test_image, OutpaintDirection.ALL, 64, "edge"
        )
        self.assertIsNotNone(expanded_image_edge)
        
        # Test constant fill mode
        expanded_image_constant, _ = self.service._expand_canvas_for_outpainting(
            self.test_image, OutpaintDirection.ALL, 64, "constant"
        )
        self.assertIsNotNone(expanded_image_constant)
        
        # Test reflect fill mode
        expanded_image_reflect, _ = self.service._expand_canvas_for_outpainting(
            self.test_image, OutpaintDirection.ALL, 64, "reflect"
        )
        self.assertIsNotNone(expanded_image_reflect)
    
    def test_outpaint_request_validation(self):
        """Test OutpaintRequest validation"""
        from diffsynth_models import OutpaintRequest, OutpaintDirection
        
        # Valid request
        valid_request = OutpaintRequest(
            prompt="Test prompt",
            image_base64="test_data",
            direction=OutpaintDirection.ALL,
            pixels=100
        )
        self.assertEqual(valid_request.direction, OutpaintDirection.ALL)
        self.assertEqual(valid_request.pixels, 100)
        
        # Test default values
        self.assertTrue(valid_request.auto_expand_canvas)
        self.assertEqual(valid_request.fill_mode, "edge")


class TestDiffSynthServiceStyleTransfer(unittest.TestCase):
    """Test DiffSynth service style transfer functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = DiffSynthConfig(device="cpu")
        self.service = DiffSynthService(self.config)
        
        # Mock pipeline for testing
        self.mock_pipeline = Mock()
        self.mock_pipeline.return_value = Image.new("RGB", (100, 100), color="green")
        self.service.pipeline = self.mock_pipeline
        self.service.status = DiffSynthServiceStatus.READY
        
        # Create test images
        self.test_content_image = Image.new("RGB", (100, 100), color="red")
        self.test_style_image = Image.new("RGB", (100, 100), color="blue")
        
        # Create test StyleTransferRequest
        from diffsynth_models import StyleTransferRequest
        self.test_request = StyleTransferRequest(
            prompt="Apply artistic style to this photo",
            image_base64="test_content_data",
            style_image_base64="test_style_data",
            style_strength=0.7,
            content_strength=0.3,
            num_inference_steps=25,
            guidance_scale=8.0
        )
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self.service, 'shutdown'):
            self.service.shutdown()
    
    @patch('diffsynth_service.DiffSynthService._load_image_from_request')
    @patch('diffsynth_service.DiffSynthService._load_style_image_from_request')
    @patch('diffsynth_service.DiffSynthService._save_edited_image')
    def test_style_transfer_success(self, mock_save, mock_load_style, mock_load_content):
        """Test successful style transfer operation"""
        # Setup mocks
        mock_load_content.return_value = self.test_content_image
        mock_load_style.return_value = self.test_style_image
        mock_save.return_value = "output/styled.jpg"
        
        # Perform style transfer
        response = self.service.style_transfer(self.test_request)
        
        # Verify response
        self.assertTrue(response.success)
        self.assertIn("successfully", response.message)
        self.assertEqual(response.operation, "style_transfer")
        self.assertEqual(response.image_path, "output/styled.jpg")
        self.assertIsNotNone(response.processing_time)
        self.assertIsNotNone(response.parameters)
        
        # Verify parameters in response
        self.assertEqual(response.parameters["style_strength"], 0.7)
        self.assertEqual(response.parameters["content_strength"], 0.3)
        
        # Verify service state
        self.assertEqual(self.service.operation_count, 1)
        self.assertGreater(self.service.last_operation_time, 0)
        
        # Verify pipeline was called
        self.mock_pipeline.assert_called_once()
    
    def test_style_transfer_service_not_ready(self):
        """Test style transfer when service is not ready"""
        self.service.status = DiffSynthServiceStatus.NOT_INITIALIZED
        
        with patch.object(self.service, 'initialize', return_value=False):
            response = self.service.style_transfer(self.test_request)
        
        self.assertFalse(response.success)
        self.assertIn("not available", response.message)
    
    @patch('diffsynth_service.DiffSynthService._load_image_from_request')
    def test_style_transfer_invalid_content_image(self, mock_load_content):
        """Test style transfer with invalid content image"""
        mock_load_content.return_value = None
        
        response = self.service.style_transfer(self.test_request)
        
        self.assertFalse(response.success)
        self.assertIn("Failed to load content image", response.message)
    
    @patch('diffsynth_service.DiffSynthService._load_image_from_request')
    @patch('diffsynth_service.DiffSynthService._load_style_image_from_request')
    def test_style_transfer_invalid_style_image(self, mock_load_style, mock_load_content):
        """Test style transfer with invalid style image"""
        mock_load_content.return_value = self.test_content_image
        mock_load_style.return_value = None
        
        response = self.service.style_transfer(self.test_request)
        
        self.assertFalse(response.success)
        self.assertIn("Failed to load style image", response.message)
    
    def test_load_style_image_from_request_path(self):
        """Test loading style image from file path"""
        # Create temporary style image file with minimum size (256x256)
        large_style_image = Image.new("RGB", (256, 256), color="blue")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            large_style_image.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Create request with style image path
            request = self.test_request.model_copy()
            request.style_image_path = tmp_path
            request.style_image_base64 = None
            
            style_image = self.service._load_style_image_from_request(request)
            
            self.assertIsNotNone(style_image)
            self.assertEqual(style_image.mode, "RGB")
            self.assertEqual(style_image.size, (256, 256))
        finally:
            os.unlink(tmp_path)
    
    def test_load_style_image_from_request_base64(self):
        """Test loading style image from base64"""
        with patch('src.diffsynth_models.decode_base64_to_image', return_value=self.test_style_image) as mock_decode, \
             patch.object(self.service.preprocessor, 'load_and_validate_image', return_value=self.test_style_image):
            style_image = self.service._load_style_image_from_request(self.test_request)
        
        self.assertIsNotNone(style_image)
        mock_decode.assert_called_once_with("test_style_data")
    
    def test_load_style_image_from_request_no_input(self):
        """Test loading style image with no input provided"""
        request = self.test_request.model_copy()
        request.style_image_path = None
        request.style_image_base64 = None
        
        style_image = self.service._load_style_image_from_request(request)
        
        self.assertIsNone(style_image)
    
    def test_style_transfer_single_pass(self):
        """Test single pass style transfer"""
        with patch.object(self.service, '_create_style_transfer_prompt', return_value="enhanced prompt") as mock_prompt, \
             patch.object(self.service, '_blend_style_and_content', return_value=self.test_content_image) as mock_blend:
            
            result = self.service._style_transfer_single(
                self.test_request, 
                self.test_content_image, 
                self.test_style_image
            )
        
        self.assertIsNotNone(result)
        mock_prompt.assert_called_once()
        mock_blend.assert_called_once()
        self.mock_pipeline.assert_called_once()
    
    def test_style_transfer_single_pass_pipeline_error(self):
        """Test single pass style transfer with pipeline error"""
        self.mock_pipeline.side_effect = Exception("Pipeline error")
        
        result = self.service._style_transfer_single(
            self.test_request, 
            self.test_content_image, 
            self.test_style_image
        )
        
        self.assertIsNone(result)
    
    @patch('diffsynth_service.DiffSynthService._style_transfer_single')
    def test_style_transfer_tiled_processing(self, mock_single):
        """Test style transfer with tiled processing"""
        # Create large image to trigger tiled processing
        large_content = Image.new("RGB", (2048, 2048), color="red")
        large_style = Image.new("RGB", (2048, 2048), color="blue")
        
        # Mock single pass to return a tile
        mock_single.return_value = Image.new("RGB", (512, 512), color="green")
        
        with patch.object(self.service.tiled_processor, 'calculate_tiles', 
                         return_value=[(0, 0, 512, 512), (512, 0, 1024, 512)]) as mock_calc, \
             patch.object(self.service.tiled_processor, 'merge_tiles', 
                         return_value=large_content) as mock_merge:
            
            result = self.service._style_transfer_tiled(
                self.test_request, 
                large_content, 
                large_style
            )
        
        self.assertIsNotNone(result)
        mock_calc.assert_called_once()
        mock_merge.assert_called_once()
        # Should call single pass for each tile
        self.assertEqual(mock_single.call_count, 2)
    
    def test_create_style_transfer_prompt_high_style_strength(self):
        """Test style transfer prompt creation with high style strength"""
        prompt = self.service._create_style_transfer_prompt(
            "test prompt", 
            style_strength=0.9, 
            content_strength=0.1
        )
        
        self.assertIn("test prompt", prompt)
        self.assertIn("heavily stylized", prompt)
        self.assertIn("creative interpretation", prompt)
    
    def test_create_style_transfer_prompt_medium_style_strength(self):
        """Test style transfer prompt creation with medium style strength"""
        prompt = self.service._create_style_transfer_prompt(
            "test prompt", 
            style_strength=0.6, 
            content_strength=0.4
        )
        
        self.assertIn("test prompt", prompt)
        self.assertIn("stylized", prompt)
        self.assertIn("allowing creative interpretation", prompt)
    
    def test_create_style_transfer_prompt_low_style_strength(self):
        """Test style transfer prompt creation with low style strength"""
        prompt = self.service._create_style_transfer_prompt(
            "test prompt", 
            style_strength=0.3, 
            content_strength=0.8
        )
        
        self.assertIn("test prompt", prompt)
        self.assertIn("subtle artistic influence", prompt)
        self.assertIn("preserving original details", prompt)
    
    def test_blend_style_and_content_same_size(self):
        """Test blending style and content images of same size"""
        styled_image = Image.new("RGB", (100, 100), color="green")
        content_image = Image.new("RGB", (100, 100), color="red")
        
        blended = self.service._blend_style_and_content(
            styled_image, 
            content_image, 
            content_strength=0.5
        )
        
        self.assertIsNotNone(blended)
        self.assertEqual(blended.size, (100, 100))
        self.assertEqual(blended.mode, "RGB")
    
    def test_blend_style_and_content_different_sizes(self):
        """Test blending style and content images of different sizes"""
        styled_image = Image.new("RGB", (100, 100), color="green")
        content_image = Image.new("RGB", (200, 200), color="red")
        
        blended = self.service._blend_style_and_content(
            styled_image, 
            content_image, 
            content_strength=0.3
        )
        
        self.assertIsNotNone(blended)
        self.assertEqual(blended.size, (100, 100))  # Should match styled image size
    
    @patch('diffsynth_service.DiffSynthService._load_image_from_request')
    @patch('diffsynth_service.DiffSynthService._load_style_image_from_request')
    def test_style_transfer_with_fallback(self, mock_load_style, mock_load_content):
        """Test style transfer with fallback enabled"""
        self.service.config.enable_fallback = True
        mock_load_content.return_value = self.test_content_image
        mock_load_style.return_value = self.test_style_image
        self.mock_pipeline.side_effect = Exception("Pipeline error")
        
        response = self.service.style_transfer(self.test_request)
        
        self.assertFalse(response.success)
        self.assertIn("failed", response.message.lower())
        self.assertIsNotNone(response.suggested_fixes)
        self.assertIn("style transfer", response.suggested_fixes[0])
    
    def test_fallback_style_transfer_response(self):
        """Test fallback response for style transfer"""
        response = self.service._fallback_style_transfer_response(self.test_request)
        
        self.assertFalse(response.success)
        self.assertIn("failed", response.message)
        self.assertIn("style transfer", response.message)
        self.assertIsNotNone(response.suggested_fixes)
        self.assertGreater(len(response.suggested_fixes), 0)
        
        # Check that suggested fixes are relevant to style transfer
        fixes_text = " ".join(response.suggested_fixes)
        self.assertIn("style", fixes_text.lower())


class TestDiffSynthServiceStyleTransferIntegration(unittest.TestCase):
    """Integration tests for style transfer with different image types"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = DiffSynthConfig(device="cpu")
        self.service = DiffSynthService(self.config)
        self.service.status = DiffSynthServiceStatus.READY
        
        # Mock pipeline
        self.mock_pipeline = Mock()
        self.service.pipeline = self.mock_pipeline
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self.service, 'shutdown'):
            self.service.shutdown()
    
    def test_style_transfer_photo_to_painting(self):
        """Test style transfer from photo to painting style"""
        # Create realistic test images
        photo_image = Image.new("RGB", (512, 512), color=(128, 64, 32))  # Brown photo
        painting_style = Image.new("RGB", (256, 256), color=(255, 100, 50))  # Orange painting
        
        # Mock the pipeline to return a styled result
        styled_result = Image.new("RGB", (512, 512), color=(200, 80, 40))  # Blended result
        self.mock_pipeline.return_value = styled_result
        
        from diffsynth_models import StyleTransferRequest
        request = StyleTransferRequest(
            prompt="Transform this photo into an impressionist painting",
            image_base64="photo_data",
            style_image_base64="painting_data",
            style_strength=0.8,
            content_strength=0.2,
            num_inference_steps=30
        )
        
        with patch.object(self.service, '_load_image_from_request', return_value=photo_image), \
             patch.object(self.service, '_load_style_image_from_request', return_value=painting_style), \
             patch.object(self.service, '_save_edited_image', return_value="output/photo_to_painting.jpg"):
            
            response = self.service.style_transfer(request)
        
        self.assertTrue(response.success)
        self.assertEqual(response.parameters["style_strength"], 0.8)
        self.assertEqual(response.parameters["content_strength"], 0.2)
    
    def test_style_transfer_different_aspect_ratios(self):
        """Test style transfer with different aspect ratios"""
        # Portrait content image
        portrait_image = Image.new("RGB", (400, 600), color="red")
        # Landscape style image  
        landscape_style = Image.new("RGB", (800, 400), color="blue")
        
        styled_result = Image.new("RGB", (400, 600), color="purple")
        self.mock_pipeline.return_value = styled_result
        
        from diffsynth_models import StyleTransferRequest
        request = StyleTransferRequest(
            prompt="Apply landscape style to portrait",
            image_base64="portrait_data",
            style_image_base64="landscape_data",
            style_strength=0.6,
            content_strength=0.4
        )
        
        with patch.object(self.service, '_load_image_from_request', return_value=portrait_image), \
             patch.object(self.service, '_load_style_image_from_request', return_value=landscape_style), \
             patch.object(self.service, '_save_edited_image', return_value="output/aspect_ratio_test.jpg"):
            
            response = self.service.style_transfer(request)
        
        self.assertTrue(response.success)
        # Verify that preprocessing handled the aspect ratio difference
        self.assertIsNotNone(response.processing_time)
    
    def test_style_transfer_extreme_parameters(self):
        """Test style transfer with extreme parameter values"""
        content_image = Image.new("RGB", (256, 256), color="gray")
        style_image = Image.new("RGB", (256, 256), color="yellow")
        
        styled_result = Image.new("RGB", (256, 256), color="orange")
        self.mock_pipeline.return_value = styled_result
        
        from diffsynth_models import StyleTransferRequest
        request = StyleTransferRequest(
            prompt="Extreme style transfer test",
            image_base64="content_data",
            style_image_base64="style_data",
            style_strength=1.0,  # Maximum style
            content_strength=0.1,  # Minimum content
            num_inference_steps=50,
            guidance_scale=15.0
        )
        
        with patch.object(self.service, '_load_image_from_request', return_value=content_image), \
             patch.object(self.service, '_load_style_image_from_request', return_value=style_image), \
             patch.object(self.service, '_save_edited_image', return_value="output/extreme_params.jpg"):
            
            response = self.service.style_transfer(request)
        
        self.assertTrue(response.success)
        self.assertEqual(response.parameters["style_strength"], 1.0)
        self.assertEqual(response.parameters["content_strength"], 0.1)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)