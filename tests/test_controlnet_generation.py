"""
Tests for ControlNet-guided generation functionality
Tests ControlNet generation with different control types and parameters
"""

import pytest
import numpy as np
from PIL import Image
import tempfile
import os
import base64
from io import BytesIO
from unittest.mock import Mock, patch

from src.controlnet_service import (
    ControlNetService, ControlNetType, ControlNetRequest
)


class TestControlNetGeneration:
    """Test ControlNet-guided generation functionality"""
    
    @pytest.fixture
    def controlnet_service(self):
        """Create ControlNet service for testing"""
        return ControlNetService(device="cpu")
    
    @pytest.fixture
    def test_image(self):
        """Create test image for generation"""
        image_array = np.zeros((128, 128, 3), dtype=np.uint8)
        image_array[32:96, 32:96] = [128, 128, 128]
        return Image.fromarray(image_array)
    
    @pytest.fixture
    def control_image(self):
        """Create control image for testing"""
        # Simple edge-like control image
        image_array = np.zeros((128, 128, 3), dtype=np.uint8)
        image_array[32:33, 32:96] = [255, 255, 255]  # Top edge
        image_array[95:96, 32:96] = [255, 255, 255]  # Bottom edge
        image_array[32:96, 32:33] = [255, 255, 255]  # Left edge
        image_array[32:96, 95:96] = [255, 255, 255]  # Right edge
        return Image.fromarray(image_array)
    
    def test_process_with_control_basic_request(self, controlnet_service, test_image):
        """Test basic ControlNet generation request"""
        request = ControlNetRequest(
            prompt="a beautiful landscape",
            image_path="dummy_path.jpg",  # Provide a path so the method tries to load it
            control_type=ControlNetType.CANNY
        )
        
        # Mock the image loading to use our test image
        with patch.object(controlnet_service, '_load_image_from_request', return_value=test_image):
            result = controlnet_service.process_with_control(request)
        
        assert isinstance(result, dict)
        assert result["success"] is True
        assert "processing_time" in result
        assert "control_type" in result
        assert result["control_type"] == "canny"
        assert "image_path" in result
    
    def test_process_with_control_auto_detection(self, controlnet_service, test_image):
        """Test ControlNet generation with automatic control type detection"""
        request = ControlNetRequest(
            prompt="test prompt",
            image_path="dummy_path.jpg",  # Provide a path
            control_type=ControlNetType.AUTO
        )
        
        with patch.object(controlnet_service, '_load_image_from_request', return_value=test_image):
            result = controlnet_service.process_with_control(request)
        
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["control_type"] in ["canny", "depth", "pose", "normal", "segmentation"]
    
    def test_process_with_control_provided_control_image(self, controlnet_service, test_image, control_image):
        """Test ControlNet generation with provided control image"""
        request = ControlNetRequest(
            prompt="test prompt",
            control_image_path="dummy_control.jpg",  # Provide control image path
            control_type=ControlNetType.CANNY
        )
        
        # Mock loading both input and control images
        with patch.object(controlnet_service, '_load_image_from_request', return_value=None), \
             patch.object(controlnet_service, '_load_control_image_from_request', return_value=control_image):
            result = controlnet_service.process_with_control(request)
        
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["control_type"] == "canny"
    
    def test_process_with_control_custom_parameters(self, controlnet_service, test_image):
        """Test ControlNet generation with custom parameters"""
        request = ControlNetRequest(
            prompt="detailed artwork",
            image_path="dummy_path.jpg",  # Provide image path
            control_type=ControlNetType.DEPTH,
            controlnet_conditioning_scale=0.8,
            control_guidance_start=0.1,
            control_guidance_end=0.9,
            num_inference_steps=30,
            guidance_scale=8.0,
            width=256,
            height=256,
            seed=42
        )
        
        with patch.object(controlnet_service, '_load_image_from_request', return_value=test_image):
            result = controlnet_service.process_with_control(request)
        
        assert isinstance(result, dict)
        assert result["success"] is True
        
        # Check that parameters are preserved
        params = result["parameters"]
        assert params["controlnet_conditioning_scale"] == 0.8
        assert params["control_guidance_start"] == 0.1
        assert params["control_guidance_end"] == 0.9
        assert params["num_inference_steps"] == 30
        assert params["guidance_scale"] == 8.0
        assert params["width"] == 256
        assert params["height"] == 256
        assert params["seed"] == 42
    
    def test_process_with_control_all_types(self, controlnet_service, test_image):
        """Test ControlNet generation with all control types"""
        control_types = [
            ControlNetType.CANNY,
            ControlNetType.DEPTH,
            ControlNetType.POSE,
            ControlNetType.NORMAL,
            ControlNetType.SEGMENTATION
        ]
        
        for control_type in control_types:
            request = ControlNetRequest(
                prompt=f"test with {control_type.value}",
                image_path="dummy_path.jpg",  # Provide image path
                control_type=control_type
            )
            
            with patch.object(controlnet_service, '_load_image_from_request', return_value=test_image):
                result = controlnet_service.process_with_control(request)
            
            assert isinstance(result, dict)
            assert result["success"] is True
            assert result["control_type"] == control_type.value
    
    def test_process_with_control_file_path_input(self, controlnet_service, test_image):
        """Test ControlNet generation with file path input"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            test_image.save(tmp_file.name)
            
            try:
                request = ControlNetRequest(
                    prompt="test prompt",
                    image_path=tmp_file.name,
                    control_type=ControlNetType.CANNY
                )
                
                result = controlnet_service.process_with_control(request)
                
                assert isinstance(result, dict)
                assert result["success"] is True
            finally:
                os.unlink(tmp_file.name)
    
    def test_process_with_control_base64_input(self, controlnet_service, test_image):
        """Test ControlNet generation with base64 input"""
        # Convert image to base64
        buffer = BytesIO()
        test_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        request = ControlNetRequest(
            prompt="test prompt",
            image_base64=image_base64,
            control_type=ControlNetType.CANNY
        )
        
        result = controlnet_service.process_with_control(request)
        
        assert isinstance(result, dict)
        assert result["success"] is True
    
    def test_process_with_control_error_handling_no_input(self, controlnet_service):
        """Test error handling when no input image is provided"""
        request = ControlNetRequest(
            prompt="test prompt",
            control_type=ControlNetType.CANNY
        )
        
        result = controlnet_service.process_with_control(request)
        
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "error_details" in result
        assert "Either input image or control image must be provided" in result["error_details"]
    
    def test_process_with_control_error_handling_invalid_image(self, controlnet_service):
        """Test error handling with invalid image path"""
        request = ControlNetRequest(
            prompt="test prompt",
            image_path="non_existent_file.jpg",
            control_type=ControlNetType.CANNY
        )
        
        result = controlnet_service.process_with_control(request)
        
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "error_details" in result
    
    def test_process_with_control_error_handling_invalid_base64(self, controlnet_service):
        """Test error handling with invalid base64 data"""
        request = ControlNetRequest(
            prompt="test prompt",
            image_base64="invalid_base64_data",
            control_type=ControlNetType.CANNY
        )
        
        result = controlnet_service.process_with_control(request)
        
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "error_details" in result
    
    def test_controlnet_conditioning_scale_range(self, controlnet_service, test_image):
        """Test ControlNet conditioning scale parameter range"""
        scales = [0.0, 0.5, 1.0, 1.5, 2.0]
        
        for scale in scales:
            request = ControlNetRequest(
                prompt="test prompt",
                image_path="dummy_path.jpg",  # Provide image path
                control_type=ControlNetType.CANNY,
                controlnet_conditioning_scale=scale
            )
            
            with patch.object(controlnet_service, '_load_image_from_request', return_value=test_image):
                result = controlnet_service.process_with_control(request)
            
            assert isinstance(result, dict)
            assert result["success"] is True
            assert result["parameters"]["controlnet_conditioning_scale"] == scale
    
    def test_control_guidance_parameters(self, controlnet_service, test_image):
        """Test control guidance start and end parameters"""
        guidance_configs = [
            (0.0, 1.0),  # Full guidance
            (0.2, 0.8),  # Partial guidance
            (0.5, 1.0),  # Late guidance
            (0.0, 0.5),  # Early guidance only
        ]
        
        for start, end in guidance_configs:
            request = ControlNetRequest(
                prompt="test prompt",
                image_path="dummy_path.jpg",  # Provide image path
                control_type=ControlNetType.CANNY,
                control_guidance_start=start,
                control_guidance_end=end
            )
            
            with patch.object(controlnet_service, '_load_image_from_request', return_value=test_image):
                result = controlnet_service.process_with_control(request)
            
            assert isinstance(result, dict)
            assert result["success"] is True
            assert result["parameters"]["control_guidance_start"] == start
            assert result["parameters"]["control_guidance_end"] == end
    
    def test_generation_parameters_validation(self, controlnet_service, test_image):
        """Test that generation parameters are properly validated and applied"""
        request = ControlNetRequest(
            prompt="test prompt",
            image_path="dummy_path.jpg",  # Provide image path
            negative_prompt="blurry, low quality",
            control_type=ControlNetType.CANNY,
            num_inference_steps=25,
            guidance_scale=7.5,
            width=512,
            height=512,
            seed=123
        )
        
        with patch.object(controlnet_service, '_load_image_from_request', return_value=test_image):
            result = controlnet_service.process_with_control(request)
        
        assert isinstance(result, dict)
        assert result["success"] is True
        
        params = result["parameters"]
        assert params["prompt"] == "test prompt"
        assert params["negative_prompt"] == "blurry, low quality"
        assert params["num_inference_steps"] == 25
        assert params["guidance_scale"] == 7.5
        assert params["width"] == 512
        assert params["height"] == 512
        assert params["seed"] == 123


class TestControlNetRequestModel:
    """Test ControlNetRequest model functionality"""
    
    def test_controlnet_request_defaults(self):
        """Test ControlNetRequest default values"""
        request = ControlNetRequest(prompt="test")
        
        assert request.prompt == "test"
        assert request.image_path is None
        assert request.image_base64 is None
        assert request.control_image_path is None
        assert request.control_image_base64 is None
        assert request.control_type == ControlNetType.AUTO
        assert request.controlnet_conditioning_scale == 1.0
        assert request.control_guidance_start == 0.0
        assert request.control_guidance_end == 1.0
        assert request.negative_prompt is None
        assert request.num_inference_steps == 20
        assert request.guidance_scale == 7.5
        assert request.width == 768
        assert request.height == 768
        assert request.seed is None
        assert request.use_tiled_processing is None
        assert request.additional_params is None
    
    def test_controlnet_request_custom_values(self):
        """Test ControlNetRequest with custom values"""
        additional_params = {"custom_param": "value"}
        
        request = ControlNetRequest(
            prompt="custom prompt",
            image_path="/path/to/image.jpg",
            image_base64="base64data",
            control_image_path="/path/to/control.jpg",
            control_image_base64="control_base64",
            control_type=ControlNetType.DEPTH,
            controlnet_conditioning_scale=0.8,
            control_guidance_start=0.1,
            control_guidance_end=0.9,
            negative_prompt="bad quality",
            num_inference_steps=30,
            guidance_scale=8.0,
            width=512,
            height=512,
            seed=42,
            use_tiled_processing=True,
            additional_params=additional_params
        )
        
        assert request.prompt == "custom prompt"
        assert request.image_path == "/path/to/image.jpg"
        assert request.image_base64 == "base64data"
        assert request.control_image_path == "/path/to/control.jpg"
        assert request.control_image_base64 == "control_base64"
        assert request.control_type == ControlNetType.DEPTH
        assert request.controlnet_conditioning_scale == 0.8
        assert request.control_guidance_start == 0.1
        assert request.control_guidance_end == 0.9
        assert request.negative_prompt == "bad quality"
        assert request.num_inference_steps == 30
        assert request.guidance_scale == 8.0
        assert request.width == 512
        assert request.height == 512
        assert request.seed == 42
        assert request.use_tiled_processing is True
        assert request.additional_params == additional_params


class TestControlNetImageLoading:
    """Test image loading functionality for ControlNet"""
    
    @pytest.fixture
    def controlnet_service(self):
        """Create ControlNet service for testing"""
        return ControlNetService(device="cpu")
    
    @pytest.fixture
    def test_image(self):
        """Create test image"""
        image_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        return Image.fromarray(image_array)
    
    def test_load_image_from_request_file_path(self, controlnet_service, test_image):
        """Test loading image from file path"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            test_image.save(tmp_file.name)
            
            try:
                request = ControlNetRequest(
                    prompt="test",
                    image_path=tmp_file.name
                )
                
                loaded_image = controlnet_service._load_image_from_request(request)
                
                assert loaded_image is not None
                assert isinstance(loaded_image, Image.Image)
                assert loaded_image.mode == "RGB"
            finally:
                os.unlink(tmp_file.name)
    
    def test_load_image_from_request_base64(self, controlnet_service, test_image):
        """Test loading image from base64 data"""
        # Convert to base64
        buffer = BytesIO()
        test_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        request = ControlNetRequest(
            prompt="test",
            image_base64=image_base64
        )
        
        loaded_image = controlnet_service._load_image_from_request(request)
        
        assert loaded_image is not None
        assert isinstance(loaded_image, Image.Image)
        assert loaded_image.mode == "RGB"
    
    def test_load_control_image_from_request_file_path(self, controlnet_service, test_image):
        """Test loading control image from file path"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            test_image.save(tmp_file.name)
            
            try:
                request = ControlNetRequest(
                    prompt="test",
                    control_image_path=tmp_file.name
                )
                
                loaded_image = controlnet_service._load_control_image_from_request(request)
                
                assert loaded_image is not None
                assert isinstance(loaded_image, Image.Image)
                assert loaded_image.mode == "RGB"
            finally:
                os.unlink(tmp_file.name)
    
    def test_load_control_image_from_request_base64(self, controlnet_service, test_image):
        """Test loading control image from base64 data"""
        # Convert to base64
        buffer = BytesIO()
        test_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        request = ControlNetRequest(
            prompt="test",
            control_image_base64=image_base64
        )
        
        loaded_image = controlnet_service._load_control_image_from_request(request)
        
        assert loaded_image is not None
        assert isinstance(loaded_image, Image.Image)
        assert loaded_image.mode == "RGB"
    
    def test_load_image_from_request_no_input(self, controlnet_service):
        """Test loading image when no input is provided"""
        request = ControlNetRequest(prompt="test")
        
        loaded_image = controlnet_service._load_image_from_request(request)
        
        assert loaded_image is None
    
    def test_load_image_from_request_invalid_path(self, controlnet_service):
        """Test loading image with invalid file path"""
        request = ControlNetRequest(
            prompt="test",
            image_path="non_existent_file.jpg"
        )
        
        loaded_image = controlnet_service._load_image_from_request(request)
        
        assert loaded_image is None
    
    def test_load_image_from_request_invalid_base64(self, controlnet_service):
        """Test loading image with invalid base64 data"""
        request = ControlNetRequest(
            prompt="test",
            image_base64="invalid_base64_data"
        )
        
        loaded_image = controlnet_service._load_image_from_request(request)
        
        assert loaded_image is None


if __name__ == "__main__":
    pytest.main([__file__])