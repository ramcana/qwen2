"""
Tests for DiffSynth base image editing functionality
Tests the core image editing operations, data models, and utilities
"""

import pytest
import os
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

# Import modules to test
from src.diffsynth_models import (
    ImageEditRequest, ImageEditResponse, EditOperation, ProcessingMetrics,
    encode_image_to_base64, decode_base64_to_image, validate_image_format
)
from src.diffsynth_utils import (
    ImagePreprocessor, ImagePostprocessor, TiledProcessor,
    estimate_processing_time, get_optimal_dimensions
)
from src.diffsynth_service import DiffSynthService, DiffSynthConfig


class TestDiffSynthModels:
    """Test DiffSynth data models"""
    
    def test_image_edit_request_validation(self):
        """Test ImageEditRequest validation"""
        # Valid request with image path
        request = ImageEditRequest(
            prompt="Make this image more vibrant",
            image_path="/path/to/image.jpg"
        )
        assert request.prompt == "Make this image more vibrant"
        assert request.image_path == "/path/to/image.jpg"
        assert request.num_inference_steps == 20  # default
        
        # Valid request with base64
        request = ImageEditRequest(
            prompt="Edit this image",
            image_base64="base64encodedimage"
        )
        assert request.image_base64 == "base64encodedimage"
        
        # Invalid request - no image input
        with pytest.raises(ValueError, match="Either image_path or image_base64 must be provided"):
            ImageEditRequest(prompt="Test prompt")
    
    def test_image_edit_request_defaults(self):
        """Test default values in ImageEditRequest"""
        request = ImageEditRequest(
            prompt="Test",
            image_path="/test.jpg"
        )
        
        assert request.num_inference_steps == 20
        assert request.guidance_scale == 7.5
        assert request.strength == 0.8
        assert request.negative_prompt is None
        assert request.seed is None
    
    def test_image_edit_response_creation(self):
        """Test ImageEditResponse creation"""
        response = ImageEditResponse(
            success=True,
            message="Image edited successfully",
            image_path="/output/edited.jpg",
            operation=EditOperation.EDIT,
            processing_time=3.45
        )
        
        assert response.success is True
        assert response.message == "Image edited successfully"
        assert response.operation == EditOperation.EDIT
        assert response.processing_time == 3.45
    
    def test_processing_metrics(self):
        """Test ProcessingMetrics model"""
        metrics = ProcessingMetrics(
            operation_type="edit",
            processing_time=2.5,
            gpu_memory_used=1.8,
            input_resolution=(768, 768),
            output_resolution=(768, 768)
        )
        
        assert metrics.operation_type == "edit"
        assert metrics.processing_time == 2.5
        assert metrics.gpu_memory_used == 1.8
        assert metrics.input_resolution == (768, 768)


class TestImageUtilities:
    """Test image utility functions"""
    
    def setup_method(self):
        """Setup test images"""
        self.test_image = Image.new('RGB', (256, 256), color='red')
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test_image.jpg")
        self.test_image.save(self.test_image_path)
    
    def test_encode_decode_base64(self):
        """Test base64 encoding/decoding"""
        # Test with PIL Image
        base64_str = encode_image_to_base64(self.test_image)
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
        
        # Decode back
        decoded_image = decode_base64_to_image(base64_str)
        assert isinstance(decoded_image, Image.Image)
        assert decoded_image.size == self.test_image.size
        
        # Test with file path
        base64_str = encode_image_to_base64(self.test_image_path)
        decoded_image = decode_base64_to_image(base64_str)
        assert decoded_image.size == self.test_image.size
    
    def test_validate_image_format(self):
        """Test image format validation"""
        # Valid image
        valid_image = Image.new('RGB', (512, 512), color='blue')
        assert validate_image_format(valid_image) is True
        
        # Too small image
        small_image = Image.new('RGB', (32, 32), color='blue')
        assert validate_image_format(small_image) is False
        
        # Too large image
        large_image = Image.new('RGB', (5000, 5000), color='blue')
        assert validate_image_format(large_image) is False


class TestImagePreprocessor:
    """Test ImagePreprocessor class"""
    
    def setup_method(self):
        """Setup preprocessor and test images"""
        self.preprocessor = ImagePreprocessor()
        self.test_image = Image.new('RGB', (512, 512), color='green')
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test.jpg")
        self.test_image.save(self.test_image_path)
    
    def test_load_and_validate_image_from_path(self):
        """Test loading image from file path"""
        loaded_image = self.preprocessor.load_and_validate_image(self.test_image_path)
        assert loaded_image is not None
        assert loaded_image.mode == 'RGB'
        assert loaded_image.size == (512, 512)
    
    def test_load_and_validate_image_from_pil(self):
        """Test loading image from PIL Image"""
        loaded_image = self.preprocessor.load_and_validate_image(self.test_image)
        assert loaded_image is not None
        assert loaded_image.mode == 'RGB'
        assert loaded_image.size == (512, 512)
    
    def test_load_invalid_path(self):
        """Test loading from invalid path"""
        loaded_image = self.preprocessor.load_and_validate_image("/nonexistent/path.jpg")
        assert loaded_image is None
    
    def test_prepare_image_for_editing(self):
        """Test image preparation for editing"""
        prepared_image = self.preprocessor.prepare_image_for_editing(
            self.test_image,
            target_width=768,
            target_height=768
        )
        
        assert prepared_image is not None
        assert prepared_image.size == (768, 768)
        
        # Test with aspect ratio maintenance
        prepared_image = self.preprocessor.prepare_image_for_editing(
            self.test_image,
            target_width=1024,
            maintain_aspect_ratio=True
        )
        
        assert prepared_image is not None
        # Should maintain square aspect ratio
        assert prepared_image.size[0] == prepared_image.size[1]
    
    def test_prepare_mask(self):
        """Test mask preparation"""
        # Create test mask
        mask_image = Image.new('L', (256, 256), color=128)
        
        prepared_mask = self.preprocessor.prepare_mask(mask_image, blur_radius=2)
        assert prepared_mask is not None
        assert prepared_mask.mode == 'L'
        
        # Check that mask is binary (0 or 255)
        mask_array = np.array(prepared_mask)
        unique_values = np.unique(mask_array)
        assert len(unique_values) <= 2
        assert all(val in [0, 255] for val in unique_values)


class TestImagePostprocessor:
    """Test ImagePostprocessor class"""
    
    def setup_method(self):
        """Setup postprocessor and test images"""
        self.postprocessor = ImagePostprocessor()
        self.test_image = Image.new('RGB', (512, 512), color='blue')
        self.temp_dir = tempfile.mkdtemp()
    
    def test_postprocess_edited_image(self):
        """Test image postprocessing"""
        processed_image = self.postprocessor.postprocess_edited_image(
            self.test_image,
            enhance_quality=True
        )
        
        assert processed_image is not None
        assert processed_image.size == self.test_image.size
        assert processed_image.mode == 'RGB'
    
    def test_save_image(self):
        """Test image saving"""
        output_path = os.path.join(self.temp_dir, "output.jpg")
        
        success = self.postprocessor.save_image(
            self.test_image,
            output_path,
            quality=90
        )
        
        assert success is True
        assert os.path.exists(output_path)
        
        # Verify saved image
        saved_image = Image.open(output_path)
        assert saved_image.size == self.test_image.size
    
    def test_save_image_with_metadata(self):
        """Test saving image with metadata"""
        output_path = os.path.join(self.temp_dir, "output_with_metadata.png")
        metadata = {
            "prompt": "Test prompt",
            "operation": "edit"
        }
        
        success = self.postprocessor.save_image(
            self.test_image,
            output_path,
            format="PNG",
            metadata=metadata
        )
        
        assert success is True
        assert os.path.exists(output_path)


class TestTiledProcessor:
    """Test TiledProcessor class"""
    
    def setup_method(self):
        """Setup tiled processor"""
        self.tiled_processor = TiledProcessor(tile_size=256, overlap=32)
        self.large_image = Image.new('RGB', (1024, 1024), color='red')
        self.small_image = Image.new('RGB', (512, 512), color='blue')
    
    def test_should_use_tiled_processing(self):
        """Test tiled processing decision"""
        # Large image should use tiled processing
        should_tile = self.tiled_processor.should_use_tiled_processing(self.large_image)
        assert should_tile is False  # 1024x1024 is not large enough by default
        
        # Very large image should use tiled processing
        very_large_image = Image.new('RGB', (2048, 2048), color='red')
        should_tile = self.tiled_processor.should_use_tiled_processing(very_large_image)
        assert should_tile is True
        
        # Small image should not use tiled processing
        should_tile = self.tiled_processor.should_use_tiled_processing(self.small_image)
        assert should_tile is False
    
    def test_calculate_tiles(self):
        """Test tile calculation"""
        tiles = self.tiled_processor.calculate_tiles(self.large_image)
        
        assert len(tiles) > 0
        assert all(len(tile) == 4 for tile in tiles)  # Each tile has 4 coordinates
        
        # Check that tiles cover the image
        for x1, y1, x2, y2 in tiles:
            assert 0 <= x1 < x2 <= 1024
            assert 0 <= y1 < y2 <= 1024
    
    def test_merge_tiles(self):
        """Test tile merging"""
        # Create simple tiles
        tile_coords = [(0, 0, 256, 256), (256, 0, 512, 256)]
        tiles = [
            Image.new('RGB', (256, 256), color='red'),
            Image.new('RGB', (256, 256), color='blue')
        ]
        
        merged_image = self.tiled_processor.merge_tiles(
            tiles, tile_coords, (512, 256)
        )
        
        assert merged_image is not None
        assert merged_image.size == (512, 256)


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_estimate_processing_time(self):
        """Test processing time estimation"""
        test_image = Image.new('RGB', (512, 512), color='red')
        
        # Test different operations
        edit_time = estimate_processing_time(test_image, "edit")
        inpaint_time = estimate_processing_time(test_image, "inpaint")
        style_time = estimate_processing_time(test_image, "style_transfer")
        
        assert edit_time > 0
        assert inpaint_time > edit_time  # Inpainting should take longer
        assert style_time > inpaint_time  # Style transfer should take longest
    
    def test_get_optimal_dimensions(self):
        """Test optimal dimension calculation"""
        # Test within limits
        width, height = get_optimal_dimensions(512, 512, max_dimension=1024)
        assert width == 512
        assert height == 512
        
        # Test scaling down
        width, height = get_optimal_dimensions(2048, 1024, max_dimension=1024)
        assert width <= 1024
        assert height <= 1024
        
        # Test multiples of 8
        width, height = get_optimal_dimensions(1000, 1000, max_dimension=1024)
        assert width % 8 == 0
        assert height % 8 == 0


class TestDiffSynthService:
    """Test DiffSynthService class"""
    
    def setup_method(self):
        """Setup service for testing"""
        self.config = DiffSynthConfig(
            model_name="test-model",
            device="cpu",  # Use CPU for testing
            enable_vram_management=False
        )
        self.service = DiffSynthService(self.config)
        
        # Create test image
        self.test_image = Image.new('RGB', (512, 512), color='red')
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test.jpg")
        self.test_image.save(self.test_image_path)
    
    def test_service_initialization(self):
        """Test service initialization"""
        assert self.service.config.model_name == "test-model"
        assert self.service.config.device == "cpu"
        assert self.service.status.value == "not_initialized"
    
    @patch('src.diffsynth_service.DiffSynthService._create_optimized_pipeline')
    def test_service_initialize_success(self, mock_create_pipeline):
        """Test successful service initialization"""
        # Mock pipeline creation
        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline
        
        # Mock DiffSynth imports
        with patch.dict('sys.modules', {
            'diffsynth.pipelines.qwen_image': Mock(),
            'diffsynth': Mock()
        }):
            success = self.service.initialize()
            
        assert success is True
        assert self.service.status.value == "ready"
        assert self.service.pipeline is not None
    
    def test_load_image_from_request(self):
        """Test loading image from request"""
        # Test with image path
        request = ImageEditRequest(
            prompt="Test",
            image_path=self.test_image_path
        )
        
        loaded_image = self.service._load_image_from_request(request)
        assert loaded_image is not None
        assert loaded_image.size == (512, 512)
        
        # Test with base64
        base64_str = encode_image_to_base64(self.test_image)
        request = ImageEditRequest(
            prompt="Test",
            image_base64=base64_str
        )
        
        loaded_image = self.service._load_image_from_request(request)
        assert loaded_image is not None
        assert loaded_image.size == (512, 512)
    
    def test_save_edited_image(self):
        """Test saving edited image"""
        request = ImageEditRequest(
            prompt="Test edit",
            image_path=self.test_image_path
        )
        
        # Ensure output directory exists
        os.makedirs("generated_images", exist_ok=True)
        
        output_path = self.service._save_edited_image(self.test_image, request)
        
        assert output_path != ""
        assert "diffsynth_edit_" in output_path
        assert output_path.endswith(".jpg")
    
    @patch('src.diffsynth_service.DiffSynthService.initialize')
    def test_edit_image_service_not_ready(self, mock_initialize):
        """Test edit_image when service not ready"""
        mock_initialize.return_value = False
        
        request = ImageEditRequest(
            prompt="Test",
            image_path=self.test_image_path
        )
        
        response = self.service.edit_image(request)
        
        assert response.success is False
        assert "service not available" in response.message.lower()
    
    def test_fallback_edit_image_response(self):
        """Test fallback response"""
        request = ImageEditRequest(
            prompt="Test",
            image_path=self.test_image_path
        )
        
        response = self.service._fallback_edit_image_response(request)
        
        assert response.success is False
        assert "failed" in response.message.lower()
        assert response.suggested_fixes is not None
        assert len(response.suggested_fixes) > 0


class TestIntegrationWorkflow:
    """Test complete image editing workflow"""
    
    def setup_method(self):
        """Setup for integration tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image = Image.new('RGB', (512, 512), color='green')
        self.test_image_path = os.path.join(self.temp_dir, "input.jpg")
        self.test_image.save(self.test_image_path)
        
        # Ensure output directory exists
        os.makedirs("generated_images", exist_ok=True)
    
    def test_complete_editing_workflow_with_mocks(self):
        """Test complete editing workflow with mocked pipeline"""
        # Create service
        config = DiffSynthConfig(device="cpu", enable_vram_management=False)
        service = DiffSynthService(config)
        
        # Mock the pipeline
        mock_pipeline = Mock()
        mock_edited_image = Image.new('RGB', (512, 512), color='blue')
        mock_pipeline.return_value = mock_edited_image
        
        service.pipeline = mock_pipeline
        service.status = service.status.READY
        
        # Create request
        request = ImageEditRequest(
            prompt="Make this image blue",
            image_path=self.test_image_path,
            num_inference_steps=10,
            guidance_scale=7.0,
            strength=0.8
        )
        
        # Process request
        response = service.edit_image(request)
        
        # Verify response
        assert response.success is True
        assert response.operation == EditOperation.EDIT
        assert response.processing_time > 0
        assert response.image_path != ""
        assert "diffsynth_edit_" in response.image_path
        
        # Verify pipeline was called
        mock_pipeline.assert_called_once()
        call_args = mock_pipeline.call_args
        assert call_args[1]['prompt'] == "Make this image blue"
        assert call_args[1]['num_inference_steps'] == 10
        assert call_args[1]['guidance_scale'] == 7.0
    
    def test_preprocessing_postprocessing_chain(self):
        """Test the preprocessing and postprocessing chain"""
        preprocessor = ImagePreprocessor()
        postprocessor = ImagePostprocessor()
        
        # Load and preprocess
        loaded_image = preprocessor.load_and_validate_image(self.test_image_path)
        assert loaded_image is not None
        
        prepared_image = preprocessor.prepare_image_for_editing(
            loaded_image,
            target_width=768,
            target_height=768
        )
        assert prepared_image.size == (768, 768)
        
        # Simulate editing (just copy the image)
        edited_image = prepared_image.copy()
        
        # Postprocess
        final_image = postprocessor.postprocess_edited_image(
            edited_image,
            original_image=loaded_image
        )
        assert final_image is not None
        assert final_image.size == edited_image.size
        
        # Save
        output_path = os.path.join(self.temp_dir, "final_output.jpg")
        success = postprocessor.save_image(final_image, output_path)
        assert success is True
        assert os.path.exists(output_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])