"""
Tests for ControlNet detection system
Tests control type detection accuracy and control map generation
"""

import pytest
import numpy as np
from PIL import Image
import tempfile
import os
from unittest.mock import Mock, patch

from src.controlnet_service import (
    ControlNetService, ControlNetType, ControlNetRequest,
    ControlNetDetectionResult, ControlMapResult
)


class TestControlNetDetection:
    """Test ControlNet detection system"""
    
    @pytest.fixture
    def controlnet_service(self):
        """Create ControlNet service for testing"""
        return ControlNetService(device="cpu")  # Use CPU for testing
    
    @pytest.fixture
    def sample_edge_image(self):
        """Create a sample image with strong edges for Canny detection"""
        # Create a simple geometric image with clear edges
        image_array = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Add rectangles with clear edges
        image_array[50:100, 50:200] = [255, 255, 255]  # White rectangle
        image_array[150:200, 100:150] = [128, 128, 128]  # Gray square
        
        return Image.fromarray(image_array)
    
    @pytest.fixture
    def sample_depth_image(self):
        """Create a sample image suitable for depth estimation"""
        # Create an image with gradient that suggests depth
        image_array = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Create a gradient from dark to light (suggests depth)
        for i in range(256):
            intensity = int(i * 255 / 256)
            image_array[:, i] = [intensity, intensity, intensity]
        
        return Image.fromarray(image_array)
    
    @pytest.fixture
    def sample_pose_image(self):
        """Create a sample image with vertical structures (pose-like)"""
        image_array = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Add vertical structures that might suggest human figures
        image_array[50:200, 100:110] = [200, 150, 120]  # Skin-like color vertical line
        image_array[50:200, 150:160] = [200, 150, 120]  # Another vertical structure
        
        return Image.fromarray(image_array)
    
    @pytest.fixture
    def sample_segmentation_image(self):
        """Create a sample image with distinct regions for segmentation"""
        image_array = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Add distinct colored regions
        image_array[0:128, 0:128] = [255, 0, 0]      # Red region
        image_array[0:128, 128:256] = [0, 255, 0]    # Green region
        image_array[128:256, 0:128] = [0, 0, 255]    # Blue region
        image_array[128:256, 128:256] = [255, 255, 0] # Yellow region
        
        return Image.fromarray(image_array)
    
    def test_service_initialization(self, controlnet_service):
        """Test ControlNet service initialization"""
        assert controlnet_service.device == "cpu"
        assert controlnet_service.detection_thresholds is not None
        assert len(controlnet_service.detection_thresholds) > 0
        assert controlnet_service.error_handler is not None
    
    def test_detect_control_type_canny(self, controlnet_service, sample_edge_image):
        """Test automatic detection of Canny-suitable images"""
        result = controlnet_service.detect_control_type(sample_edge_image)
        
        assert isinstance(result, ControlNetDetectionResult)
        assert result.detected_type in [ControlNetType.CANNY, ControlNetType.SEGMENTATION]  # Both are reasonable
        assert result.confidence >= 0.0
        assert result.processing_time > 0.0
        assert ControlNetType.CANNY in result.all_scores
        assert result.all_scores[ControlNetType.CANNY] > 0.0
    
    def test_detect_control_type_depth(self, controlnet_service, sample_depth_image):
        """Test automatic detection of depth-suitable images"""
        result = controlnet_service.detect_control_type(sample_depth_image)
        
        assert isinstance(result, ControlNetDetectionResult)
        assert result.confidence >= 0.0
        assert result.processing_time > 0.0
        assert ControlNetType.DEPTH in result.all_scores
        # Depth score should be reasonable for gradient image
        assert result.all_scores[ControlNetType.DEPTH] > 0.0
    
    def test_detect_control_type_pose(self, controlnet_service, sample_pose_image):
        """Test automatic detection of pose-suitable images"""
        result = controlnet_service.detect_control_type(sample_pose_image)
        
        assert isinstance(result, ControlNetDetectionResult)
        assert result.confidence >= 0.0
        assert result.processing_time > 0.0
        assert ControlNetType.POSE in result.all_scores
    
    def test_detect_control_type_segmentation(self, controlnet_service, sample_segmentation_image):
        """Test automatic detection of segmentation-suitable images"""
        result = controlnet_service.detect_control_type(sample_segmentation_image)
        
        assert isinstance(result, ControlNetDetectionResult)
        # Segmentation should score well for distinct colored regions
        assert result.all_scores[ControlNetType.SEGMENTATION] > 0.3
        assert result.confidence >= 0.0
        assert result.processing_time > 0.0
    
    def test_detect_control_type_with_file_path(self, controlnet_service, sample_edge_image):
        """Test detection with file path input"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            sample_edge_image.save(tmp_file.name)
            
            try:
                result = controlnet_service.detect_control_type(tmp_file.name)
                assert isinstance(result, ControlNetDetectionResult)
                assert result.confidence >= 0.0
            finally:
                os.unlink(tmp_file.name)
    
    def test_detect_control_type_invalid_input(self, controlnet_service):
        """Test detection with invalid input"""
        # Test with non-existent file
        result = controlnet_service.detect_control_type("non_existent_file.jpg")
        assert isinstance(result, ControlNetDetectionResult)
        assert result.confidence == 0.0
        assert result.detected_type == ControlNetType.CANNY  # Fallback
    
    def test_detection_score_calculations(self, controlnet_service):
        """Test individual score calculation methods"""
        # Create test image
        test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test each score calculation method
        canny_score = controlnet_service._calculate_canny_score(test_array)
        assert 0.0 <= canny_score <= 1.0
        
        depth_score = controlnet_service._calculate_depth_score(test_array)
        assert 0.0 <= depth_score <= 1.0
        
        pose_score = controlnet_service._calculate_pose_score(test_array)
        assert 0.0 <= pose_score <= 1.0
        
        normal_score = controlnet_service._calculate_normal_score(test_array)
        assert 0.0 <= normal_score <= 1.0
        
        segmentation_score = controlnet_service._calculate_segmentation_score(test_array)
        assert 0.0 <= segmentation_score <= 1.0
    
    def test_detection_consistency(self, controlnet_service, sample_edge_image):
        """Test that detection results are consistent across multiple runs"""
        results = []
        for _ in range(3):
            result = controlnet_service.detect_control_type(sample_edge_image)
            results.append(result)
        
        # All results should have the same detected type
        detected_types = [r.detected_type for r in results]
        assert len(set(detected_types)) == 1, "Detection should be consistent"
        
        # Confidence scores should be similar (within 10%)
        confidences = [r.confidence for r in results]
        confidence_range = max(confidences) - min(confidences)
        assert confidence_range < 0.1, "Confidence should be consistent"
    
    def test_all_control_types_have_scores(self, controlnet_service, sample_edge_image):
        """Test that all control types get scores in detection"""
        result = controlnet_service.detect_control_type(sample_edge_image)
        
        expected_types = [
            ControlNetType.CANNY,
            ControlNetType.DEPTH,
            ControlNetType.POSE,
            ControlNetType.NORMAL,
            ControlNetType.SEGMENTATION
        ]
        
        for control_type in expected_types:
            assert control_type in result.all_scores
            assert isinstance(result.all_scores[control_type], float)
            assert 0.0 <= result.all_scores[control_type] <= 1.0


class TestControlMapGeneration:
    """Test control map generation functionality"""
    
    @pytest.fixture
    def controlnet_service(self):
        """Create ControlNet service for testing"""
        return ControlNetService(device="cpu")
    
    @pytest.fixture
    def test_image(self):
        """Create a test image for control map generation"""
        # Create a simple test image with some features
        image_array = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Add some geometric shapes
        image_array[30:70, 30:70] = [255, 255, 255]  # White square
        image_array[50:90, 80:120] = [128, 128, 128]  # Gray rectangle
        
        return Image.fromarray(image_array)
    
    def test_generate_canny_map(self, controlnet_service, test_image):
        """Test Canny edge map generation"""
        result = controlnet_service.generate_control_map(test_image, ControlNetType.CANNY)
        
        assert isinstance(result, ControlMapResult)
        assert result.control_type == ControlNetType.CANNY
        assert isinstance(result.control_image, Image.Image)
        assert result.confidence >= 0.0
        assert result.processing_time > 0.0
        assert "edge_density" in result.metadata
        
        # Check that the control image has the right dimensions
        assert result.control_image.size == test_image.size
    
    def test_generate_depth_map(self, controlnet_service, test_image):
        """Test depth map generation"""
        result = controlnet_service.generate_control_map(test_image, ControlNetType.DEPTH)
        
        assert isinstance(result, ControlMapResult)
        assert result.control_type == ControlNetType.DEPTH
        assert isinstance(result.control_image, Image.Image)
        assert result.confidence >= 0.0
        assert result.processing_time > 0.0
        assert "method" in result.metadata
        assert result.metadata["method"] == "gradient_based"
    
    def test_generate_pose_map(self, controlnet_service, test_image):
        """Test pose map generation"""
        result = controlnet_service.generate_control_map(test_image, ControlNetType.POSE)
        
        assert isinstance(result, ControlMapResult)
        assert result.control_type == ControlNetType.POSE
        assert isinstance(result.control_image, Image.Image)
        assert result.confidence >= 0.0
        assert result.processing_time > 0.0
        assert "contours_found" in result.metadata
    
    def test_generate_normal_map(self, controlnet_service, test_image):
        """Test normal map generation"""
        result = controlnet_service.generate_control_map(test_image, ControlNetType.NORMAL)
        
        assert isinstance(result, ControlMapResult)
        assert result.control_type == ControlNetType.NORMAL
        assert isinstance(result.control_image, Image.Image)
        assert result.confidence >= 0.0
        assert result.processing_time > 0.0
        assert "method" in result.metadata
        assert result.metadata["method"] == "gradient_based_normal"
    
    def test_generate_segmentation_map(self, controlnet_service, test_image):
        """Test segmentation map generation"""
        result = controlnet_service.generate_control_map(test_image, ControlNetType.SEGMENTATION)
        
        assert isinstance(result, ControlMapResult)
        assert result.control_type == ControlNetType.SEGMENTATION
        assert isinstance(result.control_image, Image.Image)
        assert result.confidence >= 0.0
        assert result.processing_time > 0.0
        assert "n_clusters" in result.metadata
    
    def test_generate_map_with_parameters(self, controlnet_service, test_image):
        """Test control map generation with custom parameters"""
        # Test Canny with custom thresholds
        result = controlnet_service.generate_control_map(
            test_image, 
            ControlNetType.CANNY,
            low_threshold=30,
            high_threshold=100,
            blur_kernel=5
        )
        
        assert result.metadata["low_threshold"] == 30
        assert result.metadata["high_threshold"] == 100
        assert result.metadata["blur_kernel"] == 5
    
    def test_generate_map_with_file_path(self, controlnet_service, test_image):
        """Test control map generation with file path input"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            test_image.save(tmp_file.name)
            
            try:
                result = controlnet_service.generate_control_map(tmp_file.name, ControlNetType.CANNY)
                assert isinstance(result, ControlMapResult)
                assert result.control_type == ControlNetType.CANNY
            finally:
                os.unlink(tmp_file.name)
    
    def test_generate_map_invalid_type(self, controlnet_service, test_image):
        """Test control map generation with invalid control type"""
        with pytest.raises(ValueError, match="Unsupported control type"):
            # This should raise an error since AUTO is not a valid generation type
            controlnet_service.generate_control_map(test_image, ControlNetType.AUTO)
    
    def test_generate_map_invalid_image(self, controlnet_service):
        """Test control map generation with invalid image"""
        with pytest.raises(ValueError, match="Failed to load input image"):
            controlnet_service.generate_control_map("non_existent_file.jpg", ControlNetType.CANNY)
    
    def test_confidence_calculation(self, controlnet_service, test_image):
        """Test confidence calculation for different control types"""
        # Generate maps and check confidence calculations
        canny_result = controlnet_service.generate_control_map(test_image, ControlNetType.CANNY)
        depth_result = controlnet_service.generate_control_map(test_image, ControlNetType.DEPTH)
        
        # Confidence should be calculated differently for each type
        assert 0.0 <= canny_result.confidence <= 1.0
        assert 0.0 <= depth_result.confidence <= 1.0
        
        # Test confidence calculation method directly
        confidence = controlnet_service._calculate_control_map_confidence(
            canny_result.control_image, 
            ControlNetType.CANNY
        )
        assert 0.0 <= confidence <= 1.0


class TestControlNetRequest:
    """Test ControlNetRequest model and validation"""
    
    def test_controlnet_request_creation(self):
        """Test ControlNetRequest creation with default values"""
        request = ControlNetRequest(prompt="test prompt")
        
        assert request.prompt == "test prompt"
        assert request.control_type == ControlNetType.AUTO
        assert request.controlnet_conditioning_scale == 1.0
        assert request.num_inference_steps == 20
        assert request.guidance_scale == 7.5
        assert request.width == 768
        assert request.height == 768
    
    def test_controlnet_request_with_parameters(self):
        """Test ControlNetRequest with custom parameters"""
        request = ControlNetRequest(
            prompt="custom prompt",
            control_type=ControlNetType.CANNY,
            controlnet_conditioning_scale=0.8,
            num_inference_steps=30,
            guidance_scale=8.0,
            width=512,
            height=512,
            seed=42
        )
        
        assert request.prompt == "custom prompt"
        assert request.control_type == ControlNetType.CANNY
        assert request.controlnet_conditioning_scale == 0.8
        assert request.num_inference_steps == 30
        assert request.guidance_scale == 8.0
        assert request.width == 512
        assert request.height == 512
        assert request.seed == 42


if __name__ == "__main__":
    pytest.main([__file__])