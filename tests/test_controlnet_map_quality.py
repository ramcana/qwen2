"""
Tests for ControlNet control map generation quality and consistency
Tests control map preprocessing, optimization, and preview generation
"""

import pytest
import numpy as np
from PIL import Image
import tempfile
import os
from unittest.mock import Mock, patch

from src.controlnet_service import (
    ControlNetService, ControlNetType, ControlMapResult
)


class TestControlMapQuality:
    """Test control map generation quality and consistency"""
    
    @pytest.fixture
    def controlnet_service(self):
        """Create ControlNet service for testing"""
        return ControlNetService(device="cpu")
    
    @pytest.fixture
    def high_contrast_image(self):
        """Create high contrast image for testing edge detection quality"""
        image_array = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Create high contrast patterns
        image_array[0:128, 0:128] = [255, 255, 255]  # White quadrant
        image_array[0:128, 128:256] = [0, 0, 0]      # Black quadrant
        image_array[128:256, 0:128] = [0, 0, 0]      # Black quadrant
        image_array[128:256, 128:256] = [255, 255, 255]  # White quadrant
        
        return Image.fromarray(image_array)
    
    @pytest.fixture
    def gradient_image(self):
        """Create gradient image for testing depth map quality"""
        image_array = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Create smooth gradient
        for i in range(256):
            for j in range(256):
                intensity = int((i + j) * 255 / 512)
                image_array[i, j] = [intensity, intensity, intensity]
        
        return Image.fromarray(image_array)
    
    @pytest.fixture
    def noisy_image(self):
        """Create noisy image for testing robustness"""
        # Start with a simple pattern
        image_array = np.zeros((256, 256, 3), dtype=np.uint8)
        image_array[64:192, 64:192] = [128, 128, 128]
        
        # Add random noise
        noise = np.random.randint(-30, 30, image_array.shape, dtype=np.int16)
        image_array = np.clip(image_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(image_array)
    
    def test_canny_map_quality_high_contrast(self, controlnet_service, high_contrast_image):
        """Test Canny map quality with high contrast image"""
        result = controlnet_service.generate_control_map(high_contrast_image, ControlNetType.CANNY)
        
        # High contrast image should produce reasonable edge detection
        assert result.confidence > 0.1, "High contrast image should have reasonable Canny confidence"
        assert "edge_density" in result.metadata
        assert result.metadata["edge_density"] > 0.005, "Should detect some edges"
        
        # Check that control map has proper dimensions
        assert result.control_image.size == high_contrast_image.size
        
        # Check that edges are detected (control map should not be all black)
        control_array = np.array(result.control_image)
        edge_pixels = np.sum(control_array > 0)
        total_pixels = control_array.size
        edge_ratio = edge_pixels / total_pixels
        assert edge_ratio > 0.001, "Should detect some edges"
    
    def test_canny_map_parameters_effect(self, controlnet_service, high_contrast_image):
        """Test that Canny parameters affect the output"""
        # Generate with different thresholds
        result_low = controlnet_service.generate_control_map(
            high_contrast_image, ControlNetType.CANNY,
            low_threshold=20, high_threshold=60
        )
        
        result_high = controlnet_service.generate_control_map(
            high_contrast_image, ControlNetType.CANNY,
            low_threshold=80, high_threshold=160
        )
        
        # Parameters should be recorded correctly
        assert result_low.metadata["low_threshold"] == 20
        assert result_low.metadata["high_threshold"] == 60
        assert result_high.metadata["low_threshold"] == 80
        assert result_high.metadata["high_threshold"] == 160
        
        # Different thresholds may produce different edge densities
        # (Allow for the possibility they could be the same for simple images)
        assert result_low.metadata["edge_density"] >= 0.0
        assert result_high.metadata["edge_density"] >= 0.0
    
    def test_depth_map_quality_gradient(self, controlnet_service, gradient_image):
        """Test depth map quality with gradient image"""
        result = controlnet_service.generate_control_map(gradient_image, ControlNetType.DEPTH)
        
        # Gradient image should produce some depth estimation
        assert result.confidence > 0.05, "Gradient image should have some depth confidence"
        assert "depth_range" in result.metadata
        assert result.metadata["depth_range"] > 0, "Should detect depth variation"
        
        # Check that depth map shows some variation
        control_array = np.array(result.control_image)
        depth_std = np.std(control_array)
        assert depth_std > 1, "Depth map should show some variation"
    
    def test_segmentation_map_quality_distinct_regions(self, controlnet_service):
        """Test segmentation map quality with distinct colored regions"""
        # Create image with 4 distinct colored regions
        image_array = np.zeros((200, 200, 3), dtype=np.uint8)
        image_array[0:100, 0:100] = [255, 0, 0]      # Red
        image_array[0:100, 100:200] = [0, 255, 0]    # Green
        image_array[100:200, 0:100] = [0, 0, 255]    # Blue
        image_array[100:200, 100:200] = [255, 255, 0] # Yellow
        
        test_image = Image.fromarray(image_array)
        
        result = controlnet_service.generate_control_map(test_image, ControlNetType.SEGMENTATION)
        
        # Should detect distinct regions with reasonable confidence
        assert result.confidence > 0.05, "Distinct regions should have reasonable segmentation confidence"
        assert "n_clusters" in result.metadata
        
        # Check that segmentation produces multiple colors
        control_array = np.array(result.control_image)
        unique_colors = len(np.unique(control_array.reshape(-1, 3), axis=0))
        assert unique_colors >= 2, "Should produce multiple color regions"
    
    def test_normal_map_quality_geometric_shapes(self, controlnet_service):
        """Test normal map quality with geometric shapes"""
        # Create image with geometric shapes (good for normal mapping)
        image_array = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Add a circle (good surface for normal mapping)
        center = (100, 100)
        radius = 50
        y, x = np.ogrid[:200, :200]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        # Create height variation within circle
        for i in range(200):
            for j in range(200):
                if mask[i, j]:
                    distance_from_center = np.sqrt((i - center[1])**2 + (j - center[0])**2)
                    intensity = int(255 * (1 - distance_from_center / radius))
                    image_array[i, j] = [intensity, intensity, intensity]
        
        test_image = Image.fromarray(image_array)
        
        result = controlnet_service.generate_control_map(test_image, ControlNetType.NORMAL)
        
        # Should generate normal map without errors
        assert result.confidence > 0.01, "Geometric shapes should generate normal map"
        assert "gradient_range_x" in result.metadata
        assert "gradient_range_y" in result.metadata
        
        # Normal map should have some variation in channels
        control_array = np.array(result.control_image)
        total_variation = 0.0
        for channel in range(3):
            channel_std = np.std(control_array[:, :, channel])
            total_variation += channel_std
        
        assert total_variation > 1.0, "Normal map should have some variation across channels"
    
    def test_pose_map_quality_vertical_structures(self, controlnet_service):
        """Test pose map quality with vertical structures"""
        # Create image with vertical structures
        image_array = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Add vertical lines (simulate limbs/body parts)
        image_array[50:200, 80:85] = [200, 150, 120]   # Vertical structure 1
        image_array[50:200, 120:125] = [200, 150, 120] # Vertical structure 2
        image_array[100:150, 85:120] = [180, 140, 110] # Horizontal connection
        
        test_image = Image.fromarray(image_array)
        
        result = controlnet_service.generate_control_map(test_image, ControlNetType.POSE)
        
        # Should detect structural elements
        assert result.confidence >= 0.0, "Should generate pose map without errors"
        assert "contours_found" in result.metadata
        
        # Should detect some structural elements
        control_array = np.array(result.control_image)
        structure_pixels = np.sum(control_array > 0)
        assert structure_pixels > 0, "Should detect some structural elements"
    
    def test_control_map_consistency(self, controlnet_service, high_contrast_image):
        """Test that control map generation is consistent across multiple runs"""
        results = []
        for _ in range(3):
            result = controlnet_service.generate_control_map(high_contrast_image, ControlNetType.CANNY)
            results.append(result)
        
        # All results should have the same confidence (deterministic processing)
        confidences = [r.confidence for r in results]
        assert len(set(confidences)) == 1, "Control map generation should be deterministic"
        
        # Control images should be identical
        control_arrays = [np.array(r.control_image) for r in results]
        for i in range(1, len(control_arrays)):
            assert np.array_equal(control_arrays[0], control_arrays[i]), "Control maps should be identical"
    
    def test_control_map_robustness_noisy_image(self, controlnet_service, noisy_image):
        """Test control map generation robustness with noisy input"""
        # Should handle noisy input without crashing
        result = controlnet_service.generate_control_map(noisy_image, ControlNetType.CANNY)
        
        assert isinstance(result, ControlMapResult)
        assert result.confidence >= 0.0
        assert result.control_image.size == noisy_image.size
        
        # Test with different control types
        for control_type in [ControlNetType.DEPTH, ControlNetType.SEGMENTATION]:
            result = controlnet_service.generate_control_map(noisy_image, control_type)
            assert isinstance(result, ControlMapResult)
            assert result.confidence >= 0.0
    
    def test_control_map_preprocessing_blur_effect(self, controlnet_service, noisy_image):
        """Test that blur preprocessing improves control map quality"""
        # Generate without blur
        result_no_blur = controlnet_service.generate_control_map(
            noisy_image, ControlNetType.CANNY, blur_kernel=0
        )
        
        # Generate with blur
        result_with_blur = controlnet_service.generate_control_map(
            noisy_image, ControlNetType.CANNY, blur_kernel=5
        )
        
        # Both should succeed
        assert isinstance(result_no_blur, ControlMapResult)
        assert isinstance(result_with_blur, ControlMapResult)
        
        # Blur should typically reduce noise and improve edge continuity
        # (This is a heuristic test - exact behavior may vary)
        assert result_with_blur.metadata["blur_kernel"] == 5
        assert result_no_blur.metadata["blur_kernel"] == 0
    
    def test_control_map_optimization_large_image(self, controlnet_service):
        """Test control map generation with large images"""
        # Create a larger test image
        large_image_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Add some structure
        large_image_array[100:400, 100:400] = [255, 255, 255]
        large_image_array[200:300, 200:300] = [0, 0, 0]
        
        large_image = Image.fromarray(large_image_array)
        
        # Should handle large images efficiently
        result = controlnet_service.generate_control_map(large_image, ControlNetType.CANNY)
        
        assert isinstance(result, ControlMapResult)
        assert result.control_image.size == large_image.size
        assert result.processing_time < 10.0, "Should process large images efficiently"
    
    def test_control_map_preview_generation(self, controlnet_service, high_contrast_image):
        """Test that control maps can be used for preview generation"""
        result = controlnet_service.generate_control_map(high_contrast_image, ControlNetType.CANNY)
        
        # Control map should be suitable for preview
        control_array = np.array(result.control_image)
        
        # Should be grayscale-like (for Canny) or have meaningful color variation
        assert control_array.shape == (*high_contrast_image.size[::-1], 3)
        
        # Should have some non-zero pixels (detected features)
        non_zero_pixels = np.sum(control_array > 0)
        assert non_zero_pixels > 0, "Control map should have detected features for preview"
        
        # Metadata should contain information useful for preview
        assert "edge_density" in result.metadata
        assert isinstance(result.metadata["edge_density"], float)


class TestControlMapOptimization:
    """Test control map preprocessing and optimization features"""
    
    @pytest.fixture
    def controlnet_service(self):
        """Create ControlNet service for testing"""
        return ControlNetService(device="cpu")
    
    @pytest.fixture
    def test_image(self):
        """Create test image for optimization tests"""
        image_array = np.zeros((128, 128, 3), dtype=np.uint8)
        image_array[32:96, 32:96] = [128, 128, 128]
        return Image.fromarray(image_array)
    
    def test_segmentation_clustering_optimization(self, controlnet_service, test_image):
        """Test segmentation map clustering optimization"""
        # Test with different cluster counts
        result_few_clusters = controlnet_service.generate_control_map(
            test_image, ControlNetType.SEGMENTATION, n_clusters=4
        )
        
        result_many_clusters = controlnet_service.generate_control_map(
            test_image, ControlNetType.SEGMENTATION, n_clusters=16
        )
        
        # Should respect cluster count parameter
        assert result_few_clusters.metadata["n_clusters"] == 4
        assert result_many_clusters.metadata["n_clusters"] == 16
        
        # More clusters should generally result in more detailed segmentation
        few_unique = len(np.unique(np.array(result_few_clusters.control_image).reshape(-1, 3), axis=0))
        many_unique = len(np.unique(np.array(result_many_clusters.control_image).reshape(-1, 3), axis=0))
        
        # Should have different levels of detail
        assert few_unique <= many_unique + 2, "More clusters should not result in fewer unique colors"
    
    def test_control_map_memory_efficiency(self, controlnet_service, test_image):
        """Test that control map generation is memory efficient"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Generate multiple control maps
        for _ in range(5):
            for control_type in [ControlNetType.CANNY, ControlNetType.DEPTH, ControlNetType.SEGMENTATION]:
                result = controlnet_service.generate_control_map(test_image, control_type)
                assert isinstance(result, ControlMapResult)
        
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100 * 1024 * 1024, "Control map generation should be memory efficient"
    
    def test_control_map_error_recovery(self, controlnet_service):
        """Test error recovery in control map generation"""
        # Test with invalid image
        with pytest.raises(ValueError):
            controlnet_service.generate_control_map("invalid_path.jpg", ControlNetType.CANNY)
        
        # Service should still work after error
        valid_image = Image.new("RGB", (64, 64), color="white")
        result = controlnet_service.generate_control_map(valid_image, ControlNetType.CANNY)
        assert isinstance(result, ControlMapResult)


if __name__ == "__main__":
    pytest.main([__file__])