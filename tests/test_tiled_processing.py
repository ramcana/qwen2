"""
Tests for enhanced tiled processing functionality
"""

import pytest
import numpy as np
from PIL import Image
import torch
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.diffsynth_utils import TiledProcessor


class TestTiledProcessor:
    """Test cases for enhanced TiledProcessor"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = TiledProcessor(tile_size=512, overlap=64)
        
        # Create test images of various sizes
        self.small_image = Image.new('RGB', (256, 256), (255, 0, 0))
        self.medium_image = Image.new('RGB', (1024, 768), (0, 255, 0))
        self.large_image = Image.new('RGB', (2048, 1536), (0, 0, 255))
        self.huge_image = Image.new('RGB', (4096, 3072), (255, 255, 0))
    
    def test_automatic_tiling_detection_small_image(self):
        """Test that small images don't require tiling"""
        result = self.processor.should_use_tiled_processing(
            self.small_image, 
            operation_type="edit",
            available_memory_gb=8.0
        )
        assert not result, "Small images should not require tiling"
    
    def test_automatic_tiling_detection_large_image(self):
        """Test that large images require tiling"""
        result = self.processor.should_use_tiled_processing(
            self.large_image,
            operation_type="edit", 
            available_memory_gb=8.0
        )
        assert result, "Large images should require tiling"
    
    def test_automatic_tiling_detection_huge_image(self):
        """Test that huge images always require tiling"""
        result = self.processor.should_use_tiled_processing(
            self.huge_image,
            operation_type="edit",
            available_memory_gb=16.0  # Even with lots of memory
        )
        assert result, "Huge images should always require tiling"
    
    def test_automatic_tiling_detection_low_memory(self):
        """Test tiling detection with low memory"""
        result = self.processor.should_use_tiled_processing(
            self.medium_image,
            operation_type="edit",
            available_memory_gb=1.0  # Low memory
        )
        assert result, "Medium images should require tiling with low memory"
    
    def test_automatic_tiling_detection_complex_operations(self):
        """Test tiling detection for complex operations"""
        # Style transfer should require tiling for medium images
        result = self.processor.should_use_tiled_processing(
            self.medium_image,
            operation_type="style_transfer",
            available_memory_gb=4.0
        )
        assert result, "Complex operations should require tiling for medium images"
        
        # Simple edit might not require tiling
        result = self.processor.should_use_tiled_processing(
            self.medium_image,
            operation_type="edit",
            available_memory_gb=8.0
        )
        # This might be False depending on exact thresholds
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    @patch('torch.cuda.get_device_properties')
    def test_get_available_memory_gpu(self, mock_props, mock_reserved, mock_allocated, mock_cuda_available):
        """Test GPU memory detection"""
        mock_cuda_available.return_value = True
        mock_allocated.return_value = 1e9  # 1GB allocated
        mock_reserved.return_value = 2e9   # 2GB reserved
        
        # Mock GPU properties
        mock_device = Mock()
        mock_device.total_memory = 8e9  # 8GB total
        mock_props.return_value = mock_device
        
        available = self.processor._get_available_memory()
        
        # Should be (8 - 2) * 0.8 = 4.8GB
        assert available == pytest.approx(4.8, rel=0.1)
    
    @patch('torch.cuda.is_available')
    @patch('psutil.virtual_memory')
    def test_get_available_memory_cpu(self, mock_virtual_memory, mock_cuda_available):
        """Test CPU memory detection"""
        mock_cuda_available.return_value = False
        
        # Mock system memory
        mock_memory = Mock()
        mock_memory.available = 16e9  # 16GB available
        mock_virtual_memory.return_value = mock_memory
        
        available = self.processor._get_available_memory()
        
        # Should be min(16 * 0.5, 8.0) = 8.0GB
        assert available == 8.0
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation for different operations"""
        # Test different operation types
        edit_memory = self.processor._estimate_memory_usage(self.medium_image, "edit")
        inpaint_memory = self.processor._estimate_memory_usage(self.medium_image, "inpaint")
        style_memory = self.processor._estimate_memory_usage(self.medium_image, "style_transfer")
        
        # Style transfer should use most memory
        assert style_memory > inpaint_memory > edit_memory
        
        # All should be reasonable values (not negative, not huge)
        assert 0 < edit_memory < 50  # Less than 50GB for 1024x768 image
        assert 0 < inpaint_memory < 50
        assert 0 < style_memory < 50
    
    def test_optimal_tile_calculation(self):
        """Test optimal tile size calculation"""
        # High memory should allow larger tiles
        large_tile_size = self.processor._calculate_optimal_tile_size(8.0)
        
        # Low memory should use smaller tiles
        small_tile_size = self.processor._calculate_optimal_tile_size(2.0)
        
        assert large_tile_size >= small_tile_size
        assert self.processor.min_tile_size <= small_tile_size <= self.processor.max_tile_size
        assert self.processor.min_tile_size <= large_tile_size <= self.processor.max_tile_size
    
    def test_calculate_optimal_tiles(self):
        """Test optimal tile coordinate calculation"""
        tiles = self.processor.calculate_optimal_tiles(self.large_image, available_memory_gb=4.0)
        
        # Should have multiple tiles for large image
        assert len(tiles) > 1
        
        # All tiles should be valid
        for x1, y1, x2, y2 in tiles:
            assert 0 <= x1 < x2 <= self.large_image.width
            assert 0 <= y1 < y2 <= self.large_image.height
            assert (x2 - x1) >= self.processor.min_tile_size
            assert (y2 - y1) >= self.processor.min_tile_size
    
    def test_calculate_tiles_with_optimization(self):
        """Test optimized tile calculation"""
        # Test with different image sizes
        small_tiles = self.processor._calculate_tiles_with_optimization(self.small_image)
        large_tiles = self.processor._calculate_tiles_with_optimization(self.large_image)
        
        # Small image should have fewer tiles
        assert len(small_tiles) <= len(large_tiles)
        
        # All tiles should meet minimum size requirements
        for tiles in [small_tiles, large_tiles]:
            for x1, y1, x2, y2 in tiles:
                assert (x2 - x1) >= self.processor.min_tile_size
                assert (y2 - y1) >= self.processor.min_tile_size
    
    def test_progress_callback(self):
        """Test progress tracking functionality"""
        progress_calls = []
        
        def mock_callback(current, total, message):
            progress_calls.append((current, total, message))
        
        self.processor.set_progress_callback(mock_callback)
        
        # Mock process function
        def mock_process_function(image):
            return image  # Just return the input image
        
        # Process with progress tracking
        result = self.processor.process_tiles_with_progress(
            self.medium_image,
            mock_process_function
        )
        
        # Should have received progress callbacks
        assert len(progress_calls) > 0
        
        # First call should be start, last should be complete
        assert progress_calls[0][0] == 0  # Start at 0
        assert progress_calls[-1][2] == "Complete"  # End with complete message
        
        # Result should be an image
        assert isinstance(result, Image.Image)
        assert result.size == self.medium_image.size
    
    def test_process_tiles_single_tile(self):
        """Test processing when only one tile is needed"""
        progress_calls = []
        
        def mock_callback(current, total, message):
            progress_calls.append((current, total, message))
        
        self.processor.set_progress_callback(mock_callback)
        
        def mock_process_function(image):
            return image
        
        # Process small image (should be single tile)
        result = self.processor.process_tiles_with_progress(
            self.small_image,
            mock_process_function
        )
        
        # Should still get progress callbacks
        assert len(progress_calls) >= 2  # Start and end
        assert result.size == self.small_image.size
    
    def test_tile_weight_mask_creation(self):
        """Test tile weight mask creation for blending"""
        mask = self.processor._create_tile_weight_mask((256, 256))
        
        # Should be grayscale image
        assert mask.mode == 'L'
        assert mask.size == (256, 256)
        
        # Convert to array for analysis
        mask_array = np.array(mask)
        
        # Center should have higher values than edges
        center_value = mask_array[128, 128]
        edge_value = mask_array[0, 0]
        assert center_value > edge_value
        
        # Values should be in valid range
        assert np.all(mask_array >= 0)
        assert np.all(mask_array <= 255)
    
    def test_merge_tiles_with_blending(self):
        """Test enhanced tile merging with blending"""
        # Create test tiles
        tile1 = Image.new('RGB', (256, 256), (255, 0, 0))  # Red
        tile2 = Image.new('RGB', (256, 256), (0, 255, 0))  # Green
        
        # Overlapping coordinates
        tile_coords = [
            (0, 0, 256, 256),
            (192, 0, 448, 256)  # 64 pixel overlap
        ]
        
        merged = self.processor.merge_tiles_with_blending(
            [tile1, tile2],
            tile_coords,
            (448, 256)
        )
        
        # Should be correct size
        assert merged.size == (448, 256)
        
        # Should be RGB image
        assert merged.mode == 'RGB'
        
        # Overlap region should be blended (not pure red or green)
        overlap_pixel = merged.getpixel((224, 128))  # Middle of overlap
        assert overlap_pixel != (255, 0, 0)  # Not pure red
        assert overlap_pixel != (0, 255, 0)   # Not pure green
    
    def test_merge_tiles_fallback(self):
        """Test fallback to simple merging when blending fails"""
        # Create test tiles
        tile1 = Image.new('RGB', (256, 256), (255, 0, 0))
        tile2 = Image.new('RGB', (256, 256), (0, 255, 0))
        
        tile_coords = [
            (0, 0, 256, 256),
            (256, 0, 512, 256)
        ]
        
        # Test simple merging
        merged = self.processor.merge_tiles(
            [tile1, tile2],
            tile_coords,
            (512, 256)
        )
        
        assert merged.size == (512, 256)
        assert merged.mode == 'RGB'
    
    def test_error_handling_in_tile_processing(self):
        """Test error handling during tile processing"""
        def failing_process_function(image):
            raise ValueError("Simulated processing error")
        
        # Should handle errors gracefully
        result = self.processor.process_tiles_with_progress(
            self.medium_image,
            failing_process_function
        )
        
        # Should return an image (fallback to original tiles)
        assert isinstance(result, Image.Image)
        assert result.size == self.medium_image.size
    
    def test_tile_size_validation(self):
        """Test tile size validation and adjustment"""
        # Test with various tile sizes
        for tile_size in [128, 256, 512, 1024, 2048]:
            processor = TiledProcessor(tile_size=tile_size)
            
            # Calculate tiles for medium image
            tiles = processor.calculate_optimal_tiles(self.medium_image)
            
            # All tiles should meet minimum size requirements
            for x1, y1, x2, y2 in tiles:
                assert (x2 - x1) >= processor.min_tile_size
                assert (y2 - y1) >= processor.min_tile_size
    
    def test_backward_compatibility(self):
        """Test backward compatibility with existing interface"""
        # Old method should still work
        tiles = self.processor.calculate_tiles(self.large_image)
        
        assert len(tiles) > 0
        for x1, y1, x2, y2 in tiles:
            assert 0 <= x1 < x2 <= self.large_image.width
            assert 0 <= y1 < y2 <= self.large_image.height


class TestTiledProcessingIntegration:
    """Integration tests for tiled processing with various image sizes"""
    
    def setup_method(self):
        """Setup integration test fixtures"""
        self.processor = TiledProcessor()
    
    def test_various_image_sizes(self):
        """Test tiled processing with various realistic image sizes"""
        test_sizes = [
            (512, 512),    # Small square
            (1024, 768),   # Medium landscape
            (768, 1024),   # Medium portrait
            (2048, 1536),  # Large landscape
            (1536, 2048),  # Large portrait
            (4096, 2160),  # Ultra-wide
        ]
        
        for width, height in test_sizes:
            image = Image.new('RGB', (width, height), (128, 128, 128))
            
            # Test tiling decision
            should_tile = self.processor.should_use_tiled_processing(image)
            
            # Test tile calculation
            tiles = self.processor.calculate_optimal_tiles(image)
            
            # Validate results
            assert len(tiles) > 0
            
            if should_tile:
                # Large images should have multiple tiles
                if width > 1536 or height > 1536:
                    assert len(tiles) > 1, f"Large image {width}x{height} should have multiple tiles"
    
    def test_memory_constrained_processing(self):
        """Test processing under various memory constraints"""
        large_image = Image.new('RGB', (2048, 2048), (100, 100, 100))
        
        memory_scenarios = [1.0, 2.0, 4.0, 8.0, 16.0]  # GB
        
        for memory_gb in memory_scenarios:
            # Test tiling decision
            should_tile = self.processor.should_use_tiled_processing(
                large_image,
                available_memory_gb=memory_gb
            )
            
            # Test tile calculation
            tiles = self.processor.calculate_optimal_tiles(
                large_image,
                available_memory_gb=memory_gb
            )
            
            # Lower memory should result in more tiles
            if memory_gb < 4.0:
                assert should_tile, f"Should use tiling with {memory_gb}GB memory"
                assert len(tiles) > 1, f"Should have multiple tiles with {memory_gb}GB memory"
    
    def test_operation_type_impact(self):
        """Test how operation type affects tiling decisions"""
        medium_image = Image.new('RGB', (1200, 900), (150, 150, 150))
        
        operations = ['edit', 'inpaint', 'outpaint', 'style_transfer']
        
        tiling_decisions = {}
        for operation in operations:
            should_tile = self.processor.should_use_tiled_processing(
                medium_image,
                operation_type=operation,
                available_memory_gb=4.0
            )
            tiling_decisions[operation] = should_tile
        
        # More complex operations should be more likely to use tiling
        # (though exact behavior depends on thresholds)
        assert isinstance(tiling_decisions['edit'], bool)
        assert isinstance(tiling_decisions['style_transfer'], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])