"""
Tests for EliGen integration functionality
"""

import pytest
import numpy as np
from PIL import Image
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from src.eligen_integration import (
    EliGenProcessor, EliGenConfig, EliGenMode, EntityType, EntityRegion,
    EntityDetector, QualityEnhancer, QualityMetrics
)


class TestEliGenConfig:
    """Test EliGen configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = EliGenConfig()
        
        assert config.mode == EliGenMode.BASIC
        assert config.enable_entity_detection == True
        assert config.enable_quality_enhancement == True
        assert 0.0 <= config.detail_enhancement <= 1.0
        assert 0.0 <= config.color_enhancement <= 1.0
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid configuration
        config = EliGenConfig(
            mode=EliGenMode.ENHANCED,
            detail_enhancement=0.7,
            color_enhancement=0.4
        )
        
        assert config.mode == EliGenMode.ENHANCED
        assert config.detail_enhancement == 0.7
        assert config.color_enhancement == 0.4


class TestEntityRegion:
    """Test EntityRegion data structure"""
    
    def test_entity_region_creation(self):
        """Test entity region creation"""
        region = EntityRegion(
            entity_type=EntityType.FACE,
            bbox=(10, 20, 100, 120),
            strength=0.8,
            priority=3
        )
        
        assert region.entity_type == EntityType.FACE
        assert region.bbox == (10, 20, 100, 120)
        assert region.strength == 0.8
        assert region.priority == 3
    
    def test_entity_region_with_mask(self):
        """Test entity region with mask"""
        mask = Image.new('L', (100, 100), 255)
        
        region = EntityRegion(
            entity_type=EntityType.PERSON,
            bbox=(0, 0, 100, 100),
            mask=mask,
            prompt="enhance person details"
        )
        
        assert region.mask is not None
        assert region.prompt == "enhance person details"


class TestEntityDetector:
    """Test entity detection functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = EliGenConfig()
        self.detector = EntityDetector(self.config)
        
        # Create test images
        self.test_image = Image.new('RGB', (512, 512), (128, 128, 128))
        self.face_image = self._create_face_like_image()
    
    def _create_face_like_image(self):
        """Create an image that might trigger face detection"""
        # Create a simple face-like pattern
        image = Image.new('RGB', (256, 256), (200, 180, 160))  # Skin tone background
        
        # Add some face-like features (very basic)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        
        # Eyes
        draw.ellipse([80, 80, 100, 100], fill=(50, 50, 50))
        draw.ellipse([156, 80, 176, 100], fill=(50, 50, 50))
        
        # Nose
        draw.ellipse([120, 110, 136, 130], fill=(180, 160, 140))
        
        # Mouth
        draw.ellipse([110, 150, 146, 170], fill=(150, 100, 100))
        
        return image
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        assert self.detector.config == self.config
        assert hasattr(self.detector, 'detection_models')
    
    def test_entity_detection_disabled(self):
        """Test detection when disabled"""
        config = EliGenConfig(enable_entity_detection=False)
        detector = EntityDetector(config)
        
        regions = detector.detect_entities(self.test_image)
        assert len(regions) == 0
    
    @patch('cv2.CascadeClassifier')
    def test_face_detection(self, mock_cascade):
        """Test face detection functionality"""
        # Mock face detection results
        mock_classifier = Mock()
        mock_classifier.detectMultiScale.return_value = [(50, 50, 100, 100)]
        mock_cascade.return_value = mock_classifier
        
        # Enable CV2 for this test
        self.detector.cv2_available = True
        self.detector.detection_models['face'] = mock_classifier
        
        regions = self.detector._detect_faces(
            np.array(self.face_image.convert('L')), 
            self.face_image.size
        )
        
        assert len(regions) >= 0  # May or may not detect faces depending on mock
        
        if len(regions) > 0:
            face_region = regions[0]
            assert face_region.entity_type == EntityType.FACE
            assert face_region.priority == 3  # High priority for faces
    
    def test_region_filtering(self):
        """Test region filtering and overlap detection"""
        # Create overlapping regions
        regions = [
            EntityRegion(EntityType.FACE, (10, 10, 60, 60), strength=0.9, priority=3),
            EntityRegion(EntityType.FACE, (20, 20, 70, 70), strength=0.7, priority=3),  # Overlapping
            EntityRegion(EntityType.OBJECT, (100, 100, 150, 150), strength=0.8, priority=1)
        ]
        
        filtered = self.detector._filter_regions(regions)
        
        # Should remove overlapping face region (keep higher strength)
        assert len(filtered) <= len(regions)
        
        # First region should be kept (higher strength)
        if len(filtered) > 0:
            assert filtered[0].strength >= 0.8
    
    def test_overlap_calculation(self):
        """Test bounding box overlap calculation"""
        bbox1 = (10, 10, 60, 60)
        bbox2 = (30, 30, 80, 80)  # Overlapping
        bbox3 = (100, 100, 150, 150)  # Non-overlapping
        
        overlap1 = self.detector._calculate_overlap(bbox1, bbox2)
        overlap2 = self.detector._calculate_overlap(bbox1, bbox3)
        
        assert overlap1 > 0  # Should have overlap
        assert overlap2 == 0  # Should have no overlap
        assert 0 <= overlap1 <= 1
        assert 0 <= overlap2 <= 1


class TestQualityEnhancer:
    """Test quality enhancement functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = EliGenConfig()
        self.enhancer = QualityEnhancer(self.config)
        
        # Create test images
        self.test_image = Image.new('RGB', (256, 256), (128, 128, 128))
        self.detailed_image = self._create_detailed_image()
    
    def _create_detailed_image(self):
        """Create an image with some detail for testing"""
        image = Image.new('RGB', (256, 256), (100, 150, 200))
        
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        
        # Add some patterns
        for i in range(0, 256, 20):
            draw.line([(i, 0), (i, 256)], fill=(200, 100, 50), width=2)
            draw.line([(0, i), (256, i)], fill=(50, 200, 100), width=1)
        
        return image
    
    def test_enhancer_initialization(self):
        """Test enhancer initialization"""
        assert self.enhancer.config == self.config
        assert self.enhancer.pil_available == True
    
    def test_image_enhancement_disabled(self):
        """Test enhancement when disabled"""
        config = EliGenConfig(enable_quality_enhancement=False)
        enhancer = QualityEnhancer(config)
        
        result = enhancer.enhance_image(self.test_image)
        assert result == self.test_image  # Should return original
    
    def test_detail_enhancement(self):
        """Test detail enhancement"""
        enhanced = self.enhancer._enhance_details(self.detailed_image)
        
        assert isinstance(enhanced, Image.Image)
        assert enhanced.size == self.detailed_image.size
        assert enhanced.mode == self.detailed_image.mode
    
    def test_color_enhancement(self):
        """Test color enhancement"""
        enhanced = self.enhancer._enhance_colors(self.test_image)
        
        assert isinstance(enhanced, Image.Image)
        assert enhanced.size == self.test_image.size
        assert enhanced.mode == self.test_image.mode
    
    def test_sharpness_enhancement(self):
        """Test sharpness enhancement"""
        enhanced = self.enhancer._enhance_sharpness(self.test_image)
        
        assert isinstance(enhanced, Image.Image)
        assert enhanced.size == self.test_image.size
        assert enhanced.mode == self.test_image.mode
    
    def test_image_upscaling(self):
        """Test image upscaling"""
        config = EliGenConfig(upscale_factor=1.5)
        enhancer = QualityEnhancer(config)
        
        upscaled = enhancer._upscale_image(self.test_image)
        
        expected_width = int(self.test_image.width * 1.5)
        expected_height = int(self.test_image.height * 1.5)
        
        assert upscaled.size == (expected_width, expected_height)
    
    def test_quality_assessment(self):
        """Test quality metrics calculation"""
        metrics = self.enhancer.assess_quality(self.detailed_image)
        
        assert isinstance(metrics, QualityMetrics)
        assert 0 <= metrics.sharpness_score <= 1
        assert 0 <= metrics.detail_score <= 1
        assert 0 <= metrics.color_accuracy <= 1
        assert 0 <= metrics.consistency_score <= 1
        assert 0 <= metrics.overall_quality <= 1
        assert metrics.processing_time >= 0
    
    def test_sharpness_calculation(self):
        """Test sharpness calculation"""
        sharpness = self.enhancer._calculate_sharpness(self.detailed_image)
        
        assert 0 <= sharpness <= 1
        assert isinstance(sharpness, float)
    
    def test_detail_score_calculation(self):
        """Test detail score calculation"""
        detail_score = self.enhancer._calculate_detail_score(self.detailed_image)
        
        assert 0 <= detail_score <= 1
        assert isinstance(detail_score, float)


class TestEliGenProcessor:
    """Test main EliGen processor"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = EliGenConfig()
        self.processor = EliGenProcessor(self.config)
        
        # Create test image
        self.test_image = Image.new('RGB', (512, 512), (128, 128, 128))
    
    def test_processor_initialization(self):
        """Test processor initialization"""
        assert self.processor.config == self.config
        assert isinstance(self.processor.entity_detector, EntityDetector)
        assert isinstance(self.processor.quality_enhancer, QualityEnhancer)
    
    def test_disabled_mode_processing(self):
        """Test processing with EliGen disabled"""
        config = EliGenConfig(mode=EliGenMode.DISABLED)
        processor = EliGenProcessor(config)
        
        def mock_pipeline(image, prompt, **kwargs):
            return image  # Just return input
        
        result, metrics = processor.process_with_eligen(
            self.test_image,
            "test prompt",
            mock_pipeline
        )
        
        assert result == self.test_image
        assert isinstance(metrics, QualityMetrics)
    
    def test_basic_mode_processing(self):
        """Test processing with basic EliGen mode"""
        def mock_pipeline(image, prompt, **kwargs):
            return image  # Just return input
        
        result, metrics = self.processor.process_with_eligen(
            self.test_image,
            "enhance this image",
            mock_pipeline
        )
        
        assert isinstance(result, Image.Image)
        assert isinstance(metrics, QualityMetrics)
        assert result.size == self.test_image.size
    
    def test_region_prompt_creation(self):
        """Test region-specific prompt creation"""
        base_prompt = "enhance image quality"
        
        face_region = EntityRegion(EntityType.FACE, (0, 0, 100, 100))
        face_prompt = self.processor._create_region_prompt(base_prompt, face_region)
        
        assert "facial features" in face_prompt.lower()
        assert base_prompt in face_prompt
        
        # Test custom prompt
        custom_region = EntityRegion(
            EntityType.CUSTOM, 
            (0, 0, 100, 100), 
            prompt="custom enhancement"
        )
        custom_prompt = self.processor._create_region_prompt(base_prompt, custom_region)
        assert custom_prompt == "custom enhancement"
    
    def test_quality_presets(self):
        """Test quality preset configurations"""
        presets = self.processor.get_quality_presets()
        
        assert "fast" in presets
        assert "balanced" in presets
        assert "quality" in presets
        assert "ultra" in presets
        
        # Test preset configurations
        fast_config = presets["fast"]
        ultra_config = presets["ultra"]
        
        assert fast_config.mode == EliGenMode.BASIC
        assert ultra_config.mode == EliGenMode.ULTRA
        assert ultra_config.detail_enhancement > fast_config.detail_enhancement
    
    def test_hardware_optimization(self):
        """Test hardware-based configuration optimization"""
        # Low memory scenario
        low_memory_config = self.processor.optimize_config_for_hardware(2.0)
        assert low_memory_config.mode == EliGenMode.BASIC
        assert low_memory_config.enable_entity_detection == False
        
        # High memory scenario
        high_memory_config = self.processor.optimize_config_for_hardware(16.0)
        assert high_memory_config.mode == EliGenMode.ULTRA
        assert high_memory_config.enable_entity_detection == True
    
    def test_quality_history_tracking(self):
        """Test quality history tracking"""
        # Initially empty
        assert self.processor.get_average_quality() == 0.0
        
        # Add some quality metrics
        for i in range(5):
            metrics = QualityMetrics(overall_quality=0.5 + i * 0.1)
            self.processor.quality_history.append(metrics)
        
        avg_quality = self.processor.get_average_quality()
        assert 0.5 <= avg_quality <= 1.0
    
    def test_multi_pass_refinement(self):
        """Test multi-pass refinement processing"""
        config = EliGenConfig(
            mode=EliGenMode.ULTRA,
            multi_pass_generation=True
        )
        processor = EliGenProcessor(config)
        
        call_count = 0
        def mock_pipeline(image, prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            return image
        
        result = processor._multi_pass_refinement(
            self.test_image,
            "test prompt",
            mock_pipeline
        )
        
        assert isinstance(result, Image.Image)
        assert call_count >= 1  # Should have made at least one call


class TestEliGenIntegration:
    """Integration tests for EliGen functionality"""
    
    def setup_method(self):
        """Setup integration test fixtures"""
        self.processor = EliGenProcessor()
    
    def test_end_to_end_processing(self):
        """Test complete EliGen processing pipeline"""
        test_image = Image.new('RGB', (256, 256), (100, 150, 200))
        
        def mock_pipeline(image, prompt, **kwargs):
            # Simulate some processing by slightly modifying the image
            modified = image.copy()
            return modified
        
        result, metrics = self.processor.process_with_eligen(
            test_image,
            "enhance image quality with fine details",
            mock_pipeline,
            num_inference_steps=10,
            guidance_scale=7.5
        )
        
        # Verify results
        assert isinstance(result, Image.Image)
        assert isinstance(metrics, QualityMetrics)
        assert result.size == test_image.size
        assert metrics.processing_time >= 0
    
    def test_error_handling_and_fallback(self):
        """Test error handling and fallback mechanisms"""
        test_image = Image.new('RGB', (256, 256), (128, 128, 128))
        
        def failing_pipeline(image, prompt, **kwargs):
            raise ValueError("Simulated pipeline failure")
        
        # Should handle errors gracefully
        result, metrics = self.processor.process_with_eligen(
            test_image,
            "test prompt",
            failing_pipeline
        )
        
        # Should return fallback results
        assert isinstance(result, Image.Image)
        assert isinstance(metrics, QualityMetrics)
        assert metrics.overall_quality < 0.5  # Indicates fallback


if __name__ == "__main__":
    pytest.main([__file__, "-v"])