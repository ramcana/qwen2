"""
EliGen Integration for Enhanced Generation Quality
Provides entity-level control and advanced generation capabilities for DiffSynth
"""

import logging
import os
import time
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass
from enum import Enum
import torch
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class EliGenMode(Enum):
    """EliGen processing modes"""
    DISABLED = "disabled"
    BASIC = "basic"
    ENHANCED = "enhanced"
    ULTRA = "ultra"


class EntityType(Enum):
    """Entity types for region-specific control"""
    PERSON = "person"
    FACE = "face"
    OBJECT = "object"
    BACKGROUND = "background"
    CLOTHING = "clothing"
    HAIR = "hair"
    CUSTOM = "custom"


@dataclass
class EntityRegion:
    """Defines a region for entity-level control"""
    entity_type: EntityType
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: Optional[Image.Image] = None
    prompt: Optional[str] = None
    strength: float = 1.0
    priority: int = 1  # Higher priority regions processed first


@dataclass
class EliGenConfig:
    """Configuration for EliGen enhanced generation"""
    
    # Core EliGen settings
    mode: EliGenMode = EliGenMode.BASIC
    enable_entity_detection: bool = True
    enable_region_control: bool = True
    enable_quality_enhancement: bool = True
    
    # Quality enhancement settings
    upscale_factor: float = 1.0
    detail_enhancement: float = 0.5
    color_enhancement: float = 0.3
    sharpness_enhancement: float = 0.2
    
    # Entity detection settings
    detection_confidence: float = 0.7
    min_region_size: int = 64
    max_regions: int = 10
    
    # Processing settings
    multi_pass_generation: bool = False
    adaptive_steps: bool = True
    quality_feedback: bool = True
    
    # Performance settings
    enable_caching: bool = True
    parallel_processing: bool = False
    memory_optimization: bool = True


@dataclass
class QualityMetrics:
    """Quality assessment metrics for generated images"""
    sharpness_score: float = 0.0
    detail_score: float = 0.0
    color_accuracy: float = 0.0
    consistency_score: float = 0.0
    overall_quality: float = 0.0
    processing_time: float = 0.0


class EntityDetector:
    """Detects and segments entities in images for region-specific control"""
    
    def __init__(self, config: EliGenConfig):
        self.config = config
        self.detection_models = {}
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize entity detection models"""
        try:
            # Try to import detection libraries
            try:
                import cv2
                self.cv2_available = True
            except ImportError:
                self.cv2_available = False
                logger.warning("OpenCV not available, basic detection only")
            
            # Initialize face detection
            if self.cv2_available:
                self._init_face_detector()
            
            logger.info("✅ Entity detectors initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize entity detectors: {e}")
    
    def _init_face_detector(self):
        """Initialize face detection using OpenCV"""
        try:
            import cv2
            # Use Haar cascade for face detection (lightweight)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detection_models['face'] = cv2.CascadeClassifier(cascade_path)
            logger.debug("Face detector initialized")
        except Exception as e:
            logger.warning(f"Face detector initialization failed: {e}")
    
    def detect_entities(self, image: Image.Image) -> List[EntityRegion]:
        """
        Detect entities in image and return regions
        
        Args:
            image: Input PIL Image
            
        Returns:
            List of detected entity regions
        """
        if not self.config.enable_entity_detection:
            return []
        
        regions = []
        
        try:
            # Convert PIL to OpenCV format
            if self.cv2_available:
                import cv2
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                regions.extend(self._detect_faces(gray, image.size))
                
                # Detect objects (basic implementation)
                regions.extend(self._detect_objects(cv_image, image.size))
            
            # Filter regions by confidence and size
            regions = self._filter_regions(regions)
            
            logger.debug(f"Detected {len(regions)} entity regions")
            return regions
            
        except Exception as e:
            logger.error(f"Entity detection failed: {e}")
            return []
    
    def _detect_faces(self, gray_image, image_size) -> List[EntityRegion]:
        """Detect faces in image"""
        regions = []
        
        if 'face' not in self.detection_models:
            return regions
        
        try:
            import cv2
            face_cascade = self.detection_models['face']
            
            faces = face_cascade.detectMultiScale(
                gray_image,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                # Calculate confidence based on size and position
                confidence = min(1.0, (w * h) / (image_size[0] * image_size[1] * 0.1))
                
                if confidence >= self.config.detection_confidence:
                    region = EntityRegion(
                        entity_type=EntityType.FACE,
                        bbox=(x, y, x + w, y + h),
                        strength=confidence,
                        priority=3  # High priority for faces
                    )
                    regions.append(region)
            
            logger.debug(f"Detected {len(regions)} faces")
            
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
        
        return regions
    
    def _detect_objects(self, cv_image, image_size) -> List[EntityRegion]:
        """Basic object detection using contours"""
        regions = []
        
        try:
            import cv2
            
            # Simple contour-based object detection
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.config.min_region_size ** 2:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate confidence based on area and aspect ratio
                    aspect_ratio = w / h if h > 0 else 1.0
                    confidence = min(1.0, area / (image_size[0] * image_size[1] * 0.05))
                    
                    if confidence >= self.config.detection_confidence and 0.3 <= aspect_ratio <= 3.0:
                        region = EntityRegion(
                            entity_type=EntityType.OBJECT,
                            bbox=(x, y, x + w, y + h),
                            strength=confidence,
                            priority=1  # Lower priority for generic objects
                        )
                        regions.append(region)
            
            logger.debug(f"Detected {len(regions)} objects")
            
        except Exception as e:
            logger.warning(f"Object detection failed: {e}")
        
        return regions
    
    def _filter_regions(self, regions: List[EntityRegion]) -> List[EntityRegion]:
        """Filter and sort regions by priority and confidence"""
        # Remove overlapping regions (keep higher priority)
        filtered_regions = []
        
        # Sort by priority and strength
        sorted_regions = sorted(regions, key=lambda r: (r.priority, r.strength), reverse=True)
        
        for region in sorted_regions:
            # Check for significant overlap with existing regions
            overlap = False
            for existing in filtered_regions:
                if self._calculate_overlap(region.bbox, existing.bbox) > 0.5:
                    overlap = True
                    break
            
            if not overlap:
                filtered_regions.append(region)
            
            # Limit number of regions
            if len(filtered_regions) >= self.config.max_regions:
                break
        
        return filtered_regions
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class QualityEnhancer:
    """Enhances image quality using various techniques"""
    
    def __init__(self, config: EliGenConfig):
        self.config = config
        self._initialize_enhancers()
    
    def _initialize_enhancers(self):
        """Initialize quality enhancement modules"""
        try:
            # Check for available enhancement libraries
            self.pil_available = True  # PIL is always available
            
            try:
                import cv2
                self.cv2_available = True
            except ImportError:
                self.cv2_available = False
            
            logger.info("✅ Quality enhancers initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize quality enhancers: {e}")
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Apply quality enhancements to image
        
        Args:
            image: Input PIL Image
            
        Returns:
            Enhanced PIL Image
        """
        if not self.config.enable_quality_enhancement:
            return image
        
        try:
            enhanced_image = image.copy()
            
            # Apply enhancements based on configuration
            if self.config.detail_enhancement > 0:
                enhanced_image = self._enhance_details(enhanced_image)
            
            if self.config.color_enhancement > 0:
                enhanced_image = self._enhance_colors(enhanced_image)
            
            if self.config.sharpness_enhancement > 0:
                enhanced_image = self._enhance_sharpness(enhanced_image)
            
            if self.config.upscale_factor > 1.0:
                enhanced_image = self._upscale_image(enhanced_image)
            
            logger.debug("✅ Image quality enhanced")
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Quality enhancement failed: {e}")
            return image
    
    def _enhance_details(self, image: Image.Image) -> Image.Image:
        """Enhance image details"""
        try:
            from PIL import ImageEnhance, ImageFilter
            
            # Apply unsharp mask for detail enhancement
            blurred = image.filter(ImageFilter.GaussianBlur(radius=1.0))
            
            # Create detail mask
            detail_mask = Image.blend(image, blurred, -self.config.detail_enhancement)
            
            return detail_mask
            
        except Exception as e:
            logger.warning(f"Detail enhancement failed: {e}")
            return image
    
    def _enhance_colors(self, image: Image.Image) -> Image.Image:
        """Enhance image colors"""
        try:
            from PIL import ImageEnhance
            
            # Color enhancement
            enhancer = ImageEnhance.Color(image)
            enhanced = enhancer.enhance(1.0 + self.config.color_enhancement)
            
            # Contrast enhancement
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.0 + self.config.color_enhancement * 0.5)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Color enhancement failed: {e}")
            return image
    
    def _enhance_sharpness(self, image: Image.Image) -> Image.Image:
        """Enhance image sharpness"""
        try:
            from PIL import ImageEnhance
            
            enhancer = ImageEnhance.Sharpness(image)
            enhanced = enhancer.enhance(1.0 + self.config.sharpness_enhancement)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Sharpness enhancement failed: {e}")
            return image
    
    def _upscale_image(self, image: Image.Image) -> Image.Image:
        """Upscale image using high-quality resampling"""
        try:
            new_width = int(image.width * self.config.upscale_factor)
            new_height = int(image.height * self.config.upscale_factor)
            
            upscaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return upscaled
            
        except Exception as e:
            logger.warning(f"Image upscaling failed: {e}")
            return image
    
    def assess_quality(self, image: Image.Image) -> QualityMetrics:
        """
        Assess image quality metrics
        
        Args:
            image: PIL Image to assess
            
        Returns:
            Quality metrics
        """
        try:
            start_time = time.time()
            
            # Calculate various quality metrics
            sharpness = self._calculate_sharpness(image)
            detail_score = self._calculate_detail_score(image)
            color_accuracy = self._calculate_color_accuracy(image)
            consistency = self._calculate_consistency(image)
            
            # Overall quality score (weighted average)
            overall = (sharpness * 0.3 + detail_score * 0.3 + 
                      color_accuracy * 0.2 + consistency * 0.2)
            
            processing_time = time.time() - start_time
            
            return QualityMetrics(
                sharpness_score=sharpness,
                detail_score=detail_score,
                color_accuracy=color_accuracy,
                consistency_score=consistency,
                overall_quality=overall,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return QualityMetrics()
    
    def _calculate_sharpness(self, image: Image.Image) -> float:
        """Calculate image sharpness using Laplacian variance"""
        try:
            if self.cv2_available:
                import cv2
                gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                # Normalize to 0-1 range
                return min(1.0, laplacian_var / 1000.0)
            else:
                # Fallback: use PIL edge detection
                from PIL import ImageFilter
                edges = image.convert('L').filter(ImageFilter.FIND_EDGES)
                edge_array = np.array(edges)
                return min(1.0, edge_array.std() / 50.0)
                
        except Exception as e:
            logger.warning(f"Sharpness calculation failed: {e}")
            return 0.5
    
    def _calculate_detail_score(self, image: Image.Image) -> float:
        """Calculate detail richness score"""
        try:
            # Convert to grayscale and calculate texture variance
            gray = image.convert('L')
            gray_array = np.array(gray)
            
            # Calculate local variance (texture measure)
            from scipy import ndimage
            local_variance = ndimage.generic_filter(gray_array.astype(float), np.var, size=3)
            detail_score = local_variance.mean() / 255.0
            
            return min(1.0, detail_score)
            
        except Exception as e:
            # Fallback without scipy
            gray_array = np.array(image.convert('L'))
            variance = gray_array.var() / (255.0 ** 2)
            return min(1.0, variance * 10)
    
    def _calculate_color_accuracy(self, image: Image.Image) -> float:
        """Calculate color accuracy/naturalness"""
        try:
            # Simple color distribution analysis
            rgb_array = np.array(image)
            
            # Check color balance
            r_mean, g_mean, b_mean = rgb_array.mean(axis=(0, 1))
            color_balance = 1.0 - abs(r_mean - g_mean) / 255.0 - abs(g_mean - b_mean) / 255.0
            
            # Check saturation distribution
            hsv_array = np.array(image.convert('HSV'))
            saturation = hsv_array[:, :, 1]
            sat_score = 1.0 - abs(saturation.mean() - 128) / 128.0
            
            return (color_balance + sat_score) / 2.0
            
        except Exception as e:
            logger.warning(f"Color accuracy calculation failed: {e}")
            return 0.7
    
    def _calculate_consistency(self, image: Image.Image) -> float:
        """Calculate image consistency (smoothness vs noise)"""
        try:
            gray_array = np.array(image.convert('L')).astype(float)
            
            # Calculate gradient magnitude
            grad_x = np.abs(np.diff(gray_array, axis=1))
            grad_y = np.abs(np.diff(gray_array, axis=0))
            
            # Consistency is inverse of gradient variance
            grad_variance = (grad_x.var() + grad_y.var()) / 2.0
            consistency = 1.0 / (1.0 + grad_variance / 1000.0)
            
            return min(1.0, consistency)
            
        except Exception as e:
            logger.warning(f"Consistency calculation failed: {e}")
            return 0.6


class EliGenProcessor:
    """Main EliGen processor for enhanced generation with entity control"""
    
    def __init__(self, config: Optional[EliGenConfig] = None):
        self.config = config or EliGenConfig()
        self.entity_detector = EntityDetector(self.config)
        self.quality_enhancer = QualityEnhancer(self.config)
        
        # Processing state
        self.processing_cache = {}
        self.quality_history = []
        
        logger.info(f"✅ EliGen processor initialized (mode: {self.config.mode.value})")
    
    def process_with_eligen(
        self,
        image: Image.Image,
        prompt: str,
        pipeline_function,
        regions: Optional[List[EntityRegion]] = None,
        **kwargs
    ) -> Tuple[Image.Image, QualityMetrics]:
        """
        Process image with EliGen enhancements
        
        Args:
            image: Input PIL Image
            prompt: Generation prompt
            pipeline_function: DiffSynth pipeline function to use
            regions: Optional predefined entity regions
            **kwargs: Additional pipeline arguments
            
        Returns:
            Tuple of (enhanced_image, quality_metrics)
        """
        if self.config.mode == EliGenMode.DISABLED:
            # Direct processing without EliGen
            result = pipeline_function(image=image, prompt=prompt, **kwargs)
            metrics = QualityMetrics(overall_quality=0.5)
            return result, metrics
        
        try:
            start_time = time.time()
            
            # Step 1: Entity detection (if enabled and no regions provided)
            if regions is None and self.config.enable_region_control:
                regions = self.entity_detector.detect_entities(image)
                logger.debug(f"Detected {len(regions)} entity regions")
            
            # Step 2: Region-specific processing
            if regions and self.config.enable_region_control:
                processed_image = self._process_with_regions(
                    image, prompt, pipeline_function, regions, **kwargs
                )
            else:
                # Standard processing
                processed_image = pipeline_function(image=image, prompt=prompt, **kwargs)
            
            # Step 3: Quality enhancement
            if self.config.enable_quality_enhancement:
                enhanced_image = self.quality_enhancer.enhance_image(processed_image)
            else:
                enhanced_image = processed_image
            
            # Step 4: Multi-pass refinement (if enabled)
            if self.config.multi_pass_generation and self.config.mode in [EliGenMode.ENHANCED, EliGenMode.ULTRA]:
                enhanced_image = self._multi_pass_refinement(
                    enhanced_image, prompt, pipeline_function, **kwargs
                )
            
            # Step 5: Quality assessment
            quality_metrics = self.quality_enhancer.assess_quality(enhanced_image)
            quality_metrics.processing_time = time.time() - start_time
            
            # Update quality history
            self.quality_history.append(quality_metrics)
            if len(self.quality_history) > 100:  # Keep last 100 results
                self.quality_history.pop(0)
            
            logger.info(f"✅ EliGen processing completed (quality: {quality_metrics.overall_quality:.2f})")
            return enhanced_image, quality_metrics
            
        except Exception as e:
            logger.error(f"EliGen processing failed: {e}")
            # Fallback: return original image with low quality score
            metrics = QualityMetrics(overall_quality=0.3)
            return image, metrics
    
    def _process_with_regions(
        self,
        image: Image.Image,
        prompt: str,
        pipeline_function,
        regions: List[EntityRegion],
        **kwargs
    ) -> Image.Image:
        """Process image with region-specific control"""
        try:
            # Sort regions by priority
            sorted_regions = sorted(regions, key=lambda r: r.priority, reverse=True)
            
            result_image = image.copy()
            
            for region in sorted_regions:
                # Extract region
                x1, y1, x2, y2 = region.bbox
                region_image = image.crop((x1, y1, x2, y2))
                
                # Create region-specific prompt
                region_prompt = self._create_region_prompt(prompt, region)
                
                # Process region with adjusted strength
                region_kwargs = kwargs.copy()
                region_kwargs['strength'] = kwargs.get('strength', 0.8) * region.strength
                
                try:
                    processed_region = pipeline_function(
                        image=region_image,
                        prompt=region_prompt,
                        **region_kwargs
                    )
                    
                    # Paste processed region back
                    result_image.paste(processed_region, (x1, y1))
                    
                except Exception as e:
                    logger.warning(f"Region processing failed for {region.entity_type}: {e}")
                    continue
            
            return result_image
            
        except Exception as e:
            logger.error(f"Region-based processing failed: {e}")
            return pipeline_function(image=image, prompt=prompt, **kwargs)
    
    def _create_region_prompt(self, base_prompt: str, region: EntityRegion) -> str:
        """Create region-specific prompt"""
        if region.prompt:
            return region.prompt
        
        # Add entity-specific context to base prompt
        entity_contexts = {
            EntityType.FACE: "focusing on facial features and expression",
            EntityType.PERSON: "focusing on the person",
            EntityType.OBJECT: "focusing on the object details",
            EntityType.BACKGROUND: "focusing on the background",
            EntityType.CLOTHING: "focusing on clothing and fabric details",
            EntityType.HAIR: "focusing on hair texture and style"
        }
        
        context = entity_contexts.get(region.entity_type, "")
        if context:
            return f"{base_prompt}, {context}"
        
        return base_prompt
    
    def _multi_pass_refinement(
        self,
        image: Image.Image,
        prompt: str,
        pipeline_function,
        **kwargs
    ) -> Image.Image:
        """Apply multi-pass refinement for ultra quality"""
        try:
            refined_image = image
            
            # First pass: Detail enhancement
            detail_kwargs = kwargs.copy()
            detail_kwargs['strength'] = 0.3
            detail_kwargs['guidance_scale'] = kwargs.get('guidance_scale', 7.5) * 1.2
            
            refined_image = pipeline_function(
                image=refined_image,
                prompt=f"{prompt}, highly detailed, sharp focus",
                **detail_kwargs
            )
            
            # Second pass: Quality refinement (if ultra mode)
            if self.config.mode == EliGenMode.ULTRA:
                quality_kwargs = kwargs.copy()
                quality_kwargs['strength'] = 0.2
                quality_kwargs['num_inference_steps'] = kwargs.get('num_inference_steps', 20) + 10
                
                refined_image = pipeline_function(
                    image=refined_image,
                    prompt=f"{prompt}, masterpiece, best quality, ultra detailed",
                    **quality_kwargs
                )
            
            return refined_image
            
        except Exception as e:
            logger.error(f"Multi-pass refinement failed: {e}")
            return image
    
    def get_quality_presets(self) -> Dict[str, EliGenConfig]:
        """Get predefined quality presets"""
        presets = {
            "fast": EliGenConfig(
                mode=EliGenMode.BASIC,
                enable_entity_detection=False,
                enable_quality_enhancement=True,
                detail_enhancement=0.2,
                color_enhancement=0.1,
                multi_pass_generation=False
            ),
            "balanced": EliGenConfig(
                mode=EliGenMode.ENHANCED,
                enable_entity_detection=True,
                enable_quality_enhancement=True,
                detail_enhancement=0.5,
                color_enhancement=0.3,
                sharpness_enhancement=0.2,
                multi_pass_generation=False
            ),
            "quality": EliGenConfig(
                mode=EliGenMode.ENHANCED,
                enable_entity_detection=True,
                enable_quality_enhancement=True,
                detail_enhancement=0.7,
                color_enhancement=0.4,
                sharpness_enhancement=0.3,
                multi_pass_generation=True,
                adaptive_steps=True
            ),
            "ultra": EliGenConfig(
                mode=EliGenMode.ULTRA,
                enable_entity_detection=True,
                enable_quality_enhancement=True,
                detail_enhancement=0.8,
                color_enhancement=0.5,
                sharpness_enhancement=0.4,
                upscale_factor=1.2,
                multi_pass_generation=True,
                adaptive_steps=True,
                quality_feedback=True
            )
        }
        
        return presets
    
    def get_average_quality(self) -> float:
        """Get average quality from recent processing history"""
        if not self.quality_history:
            return 0.0
        
        return sum(q.overall_quality for q in self.quality_history) / len(self.quality_history)
    
    def optimize_config_for_hardware(self, available_memory_gb: float) -> EliGenConfig:
        """Optimize EliGen configuration based on available hardware"""
        config = self.config
        
        if available_memory_gb < 4.0:
            # Low memory: basic mode only
            config.mode = EliGenMode.BASIC
            config.enable_entity_detection = False
            config.multi_pass_generation = False
            config.parallel_processing = False
            
        elif available_memory_gb < 8.0:
            # Medium memory: enhanced mode with limitations
            config.mode = EliGenMode.ENHANCED
            config.enable_entity_detection = True
            config.max_regions = 5
            config.multi_pass_generation = False
            
        else:
            # High memory: full capabilities
            config.mode = EliGenMode.ULTRA
            config.enable_entity_detection = True
            config.multi_pass_generation = True
            config.parallel_processing = True
        
        logger.info(f"EliGen config optimized for {available_memory_gb:.1f}GB: {config.mode.value}")
        return config