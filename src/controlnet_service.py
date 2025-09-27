"""
ControlNet Integration Service
Provides ControlNet detection, control map generation, and guided generation capabilities
"""

import logging
import os
import time
from typing import Dict, Any, Optional, Union, Tuple, List, Literal
from dataclasses import dataclass
from enum import Enum
import torch
import numpy as np
from PIL import Image
import cv2

# Import existing components for consistency
from qwen_image_config import OptimizationConfig, ModelArchitecture, create_optimization_config
from error_handler import ArchitectureAwareErrorHandler, ErrorInfo, ErrorCategory

logger = logging.getLogger(__name__)


class ControlNetType(Enum):
    """Supported ControlNet types"""
    CANNY = "canny"
    DEPTH = "depth"
    POSE = "pose"
    NORMAL = "normal"
    SEGMENTATION = "segmentation"
    SCRIBBLE = "scribble"
    LINEART = "lineart"
    AUTO = "auto"


@dataclass
class ControlNetRequest:
    """Request model for ControlNet-guided generation"""
    
    # Input data
    prompt: str
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    control_image_path: Optional[str] = None
    control_image_base64: Optional[str] = None
    
    # ControlNet configuration
    control_type: ControlNetType = ControlNetType.AUTO
    controlnet_conditioning_scale: float = 1.0
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0
    
    # Generation parameters
    negative_prompt: Optional[str] = None
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    width: int = 768
    height: int = 768
    seed: Optional[int] = None
    
    # Processing options
    use_tiled_processing: Optional[bool] = None
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class ControlMapResult:
    """Result of control map generation"""
    control_image: Image.Image
    control_type: ControlNetType
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]


@dataclass
class ControlNetDetectionResult:
    """Result of automatic control type detection"""
    detected_type: ControlNetType
    confidence: float
    all_scores: Dict[ControlNetType, float]
    processing_time: float


class ControlNetService:
    """
    ControlNet integration service with automatic detection and control map generation
    Supports multiple ControlNet types with automatic detection capabilities
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize ControlNet service"""
        self.device = device
        self.error_handler = ArchitectureAwareErrorHandler()
        
        # Detection models (lazy loaded)
        self._canny_detector = None
        self._depth_detector = None
        self._pose_detector = None
        self._normal_detector = None
        self._segmentation_detector = None
        
        # ControlNet models (lazy loaded)
        self._controlnet_models = {}
        
        # Detection thresholds
        self.detection_thresholds = {
            ControlNetType.CANNY: 0.7,
            ControlNetType.DEPTH: 0.6,
            ControlNetType.POSE: 0.8,
            ControlNetType.NORMAL: 0.6,
            ControlNetType.SEGMENTATION: 0.7
        }
        
        logger.info(f"ControlNet service initialized with device: {self.device}")
    
    def detect_control_type(self, image: Union[str, Image.Image]) -> ControlNetDetectionResult:
        """
        Automatically detect the most appropriate ControlNet type for an input image
        
        Args:
            image: Input image (path or PIL Image)
            
        Returns:
            ControlNetDetectionResult with detected type and confidence scores
        """
        start_time = time.time()
        
        try:
            # Load and validate image
            input_image = self._load_and_validate_image(image)
            if input_image is None:
                raise ValueError("Failed to load input image")
            
            logger.info("🔍 Analyzing image for ControlNet type detection...")
            
            # Convert to numpy array for analysis
            image_array = np.array(input_image)
            
            # Calculate scores for each control type
            scores = {}
            
            # Canny edge detection score
            scores[ControlNetType.CANNY] = self._calculate_canny_score(image_array)
            
            # Depth estimation score
            scores[ControlNetType.DEPTH] = self._calculate_depth_score(image_array)
            
            # Pose detection score
            scores[ControlNetType.POSE] = self._calculate_pose_score(image_array)
            
            # Normal map score
            scores[ControlNetType.NORMAL] = self._calculate_normal_score(image_array)
            
            # Segmentation score
            scores[ControlNetType.SEGMENTATION] = self._calculate_segmentation_score(image_array)
            
            # Find the best control type
            best_type = max(scores.keys(), key=lambda k: scores[k])
            best_confidence = scores[best_type]
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Control type detection completed in {processing_time:.2f}s")
            logger.info(f"🎯 Detected type: {best_type.value} (confidence: {best_confidence:.3f})")
            
            return ControlNetDetectionResult(
                detected_type=best_type,
                confidence=best_confidence,
                all_scores=scores,
                processing_time=processing_time
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_pipeline_error(
                e, "ControlNet", "Detection", {"operation": "detect_control_type"}
            )
            logger.error(f"❌ Control type detection failed: {e}")
            
            # Return default fallback
            return ControlNetDetectionResult(
                detected_type=ControlNetType.CANNY,
                confidence=0.0,
                all_scores={},
                processing_time=time.time() - start_time
            )
    
    def generate_control_map(
        self, 
        image: Union[str, Image.Image], 
        control_type: ControlNetType,
        **kwargs
    ) -> ControlMapResult:
        """
        Generate control map for specified ControlNet type
        
        Args:
            image: Input image (path or PIL Image)
            control_type: Type of control map to generate
            **kwargs: Additional parameters for specific detectors
            
        Returns:
            ControlMapResult with generated control map
        """
        start_time = time.time()
        
        try:
            # Load and validate image
            input_image = self._load_and_validate_image(image)
            if input_image is None:
                raise ValueError("Failed to load input image")
            
            logger.info(f"🎨 Generating {control_type.value} control map...")
            
            # Generate control map based on type
            if control_type == ControlNetType.CANNY:
                control_image, metadata = self._generate_canny_map(input_image, **kwargs)
            elif control_type == ControlNetType.DEPTH:
                control_image, metadata = self._generate_depth_map(input_image, **kwargs)
            elif control_type == ControlNetType.POSE:
                control_image, metadata = self._generate_pose_map(input_image, **kwargs)
            elif control_type == ControlNetType.NORMAL:
                control_image, metadata = self._generate_normal_map(input_image, **kwargs)
            elif control_type == ControlNetType.SEGMENTATION:
                control_image, metadata = self._generate_segmentation_map(input_image, **kwargs)
            else:
                raise ValueError(f"Unsupported control type: {control_type}")
            
            processing_time = time.time() - start_time
            
            # Calculate confidence based on control map quality
            confidence = self._calculate_control_map_confidence(control_image, control_type)
            
            logger.info(f"✅ Control map generated in {processing_time:.2f}s (confidence: {confidence:.3f})")
            
            return ControlMapResult(
                control_image=control_image,
                control_type=control_type,
                confidence=confidence,
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_pipeline_error(
                e, "ControlNet", "MapGeneration", {"control_type": control_type.value}
            )
            logger.error(f"❌ Control map generation failed: {e}")
            raise
    
    def process_with_control(self, request: ControlNetRequest) -> Dict[str, Any]:
        """
        Process ControlNet-guided generation request
        
        Args:
            request: ControlNetRequest with all parameters
            
        Returns:
            Dictionary with generation results and metadata
        """
        start_time = time.time()
        
        try:
            # Load input image if provided
            input_image = None
            if request.image_path or request.image_base64:
                input_image = self._load_image_from_request(request)
                if input_image is None:
                    raise ValueError("Failed to load input image")
            
            # Load or generate control image
            control_image = None
            control_type = request.control_type
            
            if request.control_image_path or request.control_image_base64:
                # Load provided control image
                control_image = self._load_control_image_from_request(request)
                if control_image is None:
                    raise ValueError("Failed to load control image")
                
                # Auto-detect control type if needed
                if control_type == ControlNetType.AUTO:
                    detection_result = self.detect_control_type(control_image)
                    control_type = detection_result.detected_type
                    logger.info(f"🎯 Auto-detected control type: {control_type.value}")
            
            elif input_image is not None:
                # Generate control map from input image
                if control_type == ControlNetType.AUTO:
                    detection_result = self.detect_control_type(input_image)
                    control_type = detection_result.detected_type
                    logger.info(f"🎯 Auto-detected control type: {control_type.value}")
                
                control_map_result = self.generate_control_map(input_image, control_type)
                control_image = control_map_result.control_image
            
            else:
                raise ValueError("Either input image or control image must be provided")
            
            logger.info(f"🚀 Starting ControlNet-guided generation with {control_type.value}...")
            
            # Prepare generation parameters
            generation_params = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt or "",
                "control_image": control_image,
                "control_type": control_type,
                "controlnet_conditioning_scale": request.controlnet_conditioning_scale,
                "control_guidance_start": request.control_guidance_start,
                "control_guidance_end": request.control_guidance_end,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "width": request.width,
                "height": request.height,
                "seed": request.seed
            }
            
            # Perform ControlNet-guided generation
            # Note: This would integrate with the actual ControlNet pipeline
            # For now, we'll return a structured response
            generated_image = self._generate_with_controlnet(generation_params)
            
            processing_time = time.time() - start_time
            
            # Save generated image
            output_path = self._save_generated_image(generated_image, request)
            
            logger.info(f"✅ ControlNet generation completed in {processing_time:.2f}s")
            
            return {
                "success": True,
                "message": f"ControlNet generation completed in {processing_time:.2f}s",
                "image_path": output_path,
                "control_type": control_type.value,
                "processing_time": processing_time,
                "parameters": generation_params
            }
            
        except Exception as e:
            error_info = self.error_handler.handle_pipeline_error(
                e, "ControlNet", "Generation", {"control_type": control_type.value if 'control_type' in locals() else "unknown"}
            )
            logger.error(f"❌ ControlNet generation failed: {e}")
            
            return {
                "success": False,
                "message": "ControlNet generation failed",
                "error_details": str(e),
                "suggested_fixes": error_info.suggested_fixes if error_info else []
            }
    
    # Detection score calculation methods
    
    def _calculate_canny_score(self, image_array: np.ndarray) -> float:
        """Calculate score for canny edge detection suitability"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Calculate edge density
            edge_density = np.sum(edges > 0) / edges.size
            
            # Calculate edge continuity (connected components)
            num_labels, _ = cv2.connectedComponents(edges)
            continuity_score = 1.0 / (1.0 + num_labels / 100.0)  # Normalize
            
            # Combine metrics
            score = (edge_density * 0.7 + continuity_score * 0.3)
            return min(score * 2.0, 1.0)  # Scale and cap at 1.0
            
        except Exception as e:
            logger.warning(f"Canny score calculation failed: {e}")
            return 0.0
    
    def _calculate_depth_score(self, image_array: np.ndarray) -> float:
        """Calculate score for depth estimation suitability"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate gradient magnitude (proxy for depth variation)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Calculate depth variation score
            depth_variation = np.std(gradient_magnitude) / 255.0
            
            # Calculate spatial coherence (smooth regions indicate depth planes)
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            coherence = 1.0 - np.mean(np.abs(gray.astype(float) - blurred.astype(float))) / 255.0
            
            # Combine metrics
            score = depth_variation * 0.6 + coherence * 0.4
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Depth score calculation failed: {e}")
            return 0.0
    
    def _calculate_pose_score(self, image_array: np.ndarray) -> float:
        """Calculate score for pose detection suitability"""
        try:
            # Simple heuristic: look for human-like shapes and proportions
            # This is a simplified version - real implementation would use pose detection models
            
            # Convert to grayscale
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Look for vertical structures (potential human figures)
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
            vertical_features = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # Calculate vertical feature density
            vertical_density = np.sum(vertical_features > 0) / vertical_features.size
            
            # Look for skin-like colors (simple heuristic)
            hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
            skin_mask = cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))
            skin_density = np.sum(skin_mask > 0) / skin_mask.size
            
            # Combine metrics
            score = vertical_density * 0.4 + skin_density * 0.6
            return min(score * 3.0, 1.0)  # Scale up since these are weak signals
            
        except Exception as e:
            logger.warning(f"Pose score calculation failed: {e}")
            return 0.0
    
    def _calculate_normal_score(self, image_array: np.ndarray) -> float:
        """Calculate score for normal map suitability"""
        try:
            # Normal maps work well with surfaces that have clear geometric structure
            # Look for surface-like features and geometric patterns
            
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate surface variation using Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            surface_variation = np.std(laplacian) / 255.0
            
            # Look for geometric patterns using template matching
            # Simple geometric shapes indicate good normal map candidates
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
            geometric_score = 0.5 if circles is not None else 0.0
            
            # Combine metrics
            score = surface_variation * 0.7 + geometric_score * 0.3
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Normal score calculation failed: {e}")
            return 0.0
    
    def _calculate_segmentation_score(self, image_array: np.ndarray) -> float:
        """Calculate score for segmentation suitability"""
        try:
            # Segmentation works well with images that have distinct regions
            # Look for color coherence and region boundaries
            
            # Convert to LAB color space for better color distance
            lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
            
            # Calculate color variance (high variance suggests multiple distinct regions)
            color_variance = np.var(lab, axis=(0, 1)).mean() / 255.0
            
            # Use watershed to find potential segments
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Count distinct regions
            num_labels, _ = cv2.connectedComponents(binary)
            region_score = min(num_labels / 20.0, 1.0)  # Normalize to 0-1
            
            # Combine metrics
            score = color_variance * 0.6 + region_score * 0.4
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Segmentation score calculation failed: {e}")
            return 0.0   
 
    # Control map generation methods
    
    def _generate_canny_map(self, image: Image.Image, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Generate Canny edge control map"""
        try:
            # Extract parameters
            low_threshold = kwargs.get('low_threshold', 50)
            high_threshold = kwargs.get('high_threshold', 150)
            blur_kernel = kwargs.get('blur_kernel', 3)
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur to reduce noise
            if blur_kernel > 0:
                gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            
            # Convert back to RGB
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            # Convert to PIL Image
            control_image = Image.fromarray(edges_rgb)
            
            metadata = {
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "blur_kernel": blur_kernel,
                "edge_density": np.sum(edges > 0) / edges.size
            }
            
            logger.debug(f"✅ Canny map generated (edge density: {metadata['edge_density']:.3f})")
            return control_image, metadata
            
        except Exception as e:
            logger.error(f"❌ Canny map generation failed: {e}")
            raise
    
    def _generate_depth_map(self, image: Image.Image, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Generate depth estimation control map"""
        try:
            # For now, use a simple depth estimation based on gradients
            # In a full implementation, this would use a depth estimation model like MiDaS
            
            image_array = np.array(image)
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate gradient-based depth approximation
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Normalize to 0-255
            depth_map = ((gradient_magnitude / gradient_magnitude.max()) * 255).astype(np.uint8)
            
            # Apply smoothing
            depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
            
            # Convert to RGB
            depth_rgb = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
            
            control_image = Image.fromarray(depth_rgb)
            
            metadata = {
                "method": "gradient_based",
                "depth_range": float(gradient_magnitude.max()),
                "smoothing_applied": True
            }
            
            logger.debug("✅ Depth map generated (gradient-based approximation)")
            return control_image, metadata
            
        except Exception as e:
            logger.error(f"❌ Depth map generation failed: {e}")
            raise
    
    def _generate_pose_map(self, image: Image.Image, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Generate pose detection control map"""
        try:
            # Simplified pose detection - in full implementation would use OpenPose or similar
            image_array = np.array(image)
            
            # Create a simple skeleton-like structure based on edge detection
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Apply morphological operations to find potential limb structures
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            # Find contours that might represent body parts
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create pose map
            pose_map = np.zeros_like(image_array)
            
            # Draw simplified pose structure
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small contours
                    cv2.drawContours(pose_map, [contour], -1, (255, 255, 255), 2)
            
            control_image = Image.fromarray(pose_map)
            
            metadata = {
                "method": "contour_based",
                "contours_found": len(contours),
                "note": "Simplified pose detection - use dedicated pose model for production"
            }
            
            logger.debug(f"✅ Pose map generated ({len(contours)} contours found)")
            return control_image, metadata
            
        except Exception as e:
            logger.error(f"❌ Pose map generation failed: {e}")
            raise
    
    def _generate_normal_map(self, image: Image.Image, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Generate normal map control map"""
        try:
            # Generate normal map from height information
            image_array = np.array(image)
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
            
            # Calculate gradients
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            
            # Calculate normal vectors
            # Normal map encoding: R=X, G=Y, B=Z
            normal_x = grad_x / 255.0
            normal_y = grad_y / 255.0
            normal_z = np.ones_like(normal_x)
            
            # Normalize
            length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
            normal_x /= length
            normal_y /= length
            normal_z /= length
            
            # Convert to 0-255 range
            normal_map = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
            normal_map[:, :, 0] = ((normal_x + 1.0) * 127.5).astype(np.uint8)  # R channel
            normal_map[:, :, 1] = ((normal_y + 1.0) * 127.5).astype(np.uint8)  # G channel
            normal_map[:, :, 2] = ((normal_z + 1.0) * 127.5).astype(np.uint8)  # B channel
            
            control_image = Image.fromarray(normal_map)
            
            metadata = {
                "method": "gradient_based_normal",
                "gradient_range_x": float(np.max(grad_x) - np.min(grad_x)),
                "gradient_range_y": float(np.max(grad_y) - np.min(grad_y))
            }
            
            logger.debug("✅ Normal map generated")
            return control_image, metadata
            
        except Exception as e:
            logger.error(f"❌ Normal map generation failed: {e}")
            raise
    
    def _generate_segmentation_map(self, image: Image.Image, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Generate segmentation control map"""
        try:
            # Simple segmentation using K-means clustering
            n_clusters = kwargs.get('n_clusters', 8)
            
            image_array = np.array(image)
            
            # Reshape for K-means
            pixel_values = image_array.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)
            
            # Apply K-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(pixel_values, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert back to uint8 and reshape
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]
            segmented_image = segmented_data.reshape(image_array.shape)
            
            control_image = Image.fromarray(segmented_image)
            
            metadata = {
                "method": "kmeans_clustering",
                "n_clusters": n_clusters,
                "cluster_centers": centers.tolist()
            }
            
            logger.debug(f"✅ Segmentation map generated ({n_clusters} clusters)")
            return control_image, metadata
            
        except Exception as e:
            logger.error(f"❌ Segmentation map generation failed: {e}")
            raise
    
    # Utility methods
    
    def _load_and_validate_image(self, image: Union[str, Image.Image]) -> Optional[Image.Image]:
        """Load and validate input image"""
        try:
            if isinstance(image, str):
                if os.path.exists(image):
                    return Image.open(image).convert("RGB")
                else:
                    logger.error(f"Image file not found: {image}")
                    return None
            elif isinstance(image, Image.Image):
                return image.convert("RGB")
            else:
                logger.error(f"Unsupported image type: {type(image)}")
                return None
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None
    
    def _load_image_from_request(self, request: ControlNetRequest) -> Optional[Image.Image]:
        """Load image from ControlNet request"""
        try:
            if request.image_path:
                return self._load_and_validate_image(request.image_path)
            elif request.image_base64:
                # Decode base64 image
                import base64
                from io import BytesIO
                
                image_data = base64.b64decode(request.image_base64)
                image = Image.open(BytesIO(image_data))
                return image.convert("RGB")
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to load image from request: {e}")
            return None
    
    def _load_control_image_from_request(self, request: ControlNetRequest) -> Optional[Image.Image]:
        """Load control image from ControlNet request"""
        try:
            if request.control_image_path:
                return self._load_and_validate_image(request.control_image_path)
            elif request.control_image_base64:
                # Decode base64 image
                import base64
                from io import BytesIO
                
                image_data = base64.b64decode(request.control_image_base64)
                image = Image.open(BytesIO(image_data))
                return image.convert("RGB")
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to load control image from request: {e}")
            return None
    
    def _calculate_control_map_confidence(self, control_image: Image.Image, control_type: ControlNetType) -> float:
        """Calculate confidence score for generated control map"""
        try:
            control_array = np.array(control_image)
            
            if control_type == ControlNetType.CANNY:
                # For Canny, confidence is based on edge density and distribution
                gray = cv2.cvtColor(control_array, cv2.COLOR_RGB2GRAY)
                edge_density = np.sum(gray > 0) / gray.size
                
                # Calculate edge continuity (fewer disconnected components = higher confidence)
                num_labels, _ = cv2.connectedComponents(gray)
                continuity_score = 1.0 / (1.0 + num_labels / 10.0)
                
                # Combine edge density and continuity
                confidence = (edge_density * 10.0 + continuity_score) / 2.0
                return min(confidence, 1.0)
            
            elif control_type == ControlNetType.DEPTH:
                # For depth, confidence is based on depth variation and smoothness
                gray = cv2.cvtColor(control_array, cv2.COLOR_RGB2GRAY)
                depth_variation = np.std(gray) / 255.0
                
                # Calculate gradient smoothness
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                smoothness = 1.0 - (np.std(gradient_magnitude) / 255.0)
                
                confidence = (depth_variation * 3.0 + smoothness) / 2.0
                return min(confidence, 1.0)
            
            elif control_type == ControlNetType.POSE:
                # For pose, confidence is based on structure presence and organization
                gray = cv2.cvtColor(control_array, cv2.COLOR_RGB2GRAY)
                structure_density = np.sum(gray > 0) / gray.size
                
                # Look for organized structures (lines, connections)
                contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                structure_organization = min(len(contours) / 20.0, 1.0)
                
                confidence = (structure_density * 5.0 + structure_organization) / 2.0
                return min(confidence, 1.0)
            
            elif control_type == ControlNetType.NORMAL:
                # For normal maps, confidence is based on gradient variation across all channels
                total_variation = 0.0
                for channel in range(3):
                    channel_std = np.std(control_array[:, :, channel])
                    total_variation += channel_std / 255.0
                
                # Average variation across channels
                confidence = total_variation / 3.0 * 2.0
                return min(confidence, 1.0)
            
            elif control_type == ControlNetType.SEGMENTATION:
                # For segmentation, confidence is based on region distinctness and distribution
                unique_colors = len(np.unique(control_array.reshape(-1, 3), axis=0))
                color_diversity = min(unique_colors / 20.0, 1.0)
                
                # Calculate color distribution uniformity
                pixel_values = control_array.reshape(-1, 3)
                color_counts = {}
                for pixel in pixel_values:
                    key = tuple(pixel)
                    color_counts[key] = color_counts.get(key, 0) + 1
                
                # More uniform distribution = higher confidence
                count_values = list(color_counts.values())
                distribution_uniformity = 1.0 - (np.std(count_values) / np.mean(count_values)) if count_values else 0.0
                distribution_uniformity = max(0.0, min(distribution_uniformity, 1.0))
                
                confidence = (color_diversity + distribution_uniformity) / 2.0
                return min(confidence, 1.0)
            
            else:
                return 0.5  # Default confidence
                
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _generate_with_controlnet(self, params: Dict[str, Any]) -> Image.Image:
        """
        Generate image with ControlNet guidance
        
        Note: This is a placeholder for the actual ControlNet pipeline integration
        In a full implementation, this would use the DiffSynth ControlNet pipeline
        """
        try:
            # For now, return the control image as a placeholder
            # In actual implementation, this would:
            # 1. Load the appropriate ControlNet model
            # 2. Set up the pipeline with control conditioning
            # 3. Generate the image with ControlNet guidance
            
            control_image = params["control_image"]
            
            logger.info("🔧 ControlNet generation (placeholder implementation)")
            logger.info(f"   Control type: {params['control_type'].value}")
            logger.info(f"   Conditioning scale: {params['controlnet_conditioning_scale']}")
            logger.info(f"   Prompt: {params['prompt'][:50]}...")
            
            # Placeholder: return a modified version of the control image
            # This would be replaced with actual ControlNet pipeline call
            return control_image
            
        except Exception as e:
            logger.error(f"❌ ControlNet generation failed: {e}")
            raise
    
    def _save_generated_image(self, image: Image.Image, request: ControlNetRequest) -> str:
        """Save generated image and return path"""
        try:
            # Generate output filename
            timestamp = int(time.time())
            output_filename = f"controlnet_generated_{timestamp}.jpg"
            output_path = os.path.join("generated_images", output_filename)
            
            # Ensure output directory exists
            os.makedirs("generated_images", exist_ok=True)
            
            # Save image
            image.save(output_path, quality=95)
            
            logger.debug(f"✅ Image saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save generated image: {e}")
            return ""