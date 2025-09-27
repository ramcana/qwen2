"""
DiffSynth Service Foundation
Provides a service wrapper for DiffSynth integration with memory optimization and resource management
"""

import logging
import os
import time
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass
from enum import Enum
import torch
from PIL import Image

# Import existing components for consistency
from qwen_image_config import OptimizationConfig, ModelArchitecture, create_optimization_config
from error_handler import ArchitectureAwareErrorHandler, ErrorInfo, ErrorCategory

# Import DiffSynth models and utilities
from diffsynth_models import (
    ImageEditRequest, ImageEditResponse, EditOperation, ProcessingMetrics,
    InpaintRequest, OutpaintRequest, StyleTransferRequest, encode_image_to_base64, decode_base64_to_image
)
from diffsynth_utils import ImagePreprocessor, ImagePostprocessor, TiledProcessor, estimate_processing_time
from eligen_integration import EliGenProcessor, EliGenConfig, EliGenMode, QualityMetrics
from resource_manager import get_resource_manager, ServiceType, ResourcePriority

logger = logging.getLogger(__name__)


class DiffSynthServiceStatus(Enum):
    """Service status enumeration"""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class DiffSynthConfig:
    """Configuration for DiffSynth service"""
    
    # Model configuration
    model_name: str = "Qwen/Qwen-Image-Edit"
    torch_dtype: torch.dtype = torch.bfloat16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Memory optimization settings (from diffsynth_qwen_setup.py)
    enable_vram_management: bool = True
    enable_cpu_offload: bool = True
    enable_layer_offload: bool = True
    low_cpu_mem_usage: bool = True
    use_tiled_processing: bool = True
    
    # Processing settings
    default_num_inference_steps: int = 20
    default_guidance_scale: float = 7.5
    default_height: int = 768
    default_width: int = 768
    
    # Resource management
    max_memory_usage_gb: float = 4.0
    enable_memory_monitoring: bool = True
    auto_cleanup: bool = True
    
    # Error handling
    enable_fallback: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # EliGen integration
    enable_eligen: bool = True
    eligen_mode: EliGenMode = EliGenMode.BASIC
    eligen_quality_preset: str = "balanced"


@dataclass
class ResourceUsage:
    """Resource usage tracking"""
    gpu_memory_allocated: float = 0.0
    gpu_memory_reserved: float = 0.0
    cpu_memory_used: float = 0.0
    processing_time: float = 0.0
    last_updated: float = 0.0


class DiffSynthService:
    """
    DiffSynth service wrapper with memory optimization and resource management
    Integrates DiffSynth capabilities while maintaining compatibility with existing Qwen services
    """
    
    def __init__(self, config: Optional[DiffSynthConfig] = None):
        """Initialize DiffSynth service with configuration"""
        self.config = config or DiffSynthConfig()
        self.status = DiffSynthServiceStatus.NOT_INITIALIZED
        self.pipeline = None
        self.error_handler = ArchitectureAwareErrorHandler()
        self.resource_usage = ResourceUsage()
        
        # Service state
        self.initialization_time: Optional[float] = None
        self.last_operation_time: Optional[float] = None
        self.operation_count: int = 0
        self.error_count: int = 0
        self.service_id = f"diffsynth_service_{id(self)}"
        
        # Resource management integration
        self.resource_manager = get_resource_manager()
        self._register_with_resource_manager()
        
        # Image processing utilities
        self.preprocessor = ImagePreprocessor()
        self.postprocessor = ImagePostprocessor()
        self.tiled_processor = TiledProcessor()
        
        # EliGen integration
        self.eligen_processor = None
        if self.config.enable_eligen:
            self._initialize_eligen()
        
        # Memory optimization patterns from diffsynth_qwen_setup.py
        self._setup_memory_optimizations()
        
        logger.info(f"DiffSynth service created with device: {self.config.device}")
    
    def __del__(self):
        """Cleanup resources when service is destroyed"""
        try:
            if hasattr(self, 'resource_manager') and hasattr(self, 'service_id'):
                self.resource_manager.unregister_service(self.service_id)
        except Exception as e:
            logger.debug(f"Cleanup during destruction failed: {e}")
    
    def _initialize_eligen(self) -> None:
        """Initialize EliGen processor with configuration"""
        try:
            # Get EliGen configuration
            eligen_config = self._create_eligen_config()
            
            # Initialize EliGen processor
            self.eligen_processor = EliGenProcessor(eligen_config)
            
            logger.info(f"✅ EliGen initialized (mode: {eligen_config.mode.value})")
            
        except Exception as e:
            logger.error(f"Failed to initialize EliGen: {e}")
            self.config.enable_eligen = False
            self.eligen_processor = None
    
    def _create_eligen_config(self) -> EliGenConfig:
        """Create EliGen configuration based on service settings"""
        # Get preset configuration
        presets = {
            "fast": EliGenConfig(
                mode=EliGenMode.BASIC,
                enable_entity_detection=False,
                enable_quality_enhancement=True,
                detail_enhancement=0.2,
                color_enhancement=0.1
            ),
            "balanced": EliGenConfig(
                mode=EliGenMode.ENHANCED,
                enable_entity_detection=True,
                enable_quality_enhancement=True,
                detail_enhancement=0.5,
                color_enhancement=0.3,
                sharpness_enhancement=0.2
            ),
            "quality": EliGenConfig(
                mode=EliGenMode.ENHANCED,
                enable_entity_detection=True,
                enable_quality_enhancement=True,
                detail_enhancement=0.7,
                color_enhancement=0.4,
                sharpness_enhancement=0.3,
                multi_pass_generation=True
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
                adaptive_steps=True
            )
        }
        
        # Get base configuration from preset
        base_config = presets.get(self.config.eligen_quality_preset, presets["balanced"])
        
        # Override with service-specific settings
        base_config.mode = self.config.eligen_mode
        
        # Optimize for available hardware
        if self.eligen_processor:
            available_memory = self.tiled_processor._get_available_memory()
            base_config = self.eligen_processor.optimize_config_for_hardware(available_memory)
        
        return base_config
    
    def _register_with_resource_manager(self) -> None:
        """Register this service with the resource manager"""
        try:
            success = self.resource_manager.register_service(
                service_type=ServiceType.DIFFSYNTH_SERVICE,
                service_id=self.service_id,
                priority=ResourcePriority.NORMAL,
                cleanup_callback=self._cleanup_resources
            )
            
            if success:
                logger.info(f"✅ Registered DiffSynth service with ResourceManager: {self.service_id}")
            else:
                logger.warning(f"⚠️ Failed to register DiffSynth service with ResourceManager")
                
        except Exception as e:
            logger.error(f"❌ ResourceManager registration failed: {e}")
    
    def _setup_memory_optimizations(self) -> None:
        """Setup memory optimizations based on diffsynth_qwen_setup.py patterns"""
        if torch.cuda.is_available():
            # Enable memory efficient attention (from diffsynth_qwen_setup.py)
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            logger.info("✅ Memory optimizations configured for DiffSynth")
    
    def _request_gpu_memory(self, required_memory_gb: float) -> bool:
        """Request GPU memory allocation from resource manager"""
        try:
            success = self.resource_manager.request_memory(
                service_id=self.service_id,
                requested_memory_gb=required_memory_gb,
                force=False  # Don't force allocation initially
            )
            
            if success:
                logger.info(f"✅ Allocated {required_memory_gb:.1f}GB GPU memory for DiffSynth")
                return True
            else:
                # Try with force if initial request failed
                logger.warning(f"⚠️ Initial memory request failed, trying with force...")
                success = self.resource_manager.request_memory(
                    service_id=self.service_id,
                    requested_memory_gb=required_memory_gb,
                    force=True
                )
                
                if success:
                    logger.info(f"✅ Force-allocated {required_memory_gb:.1f}GB GPU memory for DiffSynth")
                    return True
                else:
                    logger.error(f"❌ Failed to allocate {required_memory_gb:.1f}GB GPU memory")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Memory allocation request failed: {e}")
            return False
    
    def _release_gpu_memory(self) -> None:
        """Release GPU memory allocation"""
        try:
            self.resource_manager.release_memory(self.service_id)
            logger.info("✅ Released GPU memory for DiffSynth")
        except Exception as e:
            logger.error(f"❌ Memory release failed: {e}")
    
    def _cleanup_resources(self) -> None:
        """Cleanup callback for resource manager"""
        try:
            if self.pipeline is not None:
                # Move pipeline to CPU if possible
                if hasattr(self.pipeline, 'to'):
                    self.pipeline.to('cpu')
                
                # Clear pipeline reference
                self.pipeline = None
                
            # Force GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("✅ DiffSynth resources cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Resource cleanup failed: {e}")
    
    def _update_resource_usage(self) -> None:
        """Update resource usage tracking"""
        try:
            if torch.cuda.is_available():
                self.resource_usage.gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
                self.resource_usage.gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9
            
            import psutil
            process = psutil.Process()
            self.resource_usage.cpu_memory_used = process.memory_info().rss / 1e9
            self.resource_usage.last_updated = time.time()
            
        except Exception as e:
            logger.debug(f"Resource usage update failed: {e}")
    
    def initialize(self) -> bool:
        """
        Initialize DiffSynth pipeline with memory optimization
        Returns True if successful, False otherwise
        """
        if self.status == DiffSynthServiceStatus.READY:
            logger.info("DiffSynth service already initialized")
            return True
        
        self.status = DiffSynthServiceStatus.INITIALIZING
        start_time = time.time()
        
        try:
            logger.info("🚀 Initializing DiffSynth service...")
            
            # Request GPU memory allocation
            if not self._request_gpu_memory(self.config.max_memory_usage_gb):
                logger.error("❌ Failed to allocate required GPU memory")
                self.status = DiffSynthServiceStatus.ERROR
                return False
            
            # Import DiffSynth components
            try:
                from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
                logger.info("✅ DiffSynth imports successful")
            except ImportError as e:
                error_info = self._handle_import_error(e)
                self._release_gpu_memory()  # Release allocated memory on failure
                self.status = DiffSynthServiceStatus.ERROR
                return False
            
            # Check system requirements
            if not self._check_system_requirements():
                self._release_gpu_memory()  # Release allocated memory on failure
                self.status = DiffSynthServiceStatus.ERROR
                return False
            
            # Create pipeline with memory optimization (based on diffsynth_qwen_setup.py)
            self.pipeline = self._create_optimized_pipeline(QwenImagePipeline, ModelConfig)
            
            if self.pipeline is None:
                self._release_gpu_memory()  # Release allocated memory on failure
                self.status = DiffSynthServiceStatus.ERROR
                return False
            
            # Enable VRAM management
            if hasattr(self.pipeline, 'enable_vram_management'):
                self.pipeline.enable_vram_management()
                logger.info("✅ VRAM management enabled")
            
            # Verify initialization
            if not self._verify_initialization():
                self._release_gpu_memory()  # Release allocated memory on failure
                self.status = DiffSynthServiceStatus.ERROR
                return False
            
            self.initialization_time = time.time() - start_time
            self.status = DiffSynthServiceStatus.READY
            
            logger.info(f"✅ DiffSynth service initialized successfully in {self.initialization_time:.2f}s")
            return True
            
        except Exception as e:
            error_info = self.error_handler.handle_pipeline_error(
                e, self.config.model_name, "DiffSynth", {"operation": "initialization"}
            )
            self._log_error(error_info)
            self.status = DiffSynthServiceStatus.ERROR
            self.error_count += 1
            return False
    
    def _create_optimized_pipeline(self, QwenImagePipeline, ModelConfig):
        """Create optimized DiffSynth pipeline based on diffsynth_qwen_setup.py"""
        try:
            logger.info("🔧 Creating optimized DiffSynth pipeline...")
            
            # Model configurations with layer-by-layer offload (from diffsynth_qwen_setup.py)
            model_configs = [
                ModelConfig(
                    model_id=self.config.model_name,
                    origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                    offload_device="cpu" if self.config.enable_cpu_offload else None
                ),
                ModelConfig(
                    model_id=self.config.model_name,
                    origin_file_pattern="text_encoder/model*.safetensors",
                    offload_device="cpu" if self.config.enable_cpu_offload else None
                ),
                ModelConfig(
                    model_id=self.config.model_name,
                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                    # Keep VAE on GPU for better quality
                ),
            ]
            
            tokenizer_config = ModelConfig(
                model_id=self.config.model_name,
                origin_file_pattern="tokenizer/"
            )
            
            # Create pipeline with optimization settings
            pipeline = QwenImagePipeline.from_pretrained(
                torch_dtype=self.config.torch_dtype,
                device=self.config.device,
                model_configs=model_configs,
                tokenizer_config=tokenizer_config,
            )
            
            logger.info("✅ DiffSynth pipeline created with memory optimizations")
            return pipeline
            
        except Exception as e:
            logger.error(f"❌ Failed to create DiffSynth pipeline: {e}")
            return None
    
    def _check_system_requirements(self) -> bool:
        """Check system requirements for DiffSynth"""
        try:
            # Check GPU availability
            if self.config.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.config.device = "cpu"
            
            # Check memory requirements
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory_gb < self.config.max_memory_usage_gb:
                    logger.warning(f"GPU memory ({gpu_memory_gb:.1f}GB) below recommended ({self.config.max_memory_usage_gb}GB)")
                    # Enable more aggressive optimizations
                    self.config.enable_cpu_offload = True
                    self.config.use_tiled_processing = True
            
            # Check disk space for model cache
            import psutil
            disk_usage = psutil.disk_usage('.')
            free_gb = disk_usage.free / 1e9
            if free_gb < 10:  # Minimum 10GB for model files
                logger.error(f"Insufficient disk space: {free_gb:.1f}GB available, need at least 10GB")
                return False
            
            logger.info("✅ System requirements check passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ System requirements check failed: {e}")
            return False
    
    def _verify_initialization(self) -> bool:
        """Verify that the pipeline was initialized correctly"""
        try:
            if self.pipeline is None:
                logger.error("Pipeline is None after initialization")
                return False
            
            # Check if pipeline has required methods
            required_methods = ['__call__']
            for method in required_methods:
                if not hasattr(self.pipeline, method):
                    logger.error(f"Pipeline missing required method: {method}")
                    return False
            
            logger.info("✅ Pipeline verification passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline verification failed: {e}")
            return False
    
    def _handle_import_error(self, error: Exception) -> ErrorInfo:
        """Handle DiffSynth import errors"""
        suggested_fixes = [
            "Install DiffSynth-Studio: pip install diffsynth-studio",
            "Clone and install from source: git clone https://github.com/modelscope/DiffSynth-Studio.git",
            "Check Python environment and dependencies",
            "Verify PyTorch installation compatibility",
            "Update pip and try reinstalling: pip install --upgrade diffsynth-studio"
        ]
        
        error_info = ErrorInfo(
            category=ErrorCategory.CONFIGURATION,
            severity="HIGH",
            message="DiffSynth import failed",
            details=str(error),
            suggested_fixes=suggested_fixes,
            user_feedback="❌ DiffSynth not installed. Please install DiffSynth-Studio to use editing features."
        )
        
        self._log_error(error_info)
        return error_info
    
    def _generate_image_from_text(self, request: ImageEditRequest, start_time: float) -> ImageEditResponse:
        """
        Generate image from text prompt (text-to-image)
        
        Args:
            request: ImageEditRequest with text prompt
            start_time: Start time for processing metrics
            
        Returns:
            ImageEditResponse with generated image
        """
        try:
            logger.info(f"🎨 Starting text-to-image generation: {request.prompt[:50]}...")
            
            # Set default dimensions if not provided
            width = request.width or 1024
            height = request.height or 1024
            
            # Estimate processing time for generation
            # Create a dummy image for estimation
            dummy_image = Image.new('RGB', (width, height))
            estimated_time = estimate_processing_time(dummy_image, "edit")  # Use "edit" as closest operation
            logger.info(f"⏱️ Estimated processing time: {estimated_time:.1f}s")
            
            # Log pipeline parameters for debugging
            logger.info(f"🔧 Pipeline parameters: width={width}, height={height}, steps={request.num_inference_steps or 20}, cfg_scale={request.guidance_scale or 7.5}")
            
            # Check if pipeline is ready
            if self.pipeline is None:
                raise RuntimeError("Pipeline not initialized")
            
            logger.info("🚀 Calling DiffSynth pipeline...")
            
            # Generate image using DiffSynth pipeline
            # Note: QwenImagePipeline uses specific parameter names
            pipeline_params = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt or "",
                "height": height,
                "width": width,
                "num_inference_steps": request.num_inference_steps or 20,
                "cfg_scale": request.guidance_scale or 7.5,  # QwenImagePipeline uses cfg_scale
            }
            
            # Add seed if provided
            if request.seed:
                pipeline_params["seed"] = request.seed
            
            logger.info("⚡ Pipeline execution starting...")
            generated_image = self.pipeline(**pipeline_params)
            logger.info("✅ Pipeline execution completed")
            
            if generated_image is None:
                return ImageEditResponse(
                    success=False,
                    message="Text-to-image generation failed",
                    error_details="Pipeline returned None"
                )
            
            # Skip EliGen enhancement for now (can be enabled later)
            # TODO: Fix EliGen integration
            final_image = generated_image
            quality_metrics = QualityMetrics(overall_quality=0.7)
            logger.info("✅ Using generated image without EliGen enhancement")
            
            processing_time = time.time() - start_time
            self.last_operation_time = processing_time
            self.operation_count += 1
            
            # Save image and prepare response
            output_path = self._save_generated_image(final_image, request)
            
            # Create processing metrics
            metrics = ProcessingMetrics(
                operation_type="generate",
                processing_time=processing_time,
                gpu_memory_used=self.resource_usage.gpu_memory_allocated,
                cpu_memory_used=self.resource_usage.cpu_memory_used,
                input_resolution=None,
                output_resolution=(final_image.width, final_image.height),
                tiled_processing_used=False
            )
            
            # Update resource usage
            self._update_resource_usage()
            
            logger.info(f"✅ Text-to-image generation completed in {processing_time:.2f}s")
            
            return ImageEditResponse(
                success=True,
                message="Image generated successfully",
                image_path=output_path,
                operation=EditOperation.GENERATE,
                processing_time=processing_time,
                parameters={
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "width": width,
                    "height": height,
                    "steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "seed": request.seed
                },
                resource_usage={
                    "gpu_memory": f"{self.resource_usage.gpu_memory_allocated:.2f}GB",
                    "cpu_memory": f"{self.resource_usage.cpu_memory_used:.2f}GB",
                    "processing_time": f"{processing_time:.2f}s"
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Text-to-image generation failed: {str(e)}"
            logger.error(error_msg)
            
            return ImageEditResponse(
                success=False,
                message="Text-to-image generation failed",
                error_details=error_msg,
                processing_time=processing_time
            )
        finally:
            self.status = DiffSynthServiceStatus.READY
    
    def edit_image(self, request: ImageEditRequest) -> ImageEditResponse:
        """
        Main image editing interface using structured request/response
        
        Args:
            request: ImageEditRequest with all parameters
            
        Returns:
            ImageEditResponse with results and metadata
        """
        if self.status != DiffSynthServiceStatus.READY:
            if not self.initialize():
                return ImageEditResponse(
                    success=False,
                    message="DiffSynth service not available",
                    error_details="Service initialization failed"
                )
        
        self.status = DiffSynthServiceStatus.BUSY
        start_time = time.time()
        
        try:
            # Update resource usage
            self._update_resource_usage()
            
            # Handle text-to-image generation (no input image needed)
            if request.operation == EditOperation.GENERATE:
                return self._generate_image_from_text(request, start_time)
            
            # Load and preprocess input image for editing operations
            input_image = self._load_image_from_request(request)
            if input_image is None:
                return ImageEditResponse(
                    success=False,
                    message="Failed to load input image",
                    error_details="Invalid image input or unsupported format"
                )
            
            # Preprocess image
            processed_image = self.preprocessor.prepare_image_for_editing(
                input_image,
                target_width=request.width,
                target_height=request.height
            )
            
            # Estimate processing time
            estimated_time = estimate_processing_time(processed_image, "edit")
            logger.info(f"🎨 Starting image editing (estimated: {estimated_time:.1f}s): {request.prompt[:50]}...")
            
            # Check if tiled processing should be used
            use_tiled = request.use_tiled_processing
            if use_tiled is None:
                use_tiled = self.tiled_processor.should_use_tiled_processing(processed_image)
            
            # Perform image editing with EliGen enhancement
            if self.config.enable_eligen and self.eligen_processor:
                edited_image, quality_metrics = self._edit_image_with_eligen(request, processed_image, use_tiled)
            else:
                # Standard processing without EliGen
                if use_tiled and self.tiled_processor.should_use_tiled_processing(processed_image):
                    edited_image = self._edit_image_tiled(request, processed_image)
                else:
                    edited_image = self._edit_image_single(request, processed_image)
                quality_metrics = QualityMetrics(overall_quality=0.5)
            
            if edited_image is None:
                return ImageEditResponse(
                    success=False,
                    message="Image editing failed",
                    error_details="Pipeline processing returned None"
                )
            
            # Postprocess edited image
            final_image = self.postprocessor.postprocess_edited_image(
                edited_image,
                original_image=input_image,
                enhance_quality=True
            )
            
            processing_time = time.time() - start_time
            self.last_operation_time = processing_time
            self.operation_count += 1
            
            # Save image and prepare response
            output_path = self._save_edited_image(final_image, request)
            
            # Create processing metrics
            metrics = ProcessingMetrics(
                operation_type="edit",
                processing_time=processing_time,
                gpu_memory_used=self.resource_usage.gpu_memory_allocated,
                cpu_memory_used=self.resource_usage.cpu_memory_used,
                input_resolution=(input_image.width, input_image.height),
                output_resolution=(final_image.width, final_image.height),
                tiled_processing_used=use_tiled
            )
            
            # Add EliGen quality metrics if available
            eligen_quality = getattr(quality_metrics, 'overall_quality', 0.5) if 'quality_metrics' in locals() else 0.5
            
            # Update resource usage
            self._update_resource_usage()
            
            # Auto cleanup if enabled
            if self.config.auto_cleanup:
                self._cleanup_resources()
            
            self.status = DiffSynthServiceStatus.READY
            
            logger.info(f"✅ Image editing completed in {processing_time:.2f}s")
            
            return ImageEditResponse(
                success=True,
                message=f"Image edited successfully in {processing_time:.2f}s",
                image_path=output_path,
                operation=EditOperation.EDIT,
                processing_time=processing_time,
                parameters={
                    "prompt": request.prompt,
                    "num_inference_steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "strength": request.strength,
                    "seed": request.seed,
                    "eligen_enabled": self.config.enable_eligen,
                    "eligen_quality": eligen_quality
                },
                resource_usage=metrics.dict()
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_pipeline_error(
                e, self.config.model_name, "DiffSynth", {"operation": "edit_image"}
            )
            self._log_error(error_info)
            self.status = DiffSynthServiceStatus.READY
            self.error_count += 1
            
            # Try fallback if enabled
            if self.config.enable_fallback:
                return self._fallback_edit_image_response(request)
            
            return ImageEditResponse(
                success=False,
                message="Image editing failed",
                error_details=str(e),
                suggested_fixes=error_info.suggested_fixes if error_info else None
            )
    
    def _prepare_generation_params(
        self,
        prompt: str,
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        height: Optional[int],
        width: Optional[int],
        seed: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare generation parameters with defaults"""
        return {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps or self.config.default_num_inference_steps,
            "guidance_scale": guidance_scale or self.config.default_guidance_scale,
            "height": height or self.config.default_height,
            "width": width or self.config.default_width,
            "seed": seed,
        }
    
    def _load_image(self, image: Union[str, Image.Image]) -> Optional[Image.Image]:
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
    
    def _load_image_from_request(self, request: ImageEditRequest) -> Optional[Image.Image]:
        """Load image from request (path or base64)"""
        try:
            if request.image_path:
                return self.preprocessor.load_and_validate_image(request.image_path)
            elif request.image_base64:
                image = decode_base64_to_image(request.image_base64)
                return self.preprocessor.load_and_validate_image(image)
            else:
                logger.error("No image input provided in request")
                return None
        except Exception as e:
            logger.error(f"Failed to load image from request: {e}")
            return None
    
    def _edit_image_with_eligen(
        self, 
        request: ImageEditRequest, 
        image: Image.Image, 
        use_tiled: bool = False
    ) -> Tuple[Optional[Image.Image], QualityMetrics]:
        """
        Edit image using EliGen enhancement
        
        Args:
            request: Image edit request
            image: Preprocessed input image
            use_tiled: Whether to use tiled processing
            
        Returns:
            Tuple of (edited_image, quality_metrics)
        """
        try:
            # Define pipeline function for EliGen
            def pipeline_function(image, prompt, **kwargs):
                if use_tiled and self.tiled_processor.should_use_tiled_processing(image):
                    return self._edit_image_tiled_internal(image, prompt, **kwargs)
                else:
                    return self._edit_image_single_internal(image, prompt, **kwargs)
            
            # Prepare pipeline arguments
            pipeline_kwargs = {
                'negative_prompt': request.negative_prompt or "",
                'num_inference_steps': request.num_inference_steps or self.config.default_num_inference_steps,
                'guidance_scale': request.guidance_scale or self.config.default_guidance_scale,
                'strength': request.strength or 0.8,
                'seed': request.seed or 42
            }
            
            # Add additional parameters
            if request.additional_params:
                pipeline_kwargs.update(request.additional_params)
            
            # Process with EliGen
            edited_image, quality_metrics = self.eligen_processor.process_with_eligen(
                image=image,
                prompt=request.prompt,
                pipeline_function=pipeline_function,
                **pipeline_kwargs
            )
            
            logger.debug(f"✅ EliGen processing completed (quality: {quality_metrics.overall_quality:.2f})")
            return edited_image, quality_metrics
            
        except Exception as e:
            logger.error(f"EliGen-enhanced editing failed: {e}")
            # Fallback to standard processing
            if use_tiled:
                fallback_image = self._edit_image_tiled(request, image)
            else:
                fallback_image = self._edit_image_single(request, image)
            
            fallback_metrics = QualityMetrics(overall_quality=0.3)
            return fallback_image, fallback_metrics
    
    def _edit_image_single_internal(self, image: Image.Image, prompt: str, **kwargs) -> Optional[Image.Image]:
        """Internal method for single-pass editing (used by EliGen)"""
        try:
            edited_image = self.pipeline(
                prompt=prompt,
                image=image,
                **kwargs
            )
            return edited_image
        except Exception as e:
            logger.error(f"Single pass editing failed: {e}")
            return None
    
    def _edit_image_tiled_internal(self, image: Image.Image, prompt: str, **kwargs) -> Optional[Image.Image]:
        """Internal method for tiled editing (used by EliGen)"""
        try:
            logger.info("🔧 Using tiled processing for large image")
            
            # Calculate tiles
            tile_coords = self.tiled_processor.calculate_tiles(image)
            edited_tiles = []
            
            for i, (x1, y1, x2, y2) in enumerate(tile_coords):
                logger.debug(f"Processing tile {i+1}/{len(tile_coords)}: {x1},{y1} to {x2},{y2}")
                
                # Extract tile
                tile = image.crop((x1, y1, x2, y2))
                
                # Edit tile
                edited_tile = self._edit_image_single_internal(tile, prompt, **kwargs)
                if edited_tile is None:
                    logger.warning(f"Failed to edit tile {i+1}, using original")
                    edited_tile = tile
                
                edited_tiles.append(edited_tile)
            
            # Merge tiles
            merged_image = self.tiled_processor.merge_tiles(
                edited_tiles, tile_coords, image.size
            )
            
            logger.info(f"✅ Tiled processing completed with {len(tile_coords)} tiles")
            return merged_image
            
        except Exception as e:
            logger.error(f"Tiled processing failed: {e}")
            # Fallback to single pass
            return self._edit_image_single_internal(image, prompt, **kwargs)

    def _edit_image_single(self, request: ImageEditRequest, image: Image.Image) -> Optional[Image.Image]:
        """Edit image using single pass (non-tiled)"""
        try:
            edited_image = self.pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or "",
                image=image,
                num_inference_steps=request.num_inference_steps or self.config.default_num_inference_steps,
                guidance_scale=request.guidance_scale or self.config.default_guidance_scale,
                strength=request.strength or 0.8,
                seed=request.seed or 42,
                **(request.additional_params or {})
            )
            return edited_image
        except Exception as e:
            logger.error(f"Single pass editing failed: {e}")
            return None
    
    def _edit_image_tiled(self, request: ImageEditRequest, image: Image.Image) -> Optional[Image.Image]:
        """Edit image using tiled processing"""
        try:
            logger.info("🔧 Using tiled processing for large image")
            
            # Calculate tiles
            tile_coords = self.tiled_processor.calculate_tiles(image)
            edited_tiles = []
            
            for i, (x1, y1, x2, y2) in enumerate(tile_coords):
                logger.debug(f"Processing tile {i+1}/{len(tile_coords)}: {x1},{y1} to {x2},{y2}")
                
                # Extract tile
                tile = image.crop((x1, y1, x2, y2))
                
                # Edit tile
                edited_tile = self._edit_image_single(request, tile)
                if edited_tile is None:
                    logger.warning(f"Failed to edit tile {i+1}, using original")
                    edited_tile = tile
                
                edited_tiles.append(edited_tile)
            
            # Merge tiles
            merged_image = self.tiled_processor.merge_tiles(
                edited_tiles, tile_coords, image.size
            )
            
            logger.info(f"✅ Tiled processing completed with {len(tile_coords)} tiles")
            return merged_image
            
        except Exception as e:
            logger.error(f"Tiled processing failed: {e}")
            # Fallback to single pass
            return self._edit_image_single(request, image)
    
    def _save_edited_image(self, image: Image.Image, request: ImageEditRequest) -> str:
        """Save edited image and return path"""
        try:
            # Generate output filename
            timestamp = int(time.time())
            output_filename = f"diffsynth_edit_{timestamp}.jpg"
            output_path = os.path.join("generated_images", output_filename)
            
            # Prepare metadata
            metadata = {
                "prompt": request.prompt,
                "operation": "edit",
                "timestamp": timestamp,
                "model": self.config.model_name
            }
            
            # Save image
            success = self.postprocessor.save_image(
                image, output_path, quality=95, metadata=metadata
            )
            
            if success:
                logger.debug(f"✅ Image saved: {output_path}")
                return output_path
            else:
                logger.error("Failed to save edited image")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to save edited image: {e}")
            return ""
    
    def _save_generated_image(self, image: Image.Image, request: ImageEditRequest) -> str:
        """Save generated image and return path"""
        try:
            # Generate output filename
            timestamp = int(time.time())
            output_filename = f"diffsynth_gen_{timestamp}.jpg"
            output_path = os.path.join("generated_images", output_filename)
            
            # Prepare metadata
            metadata = {
                "prompt": request.prompt,
                "operation": "generate",
                "timestamp": timestamp,
                "model": self.config.model_name,
                "width": request.width,
                "height": request.height,
                "steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed
            }
            
            # Save image
            success = self.postprocessor.save_image(
                image, output_path, quality=95, metadata=metadata
            )
            
            if success:
                logger.debug(f"✅ Generated image saved: {output_path}")
                return output_path
            else:
                logger.error("Failed to save generated image")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to save generated image: {e}")
            return ""
    
    def inpaint(self, request: "InpaintRequest") -> ImageEditResponse:
        """
        Perform inpainting using DiffSynth pipeline with mask support
        
        Args:
            request: InpaintRequest with image, mask, and parameters
            
        Returns:
            ImageEditResponse with inpainted result
        """
        if self.status != DiffSynthServiceStatus.READY:
            if not self.initialize():
                return ImageEditResponse(
                    success=False,
                    message="DiffSynth service not available",
                    error_details="Service initialization failed"
                )
        
        self.status = DiffSynthServiceStatus.BUSY
        start_time = time.time()
        
        try:
            # Update resource usage
            self._update_resource_usage()
            
            # Load and validate input image
            input_image = self._load_image_from_request(request)
            if input_image is None:
                return ImageEditResponse(
                    success=False,
                    message="Failed to load input image",
                    error_details="Invalid image input or unsupported format"
                )
            
            # Load and validate mask
            mask_image = self._load_mask_from_request(request)
            if mask_image is None:
                return ImageEditResponse(
                    success=False,
                    message="Failed to load mask image",
                    error_details="Invalid mask input or unsupported format"
                )
            
            # Validate mask compatibility with image
            if not self._validate_mask_compatibility(input_image, mask_image):
                return ImageEditResponse(
                    success=False,
                    message="Mask incompatible with input image",
                    error_details="Mask dimensions must match input image dimensions"
                )
            
            # Preprocess image and mask
            processed_image = self.preprocessor.prepare_image_for_editing(
                input_image,
                target_width=request.width,
                target_height=request.height
            )
            
            processed_mask = self._preprocess_mask(
                mask_image, 
                processed_image.size,
                blur_radius=request.mask_blur or 4,
                invert=request.invert_mask or False
            )
            
            # Estimate processing time
            estimated_time = estimate_processing_time(processed_image, "inpaint")
            logger.info(f"🎨 Starting inpainting (estimated: {estimated_time:.1f}s): {request.prompt[:50]}...")
            
            # Check if tiled processing should be used
            use_tiled = request.use_tiled_processing
            if use_tiled is None:
                use_tiled = self.tiled_processor.should_use_tiled_processing(processed_image)
            
            # Perform inpainting
            if use_tiled and self.tiled_processor.should_use_tiled_processing(processed_image):
                inpainted_image = self._inpaint_image_tiled(request, processed_image, processed_mask)
            else:
                inpainted_image = self._inpaint_image_single(request, processed_image, processed_mask)
            
            if inpainted_image is None:
                self.error_count += 1
                self.status = DiffSynthServiceStatus.READY
                
                # Try fallback if enabled
                if self.config.enable_fallback:
                    return self._fallback_inpaint_response(request)
                
                return ImageEditResponse(
                    success=False,
                    message="Inpainting failed",
                    error_details="Pipeline processing returned None",
                    suggested_fixes=[
                        "Check DiffSynth installation and dependencies",
                        "Verify mask format and compatibility",
                        "Try reducing image or mask complexity"
                    ]
                )
            
            # Postprocess inpainted image
            final_image = self.postprocessor.postprocess_edited_image(
                inpainted_image,
                original_image=input_image,
                enhance_quality=True
            )
            
            processing_time = time.time() - start_time
            self.last_operation_time = processing_time
            self.operation_count += 1
            
            # Save image and prepare response
            output_path = self._save_edited_image(final_image, request)
            
            # Create processing metrics
            metrics = ProcessingMetrics(
                operation_type="inpaint",
                processing_time=processing_time,
                gpu_memory_used=self.resource_usage.gpu_memory_allocated,
                cpu_memory_used=self.resource_usage.cpu_memory_used,
                input_resolution=(input_image.width, input_image.height),
                output_resolution=(final_image.width, final_image.height),
                tiled_processing_used=use_tiled
            )
            
            # Update resource usage
            self._update_resource_usage()
            
            # Auto cleanup if enabled
            if self.config.auto_cleanup:
                self._cleanup_resources()
            
            self.status = DiffSynthServiceStatus.READY
            
            logger.info(f"✅ Inpainting completed in {processing_time:.2f}s")
            
            return ImageEditResponse(
                success=True,
                message=f"Inpainting completed successfully in {processing_time:.2f}s",
                image_path=output_path,
                operation=EditOperation.INPAINT,
                processing_time=processing_time,
                parameters={
                    "prompt": request.prompt,
                    "num_inference_steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "strength": request.strength,
                    "seed": request.seed,
                    "mask_blur": request.mask_blur,
                    "invert_mask": request.invert_mask
                },
                resource_usage=metrics.dict()
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_pipeline_error(
                e, self.config.model_name, "DiffSynth", {"operation": "inpaint"}
            )
            self._log_error(error_info)
            self.status = DiffSynthServiceStatus.READY
            self.error_count += 1
            
            # Try fallback if enabled
            if self.config.enable_fallback:
                return self._fallback_inpaint_response(request)
            
            return ImageEditResponse(
                success=False,
                message="Inpainting failed",
                error_details=str(e),
                suggested_fixes=error_info.suggested_fixes if error_info else [
                    "Check DiffSynth installation and dependencies",
                    "Verify mask format and compatibility",
                    "Try reducing image or mask complexity"
                ]
            )
    
    def outpaint(self, request: "OutpaintRequest") -> ImageEditResponse:
        """
        Perform outpainting using DiffSynth pipeline with automatic canvas expansion
        
        Args:
            request: OutpaintRequest with image, direction, and parameters
            
        Returns:
            ImageEditResponse with outpainted result
        """
        if self.status != DiffSynthServiceStatus.READY:
            if not self.initialize():
                return ImageEditResponse(
                    success=False,
                    message="DiffSynth service not available",
                    error_details="Service initialization failed"
                )
        
        self.status = DiffSynthServiceStatus.BUSY
        start_time = time.time()
        
        try:
            # Update resource usage
            self._update_resource_usage()
            
            # Load and validate input image
            input_image = self._load_image_from_request(request)
            if input_image is None:
                return ImageEditResponse(
                    success=False,
                    message="Failed to load input image",
                    error_details="Invalid image input or unsupported format"
                )
            
            # Expand canvas based on direction and pixels
            expanded_image, expanded_mask = self._expand_canvas_for_outpainting(
                input_image, 
                request.direction, 
                request.pixels,
                request.fill_mode or "edge"
            )
            
            if expanded_image is None or expanded_mask is None:
                return ImageEditResponse(
                    success=False,
                    message="Failed to expand canvas for outpainting",
                    error_details="Canvas expansion failed"
                )
            
            # Preprocess expanded image
            processed_image = self.preprocessor.prepare_image_for_editing(
                expanded_image,
                target_width=request.width,
                target_height=request.height
            )
            
            # Preprocess mask for outpainting areas
            processed_mask = self._preprocess_mask(
                expanded_mask,
                processed_image.size,
                blur_radius=4,  # Soft edges for outpainting
                invert=False
            )
            
            # Estimate processing time
            estimated_time = estimate_processing_time(processed_image, "outpaint")
            logger.info(f"🎨 Starting outpainting (estimated: {estimated_time:.1f}s): {request.prompt[:50]}...")
            
            # Check if tiled processing should be used
            use_tiled = request.use_tiled_processing
            if use_tiled is None:
                use_tiled = self.tiled_processor.should_use_tiled_processing(processed_image)
            
            # Perform outpainting using inpainting pipeline with expanded canvas
            if use_tiled and self.tiled_processor.should_use_tiled_processing(processed_image):
                outpainted_image = self._outpaint_image_tiled(request, processed_image, processed_mask)
            else:
                outpainted_image = self._outpaint_image_single(request, processed_image, processed_mask)
            
            if outpainted_image is None:
                self.error_count += 1
                self.status = DiffSynthServiceStatus.READY
                
                # Try fallback if enabled
                if self.config.enable_fallback:
                    return self._fallback_outpaint_response(request)
                
                return ImageEditResponse(
                    success=False,
                    message="Outpainting failed",
                    error_details="Pipeline processing returned None",
                    suggested_fixes=[
                        "Check DiffSynth installation and dependencies",
                        "Verify image format and size compatibility",
                        "Try reducing outpainting area or complexity"
                    ]
                )
            
            # Postprocess outpainted image
            final_image = self.postprocessor.postprocess_edited_image(
                outpainted_image,
                original_image=input_image,
                enhance_quality=True
            )
            
            processing_time = time.time() - start_time
            self.last_operation_time = processing_time
            self.operation_count += 1
            
            # Save image and prepare response
            output_path = self._save_edited_image(final_image, request)
            
            # Create processing metrics
            metrics = ProcessingMetrics(
                operation_type="outpaint",
                processing_time=processing_time,
                gpu_memory_used=self.resource_usage.gpu_memory_allocated,
                cpu_memory_used=self.resource_usage.cpu_memory_used,
                input_resolution=(input_image.width, input_image.height),
                output_resolution=(final_image.width, final_image.height),
                tiled_processing_used=use_tiled
            )
            
            # Update resource usage
            self._update_resource_usage()
            
            # Auto cleanup if enabled
            if self.config.auto_cleanup:
                self._cleanup_resources()
            
            self.status = DiffSynthServiceStatus.READY
            
            logger.info(f"✅ Outpainting completed in {processing_time:.2f}s")
            
            return ImageEditResponse(
                success=True,
                message=f"Outpainting completed successfully in {processing_time:.2f}s",
                image_path=output_path,
                operation=EditOperation.OUTPAINT,
                processing_time=processing_time,
                parameters={
                    "prompt": request.prompt,
                    "direction": request.direction,
                    "pixels": request.pixels,
                    "num_inference_steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "strength": request.strength,
                    "seed": request.seed
                },
                resource_usage=metrics.dict()
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_pipeline_error(
                e, self.config.model_name, "DiffSynth", {"operation": "outpaint"}
            )
            self._log_error(error_info)
            self.status = DiffSynthServiceStatus.READY
            self.error_count += 1
            
            # Try fallback if enabled
            if self.config.enable_fallback:
                return self._fallback_outpaint_response(request)
            
            return ImageEditResponse(
                success=False,
                message="Outpainting failed",
                error_details=str(e),
                suggested_fixes=error_info.suggested_fixes if error_info else None
            )
    
    def _load_mask_from_request(self, request: "InpaintRequest") -> Optional[Image.Image]:
        """Load mask image from request (path or base64)"""
        try:
            if request.mask_path:
                return self._load_and_validate_mask(request.mask_path)
            elif request.mask_base64:
                from diffsynth_models import decode_base64_to_image
                mask = decode_base64_to_image(request.mask_base64)
                return self._load_and_validate_mask(mask)
            else:
                logger.error("No mask input provided in request")
                return None
        except Exception as e:
            logger.error(f"Failed to load mask from request: {e}")
            return None
    
    def _load_and_validate_mask(self, mask: Union[str, Image.Image]) -> Optional[Image.Image]:
        """Load and validate mask image"""
        try:
            if isinstance(mask, str):
                if os.path.exists(mask):
                    mask_image = Image.open(mask)
                else:
                    logger.error(f"Mask file not found: {mask}")
                    return None
            elif isinstance(mask, Image.Image):
                mask_image = mask
            else:
                logger.error(f"Unsupported mask type: {type(mask)}")
                return None
            
            # Convert to grayscale for mask processing
            if mask_image.mode != 'L':
                mask_image = mask_image.convert('L')
            
            # Validate mask properties
            if not self._validate_mask_properties(mask_image):
                return None
            
            return mask_image
            
        except Exception as e:
            logger.error(f"Failed to load and validate mask: {e}")
            return None
    
    def _validate_mask_properties(self, mask: Image.Image) -> bool:
        """Validate mask image properties"""
        try:
            # Check dimensions
            width, height = mask.size
            if width < 64 or height < 64:
                logger.error(f"Mask too small: {width}x{height} (minimum 64x64)")
                return False
            if width > 4096 or height > 4096:
                logger.error(f"Mask too large: {width}x{height} (maximum 4096x4096)")
                return False
            
            # Check if mask has valid content (not all black or all white)
            import numpy as np
            mask_array = np.array(mask)
            unique_values = np.unique(mask_array)
            
            if len(unique_values) < 2:
                logger.warning("Mask appears to be uniform (all same value)")
                # Allow it but warn - might be intentional
            
            # Check for reasonable mask coverage (not too small or too large)
            white_pixels = np.sum(mask_array > 127)
            total_pixels = mask_array.size
            coverage = white_pixels / total_pixels
            
            if coverage < 0.001:
                logger.warning(f"Very small mask coverage: {coverage:.1%}")
            elif coverage > 0.99:
                logger.warning(f"Very large mask coverage: {coverage:.1%}")
            
            logger.debug(f"✅ Mask validation passed: {width}x{height}, coverage: {coverage:.1%}")
            return True
            
        except Exception as e:
            logger.error(f"Mask validation failed: {e}")
            return False
    
    def _validate_mask_compatibility(self, image: Image.Image, mask: Image.Image) -> bool:
        """Validate that mask is compatible with input image"""
        try:
            # Check if dimensions match
            if image.size != mask.size:
                logger.error(f"Image size {image.size} doesn't match mask size {mask.size}")
                return False
            
            # Check aspect ratio consistency
            img_ratio = image.width / image.height
            mask_ratio = mask.width / mask.height
            
            if abs(img_ratio - mask_ratio) > 0.01:  # Allow small floating point differences
                logger.error(f"Aspect ratio mismatch: image {img_ratio:.3f}, mask {mask_ratio:.3f}")
                return False
            
            logger.debug("✅ Mask compatibility validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Mask compatibility validation failed: {e}")
            return False
    
    def _preprocess_mask(
        self, 
        mask: Image.Image, 
        target_size: Tuple[int, int],
        blur_radius: int = 4,
        invert: bool = False
    ) -> Image.Image:
        """Preprocess mask for inpainting"""
        try:
            # Resize mask to match target size if needed
            if mask.size != target_size:
                mask = mask.resize(target_size, Image.Resampling.LANCZOS)
                logger.debug(f"Resized mask to {target_size}")
            
            # Apply blur to soften mask edges
            if blur_radius > 0:
                from PIL import ImageFilter
                mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                logger.debug(f"Applied blur with radius {blur_radius}")
            
            # Invert mask if requested
            if invert:
                from PIL import ImageOps
                mask = ImageOps.invert(mask)
                logger.debug("Inverted mask")
            
            # Ensure mask is in correct format (0-255 grayscale)
            import numpy as np
            mask_array = np.array(mask)
            
            # Normalize to 0-255 range
            if mask_array.max() <= 1.0:
                mask_array = (mask_array * 255).astype(np.uint8)
            
            # Apply threshold to create clean binary mask
            mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
            
            processed_mask = Image.fromarray(mask_array, mode='L')
            logger.debug("✅ Mask preprocessing completed")
            
            return processed_mask
            
        except Exception as e:
            logger.error(f"Mask preprocessing failed: {e}")
            return mask  # Return original mask as fallback
    
    def _inpaint_image_single(
        self, 
        request: "InpaintRequest", 
        image: Image.Image, 
        mask: Image.Image
    ) -> Optional[Image.Image]:
        """Perform inpainting using single pass (non-tiled)"""
        try:
            # Prepare parameters for DiffSynth inpainting
            inpaint_params = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt or "",
                "image": image,
                "mask_image": mask,
                "num_inference_steps": request.num_inference_steps or self.config.default_num_inference_steps,
                "guidance_scale": request.guidance_scale or self.config.default_guidance_scale,
                "strength": request.strength or 0.9,  # Higher default strength for inpainting
                "seed": request.seed or 42,
            }
            
            # Add any additional parameters
            if request.additional_params:
                inpaint_params.update(request.additional_params)
            
            logger.debug(f"Inpainting with parameters: {list(inpaint_params.keys())}")
            
            # Perform inpainting
            inpainted_image = self.pipeline(**inpaint_params)
            
            logger.debug("✅ Single pass inpainting completed")
            return inpainted_image
            
        except Exception as e:
            logger.error(f"Single pass inpainting failed: {e}")
            return None
    
    def _inpaint_image_tiled(
        self, 
        request: "InpaintRequest", 
        image: Image.Image, 
        mask: Image.Image
    ) -> Optional[Image.Image]:
        """Perform inpainting using tiled processing"""
        try:
            logger.info("🔧 Using tiled processing for large image inpainting")
            
            # Calculate tiles
            tile_coords = self.tiled_processor.calculate_tiles(image)
            inpainted_tiles = []
            
            for i, (x1, y1, x2, y2) in enumerate(tile_coords):
                logger.debug(f"Processing inpaint tile {i+1}/{len(tile_coords)}: {x1},{y1} to {x2},{y2}")
                
                # Extract tile and corresponding mask tile
                image_tile = image.crop((x1, y1, x2, y2))
                mask_tile = mask.crop((x1, y1, x2, y2))
                
                # Check if this tile has any mask content
                import numpy as np
                mask_array = np.array(mask_tile)
                if np.sum(mask_array > 127) == 0:
                    # No mask content in this tile, use original
                    logger.debug(f"Tile {i+1} has no mask content, using original")
                    inpainted_tiles.append(image_tile)
                    continue
                
                # Create tile-specific request (pass dummy data to satisfy validation)
                tile_request = InpaintRequest(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    image_base64="dummy",  # Dummy data for validation
                    mask_base64="dummy",   # Dummy data for validation
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    strength=request.strength,
                    seed=request.seed,
                    additional_params=request.additional_params
                )
                
                # Inpaint tile
                inpainted_tile = self._inpaint_image_single(tile_request, image_tile, mask_tile)
                if inpainted_tile is None:
                    logger.warning(f"Failed to inpaint tile {i+1}, using original")
                    inpainted_tile = image_tile
                
                inpainted_tiles.append(inpainted_tile)
            
            # Merge tiles
            merged_image = self.tiled_processor.merge_tiles(
                inpainted_tiles, tile_coords, image.size
            )
            
            logger.info(f"✅ Tiled inpainting completed with {len(tile_coords)} tiles")
            return merged_image
            
        except Exception as e:
            logger.error(f"Tiled inpainting failed: {e}")
            # Fallback to single pass
            return self._inpaint_image_single(request, image, mask)
    
    def _fallback_inpaint_response(self, request: "InpaintRequest") -> ImageEditResponse:
        """Fallback response when DiffSynth inpainting fails"""
        logger.info("🔄 Attempting fallback for inpainting...")
        
        return ImageEditResponse(
            success=False,
            message="DiffSynth inpainting failed. Consider using basic image editing instead.",
            error_details="DiffSynth inpainting pipeline unavailable",
            suggested_fixes=[
                "Check DiffSynth installation and inpainting support",
                "Verify mask format and compatibility",
                "Try reducing image or mask complexity",
                "Check GPU memory availability for inpainting",
                "Ensure mask has proper contrast and coverage"
            ]
        )

    def _fallback_edit_image_response(self, request: ImageEditRequest) -> ImageEditResponse:
        """Fallback response when DiffSynth editing fails"""
        logger.info("🔄 Attempting fallback image editing...")
        
        return ImageEditResponse(
            success=False,
            message="DiffSynth editing failed. Consider using text-to-image generation instead.",
            error_details="DiffSynth pipeline unavailable",
            suggested_fixes=[
                "Check DiffSynth installation and dependencies",
                "Try reducing image size or complexity",
                "Use text-to-image generation as alternative",
                "Check GPU memory availability"
            ]
        )
    
    def _update_resource_usage(self) -> None:
        """Update resource usage tracking"""
        try:
            if torch.cuda.is_available():
                self.resource_usage.gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
                self.resource_usage.gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9
            
            import psutil
            process = psutil.Process()
            self.resource_usage.cpu_memory_used = process.memory_info().rss / 1e9
            self.resource_usage.last_updated = time.time()
            
        except Exception as e:
            logger.warning(f"Failed to update resource usage: {e}")
    
    def _cleanup_resources(self) -> None:
        """Cleanup GPU resources"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.debug("🧹 Resources cleaned up")
        except Exception as e:
            logger.warning(f"Resource cleanup failed: {e}")
    
    def _log_error(self, error_info: ErrorInfo) -> None:
        """Log error information"""
        logger.error(f"DiffSynth Error [{error_info.category.value}]: {error_info.message}")
        logger.error(f"Details: {error_info.details}")
        if error_info.suggested_fixes:
            logger.info("Suggested fixes:")
            for fix in error_info.suggested_fixes[:3]:  # Show top 3 fixes
                logger.info(f"  • {fix}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status and metrics"""
        return {
            "status": self.status.value,
            "initialized": self.status == DiffSynthServiceStatus.READY,
            "model_name": self.config.model_name,
            "device": self.config.device,
            "initialization_time": self.initialization_time,
            "last_operation_time": self.last_operation_time,
            "operation_count": self.operation_count,
            "error_count": self.error_count,
            "resource_usage": {
                "gpu_memory_allocated_gb": self.resource_usage.gpu_memory_allocated,
                "gpu_memory_reserved_gb": self.resource_usage.gpu_memory_reserved,
                "cpu_memory_used_gb": self.resource_usage.cpu_memory_used,
                "last_updated": self.resource_usage.last_updated,
            },
            "config": {
                "enable_vram_management": self.config.enable_vram_management,
                "enable_cpu_offload": self.config.enable_cpu_offload,
                "use_tiled_processing": self.config.use_tiled_processing,
                "max_memory_usage_gb": self.config.max_memory_usage_gb,
            }
        }
    
    def _expand_canvas_for_outpainting(
        self, 
        image: Image.Image, 
        direction: str, 
        pixels: int, 
        fill_mode: str = "edge"
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """
        Expand canvas for outpainting and create corresponding mask
        
        Args:
            image: Input image
            direction: Outpainting direction (left, right, top, bottom, all)
            pixels: Number of pixels to extend
            fill_mode: Fill mode for new areas (edge, constant, reflect)
            
        Returns:
            Tuple of (expanded_image, mask) or (None, None) if failed
        """
        try:
            from diffsynth_models import OutpaintDirection
            
            original_width, original_height = image.size
            
            # Calculate new dimensions based on direction
            if direction == OutpaintDirection.LEFT:
                new_width = original_width + pixels
                new_height = original_height
                paste_x = pixels
                paste_y = 0
            elif direction == OutpaintDirection.RIGHT:
                new_width = original_width + pixels
                new_height = original_height
                paste_x = 0
                paste_y = 0
            elif direction == OutpaintDirection.TOP:
                new_width = original_width
                new_height = original_height + pixels
                paste_x = 0
                paste_y = pixels
            elif direction == OutpaintDirection.BOTTOM:
                new_width = original_width
                new_height = original_height + pixels
                paste_x = 0
                paste_y = 0
            elif direction == OutpaintDirection.ALL:
                new_width = original_width + (pixels * 2)
                new_height = original_height + (pixels * 2)
                paste_x = pixels
                paste_y = pixels
            else:
                logger.error(f"Invalid outpainting direction: {direction}")
                return None, None
            
            # Create expanded canvas
            expanded_image = self._create_expanded_canvas(
                image, new_width, new_height, paste_x, paste_y, fill_mode
            )
            
            # Create mask for outpainting areas (white = areas to fill, black = preserve)
            mask = self._create_outpainting_mask(
                original_width, original_height, new_width, new_height, paste_x, paste_y
            )
            
            logger.debug(f"✅ Canvas expanded from {original_width}x{original_height} to {new_width}x{new_height}")
            return expanded_image, mask
            
        except Exception as e:
            logger.error(f"Canvas expansion failed: {e}")
            return None, None
    
    def _create_expanded_canvas(
        self, 
        image: Image.Image, 
        new_width: int, 
        new_height: int, 
        paste_x: int, 
        paste_y: int, 
        fill_mode: str
    ) -> Image.Image:
        """Create expanded canvas with specified fill mode"""
        try:
            # Create new canvas
            expanded_image = Image.new("RGB", (new_width, new_height), (128, 128, 128))
            
            # Apply fill mode for background
            if fill_mode == "edge":
                # Extend edges of the original image
                expanded_image = self._fill_with_edge_extension(
                    image, expanded_image, paste_x, paste_y
                )
            elif fill_mode == "reflect":
                # Reflect the image content
                expanded_image = self._fill_with_reflection(
                    image, expanded_image, paste_x, paste_y
                )
            elif fill_mode == "constant":
                # Use constant color (already set to gray)
                pass
            else:
                logger.warning(f"Unknown fill mode: {fill_mode}, using edge extension")
                expanded_image = self._fill_with_edge_extension(
                    image, expanded_image, paste_x, paste_y
                )
            
            # Paste original image on top
            expanded_image.paste(image, (paste_x, paste_y))
            
            return expanded_image
            
        except Exception as e:
            logger.error(f"Failed to create expanded canvas: {e}")
            # Fallback: simple paste on gray background
            expanded_image = Image.new("RGB", (new_width, new_height), (128, 128, 128))
            expanded_image.paste(image, (paste_x, paste_y))
            return expanded_image
    
    def _fill_with_edge_extension(
        self, 
        image: Image.Image, 
        canvas: Image.Image, 
        paste_x: int, 
        paste_y: int
    ) -> Image.Image:
        """Fill canvas with edge extension"""
        try:
            import numpy as np
            
            # Convert to numpy for easier manipulation
            img_array = np.array(image)
            canvas_array = np.array(canvas)
            
            # Get dimensions
            img_h, img_w = img_array.shape[:2]
            canvas_h, canvas_w = canvas_array.shape[:2]
            
            # Fill left area
            if paste_x > 0:
                left_edge = img_array[:, 0:1]  # First column
                for x in range(paste_x):
                    canvas_array[paste_y:paste_y+img_h, x:x+1] = left_edge
            
            # Fill right area
            if paste_x + img_w < canvas_w:
                right_edge = img_array[:, -1:]  # Last column
                for x in range(paste_x + img_w, canvas_w):
                    canvas_array[paste_y:paste_y+img_h, x:x+1] = right_edge
            
            # Fill top area
            if paste_y > 0:
                top_edge = img_array[0:1, :]  # First row
                for y in range(paste_y):
                    canvas_array[y:y+1, paste_x:paste_x+img_w] = top_edge
            
            # Fill bottom area
            if paste_y + img_h < canvas_h:
                bottom_edge = img_array[-1:, :]  # Last row
                for y in range(paste_y + img_h, canvas_h):
                    canvas_array[y:y+1, paste_x:paste_x+img_w] = bottom_edge
            
            # Fill corners with corner pixels
            if paste_x > 0 and paste_y > 0:
                # Top-left corner
                corner_color = img_array[0, 0]
                canvas_array[:paste_y, :paste_x] = corner_color
            
            if paste_x + img_w < canvas_w and paste_y > 0:
                # Top-right corner
                corner_color = img_array[0, -1]
                canvas_array[:paste_y, paste_x+img_w:] = corner_color
            
            if paste_x > 0 and paste_y + img_h < canvas_h:
                # Bottom-left corner
                corner_color = img_array[-1, 0]
                canvas_array[paste_y+img_h:, :paste_x] = corner_color
            
            if paste_x + img_w < canvas_w and paste_y + img_h < canvas_h:
                # Bottom-right corner
                corner_color = img_array[-1, -1]
                canvas_array[paste_y+img_h:, paste_x+img_w:] = corner_color
            
            return Image.fromarray(canvas_array.astype(np.uint8))
            
        except Exception as e:
            logger.warning(f"Edge extension failed: {e}")
            return canvas
    
    def _fill_with_reflection(
        self, 
        image: Image.Image, 
        canvas: Image.Image, 
        paste_x: int, 
        paste_y: int
    ) -> Image.Image:
        """Fill canvas with reflection (simplified implementation)"""
        try:
            # For simplicity, use edge extension as fallback
            # A full reflection implementation would be more complex
            return self._fill_with_edge_extension(image, canvas, paste_x, paste_y)
            
        except Exception as e:
            logger.warning(f"Reflection fill failed: {e}")
            return canvas
    
    def _create_outpainting_mask(
        self, 
        original_width: int, 
        original_height: int, 
        new_width: int, 
        new_height: int, 
        paste_x: int, 
        paste_y: int
    ) -> Image.Image:
        """Create mask for outpainting (white = fill areas, black = preserve)"""
        try:
            # Create mask (black background)
            mask = Image.new("L", (new_width, new_height), 0)
            
            # Create white areas for regions to be filled
            import numpy as np
            mask_array = np.array(mask)
            
            # Mark areas outside the original image as white (to be filled)
            # Left area
            if paste_x > 0:
                mask_array[:, :paste_x] = 255
            
            # Right area
            if paste_x + original_width < new_width:
                mask_array[:, paste_x + original_width:] = 255
            
            # Top area
            if paste_y > 0:
                mask_array[:paste_y, :] = 255
            
            # Bottom area
            if paste_y + original_height < new_height:
                mask_array[paste_y + original_height:, :] = 255
            
            return Image.fromarray(mask_array)
            
        except Exception as e:
            logger.error(f"Failed to create outpainting mask: {e}")
            # Fallback: create simple mask
            mask = Image.new("L", (new_width, new_height), 255)
            return mask
    
    def _outpaint_image_single(
        self, 
        request: "OutpaintRequest", 
        image: Image.Image, 
        mask: Image.Image
    ) -> Optional[Image.Image]:
        """Perform outpainting using single pass (non-tiled)"""
        try:
            # Use inpainting pipeline for outpainting with expanded canvas
            outpaint_params = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt or "",
                "image": image,
                "mask_image": mask,
                "num_inference_steps": request.num_inference_steps or self.config.default_num_inference_steps,
                "guidance_scale": request.guidance_scale or self.config.default_guidance_scale,
                "strength": request.strength or 0.8,  # Slightly lower strength for outpainting
                "seed": request.seed or 42,
            }
            
            # Add any additional parameters
            if request.additional_params:
                outpaint_params.update(request.additional_params)
            
            logger.debug(f"Outpainting with parameters: {list(outpaint_params.keys())}")
            
            # Perform outpainting using inpainting pipeline
            outpainted_image = self.pipeline(**outpaint_params)
            
            logger.debug("✅ Single pass outpainting completed")
            return outpainted_image
            
        except Exception as e:
            logger.error(f"Single pass outpainting failed: {e}")
            return None
    
    def _outpaint_image_tiled(
        self, 
        request: "OutpaintRequest", 
        image: Image.Image, 
        mask: Image.Image
    ) -> Optional[Image.Image]:
        """Perform outpainting using tiled processing"""
        try:
            logger.info("🔧 Using tiled processing for large image outpainting")
            
            # Calculate tiles
            tile_coords = self.tiled_processor.calculate_tiles(image)
            outpainted_tiles = []
            
            for i, (x1, y1, x2, y2) in enumerate(tile_coords):
                logger.debug(f"Processing outpaint tile {i+1}/{len(tile_coords)}: {x1},{y1} to {x2},{y2}")
                
                # Extract tile and corresponding mask tile
                image_tile = image.crop((x1, y1, x2, y2))
                mask_tile = mask.crop((x1, y1, x2, y2))
                
                # Check if this tile has any mask content
                import numpy as np
                mask_array = np.array(mask_tile)
                if np.sum(mask_array > 127) == 0:
                    # No mask content in this tile, use original
                    logger.debug(f"Tile {i+1} has no mask content, using original")
                    outpainted_tiles.append(image_tile)
                    continue
                
                # Create tile-specific request
                from diffsynth_models import OutpaintRequest
                tile_request = OutpaintRequest(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    image_base64="dummy",  # Dummy data for validation
                    direction=request.direction,
                    pixels=request.pixels,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    strength=request.strength,
                    seed=request.seed,
                    additional_params=request.additional_params
                )
                
                # Outpaint tile
                outpainted_tile = self._outpaint_image_single(tile_request, image_tile, mask_tile)
                if outpainted_tile is None:
                    logger.warning(f"Failed to outpaint tile {i+1}, using original")
                    outpainted_tile = image_tile
                
                outpainted_tiles.append(outpainted_tile)
            
            # Merge tiles
            merged_image = self.tiled_processor.merge_tiles(
                outpainted_tiles, tile_coords, image.size
            )
            
            logger.info(f"✅ Tiled outpainting completed with {len(tile_coords)} tiles")
            return merged_image
            
        except Exception as e:
            logger.error(f"Tiled outpainting failed: {e}")
            # Fallback to single pass
            return self._outpaint_image_single(request, image, mask)
    
    def _fallback_outpaint_response(self, request: "OutpaintRequest") -> ImageEditResponse:
        """Fallback response when DiffSynth outpainting fails"""
        logger.info("🔄 Attempting fallback for outpainting...")
        
        return ImageEditResponse(
            success=False,
            message="DiffSynth outpainting failed. Consider using manual image extension instead.",
            error_details="DiffSynth outpainting pipeline unavailable",
            suggested_fixes=[
                "Check DiffSynth installation and outpainting support",
                "Verify image format and size compatibility",
                "Try reducing outpainting area or complexity",
                "Check GPU memory availability for outpainting",
                "Consider using smaller pixel extension values"
            ]
        )
    
    def style_transfer(self, request: "StyleTransferRequest") -> ImageEditResponse:
        """
        Perform style transfer using DiffSynth pipeline with style image support
        
        Args:
            request: StyleTransferRequest with content image, style image, and parameters
            
        Returns:
            ImageEditResponse with style transfer result
        """
        if self.status != DiffSynthServiceStatus.READY:
            if not self.initialize():
                return ImageEditResponse(
                    success=False,
                    message="DiffSynth service not available",
                    error_details="Service initialization failed"
                )
        
        self.status = DiffSynthServiceStatus.BUSY
        start_time = time.time()
        
        try:
            # Update resource usage
            self._update_resource_usage()
            
            # Load and validate content image
            content_image = self._load_image_from_request(request)
            if content_image is None:
                return ImageEditResponse(
                    success=False,
                    message="Failed to load content image",
                    error_details="Invalid content image input or unsupported format"
                )
            
            # Load and validate style image
            style_image = self._load_style_image_from_request(request)
            if style_image is None:
                return ImageEditResponse(
                    success=False,
                    message="Failed to load style image",
                    error_details="Invalid style image input or unsupported format"
                )
            
            # Preprocess content and style images
            processed_content = self.preprocessor.prepare_image_for_editing(
                content_image,
                target_width=request.width,
                target_height=request.height
            )
            
            processed_style = self.preprocessor.prepare_image_for_editing(
                style_image,
                target_width=processed_content.width,
                target_height=processed_content.height
            )
            
            # Estimate processing time
            estimated_time = estimate_processing_time(processed_content, "style_transfer")
            logger.info(f"🎨 Starting style transfer (estimated: {estimated_time:.1f}s): {request.prompt[:50]}...")
            
            # Check if tiled processing should be used
            use_tiled = request.use_tiled_processing
            if use_tiled is None:
                use_tiled = self.tiled_processor.should_use_tiled_processing(processed_content)
            
            # Perform style transfer
            if use_tiled and self.tiled_processor.should_use_tiled_processing(processed_content):
                styled_image = self._style_transfer_tiled(request, processed_content, processed_style)
            else:
                styled_image = self._style_transfer_single(request, processed_content, processed_style)
            
            if styled_image is None:
                self.error_count += 1
                self.status = DiffSynthServiceStatus.READY
                
                # Try fallback if enabled
                if self.config.enable_fallback:
                    return self._fallback_style_transfer_response(request)
                
                return ImageEditResponse(
                    success=False,
                    message="Style transfer failed",
                    error_details="Pipeline processing returned None",
                    suggested_fixes=[
                        "Check DiffSynth installation and dependencies",
                        "Verify style image format and compatibility",
                        "Try reducing style strength or image complexity",
                        "Ensure sufficient GPU memory for style transfer"
                    ]
                )
            
            # Postprocess styled image
            final_image = self.postprocessor.postprocess_edited_image(
                styled_image,
                original_image=content_image,
                enhance_quality=True
            )
            
            processing_time = time.time() - start_time
            self.last_operation_time = processing_time
            self.operation_count += 1
            
            # Save image and prepare response
            output_path = self._save_edited_image(final_image, request)
            
            # Create processing metrics
            metrics = ProcessingMetrics(
                operation_type="style_transfer",
                processing_time=processing_time,
                gpu_memory_used=self.resource_usage.gpu_memory_allocated,
                cpu_memory_used=self.resource_usage.cpu_memory_used,
                input_resolution=(content_image.width, content_image.height),
                output_resolution=(final_image.width, final_image.height),
                tiled_processing_used=use_tiled
            )
            
            # Update resource usage
            self._update_resource_usage()
            
            # Auto cleanup if enabled
            if self.config.auto_cleanup:
                self._cleanup_resources()
            
            self.status = DiffSynthServiceStatus.READY
            
            logger.info(f"✅ Style transfer completed in {processing_time:.2f}s")
            
            return ImageEditResponse(
                success=True,
                message=f"Style transfer completed successfully in {processing_time:.2f}s",
                image_path=output_path,
                operation=EditOperation.STYLE_TRANSFER,
                processing_time=processing_time,
                parameters={
                    "prompt": request.prompt,
                    "num_inference_steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "style_strength": request.style_strength,
                    "content_strength": request.content_strength,
                    "seed": request.seed
                },
                resource_usage=metrics.dict()
            )
            
        except Exception as e:
            error_info = self.error_handler.handle_pipeline_error(
                e, self.config.model_name, "DiffSynth", {"operation": "style_transfer"}
            )
            self._log_error(error_info)
            self.status = DiffSynthServiceStatus.READY
            self.error_count += 1
            
            # Try fallback if enabled
            if self.config.enable_fallback:
                return self._fallback_style_transfer_response(request)
            
            return ImageEditResponse(
                success=False,
                message="Style transfer failed",
                error_details=str(e),
                suggested_fixes=error_info.suggested_fixes if error_info else None
            )
    
    def _load_style_image_from_request(self, request: "StyleTransferRequest") -> Optional[Image.Image]:
        """Load style image from request (path or base64)"""
        try:
            if request.style_image_path:
                return self.preprocessor.load_and_validate_image(request.style_image_path)
            elif request.style_image_base64:
                from diffsynth_models import decode_base64_to_image
                style_image = decode_base64_to_image(request.style_image_base64)
                return self.preprocessor.load_and_validate_image(style_image)
            else:
                logger.error("No style image input provided in request")
                return None
        except Exception as e:
            logger.error(f"Failed to load style image from request: {e}")
            return None
    
    def _style_transfer_single(self, request: "StyleTransferRequest", content_image: Image.Image, style_image: Image.Image) -> Optional[Image.Image]:
        """Perform style transfer using single pass (non-tiled)"""
        try:
            # Create style-aware prompt
            style_prompt = self._create_style_transfer_prompt(request.prompt, request.style_strength, request.content_strength)
            
            # Perform style transfer using DiffSynth pipeline
            # Note: This uses the edit pipeline with style conditioning
            styled_image = self.pipeline(
                prompt=style_prompt,
                negative_prompt=request.negative_prompt or "blurry, low quality, distorted",
                image=content_image,
                num_inference_steps=request.num_inference_steps or self.config.default_num_inference_steps,
                guidance_scale=request.guidance_scale or self.config.default_guidance_scale,
                strength=request.style_strength or 0.7,
                seed=request.seed or 42,
                # Additional style transfer parameters
                **(request.additional_params or {})
            )
            
            # Blend with original content based on content_strength
            if request.content_strength and request.content_strength > 0:
                styled_image = self._blend_style_and_content(
                    styled_image, content_image, request.content_strength
                )
            
            return styled_image
            
        except Exception as e:
            logger.error(f"Single pass style transfer failed: {e}")
            return None
    
    def _style_transfer_tiled(self, request: "StyleTransferRequest", content_image: Image.Image, style_image: Image.Image) -> Optional[Image.Image]:
        """Perform style transfer using tiled processing"""
        try:
            logger.info("🔧 Using tiled processing for large image style transfer")
            
            # Calculate tiles for content image
            tile_coords = self.tiled_processor.calculate_tiles(content_image)
            styled_tiles = []
            
            for i, (x1, y1, x2, y2) in enumerate(tile_coords):
                logger.debug(f"Processing style transfer tile {i+1}/{len(tile_coords)}: {x1},{y1} to {x2},{y2}")
                
                # Extract content tile
                content_tile = content_image.crop((x1, y1, x2, y2))
                
                # Resize style image to match tile size
                style_tile = style_image.resize(content_tile.size, Image.Resampling.LANCZOS)
                
                # Perform style transfer on tile
                styled_tile = self._style_transfer_single(request, content_tile, style_tile)
                if styled_tile is None:
                    logger.warning(f"Failed to style transfer tile {i+1}, using original")
                    styled_tile = content_tile
                
                styled_tiles.append(styled_tile)
            
            # Merge tiles
            merged_image = self.tiled_processor.merge_tiles(
                styled_tiles, tile_coords, content_image.size
            )
            
            logger.info(f"✅ Tiled style transfer completed with {len(tile_coords)} tiles")
            return merged_image
            
        except Exception as e:
            logger.error(f"Tiled style transfer failed: {e}")
            # Fallback to single pass
            return self._style_transfer_single(request, content_image, style_image)
    
    def _create_style_transfer_prompt(self, base_prompt: str, style_strength: float, content_strength: float) -> str:
        """Create enhanced prompt for style transfer"""
        try:
            # Enhance prompt based on style and content strengths
            if style_strength > 0.8:
                style_emphasis = "in the artistic style of, heavily stylized, artistic interpretation"
            elif style_strength > 0.5:
                style_emphasis = "with artistic style applied, stylized"
            else:
                style_emphasis = "with subtle artistic influence"
            
            if content_strength > 0.7:
                content_emphasis = "preserving original details and structure"
            elif content_strength > 0.4:
                content_emphasis = "maintaining key features"
            else:
                content_emphasis = "allowing creative interpretation"
            
            enhanced_prompt = f"{base_prompt}, {style_emphasis}, {content_emphasis}, high quality, detailed"
            
            return enhanced_prompt
            
        except Exception as e:
            logger.warning(f"Failed to create enhanced style transfer prompt: {e}")
            return base_prompt
    
    def _blend_style_and_content(self, styled_image: Image.Image, content_image: Image.Image, content_strength: float) -> Image.Image:
        """Blend styled image with original content based on content strength"""
        try:
            from PIL import ImageChops
            
            # Ensure images are the same size
            if styled_image.size != content_image.size:
                content_image = content_image.resize(styled_image.size, Image.Resampling.LANCZOS)
            
            # Calculate blend ratio (content_strength determines how much original content to preserve)
            style_ratio = 1.0 - content_strength
            content_ratio = content_strength
            
            # Blend images
            blended = Image.blend(styled_image, content_image, content_ratio)
            
            return blended
            
        except Exception as e:
            logger.warning(f"Failed to blend style and content: {e}")
            return styled_image
    
    def _fallback_style_transfer_response(self, request: "StyleTransferRequest") -> ImageEditResponse:
        """Fallback response when DiffSynth style transfer fails"""
        logger.info("🔄 Attempting fallback for style transfer...")
        
        return ImageEditResponse(
            success=False,
            message="DiffSynth style transfer failed. Consider using alternative style transfer methods.",
            error_details="DiffSynth style transfer pipeline unavailable",
            suggested_fixes=[
                "Check DiffSynth installation and style transfer support",
                "Verify both content and style image formats",
                "Try reducing style strength or image complexity",
                "Check GPU memory availability for style transfer",
                "Consider using smaller images or different style references",
                "Ensure style and content images are compatible formats"
            ]
        )
    
    def shutdown(self) -> None:
        """Shutdown the service and cleanup resources"""
        logger.info("🔄 Shutting down DiffSynth service...")
        
        self.status = DiffSynthServiceStatus.OFFLINE
        
        # Cleanup pipeline
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        # Cleanup GPU memory
        self._cleanup_resources()
        
        logger.info("✅ DiffSynth service shutdown complete")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        if hasattr(self, 'status') and self.status != DiffSynthServiceStatus.OFFLINE:
            self.shutdown()


# Factory function for easy service creation
def create_diffsynth_service(
    model_name: str = "Qwen/Qwen-Image-Edit",
    device: Optional[str] = None,
    enable_optimizations: bool = True,
    **kwargs
) -> DiffSynthService:
    """
    Factory function to create DiffSynth service with optimal configuration
    
    Args:
        model_name: Model to use for DiffSynth
        device: Device to use (auto-detected if None)
        enable_optimizations: Whether to enable memory optimizations
        **kwargs: Additional configuration options
        
    Returns:
        Configured DiffSynthService instance
    """
    config = DiffSynthConfig(
        model_name=model_name,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs
    )
    
    if not enable_optimizations:
        config.enable_vram_management = False
        config.enable_cpu_offload = False
        config.use_tiled_processing = False
    
    return DiffSynthService(config)