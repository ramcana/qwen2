"""
Pipeline Optimizer for Modern GPU Performance with MMDiT Architecture
Optimizes Qwen-Image generation pipelines for RTX 4080 and modern hardware
Includes modern attention and memory optimizations
"""

import logging
import torch
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from diffusers import DiffusionPipeline, AutoPipelineForText2Image
from diffusers.pipelines.auto_pipeline import AUTO_TEXT2IMAGE_PIPELINES_MAPPING

# Import error handling system
try:
    from .error_handler import ArchitectureAwareErrorHandler, ErrorCategory
    ERROR_HANDLER_AVAILABLE = True
except ImportError:
    try:
        from error_handler import ArchitectureAwareErrorHandler, ErrorCategory
        ERROR_HANDLER_AVAILABLE = True
    except ImportError:
        ERROR_HANDLER_AVAILABLE = False

# Import attention optimization system
try:
    from .attention_optimizer import AttentionOptimizer, create_attention_config
    ATTENTION_OPTIMIZER_AVAILABLE = True
except ImportError:
    try:
        from attention_optimizer import AttentionOptimizer, create_attention_config
        ATTENTION_OPTIMIZER_AVAILABLE = True
    except ImportError:
        ATTENTION_OPTIMIZER_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for pipeline optimization"""
    # Model settings
    torch_dtype: torch.dtype = torch.float16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Memory optimization settings (disabled for high-VRAM GPUs)
    enable_attention_slicing: bool = False
    enable_vae_slicing: bool = False
    enable_vae_tiling: bool = False
    enable_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    low_cpu_mem_usage: bool = False
    
    # Performance optimization settings
    enable_tf32: bool = True
    enable_cudnn_benchmark: bool = True
    enable_scaled_dot_product_attention: bool = True
    enable_memory_efficient_attention: bool = True
    enable_flash_attention: bool = False  # Disabled for Qwen-Image compatibility
    
    # Generation settings for MMDiT architecture
    optimal_steps: int = 20
    optimal_cfg_scale: float = 3.5
    optimal_width: int = 1024
    optimal_height: int = 1024
    
    # Architecture-specific settings
    architecture_type: str = "MMDiT"  # "MMDiT" or "UNet"
    disable_gradient_computation: bool = True
    use_torch_compile: bool = False  # Disabled due to tensor unpacking issues
    
    # Modern attention optimizations
    enable_attention_optimizations: bool = True
    attention_optimization_level: str = "balanced"  # "ultra_fast", "balanced", "quality", "experimental"
    enable_dynamic_batch_sizing: bool = True
    enable_memory_efficient_attention: bool = True


class PipelineOptimizer:
    """Optimizes diffusion pipelines for modern GPU performance with MMDiT architecture"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.device = self.config.device
        
        # Initialize error handler
        self.error_handler = ArchitectureAwareErrorHandler() if ERROR_HANDLER_AVAILABLE else None
        if self.error_handler:
            self.error_handler.add_user_feedback_callback(self._handle_error_feedback)
        
        # Initialize attention optimizer
        self.attention_optimizer = None
        if ATTENTION_OPTIMIZER_AVAILABLE and self.config.enable_attention_optimizations:
            try:
                attention_config = create_attention_config(
                    architecture=self.config.architecture_type,
                    optimization_level=self.config.attention_optimization_level,
                    enable_dynamic_batch_sizing=self.config.enable_dynamic_batch_sizing,
                    use_memory_efficient_attention=self.config.enable_memory_efficient_attention,
                    enable_torch_compile=self.config.use_torch_compile
                )
                self.attention_optimizer = AttentionOptimizer(attention_config)
                logger.info("✅ Attention optimizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize attention optimizer: {e}")
                self.attention_optimizer = None
        
        # Validate device availability
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
            self.config.device = "cpu"
        
        logger.info(f"PipelineOptimizer initialized for device: {self.device}")
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name()
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name}, VRAM: {total_memory:.1f}GB")
    
    def _handle_error_feedback(self, message: str):
        """Handle error feedback from error handler"""
        logger.info(f"🔧 Pipeline Error Handler: {message}")
    
    def create_optimized_pipeline(
        self, 
        model_path: str, 
        architecture_type: str = "MMDiT"
    ) -> Union[DiffusionPipeline, AutoPipelineForText2Image]:
        """
        Create an optimized pipeline configured for modern GPU performance
        
        Args:
            model_path: Path to the model (local or HuggingFace model ID)
            architecture_type: Model architecture type ("MMDiT" or "UNet")
            
        Returns:
            Optimized pipeline instance
        """
        logger.info(f"🚀 Creating optimized pipeline for {architecture_type} architecture")
        logger.info(f"Model: {model_path}")
        
        # Update architecture type in config
        self.config.architecture_type = architecture_type
        
        try:
            # Choose appropriate pipeline class based on architecture and use case
            pipeline_class = self._select_pipeline_class(model_path, architecture_type)
            logger.info(f"Using pipeline class: {pipeline_class.__name__}")
            
            # Prepare loading arguments
            loading_kwargs = self._prepare_loading_kwargs()
            
            # Load the pipeline
            logger.info("📥 Loading pipeline with optimized configuration...")
            pipeline = pipeline_class.from_pretrained(model_path, **loading_kwargs)
            
            # Move to device
            if self.device == "cuda":
                logger.info(f"🔄 Moving pipeline to {self.device}")
                pipeline = pipeline.to(self.device)
            
            # Apply optimizations
            self._apply_gpu_optimizations(pipeline)
            self._configure_memory_settings(pipeline)
            self._setup_attention_processors(pipeline, architecture_type)
            
            # Apply modern attention optimizations
            if self.attention_optimizer:
                logger.info("🚀 Applying modern attention optimizations...")
                attention_success = self.attention_optimizer.optimize_pipeline_attention(pipeline)
                if attention_success:
                    logger.info("✅ Modern attention optimizations applied")
                else:
                    logger.warning("⚠️ Some attention optimizations failed")
            
            logger.info("✅ Pipeline optimization complete")
            return pipeline
            
        except Exception as e:
            logger.error(f"❌ Failed to create optimized pipeline: {e}")
            
            # Use error handler if available
            if self.error_handler:
                error_info = self.error_handler.handle_pipeline_error(
                    e, model_path, architecture_type, 
                    {"config": self.config, "loading_kwargs": loading_kwargs}
                )
                self.error_handler.log_error(error_info)
                
                # Attempt recovery
                recovery_success = self.error_handler.execute_recovery_actions(error_info)
                if recovery_success:
                    logger.info("🔄 Recovery actions completed, applying fallback configuration...")
                    
                    # Apply fallback configuration
                    fallback_config = self.error_handler._apply_architecture_fallback(model_path, architecture_type)
                    
                    # Try loading with fallback configuration
                    try:
                        logger.info("🔄 Retrying pipeline creation with fallback configuration...")
                        fallback_kwargs = loading_kwargs.copy()
                        fallback_kwargs.update({
                            "torch_dtype": torch.float16,
                            "device_map": None,
                            "low_cpu_mem_usage": True
                        })
                        
                        pipeline_class = DiffusionPipeline  # Use generic pipeline as fallback
                        pipeline = pipeline_class.from_pretrained(model_path, **fallback_kwargs)
                        
                        if self.device == "cuda":
                            pipeline = pipeline.to(self.device)
                        
                        logger.info("✅ Pipeline created successfully with fallback configuration")
                        return pipeline
                        
                    except Exception as fallback_error:
                        logger.error(f"❌ Fallback pipeline creation also failed: {fallback_error}")
            
            raise
    
    def _select_pipeline_class(self, model_path: str, architecture_type: str):
        """Select the appropriate pipeline class based on model and architecture"""
        
        # For text-to-image tasks, prefer AutoPipelineForText2Image for MMDiT
        if architecture_type == "MMDiT" and "edit" not in model_path.lower():
            return AutoPipelineForText2Image
        
        # For image editing models or UNet architecture, use generic DiffusionPipeline
        return DiffusionPipeline
    
    def _prepare_loading_kwargs(self) -> Dict[str, Any]:
        """Prepare keyword arguments for pipeline loading"""
        kwargs = {
            "torch_dtype": self.config.torch_dtype,
            "use_safetensors": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
            "resume_download": True,
            "force_download": False,
        }
        
        # Add device map for CPU offloading if enabled
        if self.config.enable_cpu_offload or self.config.enable_sequential_cpu_offload:
            kwargs["device_map"] = "balanced"
        else:
            kwargs["device_map"] = None
        
        logger.info(f"Loading configuration: {kwargs}")
        return kwargs
    
    def _apply_gpu_optimizations(self, pipeline) -> None:
        """Apply modern GPU optimizations (TF32, cuDNN benchmark, etc.)"""
        if self.device != "cuda":
            logger.info("Skipping GPU optimizations (not using CUDA)")
            return
        
        logger.info("🚀 Applying modern GPU optimizations...")
        
        try:
            # Enable TF32 for Tensor Cores (RTX 30/40 series)
            if self.config.enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("✅ TF32 Tensor Cores enabled")
            
            # Enable cuDNN benchmark for consistent input sizes
            if self.config.enable_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                logger.info("✅ cuDNN benchmark enabled")
            
            # Configure scaled dot-product attention (PyTorch 2.0+)
            if self.config.enable_scaled_dot_product_attention:
                try:
                    torch.backends.cuda.enable_math_sdp(True)
                    if self.config.enable_memory_efficient_attention:
                        torch.backends.cuda.enable_mem_efficient_sdp(True)
                        logger.info("✅ Memory-efficient scaled dot-product attention enabled")
                    
                    # Flash attention disabled for Qwen-Image compatibility
                    torch.backends.cuda.enable_flash_sdp(self.config.enable_flash_attention)
                    if not self.config.enable_flash_attention:
                        logger.info("⚠️ Flash attention disabled for Qwen-Image MMDiT compatibility")
                    
                except Exception as e:
                    logger.warning(f"Could not configure scaled dot-product attention: {e}")
            
            # Disable gradient computation for inference
            if self.config.disable_gradient_computation:
                torch.set_grad_enabled(False)
                logger.info("✅ Gradient computation disabled for inference")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        except Exception as e:
            logger.warning(f"Some GPU optimizations failed: {e}")
    
    def _configure_memory_settings(self, pipeline) -> None:
        """Configure memory optimization settings (disabled for high-VRAM GPUs)"""
        logger.info("🔧 Configuring memory settings for high-VRAM GPU...")
        
        try:
            # DISABLE memory-saving features that hurt performance on high-VRAM GPUs
            if hasattr(pipeline, 'disable_attention_slicing') and not self.config.enable_attention_slicing:
                pipeline.disable_attention_slicing()
                logger.info("✅ Attention slicing DISABLED for performance")
            elif hasattr(pipeline, 'enable_attention_slicing') and self.config.enable_attention_slicing:
                pipeline.enable_attention_slicing()
                logger.info("⚠️ Attention slicing enabled (may reduce performance)")
            
            if hasattr(pipeline, 'disable_vae_slicing') and not self.config.enable_vae_slicing:
                pipeline.disable_vae_slicing()
                logger.info("✅ VAE slicing DISABLED for performance")
            elif hasattr(pipeline, 'enable_vae_slicing') and self.config.enable_vae_slicing:
                pipeline.enable_vae_slicing()
                logger.info("⚠️ VAE slicing enabled (may reduce performance)")
            
            if hasattr(pipeline, 'disable_vae_tiling') and not self.config.enable_vae_tiling:
                pipeline.disable_vae_tiling()
                logger.info("✅ VAE tiling DISABLED for performance")
            elif hasattr(pipeline, 'enable_vae_tiling') and self.config.enable_vae_tiling:
                pipeline.enable_vae_tiling()
                logger.info("⚠️ VAE tiling enabled (may reduce performance)")
            
            # CPU offloading settings
            if self.config.enable_cpu_offload:
                if hasattr(pipeline, 'enable_model_cpu_offload'):
                    pipeline.enable_model_cpu_offload()
                    logger.info("⚠️ Model CPU offload enabled (may reduce performance)")
            
            if self.config.enable_sequential_cpu_offload:
                if hasattr(pipeline, 'enable_sequential_cpu_offload'):
                    pipeline.enable_sequential_cpu_offload()
                    logger.info("⚠️ Sequential CPU offload enabled (may reduce performance)")
            
        except Exception as e:
            logger.warning(f"Some memory settings could not be configured: {e}")
    
    def _setup_attention_processors(self, pipeline, architecture_type: str) -> None:
        """Setup attention processors based on architecture type"""
        logger.info(f"🔧 Setting up attention processors for {architecture_type} architecture...")
        
        try:
            if architecture_type == "MMDiT":
                # MMDiT (Qwen-Image) uses custom attention mechanism
                # AttnProcessor2_0 is NOT compatible and causes tensor unpacking errors
                logger.info("⚠️ Using default attention processor for MMDiT compatibility")
                logger.info("   AttnProcessor2_0 disabled due to tensor unpacking issues with Qwen-Image")
                
                # Keep the default attention processor - it's already optimized for MMDiT
                # The Qwen-Image transformer has built-in optimized attention
                
            elif architecture_type == "UNet":
                # UNet architecture can use AttnProcessor2_0 for Flash Attention
                try:
                    from diffusers.models.attention_processor import AttnProcessor2_0
                    
                    # Check if the pipeline has UNet component
                    if hasattr(pipeline, 'unet') and pipeline.unet is not None:
                        pipeline.unet.set_attn_processor(AttnProcessor2_0())
                        logger.info("✅ AttnProcessor2_0 (Flash Attention) enabled for UNet")
                    else:
                        logger.warning("UNet component not found, keeping default attention")
                        
                except ImportError:
                    logger.warning("AttnProcessor2_0 not available, using default attention")
                except Exception as e:
                    logger.warning(f"Could not set AttnProcessor2_0: {e}")
            
            else:
                logger.info("Unknown architecture type, using default attention processors")
                
        except Exception as e:
            logger.warning(f"Attention processor setup failed: {e}")
    
    def configure_generation_settings(self, architecture_type: str = None) -> Dict[str, Any]:
        """
        Configure optimal generation settings for MMDiT architecture with dynamic batch sizing
        
        Args:
            architecture_type: Override architecture type for settings
            
        Returns:
            Dictionary of optimal generation parameters
        """
        arch_type = architecture_type or self.config.architecture_type
        
        # Calculate optimal batch size if attention optimizer is available
        optimal_batch_size = 1
        if self.attention_optimizer and self.config.enable_dynamic_batch_sizing:
            optimal_batch_size = self.attention_optimizer.calculate_optimal_batch_size(
                self.config.optimal_width,
                self.config.optimal_height,
                self.config.optimal_steps
            )
        
        if arch_type == "MMDiT":
            # Optimized settings for MMDiT (Qwen-Image) architecture
            settings = {
                "width": self.config.optimal_width,
                "height": self.config.optimal_height,
                "num_inference_steps": self.config.optimal_steps,
                "true_cfg_scale": self.config.optimal_cfg_scale,  # Qwen-Image uses true_cfg_scale
                "output_type": "pil",
                "return_dict": True,
            }
            
            # Add batch size if > 1
            if optimal_batch_size > 1:
                settings["batch_size"] = optimal_batch_size
                logger.info(f"Dynamic batch size: {optimal_batch_size}")
            
            logger.info(f"MMDiT generation settings: {settings}")
            
        elif arch_type == "UNet":
            # Standard settings for UNet architecture
            settings = {
                "width": self.config.optimal_width,
                "height": self.config.optimal_height,
                "num_inference_steps": self.config.optimal_steps,
                "guidance_scale": self.config.optimal_cfg_scale,  # UNet uses guidance_scale
                "output_type": "pil",
                "return_dict": True,
            }
            
            # Add batch size if > 1
            if optimal_batch_size > 1:
                settings["batch_size"] = optimal_batch_size
                logger.info(f"Dynamic batch size: {optimal_batch_size}")
            
            logger.info(f"UNet generation settings: {settings}")
            
        else:
            # Generic settings for unknown architectures
            settings = {
                "width": self.config.optimal_width,
                "height": self.config.optimal_height,
                "num_inference_steps": self.config.optimal_steps,
                "output_type": "pil",
                "return_dict": True,
            }
            
            # Try both CFG scale parameter names
            settings["guidance_scale"] = self.config.optimal_cfg_scale
            settings["true_cfg_scale"] = self.config.optimal_cfg_scale
            
            # Add batch size if > 1
            if optimal_batch_size > 1:
                settings["batch_size"] = optimal_batch_size
                logger.info(f"Dynamic batch size: {optimal_batch_size}")
            
            logger.info(f"Generic generation settings: {settings}")
        
        return settings
    
    def apply_torch_compile_optimization(self, pipeline) -> bool:
        """
        Apply torch.compile optimization (experimental)
        
        Args:
            pipeline: Pipeline to optimize
            
        Returns:
            True if successful, False otherwise
        """
        if not self.config.use_torch_compile:
            logger.info("torch.compile optimization disabled in configuration")
            return False
        
        if self.config.architecture_type == "MMDiT":
            logger.warning("torch.compile disabled for MMDiT due to tensor unpacking issues")
            logger.warning("Qwen-Image transformer returns single tensor instead of tuple")
            return False
        
        try:
            logger.info("🔧 Applying torch.compile optimization...")
            
            # Compile the most compute-intensive components
            if hasattr(pipeline, 'unet') and pipeline.unet is not None:
                pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune")
                logger.info("✅ UNet compiled with torch.compile")
                return True
            elif hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                # This would cause issues with MMDiT, so we skip it
                logger.warning("Transformer compilation skipped for compatibility")
                return False
            
        except Exception as e:
            logger.error(f"torch.compile optimization failed: {e}")
            return False
        
        return False
    
    def validate_optimization(self, pipeline) -> Dict[str, Any]:
        """
        Validate that optimizations were applied correctly
        
        Args:
            pipeline: Pipeline to validate
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            "device_placement": "unknown",
            "memory_optimizations": "unknown",
            "attention_setup": "unknown",
            "gpu_optimizations": "unknown",
            "overall_status": "unknown"
        }
        
        try:
            # Check device placement
            if hasattr(pipeline, 'device'):
                validation_results["device_placement"] = f"Pipeline on {pipeline.device}"
            elif self.device == "cuda":
                # Check component devices
                components_on_gpu = 0
                total_components = 0
                
                for comp_name in ['unet', 'vae', 'text_encoder', 'transformer']:
                    if hasattr(pipeline, comp_name):
                        component = getattr(pipeline, comp_name)
                        if component is not None:
                            total_components += 1
                            try:
                                comp_device = str(next(component.parameters()).device)
                                if "cuda" in comp_device:
                                    components_on_gpu += 1
                            except (StopIteration, AttributeError):
                                pass
                
                if total_components > 0:
                    validation_results["device_placement"] = f"{components_on_gpu}/{total_components} components on GPU"
            
            # Check memory optimizations
            memory_features = []
            if hasattr(pipeline, '_attention_slicing_enabled'):
                memory_features.append(f"attention_slicing: {pipeline._attention_slicing_enabled}")
            if hasattr(pipeline, '_vae_slicing_enabled'):
                memory_features.append(f"vae_slicing: {pipeline._vae_slicing_enabled}")
            
            validation_results["memory_optimizations"] = ", ".join(memory_features) if memory_features else "default"
            
            # Check GPU optimizations
            gpu_opts = []
            if torch.backends.cuda.matmul.allow_tf32:
                gpu_opts.append("TF32")
            if torch.backends.cudnn.benchmark:
                gpu_opts.append("cuDNN_benchmark")
            if not torch.is_grad_enabled():
                gpu_opts.append("no_grad")
            
            validation_results["gpu_optimizations"] = ", ".join(gpu_opts) if gpu_opts else "none"
            
            # Overall status
            if (validation_results["device_placement"] != "unknown" and 
                "cuda" in validation_results["device_placement"].lower()):
                validation_results["overall_status"] = "optimized"
            else:
                validation_results["overall_status"] = "basic"
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_results["overall_status"] = "error"
        
        logger.info(f"Optimization validation: {validation_results}")
        return validation_results
    
    def get_performance_recommendations(self, gpu_memory_gb: float = None) -> Dict[str, Any]:
        """
        Get performance recommendations based on hardware
        
        Args:
            gpu_memory_gb: Available GPU memory in GB
            
        Returns:
            Performance recommendations
        """
        if gpu_memory_gb is None and torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        recommendations = {
            "optimal_settings": {},
            "memory_strategy": "unknown",
            "expected_performance": "unknown",
            "warnings": []
        }
        
        if gpu_memory_gb is None:
            recommendations["memory_strategy"] = "cpu_only"
            recommendations["expected_performance"] = "slow (CPU inference)"
            recommendations["warnings"].append("No GPU detected - performance will be limited")
            return recommendations
        
        if gpu_memory_gb >= 16:
            # High-VRAM GPU (RTX 4080, RTX 4090, etc.)
            recommendations["optimal_settings"] = {
                "enable_attention_slicing": False,
                "enable_vae_slicing": False,
                "enable_cpu_offload": False,
                "torch_dtype": "float16",
                "batch_size": 1,
                "resolution": "1024x1024"
            }
            recommendations["memory_strategy"] = "full_gpu"
            recommendations["expected_performance"] = "excellent (2-5 seconds per step)"
            
        elif gpu_memory_gb >= 12:
            # Medium-VRAM GPU (RTX 3080, RTX 4070, etc.)
            recommendations["optimal_settings"] = {
                "enable_attention_slicing": False,
                "enable_vae_slicing": False,
                "enable_cpu_offload": False,
                "torch_dtype": "float16",
                "batch_size": 1,
                "resolution": "1024x1024"
            }
            recommendations["memory_strategy"] = "full_gpu"
            recommendations["expected_performance"] = "good (3-8 seconds per step)"
            
        elif gpu_memory_gb >= 8:
            # Lower-VRAM GPU (RTX 3070, RTX 4060, etc.)
            recommendations["optimal_settings"] = {
                "enable_attention_slicing": True,
                "enable_vae_slicing": True,
                "enable_cpu_offload": False,
                "torch_dtype": "float16",
                "batch_size": 1,
                "resolution": "768x768"
            }
            recommendations["memory_strategy"] = "memory_efficient"
            recommendations["expected_performance"] = "moderate (5-15 seconds per step)"
            recommendations["warnings"].append("Consider enabling memory optimizations for stability")
            
        else:
            # Very low VRAM
            recommendations["optimal_settings"] = {
                "enable_attention_slicing": True,
                "enable_vae_slicing": True,
                "enable_cpu_offload": True,
                "torch_dtype": "float16",
                "batch_size": 1,
                "resolution": "512x512"
            }
            recommendations["memory_strategy"] = "cpu_offload"
            recommendations["expected_performance"] = "slow (15+ seconds per step)"
            recommendations["warnings"].append("Low VRAM detected - consider upgrading GPU")
        
        logger.info(f"Performance recommendations for {gpu_memory_gb:.1f}GB GPU: {recommendations['memory_strategy']}")
        return recommendations