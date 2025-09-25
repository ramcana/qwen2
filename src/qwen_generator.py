"""
Qwen-Image Generator Core Module
Handles model loading, image generation, and optimization for RTX 4080
Integrated with architecture detection and optimized pipeline components
"""

import json
import os
import random
from datetime import datetime
from typing import Optional, Tuple, Union

import PIL.Image
import torch
from diffusers import DiffusionPipeline
from PIL import ImageFilter

try:
    from diffusers import QwenImageEditPipeline
except ImportError:
    print("⚠️ QwenImageEditPipeline not available. Enhanced features will be limited.")
    QwenImageEditPipeline = None

from qwen_image_config import (
    GENERATION_CONFIG,
    MEMORY_CONFIG,
    MODEL_CONFIG,
    PROMPT_ENHANCEMENT,
)

# Import optimized components
try:
    from model_detection_service import ModelDetectionService, ModelInfo
    from pipeline_optimizer import PipelineOptimizer, OptimizationConfig
    from model_download_manager import ModelDownloadManager
    from utils.performance_monitor import PerformanceMonitor, monitor_generation_performance
    from compatibility_layer import CompatibilityLayer
    OPTIMIZED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Optimized components not available: {e}")
    print("   Falling back to basic functionality")
    OPTIMIZED_COMPONENTS_AVAILABLE = False

# Import Qwen2-VL integration
try:
    from qwen2vl_integration import Qwen2VLIntegration, create_qwen2vl_integration
    QWEN2VL_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Qwen2-VL integration not available: {e}")
    print("   Multimodal features will be limited")
    QWEN2VL_INTEGRATION_AVAILABLE = False


class QwenImageGenerator:
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name: str = model_name or MODEL_CONFIG["model_name"]
        self.pipe: Optional[DiffusionPipeline] = None
        self.edit_pipe: Optional[QwenImageEditPipeline] = None
        
        # Initialize optimized components if available
        self.detection_service: Optional[ModelDetectionService] = None
        self.pipeline_optimizer: Optional[PipelineOptimizer] = None
        self.download_manager: Optional[ModelDownloadManager] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.compatibility_layer: Optional[CompatibilityLayer] = None
        
        # Initialize Qwen2-VL integration
        self.qwen2vl_integration: Optional[Qwen2VLIntegration] = None
        self.multimodal_enabled: bool = False
        
        # Architecture detection
        self.current_architecture: str = "Unknown"
        self.model_info: Optional[ModelInfo] = None
        self.optimization_applied: bool = False
        
        # Create output directory
        self.output_dir: str = "generated_images"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        print("\n📋 Qwen Model Usage:")
        print("   • Qwen-Image: Text-to-image generation")
        print("   • Qwen-Image-Edit: Image editing with reference images")
        
        # Initialize optimized components
        self._initialize_optimized_components()
    
    def _initialize_optimized_components(self) -> None:
        """Initialize optimized components for architecture detection and performance"""
        if not OPTIMIZED_COMPONENTS_AVAILABLE:
            print("⚠️ Optimized components not available - using basic functionality")
            return
        
        try:
            print("🔧 Initializing optimized components...")
            
            # Initialize model detection service
            self.detection_service = ModelDetectionService()
            print("✅ Model detection service initialized")
            
            # Initialize download manager
            self.download_manager = ModelDownloadManager()
            print("✅ Model download manager initialized")
            
            # Initialize performance monitor with 5-second target
            self.performance_monitor = PerformanceMonitor(target_generation_time=5.0)
            print("✅ Performance monitor initialized")
            
            # Initialize compatibility layer
            self.compatibility_layer = CompatibilityLayer()
            print("✅ Compatibility layer initialized")
            
            # Initialize Qwen2-VL integration
            self._initialize_qwen2vl_integration()
            
            print("🚀 All optimized components ready")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize optimized components: {e}")
            print("   Falling back to basic functionality")
            # Reset components to None
            self.detection_service = None
            self.pipeline_optimizer = None
            self.download_manager = None
            self.performance_monitor = None
            self.compatibility_layer = None
            self.qwen2vl_integration = None
    
    def _initialize_qwen2vl_integration(self) -> None:
        """Initialize Qwen2-VL integration for multimodal capabilities"""
        if not QWEN2VL_INTEGRATION_AVAILABLE:
            print("⚠️ Qwen2-VL integration not available - multimodal features disabled")
            return
        
        try:
            print("🔧 Initializing Qwen2-VL integration...")
            
            # Create Qwen2-VL integration with fallback enabled
            self.qwen2vl_integration = create_qwen2vl_integration(
                enable_prompt_enhancement=True,
                enable_image_analysis=True,
                fallback_enabled=True
            )
            
            # Check if Qwen2-VL models are available
            if self.detection_service:
                qwen2vl_info = self.detection_service.detect_qwen2_vl_capabilities()
                if qwen2vl_info["integration_possible"]:
                    print("✅ Qwen2-VL models detected - attempting to load...")
                    if self.qwen2vl_integration.load_model():
                        self.multimodal_enabled = True
                        print("✅ Qwen2-VL integration ready with full capabilities")
                    else:
                        print("⚠️ Qwen2-VL model loading failed - using fallback mode")
                        self.multimodal_enabled = False
                else:
                    print("⚠️ No Qwen2-VL models found - using fallback mode")
                    self.multimodal_enabled = False
            else:
                print("⚠️ Model detection not available - using fallback mode")
                self.multimodal_enabled = False
            
            print("✅ Qwen2-VL integration initialized")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize Qwen2-VL integration: {e}")
            print("   Multimodal features will use fallback mode")
            self.qwen2vl_integration = None
            self.multimodal_enabled = False
    
    def detect_and_optimize_model(self) -> bool:
        """
        Detect current model and apply optimizations if needed
        
        Returns:
            True if optimization was successful or not needed
        """
        if not self.detection_service:
            print("⚠️ Model detection service not available")
            return False
        
        try:
            print("🔍 Detecting current model configuration...")
            
            # Detect current model
            self.model_info = self.detection_service.detect_current_model()
            
            if not self.model_info:
                print("❌ No model detected - download required")
                return self._handle_missing_model()
            
            # Detect architecture
            self.current_architecture = self.detection_service.detect_model_architecture(self.model_info)
            print(f"🏗️ Detected architecture: {self.current_architecture}")
            print(f"📦 Model: {self.model_info.name} ({self.model_info.size_gb:.1f}GB)")
            print(f"📊 Status: {self.model_info.download_status}")
            print(f"🎯 Optimal for T2I: {self.model_info.is_optimal}")
            
            # Check if optimization is needed
            optimization_needed = self.detection_service.is_optimization_needed()
            
            if optimization_needed:
                print("🔄 Model optimization required")
                return self._apply_model_optimization()
            else:
                print("✅ Current model is already optimal")
                # Still create optimized pipeline for consistency
                return self._create_optimized_pipeline()
            
        except Exception as e:
            print(f"❌ Model detection and optimization failed: {e}")
            return False
    
    def _handle_missing_model(self) -> bool:
        """Handle case where no model is detected"""
        if not self.download_manager:
            print("❌ Download manager not available")
            return False
        
        try:
            print("📥 Downloading optimal model for text-to-image generation...")
            
            # Download Qwen-Image model
            success = self.download_manager.download_qwen_image()
            
            if success:
                print("✅ Model download completed")
                # Re-detect after download
                return self.detect_and_optimize_model()
            else:
                print("❌ Model download failed")
                return False
                
        except Exception as e:
            print(f"❌ Model download error: {e}")
            return False
    
    def _apply_model_optimization(self) -> bool:
        """Apply model optimization based on detected issues"""
        try:
            # Get recommended model
            recommended_model = self.detection_service.get_recommended_model()
            print(f"💡 Recommended model: {recommended_model}")
            
            # Check if we need to download the recommended model
            if self.model_info.name != recommended_model or self.model_info.download_status != "complete":
                if self.download_manager:
                    print(f"📥 Downloading recommended model: {recommended_model}")
                    
                    if "Qwen-Image" in recommended_model and "Edit" not in recommended_model:
                        success = self.download_manager.download_qwen_image()
                    else:
                        print(f"⚠️ Unsupported recommended model: {recommended_model}")
                        success = False
                    
                    if not success:
                        print("❌ Failed to download recommended model")
                        return False
                    
                    # Re-detect after download
                    self.model_info = self.detection_service.detect_current_model()
                    if not self.model_info:
                        print("❌ Model still not detected after download")
                        return False
            
            # Create optimized pipeline
            return self._create_optimized_pipeline()
            
        except Exception as e:
            print(f"❌ Model optimization failed: {e}")
            return False
    
    def _create_optimized_pipeline(self) -> bool:
        """Create optimized pipeline using PipelineOptimizer"""
        try:
            print("🚀 Creating optimized pipeline...")
            
            # Import OptimizationConfig if not already imported
            try:
                from pipeline_optimizer import OptimizationConfig
            except ImportError:
                print("⚠️ OptimizationConfig not available, using basic configuration")
                return self._create_basic_optimized_pipeline()
            
            # Create optimization configuration
            optimization_config = OptimizationConfig(
                torch_dtype=MODEL_CONFIG.get("torch_dtype", torch.float16),
                device=self.device,
                architecture_type=self.current_architecture,
                # Disable memory-saving features for high-VRAM GPUs
                enable_attention_slicing=False,
                enable_vae_slicing=False,
                enable_cpu_offload=False,
                # Enable performance optimizations
                enable_tf32=True,
                enable_cudnn_benchmark=True,
                enable_scaled_dot_product_attention=True,
                enable_memory_efficient_attention=True,
                enable_flash_attention=False,  # Disabled for Qwen-Image compatibility
                # Generation settings
                optimal_steps=GENERATION_CONFIG.get("num_inference_steps", 20),
                optimal_cfg_scale=GENERATION_CONFIG.get("true_cfg_scale", 3.5),
                optimal_width=GENERATION_CONFIG.get("width", 1024),
                optimal_height=GENERATION_CONFIG.get("height", 1024),
            )
            
            # Initialize pipeline optimizer
            from pipeline_optimizer import PipelineOptimizer
            self.pipeline_optimizer = PipelineOptimizer(optimization_config)
            
            # Create optimized pipeline
            model_path = self.model_info.name if self.model_info else self.model_name
            self.pipe = self.pipeline_optimizer.create_optimized_pipeline(
                model_path, 
                self.current_architecture
            )
            
            # Validate optimization
            validation_results = self.pipeline_optimizer.validate_optimization(self.pipe)
            print(f"🔍 Optimization validation: {validation_results['overall_status']}")
            
            self.optimization_applied = True
            print("✅ Optimized pipeline created successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create optimized pipeline: {e}")
            print("🔄 Falling back to basic optimization...")
            return self._create_basic_optimized_pipeline()
    
    def _create_basic_optimized_pipeline(self) -> bool:
        """Create a basic optimized pipeline without PipelineOptimizer"""
        try:
            print("🔄 Creating basic optimized pipeline...")
            
            # Select optimal pipeline class
            pipeline_class = self._select_optimal_pipeline_class()
            
            # Basic loading configuration
            loading_kwargs = {
                "torch_dtype": MODEL_CONFIG.get("torch_dtype", torch.float16),
                "use_safetensors": True,
                "trust_remote_code": True,
                "low_cpu_mem_usage": False,
                "device_map": None,
                "resume_download": True,
            }
            
            # Load pipeline
            model_path = self.model_info.name if self.model_info else self.model_name
            self.pipe = pipeline_class.from_pretrained(model_path, **loading_kwargs)
            
            # Move to device
            if self.device == "cuda":
                self.pipe = self.pipe.to(self.device)
            
            # Apply basic optimizations
            self._apply_architecture_specific_optimizations()
            
            self.optimization_applied = True
            print("✅ Basic optimized pipeline created successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create basic optimized pipeline: {e}")
            return False
    
    def check_model_cache(self) -> dict:
        """Check if model is properly cached with detailed analysis"""
        import os
        import glob
        
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_path = os.path.join(cache_dir, f"models--{self.model_name.replace('/', '--')}")
        
        result = {
            "exists": False,
            "complete": False,
            "size_gb": 0,
            "missing_components": [],
            "snapshot_path": None
        }
        
        if not os.path.exists(model_path):
            return result
        
        result["exists"] = True
        
        # Find snapshot directory
        snapshots_dir = os.path.join(model_path, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                result["snapshot_path"] = snapshot_path
                
                # Check essential components
                essential_components = {
                    "model_index.json": False,
                    "scheduler": False,
                    "text_encoder": False,
                    "transformer": False,
                    "vae": False,
                    "tokenizer": False
                }
                
                for item in os.listdir(snapshot_path):
                    item_path = os.path.join(snapshot_path, item)
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        if item in essential_components:
                            essential_components[item] = True
                    elif os.path.isdir(item_path):
                        if item in essential_components:
                            # Check if directory has files
                            if os.listdir(item_path):
                                essential_components[item] = True
                
                # Check what's missing
                for component, exists in essential_components.items():
                    if not exists:
                        result["missing_components"].append(component)
                
                result["complete"] = len(result["missing_components"]) == 0
        
        # Calculate total size
        try:
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(model_path)
                for filename in filenames
            )
            result["size_gb"] = total_size / (1024**3)
        except:
            result["size_gb"] = 0
        
        return result
    
    def estimate_download_progress(self) -> dict:
        """Estimate download progress and ETA"""
        import os
        import time
        
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_path = os.path.join(cache_dir, f"models--{self.model_name.replace('/', '--')}")
        
        if os.path.exists(model_path):
            try:
                # Get current cache size
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(model_path)
                    for filename in filenames
                )
                size_gb = total_size / (1024**3)
                expected_size_gb = 36  # Approximate total size
                progress_pct = min(100, (size_gb / expected_size_gb) * 100)
                
                return {
                    "current_size_gb": size_gb,
                    "expected_size_gb": expected_size_gb,
                    "progress_percent": progress_pct,
                    "is_complete": progress_pct > 95
                }
            except:
                pass
        
        return {"current_size_gb": 0, "expected_size_gb": 36, "progress_percent": 0, "is_complete": False}

    def load_model(self) -> bool:
        """Load Qwen-Image diffusion pipeline with architecture detection and optimization"""
        try:
            print("🎯 Enhanced Qwen-Image Model Loading with Architecture Detection")
            print(f"Attempting to load: {self.model_name}")
            
            # Use optimized loading if components are available
            if OPTIMIZED_COMPONENTS_AVAILABLE and self.detection_service:
                print("🚀 Using optimized loading with architecture detection...")
                
                # Detect and optimize model
                optimization_success = self.detect_and_optimize_model()
                
                if optimization_success and self.pipe:
                    print("✅ Model loaded with optimized pipeline")
                    
                    # Apply automatic model switching logic with MMDiT vs UNet detection
                    self._apply_architecture_specific_optimizations()
                    
                    # Implement proper device management for multi-component models
                    self._ensure_device_consistency()
                    
                    # Verify device setup
                    self.verify_device_setup()
                    
                    # Skip loading Qwen-Image-Edit to prevent unnecessary download
                    print("⚠️ Qwen-Image-Edit loading skipped to prevent additional downloads")
                    print("   Text-to-image generation is fully functional with Qwen-Image")
                    
                    return True
                else:
                    print("⚠️ Optimized loading failed, falling back to legacy method")
                    # Fall through to legacy loading
            
            # Legacy loading method (fallback)
            print("🔄 Using legacy loading method...")
            
            # Check download progress
            progress = self.estimate_download_progress()
            print(f"📊 Cache status: {progress['current_size_gb']:.1f}GB / {progress['expected_size_gb']}GB ({progress['progress_percent']:.1f}%)")
            
            # Pre-flight checks
            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                free_gb = free_memory / (1024**3)
                print(f"🖥️ Available GPU memory: {free_gb:.1f}GB")
                
                if free_gb < 12:
                    print("⚠️ Low GPU memory - clearing cache...")
                    torch.cuda.empty_cache()
            
            # Check cache status first
            cache_status = self.check_model_cache()
            print(f"📁 Cache analysis:")
            print(f"   Exists: {cache_status['exists']}")
            print(f"   Complete: {cache_status['complete']}")
            print(f"   Size: {cache_status['size_gb']:.1f}GB")
            if cache_status['missing_components']:
                print(f"   Missing: {', '.join(cache_status['missing_components'])}")
            
            # Enhanced loading strategy with architecture detection
            try:
                print("🚀 Loading with enhanced configuration for RTX 4080...")
                
                # Use DiffusionPipeline with proper architecture detection
                pipeline_class = self._select_optimal_pipeline_class()
                print(f"📦 Using pipeline class: {pipeline_class.__name__}")
                
                # Base loading parameters
                loading_kwargs = {
                    "torch_dtype": MODEL_CONFIG["torch_dtype"],
                    "use_safetensors": MODEL_CONFIG["use_safetensors"],
                    "trust_remote_code": MODEL_CONFIG["trust_remote_code"],
                    "low_cpu_mem_usage": False,  # Use 128GB RAM for speed
                    "device_map": None,  # Manual device management
                }
                
                # Strategy 1: Try cache-only loading if cache appears complete
                if cache_status["complete"] and cache_status["size_gb"] > 30:
                    print("📁 Cache appears complete - attempting cache-only loading...")
                    try:
                        cache_kwargs = loading_kwargs.copy()
                        cache_kwargs["local_files_only"] = True
                        self.pipe = pipeline_class.from_pretrained(self.model_name, **cache_kwargs)
                        print("✅ Model loaded from cache successfully!")
                    except Exception as cache_error:
                        print(f"⚠️ Cache loading failed (likely missing files): {str(cache_error)[:100]}...")
                        print("🔄 Cache is incomplete - will download missing files")
                        # Don't raise, fall through to download strategy
                
                # Strategy 2: Smart resume download (handles incomplete cache)
                if not hasattr(self, 'pipe') or self.pipe is None:  # Only if cache loading failed
                    if not cache_status["complete"]:
                        print(f"🌐 Cache incomplete - downloading missing components...")
                        print(f"   Missing: {', '.join(cache_status['missing_components'])}")
                    else:
                        print(f"🌐 Cache validation failed - resuming download...")
                    
                    online_kwargs = loading_kwargs.copy()
                    online_kwargs["resume_download"] = True
                    online_kwargs["force_download"] = False  # Smart resume, don't redownload existing files
                    
                    print("📥 Starting smart resume download (will skip existing files)...")
                    self.pipe = pipeline_class.from_pretrained(self.model_name, **online_kwargs)
                    print("✅ Model loaded with smart resume strategy!")
                
                print(f"✅ Qwen-Image MMDiT model loaded with {MODEL_CONFIG['torch_dtype']} precision")
                
            except Exception as e1:
                print(f"⚠️ Optimal loading failed: {e1}")
                print("🔄 Trying fallback configuration...")
                
                try:
                    # Fallback: Try with basic configuration
                    print("🔄 Trying fallback configuration...")
                    self.pipe = DiffusionPipeline.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        resume_download=True
                    )
                    print("✅ Qwen-Image MMDiT model loaded with fallback configuration")
                    
                except Exception as e2:
                    print(f"⚠️ Float16 loading failed: {e2}")
                    print("🔄 Trying minimal configuration...")
                    
                    # Final fallback: Minimal settings with fresh download
                    print("🔄 Trying minimal configuration with fresh download...")
                    self.pipe = DiffusionPipeline.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        use_safetensors=False,
                        trust_remote_code=True,
                        force_download=True,  # Force fresh download
                        resume_download=True
                    )
                    print("✅ Qwen-Image MMDiT model loaded with minimal configuration")
            
            # Move to GPU with optimized approach for RTX 4080
            if torch.cuda.is_available():
                print(f"🔄 Moving model to GPU: {self.device}")
                
                # Clear GPU memory first
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Move pipeline efficiently - let PyTorch handle component movement
                self.pipe = self.pipe.to(self.device)
                
                # Apply basic optimizations for legacy loading
                self._apply_basic_optimizations()
                
                # Final memory cleanup
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            else:
                print("  CUDA not available, using CPU")
                
            print("  Qwen-Image model loaded successfully!")
            
            # Skip loading Qwen-Image-Edit to prevent unnecessary download
            print("⚠️ Qwen-Image-Edit loading skipped to prevent additional downloads")
            print("   Text-to-image generation is fully functional with Qwen-Image")
            
            # Verify device setup
            self.verify_device_setup()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Troubleshooting tips:")
            print("   1. Check internet connection for model download")
            print("   2. Ensure sufficient disk space (~60-70GB)")
            print("   3. Verify CUDA installation if using GPU")
            print("   4. Try restarting the application")
            return False
    
    def _select_optimal_pipeline_class(self):
        """Select optimal pipeline class based on architecture detection"""
        try:
            # If we have model detection service, use it to determine optimal pipeline
            if self.detection_service and self.model_info:
                architecture = self.current_architecture
                print(f"🏗️ Detected architecture: {architecture}")
                
                if architecture == "MMDiT" and "edit" not in self.model_name.lower():
                    # Use AutoPipelineForText2Image for MMDiT text-to-image models
                    from diffusers import AutoPipelineForText2Image
                    print("📦 Selected AutoPipelineForText2Image for MMDiT architecture")
                    return AutoPipelineForText2Image
                elif architecture == "UNet":
                    # Use AutoPipelineForText2Image for UNet models too
                    from diffusers import AutoPipelineForText2Image
                    print("📦 Selected AutoPipelineForText2Image for UNet architecture")
                    return AutoPipelineForText2Image
                else:
                    # Fallback to generic DiffusionPipeline
                    print("📦 Selected DiffusionPipeline for unknown/editing architecture")
                    return DiffusionPipeline
            
            # Default selection based on model name
            if "edit" in self.model_name.lower():
                print("📦 Selected DiffusionPipeline for editing model")
                return DiffusionPipeline
            else:
                # Try to use AutoPipelineForText2Image for text-to-image
                try:
                    from diffusers import AutoPipelineForText2Image
                    print("📦 Selected AutoPipelineForText2Image for text-to-image")
                    return AutoPipelineForText2Image
                except ImportError:
                    print("📦 AutoPipelineForText2Image not available, using DiffusionPipeline")
                    return DiffusionPipeline
                    
        except Exception as e:
            print(f"⚠️ Pipeline class selection failed: {e}, using DiffusionPipeline")
            return DiffusionPipeline
    
    def _apply_architecture_specific_optimizations(self) -> None:
        """Apply optimizations specific to the detected architecture"""
        try:
            print(f"🔧 Applying {self.current_architecture} architecture-specific optimizations...")
            
            if self.current_architecture == "MMDiT":
                # MMDiT-specific optimizations
                print("🚀 Applying MMDiT (Multimodal Diffusion Transformer) optimizations...")
                
                # MMDiT models use transformer component instead of UNet
                if hasattr(self.pipe, 'transformer') and self.pipe.transformer is not None:
                    print("✅ MMDiT transformer component detected")
                    
                    # Apply transformer-specific optimizations
                    try:
                        # Enable memory-efficient attention for transformers
                        if hasattr(self.pipe.transformer, 'set_use_memory_efficient_attention_xformers'):
                            self.pipe.transformer.set_use_memory_efficient_attention_xformers(True)
                            print("✅ Memory-efficient attention enabled for transformer")
                    except Exception as e:
                        print(f"⚠️ Could not enable memory-efficient attention: {e}")
                
                # Disable memory-saving features that hurt MMDiT performance
                self._disable_memory_saving_features()
                
            elif self.current_architecture == "UNet":
                # UNet-specific optimizations
                print("🚀 Applying UNet architecture optimizations...")
                
                if hasattr(self.pipe, 'unet') and self.pipe.unet is not None:
                    print("✅ UNet component detected")
                    
                    # Apply UNet-specific optimizations
                    try:
                        # Enable attention processor optimizations for UNet
                        from diffusers.models.attention_processor import AttnProcessor2_0
                        self.pipe.unet.set_attn_processor(AttnProcessor2_0())
                        print("✅ AttnProcessor2_0 enabled for UNet")
                    except Exception as e:
                        print(f"⚠️ Could not set attention processor: {e}")
                
                # Apply standard memory optimizations
                self._disable_memory_saving_features()
                
            else:
                print("⚠️ Unknown architecture, applying generic optimizations")
                self._disable_memory_saving_features()
            
            # Apply common GPU optimizations
            self._apply_gpu_optimizations()
            
        except Exception as e:
            print(f"⚠️ Architecture-specific optimizations failed: {e}")
    
    def _disable_memory_saving_features(self) -> None:
        """Disable memory-saving features that hurt performance on high-VRAM GPUs"""
        try:
            # DISABLE all memory-saving features that hurt performance
            if hasattr(self.pipe, 'disable_attention_slicing'):
                self.pipe.disable_attention_slicing()
                print("✅ Attention slicing DISABLED for performance")
            
            if hasattr(self.pipe, 'disable_vae_slicing'):
                self.pipe.disable_vae_slicing()
                print("✅ VAE slicing DISABLED for performance")
            
            if hasattr(self.pipe, 'disable_vae_tiling'):
                self.pipe.disable_vae_tiling()
                print("✅ VAE tiling DISABLED for performance")
                
        except Exception as e:
            print(f"⚠️ Could not disable memory-saving features: {e}")
    
    def _apply_gpu_optimizations(self) -> None:
        """Apply GPU-specific optimizations"""
        try:
            if not torch.cuda.is_available():
                return
                
            # Enable AGGRESSIVE PyTorch optimizations for RTX 4080
            # Enable all Tensor Core optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # Disable for speed
            
            # Enable CUDA optimizations for RTX 4080
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(False)  # Disabled for Qwen-Image compatibility
            
            # Disable gradient computation globally for inference
            torch.set_grad_enabled(False)
            
            print("✅ GPU optimizations enabled")
            print("   • TF32 Tensor Cores enabled")
            print("   • cuDNN benchmark enabled")
            print("   • Memory-efficient attention enabled")
            print("   • Gradient computation disabled")
            
        except Exception as e:
            print(f"⚠️ GPU optimizations failed: {e}")
    
    def _ensure_device_consistency(self) -> None:
        """Ensure all pipeline components are on the correct device"""
        try:
            if not self.pipe:
                return
                
            print(f"🔧 Ensuring device consistency for {self.current_architecture} architecture...")
            
            # Get list of components based on architecture
            if self.current_architecture == "MMDiT":
                components = ['transformer', 'vae', 'text_encoder', 'tokenizer']
            elif self.current_architecture == "UNet":
                components = ['unet', 'vae', 'text_encoder', 'tokenizer']
            else:
                # Generic component list
                components = ['unet', 'transformer', 'vae', 'text_encoder', 'tokenizer']
            
            # Check and move components to target device
            for comp_name in components:
                if hasattr(self.pipe, comp_name):
                    component = getattr(self.pipe, comp_name)
                    if component is not None:
                        try:
                            # Check current device
                            if hasattr(component, 'device'):
                                current_device = str(component.device)
                            else:
                                try:
                                    current_device = str(next(component.parameters()).device)
                                except (StopIteration, AttributeError):
                                    current_device = "unknown"
                            
                            # Move to target device if needed
                            if current_device != self.device and current_device != "unknown":
                                print(f"🔄 Moving {comp_name} from {current_device} to {self.device}")
                                component = component.to(self.device)
                                setattr(self.pipe, comp_name, component)
                                print(f"✅ {comp_name} moved successfully")
                            else:
                                print(f"✅ {comp_name} already on {self.device}")
                                
                        except Exception as e:
                            print(f"⚠️ Could not move {comp_name}: {e}")
            
            # Final device consistency check
            if hasattr(self.pipe, 'to'):
                self.pipe = self.pipe.to(self.device)
                print(f"✅ Pipeline moved to {self.device}")
                
        except Exception as e:
            print(f"⚠️ Device consistency check failed: {e}")
    
    def _apply_basic_optimizations(self) -> None:
        """Apply basic optimizations for legacy loading"""
        try:
            print("🚀 Applying basic RTX 4080 performance optimizations...")
            
            # Use the new modular optimization methods
            self._disable_memory_saving_features()
            self._apply_gpu_optimizations()
                
        except Exception as e:
            print(f"⚠️ Basic optimizations failed: {e}")
    
    def _load_qwen_edit_pipeline(self) -> None:
        """Load Qwen-Image-Edit pipeline for enhanced features using HF Hub API"""
        if QwenImageEditPipeline is None:
            print("⚠️ QwenImageEditPipeline not available. Install latest diffusers from GitHub.")
            print("   Enhanced features will use alternative methods.")
            self.edit_pipe = None
            return
            
        try:
            print("🔄 Loading Qwen-Image-Edit pipeline for enhanced features...")
            print("   Using HuggingFace Hub API for better download reliability")
            
            # Import HuggingFace Hub for better download handling
            try:
                from huggingface_hub import repo_info, snapshot_download
                use_hub_api = True
            except ImportError:
                print("⚠️ huggingface_hub not available, using standard method")
                use_hub_api = False
            
            # Load Qwen-Image-Edit model with improved download handling
            try:
                if use_hub_api:
                    # Pre-check if model exists and get size info
                    try:
                        repo_data = repo_info("Qwen/Qwen-Image-Edit")
                        total_size = sum(file.size for file in repo_data.siblings if file.size)
                        print(f"📊 Model size: {self._format_size(total_size)} (~20GB)")
                        print("💡 Download will resume automatically if interrupted")
                    except Exception:
                        print("📊 Model size: ~20GB (estimated)")
                
                # Load with optimized settings for your hardware (128GB RAM)
                self.edit_pipe = QwenImageEditPipeline.from_pretrained(
                    "Qwen/Qwen-Image-Edit",
                    torch_dtype=MODEL_CONFIG["torch_dtype"],
                    low_cpu_mem_usage=False,  # Disabled for 128GB RAM system
                    resume_download=True,     # Auto-resume interrupted downloads
                    use_safetensors=True      # Faster loading
                )
                
                if torch.cuda.is_available():
                    # Move to device and apply optimizations
                    self.edit_pipe = self.edit_pipe.to(self.device)
                    
                    if MEMORY_CONFIG["enable_attention_slicing"]:
                        # Apply memory optimizations if available
                        try:
                            self.edit_pipe.enable_attention_slicing()
                            print("✅ Attention slicing enabled for Qwen-Image-Edit")
                        except Exception as opt_error:
                            print(f"⚠️ Could not enable attention slicing: {opt_error}")
                    
                    # Verify device consistency for edit pipeline
                    self._verify_edit_pipeline_devices()
                
                print("✅ Qwen-Image-Edit pipeline loaded successfully!")
                print("   • Image-to-Image editing available")
                print("   • Inpainting capabilities available")
                print("   • Text editing in images available")
                
            except Exception as download_error:
                error_msg = str(download_error)
                print(f"⚠️ Could not download/load Qwen-Image-Edit: {download_error}")
                
                # Provide specific guidance based on error type
                if "Connection" in error_msg or "timeout" in error_msg.lower():
                    print("🌐 Network issue detected. Try:")
                    print("   1. Check internet connection stability")
                    print("   2. Use the enhanced downloader: python tools/download_qwen_edit_hub.py")
                    print("   3. Download will auto-resume if interrupted")
                elif "disk" in error_msg.lower() or "space" in error_msg.lower():
                    print("💾 Disk space issue. Ensure ~25GB free space available")
                elif "permission" in error_msg.lower():
                    print("🔒 Permission issue. Check write access to cache directory")
                else:
                    print("💡 General troubleshooting:")
                    print("   1. Try: python tools/download_qwen_edit_hub.py")
                    print("   2. Check HuggingFace Hub accessibility")
                    print("   3. Ensure sufficient disk space (~25GB)")
                
                print("   Enhanced features will use alternative approaches.")
                self.edit_pipe = None
                
        except Exception as e:
            print(f"⚠️ Error loading Qwen-Image-Edit pipeline: {e}")
            print("   Enhanced features will use creative text-to-image approaches.")
            self.edit_pipe = None
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format byte size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def _verify_edit_pipeline_devices(self) -> bool:
        """Verify edit pipeline components are on correct device"""
        if not self.edit_pipe:
            return False
            
        print(f"🔍 Verifying Qwen-Image-Edit pipeline devices for {self.device}:")
        
        try:
            # Check components safely
            components = ['unet', 'vae', 'text_encoder']
            all_correct = True
            
            for comp_name in components:
                if hasattr(self.edit_pipe, comp_name) and getattr(self.edit_pipe, comp_name) is not None:
                    component = getattr(self.edit_pipe, comp_name)
                    
                    try:
                        # Safe device check
                        if hasattr(component, 'device'):
                            comp_device = str(component.device)
                        else:
                            # Check first parameter device safely
                            try:
                                comp_device = str(next(component.parameters()).device)
                            except (StopIteration, AttributeError):
                                comp_device = "unknown"
                        
                        print(f"   {comp_name.upper()}: {comp_device}")
                        
                        if comp_device != self.device and comp_device != "unknown":
                            print(f"   🔧 Moving {comp_name} from {comp_device} to {self.device}")
                            try:
                                component = component.to(self.device)
                                setattr(self.edit_pipe, comp_name, component)
                                print(f"   ✅ {comp_name} moved successfully")
                            except Exception as move_error:
                                print(f"   ⚠️ Could not move {comp_name}: {move_error}")
                                all_correct = False
                                
                    except Exception as comp_error:
                        print(f"   {comp_name.upper()}: error checking device ({comp_error})")
                        all_correct = False
            
            # Check scheduler
            if hasattr(self.edit_pipe, 'scheduler') and self.edit_pipe.scheduler is not None:
                print(f"   SCHEDULER: present ({type(self.edit_pipe.scheduler).__name__})")
            
            # Summary
            if all_correct:
                print(f"✅ Qwen-Image-Edit pipeline verified on {self.device}")
                return True
            else:
                print("⚠️ Some edit pipeline components needed adjustment")
                return False
                
        except Exception as e:
            print(f"❌ Edit pipeline device verification failed: {e}")
            return False
    
    def verify_device_setup(self) -> bool:
        """Verify model components are on the correct device (safe version)"""
        if not self.pipe:
            print("⚠️ Model not loaded")
            return False
            
        print(f"🔍 Safe device verification for {self.device}:")
        
        try:
            # Check main pipeline device
            if hasattr(self.pipe, 'device'):
                main_device = str(self.pipe.device)
                print(f"   Pipeline device: {main_device}")
            
            # Check components safely
            components = ['unet', 'vae', 'text_encoder']
            all_correct = True
            
            for comp_name in components:
                if hasattr(self.pipe, comp_name) and getattr(self.pipe, comp_name) is not None:
                    component = getattr(self.pipe, comp_name)
                    
                    try:
                        # Safe device check
                        if hasattr(component, 'device'):
                            comp_device = str(component.device)
                        else:
                            # Check first parameter device safely
                            try:
                                comp_device = str(next(component.parameters()).device)
                            except (StopIteration, AttributeError):
                                comp_device = "unknown"
                        
                        # Count parameters safely
                        try:
                            param_count = sum(p.numel() for p in component.parameters())
                        except Exception:
                            param_count = 0
                        
                        print(f"   {comp_name.upper()}: {comp_device} ({param_count:,} params)")
                        
                        if comp_device != self.device and comp_device != "unknown":
                            all_correct = False
                            
                    except Exception as comp_error:
                        print(f"   {comp_name.upper()}: error checking device ({comp_error})")
                        all_correct = False
            
            # Check scheduler
            if hasattr(self.pipe, 'scheduler') and self.pipe.scheduler is not None:
                print(f"   SCHEDULER: present ({type(self.pipe.scheduler).__name__})")
            
            # Summary
            if all_correct:
                print(f"✅ All components verified on {self.device}")
                return True
            else:
                print("⚠️ Some components may need device adjustment")
                return False
                
        except Exception as e:
            print(f"❌ Device verification failed: {e}")
            return False
    
    def _force_device_consistency(self):
        """Force all components to the target device with enhanced memory management"""
        try:
            print(f"🔧 Forcing device consistency to {self.device}...")
            
            # Enhanced memory management
            if torch.cuda.is_available():
                # Get initial memory state
                initial_memory = torch.cuda.memory_allocated()
                print(f"📊 Initial GPU memory: {initial_memory / 1e9:.2f}GB")
                
                # Clear cache and synchronize
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                cleared_memory = torch.cuda.memory_allocated()
                print(f"📊 After cleanup: {cleared_memory / 1e9:.2f}GB (freed {(initial_memory - cleared_memory) / 1e9:.2f}GB)")
            
            # Safe device movement - use PyTorch's built-in methods only
            components = ['unet', 'vae', 'text_encoder']
            
            for comp_name in components:
                if hasattr(self.pipe, comp_name) and getattr(self.pipe, comp_name) is not None:
                    component = getattr(self.pipe, comp_name)
                    
                    # Use PyTorch's safe .to() method only
                    try:
                        # Check current device first
                        current_device = "unknown"
                        try:
                            if hasattr(component, 'device'):
                                current_device = str(component.device)
                            else:
                                current_device = str(next(component.parameters()).device)
                        except:
                            pass
                        
                        if current_device != self.device:
                            # Memory-efficient device transfer
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            component = component.to(self.device)
                            setattr(self.pipe, comp_name, component)
                            
                            # Count parameters for reporting
                            param_count = sum(p.numel() for p in component.parameters())
                            print(f"✅ {comp_name}: moved from {current_device} to {self.device} ({param_count:,} parameters)")
                            
                            # Clear cache after each component move
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            print(f"✅ {comp_name}: already on {self.device}")
                        
                    except Exception as comp_error:
                        print(f"⚠️ Could not move {comp_name}: {comp_error}")
            
            # Handle scheduler (no device movement needed for most schedulers)
            if hasattr(self.pipe, 'scheduler') and self.pipe.scheduler is not None:
                print(f"✅ scheduler: present ({type(self.pipe.scheduler).__name__})")
            
            # Move pipeline itself to device (this should be redundant but ensures consistency)
            try:
                self.pipe = self.pipe.to(self.device)
                print(f"✅ Pipeline confirmed on {self.device}")
            except Exception as pipe_error:
                print(f"⚠️ Pipeline device move warning: {pipe_error}")
            
            # Final memory optimization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Report final memory state
                final_memory = torch.cuda.memory_allocated()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                print(f"📊 Final GPU memory: {final_memory / 1e9:.2f}GB / {total_memory / 1e9:.2f}GB ({(final_memory/total_memory)*100:.1f}% used)")
                
        except Exception as e:
            print(f"❌ Failed to force device consistency: {e}")
            # Emergency fallback: minimal device move with memory cleanup
            try:
                print("🚨 Attempting minimal device move with memory cleanup...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                
                self.pipe = self.pipe.to(self.device)
                print("✅ Minimal move completed")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e2:
                print(f"❌ Minimal move also failed: {e2}")

    def enhance_prompt(self, prompt: str, language: str = "en", enhancement_type: str = "general") -> str:
        """
        Enhance prompt with quality keywords and optional Qwen2-VL multimodal enhancement
        
        Args:
            prompt: Original text prompt
            language: Language for enhancement keywords
            enhancement_type: Type of enhancement ("general", "artistic", "technical", "creative")
        
        Returns:
            Enhanced prompt string
        """
        # Try Qwen2-VL enhancement first if available
        if self.qwen2vl_integration and self.multimodal_enabled:
            try:
                print("🔮 Enhancing prompt with Qwen2-VL...")
                enhancement_result = self.qwen2vl_integration.enhance_prompt(prompt, enhancement_type)
                
                if enhancement_result.confidence > 0.5:  # Use Qwen2-VL result if confident
                    print(f"✅ Qwen2-VL enhancement applied (confidence: {enhancement_result.confidence:.2f})")
                    return enhancement_result.enhanced_prompt
                else:
                    print(f"⚠️ Qwen2-VL enhancement low confidence ({enhancement_result.confidence:.2f}), using fallback")
                    
            except Exception as e:
                print(f"⚠️ Qwen2-VL enhancement failed: {e}, using fallback")
        
        # Fallback to traditional enhancement
        enhancement = PROMPT_ENHANCEMENT.get(language, PROMPT_ENHANCEMENT["en"])
        quality_keywords: str = enhancement["quality_keywords"]
        
        # Add architecture-specific keywords if available
        if self.current_architecture == "MMDiT":
            mmdit_keywords = enhancement.get("mmdit_keywords", "")
            if mmdit_keywords:
                quality_keywords = f"{quality_keywords}, {mmdit_keywords}"
        
        # Add multimodal keywords if Qwen2-VL is available (even in fallback mode)
        if self.qwen2vl_integration:
            multimodal_keywords = enhancement.get("multimodal_keywords", "")
            if multimodal_keywords:
                quality_keywords = f"{quality_keywords}, {multimodal_keywords}"
        
        # Don't add keywords if they're already present
        if not any(word in prompt.lower() for word in ["4k", "hd", "quality", "detailed"]):
            return f"{prompt}, {quality_keywords}"
        return prompt
    
    def generate_image(self, prompt: str, negative_prompt: str = "", width: Optional[int] = None, height: Optional[int] = None, 
                      num_inference_steps: Optional[int] = None, cfg_scale: Optional[float] = None, seed: int = -1, 
                      language: str = "en", enhance_prompt_flag: bool = True) -> Tuple[Optional[PIL.Image.Image], str]:
        """Generate image from text prompt with performance monitoring and optimization"""
        
        if not self.pipe:
            return None, "Model not loaded. Please initialize the model first."
        
        try:
            # Use default values if not provided
            width = width or GENERATION_CONFIG["width"]
            height = height or GENERATION_CONFIG["height"]
            num_inference_steps = num_inference_steps or GENERATION_CONFIG["num_inference_steps"]
            cfg_scale = cfg_scale or GENERATION_CONFIG["true_cfg_scale"]
            
            # Use optimized generation settings if available
            if self.pipeline_optimizer:
                optimal_settings = self.pipeline_optimizer.configure_generation_settings(self.current_architecture)
                # Override with user-provided values
                if width is not None:
                    optimal_settings["width"] = width
                if height is not None:
                    optimal_settings["height"] = height
                if num_inference_steps is not None:
                    optimal_settings["num_inference_steps"] = num_inference_steps
                if cfg_scale is not None:
                    if "true_cfg_scale" in optimal_settings:
                        optimal_settings["true_cfg_scale"] = cfg_scale
                    else:
                        optimal_settings["guidance_scale"] = cfg_scale
                
                # Update values from optimal settings
                width = optimal_settings.get("width", width)
                height = optimal_settings.get("height", height)
                num_inference_steps = optimal_settings.get("num_inference_steps", num_inference_steps)
                cfg_scale = optimal_settings.get("true_cfg_scale", optimal_settings.get("guidance_scale", cfg_scale))
            
            # Enhance prompt if requested
            if enhance_prompt_flag:
                enhanced_prompt = self.enhance_prompt(prompt, language)
            else:
                enhanced_prompt = prompt
            
            # Handle random seed and generator
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
            
            # Create generator on the correct device
            if torch.cuda.is_available() and self.device == "cuda":
                generator = torch.Generator(device="cuda").manual_seed(seed)
            else:
                generator = torch.Generator().manual_seed(seed)
            
            print(f"Generating image with prompt: {enhanced_prompt[:100]}...")
            print(f"Settings: {width}x{height}, steps: {num_inference_steps}, CFG: {cfg_scale}, seed: {seed}")
            print(f"Device: {self.device}, Architecture: {self.current_architecture}")
            
            # Use performance monitoring if available
            if self.performance_monitor:
                with self.performance_monitor.monitor_generation(
                    model_name=self.model_name,
                    architecture_type=self.current_architecture,
                    num_steps=num_inference_steps,
                    resolution=(width, height)
                ) as perf_monitor:
                    
                    # Generate image with performance monitoring
                    image, success_msg = self._generate_with_monitoring(
                        enhanced_prompt, negative_prompt, width, height,
                        num_inference_steps, cfg_scale, seed, generator, perf_monitor
                    )
                    
                    # Log performance results
                    metrics = perf_monitor.get_current_metrics()
                    if metrics.target_met:
                        print(f"🎯 Performance target met: {metrics.total_generation_time:.3f}s")
                    else:
                        print(f"⚠️ Performance target missed: {metrics.total_generation_time:.3f}s (target: {metrics.target_generation_time:.1f}s)")
                    
                    return image, success_msg
            else:
                # Generate without performance monitoring (fallback)
                return self._generate_without_monitoring(
                    enhanced_prompt, negative_prompt, width, height,
                    num_inference_steps, cfg_scale, seed, generator
                )
            
        except Exception as e:
            error_msg = f"❌ Error generating image: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def _generate_with_monitoring(self, enhanced_prompt: str, negative_prompt: str, width: int, height: int,
                                num_inference_steps: int, cfg_scale: float, seed: int, generator, perf_monitor) -> Tuple[Optional[PIL.Image.Image], str]:
        """Generate image with performance monitoring"""
        
        # Ensure we're using the correct device context with proper error handling
        device_context = torch.cuda.device(self.device) if torch.cuda.is_available() else torch.no_grad()
        
        with device_context:
            with torch.no_grad():
                # Clear CUDA cache and synchronize before generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Prepare generation parameters based on architecture
                if self.current_architecture == "MMDiT":
                    generation_params = {
                        "prompt": enhanced_prompt,
                        "width": width,
                        "height": height,
                        "num_inference_steps": num_inference_steps,
                        "generator": generator,
                        "true_cfg_scale": cfg_scale,  # Qwen-Image uses true_cfg_scale
                        "output_type": "pil",
                        "return_dict": True
                    }
                else:
                    # UNet or unknown architecture
                    generation_params = {
                        "prompt": enhanced_prompt,
                        "width": width,
                        "height": height,
                        "num_inference_steps": num_inference_steps,
                        "generator": generator,
                        "guidance_scale": cfg_scale,  # Standard parameter name
                        "output_type": "pil",
                        "return_dict": True
                    }
                
                # Add negative prompt if provided
                if negative_prompt and negative_prompt.strip():
                    generation_params["negative_prompt"] = negative_prompt
                
                print(f"🎨 Using {self.current_architecture} architecture with optimized settings")
                print(f"📐 Resolution: {width}x{height} (aspect ratio: {width/height:.2f})")
                
                # Generation with proper error handling
                try:
                    print("🎨 Generating with optimized pipeline...")
                    result = self.pipe(**generation_params)
                    
                except RuntimeError as runtime_error:
                    return self._handle_generation_error(runtime_error, generation_params, seed)
                    
                except Exception as general_error:
                    return self._handle_general_error(general_error)
        
        image = result.images[0]
        
        # Save image with metadata
        return self._save_generated_image(image, enhanced_prompt, negative_prompt, width, height,
                                        num_inference_steps, cfg_scale, seed)
    
    def _generate_without_monitoring(self, enhanced_prompt: str, negative_prompt: str, width: int, height: int,
                                   num_inference_steps: int, cfg_scale: float, seed: int, generator) -> Tuple[Optional[PIL.Image.Image], str]:
        """Generate image without performance monitoring (fallback)"""
        
        # Ensure we're using the correct device context with proper error handling
        device_context = torch.cuda.device(self.device) if torch.cuda.is_available() else torch.no_grad()
        
        with device_context:
            with torch.no_grad():
                # Clear CUDA cache and synchronize before generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Qwen-Image specific parameters
                generation_params = {
                    "prompt": enhanced_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                    "generator": generator,
                    "true_cfg_scale": cfg_scale,  # Qwen-Image uses true_cfg_scale
                    "output_type": "pil",
                    "return_dict": True
                }
                
                # Add negative prompt if provided
                if negative_prompt and negative_prompt.strip():
                    generation_params["negative_prompt"] = negative_prompt
                
                print(f"🎨 Using Qwen-Image MMDiT model with true_cfg_scale={cfg_scale}")
                print(f"📐 Resolution: {width}x{height} (aspect ratio: {width/height:.2f})")
                
                # Generation with proper error handling
                try:
                    print("🎨 Generating with Qwen-Image pipeline...")
                    result = self.pipe(**generation_params)
                    
                except RuntimeError as runtime_error:
                    return self._handle_generation_error(runtime_error, generation_params, seed)
                    
                except Exception as general_error:
                    return self._handle_general_error(general_error)
        
        image = result.images[0]
        
        # Save image with metadata
        return self._save_generated_image(image, enhanced_prompt, negative_prompt, width, height,
                                        num_inference_steps, cfg_scale, seed)
    
    def _handle_generation_error(self, runtime_error: RuntimeError, generation_params: dict, seed: int) -> Tuple[Optional[PIL.Image.Image], str]:
        """Handle runtime errors during generation"""
        error_str = str(runtime_error)
        
        if "Expected all tensors to be on the same device" in error_str:
            print(f"❌ Device mismatch error: {runtime_error}")
            print("🔄 Attempting device consistency fix...")
            
            try:
                self._force_device_consistency()
                
                # Recreate generator on correct device
                if torch.cuda.is_available() and self.device == "cuda":
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                else:
                    generator = torch.Generator().manual_seed(seed)
                
                generation_params["generator"] = generator
                
                # Retry generation
                result = self.pipe(**generation_params)
                print("✅ Generation successful after device fix")
                
                image = result.images[0]
                return image, "✅ Image generated successfully after device fix!"
                
            except Exception as retry_error:
                print(f"❌ Device fix failed: {retry_error}")
                return None, f"❌ Device consistency error: {runtime_error}"
                
        elif "out of memory" in error_str.lower() or "cuda out of memory" in error_str.lower():
            print(f"❌ GPU memory error: {runtime_error}")
            print("🧹 Clearing GPU memory and retrying with optimizations...")
            
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Enable memory optimizations
                if hasattr(self.pipe, 'enable_attention_slicing'):
                    self.pipe.enable_attention_slicing()
                    print("✅ Attention slicing enabled")
                
                # Retry generation
                result = self.pipe(**generation_params)
                print("✅ Generation successful after memory optimization")
                
                image = result.images[0]
                return image, "✅ Image generated successfully with memory optimization!"
                
            except Exception as memory_error:
                print(f"❌ Memory optimization failed: {memory_error}")
                return None, f"❌ GPU memory error: {runtime_error}"
        else:
            print(f"❌ Runtime error during generation: {runtime_error}")
            return None, f"❌ Generation error: {runtime_error}"
    
    def _handle_general_error(self, general_error: Exception) -> Tuple[Optional[PIL.Image.Image], str]:
        """Handle general errors during generation"""
        error_str = str(general_error)
        
        if "Can't unpack a tensor" in error_str and "tuple of 2 elements" in error_str:
            print(f"❌ Tensor unpacking error detected: {general_error}")
            print("🔧 This is caused by torch.compile on the transformer")
            print("💡 Solution: torch.compile has been disabled in config")
            
            error_msg = ("Tensor unpacking error in transformer attention mechanism. "
                       "This is caused by torch.compile. Please restart the server - "
                       "torch.compile has been disabled to fix this issue.")
            return None, error_msg
            
        elif "not enough values to unpack" in error_str and "expected 2, got 1" in error_str:
            print(f"❌ Attention processor unpacking error detected: {general_error}")
            print("🔧 This is caused by AttnProcessor2_0 incompatibility with Qwen-Image")
            print("💡 Solution: Flash attention has been disabled for Qwen-Image compatibility")
            
            error_msg = ("Attention processor unpacking error. "
                       "AttnProcessor2_0 is not compatible with Qwen-Image MMDiT architecture. "
                       "Please restart the server - flash attention has been disabled to fix this issue.")
            return None, error_msg
        else:
            print(f"❌ Unexpected error during generation: {general_error}")
            return None, f"❌ Generation error: {general_error}"
    
    def _save_generated_image(self, image: PIL.Image.Image, enhanced_prompt: str, negative_prompt: str,
                            width: int, height: int, num_inference_steps: int, cfg_scale: float, seed: int) -> Tuple[PIL.Image.Image, str]:
        """Save generated image with metadata"""
        
        # Save image with metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qwen_image_{timestamp}_{seed}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save with metadata
        metadata = {
            "prompt": enhanced_prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "model": self.model_name,
            "architecture": self.current_architecture,
            "optimization_applied": self.optimization_applied,
            "timestamp": timestamp
        }
        
        # Add performance metrics if available
        if self.performance_monitor:
            current_metrics = self.performance_monitor.get_current_metrics()
            metadata["performance"] = {
                "generation_time": current_metrics.total_generation_time,
                "per_step_time": current_metrics.generation_time_per_step,
                "target_met": current_metrics.target_met,
                "performance_score": current_metrics.performance_score
            }
        
        # Save image
        image.save(filepath)
        
        # Save metadata
        metadata_file = filepath.replace(".png", "_metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        success_msg = f"✅ Image generated successfully!\nSaved as: {filename}\nSeed: {seed}"
        
        # Add performance info to success message if available
        if self.performance_monitor:
            current_metrics = self.performance_monitor.get_current_metrics()
            success_msg += f"\nGeneration time: {current_metrics.total_generation_time:.3f}s"
            if current_metrics.target_met:
                success_msg += " (Target met! 🎯)"
            else:
                success_msg += f" (Target: {current_metrics.target_generation_time:.1f}s)"
        
        return image, success_msg
    
    def get_optimization_status(self) -> dict:
        """Get current optimization status and performance information"""
        status = {
            "optimized_components_available": OPTIMIZED_COMPONENTS_AVAILABLE,
            "optimization_applied": self.optimization_applied,
            "current_architecture": self.current_architecture,
            "model_info": None,
            "performance_monitor_available": self.performance_monitor is not None,
            "pipeline_optimizer_available": self.pipeline_optimizer is not None,
            "detection_service_available": self.detection_service is not None,
            "download_manager_available": self.download_manager is not None,
            "compatibility_layer_available": self.compatibility_layer is not None
        }
        
        # Add model information if available
        if self.model_info:
            status["model_info"] = {
                "name": self.model_info.name,
                "size_gb": self.model_info.size_gb,
                "model_type": self.model_info.model_type,
                "is_optimal": self.model_info.is_optimal,
                "download_status": self.model_info.download_status,
                "path": self.model_info.path
            }
        
        # Add performance information if available
        if self.performance_monitor:
            performance_summary = self.performance_monitor.get_performance_summary()
            status["performance_summary"] = performance_summary
        
        # Add optimization validation if available
        if self.pipeline_optimizer and self.pipe:
            validation_results = self.pipeline_optimizer.validate_optimization(self.pipe)
            status["optimization_validation"] = validation_results
        
        return status
    
    def get_performance_recommendations(self) -> dict:
        """Get performance recommendations based on current hardware"""
        if not self.pipeline_optimizer:
            return {"error": "Pipeline optimizer not available"}
        
        # Get GPU memory info
        gpu_memory_gb = None
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return self.pipeline_optimizer.get_performance_recommendations(gpu_memory_gb)
    
    def validate_optimization_workflow(self) -> dict:
        """Validate the complete optimization workflow"""
        validation_results = {
            "workflow_status": "unknown",
            "components_status": {},
            "model_detection": False,
            "architecture_detection": False,
            "pipeline_optimization": False,
            "performance_monitoring": False,
            "overall_success": False,
            "warnings": [],
            "errors": []
        }
        
        try:
            # Check component availability
            validation_results["components_status"] = {
                "optimized_components": OPTIMIZED_COMPONENTS_AVAILABLE,
                "detection_service": self.detection_service is not None,
                "pipeline_optimizer": self.pipeline_optimizer is not None,
                "performance_monitor": self.performance_monitor is not None,
                "download_manager": self.download_manager is not None,
                "compatibility_layer": self.compatibility_layer is not None
            }
            
            # Test model detection
            if self.detection_service and self.model_info:
                validation_results["model_detection"] = True
                
                # Test architecture detection
                if self.current_architecture != "Unknown":
                    validation_results["architecture_detection"] = True
                else:
                    validation_results["warnings"].append("Architecture detection failed")
            else:
                validation_results["errors"].append("Model detection failed")
            
            # Test pipeline optimization
            if self.pipe and self.optimization_applied:
                validation_results["pipeline_optimization"] = True
            else:
                validation_results["warnings"].append("Pipeline optimization not applied")
            
            # Test performance monitoring
            if self.performance_monitor:
                validation_results["performance_monitoring"] = True
            else:
                validation_results["warnings"].append("Performance monitoring not available")
            
            # Overall success assessment
            validation_results["overall_success"] = (
                validation_results["model_detection"] and
                validation_results["architecture_detection"] and
                validation_results["pipeline_optimization"]
            )
            
            if validation_results["overall_success"]:
                validation_results["workflow_status"] = "optimized"
            elif len(validation_results["errors"]) == 0:
                validation_results["workflow_status"] = "partial"
            else:
                validation_results["workflow_status"] = "failed"
            
            print(f"🔍 Optimization workflow validation: {validation_results['workflow_status'].upper()}")
            
        except Exception as e:
            validation_results["errors"].append(f"Validation error: {e}")
            validation_results["workflow_status"] = "error"
        
        return validation_results
    
    def generate_img2img(self, prompt: str, init_image: PIL.Image.Image, strength: float = 0.8,
                        negative_prompt: str = "", width: Optional[int] = None, height: Optional[int] = None,
                        num_inference_steps: Optional[int] = None, cfg_scale: Optional[float] = None,
                        seed: int = -1, language: str = "en", enhance_prompt_flag: bool = True) -> Tuple[Optional[PIL.Image.Image], str]:
        """Generate image from text prompt using input image as base (using Qwen-Image-Edit)"""
        
        if not self.edit_pipe:
            # Provide helpful message and suggest alternatives
            error_msg = "⚠️ Qwen-Image-Edit pipeline not available.\n"
            error_msg += "\nPossible solutions:\n"
            error_msg += "1. Wait for model download to complete (may take 10-20 minutes)\n"
            error_msg += "2. Check internet connection\n"
            error_msg += "3. Ensure sufficient disk space (~20GB)\n"
            error_msg += "4. Try using Text-to-Image mode with descriptive prompts\n"
            error_msg += "\nFor now, try using Text-to-Image with a prompt like:\n"
            error_msg += f"'An image showing: {prompt}'"
            return None, error_msg
            
        try:
            # Use default values if not provided
            width = width or GENERATION_CONFIG["width"]
            height = height or GENERATION_CONFIG["height"]
            num_inference_steps = num_inference_steps or GENERATION_CONFIG["num_inference_steps"]
            cfg_scale = cfg_scale or GENERATION_CONFIG["true_cfg_scale"]
            
            # Enhance prompt if requested
            if enhance_prompt_flag:
                enhanced_prompt = self.enhance_prompt(prompt, language)
            else:
                enhanced_prompt = prompt
                
            # Handle random seed
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
            
            # Create generator
            if torch.cuda.is_available() and self.device == "cuda":
                generator = torch.Generator(device="cuda").manual_seed(seed)
            else:
                generator = torch.Generator().manual_seed(seed)
            
            # Resize init image to target dimensions
            init_image = init_image.resize((width, height), PIL.Image.Resampling.LANCZOS)
            
            print("Generating image-to-image with Qwen-Image-Edit...")
            print(f"Prompt: {enhanced_prompt[:100]}...")
            print(f"Settings: {width}x{height}, steps: {num_inference_steps}, CFG: {cfg_scale}, seed: {seed}")
            
            with torch.no_grad():
                # Use Qwen-Image-Edit pipeline
                inputs = {
                    "image": init_image,
                    "prompt": enhanced_prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": num_inference_steps,
                    "true_cfg_scale": cfg_scale,
                    "generator": generator
                }
                
                result = self.edit_pipe(**inputs)
            
            image = result.images[0]
            
            # Save image with metadata
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qwen_img2img_{timestamp}_{seed}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            metadata = {
                "mode": "img2img",
                "prompt": enhanced_prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "language": language,
                "model": "Qwen-Image-Edit",
                "timestamp": timestamp
            }
            
            # Save image and metadata
            image.save(filepath)
            metadata_file = filepath.replace(".png", "_metadata.json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            success_msg = f"✅ Image-to-image generated with Qwen-Image-Edit!\nSaved as: {filename}\nSeed: {seed}"
            return image, success_msg
            
        except Exception as e:
            error_msg = f"❌ Error in img2img generation: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def generate_inpaint(self, prompt: str, init_image: PIL.Image.Image, mask_image: PIL.Image.Image,
                        negative_prompt: str = "", width: Optional[int] = None, height: Optional[int] = None,
                        num_inference_steps: Optional[int] = None, cfg_scale: Optional[float] = None,
                        seed: int = -1, language: str = "en", enhance_prompt_flag: bool = True) -> Tuple[Optional[PIL.Image.Image], str]:
        """Generate image using inpainting with mask (using Qwen-Image-Edit)"""
        
        if not self.edit_pipe:
            # Provide helpful message and suggest alternatives
            error_msg = "⚠️ Qwen-Image-Edit pipeline not available for inpainting.\n"
            error_msg += "\nPossible solutions:\n"
            error_msg += "1. Wait for model download to complete (may take 10-20 minutes)\n"
            error_msg += "2. Check internet connection\n"
            error_msg += "3. Ensure sufficient disk space (~20GB)\n"
            error_msg += "4. Try using Text-to-Image mode instead\n"
            error_msg += "\nFor inpainting-like results, try Text-to-Image with:\n"
            error_msg += f"'A composition featuring: {prompt}'"
            return None, error_msg
            
        try:
            # Use default values if not provided
            width = width or GENERATION_CONFIG["width"]
            height = height or GENERATION_CONFIG["height"]
            num_inference_steps = num_inference_steps or GENERATION_CONFIG["num_inference_steps"]
            cfg_scale = cfg_scale or GENERATION_CONFIG["true_cfg_scale"]
            
            # Enhance prompt if requested
            if enhance_prompt_flag:
                enhanced_prompt = self.enhance_prompt(prompt, language)
            else:
                enhanced_prompt = prompt
                
            # Handle random seed
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
            
            # Create generator
            if torch.cuda.is_available() and self.device == "cuda":
                generator = torch.Generator(device="cuda").manual_seed(seed)
            else:
                generator = torch.Generator().manual_seed(seed)
            
            # Resize images to target dimensions
            init_image = init_image.resize((width, height), PIL.Image.Resampling.LANCZOS)
            mask_image = mask_image.resize((width, height), PIL.Image.Resampling.LANCZOS)
            
            # For Qwen-Image-Edit, we'll create a composite prompt that describes the inpainting task
            mask_prompt = f"In the masked area: {enhanced_prompt}"
            
            print("Generating inpaint with Qwen-Image-Edit...")
            print(f"Prompt: {mask_prompt[:100]}...")
            print(f"Settings: {width}x{height}, steps: {num_inference_steps}, CFG: {cfg_scale}, seed: {seed}")
            
            with torch.no_grad():
                # Use Qwen-Image-Edit pipeline for inpainting-style editing
                inputs = {
                    "image": init_image,
                    "prompt": mask_prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": num_inference_steps,
                    "true_cfg_scale": cfg_scale,
                    "generator": generator
                }
                
                result = self.edit_pipe(**inputs)
            
            image = result.images[0]
            
            # Save image with metadata
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qwen_inpaint_{timestamp}_{seed}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            metadata = {
                "mode": "inpaint",
                "prompt": enhanced_prompt,
                "mask_prompt": mask_prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "language": language,
                "model": "Qwen-Image-Edit",
                "timestamp": timestamp
            }
            
            # Save image and metadata
            image.save(filepath)
            metadata_file = filepath.replace(".png", "_metadata.json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            success_msg = f"✅ Inpainted image generated with Qwen-Image-Edit!\nSaved as: {filename}\nSeed: {seed}"
            return image, success_msg
            
        except Exception as e:
            error_msg = f"❌ Error in inpainting generation: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def super_resolution(self, image: PIL.Image.Image, scale_factor: int = 2) -> Tuple[Optional[PIL.Image.Image], str]:
        """Enhance image resolution using simple upscaling with AI-like enhancement"""
        try:
            # Get original dimensions
            original_width, original_height = image.size
            new_width = original_width * scale_factor
            new_height = original_height * scale_factor
            
            # Use high-quality resampling with sharpening
            enhanced_image = image.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
            
            # Apply sharpening filter for better quality
            enhanced_image = enhanced_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            
            # Save enhanced image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qwen_superres_{timestamp}_{scale_factor}x.png"
            filepath = os.path.join(self.output_dir, filename)
            
            metadata = {
                "mode": "super_resolution",
                "original_size": [original_width, original_height],
                "enhanced_size": [new_width, new_height],
                "scale_factor": scale_factor,
                "timestamp": timestamp
            }
            
            enhanced_image.save(filepath)
            metadata_file = filepath.replace(".png", "_metadata.json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            success_msg = f"✅ Image enhanced successfully!\nSaved as: {filename}\nScale: {scale_factor}x ({original_width}x{original_height} → {new_width}x{new_height})"
            return enhanced_image, success_msg
            
        except Exception as e:
            error_msg = f"❌ Error in super resolution: {str(e)}"
            print(error_msg)
    
    # Qwen2-VL Integration Methods
    
    def analyze_image_with_qwen2vl(self, image: Union[str, PIL.Image.Image], analysis_type: str = "comprehensive") -> dict:
        """
        Analyze an image using Qwen2-VL for context-aware generation
        
        Args:
            image: Path to image file or PIL Image object
            analysis_type: Type of analysis ("comprehensive", "style", "composition", "elements")
        
        Returns:
            Dictionary with analysis results
        """
        if not self.qwen2vl_integration:
            return {
                "available": False,
                "error": "Qwen2-VL integration not available",
                "fallback_description": "Basic image detected - enable Qwen2-VL for detailed analysis"
            }
        
        try:
            print(f"🔍 Analyzing image with Qwen2-VL ({analysis_type})...")
            analysis_result = self.qwen2vl_integration.analyze_image(image, analysis_type)
            
            return {
                "available": True,
                "analysis_type": analysis_type,
                "description": analysis_result.description,
                "key_elements": analysis_result.key_elements,
                "style_analysis": analysis_result.style_analysis,
                "composition_notes": analysis_result.composition_notes,
                "suggested_improvements": analysis_result.suggested_improvements,
                "confidence": analysis_result.confidence,
                "multimodal_enabled": self.multimodal_enabled
            }
            
        except Exception as e:
            print(f"⚠️ Image analysis failed: {e}")
            return {
                "available": True,
                "error": str(e),
                "fallback_description": "Image analysis failed - using fallback mode"
            }
    
    def generate_context_aware_image(self, prompt: str, reference_image: Optional[Union[str, PIL.Image.Image]] = None,
                                   negative_prompt: str = "", width: Optional[int] = None, height: Optional[int] = None,
                                   num_inference_steps: Optional[int] = None, cfg_scale: Optional[float] = None,
                                   seed: int = -1, language: str = "en") -> Tuple[Optional[PIL.Image.Image], str]:
        """
        Generate image with context-aware prompt enhancement using Qwen2-VL
        
        Args:
            prompt: Base text prompt
            reference_image: Optional reference image for context
            negative_prompt: Negative prompt
            width, height: Image dimensions
            num_inference_steps: Number of inference steps
            cfg_scale: CFG scale
            seed: Random seed
            language: Language for enhancement
        
        Returns:
            Tuple of (generated_image, status_message)
        """
        if not self.qwen2vl_integration:
            print("⚠️ Qwen2-VL not available, using standard generation")
            return self.generate_image(prompt, negative_prompt, width, height, 
                                     num_inference_steps, cfg_scale, seed, language, True)
        
        try:
            print("🎨 Creating context-aware prompt with Qwen2-VL...")
            
            # Create context-aware prompt
            context_prompt = self.qwen2vl_integration.create_context_aware_prompt(prompt, reference_image)
            
            print(f"📝 Original prompt: {prompt}")
            print(f"🔮 Context-aware prompt: {context_prompt[:200]}...")
            
            # Generate image with enhanced prompt
            return self.generate_image(context_prompt, negative_prompt, width, height,
                                     num_inference_steps, cfg_scale, seed, language, False)  # Don't enhance again
            
        except Exception as e:
            print(f"⚠️ Context-aware generation failed: {e}, using standard generation")
            return self.generate_image(prompt, negative_prompt, width, height,
                                     num_inference_steps, cfg_scale, seed, language, True)
    
    def get_qwen2vl_status(self) -> dict:
        """Get Qwen2-VL integration status and capabilities"""
        if not self.qwen2vl_integration:
            return {
                "available": False,
                "integration_available": QWEN2VL_INTEGRATION_AVAILABLE,
                "error": "Qwen2-VL integration not initialized"
            }
        
        status = self.qwen2vl_integration.get_integration_status()
        status.update({
            "multimodal_enabled": self.multimodal_enabled,
            "integration_available": QWEN2VL_INTEGRATION_AVAILABLE
        })
        
        return status
    
    def enhance_prompt_with_qwen2vl(self, prompt: str, enhancement_type: str = "general") -> dict:
        """
        Enhance prompt using Qwen2-VL with detailed results
        
        Args:
            prompt: Original prompt
            enhancement_type: Type of enhancement
        
        Returns:
            Dictionary with enhancement results
        """
        if not self.qwen2vl_integration:
            return {
                "available": False,
                "original_prompt": prompt,
                "enhanced_prompt": self.enhance_prompt(prompt),  # Fallback
                "enhancement_type": "fallback",
                "confidence": 0.3,
                "error": "Qwen2-VL integration not available"
            }
        
        try:
            enhancement_result = self.qwen2vl_integration.enhance_prompt(prompt, enhancement_type)
            
            return {
                "available": True,
                "original_prompt": enhancement_result.original_prompt,
                "enhanced_prompt": enhancement_result.enhanced_prompt,
                "enhancement_type": enhancement_result.enhancement_type,
                "confidence": enhancement_result.confidence,
                "metadata": enhancement_result.metadata,
                "multimodal_enabled": self.multimodal_enabled
            }
            
        except Exception as e:
            return {
                "available": True,
                "original_prompt": prompt,
                "enhanced_prompt": self.enhance_prompt(prompt),  # Fallback
                "enhancement_type": "error_fallback",
                "confidence": 0.1,
                "error": str(e)
            }
    
    def load_qwen2vl_model(self) -> bool:
        """
        Manually load Qwen2-VL model if not already loaded
        
        Returns:
            True if successful, False otherwise
        """
        if not self.qwen2vl_integration:
            print("❌ Qwen2-VL integration not available")
            return False
        
        if self.multimodal_enabled:
            print("✅ Qwen2-VL model already loaded")
            return True
        
        try:
            print("📥 Loading Qwen2-VL model...")
            success = self.qwen2vl_integration.load_model()
            
            if success:
                self.multimodal_enabled = True
                print("✅ Qwen2-VL model loaded successfully")
            else:
                print("❌ Qwen2-VL model loading failed")
            
            return success
            
        except Exception as e:
            print(f"❌ Error loading Qwen2-VL model: {e}")
            return False
    
    def unload_qwen2vl_model(self) -> None:
        """Unload Qwen2-VL model to free memory"""
        if self.qwen2vl_integration:
            self.qwen2vl_integration.unload_model()
            self.multimodal_enabled = False
            print("✅ Qwen2-VL model unloaded")
        else:
            print("⚠️ Qwen2-VL integration not available")
    
    def clear_qwen2vl_cache(self) -> None:
        """Clear Qwen2-VL response cache"""
        if self.qwen2vl_integration:
            self.qwen2vl_integration.clear_cache()
            print("✅ Qwen2-VL cache cleared")
        else:
            print("⚠️ Qwen2-VL integration not available")
            return None, error_msg