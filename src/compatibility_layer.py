"""
Backward Compatibility Layer with Multimodal Support
Maintains existing API interface while using optimized backend components
"""

import json
import os
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
from PIL import Image

# Import existing components
try:
    # Try relative imports first (when running from src/)
    from model_detection_service import ModelDetectionService, ModelInfo
    from pipeline_optimizer import PipelineOptimizer, OptimizationConfig
    from qwen_image_config import MODEL_CONFIG, GENERATION_CONFIG, MEMORY_CONFIG
except ImportError:
    # Fall back to absolute imports (when running from project root)
    from src.model_detection_service import ModelDetectionService, ModelInfo
    from src.pipeline_optimizer import PipelineOptimizer, OptimizationConfig
    from src.qwen_image_config import MODEL_CONFIG, GENERATION_CONFIG, MEMORY_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LegacyConfig:
    """Legacy configuration structure for backward compatibility"""
    model_name: str
    device: str
    torch_dtype: torch.dtype
    generation_settings: Dict[str, Any]
    memory_settings: Dict[str, bool]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        config_dict = asdict(self)
        # Handle torch.dtype serialization
        config_dict['torch_dtype'] = str(self.torch_dtype)
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LegacyConfig':
        """Create from dictionary"""
        # Handle torch.dtype deserialization
        if isinstance(config_dict.get('torch_dtype'), str):
            dtype_str = config_dict['torch_dtype']
            if 'float16' in dtype_str:
                config_dict['torch_dtype'] = torch.float16
            elif 'bfloat16' in dtype_str:
                config_dict['torch_dtype'] = torch.bfloat16
            elif 'float32' in dtype_str:
                config_dict['torch_dtype'] = torch.float32
            else:
                config_dict['torch_dtype'] = torch.float16  # Default
        
        return cls(**config_dict)


class Qwen2VLIntegration:
    """Optional Qwen2-VL integration for enhanced text understanding"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.available = False
        self.model_path = None
        
        # Try to detect and load Qwen2-VL
        self._detect_and_load()
    
    def _detect_and_load(self) -> None:
        """Detect and load available Qwen2-VL model"""
        try:
            detection_service = ModelDetectionService()
            qwen2_vl_info = detection_service.detect_qwen2_vl_capabilities()
            
            if qwen2_vl_info["integration_possible"] and qwen2_vl_info["recommended_model"]:
                recommended = qwen2_vl_info["recommended_model"]
                self.model_path = recommended["path"]
                
                logger.info(f"🔍 Found Qwen2-VL model: {recommended['name']}")
                logger.info(f"   Size: {recommended['size_gb']:.1f}GB")
                logger.info(f"   Capabilities: {', '.join(recommended['capabilities'])}")
                
                # Load the model (lazy loading - only when needed)
                self.available = True
                logger.info("✅ Qwen2-VL integration available")
            else:
                logger.info("💡 Qwen2-VL not available - text understanding will use basic methods")
                
        except Exception as e:
            logger.warning(f"Could not initialize Qwen2-VL integration: {e}")
            self.available = False
    
    def _load_model(self) -> bool:
        """Lazy load the Qwen2-VL model when first needed"""
        if self.model is not None:
            return True
        
        if not self.available or not self.model_path:
            return False
        
        try:
            logger.info("📥 Loading Qwen2-VL model for text enhancement...")
            
            # Import transformers components
            from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
            
            # Load model components
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            logger.info("✅ Qwen2-VL model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Qwen2-VL model: {e}")
            self.available = False
            return False
    
    def enhance_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Enhance prompt using Qwen2-VL for better text understanding
        
        Args:
            prompt: Original prompt
            context: Optional context information
            
        Returns:
            Enhanced prompt
        """
        if not self.available:
            return prompt
        
        try:
            # Lazy load model
            if not self._load_model():
                return prompt
            
            # Create enhancement prompt for Qwen2-VL
            enhancement_prompt = f"""
            Please enhance this image generation prompt to be more descriptive and specific:
            
            Original prompt: "{prompt}"
            {f"Context: {context}" if context else ""}
            
            Enhanced prompt:"""
            
            # Process with Qwen2-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": enhancement_prompt}
                    ]
                }
            ]
            
            # Prepare inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            )
            
            # Generate enhanced prompt
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0.7
                )
            
            # Decode response
            generated_text = self.processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0]
            
            # Extract enhanced prompt from response
            if "Enhanced prompt:" in generated_text:
                enhanced = generated_text.split("Enhanced prompt:")[-1].strip()
                if enhanced and len(enhanced) > len(prompt) * 0.8:  # Sanity check
                    logger.info(f"✨ Prompt enhanced by Qwen2-VL")
                    return enhanced
            
            return prompt
            
        except Exception as e:
            logger.warning(f"Prompt enhancement failed: {e}")
            return prompt
    
    def analyze_image_for_context(self, image: Image.Image) -> str:
        """
        Analyze image to provide context for generation
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Context description
        """
        if not self.available:
            return ""
        
        try:
            # Lazy load model
            if not self._load_model():
                return ""
            
            # Create analysis prompt
            analysis_prompt = "Describe this image in detail, focusing on style, composition, and visual elements:"
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": analysis_prompt}
                    ]
                }
            ]
            
            # Process inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            
            # Generate analysis
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.7
                )
            
            # Decode response
            analysis = self.processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0]
            
            logger.info("🔍 Image analyzed by Qwen2-VL")
            return analysis.strip()
            
        except Exception as e:
            logger.warning(f"Image analysis failed: {e}")
            return ""


class CompatibilityLayer:
    """
    Backward compatibility layer that maintains existing API interface
    while using optimized backend components
    """
    
    def __init__(self):
        self.detection_service = ModelDetectionService()
        self.pipeline_optimizer = None
        self.optimized_pipeline = None
        self.qwen2_vl = Qwen2VLIntegration()
        
        # Legacy configuration tracking
        self.legacy_config = None
        self.config_file_path = "config/legacy_config.json"
        
        # Architecture detection
        self.current_architecture = "Unknown"
        self.backend_switched = False
        
        logger.info("🔧 CompatibilityLayer initialized")
    
    def migrate_existing_config(self, existing_config: Optional[Dict[str, Any]] = None) -> LegacyConfig:
        """
        Migrate existing user configuration to new format
        
        Args:
            existing_config: Existing configuration dictionary
            
        Returns:
            Migrated legacy configuration
        """
        logger.info("🔄 Migrating existing configuration...")
        
        # Load existing config if not provided
        if existing_config is None:
            existing_config = self._load_existing_config()
        
        # Create default legacy config
        legacy_config = LegacyConfig(
            model_name=MODEL_CONFIG.get("model_name", "Qwen/Qwen-Image"),
            device=MODEL_CONFIG.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            torch_dtype=MODEL_CONFIG.get("torch_dtype", torch.float16),
            generation_settings=GENERATION_CONFIG.copy(),
            memory_settings=MEMORY_CONFIG.copy()
        )
        
        # Migrate existing settings
        if existing_config:
            # Model settings
            if "model_name" in existing_config:
                legacy_config.model_name = existing_config["model_name"]
            
            if "device" in existing_config:
                legacy_config.device = existing_config["device"]
            
            if "torch_dtype" in existing_config:
                if isinstance(existing_config["torch_dtype"], str):
                    if "bfloat16" in existing_config["torch_dtype"]:
                        legacy_config.torch_dtype = torch.bfloat16
                    elif "float16" in existing_config["torch_dtype"]:
                        legacy_config.torch_dtype = torch.float16
                    elif "float32" in existing_config["torch_dtype"]:
                        legacy_config.torch_dtype = torch.float32
                else:
                    legacy_config.torch_dtype = existing_config["torch_dtype"]
            
            # Generation settings
            if "generation_settings" in existing_config:
                legacy_config.generation_settings.update(existing_config["generation_settings"])
            
            # Memory settings
            if "memory_settings" in existing_config:
                legacy_config.memory_settings.update(existing_config["memory_settings"])
            
            # Handle individual settings for backward compatibility
            for key in ["width", "height", "num_inference_steps", "true_cfg_scale"]:
                if key in existing_config:
                    legacy_config.generation_settings[key] = existing_config[key]
        
        # Save migrated config
        self._save_legacy_config(legacy_config)
        self.legacy_config = legacy_config
        
        logger.info("✅ Configuration migration complete")
        logger.info(f"   Model: {legacy_config.model_name}")
        logger.info(f"   Device: {legacy_config.device}")
        logger.info(f"   Precision: {legacy_config.torch_dtype}")
        
        return legacy_config
    
    def _load_existing_config(self) -> Dict[str, Any]:
        """Load existing configuration from various sources"""
        config = {}
        
        # Try to load from various config files
        config_paths = [
            "config/ui_config.json",
            "qwen_image_config.py",
            self.config_file_path
        ]
        
        for config_path in config_paths:
            try:
                if config_path.endswith('.json') and os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        file_config = json.load(f)
                        config.update(file_config)
                        logger.info(f"📁 Loaded config from {config_path}")
                        
                elif config_path.endswith('.py') and os.path.exists(config_path):
                    # Import configuration from Python file
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("config", config_path)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    
                    # Extract relevant configuration
                    for attr_name in dir(config_module):
                        if attr_name.isupper() and not attr_name.startswith('_'):
                            attr_value = getattr(config_module, attr_name)
                            if isinstance(attr_value, (dict, str, int, float, bool)):
                                config[attr_name.lower()] = attr_value
                    
                    logger.info(f"📁 Loaded config from {config_path}")
                    
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
        
        return config
    
    def _save_legacy_config(self, config: LegacyConfig) -> None:
        """Save legacy configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file_path), exist_ok=True)
            
            with open(self.config_file_path, 'w', encoding='utf-8') as f:
                json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Legacy configuration saved to {self.config_file_path}")
            
        except Exception as e:
            logger.warning(f"Could not save legacy config: {e}")
    
    def detect_and_switch_backend(self) -> bool:
        """
        Detect current model and switch to optimal backend if needed
        
        Returns:
            True if backend was switched or is already optimal
        """
        logger.info("🔍 Detecting current model and backend requirements...")
        
        try:
            # Detect current model
            current_model = self.detection_service.detect_current_model()
            
            if not current_model:
                logger.warning("❌ No model detected - optimization needed")
                return False
            
            # Detect architecture
            self.current_architecture = self.detection_service.detect_model_architecture(current_model)
            logger.info(f"🏗️ Detected architecture: {self.current_architecture}")
            
            # Check if optimization is needed
            optimization_needed = self.detection_service.is_optimization_needed()
            
            if optimization_needed:
                logger.info("🔄 Backend switching required for optimal performance")
                
                # Get recommended model
                recommended_model = self.detection_service.get_recommended_model()
                
                # Create optimized pipeline configuration
                optimization_config = OptimizationConfig(
                    torch_dtype=self.legacy_config.torch_dtype if self.legacy_config else torch.float16,
                    device=self.legacy_config.device if self.legacy_config else "cuda",
                    architecture_type=self.current_architecture,
                    # Disable memory-saving features for high-VRAM GPUs
                    enable_attention_slicing=False,
                    enable_vae_slicing=False,
                    enable_cpu_offload=False,
                )
                
                # Initialize pipeline optimizer
                self.pipeline_optimizer = PipelineOptimizer(optimization_config)
                
                # Create optimized pipeline
                self.optimized_pipeline = self.pipeline_optimizer.create_optimized_pipeline(
                    recommended_model, 
                    self.current_architecture
                )
                
                self.backend_switched = True
                logger.info("✅ Backend successfully switched to optimized pipeline")
                
            else:
                logger.info("✅ Current backend is already optimal")
                self.backend_switched = False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Backend switching failed: {e}")
            return False
    
    def get_legacy_interface(self) -> 'LegacyQwenImageGenerator':
        """
        Get a legacy-compatible interface that preserves existing API
        
        Returns:
            Legacy interface wrapper
        """
        return LegacyQwenImageGenerator(self)
    
    def generate_image_with_compatibility(
        self, 
        prompt: str, 
        **kwargs
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Generate image with full backward compatibility
        
        Args:
            prompt: Text prompt for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated image(s)
        """
        try:
            # Enhance prompt with Qwen2-VL if available
            if self.qwen2_vl.available:
                enhanced_prompt = self.qwen2_vl.enhance_prompt(prompt)
                if enhanced_prompt != prompt:
                    logger.info("✨ Using Qwen2-VL enhanced prompt")
                    prompt = enhanced_prompt
            
            # Use optimized pipeline if available
            if self.optimized_pipeline and self.pipeline_optimizer:
                # Get optimal generation settings
                generation_settings = self.pipeline_optimizer.configure_generation_settings(
                    self.current_architecture
                )
                
                # Merge with user-provided kwargs
                generation_settings.update(kwargs)
                
                # Generate with optimized pipeline
                logger.info("🚀 Generating with optimized pipeline")
                result = self.optimized_pipeline(prompt, **generation_settings)
                
                if hasattr(result, 'images'):
                    return result.images[0] if len(result.images) == 1 else result.images
                else:
                    return result
            
            else:
                # Fallback to basic generation (should not happen in normal operation)
                logger.warning("⚠️ Using fallback generation method")
                raise NotImplementedError("Fallback generation not implemented - optimized pipeline required")
                
        except Exception as e:
            logger.error(f"❌ Image generation failed: {e}")
            raise
    
    def validate_compatibility(self) -> Dict[str, Any]:
        """
        Validate that compatibility layer is working correctly
        
        Returns:
            Validation results
        """
        validation_results = {
            "config_migration": False,
            "backend_detection": False,
            "pipeline_optimization": False,
            "qwen2_vl_integration": False,
            "overall_compatibility": False,
            "warnings": [],
            "errors": []
        }
        
        try:
            # Test config migration
            if self.legacy_config:
                validation_results["config_migration"] = True
            else:
                validation_results["warnings"].append("Legacy configuration not migrated")
            
            # Test backend detection
            current_model = self.detection_service.detect_current_model()
            if current_model:
                validation_results["backend_detection"] = True
            else:
                validation_results["errors"].append("No model detected")
            
            # Test pipeline optimization
            if self.optimized_pipeline:
                validation_results["pipeline_optimization"] = True
            else:
                validation_results["warnings"].append("Optimized pipeline not available")
            
            # Test Qwen2-VL integration
            validation_results["qwen2_vl_integration"] = self.qwen2_vl.available
            if not self.qwen2_vl.available:
                validation_results["warnings"].append("Qwen2-VL integration not available")
            
            # Overall compatibility
            validation_results["overall_compatibility"] = (
                validation_results["config_migration"] and
                validation_results["backend_detection"] and
                (validation_results["pipeline_optimization"] or len(validation_results["errors"]) == 0)
            )
            
            logger.info(f"🔍 Compatibility validation: {'✅ PASSED' if validation_results['overall_compatibility'] else '⚠️ PARTIAL'}")
            
        except Exception as e:
            validation_results["errors"].append(f"Validation failed: {e}")
            logger.error(f"❌ Compatibility validation error: {e}")
        
        return validation_results


class LegacyQwenImageGenerator:
    """
    Legacy-compatible wrapper that preserves the original QwenImageGenerator interface
    while using the optimized backend transparently
    """
    
    def __init__(self, compatibility_layer: CompatibilityLayer):
        self.compatibility_layer = compatibility_layer
        self.device = compatibility_layer.legacy_config.device if compatibility_layer.legacy_config else "cuda"
        self.model_name = compatibility_layer.legacy_config.model_name if compatibility_layer.legacy_config else "Qwen/Qwen-Image"
        self.pipe = compatibility_layer.optimized_pipeline
        self.edit_pipe = None  # Maintained for compatibility but not used
        self.output_dir = "generated_images"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("🔧 Legacy QwenImageGenerator interface initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   Backend: {'Optimized' if self.pipe else 'Standard'}")
    
    def load_model(self) -> bool:
        """
        Legacy load_model method - now handled by compatibility layer
        
        Returns:
            True if model is loaded successfully
        """
        logger.info("📥 Loading model (compatibility mode)...")
        
        try:
            # Migrate configuration if not done
            if not self.compatibility_layer.legacy_config:
                self.compatibility_layer.migrate_existing_config()
            
            # Detect and switch backend
            success = self.compatibility_layer.detect_and_switch_backend()
            
            if success:
                self.pipe = self.compatibility_layer.optimized_pipeline
                logger.info("✅ Model loaded successfully (optimized backend)")
                return True
            else:
                logger.error("❌ Failed to load optimized model")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def generate_image(
        self, 
        prompt: str, 
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        true_cfg_scale: Optional[float] = None,
        **kwargs
    ) -> Image.Image:
        """
        Legacy generate_image method with preserved signature
        
        Args:
            prompt: Text prompt for generation
            width: Image width (optional)
            height: Image height (optional)
            num_inference_steps: Number of inference steps (optional)
            true_cfg_scale: CFG scale for Qwen-Image (optional)
            **kwargs: Additional parameters
            
        Returns:
            Generated PIL Image
        """
        # Prepare generation parameters
        generation_params = {}
        
        if width is not None:
            generation_params["width"] = width
        if height is not None:
            generation_params["height"] = height
        if num_inference_steps is not None:
            generation_params["num_inference_steps"] = num_inference_steps
        if true_cfg_scale is not None:
            generation_params["true_cfg_scale"] = true_cfg_scale
        
        # Add any additional kwargs
        generation_params.update(kwargs)
        
        # Generate using compatibility layer
        return self.compatibility_layer.generate_image_with_compatibility(
            prompt, **generation_params
        )
    
    def verify_device_setup(self) -> bool:
        """Legacy device verification method"""
        if self.compatibility_layer.pipeline_optimizer:
            validation = self.compatibility_layer.pipeline_optimizer.validate_optimization(self.pipe)
            return validation.get("overall_status") == "optimized"
        return self.pipe is not None
    
    def check_model_cache(self) -> Dict[str, Any]:
        """Legacy cache checking method"""
        current_model = self.compatibility_layer.detection_service.detect_current_model()
        if current_model:
            return {
                "exists": True,
                "complete": current_model.download_status == "complete",
                "size_gb": current_model.size_gb,
                "missing_components": [
                    comp for comp, exists in current_model.components.items() 
                    if not exists
                ],
                "snapshot_path": current_model.path
            }
        else:
            return {
                "exists": False,
                "complete": False,
                "size_gb": 0,
                "missing_components": [],
                "snapshot_path": None
            }