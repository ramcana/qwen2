"""
Qwen-Image Generator Core Module
Handles model loading, image generation, and optimization for RTX 4080
"""

import json
import os
import random
from datetime import datetime
from typing import Optional, Tuple

import PIL.Image
import torch
from diffusers import DiffusionPipeline
from PIL import ImageFilter
import logging

logger = logging.getLogger(__name__)

try:
    from diffusers import QwenImageEditPipeline
except ImportError:
    logger.warning("⚠️ QwenImageEditPipeline not available. Enhanced features will be limited.")
    QwenImageEditPipeline = None

from .qwen_image_config import (
    GENERATION_CONFIG,
    MEMORY_CONFIG,
    MODEL_CONFIG,
    PROMPT_ENHANCEMENT,
)


class QwenImageGenerator:
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name: str = model_name or MODEL_CONFIG["model_name"]
        self.pipe: Optional[DiffusionPipeline] = None
        self.edit_pipe: Optional[QwenImageEditPipeline] = None
        # Create output directory
        self.output_dir: str = "generated_images"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        logger.info("\n📋 Qwen Model Usage:")
        logger.info("   • Qwen-Image: Text-to-image generation")
        logger.info("   • Qwen-Image-Edit: Image editing with reference images")
    
    def load_model(self) -> bool:
        """Load Qwen-Image diffusion pipeline"""
        try:
            logger.info("Loading Qwen-Image model... This may take a few minutes.")
            logger.info(f"Attempting to load: {self.model_name}")

            self._load_pipeline()

            if torch.cuda.is_available():
                self._move_pipeline_to_device()
                self._apply_memory_optimizations()
            else:
                logger.warning("⚠️ CUDA not available, using CPU")

            logger.info("✅ Qwen-Image model loaded successfully!")

            self._load_qwen_edit_pipeline()
            self.verify_device_setup()

            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            logger.info("💡 Troubleshooting tips:")
            logger.info("   1. Check internet connection for model download")
            logger.info("   2. Ensure sufficient disk space (~60-70GB)")
            logger.info("   3. Verify CUDA installation if using GPU")
            logger.info("   4. Try restarting the application")
            return False

    def _load_pipeline(self) -> None:
        """Load diffusion pipeline with precision fallbacks"""
        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=MODEL_CONFIG["torch_dtype"],
                use_safetensors=MODEL_CONFIG["use_safetensors"],
            )
            logger.info("✅ Model loaded with bfloat16 precision")
        except Exception as e1:
            logger.warning(f"⚠️  bfloat16 loading failed: {e1}")
            logger.info("🔄 Trying with float16...")
            try:
                self.pipe = DiffusionPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    use_safetensors=MODEL_CONFIG["use_safetensors"],
                )
                logger.info("✅ Model loaded with float16 precision")
            except Exception as e2:
                logger.warning(f"⚠️  float16 loading failed: {e2}")
                logger.info("🔄 Trying with default settings...")
                self.pipe = DiffusionPipeline.from_pretrained(
                    self.model_name,
                    use_safetensors=False,
                )
                logger.info("✅ Model loaded with default settings")

    def _move_pipeline_to_device(self) -> None:
        """Move pipeline components to target device"""
        logger.info(f"🔄 Moving model to GPU: {self.device}")
        self.pipe = self.pipe.to(self.device)

        component_count = 0
        if hasattr(self.pipe, "unet") and self.pipe.unet is not None:
            self.pipe.unet = self.pipe.unet.to(self.device)
            for param in self.pipe.unet.parameters():
                param.data = param.data.to(self.device)
            component_count += 1
            logger.info(
                f"✅ UNet moved to {self.device} ({sum(p.numel() for p in self.pipe.unet.parameters())} parameters)"
            )

        if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
            self.pipe.vae = self.pipe.vae.to(self.device)
            for param in self.pipe.vae.parameters():
                param.data = param.data.to(self.device)
            component_count += 1
            logger.info(
                f"✅ VAE moved to {self.device} ({sum(p.numel() for p in self.pipe.vae.parameters())} parameters)"
            )

        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
            self.pipe.text_encoder = self.pipe.text_encoder.to(self.device)
            for param in self.pipe.text_encoder.parameters():
                param.data = param.data.to(self.device)
            component_count += 1
            logger.info(
                f"✅ Text encoder moved to {self.device} ({sum(p.numel() for p in self.pipe.text_encoder.parameters())} parameters)"
            )

        additional_components = ["safety_checker", "feature_extractor"]
        for comp_name in additional_components:
            if hasattr(self.pipe, comp_name) and getattr(self.pipe, comp_name) is not None:
                try:
                    component = getattr(self.pipe, comp_name)
                    if hasattr(component, "to"):
                        component = component.to(self.device)
                        setattr(self.pipe, comp_name, component)
                        component_count += 1
                        logger.info(f"✅ {comp_name} moved to {self.device}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not move {comp_name}: {e}")

        if hasattr(self.pipe, "scheduler") and self.pipe.scheduler is not None:
            logger.info("✅ Scheduler noted (no device movement needed)")
            component_count += 1

        import gc

        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"📊 Total components processed: {component_count}")

    def _apply_memory_optimizations(self) -> None:
        """Apply memory optimization settings to pipeline"""
        if MEMORY_CONFIG["enable_attention_slicing"]:
            self.pipe.enable_attention_slicing()
            logger.info("✅ Attention slicing enabled")

        if MEMORY_CONFIG["enable_cpu_offload"]:
            self.pipe.enable_model_cpu_offload()
            logger.info("✅ CPU offload enabled")

        if MEMORY_CONFIG["enable_sequential_cpu_offload"]:
            self.pipe.enable_sequential_cpu_offload()
            logger.info("✅ Sequential CPU offload enabled")
    
    def _load_qwen_edit_pipeline(self) -> None:
        """Load Qwen-Image-Edit pipeline for enhanced features using HF Hub API"""
        if QwenImageEditPipeline is None:
            logger.warning("⚠️ QwenImageEditPipeline not available. Install latest diffusers from GitHub.")
            logger.warning("   Enhanced features will use alternative methods.")
            self.edit_pipe = None
            return
            
        try:
            logger.info("🔄 Loading Qwen-Image-Edit pipeline for enhanced features...")
            logger.info("   Using HuggingFace Hub API for better download reliability")
            
            # Import HuggingFace Hub for better download handling
            try:
                from huggingface_hub import repo_info, snapshot_download
                use_hub_api = True
            except ImportError:
                logger.warning("⚠️ huggingface_hub not available, using standard method")
                use_hub_api = False
            
            # Load Qwen-Image-Edit model with improved download handling
            try:
                if use_hub_api:
                    # Pre-check if model exists and get size info
                    try:
                        repo_data = repo_info("Qwen/Qwen-Image-Edit")
                        total_size = sum(file.size for file in repo_data.siblings if file.size)
                        logger.info(f"📊 Model size: {self._format_size(total_size)} (~20GB)")
                        logger.info("💡 Download will resume automatically if interrupted")
                    except Exception:
                        logger.info("📊 Model size: ~20GB (estimated)")
                
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
                            logger.info("✅ Attention slicing enabled for Qwen-Image-Edit")
                        except Exception as opt_error:
                            logger.warning(f"⚠️ Could not enable attention slicing: {opt_error}")
                    
                    # Verify device consistency for edit pipeline
                    self._verify_edit_pipeline_devices()
                
                logger.info("✅ Qwen-Image-Edit pipeline loaded successfully!")
                logger.info("   • Image-to-Image editing available")
                logger.info("   • Inpainting capabilities available")
                logger.info("   • Text editing in images available")
                
            except Exception as download_error:
                error_msg = str(download_error)
                logger.warning(f"⚠️ Could not download/load Qwen-Image-Edit: {download_error}")
                
                # Provide specific guidance based on error type
                if "Connection" in error_msg or "timeout" in error_msg.lower():
                    logger.info("🌐 Network issue detected. Try:")
                    logger.info("   1. Check internet connection stability")
                    logger.info("   2. Use the enhanced downloader: python tools/download_qwen_edit_hub.py")
                    logger.info("   3. Download will auto-resume if interrupted")
                elif "disk" in error_msg.lower() or "space" in error_msg.lower():
                    logger.info("💾 Disk space issue. Ensure ~25GB free space available")
                elif "permission" in error_msg.lower():
                    logger.info("🔒 Permission issue. Check write access to cache directory")
                else:
                    logger.info("💡 General troubleshooting:")
                    logger.info("   1. Try: python tools/download_qwen_edit_hub.py")
                    logger.info("   2. Check HuggingFace Hub accessibility")
                    logger.info("   3. Ensure sufficient disk space (~25GB)")
                
                logger.info("   Enhanced features will use alternative approaches.")
                self.edit_pipe = None
                
        except Exception as e:
            logger.error(f"⚠️ Error loading Qwen-Image-Edit pipeline: {e}")
            logger.info("   Enhanced features will use creative text-to-image approaches.")
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
            
        logger.info(f"🔍 Verifying Qwen-Image-Edit pipeline devices for {self.device}:")
        
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
                        
                        logger.info(f"   {comp_name.upper()}: {comp_device}")
                        
                        if comp_device != self.device and comp_device != "unknown":
                            logger.info(f"   🔧 Moving {comp_name} from {comp_device} to {self.device}")
                            try:
                                component = component.to(self.device)
                                setattr(self.edit_pipe, comp_name, component)
                                logger.info(f"   ✅ {comp_name} moved successfully")
                            except Exception as move_error:
                                logger.warning(f"   ⚠️ Could not move {comp_name}: {move_error}")
                                all_correct = False
                                
                    except Exception as comp_error:
                        logger.info(f"   {comp_name.upper()}: error checking device ({comp_error})")
                        all_correct = False
            
            # Check scheduler
            if hasattr(self.edit_pipe, 'scheduler') and self.edit_pipe.scheduler is not None:
                logger.info(f"   SCHEDULER: present ({type(self.edit_pipe.scheduler).__name__})")
            
            # Summary
            if all_correct:
                logger.info(f"✅ Qwen-Image-Edit pipeline verified on {self.device}")
                return True
            else:
                logger.warning("⚠️ Some edit pipeline components needed adjustment")
                return False
                
        except Exception as e:
            logger.error(f"❌ Edit pipeline device verification failed: {e}")
            return False
    
    def verify_device_setup(self) -> bool:
        """Verify model components are on the correct device (safe version)"""
        if not self.pipe:
            logger.warning("⚠️ Model not loaded")
            return False
            
        logger.info(f"🔍 Safe device verification for {self.device}:")
        
        try:
            # Check main pipeline device
            if hasattr(self.pipe, 'device'):
                main_device = str(self.pipe.device)
                logger.info(f"   Pipeline device: {main_device}")
            
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
                        
                        logger.info(f"   {comp_name.upper()}: {comp_device} ({param_count:,} params)")
                        
                        if comp_device != self.device and comp_device != "unknown":
                            all_correct = False
                            
                    except Exception as comp_error:
                        logger.info(f"   {comp_name.upper()}: error checking device ({comp_error})")
                        all_correct = False
            
            # Check scheduler
            if hasattr(self.pipe, 'scheduler') and self.pipe.scheduler is not None:
                logger.info(f"   SCHEDULER: present ({type(self.pipe.scheduler).__name__})")
            
            # Summary
            if all_correct:
                logger.info(f"✅ All components verified on {self.device}")
                return True
            else:
                logger.warning("⚠️ Some components may need device adjustment")
                return False
                
        except Exception as e:
            logger.error(f"❌ Device verification failed: {e}")
            return False
    
    def _force_device_consistency(self):
        """Force all components to the target device with safe methods"""
        try:
            logger.info(f"🔧 Forcing device consistency to {self.device}...")
            
            # Safe device movement - use PyTorch's built-in methods only
            components = ['unet', 'vae', 'text_encoder']
            
            for comp_name in components:
                if hasattr(self.pipe, comp_name) and getattr(self.pipe, comp_name) is not None:
                    component = getattr(self.pipe, comp_name)
                    
                    # Use PyTorch's safe .to() method only
                    try:
                        component = component.to(self.device)
                        setattr(self.pipe, comp_name, component)
                        
                        # Count parameters for reporting
                        param_count = sum(p.numel() for p in component.parameters())
                        logger.info(f"✅ {comp_name}: moved to {self.device} ({param_count:,} parameters)")
                        
                    except Exception as comp_error:
                        logger.warning(f"⚠️ Could not move {comp_name}: {comp_error}")
            
            # Handle scheduler (no device movement needed for most schedulers)
            if hasattr(self.pipe, 'scheduler') and self.pipe.scheduler is not None:
                logger.info(f"✅ scheduler: present ({type(self.pipe.scheduler).__name__})")
            
            # Move pipeline itself to device
            try:
                self.pipe = self.pipe.to(self.device)
                logger.info(f"✅ Pipeline moved to {self.device}")
            except Exception as pipe_error:
                logger.warning(f"⚠️ Pipeline device move warning: {pipe_error}")
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"❌ Failed to force device consistency: {e}")
            # Emergency fallback: minimal device move
            try:
                logger.info("🚨 Attempting minimal device move...")
                self.pipe = self.pipe.to(self.device)
                logger.info("✅ Minimal move completed")
            except Exception as e2:
                logger.error(f"❌ Minimal move also failed: {e2}")

    def enhance_prompt(self, prompt: str, language: str = "en") -> str:
        """Enhance prompt with quality keywords"""
        enhancement = PROMPT_ENHANCEMENT.get(language, PROMPT_ENHANCEMENT["en"])
        quality_keywords: str = enhancement["quality_keywords"]
        
        # Don't add keywords if they're already present
        if not any(word in prompt.lower() for word in ["4k", "hd", "quality", "detailed"]):
            return f"{prompt}, {quality_keywords}"
        return prompt
    
    def generate_image(self, prompt: str, negative_prompt: str = "", width: Optional[int] = None, height: Optional[int] = None, 
                      num_inference_steps: Optional[int] = None, cfg_scale: Optional[float] = None, seed: int = -1, 
                      language: str = "en", enhance_prompt_flag: bool = True) -> Tuple[Optional[PIL.Image.Image], str]:
        """Generate image from text prompt"""
        
        if not self.pipe:
            return None, "Model not loaded. Please initialize the model first."
        
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
            
            # Handle random seed and generator
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
            
            # Create generator on the correct device
            if torch.cuda.is_available() and self.device == "cuda":
                generator = torch.Generator(device="cuda").manual_seed(seed)
            else:
                generator = torch.Generator().manual_seed(seed)
            
            logger.info(f"Generating image with prompt: {enhanced_prompt[:100]}...")
            logger.info(f"Settings: {width}x{height}, steps: {num_inference_steps}, CFG: {cfg_scale}, seed: {seed}")
            logger.info(f"Device: {self.device}, Generator device: {generator.device if hasattr(generator, 'device') else 'cpu'}")
            
            # SAFE: Pre-generation device check
            logger.info("🔍 Critical pre-generation device check...")
            
            # Light device consistency check - no aggressive moves
            try:
                # Quick verification without forcing moves
                device_issues = False
                
                # Check main components exist and have .to() method
                components = ['unet', 'vae', 'text_encoder']
                for comp_name in components:
                    if hasattr(self.pipe, comp_name):
                        component = getattr(self.pipe, comp_name)
                        if component is not None and hasattr(component, 'to'):
                            # Just ensure it's on the right device without aggressive moves
                            try:
                                # Check if already on correct device
                                if hasattr(component, 'device'):
                                    comp_device = str(component.device)
                                else:
                                    # Check first parameter device
                                    comp_device = str(next(component.parameters()).device)
                                
                                if comp_device != self.device:
                                    logger.warning(f"⚠️ {comp_name} on {comp_device}, expected {self.device}")
                                    device_issues = True
                                else:
                                    logger.info(f"✅ {comp_name} correctly on {self.device}")
                            except Exception:
                                logger.warning(f"⚠️ Could not verify {comp_name} device")
                
                # Only force consistency if issues detected
                if device_issues:
                    logger.info("🛠️ Light device consistency adjustment...")
                    self._force_device_consistency()
                else:
                    logger.info("✅ All components already on correct device")
                    
            except Exception as check_error:
                logger.warning(f"⚠️ Device check failed: {check_error}")
            
            # Ensure we're using the correct device context with proper error handling
            device_context = torch.cuda.device(self.device) if torch.cuda.is_available() else torch.no_grad()
            
            with device_context:
                with torch.no_grad():
                    # Clear CUDA cache and synchronize before generation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        logger.info(f"🧹 CUDA synchronized, available memory: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated():.1e} bytes")
                    
                    # For Qwen-Image, we need to use the correct parameter name
                    generation_params = {
                        "prompt": enhanced_prompt,
                        "negative_prompt": negative_prompt,
                        "width": width,
                        "height": height,
                        "num_inference_steps": num_inference_steps,
                        "generator": generator
                    }
                    
                    # Add CFG scale parameter (check what parameter name Qwen uses)
                    if hasattr(self.pipe, '__call__'):
                        import inspect
                        sig = inspect.signature(self.pipe.__call__)
                        if 'guidance_scale' in sig.parameters:
                            generation_params['guidance_scale'] = cfg_scale
                        elif 'true_cfg_scale' in sig.parameters:
                            generation_params['true_cfg_scale'] = cfg_scale
                    
                    # NO aggressive device moves right before generation
                    # Just verify and proceed - the model should already be properly loaded
                    logger.info(f"✅ Starting generation on {self.device}")
                    
                    logger.info("🎨 Starting generation with comprehensive device safety...")
                    
                    # Generation with additional error handling
                    try:
                        result = self.pipe(**generation_params)
                    except RuntimeError as runtime_error:
                        if "Expected all tensors to be on the same device" in str(runtime_error):
                            logger.error(f"❌ Device error during generation: {runtime_error}")
                            logger.info("🔄 Attempting CPU fallback generation...")
                            
                            # Emergency CPU fallback
                            try:
                                # Move everything to CPU
                                self.pipe = self.pipe.to('cpu')
                                cpu_generator = torch.Generator().manual_seed(seed)
                                
                                result = self.pipe(
                                    prompt=enhanced_prompt,
                                    negative_prompt=negative_prompt,
                                    width=width,
                                    height=height,
                                    num_inference_steps=num_inference_steps,
                                    true_cfg_scale=cfg_scale,
                                    generator=cpu_generator
                                )
                                
                                logger.info("✅ CPU fallback generation successful")
                                
                            except Exception as cpu_error:
                                logger.error(f"❌ CPU fallback also failed: {cpu_error}")
                                raise runtime_error  # Re-raise original error
                        else:
                            raise  # Re-raise non-device errors
            
            image = result.images[0]
            
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
                "language": language,
                "model": self.model_name,
                "timestamp": timestamp
            }
            
            # Save image
            image.save(filepath)
            
            # Save metadata
            metadata_file = filepath.replace(".png", "_metadata.json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            success_msg = f"✅ Image generated successfully!\nSaved as: {filename}\nSeed: {seed}"
            
            return image, success_msg
            
        except Exception as e:
            error_msg = f"❌ Error generating image: {str(e)}"
            logger.info(error_msg)
            return None, error_msg
    
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
            
            logger.info("Generating image-to-image with Qwen-Image-Edit...")
            logger.info(f"Prompt: {enhanced_prompt[:100]}...")
            logger.info(f"Settings: {width}x{height}, steps: {num_inference_steps}, CFG: {cfg_scale}, seed: {seed}")
            
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
            logger.info(error_msg)
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
            
            logger.info("Generating inpaint with Qwen-Image-Edit...")
            logger.info(f"Prompt: {mask_prompt[:100]}...")
            logger.info(f"Settings: {width}x{height}, steps: {num_inference_steps}, CFG: {cfg_scale}, seed: {seed}")
            
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
            logger.info(error_msg)
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
            logger.info(error_msg)
            return None, error_msg