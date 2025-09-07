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
        # Create output directory
        self.output_dir: str = "generated_images"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    def load_model(self) -> bool:
        """Load Qwen-Image diffusion pipeline"""
        try:
            print("Loading Qwen-Image model... This may take a few minutes.")
            print(f"Attempting to load: {self.model_name}")
            
            # First try: Load with optimal settings for RTX 4080
            try:
                self.pipe = DiffusionPipeline.from_pretrained(
                    self.model_name, 
                    torch_dtype=MODEL_CONFIG["torch_dtype"],
                    use_safetensors=MODEL_CONFIG["use_safetensors"]
                )
                print("✅ Model loaded with bfloat16 precision")
            except Exception as e1:
                print(f"⚠️  bfloat16 loading failed: {e1}")
                print("🔄 Trying with float16...")
                
                # Fallback: Try with float16
                try:
                    self.pipe = DiffusionPipeline.from_pretrained(
                        self.model_name, 
                        torch_dtype=torch.float16,
                        use_safetensors=MODEL_CONFIG["use_safetensors"]
                    )
                    print("✅ Model loaded with float16 precision")
                except Exception as e2:
                    print(f"⚠️  float16 loading failed: {e2}")
                    print("🔄 Trying with default settings...")
                    
                    # Final fallback: Default settings
                    self.pipe = DiffusionPipeline.from_pretrained(
                        self.model_name,
                        use_safetensors=False  # Try without safetensors if needed
                    )
                    print("✅ Model loaded with default settings")
            
            # Move to GPU and apply optimizations
            if torch.cuda.is_available():
                print(f"🔄 Moving model to GPU: {self.device}")
                
                # First: Move the entire pipeline
                self.pipe = self.pipe.to(self.device)
                
                # Second: Explicitly move all subcomponents
                component_count = 0
                if hasattr(self.pipe, 'unet') and self.pipe.unet is not None:
                    self.pipe.unet = self.pipe.unet.to(self.device)
                    # Ensure all parameters are on device
                    for param in self.pipe.unet.parameters():
                        param.data = param.data.to(self.device)
                    component_count += 1
                    print(f"✅ UNet moved to {self.device} ({sum(p.numel() for p in self.pipe.unet.parameters())} parameters)")
                
                if hasattr(self.pipe, 'vae') and self.pipe.vae is not None:
                    self.pipe.vae = self.pipe.vae.to(self.device)
                    # Ensure all parameters are on device
                    for param in self.pipe.vae.parameters():
                        param.data = param.data.to(self.device)
                    component_count += 1
                    print(f"✅ VAE moved to {self.device} ({sum(p.numel() for p in self.pipe.vae.parameters())} parameters)")
                
                if hasattr(self.pipe, 'text_encoder') and self.pipe.text_encoder is not None:
                    self.pipe.text_encoder = self.pipe.text_encoder.to(self.device)
                    # Ensure all parameters are on device
                    for param in self.pipe.text_encoder.parameters():
                        param.data = param.data.to(self.device)
                    component_count += 1
                    print(f"✅ Text encoder moved to {self.device} ({sum(p.numel() for p in self.pipe.text_encoder.parameters())} parameters)")
                
                # Check for additional components (excluding scheduler)
                additional_components = ['safety_checker', 'feature_extractor']
                for comp_name in additional_components:
                    if hasattr(self.pipe, comp_name) and getattr(self.pipe, comp_name) is not None:
                        try:
                            component = getattr(self.pipe, comp_name)
                            if hasattr(component, 'to'):
                                component = component.to(self.device)
                                setattr(self.pipe, comp_name, component)
                                component_count += 1
                                print(f"✅ {comp_name} moved to {self.device}")
                        except Exception as e:
                            print(f"⚠️ Could not move {comp_name}: {e}")
                
                # Handle scheduler separately (it typically doesn't have parameters)
                if hasattr(self.pipe, 'scheduler') and self.pipe.scheduler is not None:
                    # Schedulers don't typically need device movement
                    print("✅ Scheduler noted (no device movement needed)")
                    component_count += 1
                
                # Force garbage collection and ensure all tensors are moved
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                print(f"📊 Total components processed: {component_count}")
                
                # Apply memory optimizations
                if MEMORY_CONFIG["enable_attention_slicing"]:
                    self.pipe.enable_attention_slicing()
                    print("✅ Attention slicing enabled")
                
                if MEMORY_CONFIG["enable_cpu_offload"]:
                    self.pipe.enable_model_cpu_offload()
                    print("✅ CPU offload enabled")
                
                if MEMORY_CONFIG["enable_sequential_cpu_offload"]:
                    self.pipe.enable_sequential_cpu_offload()
                    print("✅ Sequential CPU offload enabled")
            else:
                print("⚠️ CUDA not available, using CPU")
                
            print("✅ Qwen-Image model loaded successfully!")
            
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
        """Force all components to the target device with safe methods"""
        try:
            print(f"🔧 Forcing device consistency to {self.device}...")
            
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
                        print(f"✅ {comp_name}: moved to {self.device} ({param_count:,} parameters)")
                        
                    except Exception as comp_error:
                        print(f"⚠️ Could not move {comp_name}: {comp_error}")
            
            # Handle scheduler (no device movement needed for most schedulers)
            if hasattr(self.pipe, 'scheduler') and self.pipe.scheduler is not None:
                print(f"✅ scheduler: present ({type(self.pipe.scheduler).__name__})")
            
            # Move pipeline itself to device
            try:
                self.pipe = self.pipe.to(self.device)
                print(f"✅ Pipeline moved to {self.device}")
            except Exception as pipe_error:
                print(f"⚠️ Pipeline device move warning: {pipe_error}")
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Failed to force device consistency: {e}")
            # Emergency fallback: minimal device move
            try:
                print("🚨 Attempting minimal device move...")
                self.pipe = self.pipe.to(self.device)
                print("✅ Minimal move completed")
            except Exception as e2:
                print(f"❌ Minimal move also failed: {e2}")

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
            
            print(f"Generating image with prompt: {enhanced_prompt[:100]}...")
            print(f"Settings: {width}x{height}, steps: {num_inference_steps}, CFG: {cfg_scale}, seed: {seed}")
            print(f"Device: {self.device}, Generator device: {generator.device if hasattr(generator, 'device') else 'cpu'}")
            
            # SAFE: Pre-generation device check
            print("🔍 Critical pre-generation device check...")
            
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
                                    print(f"⚠️ {comp_name} on {comp_device}, expected {self.device}")
                                    device_issues = True
                                else:
                                    print(f"✅ {comp_name} correctly on {self.device}")
                            except Exception:
                                print(f"⚠️ Could not verify {comp_name} device")
                
                # Only force consistency if issues detected
                if device_issues:
                    print("🛠️ Light device consistency adjustment...")
                    self._force_device_consistency()
                else:
                    print("✅ All components already on correct device")
                    
            except Exception as check_error:
                print(f"⚠️ Device check failed: {check_error}")
            
            # Ensure we're using the correct device context with proper error handling
            device_context = torch.cuda.device(self.device) if torch.cuda.is_available() else torch.no_grad()
            
            with device_context:
                with torch.no_grad():
                    # Clear CUDA cache and synchronize before generation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        print(f"🧹 CUDA synchronized, available memory: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated():.1e} bytes")
                    
                    # NO aggressive device moves right before generation
                    # Just verify and proceed - the model should already be properly loaded
                    print(f"✅ Starting generation on {self.device}")
                    
                    print("🎨 Starting generation with comprehensive device safety...")
                    
                    # Generation with additional error handling
                    try:
                        result = self.pipe(
                            prompt=enhanced_prompt,
                            negative_prompt=negative_prompt,
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps,
                            true_cfg_scale=cfg_scale,
                            generator=generator
                        )
                    except RuntimeError as runtime_error:
                        if "Expected all tensors to be on the same device" in str(runtime_error):
                            print(f"❌ Device error during generation: {runtime_error}")
                            print("🔄 Attempting CPU fallback generation...")
                            
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
                                
                                print("✅ CPU fallback generation successful")
                                
                            except Exception as cpu_error:
                                print(f"❌ CPU fallback also failed: {cpu_error}")
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
            print(error_msg)
            return None, error_msg