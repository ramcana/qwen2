"""
Robust Qwen-Image Generator with timeout handling and better error recovery
"""

import json
import os
import random
import signal
import threading
import time
from datetime import datetime
from typing import Optional, Tuple

import PIL.Image
import torch
from diffusers import DiffusionPipeline
from PIL import ImageFilter

from qwen_image_config import (
    GENERATION_CONFIG,
    MEMORY_CONFIG,
    MODEL_CONFIG,
    PROMPT_ENHANCEMENT,
)


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


class QwenImageGeneratorRobust:
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
        
        print("\n📋 Qwen Model Usage:")
        print("   • Qwen-Image: Text-to-image generation with robust loading")
    
    def load_model_with_timeout(self, timeout_seconds: int = 600) -> bool:
        """Load model with timeout protection"""
        
        def load_model_thread():
            try:
                return self._load_model_internal()
            except Exception as e:
                print(f"❌ Model loading thread failed: {e}")
                return False
        
        print(f"🔄 Loading model with {timeout_seconds}s timeout...")
        
        # Use threading for timeout control
        result = [False]  # Mutable container for thread result
        
        def target():
            result[0] = self._load_model_internal()
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            print(f"❌ Model loading timed out after {timeout_seconds}s")
            print("💡 Try:")
            print("   1. Check internet connection")
            print("   2. Clear HuggingFace cache: rm -rf ~/.cache/huggingface")
            print("   3. Restart with more time")
            return False
        
        return result[0]
    
    def _load_model_internal(self) -> bool:
        """Internal model loading with progressive fallbacks"""
        try:
            print("🔄 Step 1: Attempting optimal Qwen-Image loading...")
            
            # Clear GPU memory first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Try optimal loading first
            try:
                print("   Loading with bfloat16 + trust_remote_code...")
                self.pipe = DiffusionPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    low_cpu_mem_usage=True,  # Enable for safer loading
                    trust_remote_code=True,
                    local_files_only=False,
                    resume_download=True,  # Resume interrupted downloads
                    force_download=False   # Don't re-download if cached
                )
                print("✅ Optimal loading successful!")
                
            except Exception as e1:
                print(f"⚠️ Optimal loading failed: {e1}")
                print("🔄 Step 2: Trying float16 fallback...")
                
                try:
                    self.pipe = DiffusionPipeline.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        variant="fp16",
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        resume_download=True
                    )
                    print("✅ Float16 fallback successful!")
                    
                except Exception as e2:
                    print(f"⚠️ Float16 failed: {e2}")
                    print("🔄 Step 3: Trying minimal configuration...")
                    
                    self.pipe = DiffusionPipeline.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        use_safetensors=False,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    print("✅ Minimal configuration successful!")
            
            # Move to GPU efficiently
            if torch.cuda.is_available():
                print("🔄 Step 4: Moving to GPU...")
                
                # Clear memory before moving
                torch.cuda.empty_cache()
                
                # Move pipeline
                self.pipe = self.pipe.to(self.device)
                
                # Verify components
                components = ['unet', 'vae', 'text_encoder', 'scheduler']
                for comp_name in components:
                    if hasattr(self.pipe, comp_name) and getattr(self.pipe, comp_name) is not None:
                        print(f"   ✅ {comp_name} ready")
                
                # Apply RTX 4080 optimizations
                print("🔄 Step 5: Applying RTX 4080 optimizations...")
                
                # Disable performance-killing features
                if hasattr(self.pipe, 'disable_attention_slicing'):
                    self.pipe.disable_attention_slicing()
                    print("   ✅ Attention slicing disabled for performance")
                
                # Enable flash attention if available
                try:
                    if hasattr(self.pipe, 'unet') and hasattr(self.pipe.unet, 'set_attn_processor'):
                        from diffusers.models.attention_processor import AttnProcessor2_0
                        self.pipe.unet.set_attn_processor(AttnProcessor2_0())
                        print("   ✅ Flash attention enabled")
                except Exception as flash_error:
                    print(f"   ⚠️ Flash attention not available: {flash_error}")
                
                # Final cleanup
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Report memory usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1e9
                    total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"   📊 GPU Memory: {allocated:.2f}GB / {total:.2f}GB ({allocated/total*100:.1f}%)")
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print("💡 Troubleshooting:")
            print("   1. Check internet connection")
            print("   2. Ensure sufficient disk space (~20GB)")
            print("   3. Clear cache: rm -rf ~/.cache/huggingface")
            print("   4. Check CUDA installation")
            return False
    
    def load_model(self) -> bool:
        """Public interface for model loading"""
        return self.load_model_with_timeout(timeout_seconds=600)  # 10 minute timeout
    
    def enhance_prompt(self, prompt: str, language: str = "en") -> str:
        """Enhance prompt with quality keywords"""
        enhancement = PROMPT_ENHANCEMENT.get(language, PROMPT_ENHANCEMENT["en"])
        quality_keywords: str = enhancement["quality_keywords"]
        
        # Don't add keywords if they're already present
        if not any(word in prompt.lower() for word in ["4k", "hd", "quality", "detailed"]):
            return f"{prompt}, {quality_keywords}"
        return prompt
    
    def generate_image(self, prompt: str, negative_prompt: str = "", width: Optional[int] = None, 
                      height: Optional[int] = None, num_inference_steps: Optional[int] = None, 
                      cfg_scale: Optional[float] = None, seed: int = -1, language: str = "en", 
                      enhance_prompt_flag: bool = True) -> Tuple[Optional[PIL.Image.Image], str]:
        """Generate image from text prompt with robust error handling"""
        
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
            
            # Handle random seed
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
            
            # Create generator
            if torch.cuda.is_available() and self.device == "cuda":
                generator = torch.Generator(device="cuda").manual_seed(seed)
            else:
                generator = torch.Generator().manual_seed(seed)
            
            print(f"🎨 Generating: {enhanced_prompt[:100]}...")
            print(f"⚙️ Settings: {width}x{height}, {num_inference_steps} steps, CFG: {cfg_scale}, seed: {seed}")
            
            # Clear GPU memory before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Generation parameters for Qwen-Image
            generation_params = {
                "prompt": enhanced_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
                "true_cfg_scale": cfg_scale,  # Qwen-Image specific parameter
                "output_type": "pil"
            }
            
            # Add negative prompt if provided
            if negative_prompt and negative_prompt.strip():
                generation_params["negative_prompt"] = negative_prompt
            
            # Generate with error handling
            start_time = time.time()
            
            try:
                result = self.pipe(**generation_params)
                generation_time = time.time() - start_time
                
            except RuntimeError as e:
                error_str = str(e)
                if "out of memory" in error_str.lower():
                    print("❌ GPU out of memory. Trying memory optimizations...")
                    
                    # Clear memory and try with optimizations
                    torch.cuda.empty_cache()
                    
                    # Enable memory saving features temporarily
                    if hasattr(self.pipe, 'enable_attention_slicing'):
                        self.pipe.enable_attention_slicing()
                    
                    # Retry
                    result = self.pipe(**generation_params)
                    generation_time = time.time() - start_time
                    
                    # Disable attention slicing again for future generations
                    if hasattr(self.pipe, 'disable_attention_slicing'):
                        self.pipe.disable_attention_slicing()
                else:
                    raise
            
            image = result.images[0]
            
            # Save image with metadata
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qwen_robust_{timestamp}_{seed}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save image
            image.save(filepath)
            
            # Save metadata
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
                "generation_time": generation_time,
                "timestamp": timestamp
            }
            
            metadata_file = filepath.replace(".png", "_metadata.json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            success_msg = f"✅ Image generated in {generation_time:.2f}s!\nSaved as: {filename}\nSeed: {seed}"
            
            return image, success_msg
            
        except Exception as e:
            error_msg = f"❌ Error generating image: {str(e)}"
            print(error_msg)
            return None, error_msg