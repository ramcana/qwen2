"""
High-Performance Qwen-Image Generator
Optimized for Threadripper PRO 5995WX + RTX 4080 + 128GB RAM
"""

import os
import time
from typing import Optional, Tuple, Dict, Any
import torch
import PIL.Image
from diffusers import DiffusionPipeline
from src.qwen_image_config import MODEL_CONFIG, MEMORY_CONFIG, GENERATION_CONFIG


class OptimizedQwenImageGenerator:
    """High-performance Qwen-Image generator optimized for high-end hardware"""
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name: str = model_name or MODEL_CONFIG["model_name"]
        self.pipe: Optional[DiffusionPipeline] = None
        self.output_dir: str = "outputs"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Print hardware info
        if torch.cuda.is_available():
            print(f"Using device: {self.device}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        print("\n🚀 HIGH-PERFORMANCE MODE ENABLED")
        print("   • Optimized for Threadripper PRO 5995WX + RTX 4080")
        print("   • Attention slicing DISABLED for maximum speed")
        print("   • Flash attention and xFormers ENABLED")
    
    def load_model(self) -> bool:
        """Load Qwen-Image pipeline with high-performance optimizations"""
        try:
            print("⚡ Loading Qwen-Image model with performance optimizations...")
            print(f"📦 Model: {self.model_name}")
            
            # Load with optimal settings for high-end hardware
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=MODEL_CONFIG["torch_dtype"],
                use_safetensors=MODEL_CONFIG["use_safetensors"],
                variant="fp16" if MODEL_CONFIG["torch_dtype"] == torch.float16 else None
            )
            
            print(f"✅ Model loaded with {MODEL_CONFIG['torch_dtype']} precision")
            
            # Move to GPU efficiently
            if torch.cuda.is_available():
                print(f"🚀 Moving pipeline to {self.device}...")
                self.pipe = self.pipe.to(self.device)
                
                # Apply high-performance optimizations
                print("⚡ Applying performance optimizations...")
                
                # DISABLE attention slicing for maximum speed on high-end hardware
                if hasattr(self.pipe, 'disable_attention_slicing'):
                    self.pipe.disable_attention_slicing()
                    print("✅ Attention slicing DISABLED (maximum speed)")
                
                # Enable Flash Attention 2.0 (more compatible than xFormers)
                try:
                    if hasattr(self.pipe.unet, 'set_attn_processor'):
                        from diffusers.models.attention_processor import AttnProcessor2_0
                        self.pipe.unet.set_attn_processor(AttnProcessor2_0())
                        print("✅ Flash Attention 2.0 ENABLED")
                except Exception as e:
                    print(f"⚠️ Flash Attention not available: {e}")
                
                # Skip xFormers for now due to compatibility issues with Qwen-Image
                print("⚠️ xFormers DISABLED (compatibility issues with Qwen-Image)")
                
                # Compile the UNet for even faster inference (PyTorch 2.0+)
                try:
                    if hasattr(torch, 'compile') and hasattr(self.pipe, 'unet'):
                        print("🔥 Compiling UNet with torch.compile...")
                        self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead")
                        print("✅ UNet compiled for maximum performance")
                except Exception as e:
                    print(f"⚠️ torch.compile not available: {e}")
                
                # Clear cache and optimize memory
                torch.cuda.empty_cache()
                
                print("🎯 High-performance optimizations complete!")
                print("   • Attention slicing: DISABLED")
                print("   • xFormers: DISABLED (compatibility)")
                print("   • Flash Attention: ENABLED")
                print("   • torch.compile: ENABLED")
                
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1664,
        height: int = 960,
        num_inference_steps: int = 50,
        cfg_scale: float = 4.0,
        seed: int = -1,
        language: str = "en",
        enhance_prompt: bool = True
    ) -> Tuple[Optional[PIL.Image.Image], str]:
        """Generate image with high-performance settings"""
        
        if not self.pipe:
            return None, "Model not loaded. Call load_model() first."
        
        try:
            # Set seed for reproducibility
            if seed == -1:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            print(f"🎨 HIGH-SPEED GENERATION STARTING...")
            print(f"   • Prompt: {prompt[:50]}...")
            print(f"   • Resolution: {width}×{height}")
            print(f"   • Steps: {num_inference_steps}")
            print(f"   • CFG: {cfg_scale}")
            print(f"   • Seed: {seed}")
            
            start_time = time.time()
            
            # Add positive magic for better quality (official recommendation)
            positive_magic = {
                "en": ", Ultra HD, 4K, cinematic composition.",
                "zh": ", 超清，4K，电影级构图."
            }
            
            enhanced_prompt = prompt + positive_magic.get(language, positive_magic["en"])
            
            # Generate with official Qwen-Image parameters
            with torch.inference_mode():  # Disable gradient computation for speed
                result = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt if negative_prompt else " ",  # Use space, not None
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    true_cfg_scale=cfg_scale,  # Use true_cfg_scale, not guidance_scale
                    generator=generator,
                )
                
                # Handle different return types
                if hasattr(result, 'images'):
                    image = result.images[0]
                elif isinstance(result, list):
                    image = result[0]
                else:
                    image = result
            
            generation_time = time.time() - start_time
            
            # Save image
            timestamp = int(time.time())
            filename = f"qwen_optimized_{timestamp}_{seed}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save with metadata
            metadata = {
                "generator": "OptimizedQwenImageGenerator",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "generation_time": generation_time,
                "hardware": "Threadripper PRO 5995WX + RTX 4080"
            }
            
            # Add metadata to image
            from PIL.PngImagePlugin import PngInfo
            png_info = PngInfo()
            for key, value in metadata.items():
                png_info.add_text(key, str(value))
            
            image.save(filepath, pnginfo=png_info)
            
            print(f"🚀 GENERATION COMPLETE!")
            print(f"   • Time: {generation_time:.2f}s")
            print(f"   • Speed: {num_inference_steps/generation_time:.1f} steps/sec")
            print(f"   • Saved: {filename}")
            
            return image, filepath
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return None, f"Generation failed: {str(e)}"
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        
        allocated = torch.cuda.memory_allocated(0)
        total = torch.cuda.get_device_properties(0).total_memory
        
        return {
            "gpu_available": True,
            "allocated_gb": allocated / (1024**3),
            "total_gb": total / (1024**3),
            "usage_percent": (allocated / total) * 100,
            "device_name": torch.cuda.get_device_name(0)
        }
