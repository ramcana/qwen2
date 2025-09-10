"""
High-Performance Qwen-Image Generator
Optimized for AMD Threadripper PRO 5995WX + RTX 4080 + 128GB RAM
Target: 15-60 second generation time (not 500+ seconds!)
"""

import gc
import json
import os
import random
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Tuple

import PIL.Image
import torch
from diffusers import DiffusionPipeline

try:
    from diffusers import QwenImageEditPipeline
except ImportError:
    print("⚠️ QwenImageEditPipeline not available. Enhanced features will be limited.")
    QwenImageEditPipeline = None

from .qwen_highend_config import (
    HIGH_END_GENERATION_CONFIG,
    HIGH_END_MODEL_CONFIG,
    apply_high_end_optimizations,
    get_high_end_config,
    quick_performance_test,
    setup_high_end_environment,
    verify_high_end_setup,
)
from .qwen_image_config import PROMPT_ENHANCEMENT

# Import our device policy helper
from .utils.devices import (
    check_vram_availability,
    clamp_image_size,
    clear_gpu_memory,
    create_processor_eager,
    get_cache_dir,
    get_device_config,
    load_model_lazy,
    load_model_with_retry,
    perform_preflight_checks,
    safe_model_switch_context,
)


class HighEndQwenImageGenerator:
    """High-performance version optimized for powerful hardware"""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.device: str = "cuda"
        self.model_name: str = model_name or HIGH_END_MODEL_CONFIG["model_name"]
        self.pipe: Optional[DiffusionPipeline] = None
        self.edit_pipe: Optional[QwenImageEditPipeline] = None
        self.processor = None  # Eagerly loaded processor
        # Create output directory
        self.output_dir: str = "generated_images"
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup high-end environment
        setup_high_end_environment()

        print("🚀 HIGH-PERFORMANCE QWEN GENERATOR INITIALIZED")
        print("=" * 60)
        print("Target Hardware:")
        print("  • AMD Threadripper PRO 5995WX (64 cores)")
        print("  • RTX 4080 (16GB VRAM)")
        print("  • 128GB System RAM")
        print("Target Performance: 15-60 seconds per image")
        print("=" * 60)

        if torch.cuda.is_available():
            vram_info = check_vram_availability()
            print(f"Detected GPU: {torch.cuda.get_device_name()}")
            print(f"Available VRAM: {vram_info['total_gb']:.1f}GB")

            if (
                "RTX 4080" not in torch.cuda.get_device_name()
                and vram_info["total_gb"] < 15
            ):
                print("⚠️ WARNING: Hardware mismatch detected!")
                print("   Performance may be suboptimal")
        else:
            print("❌ CUDA not available - this will be VERY slow!")

    @contextmanager
    def safe_model_switch(self):
        """Context manager for safe model switching with memory cleanup"""
        # Clear existing model from memory
        if self.pipe is not None:
            del self.pipe
            self.pipe = None

        if self.edit_pipe is not None:
            del self.edit_pipe
            self.edit_pipe = None

        # Clear GPU memory and run garbage collection
        clear_gpu_memory()

        # Short sleep to allow memory cleanup
        time.sleep(0.2)  # 200ms

        try:
            yield
        finally:
            # Ensure cleanup after operation
            clear_gpu_memory()

    def switch_to_model(self, model_name: str) -> bool:
        """Safely switch to a different model"""
        print(f"🔄 Switching to model: {model_name}")

        with self.safe_model_switch():
            self.model_name = model_name
            return self.load_model()

    def load_model(self) -> bool:
        """Load model with high-end optimizations and canonical device policy"""
        try:
            print("\n🚀 LOADING MODEL WITH HIGH-END OPTIMIZATIONS...")
            print("=" * 60)

            # Pre-flight checks
            self._perform_preflight_checks()

            # Eagerly create processor (fast)
            try:
                self.processor = create_processor_eager(self.model_name)
                print("✅ Processor loaded eagerly")
            except Exception as e:
                print(f"⚠️ Could not load processor: {e}")
                self.processor = None

            # Safe model switching context
            safe_model_switch_context()

            # Verify setup first
            if not verify_high_end_setup():
                print("⚠️ High-end setup verification failed")
                print("   Proceeding anyway...")

            # Get optimized config
            config = get_high_end_config()

            # Get canonical device configuration
            device_config = get_device_config()
            print(f"🔧 Device config: {device_config}")

            print(f"Loading: {self.model_name}")
            print("Configuration:")
            print(f"  • torch_dtype: {config['torch_dtype']}")
            print(f"  • device: {config['device']}")
            print(f"  • low_cpu_mem_usage: {config['low_cpu_mem_usage']}")
            print(f"  • device_map: {config['device_map']}")
            print(f"  • max_memory: {config['max_memory']}")

            # Clear cache before loading
            if torch.cuda.is_available():
                clear_gpu_memory()
                torch.cuda.synchronize()
                print("✅ GPU cache cleared")

            # Load with high-end settings using our device policy and retry mechanism
            print("🔄 Loading model...")
            start_time = datetime.now()

            try:
                # Load with optimal settings for your hardware using our device policy
                self.pipe = load_model_with_retry(
                    DiffusionPipeline,
                    self.model_name,
                    use_safetensors=config["use_safetensors"],
                    low_cpu_mem_usage=config["low_cpu_mem_usage"],
                    device_map=config["device_map"],
                    max_memory=config["max_memory"],
                    variant=config["variant"],
                    cache_dir=config["cache_dir"],
                    local_files_only=config["local_files_only"],
                )

                load_time = (datetime.now() - start_time).total_seconds()
                print(f"✅ Model loaded in {load_time:.1f} seconds")

            except Exception as e:
                print(f"❌ High-end loading failed: {e}")
                print("🔄 Trying fallback method...")

                # Fallback to standard loading with our device policy
                self.pipe = load_model_lazy(
                    DiffusionPipeline,
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                )

                load_time = (datetime.now() - start_time).total_seconds()
                print(f"✅ Model loaded with fallback in {load_time:.1f} seconds")

            # Move to GPU explicitly (ensuring no CPU offloading)
            print("🔄 Moving to GPU...")
            start_time = datetime.now()

            self.pipe = self.pipe.to("cuda")

            # Ensure all components are on GPU
            components = ["unet", "vae", "text_encoder"]
            for comp_name in components:
                if hasattr(self.pipe, comp_name):
                    component = getattr(self.pipe, comp_name)
                    if component is not None:
                        component = component.to("cuda")
                        setattr(self.pipe, comp_name, component)

                        # Count parameters
                        param_count = sum(p.numel() for p in component.parameters())
                        print(
                            f"✅ {comp_name.upper()}: {param_count:,} parameters on GPU"
                        )

            gpu_time = (datetime.now() - start_time).total_seconds()
            print(f"✅ GPU transfer completed in {gpu_time:.1f} seconds")

            # Apply high-end optimizations
            print("🔄 Applying high-end optimizations...")
            self.pipe = apply_high_end_optimizations(self.pipe)

            # Verify final setup
            print("🔍 Final verification...")
            self._verify_gpu_setup()

            # Run performance test
            print("🧪 Running performance test...")
            if quick_performance_test():
                print("🚀 Performance test PASSED!")
            else:
                print("⚠️ Performance test indicates potential issues")

            print("✅ HIGH-END MODEL LOADING COMPLETE!")
            print("=" * 60)

            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\n💡 Troubleshooting for high-end hardware:")
            print("   1. Check CUDA drivers are up to date")
            print("   2. Ensure PyTorch CUDA version matches CUDA drivers")
            print("   3. Verify 15+ GB VRAM available")
            print("   4. Check internet connection for model download")
            return False

    def _verify_gpu_setup(self) -> bool:
        """Verify everything is properly on GPU"""

        print("🔍 Verifying GPU setup...")

        if not self.pipe:
            print("❌ Pipeline not loaded")
            return False

        all_good = True

        # Check components
        components = ["unet", "vae", "text_encoder"]
        for comp_name in components:
            if hasattr(self.pipe, comp_name):
                component = getattr(self.pipe, comp_name)
                if component is not None:
                    try:
                        # Check device
                        if hasattr(component, "device"):
                            device = str(component.device)
                        else:
                            device = str(next(component.parameters()).device)

                        if "cuda" in device:
                            print(f"✅ {comp_name.upper()}: GPU ({device})")
                        else:
                            print(f"❌ {comp_name.upper()}: NOT on GPU ({device})")
                            all_good = False

                    except Exception as e:
                        print(f"⚠️ {comp_name.upper()}: Error checking device ({e})")
                        all_good = False

        # Check memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            usage_percent = (allocated / total) * 100

            print(
                f"💾 VRAM Usage: {allocated:.1f}GB / {total:.1f}GB ({usage_percent:.1f}%)"
            )

            if usage_percent > 90:
                print("⚠️ Very high VRAM usage - may cause OOM errors")
            elif usage_percent < 50:
                print("💡 Low VRAM usage - good for performance")

        return all_good

    def _perform_preflight_checks(self) -> None:
        """Perform pre-flight checks for VRAM, disk space, and system resources"""
        print("🔍 Performing pre-flight checks...")

        # Perform comprehensive checks
        checks = perform_preflight_checks()

        # Display results
        vram_info = checks["vram"]
        disk_info = checks["disk"]

        print(
            f"📊 VRAM: {vram_info['used_gb']:.1f}GB/{vram_info['total_gb']:.1f}GB ({vram_info['usage_percent']:.1f}%)"
        )
        print(f"💾 Disk space: {disk_info['free_gb']:.1f}GB free")
        print(f"🔍 CUDA version: {checks['cuda_version']}")

        # Show recommendations if any
        if checks["recommendations"]:
            print("💡 Recommendations:")
            for rec in checks["recommendations"]:
                print(f"   • {rec}")

    def enhance_prompt(self, prompt: str, language: str = "en") -> str:
        """Enhance prompt with quality keywords"""
        enhancement = PROMPT_ENHANCEMENT.get(language, PROMPT_ENHANCEMENT["en"])
        quality_keywords: str = enhancement["quality_keywords"]

        # Don't add keywords if they're already present
        if not any(
            word in prompt.lower() for word in ["4k", "hd", "quality", "detailed"]
        ):
            return f"{prompt}, {quality_keywords}"
        return prompt

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = HIGH_END_GENERATION_CONFIG["width"],
        height: int = HIGH_END_GENERATION_CONFIG["height"],
        num_inference_steps: int = HIGH_END_GENERATION_CONFIG["num_inference_steps"],
        cfg_scale: float = HIGH_END_GENERATION_CONFIG["true_cfg_scale"],
        seed: Optional[int] = None,
        language: str = "en",
    ) -> Tuple[Optional[PIL.Image.Image], str]:
        """Generate an image with high-end optimizations and pixel window governor"""
        if not self.pipe:
            return None, "Model not loaded. Call load_model() first."

        try:
            # Apply pixel window governor
            original_width, original_height = width, height
            width, height = clamp_image_size(width, height)

            if width != original_width or height != original_height:
                print(
                    f"⚠️ Image size adjusted from {original_width}x{original_height} to {width}x{height} due to VRAM limitations"
                )

            # Enhance prompt if requested
            enhanced_prompt = self.enhance_prompt(prompt, language)

            # Handle random seed
            if seed is None:
                seed = random.randint(0, 2**32 - 1)

            # Create GPU generator (critical for performance)
            generator = torch.Generator(device="cuda").manual_seed(seed)

            print("🎨 HIGH-PERFORMANCE GENERATION STARTING...")
            print(f"Prompt: {enhanced_prompt[:100]}...")
            print(
                f"Settings: {width}x{height}, steps: {num_inference_steps}, CFG: {cfg_scale}"
            )
            print(f"Seed: {seed}")
            print(f"Generator device: {generator.device}")

            # Pre-generation checks
            start_vram = torch.cuda.memory_allocated(0) / 1e9
            print(f"Pre-generation VRAM: {start_vram:.1f}GB")

            # Generation timing
            generation_start = datetime.now()

            # High-performance generation with minimal overhead
            with torch.cuda.device("cuda"):
                with torch.no_grad():
                    # NO cache clearing during generation (impacts performance)
                    # NO device moves during generation
                    # NO verification during generation

                    print("🚀 Starting high-speed generation...")

                    # Optimized generation parameters
                    generation_params = {
                        "prompt": enhanced_prompt,
                        "negative_prompt": negative_prompt,
                        "width": width,
                        "height": height,
                        "num_inference_steps": num_inference_steps,
                        "generator": generator,
                    }

                    # Add CFG parameter (check what Qwen uses)
                    if hasattr(self.pipe, "__call__"):
                        import inspect

                        sig = inspect.signature(self.pipe.__call__)
                        if "guidance_scale" in sig.parameters:
                            generation_params["guidance_scale"] = cfg_scale
                        elif "true_cfg_scale" in sig.parameters:
                            generation_params["true_cfg_scale"] = cfg_scale

                    # GENERATE!
                    result = self.pipe(**generation_params)

            # Calculate timing
            generation_time = (datetime.now() - generation_start).total_seconds()

            # Post-generation stats
            end_vram = torch.cuda.memory_allocated(0) / 1e9
            vram_used = end_vram - start_vram

            print("⚡ GENERATION COMPLETE!")
            print(f"   Time: {generation_time:.1f} seconds")
            print(f"   VRAM used: {vram_used:.1f}GB")
            print(f"   Speed: {generation_time/num_inference_steps:.2f}s per step")

            if generation_time > 100:  # More than 100 seconds is too slow
                print("⚠️ WARNING: Generation took longer than expected!")
                print("   Check GPU utilization and configuration")
            elif generation_time < 30:
                print("🚀 EXCELLENT: Fast generation achieved!")

            image = result.images[0]

            # Save with metadata
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"highend_qwen_{timestamp}_{seed}.png"
            filepath = os.path.join(self.output_dir, filename)

            metadata = {
                "generator": "HighEndQwenImageGenerator",
                "prompt": enhanced_prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "language": language,
                "model": self.model_name,
                "timestamp": timestamp,
                "generation_time": generation_time,
                "performance": {
                    "seconds_per_step": generation_time / num_inference_steps,
                    "vram_used_gb": vram_used,
                    "total_vram_gb": torch.cuda.get_device_properties(0).total_memory
                    / 1e9,
                },
            }

            # Save files
            image.save(filepath)
            metadata_file = filepath.replace(".png", "_metadata.json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            success_msg = "✅ HIGH-PERFORMANCE GENERATION COMPLETE!\n"
            success_msg += f"File: {filename}\n"
            success_msg += f"Time: {generation_time:.1f}s ({generation_time/num_inference_steps:.2f}s per step)\n"
            success_msg += f"Seed: {seed}"

            return image, success_msg

        except Exception as e:
            error_msg = f"❌ High-performance generation failed: {str(e)}"
            print(error_msg)
            return None, error_msg
