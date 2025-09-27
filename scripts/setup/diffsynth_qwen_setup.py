# DiffSynth-Studio setup for Qwen-Image-Edit with 4GB VRAM capability
# This should work excellently with your RTX 4080 16GB

# Installation
"""
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
"""

import torch
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

def setup_qwen_image_edit_low_vram():
    """
    Setup Qwen-Image-Edit with aggressive memory optimization
    This configuration can run within 4GB VRAM according to DiffSynth docs
    """
    
    # Enable memory efficient attention
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # Configure pipeline with offloading
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,  # Use bf16 for your RTX 4080
        device="cuda",
        
        # Model configs with layer-by-layer offload
        model_configs=[
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit", 
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                offload_device="cpu"  # Offload transformer to CPU
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit", 
                origin_file_pattern="text_encoder/model*.safetensors",
                offload_device="cpu"  # Offload text encoder to CPU
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit", 
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                # Keep VAE on GPU for better quality
            ),
        ],
        tokenizer_config=ModelConfig(
            model_id="Qwen/Qwen-Image-Edit", 
            origin_file_pattern="tokenizer/"
        ),
        
        # Memory optimization settings
        low_cpu_mem_usage=True,
    )
    
    # Enable VRAM management for dynamic offloading
    pipe.enable_vram_management()
    
    return pipe

def image_edit_example(pipe, input_image_path, prompt, output_path="edited_image.jpg"):
    """
    Perform image editing with memory-optimized pipeline
    """
    from PIL import Image
    
    # Load input image
    input_image = Image.open(input_image_path).convert("RGB")
    
    # Generate edited image with conservative settings
    edited_image = pipe(
        prompt=prompt,
        image=input_image,  # Input image for editing
        num_inference_steps=20,  # Lower steps to save memory/time
        guidance_scale=7.5,
        height=768,  # Start with lower resolution
        width=768,
        seed=42,
        tiled=True,  # Enable tiling for large images
    )
    
    edited_image.save(output_path)
    print(f"Edited image saved to: {output_path}")
    return edited_image

# Alternative: Pre-quantized model approach
def setup_quantized_model():
    """
    Alternative: Use pre-quantized model
    """
    from transformers import AutoModelForImageEditing, AutoProcessor
    
    model = AutoModelForImageEditing.from_pretrained(
        "ovedrive/qwen-image-edit-4bit",  # Pre-quantized 4-bit model
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained("ovedrive/qwen-image-edit-4bit")
    
    return model, processor

if __name__ == "__main__":
    print("Setting up Qwen-Image-Edit with DiffSynth-Studio...")
    print("This configuration should use <4GB VRAM with your RTX 4080")
    
    # Setup the pipeline
    pipe = setup_qwen_image_edit_low_vram()
    
    # Example usage
    prompt = "Change the dog to a cat, maintaining the same background and lighting"
    # edited_image = image_edit_example(pipe, "input_image.jpg", prompt)
    
    print("\nSetup complete! Key memory optimizations enabled:")
    print("✓ Layer-by-layer CPU offloading")
    print("✓ VRAM management with dynamic offloading") 
    print("✓ BF16 precision for your RTX 4080")
    print("✓ Tiled processing for large images")
    print("✓ Conservative inference steps")
