#!/usr/bin/env python3
"""
Qwen-Image Enhanced UI - Advanced Image Generation Suite
Optimized for RTX 4080 + AMD Threadripper Setup
Features: Text-to-Image, Image-to-Image, Inpainting, Super-Resolution
"""

import os
import sys

import gradio as gr

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qwen_generator import QwenImageGenerator
from src.qwen_image_config import ASPECT_RATIOS

# Initialize the generator
qwen_generator = QwenImageGenerator()

def initialize_model():
    """Initialize the model and return status"""
    success = qwen_generator.load_model()
    if success:
        return "✅ Qwen-Image model loaded successfully! Ready for image generation."
    else:
        return "❌ Failed to load model. Check console for errors."

def generate_image_ui(prompt, negative_prompt, width, height, steps, cfg_scale, 
                     seed, language, enhance_prompt):
    """UI wrapper for text-to-image generation"""
    if not prompt.strip():
        return None, "Please enter a prompt to generate an image."
    
    image, message = qwen_generator.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        language=language,
        enhance_prompt_flag=enhance_prompt
    )
    
    return image, message

def generate_img2img_ui(prompt, negative_prompt, init_image, strength, width, height, 
                       steps, cfg_scale, seed, language, enhance_prompt):
    """UI wrapper for image-to-image generation"""
    if not prompt.strip():
        return None, "Please enter a prompt for image-to-image generation."
    
    if init_image is None:
        return None, "Please upload an input image for image-to-image generation."
    
    image, message = qwen_generator.generate_img2img(
        prompt=prompt,
        init_image=init_image,
        strength=strength,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        language=language,
        enhance_prompt_flag=enhance_prompt
    )
    
    return image, message

def generate_inpaint_ui(prompt, negative_prompt, init_image, mask_image, width, height,
                       steps, cfg_scale, seed, language, enhance_prompt):
    """UI wrapper for inpainting generation"""
    if not prompt.strip():
        return None, "Please enter a prompt for inpainting."
    
    if init_image is None:
        return None, "Please upload an input image for inpainting."
        
    if mask_image is None:
        return None, "Please provide a mask for inpainting."
    
    image, message = qwen_generator.generate_inpaint(
        prompt=prompt,
        init_image=init_image,
        mask_image=mask_image,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        language=language,
        enhance_prompt_flag=enhance_prompt
    )
    
    return image, message

def super_resolution_ui(input_image, scale_factor):
    """UI wrapper for super resolution"""
    if input_image is None:
        return None, "Please upload an image to enhance."
    
    image, message = qwen_generator.super_resolution(
        image=input_image,
        scale_factor=scale_factor
    )
    
    return image, message

# Convert aspect ratios for UI
UI_ASPECT_RATIOS = {
    "Square (1:1)": ASPECT_RATIOS["1:1"],
    "Landscape (16:9)": ASPECT_RATIOS["16:9"], 
    "Portrait (9:16)": ASPECT_RATIOS["9:16"],
    "Photo (4:3)": ASPECT_RATIOS["4:3"],
    "Portrait Photo (3:4)": ASPECT_RATIOS["3:4"],
    "Classic Photo (3:2)": ASPECT_RATIOS["3:2"],  # New official ratio
    "Portrait Classic (2:3)": ASPECT_RATIOS["2:3"],  # New official ratio
    "Widescreen (21:9)": ASPECT_RATIOS["21:9"],
    "Custom": (1024, 1024)  # Will be overridden by manual input
}

def update_dimensions(aspect_ratio):
    """Update width/height based on aspect ratio selection"""
    if aspect_ratio in UI_ASPECT_RATIOS and aspect_ratio != "Custom":
        w, h = UI_ASPECT_RATIOS[aspect_ratio]
        return w, h
    return gr.update(), gr.update()  # Don't change for custom

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="Qwen-Image Generator", 
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 1400px !important; }
        .aspect-ratio-gallery { height: 120px; }
        .generate-btn { background: linear-gradient(45deg, #FF6B6B, #4ECDC4) !important; }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🎨 Qwen-Image Local Generator
        ### Professional Text-to-Image Generation on your RTX 4080
        Generate high-quality images with advanced text rendering and artistic control
        """)
        
        with gr.Row():
            # Model initialization
            init_btn = gr.Button("🚀 Initialize Qwen-Image Model", variant="primary", scale=1)
            status_display = gr.Textbox(
                value="Click 'Initialize Model' to load Qwen-Image",
                label="Status",
                interactive=False,
                scale=2
            )
        
        with gr.Row():
            # Left Panel - Controls
            with gr.Column(scale=1):
                
                # Prompt inputs
                with gr.Group():
                    gr.Markdown("### 📝 Prompt Settings")
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="A coffee shop entrance with a chalkboard sign reading 'Qwen Coffee ☕ $2 per cup'...",
                        lines=4,
                        max_lines=8
                    )
                    
                    negative_prompt_input = gr.Textbox(
                        label="Negative Prompt (optional)",
                        placeholder="blurry, low quality, distorted...",
                        lines=2
                    )
                    
                    language_choice = gr.Radio(
                        choices=["en", "zh"], 
                        value="en", 
                        label="Language",
                        info="Choose prompt language for better enhancement"
                    )
                    
                    enhance_prompt_toggle = gr.Checkbox(
                        value=True, 
                        label="Enhance Prompt",
                        info="Add quality keywords automatically"
                    )
                
                # Image dimensions
                with gr.Group():
                    gr.Markdown("### 📐 Image Dimensions")
                    aspect_ratio_dropdown = gr.Dropdown(
                        choices=list(UI_ASPECT_RATIOS.keys()),
                        value="Landscape (16:9)",
                        label="Aspect Ratio Preset"
                    )
                    
                    with gr.Row():
                        width_slider = gr.Slider(
                            minimum=512, maximum=2048, step=64, value=1664,
                            label="Width"
                        )
                        height_slider = gr.Slider(
                            minimum=512, maximum=2048, step=64, value=928,
                            label="Height"
                        )
                
                # Generation settings
                with gr.Group():
                    gr.Markdown("### ⚙️ Generation Settings")
                    steps_slider = gr.Slider(
                        minimum=10, maximum=100, step=5, value=50,
                        label="Inference Steps",
                        info="More steps = higher quality (slower)"
                    )
                    
                    cfg_slider = gr.Slider(
                        minimum=1.0, maximum=20.0, step=0.5, value=4.0,
                        label="CFG Scale",
                        info="How closely to follow the prompt"
                    )
                    
                    seed_input = gr.Number(
                        value=-1, 
                        label="Seed (-1 for random)",
                        precision=0,
                        info="Use same seed for reproducible results"
                    )
            
            # Right Panel - Generation and Results
            with gr.Column(scale=2):
                
                # Generate button
                generate_btn = gr.Button(
                    "🎨 Generate Image", 
                    variant="primary", 
                    size="lg",
                    elem_classes=["generate-btn"]
                )
                
                # Results
                result_image = gr.Image(
                    label="Generated Image",
                    height=600,
                    show_download_button=True,
                    show_share_button=False
                )
                
                result_message = gr.Textbox(
                    label="Generation Info",
                    lines=3,
                    max_lines=5,
                    interactive=False
                )
        
        # Example prompts
        with gr.Accordion("💡 Example Prompts", open=False):
            example_prompts = [
                ["A futuristic coffee shop with neon signs reading 'AI Café' and 'Welcome' in both English and Chinese, cyberpunk style", ""],
                ["A beautiful landscape painting with text overlay reading 'Qwen Mountain Resort - Est. 2025', traditional Chinese painting style", ""],
                ["A modern poster design with the text 'Innovation Summit 2025' in bold letters, minimalist design, blue and white color scheme", ""],
                ["A bookstore window display with books and a sign reading 'New Arrivals - Fantasy & Sci-Fi', cozy lighting, autumn atmosphere", ""],
                ["A vintage travel poster showing mountains with text 'Visit Beautiful Tibet - Experience the Culture', retro illustration style", ""]
            ]
            
            gr.Examples(
                examples=example_prompts,
                inputs=[prompt_input, negative_prompt_input],
                label="Click to try these example prompts"
            )
        
        # Quick settings presets
        with gr.Accordion("🎯 Quick Presets", open=False):
            with gr.Row():
                quality_preset = gr.Button("🏆 High Quality (slow)")
                balanced_preset = gr.Button("⚖️ Balanced (recommended)")  
                fast_preset = gr.Button("⚡ Fast Preview")
        
        # Event handlers
        init_btn.click(
            fn=initialize_model,
            outputs=[status_display]
        )
        
        # Aspect ratio change handler
        aspect_ratio_dropdown.change(
            fn=update_dimensions,
            inputs=[aspect_ratio_dropdown],
            outputs=[width_slider, height_slider]
        )
        
        # Generation handler
        generate_btn.click(
            fn=generate_image_ui,
            inputs=[
                prompt_input, negative_prompt_input, width_slider, height_slider,
                steps_slider, cfg_slider, seed_input, language_choice, enhance_prompt_toggle
            ],
            outputs=[result_image, result_message]
        )
        
        # Preset handlers
        quality_preset.click(
            lambda: (80, 7.0),
            outputs=[steps_slider, cfg_slider]
        )
        balanced_preset.click(
            lambda: (50, 4.0),
            outputs=[steps_slider, cfg_slider]
        )
        fast_preset.click(
            lambda: (20, 3.0),
            outputs=[steps_slider, cfg_slider]
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    
    print("""
    🎨 Starting Qwen-Image Local Generator
    
    Hardware Detected:
    - CPU: AMD Ryzen Threadripper PRO 5995WX (64 cores)
    - RAM: 128GB  
    - GPU: RTX 4080 (16GB VRAM)
    
    Model: Qwen-Image (20B parameters)
    Optimizations Applied:
    ✅ bfloat16 precision for RTX 4080
    ✅ Attention slicing for memory efficiency
    ✅ Automatic prompt enhancement
    ✅ Multiple aspect ratio presets
    
    Features:
    🎯 Advanced text rendering in images
    🌍 Multi-language support (EN/ZH)
    📏 Multiple aspect ratios
    💾 Auto-save with metadata
    🎛️ Professional controls
    
    🌐 Access your interface at: http://localhost:7860
    📁 Generated images saved to: ./generated_images/
    
    💡 Note: Open http://localhost:7860 in your Windows browser
    """)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set True for public access
        in_browser=False,  # Disabled for WSL2 compatibility
        max_file_size="50mb"
    )