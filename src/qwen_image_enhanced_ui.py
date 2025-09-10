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
        # Check if enhanced features are available
        if qwen_generator.edit_pipe:
            return (
                "✅ Qwen-Image Enhanced Generator loaded successfully! All modes ready."
            )
        else:
            return "✅ Qwen-Image loaded successfully! Text-to-Image ready.\n⚠️ Enhanced features (Image-to-Image, Inpainting) unavailable.\nRun 'python download_qwen_edit.py' to enable them."
    else:
        return "❌ Failed to load model. Check console for errors."


def generate_image_ui(
    prompt,
    negative_prompt,
    width,
    height,
    steps,
    cfg_scale,
    seed,
    language,
    enhance_prompt,
):
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
        enhance_prompt_flag=enhance_prompt,
    )

    return image, message


def generate_img2img_ui(
    prompt,
    negative_prompt,
    init_image,
    strength,
    width,
    height,
    steps,
    cfg_scale,
    seed,
    language,
    enhance_prompt,
):
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
        enhance_prompt_flag=enhance_prompt,
    )

    return image, message


def generate_inpaint_ui(
    prompt,
    negative_prompt,
    init_image,
    mask_image,
    width,
    height,
    steps,
    cfg_scale,
    seed,
    language,
    enhance_prompt,
):
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
        enhance_prompt_flag=enhance_prompt,
    )

    return image, message


def super_resolution_ui(input_image, scale_factor):
    """UI wrapper for super resolution"""
    if input_image is None:
        return None, "Please upload an image to enhance."

    image, message = qwen_generator.super_resolution(
        image=input_image, scale_factor=scale_factor
    )

    return image, message


def update_ui_for_mode(mode):
    """Update UI visibility based on selected mode"""
    # Return visibility updates for different components
    is_txt2img = mode == "Text-to-Image"
    is_img2img = mode == "Image-to-Image"
    is_inpaint = mode == "Inpainting"
    is_superres = mode == "Super-Resolution"

    return (
        gr.update(visible=True),  # prompt always visible except superres
        gr.update(visible=not is_superres),  # negative prompt
        gr.update(visible=not is_superres),  # language
        gr.update(visible=not is_superres),  # enhance prompt
        gr.update(visible=is_img2img or is_inpaint),  # init image upload
        gr.update(visible=is_inpaint),  # mask editor
        gr.update(visible=is_img2img),  # strength control
        gr.update(visible=is_superres),  # scale factor
        gr.update(visible=not is_superres),  # dimensions
        gr.update(visible=not is_superres),  # generation settings
        gr.update(visible=is_superres),  # superres upload
    )


# Convert aspect ratios for UI
UI_ASPECT_RATIOS = {
    "Square (1:1)": ASPECT_RATIOS["1:1"],
    "Landscape (16:9)": ASPECT_RATIOS["16:9"],
    "Portrait (9:16)": ASPECT_RATIOS["9:16"],
    "Photo (4:3)": ASPECT_RATIOS["4:3"],
    "Portrait Photo (3:4)": ASPECT_RATIOS["3:4"],
    "Classic Photo (3:2)": ASPECT_RATIOS["3:2"],
    "Portrait Classic (2:3)": ASPECT_RATIOS["2:3"],
    "Widescreen (21:9)": ASPECT_RATIOS["21:9"],
    "Custom": (1024, 1024),
}


def update_dimensions(aspect_ratio):
    """Update width/height based on aspect ratio selection"""
    if aspect_ratio in UI_ASPECT_RATIOS and aspect_ratio != "Custom":
        w, h = UI_ASPECT_RATIOS[aspect_ratio]
        return w, h
    return gr.update(), gr.update()


# Create Enhanced Gradio interface
def create_interface():
    with gr.Blocks(
        title="Qwen-Image Enhanced Generator",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 1600px !important; }
        .generate-btn { background: linear-gradient(45deg, #FF6B6B, #4ECDC4) !important; }
        .mode-selector { font-size: 16px !important; font-weight: bold !important; }
        """,
    ) as demo:
        gr.Markdown(
            """
        # 🎨 Qwen-Image Enhanced Generator
        ### Professional AI Image Generation Suite - RTX 4080 Optimized
        **Features:** Text-to-Image • Image-to-Image • Inpainting • Super-Resolution
        """
        )

        with gr.Row():
            # Model initialization
            init_btn = gr.Button(
                "🚀 Initialize Enhanced Generator", variant="primary", scale=1
            )
            status_display = gr.Textbox(
                value="Click 'Initialize Enhanced Generator' to load all models.\nNote: Enhanced features require Qwen-Image-Edit (~20GB download).",
                label="Status",
                interactive=False,
                scale=2,
            )

        # Generation Mode Selector
        with gr.Group():
            gr.Markdown("### 🎛️ Generation Mode")
            generation_mode = gr.Radio(
                choices=[
                    "Text-to-Image",
                    "Image-to-Image",
                    "Inpainting",
                    "Super-Resolution",
                ],
                value="Text-to-Image",
                label="Select Generation Mode",
                elem_classes=["mode-selector"],
            )

        with gr.Row():
            # Left Panel - Controls
            with gr.Column(scale=1):
                # Prompt inputs (visible for all modes except super-res)
                with gr.Group(visible=True) as prompt_group:
                    gr.Markdown("### 📝 Prompt Settings")
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="A coffee shop entrance with a chalkboard sign reading 'Qwen Coffee ☕ $2 per cup'...",
                        lines=4,
                        max_lines=8,
                    )

                    negative_prompt_input = gr.Textbox(
                        label="Negative Prompt (optional)",
                        placeholder="blurry, low quality, distorted...",
                        lines=2,
                        visible=True,
                    )

                    with gr.Row(visible=True) as lang_enhance_row:
                        language_choice = gr.Radio(
                            choices=["en", "zh"], value="en", label="Language"
                        )

                        enhance_prompt_toggle = gr.Checkbox(
                            value=True, label="Enhance Prompt"
                        )

                # Image Upload Components
                with gr.Group(visible=False) as init_image_group:
                    gr.Markdown("### 🖼️ Input Image")
                    init_image_upload = gr.Image(label="Upload Image", type="pil")

                # Inpainting Mask Component
                with gr.Group(visible=False) as mask_group:
                    gr.Markdown("### 🎭 Inpainting Mask")
                    gr.Markdown(
                        "*Draw white areas where you want to generate new content*"
                    )
                    mask_editor = gr.ImageEditor(
                        label="Draw Mask (White = Inpaint Area)",
                        type="pil",
                        brush=gr.Brush(colors=["white"], color_mode="fixed"),
                    )

                # Image-to-Image Strength Control
                with gr.Group(visible=False) as strength_group:
                    gr.Markdown("### 💪 Transformation Strength")
                    strength_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        step=0.05,
                        value=0.8,
                        label="Strength",
                        info="How much to modify the input image (0.1=subtle, 1.0=completely new)",
                    )

                # Super-Resolution Controls
                with gr.Group(visible=False) as superres_group:
                    gr.Markdown("### 🔍 Super-Resolution Settings")
                    superres_image_upload = gr.Image(
                        label="Upload Image to Enhance", type="pil"
                    )
                    scale_factor_slider = gr.Slider(
                        minimum=2,
                        maximum=4,
                        step=1,
                        value=2,
                        label="Scale Factor",
                        info="How much to enlarge the image",
                    )

                # Image dimensions (hidden for super-res)
                with gr.Group(visible=True) as dimensions_group:
                    gr.Markdown("### 📐 Image Dimensions")
                    aspect_ratio_dropdown = gr.Dropdown(
                        choices=list(UI_ASPECT_RATIOS.keys()),
                        value="Landscape (16:9)",
                        label="Aspect Ratio Preset",
                    )

                    with gr.Row():
                        width_slider = gr.Slider(
                            minimum=512,
                            maximum=2048,
                            step=64,
                            value=1664,
                            label="Width",
                        )
                        height_slider = gr.Slider(
                            minimum=512,
                            maximum=2048,
                            step=64,
                            value=928,
                            label="Height",
                        )

                # Generation settings (hidden for super-res)
                with gr.Group(visible=True) as generation_group:
                    gr.Markdown("### ⚙️ Generation Settings")
                    steps_slider = gr.Slider(
                        minimum=10,
                        maximum=100,
                        step=5,
                        value=50,
                        label="Inference Steps",
                    )

                    cfg_slider = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        step=0.5,
                        value=4.0,
                        label="CFG Scale",
                    )

                    seed_input = gr.Number(
                        value=-1, label="Seed (-1 for random)", precision=0
                    )

            # Right Panel - Generation and Results
            with gr.Column(scale=2):
                # Generate buttons (different for each mode)
                with gr.Group():
                    txt2img_btn = gr.Button(
                        "🎨 Generate Image",
                        variant="primary",
                        size="lg",
                        elem_classes=["generate-btn"],
                        visible=True,
                    )

                    img2img_btn = gr.Button(
                        "🖼️ Transform Image",
                        variant="primary",
                        size="lg",
                        elem_classes=["generate-btn"],
                        visible=False,
                    )

                    inpaint_btn = gr.Button(
                        "🎭 Inpaint Image",
                        variant="primary",
                        size="lg",
                        elem_classes=["generate-btn"],
                        visible=False,
                    )

                    superres_btn = gr.Button(
                        "🔍 Enhance Resolution",
                        variant="primary",
                        size="lg",
                        elem_classes=["generate-btn"],
                        visible=False,
                    )

                # Results
                result_image = gr.Image(
                    label="Generated Image",
                    height=600,
                    show_download_button=True,
                    show_share_button=False,
                )

                result_message = gr.Textbox(
                    label="Generation Info", lines=3, max_lines=5, interactive=False
                )

        # Quick settings presets
        with gr.Accordion("🎯 Quick Settings", open=False):
            with gr.Row():
                quality_preset = gr.Button("🏆 High Quality")
                balanced_preset = gr.Button("⚖️ Balanced")
                fast_preset = gr.Button("⚡ Fast Preview")

        # Event handlers
        init_btn.click(fn=initialize_model, outputs=[status_display])

        # Mode change handler
        generation_mode.change(
            fn=update_ui_for_mode,
            inputs=[generation_mode],
            outputs=[
                prompt_group,
                negative_prompt_input,
                language_choice,
                enhance_prompt_toggle,
                init_image_group,
                mask_group,
                strength_group,
                superres_group,
                dimensions_group,
                generation_group,
                superres_image_upload,
            ],
        )

        # Also update button visibility based on mode
        def update_buttons(mode):
            return (
                gr.update(visible=mode == "Text-to-Image"),
                gr.update(visible=mode == "Image-to-Image"),
                gr.update(visible=mode == "Inpainting"),
                gr.update(visible=mode == "Super-Resolution"),
            )

        generation_mode.change(
            fn=update_buttons,
            inputs=[generation_mode],
            outputs=[txt2img_btn, img2img_btn, inpaint_btn, superres_btn],
        )

        # Aspect ratio change handler
        aspect_ratio_dropdown.change(
            fn=update_dimensions,
            inputs=[aspect_ratio_dropdown],
            outputs=[width_slider, height_slider],
        )

        # Generation handlers for different modes
        txt2img_btn.click(
            fn=generate_image_ui,
            inputs=[
                prompt_input,
                negative_prompt_input,
                width_slider,
                height_slider,
                steps_slider,
                cfg_slider,
                seed_input,
                language_choice,
                enhance_prompt_toggle,
            ],
            outputs=[result_image, result_message],
        )

        img2img_btn.click(
            fn=generate_img2img_ui,
            inputs=[
                prompt_input,
                negative_prompt_input,
                init_image_upload,
                strength_slider,
                width_slider,
                height_slider,
                steps_slider,
                cfg_slider,
                seed_input,
                language_choice,
                enhance_prompt_toggle,
            ],
            outputs=[result_image, result_message],
        )

        inpaint_btn.click(
            fn=generate_inpaint_ui,
            inputs=[
                prompt_input,
                negative_prompt_input,
                init_image_upload,
                mask_editor,
                width_slider,
                height_slider,
                steps_slider,
                cfg_slider,
                seed_input,
                language_choice,
                enhance_prompt_toggle,
            ],
            outputs=[result_image, result_message],
        )

        superres_btn.click(
            fn=super_resolution_ui,
            inputs=[superres_image_upload, scale_factor_slider],
            outputs=[result_image, result_message],
        )

        # Preset handlers
        quality_preset.click(lambda: (80, 7.0), outputs=[steps_slider, cfg_slider])
        balanced_preset.click(lambda: (50, 4.0), outputs=[steps_slider, cfg_slider])
        fast_preset.click(lambda: (20, 3.0), outputs=[steps_slider, cfg_slider])

    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()

    print(
        """
    🎨 Starting Qwen-Image Enhanced Generator

    Hardware Detected:
    - CPU: AMD Ryzen Threadripper PRO 5995WX (64 cores)
    - RAM: 128GB
    - GPU: RTX 4080 (16GB VRAM)

    Enhanced Features:
    🎯 Text-to-Image Generation (Qwen-Image)
    🖼️ Image-to-Image Transformation
    🎭 Inpainting with Mask Editor
    🔍 Super-Resolution Enhancement
    🌍 Multi-language support (EN/ZH)
    📏 Multiple aspect ratios
    💾 Auto-save with metadata

    🌐 Access your interface at: http://localhost:7860
    📁 Generated images saved to: ./generated_images/
    """
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=False,
        max_file_size="50mb",
    )
