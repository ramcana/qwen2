#!/usr/bin/env python3
"""
Enhanced Qwen-Image Generator Launcher
Advanced Image Generation Suite with multiple modes
"""

import os
import sys

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.insert(0, src_dir)


def main():
    print("🎨 Starting Enhanced Qwen-Image Generator...")
    print("=" * 60)

    try:
        from qwen_image_enhanced_ui import create_interface

        print(
            """
🚀 Enhanced Generator Features:

Hardware Optimization:
- CPU: AMD Ryzen Threadripper PRO (Multi-core)
- RAM: 128GB (Excellent for large models)
- GPU: RTX 4080 (16GB VRAM) - Optimal for AI generation

Generation Modes:
🎯 Text-to-Image: Qwen-Image model for text rendering
🖼️ Image-to-Image: Transform existing images with AI
🎭 Inpainting: Fill masked areas with AI-generated content
🔍 Super-Resolution: Enhance image quality and size

Advanced Features:
✅ Multiple model pipeline integration
✅ Dynamic UI mode switching
✅ Strength control for transformations
✅ Interactive mask editor for inpainting
✅ Multi-language prompt enhancement
✅ Professional quality presets

Access your interface at: http://localhost:7860
Generated images saved to: ./generated_images/
        """
        )

        # Create and launch the interface
        demo = create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=False,
            max_file_size="50mb",
        )

    except Exception as e:
        print(f"❌ Error starting enhanced generator: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure all dependencies are installed")
        print("2. Check if models are accessible")
        print("3. Verify GPU drivers are up to date")
        return False


if __name__ == "__main__":
    main()
