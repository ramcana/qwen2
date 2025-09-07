#!/usr/bin/env python3
"""
Qwen-Image Local Generator Launcher
Simple launcher script for the MVP
"""

import sys
import os

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def main():
    print("🎨 Starting Qwen-Image Local Generator...")
    print("=" * 50)
    
    try:
        from qwen_image_ui import create_interface
        
        print("""
Hardware Optimization Status:
- CPU: AMD Ryzen Threadripper PRO (Multi-core)
- RAM: 128GB (Excellent for large models)
- GPU: RTX 4080 (16GB VRAM) - Optimal for Qwen-Image
        
Model: Qwen-Image (20B parameters)
Optimizations:
✅ bfloat16 precision for RTX 4080
✅ Attention slicing for memory efficiency
✅ Automatic prompt enhancement
✅ Multiple aspect ratio presets

Features Available:
🎯 Advanced text rendering in images
🌍 Multi-language support (EN/ZH)
📏 Multiple aspect ratios
💾 Auto-save with metadata
🎛️ Professional controls

Access your interface at: http://localhost:7860
Generated images saved to: ./generated_images/
        """)
        
        # Create and launch the interface
        demo = create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True,
            max_file_size="50mb"
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()