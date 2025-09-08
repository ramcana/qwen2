#!/usr/bin/env python3
"""
Qwen-Image Generator Launcher
Choose between Standard and Enhanced UI modes
"""

import argparse
import os
import sys

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def launch_standard_ui():
    """Launch the standard Qwen-Image UI"""
    print("🎨 Starting Qwen-Image Standard Generator...")
    print("="*50)
    
    try:
        from qwen_image_ui import create_interface
        
        print("""
Hardware Optimization Status:
- CPU: AMD Ryzen Threadripper PRO (Multi-core)
- RAM: 128GB (Excellent for large models)
- GPU: RTX 4080 (16GB VRAM) - Optimal for Qwen-Image
        
Model: Qwen-Image (20B parameters)
Features:
🎯 Text-to-Image Generation
🌍 Multi-language support (EN/ZH)
📏 Multiple aspect ratios
💾 Auto-save with metadata

Access your interface at: http://localhost:7860
Generated images saved to: ./generated_images/
        """)
        
        # Create and launch the interface
        demo = create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=False,
            max_file_size="50mb"
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error starting standard UI: {e}")
        return False
    return True

def launch_enhanced_ui():
    """Launch the enhanced Qwen-Image UI with advanced features"""
    print("🎨 Starting Qwen-Image Enhanced Generator...")
    print("="*50)
    
    try:
        from qwen_image_enhanced_ui import create_interface
        
        print("""
Hardware Optimization Status:
- CPU: AMD Ryzen Threadripper PRO (Multi-core)
- RAM: 128GB (Excellent for large models)
- GPU: RTX 4080 (16GB VRAM) - Optimal for Qwen-Image

Enhanced Features:
🎯 Text-to-Image Generation (Qwen-Image)
🖼️ Image-to-Image Transformation
🎭 Inpainting with Mask Editor
🔍 Super-Resolution Enhancement
🌍 Multi-language support (EN/ZH)
📏 Multiple aspect ratios
💾 Auto-save with metadata

Access your interface at: http://localhost:7860
Generated images saved to: ./generated_images/
        """)
        
        # Create and launch the enhanced interface
        demo = create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=False,
            max_file_size="50mb"
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error starting enhanced UI: {e}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Qwen-Image Generator Launcher')
    parser.add_argument('--mode', choices=['standard', 'enhanced', 'interactive'], 
                       default='interactive',
                       help='Launch mode: standard UI, enhanced UI, or interactive selection')
    
    args = parser.parse_args()
    
    if args.mode == 'standard':
        launch_standard_ui()
    elif args.mode == 'enhanced':
        launch_enhanced_ui()
    else:
        # Interactive mode - let user choose
        print("\n🎨 Qwen-Image Generator Launcher")
        print("═" * 40)
        print("\nChoose your interface:")
        print("\n1️⃣  Standard UI - Text-to-Image only")
        print("    • Qwen-Image model")
        print("    • Fast and reliable")
        print("    • Original features")
        
        print("\n2️⃣  Enhanced UI - Full Feature Suite")
        print("    • Text-to-Image (Qwen-Image)")
        print("    • Image-to-Image transformation")
        print("    • Inpainting with mask editor")
        print("    • Super-resolution enhancement")
        print("    • Advanced controls")
        
        print("\n0️⃣  Exit")
        
        while True:
            try:
                choice = input("\n🎯 Enter your choice (1/2/0): ").strip()
                
                if choice == '1':
                    print("\n🚀 Launching Standard UI...")
                    launch_standard_ui()
                    break
                elif choice == '2':
                    print("\n🚀 Launching Enhanced UI...")
                    launch_enhanced_ui()
                    break
                elif choice == '0':
                    print("\n👋 Goodbye!")
                    sys.exit(0)
                else:
                    print("❌ Invalid choice. Please enter 1, 2, or 0.")
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
            except EOFError:
                print("\n\n👋 Goodbye!")
                sys.exit(0)

if __name__ == "__main__":
    main()