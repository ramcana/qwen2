#!/bin/bash
# WSL2-Friendly Qwen2 UI Launcher with Enhanced Mode Support
# Choose between Standard and Enhanced UI

echo "🎨 Launching Qwen2 Image Generator..."
echo "======================================"

# Activate virtual environment
source venv/bin/activate

echo "✅ Virtual environment activated"
echo ""
echo "🎆 Choose your interface:"
echo ""
echo "1. Standard UI  - Text-to-Image only (Fast)"
echo "2. Enhanced UI  - Full feature suite (Advanced)"
echo "3. Interactive  - Choose from Python launcher"
echo "0. Exit"
echo ""

while true; do
    read -p "🎯 Enter your choice (1/2/3/0): " choice
    case $choice in
        1)
            echo ""
            echo "🚀 Starting Standard Qwen2 UI..."
            echo "📋 Features: Text-to-Image, Multi-language, Aspect ratios"
            echo "🌐 Server will start on http://localhost:7860"
            echo "💡 Open that URL in your Windows browser"
            echo "⏹️  Press Ctrl+C to stop the server"
            echo ""
            python src/qwen_image_ui.py
            break
            ;;
        2)
            echo ""
            echo "🚀 Starting Enhanced Qwen2 UI..."
            echo "📋 Features: Text-to-Image, Img2Img, Inpainting, Super-Resolution"
            echo "🌐 Server will start on http://localhost:7860"
            echo "💡 Open that URL in your Windows browser"
            echo "⏹️  Press Ctrl+C to stop the server"
            echo ""
            python src/qwen_image_enhanced_ui.py
            break
            ;;
        3)
            echo ""
            echo "🚀 Starting Interactive Launcher..."
            echo ""
            python launch.py
            break
            ;;
        0)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid choice. Please enter 1, 2, 3, or 0."
            ;;
    esac
done

echo "🔴 Server stopped"