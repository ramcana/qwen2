#!/bin/bash
# WSL2-Friendly Qwen2 UI Launcher
# Automatically opens browser in Windows after starting the server

echo "🎨 Launching Qwen2 Image Generator..."
echo "======================================"

# Activate virtual environment
source venv/bin/activate

echo "✅ Virtual environment activated"
echo "🚀 Starting Qwen2 UI server..."
echo ""
echo "📋 Instructions:"
echo "1. Server will start on http://localhost:7860"
echo "2. Open your Windows browser"
echo "3. Navigate to: http://localhost:7860"
echo "4. Press Ctrl+C here to stop the server"
echo ""

# Launch the UI
python src/qwen_image_ui.py

echo "🔴 Server stopped"