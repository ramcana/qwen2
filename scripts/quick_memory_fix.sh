#!/bin/bash
# Quick CUDA Memory Fix Script
# Runs the automated memory fix tool

echo "🔧 Quick CUDA Memory Fix"
echo "========================"

# Navigate to project root
cd "$(dirname "$0")/.." || exit 1

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate || {
    echo "❌ Failed to activate virtual environment"
    echo "💡 Make sure you're in the Qwen2 project directory"
    exit 1
}

# Run the automated fix
echo "🚀 Running automated CUDA memory fix..."
python tools/fix_cuda_memory.py

echo ""
echo "✅ Memory fix complete!"
echo "💡 You can now try initializing the model again in the React UI"
