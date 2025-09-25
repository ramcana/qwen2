#!/usr/bin/env bash
# Setup models with resumable downloads
set -euo pipefail

echo "🔧 Setting up models..."

# Activate virtual environment
if [ -f ".venv311/bin/activate" ]; then
    source .venv311/bin/activate
    echo "✅ Activated Python 3.11 virtual environment"
else
    echo "❌ Virtual environment not found. Run setup_env.sh first."
    exit 1
fi

# Download models
python tools/download_models.py

echo "🎉 Model setup complete!"