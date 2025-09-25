#!/usr/bin/env bash
# Quick stop script for development environment
set -e

echo "🛑 Stopping Qwen Image Edit Development Environment"
echo "=" * 60

# Kill any running processes
echo "🔍 Checking for running processes..."

# Kill Gradio/FastAPI processes
GRADIO_PIDS=$(pgrep -f "gradio\|fastapi\|uvicorn" 2>/dev/null || true)
if [ ! -z "$GRADIO_PIDS" ]; then
    echo "🔪 Stopping Gradio/FastAPI processes..."
    echo "$GRADIO_PIDS" | xargs kill -TERM 2>/dev/null || true
    sleep 2
    # Force kill if still running
    echo "$GRADIO_PIDS" | xargs kill -KILL 2>/dev/null || true
fi

# Kill Python processes in this project
PROJECT_PIDS=$(pgrep -f "$(pwd)" 2>/dev/null || true)
if [ ! -z "$PROJECT_PIDS" ]; then
    echo "🔪 Stopping project Python processes..."
    echo "$PROJECT_PIDS" | xargs kill -TERM 2>/dev/null || true
fi

# Clear GPU memory if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "🧹 Clearing GPU memory..."
    python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('✅ GPU memory cleared')
" 2>/dev/null || true
fi

# Deactivate virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "🐍 Deactivating virtual environment..."
    deactivate 2>/dev/null || true
fi

echo ""
echo "✅ Development environment stopped"
echo "💡 To restart: run './dev-start.sh'"
echo ""