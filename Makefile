# Qwen Image Generator - Development Shortcuts
.PHONY: help venv deps torch models smoke clean test lint format

# Default target
help:
	@echo "🚀 Qwen Image Generator - Available Commands:"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make venv     - Create Python 3.11 virtual environment"
	@echo "  make deps     - Install Python dependencies"
	@echo "  make torch    - Install PyTorch with CUDA support"
	@echo "  make models   - Download required models (Python method)"
	@echo "  make models-cli - Download required models (CLI method - more reliable)"
	@echo "  make clean-downloads - Clean up partial/corrupted downloads"
	@echo "  make clear-locks - Clear stale download lock files"
	@echo "  make setup    - Complete setup (venv + deps + torch + models)"
	@echo ""
	@echo "Testing Commands:"
	@echo "  make smoke    - Run smoke test"
	@echo "  make test     - Run full test suite"
	@echo ""
	@echo "Development Commands:"
	@echo "  make lint     - Run linting (ruff + flake8)"
	@echo "  make format   - Format code (black + isort)"
	@echo "  make clean    - Clean up generated files"
	@echo ""
	@echo "Application Commands:"
	@echo "  make run      - Start the main application"
	@echo "  make ui       - Start Gradio UI"
	@echo ""
	@echo "Daily Development:"
	@echo "  make dev-start - Start development environment (recommended)"
	@echo "  make dev-stop  - Stop development environment"
	@echo "  make dev-ui    - Quick UI launcher"
	@echo "  make dev-test  - Quick test runner"

# Environment setup
venv:
	@echo "🐍 Creating Python 3.11 virtual environment..."
	python3.11 -m venv .venv311
	@echo "✅ Virtual environment created at .venv311"
	@echo "💡 Activate with: source .venv311/bin/activate"

deps:
	@echo "📦 Installing dependencies..."
	. .venv311/bin/activate && pip install -U pip setuptools wheel
	. .venv311/bin/activate && pip uninstall -y diffusers || true
	. .venv311/bin/activate && pip install -r requirements.txt

torch:
	@echo "🔥 Installing PyTorch with CUDA support..."
	. .venv311/bin/activate && pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

models:
	@echo "📥 Downloading models (Python method)..."
	. .venv311/bin/activate && python tools/download_models.py

models-cli:
	@echo "📥 Downloading models (CLI method - more reliable)..."
	. .venv311/bin/activate && python tools/download_models_cli.py

clean-downloads:
	@echo "🧹 Cleaning up partial downloads..."
	. .venv311/bin/activate && python tools/clean_downloads.py

clear-locks:
	@echo "🔓 Clearing stale download locks..."
	. .venv311/bin/activate && python tools/clear_locks.py

setup: venv deps torch models
	@echo "🎉 Complete setup finished!"
	@echo "💡 Activate environment: source .venv311/bin/activate"

# Testing
smoke:
	@echo "🧪 Running smoke test..."
	. .venv311/bin/activate && python examples/qwen_edit_smoke.py

test:
	@echo "🧪 Running test suite..."
	. .venv311/bin/activate && pytest tests/ -v

# Development tools
lint:
	@echo "🔍 Running linting..."
	. .venv311/bin/activate && ruff check src/ tests/ examples/
	. .venv311/bin/activate && flake8 src/ tests/ examples/

format:
	@echo "🎨 Formatting code..."
	. .venv311/bin/activate && black src/ tests/ examples/ tools/
	. .venv311/bin/activate && isort src/ tests/ examples/ tools/

# Application
run:
	@echo "🚀 Starting main application..."
	. .venv311/bin/activate && python start.py

ui:
	@echo "🎨 Starting Gradio UI..."
	. .venv311/bin/activate && python src/qwen_image_ui.py

# Daily development shortcuts
dev-start:
	@echo "🚀 Starting development environment..."
	./dev-start.sh

dev-stop:
	@echo "🛑 Stopping development environment..."
	./dev-stop.sh

dev-ui:
	@echo "🎨 Quick UI launcher..."
	./dev-ui.sh

dev-test:
	@echo "🧪 Quick test runner..."
	./dev-test.sh

# Cleanup
clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
	@echo "✅ Cleanup complete"