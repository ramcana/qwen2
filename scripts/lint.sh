#!/bin/bash
# Comprehensive linting and code quality script for Qwen-Image project
# Optimized for RTX 4080 + AMD Threadripper development environment

set -e  # Exit on any error

echo "🔍 Starting comprehensive code quality checks..."
echo "=================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_warning "Virtual environment not detected. Activating..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_error "Virtual environment not found. Run setup.sh first."
        exit 1
    fi
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# 1. Code formatting with Black
print_status "Running Black formatter..."
if black --check --diff src/ tests/ examples/ scripts/ launch.py 2>&1 | tee logs/black.log; then
    print_success "✅ Black formatting check passed"
else
    print_warning "❌ Black formatting issues found. Run 'black src/ tests/ examples/ scripts/ launch.py' to fix"
fi

# 2. Import sorting with isort
print_status "Checking import sorting with isort..."
if isort --check-only --diff src/ tests/ examples/ scripts/ launch.py 2>&1 | tee logs/isort.log; then
    print_success "✅ Import sorting check passed"
else
    print_warning "❌ Import sorting issues found. Run 'isort src/ tests/ examples/ scripts/ launch.py' to fix"
fi

# 3. Linting with flake8
print_status "Running flake8 linter..."
if flake8 src/ tests/ examples/ scripts/ launch.py 2>&1 | tee logs/flake8.log; then
    print_success "✅ Flake8 linting passed"
else
    print_error "❌ Flake8 found linting issues"
fi

# 4. Modern linting with ruff (faster alternative)
print_status "Running ruff linter..."
if command -v ruff &> /dev/null; then
    if ruff check src/ tests/ examples/ scripts/ launch.py 2>&1 | tee logs/ruff.log; then
        print_success "✅ Ruff linting passed"
    else
        print_error "❌ Ruff found linting issues"
    fi
else
    print_warning "Ruff not installed, skipping..."
fi

# 5. Type checking with mypy
print_status "Running mypy type checker..."
if mypy src/ 2>&1 | tee logs/mypy.log; then
    print_success "✅ MyPy type checking passed"
else
    print_warning "❌ MyPy found type issues"
fi

# 6. Security scanning with bandit
print_status "Running security scan with bandit..."
if command -v bandit &> /dev/null; then
    if bandit -r src/ -f json -o logs/bandit.json 2>&1 | tee logs/bandit.log; then
        print_success "✅ Security scan passed"
    else
        print_warning "❌ Security issues found"
    fi
else
    print_warning "Bandit not installed, skipping security scan..."
fi

# 7. Spell checking
print_status "Running spell check..."
if command -v codespell &> /dev/null; then
    if codespell src/ docs/ README.md --skip="*.log,*.json,venv,generated_images,models" 2>&1 | tee logs/codespell.log; then
        print_success "✅ Spell check passed"
    else
        print_warning "❌ Spelling errors found"
    fi
else
    print_warning "Codespell not installed, skipping spell check..."
fi

# 8. Check for Python syntax errors
print_status "Checking Python syntax..."
if python -m py_compile src/*.py tests/*.py examples/*.py launch.py 2>&1 | tee logs/syntax.log; then
    print_success "✅ Python syntax check passed"
else
    print_error "❌ Python syntax errors found"
fi

# 9. Check imports
print_status "Checking import dependencies..."
if python -c "
import sys
sys.path.append('src')
try:
    from qwen_image_config import MODEL_CONFIG
    from qwen_generator import QwenImageGenerator
    print('✅ Core imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
" 2>&1 | tee logs/imports.log; then
    print_success "✅ Import check passed"
else
    print_error "❌ Import issues found"
fi

# 10. Hardware-specific checks
print_status "Running hardware optimization checks..."
python -c "
import torch
print(f'🔧 PyTorch version: {torch.__version__}')
print(f'🔧 CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'🔧 CUDA version: {torch.version.cuda}')
    print(f'🔧 GPU: {torch.cuda.get_device_name()}')
    print(f'🔧 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    print('⚠️  CUDA not available - GPU acceleration disabled')
" 2>&1 | tee logs/hardware.log

# Summary
echo ""
echo "=================================================="
echo "🏁 Code Quality Check Summary"
echo "=================================================="

# Count issues
total_issues=0

if [ -s logs/black.log ] && grep -q "would reformat" logs/black.log; then
    echo "❌ Black formatting issues found"
    ((total_issues++))
fi

if [ -s logs/isort.log ] && grep -q "Fixing" logs/isort.log; then
    echo "❌ Import sorting issues found"
    ((total_issues++))
fi

if [ -s logs/flake8.log ] && [ -s logs/flake8.log ]; then
    echo "❌ Flake8 linting issues found"
    ((total_issues++))
fi

if [ -s logs/mypy.log ] && grep -q "error:" logs/mypy.log; then
    echo "❌ MyPy type issues found"
    ((total_issues++))
fi

if [ $total_issues -eq 0 ]; then
    print_success "🎉 All code quality checks passed!"
    echo "📊 Logs saved to ./logs/ directory"
    echo "🚀 Your code is ready for deployment!"
else
    print_warning "⚠️  Found $total_issues issue(s) to fix"
    echo "📊 Check logs in ./logs/ directory for details"
    echo "🔧 Run quick fixes:"
    echo "   black src/ tests/ examples/ scripts/ launch.py"
    echo "   isort src/ tests/ examples/ scripts/ launch.py"
fi

echo "=================================================="
