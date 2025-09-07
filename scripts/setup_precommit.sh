#!/bin/bash
# Pre-commit setup and management script
# Sets up automated code quality checks before commits

set -e

echo "🔗 Setting up pre-commit hooks for automated code quality..."

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if in git repository
if [ ! -d ".git" ]; then
    print_warning "Not in a git repository. Initializing..."
    git init
    print_success "Git repository initialized"
fi

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_warning "Virtual environment not detected. Activating..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_warning "Virtual environment not found. Creating one..."
        python3 -m venv venv
        source venv/bin/activate
        print_success "Virtual environment created and activated"
    fi
fi

# Install pre-commit if not already installed
if ! command -v pre-commit &> /dev/null; then
    print_status "Installing pre-commit..."
    pip install pre-commit
    print_success "Pre-commit installed"
fi

# Install development dependencies
print_status "Installing development dependencies..."
pip install black isort flake8 mypy bandit ruff codespell autopep8 pylint

# Install pre-commit hooks
print_status "Installing pre-commit hooks..."
pre-commit install

# Install pre-push hooks (optional)
print_status "Installing pre-push hooks..."
pre-commit install --hook-type pre-push

# Run pre-commit on all files (first time setup)
print_status "Running pre-commit on all files (initial setup)..."
if pre-commit run --all-files; then
    print_success "✅ All pre-commit checks passed!"
else
    print_warning "⚠️  Some files were modified by pre-commit hooks"
    print_status "Review the changes and commit them"
fi

# Create git attributes file for better handling
cat > .gitattributes << EOF
# Handle line endings automatically for files detected as text
* text=auto

# Force specific files to have LF line endings
*.py text eol=lf
*.sh text eol=lf
*.md text eol=lf
*.yaml text eol=lf
*.yml text eol=lf
*.toml text eol=lf
*.json text eol=lf

# Binary files
*.png binary
*.jpg binary
*.jpeg binary
*.gif binary
*.ico binary
*.zip binary
*.tar.gz binary
*.pt binary
*.pth binary
*.bin binary
*.safetensors binary
EOF

print_success "✅ Pre-commit hooks setup complete!"
echo ""
echo "📋 What was configured:"
echo "  ✅ Black code formatter"
echo "  ✅ isort import sorter"
echo "  ✅ Ruff linter (fast)"
echo "  ✅ MyPy type checker"
echo "  ✅ Bandit security scanner"
echo "  ✅ General file format checks"
echo "  ✅ Spell checking"
echo ""
echo "🎯 Usage:"
echo "  • Hooks run automatically on 'git commit'"
echo "  • Manual run: 'pre-commit run --all-files'"
echo "  • Update hooks: 'pre-commit autoupdate'"
echo "  • Skip hooks: 'git commit --no-verify'"
echo ""
echo "🔧 Quick fixes:"
echo "  • Format code: 'black src/ tests/ examples/ scripts/ launch.py'"
echo "  • Sort imports: 'isort src/ tests/ examples/ scripts/ launch.py'"
echo "  • Auto-fix issues: 'ruff check --fix src/'"