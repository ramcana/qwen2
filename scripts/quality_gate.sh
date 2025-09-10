#!/bin/bash
# Quality Gate Script - Final check before deployment
# Comprehensive validation for Qwen-Image project

set -e

echo "🛡️ Running Quality Gate Checks..."
echo "=================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
PASS_COUNT=0
FAIL_COUNT=0
TOTAL_CHECKS=0

print_result() {
    local status=$1
    local message=$2
    ((TOTAL_CHECKS++))

    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✅ PASS${NC} - $message"
        ((PASS_COUNT++))
    else
        echo -e "${RED}❌ FAIL${NC} - $message"
        ((FAIL_COUNT++))
    fi
}

print_section() {
    echo -e "\n${BLUE}📋 $1${NC}"
    echo "----------------------------------------"
}

# Check 1: File Structure
print_section "File Structure Validation"

required_files=(
    "src/qwen_image_ui.py"
    "src/qwen_generator.py"
    "src/qwen_image_config.py"
    "requirements.txt"
    "README.md"
    "launch.py"
    "pyproject.toml"
    ".flake8"
    ".pre-commit-config.yaml"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_result "PASS" "Required file exists: $file"
    else
        print_result "FAIL" "Missing required file: $file"
    fi
done

# Check 2: Python Syntax
print_section "Python Syntax Validation"

if python -m py_compile src/*.py tests/*.py examples/*.py launch.py 2>/dev/null; then
    print_result "PASS" "All Python files compile successfully"
else
    print_result "FAIL" "Python syntax errors found"
fi

# Check 3: Import Validation
print_section "Import Validation"

export PYTHONPATH=$PYTHONPATH:src
if python -c "
from qwen_image_config import MODEL_CONFIG
from qwen_generator import QwenImageGenerator
print('Core imports successful')
" 2>/dev/null; then
    print_result "PASS" "Core module imports work"
else
    print_result "FAIL" "Core module import errors"
fi

# Check 4: Code Formatting
print_section "Code Formatting Check"

if black --check src/ tests/ examples/ scripts/ launch.py >/dev/null 2>&1; then
    print_result "PASS" "Code formatting (Black) compliant"
else
    print_result "FAIL" "Code formatting issues found"
fi

if isort --check-only src/ tests/ examples/ scripts/ launch.py >/dev/null 2>&1; then
    print_result "PASS" "Import sorting (isort) compliant"
else
    print_result "FAIL" "Import sorting issues found"
fi

# Check 5: Linting
print_section "Linting Validation"

if flake8 src/ tests/ examples/ scripts/ launch.py >/dev/null 2>&1; then
    print_result "PASS" "Flake8 linting passed"
else
    print_result "FAIL" "Flake8 linting issues found"
fi

if command -v ruff >/dev/null 2>&1; then
    if ruff check src/ tests/ examples/ scripts/ launch.py >/dev/null 2>&1; then
        print_result "PASS" "Ruff linting passed"
    else
        print_result "FAIL" "Ruff linting issues found"
    fi
else
    print_result "FAIL" "Ruff not installed"
fi

# Check 6: Type Checking
print_section "Type Checking"

if mypy src/ >/dev/null 2>&1; then
    print_result "PASS" "MyPy type checking passed"
else
    print_result "FAIL" "MyPy type checking issues found"
fi

# Check 7: Security Scan
print_section "Security Validation"

if command -v bandit >/dev/null 2>&1; then
    if bandit -r src/ >/dev/null 2>&1; then
        print_result "PASS" "Security scan (Bandit) passed"
    else
        print_result "FAIL" "Security issues found"
    fi
else
    print_result "FAIL" "Bandit not installed"
fi

# Check 8: Documentation
print_section "Documentation Validation"

if [ -s "README.md" ]; then
    print_result "PASS" "README.md exists and is not empty"
else
    print_result "FAIL" "README.md missing or empty"
fi

if grep -q "## " README.md; then
    print_result "PASS" "README.md has proper structure"
else
    print_result "FAIL" "README.md lacks proper section headers"
fi

# Check 9: Configuration Files
print_section "Configuration Validation"

config_files=(".flake8" "pyproject.toml" "mypy.ini")
for config in "${config_files[@]}"; do
    if [ -f "$config" ] && [ -s "$config" ]; then
        print_result "PASS" "Configuration file valid: $config"
    else
        print_result "FAIL" "Configuration file missing/empty: $config"
    fi
done

# Check 10: Hardware Detection
print_section "Hardware Detection"

if python -c "
import torch
if torch.cuda.is_available():
    print('CUDA available')
else:
    print('CPU mode')
" >/dev/null 2>&1; then
    print_result "PASS" "Hardware detection working"
else
    print_result "FAIL" "Hardware detection failed"
fi

# Check 11: Git Setup
print_section "Git Repository Validation"

if [ -d ".git" ]; then
    print_result "PASS" "Git repository initialized"
else
    print_result "FAIL" "Git repository not initialized"
fi

if [ -f ".gitignore" ]; then
    print_result "PASS" ".gitignore file exists"
else
    print_result "FAIL" ".gitignore file missing"
fi

# Check 12: Pre-commit Setup
print_section "Pre-commit Validation"

if [ -f ".pre-commit-config.yaml" ]; then
    print_result "PASS" "Pre-commit configuration exists"
else
    print_result "FAIL" "Pre-commit configuration missing"
fi

if command -v pre-commit >/dev/null 2>&1; then
    if pre-commit run --all-files >/dev/null 2>&1; then
        print_result "PASS" "Pre-commit hooks pass"
    else
        print_result "FAIL" "Pre-commit hooks failed"
    fi
else
    print_result "FAIL" "Pre-commit not installed"
fi

# Final Results
echo ""
echo "=================================="
echo "🏁 Quality Gate Results"
echo "=================================="
echo -e "Total Checks: ${BLUE}$TOTAL_CHECKS${NC}"
echo -e "Passed: ${GREEN}$PASS_COUNT${NC}"
echo -e "Failed: ${RED}$FAIL_COUNT${NC}"

PASS_PERCENTAGE=$((PASS_COUNT * 100 / TOTAL_CHECKS))
echo -e "Success Rate: ${BLUE}$PASS_PERCENTAGE%${NC}"

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "\n${GREEN}🎉 ALL QUALITY CHECKS PASSED!${NC}"
    echo -e "${GREEN}🚀 Project is ready for deployment!${NC}"
    exit 0
elif [ $PASS_PERCENTAGE -ge 80 ]; then
    echo -e "\n${YELLOW}⚠️  Quality gate passed with warnings${NC}"
    echo -e "${YELLOW}📝 $FAIL_COUNT issue(s) should be addressed${NC}"
    exit 0
else
    echo -e "\n${RED}❌ QUALITY GATE FAILED${NC}"
    echo -e "${RED}🔧 $FAIL_COUNT critical issue(s) must be fixed${NC}"
    echo ""
    echo "Quick fixes:"
    echo "• Format code: black src/ tests/ examples/ scripts/ launch.py"
    echo "• Sort imports: isort src/ tests/ examples/ scripts/ launch.py"
    echo "• Fix issues: python scripts/fix_common_issues.py"
    echo "• Run linting: ./scripts/lint.sh"
    exit 1
fi
