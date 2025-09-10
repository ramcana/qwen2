# 🛡️ Code Quality System for Qwen-Image Project

## Overview

This document describes the comprehensive code quality system implemented for the Qwen-Image text-to-image generation project. The system ensures high code quality, maintainability, and reliability through automated linting, formatting, type checking, and error detection.

## 🎯 System Components

### 1. **Linting & Formatting Tools**
- **Black**: Code formatter with 88-character line length
- **isort**: Import sorting with Black profile compatibility
- **Flake8**: Traditional Python linting
- **Ruff**: Modern, fast linting (Rust-based)
- **MyPy**: Static type checking
- **Bandit**: Security vulnerability scanning

### 2. **Pre-commit Hooks**
- Automatic code quality checks before each commit
- Prevents bad code from entering the repository
- Includes formatting, linting, security, and spell checking

### 3. **Automated Error Detection**
- Comprehensive project health checking
- Hardware compatibility validation
- Import dependency verification
- Configuration file validation

### 4. **IDE Integration**
- VS Code configuration with proper Python settings
- Debug configurations for different scenarios
- Task runners for common operations
- Extension recommendations

### 5. **CI/CD Quality Gates**
- GitHub Actions workflows for continuous integration
- Multi-Python version testing
- Documentation validation
- Hardware simulation testing

## 📁 Configuration Files

### Core Configuration
```
pyproject.toml          # Main configuration for tools
.flake8                 # Flake8 specific settings
mypy.ini               # MyPy type checking config
.pre-commit-config.yaml # Pre-commit hooks setup
.cspell.json           # Spell checking dictionary
```

### VS Code Integration
```
.vscode/
├── settings.json      # IDE settings and tool configuration
├── launch.json        # Debug configurations
├── tasks.json         # Task runners
└── extensions.json    # Recommended extensions
```

### CI/CD
```
.github/workflows/
└── code-quality.yml   # GitHub Actions workflow
```

## 🚀 Usage Guide

### Quick Setup
```bash
# 1. Install development tools
python scripts/fix_common_issues.py --install-tools

# 2. Setup pre-commit hooks
./scripts/setup_precommit.sh

# 3. Run initial quality check
./scripts/lint.sh
```

### Daily Development Workflow

#### 1. **Before Coding**
```bash
# Check current code quality
./scripts/error_detection.py
```

#### 2. **During Development**
- VS Code will automatically format on save
- Real-time linting feedback in editor
- Type hints and error detection

#### 3. **Before Committing**
```bash
# Auto-fix common issues
python scripts/fix_common_issues.py

# Comprehensive check
./scripts/lint.sh

# Quality gate validation
./scripts/quality_gate.sh
```

#### 4. **Git Workflow**
```bash
# Pre-commit hooks run automatically
git add .
git commit -m "Your commit message"
```

## 🔧 Available Scripts

### Core Scripts
| Script | Purpose | Usage |
|--------|---------|--------|
| `scripts/lint.sh` | Comprehensive linting | `./scripts/lint.sh` |
| `scripts/error_detection.py` | Project health check | `python scripts/error_detection.py` |
| `scripts/fix_common_issues.py` | Auto-fix problems | `python scripts/fix_common_issues.py` |
| `scripts/quality_gate.sh` | Final validation | `./scripts/quality_gate.sh` |
| `scripts/setup_precommit.sh` | Setup pre-commit | `./scripts/setup_precommit.sh` |

### Manual Commands
```bash
# Format code
black src/ tests/ examples/ scripts/ launch.py
isort src/ tests/ examples/ scripts/ launch.py

# Run linting
flake8 src/ tests/ examples/ scripts/ launch.py
ruff check src/ tests/ examples/ scripts/ launch.py

# Type checking
mypy src/

# Security scan
bandit -r src/

# Pre-commit (manual run)
pre-commit run --all-files
```

## 📊 Quality Metrics

### Code Quality Targets
- **Code Coverage**: >80%
- **Type Coverage**: >70%
- **Linting Score**: Zero critical issues
- **Security Score**: Zero high-severity issues
- **Documentation**: All public APIs documented

### Quality Gates
1. **Syntax Validation**: All Python files must compile
2. **Import Validation**: Core modules must import successfully
3. **Formatting**: Code must pass Black and isort checks
4. **Linting**: Must pass Flake8 and Ruff without errors
5. **Type Checking**: MyPy validation (warnings allowed)
6. **Security**: Bandit scan with no high-severity issues
7. **Documentation**: README and docstrings present

## 🎨 VS Code Integration

### Key Features
- **Auto-formatting**: Code formats on save
- **Real-time linting**: Immediate feedback on code issues
- **Type checking**: IntelliSense with type hints
- **Debug configurations**: Multiple debug scenarios
- **Task runners**: One-click access to common operations

### Recommended Extensions
- Python (Microsoft)
- Pylance (Microsoft)
- Black Formatter
- isort
- Ruff
- GitLens
- Markdown All in One

### Debug Configurations
1. **Launch Qwen-Image UI**: Debug the main application
2. **Debug Qwen Generator**: Debug core generation logic
3. **Run Quick Test**: Debug test scenarios
4. **Debug with GPU Profiling**: GPU-specific debugging

## 🔄 CI/CD Integration

### GitHub Actions Workflow
- **Multi-Python Testing**: Tests on Python 3.8-3.11
- **Code Quality Checks**: All linting tools run automatically
- **Documentation Validation**: Ensures docs are up-to-date
- **Hardware Simulation**: Tests CPU-only mode
- **Artifact Collection**: Saves reports for review

### Workflow Triggers
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

## 🛠️ Hardware-Specific Considerations

### RTX 4080 Optimizations
- CUDA availability checking
- VRAM usage monitoring
- GPU-specific debugging configurations
- Performance profiling setup

### WSL2 Environment
- Line ending handling (LF vs CRLF)
- Path resolution for cross-platform compatibility
- Virtual environment detection

## 📈 Monitoring & Reporting

### Log Files
All tools generate logs in the `logs/` directory:
```
logs/
├── black.log           # Formatting results
├── isort.log          # Import sorting results
├── flake8.log         # Linting results
├── mypy.log           # Type checking results
├── bandit.log         # Security scan results
├── ruff.log           # Modern linting results
└── error_detection_report.json  # Comprehensive health report
```

### Reports
- **Error Detection Report**: JSON format with categorized issues
- **Bandit Security Report**: JSON format with security findings
- **Quality Gate Summary**: Pass/fail status with metrics

## 🔧 Troubleshooting

### Common Issues

#### 1. **Pre-commit hooks failing**
```bash
# Update hooks
pre-commit autoupdate

# Clear cache and reinstall
pre-commit clean
pre-commit install
```

#### 2. **Import errors**
```bash
# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:src

# Or use VS Code task runner
```

#### 3. **Tool not found errors**
```bash
# Install missing tools
python scripts/fix_common_issues.py --install-tools
```

#### 4. **Line ending issues (Windows/WSL2)**
```bash
# Fix automatically
python scripts/fix_common_issues.py
```

### Performance Tips
- Use Ruff instead of Flake8 for faster linting
- Enable parallel execution in pre-commit
- Use caching for large projects
- Skip heavy checks during development (manual override)

## 📚 Best Practices

### Code Style
- Follow PEP 8 with 88-character line length
- Use type hints for all public APIs
- Document all classes and functions
- Prefer composition over inheritance

### Git Workflow
- Make small, focused commits
- Write descriptive commit messages
- Use pre-commit hooks consistently
- Fix quality issues before pushing

### Development Environment
- Use virtual environments
- Keep dependencies up-to-date
- Run quality checks regularly
- Use IDE integration features

## 🎉 Success Metrics

A successful code quality system implementation provides:

✅ **Zero Build Failures**: All CI/CD pipelines pass consistently
✅ **Fast Feedback**: Issues caught early in development
✅ **Consistent Style**: Uniform code formatting across the project
✅ **Type Safety**: Reduced runtime errors through static analysis
✅ **Security Assurance**: Automated vulnerability detection
✅ **Developer Productivity**: IDE integration and automation reduce manual work
✅ **Maintainability**: Clean, well-documented, and tested code

This system ensures that the Qwen-Image project maintains high quality standards while supporting rapid development and collaboration.
