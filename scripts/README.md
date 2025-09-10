# Scripts Directory

This directory contains all shell scripts and automation tools for the Qwen2 Image Generator project.

## 🚀 **Launch Scripts**

### `safe_restart.sh` ⭐ **RECOMMENDED**
**Purpose**: Safe restart with minimal device handling (prevents segfaults)
**Usage**: `./scripts/safe_restart.sh`
**Features**:
- ✅ Safe device management
- ✅ Light CUDA cleanup
- ✅ No aggressive memory manipulation
- ✅ WSL2 optimized

### `restart_ui.sh`
**Purpose**: Full restart with comprehensive device fixes
**Usage**: `./scripts/restart_ui.sh`
**Features**:
- 🔧 Comprehensive device diagnostics
- 🧹 Aggressive CUDA cleanup
- 📊 Device state monitoring
- 🛠️ Emergency fallback systems

### `launch_ui.sh`
**Purpose**: Basic UI launcher
**Usage**: `./scripts/launch_ui.sh`
**Features**:
- 🐍 Environment activation
- 🌐 Server startup
- 📝 WSL2 instructions

### `activate.sh`
**Purpose**: Environment activation with path setup
**Usage**: `source scripts/activate.sh`
**Features**:
- 🐍 Virtual environment activation
- 📂 Python path configuration
- 🔧 Environment variables setup

## 🛠️ **Development Scripts**

### `setup.sh`
**Purpose**: Complete project setup and installation
**Usage**: `./scripts/setup.sh`
**Features**:
- 📦 Dependency installation
- 🐍 Virtual environment creation
- 🔧 CUDA configuration
- 📋 Pre-commit hooks setup

### `setup_precommit.sh`
**Purpose**: Install and configure pre-commit hooks
**Usage**: `./scripts/setup_precommit.sh`
**Features**:
- 🔒 Code quality enforcement
- 🧹 Auto-formatting setup
- 🔍 Linting configuration
- 🛡️ Security scanning

## 🔍 **Quality Assurance Scripts**

### `lint.sh`
**Purpose**: Run all linting and formatting tools
**Usage**: `./scripts/lint.sh`
**Tools**:
- 🖤 Black (formatting)
- 🦀 Ruff (linting)
- 📏 Flake8 (style)
- 🔤 isort (imports)
- 🔍 MyPy (type checking)

### `quality_gate.sh`
**Purpose**: Comprehensive quality gate for CI/CD
**Usage**: `./scripts/quality_gate.sh`
**Checks**:
- ✅ Code formatting
- ✅ Linting compliance
- ✅ Type checking
- ✅ Security scanning
- ✅ Test execution

### `error_detection.py`
**Purpose**: Automated error detection and reporting
**Usage**: `python scripts/error_detection.py`
**Features**:
- 🔍 Code smell detection
- 📊 Error pattern analysis
- 📋 Automated reporting
- 🔧 Fix suggestions

### `fix_common_issues.py`
**Purpose**: Automated fixes for common code issues
**Usage**: `python scripts/fix_common_issues.py`
**Fixes**:
- 🔧 Import organization
- 📝 Documentation updates
- 🧹 Code cleanup
- ⚡ Performance optimizations

## 📋 **Script Usage Patterns**

### **First Time Setup**
```bash
cd /home/ramji_t/projects/Qwen2
chmod +x scripts/*.sh
./scripts/setup.sh
./scripts/setup_precommit.sh
```

### **Daily Development**
```bash
# Activate environment
source scripts/activate.sh

# Run quality checks
./scripts/lint.sh

# Start application (safe mode)
./scripts/safe_restart.sh
```

### **Before Committing**
```bash
# Run full quality gate
./scripts/quality_gate.sh

# Fix any issues
python scripts/fix_common_issues.py

# Verify fixes
./scripts/lint.sh
```

### **Troubleshooting**
```bash
# Full diagnostic restart
./scripts/restart_ui.sh

# Emergency fixes if needed
python tools/emergency_device_fix.py

# Test system health
python tools/test_device.py
```

## 🔧 **Script Configuration**

All scripts use consistent configuration:
- **Virtual Environment**: `venv/`
- **Python Path**: Auto-configured for project imports
- **Quality Tools**: Configured via `pyproject.toml`, `.flake8`, `mypy.ini`
- **Pre-commit**: Configured via `.pre-commit-config.yaml`

## 🚀 **Integration**

Scripts integrate with:
- **IDE**: VS Code tasks and launch configurations
- **CI/CD**: GitHub Actions workflows
- **Git**: Pre-commit hooks
- **Development**: Daily workflow automation
