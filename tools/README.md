# Development Tools

This directory contains diagnostic and utility tools for the Qwen2 Image Generator project.

## 🛠️ **Available Tools**

### 🔍 **Diagnostic Tools**

#### `test_device.py`
**Purpose**: Comprehensive CUDA and device diagnostics
**Usage**:
```bash
cd /home/ramji_t/projects/Qwen2
source venv/bin/activate
python tools/test_device.py
```

**What it checks**:
- ✅ CUDA availability and configuration
- ✅ GPU memory and specifications
- ✅ PyTorch tensor operations
- ✅ Generator device compatibility
- ✅ Model component initialization

**Expected output**:
```
🚀 Qwen2 Device Diagnostic Test
CUDA Setup: ✅ PASS
Model Setup: ✅ PASS
🎉 All tests passed! Device setup is correct.
```

#### `download_qwen_edit.py`
**Purpose**: Download and verify Qwen-Image-Edit model
**Usage**:
```bash
cd /home/ramji_t/projects/Qwen2
source venv/bin/activate
python tools/download_qwen_edit.py
```

**What it does**:
- 📥 Downloads Qwen-Image-Edit model (~20GB)
- 📊 Shows download progress and estimated time
- ✅ Verifies successful installation
- 🔧 Provides troubleshooting guidance
- 💾 Manages model cache efficiently

### 🚨 **Emergency Fixes**

#### `emergency_device_fix.py`
**Purpose**: Advanced device error recovery and system repair
**Usage**:
```bash
cd /home/ramji_t/projects/Qwen2
source venv/bin/activate
python tools/emergency_device_fix.py
```

**What it does**:
- 🔧 Deep CUDA context reset
- 🧹 Aggressive memory cleanup
- 🔄 Model component reinitialization
- 📊 Comprehensive device state analysis
- 🛡️ Emergency fallback configuration

## 📋 **When to Use Each Tool**

### Use `test_device.py` when:
- Setting up the project for the first time
- Verifying system compatibility
- Diagnosing basic device issues
- Regular system health checks

### Use `download_qwen_edit.py` when:
- Setting up enhanced features for the first time
- "Qwen-Image-Edit pipeline not loaded" error appears
- Want to enable image-to-image and inpainting features
- Need to re-download the model after cache issues

### Use `emergency_device_fix.py` when:
- Experiencing persistent device errors
- System becomes unresponsive during generation
- Memory leaks or CUDA out-of-memory errors
- After hardware/driver changes

## 🏃 **Quick Diagnostic Workflow**

1. **First**: Run `test_device.py` to identify issues
2. **If issues found**: Run `emergency_device_fix.py`
3. **Verify fix**: Run `test_device.py` again
4. **Start application**: Use `scripts/safe_restart.sh`

## 📝 **Tool Integration**

These tools are automatically integrated into:
- **Restart scripts**: `scripts/restart_ui.sh` runs device diagnostics
- **Quality gates**: Part of the automated testing pipeline
- **Development workflow**: VS Code tasks and launch configurations

## ⚙️ **Configuration**

Tools respect the same configuration as the main application:
- Virtual environment: `venv/`
- Python path: Auto-configured for project imports
- CUDA settings: Matches main application device preferences
- Logging: Outputs to console and `reports/` directory when applicable
