# 🚀 Qwen-Image Enhanced Launcher Guide

## Quick Start Options

Now you have multiple easy ways to start the Qwen-Image Generator with your preferred interface:

### 🎯 **Interactive Launcher (Recommended)**
```bash
# Activate environment and choose your interface
source venv/bin/activate
python launch.py
```
**Features:** Interactive menu to choose between Standard or Enhanced UI

### ⚡ **Direct Launch Options**

#### Standard UI (Original)
```bash
# Using Python
python launch.py --mode standard

# Using shell script
./scripts/launch_ui.sh  # Choose option 1
```

#### Enhanced UI (New Features)
```bash
# Using Python
python launch.py --mode enhanced

# Using shell script
./scripts/launch_ui.sh  # Choose option 2

# Direct enhanced launch
./scripts/restart_enhanced.sh
```

### 🛡️ **Safe Restart Options**

#### Safe Restart with Choice
```bash
./scripts/safe_restart.sh
```
**Features:** Kills existing processes, cleans CUDA cache, then lets you choose UI mode

#### Direct Enhanced Safe Restart
```bash
./scripts/restart_enhanced.sh
```
**Features:** Direct launch of enhanced UI with safe cleanup

## 🎨 **Interface Comparison**

| Feature | Standard UI | Enhanced UI |
|---------|-------------|-------------|
| **Text-to-Image** | ✅ Qwen-Image | ✅ Qwen-Image |
| **Image-to-Image** | ❌ | ✅ SD 2.1 |
| **Inpainting** | ❌ | ✅ Interactive Mask |
| **Super-Resolution** | ❌ | ✅ 2x-4x Scale |
| **Performance** | Faster | Slightly Slower |
| **Memory Usage** | Lower | Higher |
| **Features** | Basic | Advanced |

## 📋 **Launch Script Features**

### `launch.py` - Main Launcher
- **Interactive Mode:** Choose UI at runtime
- **Command Line:** `--mode standard|enhanced|interactive`
- **Auto-detection:** Handles missing dependencies gracefully

### `scripts/launch_ui.sh` - Shell Launcher
- **Menu-driven:** Choose from 4 options
- **WSL2 friendly:** Optimized for Windows browser access
- **Environment handling:** Auto-activates virtual environment

### `scripts/safe_restart.sh` - Safe Restart
- **Process cleanup:** Kills existing UI instances
- **CUDA cleanup:** Clears GPU memory safely
- **Choice menu:** Select UI mode after cleanup

### `scripts/restart_enhanced.sh` - Direct Enhanced
- **Quick launch:** Direct enhanced UI startup
- **Full cleanup:** Process + CUDA memory cleanup
- **Feature preview:** Shows available enhanced features

## 🔧 **Troubleshooting**

### Import Errors
```bash
# Install missing dependencies
source venv/bin/activate
pip install -r requirements.txt
```

### GPU Memory Issues
```bash
# Use safe restart to clear CUDA cache
./scripts/safe_restart.sh
```

### Port Already in Use
```bash
# Kill existing processes
pkill -f "qwen_image"
# Wait a moment, then restart
./scripts/launch_ui.sh
```

## 💡 **Pro Tips**

1. **First Time Users:** Start with `python launch.py` for interactive selection
2. **Regular Users:** Use `./scripts/launch_ui.sh` for quick menu access
3. **Enhanced Features:** Use `./scripts/restart_enhanced.sh` for direct advanced UI
4. **After Changes:** Use `./scripts/safe_restart.sh` to ensure clean startup

## 🌐 **Access Points**

All launch methods start the server on: **http://localhost:7860**

- Open this URL in your Windows browser (WSL2 users)
- Server runs until you press `Ctrl+C` in terminal
- Generated images saved to `./generated_images/`

---

**Choose your preferred launch method and start creating amazing images! 🎨**
