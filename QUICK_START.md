# Quick Reference - Qwen2 Image Generator

## 🚀 Quick Start Commands

### First Time Setup

```bash
cd Qwen2
chmod +x scripts/*.sh
./scripts/setup.sh
source scripts/activate.sh
```

### Daily Usage

```bash
# Recommended: Safe start (prevents segfaults)
./scripts/safe_restart.sh

# Alternative: Full diagnostic start  
./scripts/restart_ui.sh

# Just activate environment
source scripts/activate.sh
```

### Troubleshooting

```bash
# Test system health
python tools/test_device.py

# Emergency fixes
python tools/emergency_device_fix.py

# Code quality check
./scripts/lint.sh
```

## 📁 Important Directories

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `src/` | Main application | `qwen_image_ui.py`, `qwen_generator.py` |
| `scripts/` | All automation | `safe_restart.sh`, `setup.sh` |
| `docs/` | Documentation | Device fixes, UI guides |
| `tools/` | Diagnostics | `test_device.py` |
| `reports/` | Generated logs | Error reports |

## 🔧 Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| Segmentation fault | Use `./scripts/safe_restart.sh` |
| CUDA device errors | Run `python tools/test_device.py` |
| UI won't load | Check `docs/UI_ACCESS_GUIDE.md` |
| Import errors | Run `source scripts/activate.sh` |

## 🌐 Access Points

- **UI**: <http://localhost:7860>
- **Documentation**: `docs/README.md`
- **Scripts Help**: `scripts/README.md`
- **Tools Help**: `tools/README.md`

## ⚡ Performance Tips

- **Best launcher**: `./scripts/safe_restart.sh`
- **Device check**: `python tools/test_device.py`
- **Quality check**: `./scripts/lint.sh`
- **Emergency fix**: `python tools/emergency_device_fix.py`
