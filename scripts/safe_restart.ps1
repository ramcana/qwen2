# Safe Restart Script for Qwen-Image Generator
# PowerShell version for Windows users

Write-Host "🔄 Qwen-Image Safe Restart" -ForegroundColor Green
Write-Host "==========================" -ForegroundColor Green
Write-Host "Clearing GPU memory and restarting application..." -ForegroundColor Cyan

# Navigate to project root
Set-Location -Path ".." -ErrorAction Stop

# Clear GPU memory and restart
Write-Host "🧹 Clearing GPU memory..." -ForegroundColor Cyan
wsl -u ramji_t python -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    print('✅ GPU memory cleared')
else:
    print('⚠️ CUDA not available')
"

# Restart application
Write-Host "🚀 Restarting application..." -ForegroundColor Green
wsl -u ramji_t ./scripts/launch_ui.sh