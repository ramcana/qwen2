# Setup Script for Qwen-Image Generator
# PowerShell version for Windows users

Write-Host "🚀 Qwen-Image Generator Setup" -ForegroundColor Green
Write-Host "=============================" -ForegroundColor Green

# Navigate to project root
Set-Location -Path ".." -ErrorAction Stop

Write-Host "🔧 Checking system requirements..." -ForegroundColor Cyan

# Check if WSL is available
try {
    $wslVersion = wsl --list --verbose 2>$null
    Write-Host "✅ WSL is available" -ForegroundColor Green
}
catch {
    Write-Host "❌ WSL is not available. Please install WSL2 with Ubuntu." -ForegroundColor Red
    Write-Host "   Download from: https://aka.ms/wsl2kernel" -ForegroundColor Yellow
    exit 1
}

# Check if Ubuntu is installed
try {
    $ubuntuCheck = wsl -l 2>$null | Select-String "Ubuntu"
    if ($ubuntuCheck) {
        Write-Host "✅ Ubuntu is installed in WSL" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Ubuntu is not installed in WSL." -ForegroundColor Red
        Write-Host "   Please install Ubuntu from Microsoft Store." -ForegroundColor Yellow
        exit 1
    }
}
catch {
    Write-Host "⚠️ Could not check Ubuntu installation." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📋 Setup Options:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Full Setup (Recommended)" -ForegroundColor Yellow
Write-Host "   • Create Python virtual environment" -ForegroundColor Yellow
Write-Host "   • Install all dependencies" -ForegroundColor Yellow
Write-Host "   • Download models" -ForegroundColor Yellow
Write-Host "   • Run quick test" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. Quick Setup" -ForegroundColor Yellow
Write-Host "   • Create Python virtual environment" -ForegroundColor Yellow
Write-Host "   • Install all dependencies" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. Dependency Installation Only" -ForegroundColor Yellow
Write-Host "   • Install Python dependencies" -ForegroundColor Yellow
Write-Host ""
Write-Host "0. Exit" -ForegroundColor Red
Write-Host ""

do {
    $choice = Read-Host "🎯 Enter your choice (1/2/3/0)"
    
    switch ($choice) {
        "1" {
            Write-Host ""
            Write-Host "🚀 Starting Full Setup..." -ForegroundColor Green
            wsl -u ramji_t ./scripts/setup.sh
            break
        }
        "2" {
            Write-Host ""
            Write-Host "🚀 Starting Quick Setup..." -ForegroundColor Green
            wsl -u ramji_t ./scripts/setup.sh quick
            break
        }
        "3" {
            Write-Host ""
            Write-Host "🚀 Installing Dependencies Only..." -ForegroundColor Green
            wsl -u ramji_t ./scripts/setup.sh deps
            break
        }
        "0" {
            Write-Host "👋 Goodbye!" -ForegroundColor Green
            exit 0
        }
        default {
            Write-Host "❌ Invalid choice. Please enter 1, 2, 3, or 0." -ForegroundColor Red
        }
    }
} while ($true)

Write-Host ""
Write-Host "✅ Setup completed!" -ForegroundColor Green
Write-Host "💡 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Launch the UI: ./scripts/launch_ui.ps1" -ForegroundColor Yellow
Write-Host "   2. Or run directly: python launch.py" -ForegroundColor Yellow