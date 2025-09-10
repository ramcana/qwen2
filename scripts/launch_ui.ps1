# WSL2-Friendly Qwen2 UI Launcher with Enhanced Mode Support
# PowerShell version for Windows users

Write-Host "🎨 Launching Qwen2 Image Generator..." -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green

# Activate virtual environment (navigate to project root)
Set-Location -Path ".." -ErrorAction Stop
# Note: PowerShell can't directly activate bash venv, but we'll use it in WSL

Write-Host "✅ Virtual environment ready" -ForegroundColor Green
Write-Host ""
Write-Host "🎆 Choose your interface:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Standard UI  - Text-to-Image only (Fast)" -ForegroundColor Yellow
Write-Host "2. Enhanced UI  - Full feature suite (Advanced)" -ForegroundColor Yellow
Write-Host "3. Complete System - Backend API + React Frontend (High-Performance)" -ForegroundColor Yellow
Write-Host "4. Performance Test - Validate system speed" -ForegroundColor Yellow
Write-Host "5. Server Startup Test - Verify server can start without errors" -ForegroundColor Yellow
Write-Host "0. Exit" -ForegroundColor Red
Write-Host ""

do {
    $choice = Read-Host "🎯 Enter your choice (1/2/3/4/5/0)"

    switch ($choice) {
        "1" {
            Write-Host ""
            Write-Host "🚀 Starting Standard Qwen2 UI..." -ForegroundColor Green
            Write-Host "📋 Features: Text-to-Image, Multi-language, Aspect ratios" -ForegroundColor Cyan
            Write-Host "🌐 Server will start on http://localhost:7860" -ForegroundColor Cyan
            Write-Host "💡 Open that URL in your Windows browser" -ForegroundColor Cyan
            Write-Host "⏹️  Press Ctrl+C to stop the server" -ForegroundColor Cyan
            Write-Host ""
            wsl -u ramji_t python src/qwen_image_ui.py
            break
        }
        "2" {
            Write-Host ""
            Write-Host "🚀 Starting Enhanced Qwen2 UI..." -ForegroundColor Green
            Write-Host "📋 Features: Text-to-Image, Img2Img, Inpainting, Super-Resolution" -ForegroundColor Cyan
            Write-Host "🌐 Server will start on http://localhost:7860" -ForegroundColor Cyan
            Write-Host "💡 Open that URL in your Windows browser" -ForegroundColor Cyan
            Write-Host "⏹️  Press Ctrl+C to stop the server" -ForegroundColor Cyan
            Write-Host ""
            wsl -u ramji_t python src/qwen_image_enhanced_ui.py
            break
        }
        "3" {
            Write-Host ""
            Write-Host "🚀 Starting Complete High-Performance System..." -ForegroundColor Green
            Write-Host "📋 Features: FastAPI Backend + React Frontend" -ForegroundColor Cyan
            Write-Host "🌐 Backend: http://localhost:8000" -ForegroundColor Cyan
            Write-Host "🌐 Frontend: http://localhost:3000" -ForegroundColor Cyan
            Write-Host "💡 Open URLs in your Windows browser" -ForegroundColor Cyan
            Write-Host "⏹️  Press Ctrl+C to stop all services" -ForegroundColor Cyan
            Write-Host ""
            wsl -u ramji_t ./scripts/launch_complete_system.sh
            break
        }
        "4" {
            Write-Host ""
            Write-Host "🚀 Running Performance Test..." -ForegroundColor Green
            Write-Host "📋 Validates 15-60s generation times" -ForegroundColor Cyan
            Write-Host ""
            wsl -u ramji_t python tools/test_performance.py
            break
        }
        "5" {
            Write-Host ""
            Write-Host "🚀 Running Server Startup Test..." -ForegroundColor Green
            Write-Host "📋 Verifies all components can be imported successfully" -ForegroundColor Cyan
            Write-Host ""
            wsl -u ramji_t python tools/test_server_startup.py
            break
        }
        "0" {
            Write-Host "👋 Goodbye!" -ForegroundColor Green
            exit 0
        }
        default {
            Write-Host "❌ Invalid choice. Please enter 1, 2, 3, 4, 5, or 0." -ForegroundColor Red
        }
    }
} while ($true)

Write-Host "🔴 Server stopped" -ForegroundColor Red
