# PowerShell Dockerfile validation script
# Checks syntax and best practices for Dockerfile.api

Write-Host "Validating Dockerfile.api..." -ForegroundColor Green

# Check if Dockerfile exists
if (-not (Test-Path "Dockerfile.api")) {
    Write-Host "❌ Dockerfile.api not found" -ForegroundColor Red
    exit 1
}

# Check if required files exist
Write-Host "Checking required files..."

$requiredFiles = @(
    "requirements-docker.txt",
    "docker-entrypoint.sh",
    "src/api_server.py",
    "configs/"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "✅ $file exists" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️  $file not found (may be created at runtime)" -ForegroundColor Yellow
    }
}

# Check Dockerfile content
Write-Host ""
Write-Host "Checking Dockerfile best practices..."

$dockerfileContent = Get-Content "Dockerfile.api" -Raw

# Check for multi-stage build
if ($dockerfileContent -match "FROM.*as.*builder" -and $dockerfileContent -match "FROM.*as.*runtime") {
    Write-Host "✅ Multi-stage build detected" -ForegroundColor Green
}
else {
    Write-Host "⚠️  Multi-stage build not detected" -ForegroundColor Yellow
}

# Check for non-root user
if ($dockerfileContent -match "USER.*appuser") {
    Write-Host "✅ Non-root user configured" -ForegroundColor Green
}
else {
    Write-Host "⚠️  Non-root user not configured" -ForegroundColor Yellow
}

# Check for health check
if ($dockerfileContent -match "HEALTHCHECK") {
    Write-Host "✅ Health check configured" -ForegroundColor Green
}
else {
    Write-Host "⚠️  Health check not configured" -ForegroundColor Yellow
}

# Check for proper COPY usage
if ($dockerfileContent -match "COPY --chown=") {
    Write-Host "✅ Proper file ownership in COPY commands" -ForegroundColor Green
}
else {
    Write-Host "⚠️  File ownership not set in COPY commands" -ForegroundColor Yellow
}

# Check for .dockerignore
if (Test-Path ".dockerignore") {
    Write-Host "✅ .dockerignore file exists" -ForegroundColor Green
}
else {
    Write-Host "⚠️  .dockerignore file not found" -ForegroundColor Yellow
}

# Check for Python 3.11
if ($dockerfileContent -match "python:3.11") {
    Write-Host "✅ Python 3.11 base image" -ForegroundColor Green
}
else {
    Write-Host "⚠️  Python 3.11 not detected" -ForegroundColor Yellow
}

# Check for CUDA support
if ($dockerfileContent -match "nvidia") {
    Write-Host "✅ NVIDIA/CUDA support configured" -ForegroundColor Green
}
else {
    Write-Host "⚠️  NVIDIA/CUDA support not detected" -ForegroundColor Yellow
}

# Check Docker availability
Write-Host ""
Write-Host "Checking Docker availability..."

try {
    $dockerVersion = docker --version 2>$null
    if ($dockerVersion) {
        Write-Host "✅ Docker is available: $dockerVersion" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️  Docker not available" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "⚠️  Docker not available" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Validation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To build the container:"
Write-Host "  .\build-docker-api.ps1 -Gpu" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the container:"
Write-Host "  docker run --gpus all -p 8000:8000 qwen-image-api:gpu" -ForegroundColor Cyan