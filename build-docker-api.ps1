# PowerShell build script for Qwen-Image API Docker container
# Supports both CPU and GPU builds with optimization

param(
    [switch]$Gpu,
    [switch]$Cpu,
    [string]$Tag = "latest",
    [string]$Name = "qwen-image-api",
    [string]$Platform = "",
    [switch]$NoCache,
    [switch]$Verbose,
    [switch]$Help
)

if ($Help) {
    Write-Host "Usage: .\build-docker-api.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Gpu               Build with GPU support (default)"
    Write-Host "  -Cpu               Build with CPU-only support"
    Write-Host "  -Tag TAG           Set image tag (default: latest)"
    Write-Host "  -Name NAME         Set image name (default: qwen-image-api)"
    Write-Host "  -Platform ARCH     Set target platform (e.g., linux/amd64)"
    Write-Host "  -NoCache           Build without using cache"
    Write-Host "  -Verbose           Show verbose build output"
    Write-Host "  -Help              Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\build-docker-api.ps1 -Gpu -Tag v1.0"
    Write-Host "  .\build-docker-api.ps1 -Cpu -NoCache"
    Write-Host "  .\build-docker-api.ps1 -Platform linux/amd64 -Verbose"
    exit 0
}

# Build arguments
$BuildArgs = @()

# Set GPU/CPU mode
if ($Cpu) {
    $BuildArgs += "--build-arg", "ENABLE_GPU=false"
    if ($Tag -eq "latest") { $Tag = "cpu" }
}
else {
    $BuildArgs += "--build-arg", "ENABLE_GPU=true"
    if ($Tag -eq "latest") { $Tag = "gpu" }
}

# Platform argument
$PlatformArgs = @()
if ($Platform) {
    $PlatformArgs += "--platform", $Platform
}

# Cache argument
$CacheArgs = @()
if ($NoCache) {
    $CacheArgs += "--no-cache"
}

# Verbose argument
$VerboseArgs = @()
if ($Verbose) {
    $VerboseArgs += "--progress=plain"
}

# Build information
Write-Host "Building Qwen-Image API Docker container..." -ForegroundColor Green
Write-Host "Image name: $Name`:$Tag"
Write-Host "Build args: $($BuildArgs -join ' ')"
Write-Host "Platform: $(if ($Platform) { $Platform } else { 'default' })"
Write-Host "Cache: $(if ($NoCache) { 'disabled' } else { 'enabled' })"
Write-Host ""

# Check if Dockerfile exists
if (-not (Test-Path "Dockerfile.api")) {
    Write-Error "Dockerfile.api not found in current directory"
    exit 1
}

# Check if requirements file exists
if (-not (Test-Path "requirements-docker.txt")) {
    Write-Warning "requirements-docker.txt not found, using requirements.txt"
    if (-not (Test-Path "requirements.txt")) {
        Write-Error "No requirements file found"
        exit 1
    }
}

# Build the Docker image
Write-Host "Starting Docker build..." -ForegroundColor Yellow

$DockerArgs = @(
    "build"
    $PlatformArgs
    $CacheArgs
    $VerboseArgs
    $BuildArgs
    "-f", "Dockerfile.api"
    "-t", "$Name`:$Tag"
    "."
) | Where-Object { $_ -ne "" }

try {
    & docker @DockerArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Build completed successfully!" -ForegroundColor Green
        Write-Host "Image: $Name`:$Tag"
        Write-Host ""
        Write-Host "To run the container:"
        if ($Tag -like "*gpu*") {
            Write-Host "  docker run --gpus all -p 8000:8000 $Name`:$Tag"
        }
        else {
            Write-Host "  docker run -p 8000:8000 $Name`:$Tag"
        }
        Write-Host ""
        Write-Host "To run with volumes:"
        Write-Host "  docker run --gpus all -p 8000:8000 \"
        Write-Host "    -v `$(pwd)/models:/app/models \"
        Write-Host "    -v `$(pwd)/cache:/app/cache \"
        Write-Host "    -v `$(pwd)/generated_images:/app/generated_images \"
        Write-Host "    $Name`:$Tag"
    }
    else {
        Write-Error "Build failed!"
        exit 1
    }
}
catch {
    Write-Error "Docker build failed: $_"
    exit 1
}