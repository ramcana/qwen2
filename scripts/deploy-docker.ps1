# Docker Deployment Script for Qwen2 Image Generation Application (PowerShell)
# Usage: .\scripts\deploy-docker.ps1 [dev|prod|staging] [options]

param(
    [Parameter(Position = 0)]
    [ValidateSet("dev", "prod", "staging")]
    [string]$Environment = "dev",
    
    [switch]$Build,
    [switch]$Pull,
    [switch]$Force,
    [switch]$Foreground,
    [switch]$Verbose,
    [switch]$Help
)

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to show usage
function Show-Usage {
    @"
Docker Deployment Script for Qwen2 Application (PowerShell)

Usage: .\scripts\deploy-docker.ps1 [ENVIRONMENT] [OPTIONS]

ENVIRONMENTS:
    dev         Development environment (default)
    prod        Production environment
    staging     Staging environment

OPTIONS:
    -Build          Build images before deployment
    -Pull           Pull latest images before deployment
    -Force          Force recreate containers
    -Foreground     Run in foreground (don't detach)
    -Verbose        Enable verbose output
    -Help           Show this help message

EXAMPLES:
    .\scripts\deploy-docker.ps1 dev -Build                    # Deploy development with fresh build
    .\scripts\deploy-docker.ps1 prod -Pull -Force           # Deploy production with latest images
    .\scripts\deploy-docker.ps1 staging -Build -Foreground  # Deploy staging and watch logs

"@
}

# Show help if requested
if ($Help) {
    Show-Usage
    exit 0
}

# Set compose file based on environment
switch ($Environment) {
    "dev" { $ComposeFile = "docker-compose.dev.yml" }
    "prod" { $ComposeFile = "docker-compose.prod.yml" }
    "staging" { $ComposeFile = "docker-compose.yml" }
    default {
        Write-Error "Invalid environment: $Environment"
        exit 1
    }
}

# Check if compose file exists
if (-not (Test-Path $ComposeFile)) {
    Write-Error "Compose file not found: $ComposeFile"
    exit 1
}

Write-Status "Deploying Qwen2 application in $Environment environment"
Write-Status "Using compose file: $ComposeFile"

# Set verbose mode
if ($Verbose) {
    $VerbosePreference = "Continue"
}

# Create necessary directories
Write-Status "Creating necessary directories..."
$Directories = @(
    "models", "cache", "generated_images", "uploads", "offload",
    "logs", "logs\api", "logs\traefik",
    "data", "data\redis", "data\prometheus", "data\grafana"
)

foreach ($Dir in $Directories) {
    if (-not (Test-Path $Dir)) {
        New-Item -ItemType Directory -Path $Dir -Force | Out-Null
        Write-Verbose "Created directory: $Dir"
    }
}

# Pull images if requested
if ($Pull) {
    Write-Status "Pulling latest images..."
    & docker-compose -f $ComposeFile pull
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to pull images"
        exit 1
    }
}

# Build images if requested
if ($Build) {
    Write-Status "Building images..."
    & docker-compose -f $ComposeFile build --no-cache
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build images"
        exit 1
    }
}

# Prepare docker-compose command
$ComposeArgs = @("-f", $ComposeFile, "up")

if ($Force) {
    $ComposeArgs += "--force-recreate"
}

if (-not $Foreground) {
    $ComposeArgs += "-d"
}

# Deploy the application
Write-Status "Starting services..."
& docker-compose @ComposeArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to start services"
    exit 1
}

if (-not $Foreground) {
    Write-Success "Services started successfully!"
    Write-Status "Checking service health..."
    
    # Wait a moment for services to start
    Start-Sleep -Seconds 5
    
    # Show service status
    & docker-compose -f $ComposeFile ps
    
    Write-Status "Application URLs:"
    switch ($Environment) {
        "dev" {
            Write-Host "  Frontend: http://localhost:3000"
            Write-Host "  API: http://localhost:8000"
            Write-Host "  Traefik Dashboard: http://localhost:8080"
        }
        { $_ -in "prod", "staging" } {
            Write-Host "  Application: https://your-domain.com"
            Write-Host "  Traefik Dashboard: https://traefik.your-domain.com"
        }
    }
    
    Write-Status "To view logs: docker-compose -f $ComposeFile logs -f"
    Write-Status "To stop services: docker-compose -f $ComposeFile down"
}
else {
    Write-Status "Running in foreground mode. Press Ctrl+C to stop."
}