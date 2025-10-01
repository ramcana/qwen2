# =============================================================================
# Docker Deployment Script for Qwen2 Image Generation (PowerShell)
# =============================================================================
# This script provides easy deployment commands for different environments on Windows
# =============================================================================

param(
    [Parameter(Position = 0)]
    [string]$Command = "help",
    
    [Parameter(Position = 1)]
    [string]$Environment = "development",
    
    [Parameter(Position = 2)]
    [string]$Service = "",
    
    [Parameter(Position = 3)]
    [string]$Profiles = ""
)

# Script configuration
$ProjectName = "qwen2"

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

# Function to check prerequisites
function Test-Prerequisites {
    Write-Status "Checking prerequisites..."
    
    # Check if Docker is installed and running
    try {
        $dockerVersion = docker --version
        Write-Success "Docker found: $dockerVersion"
    }
    catch {
        Write-Error "Docker is not installed or not in PATH. Please install Docker Desktop."
        exit 1
    }
    
    try {
        docker info | Out-Null
        Write-Success "Docker is running"
    }
    catch {
        Write-Error "Docker is not running. Please start Docker Desktop."
        exit 1
    }
    
    # Check if Docker Compose is available
    try {
        $composeVersion = docker-compose --version
        Write-Success "Docker Compose found: $composeVersion"
    }
    catch {
        try {
            $composeVersion = docker compose version
            Write-Success "Docker Compose (plugin) found: $composeVersion"
        }
        catch {
            Write-Error "Docker Compose is not available. Please install Docker Compose."
            exit 1
        }
    }
    
    # Check for NVIDIA Docker (for GPU support)
    try {
        $dockerInfo = docker info
        if ($dockerInfo -match "nvidia") {
            Write-Success "NVIDIA Docker runtime detected - GPU support available"
        }
        else {
            Write-Warning "NVIDIA Docker runtime not detected - GPU support may not be available"
        }
    }
    catch {
        Write-Warning "Could not check for NVIDIA Docker runtime"
    }
    
    Write-Success "Prerequisites check completed"
}

# Function to setup environment
function Initialize-Environment {
    Write-Status "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if (-not (Test-Path ".env")) {
        Write-Status "Creating .env file from template..."
        Copy-Item ".env.example" ".env"
        Write-Warning "Please edit .env file to customize your configuration"
    }
    
    # Create necessary directories
    Write-Status "Creating necessary directories..."
    $directories = @(
        "models", "cache", "generated_images", "uploads", "offload",
        "logs\api", "logs\traefik",
        "data\redis", "data\prometheus", "data\grafana",
        "config\redis", "config\prometheus", 
        "config\grafana\dashboards", "config\grafana\datasources",
        "ssl"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    # Create acme.json file with proper permissions
    if (-not (Test-Path "acme.json")) {
        New-Item -ItemType File -Path "acme.json" | Out-Null
    }
    
    # Create external network if it doesn't exist
    try {
        docker network create traefik-public 2>$null
        Write-Success "Created traefik-public network"
    }
    catch {
        Write-Status "traefik-public network already exists or created"
    }
    
    Write-Success "Environment setup completed"
}

# Function to build images
function Build-Images {
    param([string]$Env = "development")
    
    Write-Status "Building Docker images for $Env environment..."
    
    switch ($Env) {
        { $_ -in @("development", "dev") } {
            docker-compose -f docker-compose.yml -f docker-compose.dev.yml build --parallel
        }
        { $_ -in @("production", "prod") } {
            docker-compose -f docker-compose.yml -f docker-compose.prod.yml build --parallel
        }
        default {
            docker-compose build --parallel
        }
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Docker images built successfully"
    }
    else {
        Write-Error "Failed to build Docker images"
        exit 1
    }
}

# Function to start services
function Start-Services {
    param(
        [string]$Env = "development",
        [string]$ServiceProfiles = ""
    )
    
    Write-Status "Starting services for $Env environment..."
    
    $composeArgs = @()
    
    switch ($Env) {
        { $_ -in @("development", "dev") } {
            $composeArgs += @("-f", "docker-compose.yml", "-f", "docker-compose.dev.yml")
        }
        { $_ -in @("production", "prod") } {
            $composeArgs += @("-f", "docker-compose.yml", "-f", "docker-compose.prod.yml")
        }
        default {
            # Use base compose file only
        }
    }
    
    if ($ServiceProfiles) {
        $composeArgs += @("--profile", $ServiceProfiles)
    }
    
    $composeArgs += @("up", "-d")
    
    & docker-compose @composeArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Services started successfully"
        Show-Status
    }
    else {
        Write-Error "Failed to start services"
        exit 1
    }
}

# Function to stop services
function Stop-Services {
    param([string]$Env = "development")
    
    Write-Status "Stopping services..."
    
    switch ($Env) {
        { $_ -in @("development", "dev") } {
            docker-compose -f docker-compose.yml -f docker-compose.dev.yml down
        }
        { $_ -in @("production", "prod") } {
            docker-compose -f docker-compose.yml -f docker-compose.prod.yml down
        }
        default {
            docker-compose down
        }
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Services stopped successfully"
    }
    else {
        Write-Error "Failed to stop services"
    }
}

# Function to show service status
function Show-Status {
    Write-Status "Service Status:"
    docker-compose ps
    
    Write-Status "Available URLs:"
    Write-Host "  Frontend: http://qwen.localhost" -ForegroundColor Cyan
    Write-Host "  API: http://api.localhost" -ForegroundColor Cyan
    Write-Host "  Traefik Dashboard: http://traefik.localhost:8080" -ForegroundColor Cyan
    
    $services = docker-compose ps --services
    if ($services -contains "redis") {
        Write-Host "  Redis: localhost:6379" -ForegroundColor Cyan
    }
    
    if ($services -contains "grafana") {
        Write-Host "  Grafana: http://monitoring.localhost" -ForegroundColor Cyan
    }
}

# Function to show logs
function Show-Logs {
    param(
        [string]$ServiceName = "",
        [bool]$Follow = $false
    )
    
    $logArgs = @("logs")
    
    if ($Follow) {
        $logArgs += "-f"
    }
    
    if ($ServiceName) {
        $logArgs += $ServiceName
    }
    
    & docker-compose @logArgs
}

# Function to restart services
function Restart-Services {
    param(
        [string]$Env = "development",
        [string]$ServiceName = ""
    )
    
    if ($ServiceName) {
        Write-Status "Restarting $ServiceName..."
        docker-compose restart $ServiceName
    }
    else {
        Write-Status "Restarting all services..."
        Stop-Services $Env
        Start-Services $Env
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Services restarted successfully"
    }
    else {
        Write-Error "Failed to restart services"
    }
}

# Function to clean up
function Invoke-Cleanup {
    param([bool]$FullCleanup = $false)
    
    Write-Status "Cleaning up..."
    
    # Stop and remove containers
    docker-compose down --remove-orphans
    
    if ($FullCleanup) {
        Write-Warning "Performing full cleanup (removing volumes and images)..."
        
        # Remove volumes
        docker-compose down -v
        
        # Remove images
        $images = docker images --filter "reference=*$ProjectName*" -q
        if ($images) {
            docker rmi -f $images
        }
        
        # Remove unused networks
        docker network prune -f
        
        Write-Success "Full cleanup completed"
    }
    else {
        Write-Success "Basic cleanup completed"
    }
}

# Function to show help
function Show-Help {
    Write-Host "Qwen2 Docker Deployment Script (PowerShell)" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\docker-deploy.ps1 [COMMAND] [OPTIONS]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  setup                    Setup environment and create necessary files"
    Write-Host "  build [env]             Build Docker images (env: dev|prod)"
    Write-Host "  start [env] [profiles]  Start services (env: dev|prod, profiles: monitoring|database|email|tools)"
    Write-Host "  stop [env]              Stop services"
    Write-Host "  restart [env] [service] Restart services or specific service"
    Write-Host "  status                  Show service status and URLs"
    Write-Host "  logs [service] [follow] Show logs"
    Write-Host "  cleanup [full]          Clean up containers and optionally volumes/images"
    Write-Host "  help                    Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\docker-deploy.ps1 setup                # Initial setup"
    Write-Host "  .\docker-deploy.ps1 start dev            # Start development environment"
    Write-Host "  .\docker-deploy.ps1 start prod monitoring # Start production with monitoring"
    Write-Host "  .\docker-deploy.ps1 logs api             # Show API logs"
    Write-Host "  .\docker-deploy.ps1 restart dev api      # Restart API service in dev"
    Write-Host "  .\docker-deploy.ps1 cleanup full         # Full cleanup including volumes"
    Write-Host ""
    Write-Host "Environment files:" -ForegroundColor Yellow
    Write-Host "  .env                    # Main environment configuration"
    Write-Host "  .env.example           # Environment template"
    Write-Host ""
    Write-Host "Docker Compose files:" -ForegroundColor Yellow
    Write-Host "  docker-compose.yml     # Base configuration"
    Write-Host "  docker-compose.dev.yml # Development overrides"
    Write-Host "  docker-compose.prod.yml # Production overrides"
}

# Main script logic
switch ($Command.ToLower()) {
    "setup" {
        Test-Prerequisites
        Initialize-Environment
    }
    "build" {
        Test-Prerequisites
        Build-Images $Environment
    }
    "start" {
        Test-Prerequisites
        Start-Services $Environment $Profiles
    }
    "stop" {
        Stop-Services $Environment
    }
    "restart" {
        Test-Prerequisites
        Restart-Services $Environment $Service
    }
    "status" {
        Show-Status
    }
    "logs" {
        $follow = $Service -eq "true" -or $Service -eq "follow"
        if ($follow) {
            Show-Logs $Environment $true
        }
        else {
            Show-Logs $Service $false
        }
    }
    "cleanup" {
        $fullCleanup = $Environment -eq "full"
        Invoke-Cleanup $fullCleanup
    }
    { $_ -in @("help", "--help", "-h", "?") } {
        Show-Help
    }
    default {
        Write-Error "Unknown command: $Command"
        Show-Help
        exit 1
    }
}