#!/bin/bash

# Docker Environment Setup Script for Qwen2 Application
# This script sets up the Docker environment and creates necessary configuration files

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is installed and running
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    print_success "Docker is installed and running"
}

# Function to check if Docker Compose is available
check_docker_compose() {
    print_status "Checking Docker Compose..."
    
    if docker compose version &> /dev/null; then
        print_success "Docker Compose (plugin) is available"
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose &> /dev/null; then
        print_success "Docker Compose (standalone) is available"
        COMPOSE_CMD="docker-compose"
    else
        print_error "Docker Compose is not available. Please install Docker Compose."
        echo "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi
}

# Function to check GPU support
check_gpu_support() {
    print_status "Checking GPU support..."
    
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            print_success "NVIDIA GPU detected"
            GPU_AVAILABLE=true
        else
            print_warning "NVIDIA drivers not properly installed"
            GPU_AVAILABLE=false
        fi
    else
        print_warning "No NVIDIA GPU detected"
        GPU_AVAILABLE=false
    fi
    
    if [[ "$GPU_AVAILABLE" == "true" ]]; then
        if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
            print_success "Docker GPU support is working"
        else
            print_warning "Docker GPU support not available. Install nvidia-docker2"
            echo "Visit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        fi
    fi
}

# Function to create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    local directories=(
        "models"
        "cache"
        "generated_images"
        "uploads"
        "offload"
        "logs/api"
        "logs/traefik"
        "data/redis"
        "data/prometheus"
        "data/grafana"
        "config/traefik"
        "ssl"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        fi
    done
    
    # Set proper permissions
    chmod 755 models cache generated_images uploads offload
    chmod 755 logs logs/api logs/traefik
    chmod 755 data data/redis data/prometheus data/grafana
    chmod 755 config config/traefik
    chmod 755 ssl
    
    print_success "Directory structure created"
}

# Function to create environment files
create_env_files() {
    print_status "Creating environment configuration files..."
    
    # Create .env.docker if it doesn't exist
    if [[ ! -f ".env.docker" ]]; then
        cat > .env.docker << EOF
# Docker Environment Configuration for Qwen2 Application

# Application Settings
APP_NAME=qwen2-app
APP_VERSION=latest

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Frontend Configuration
FRONTEND_PORT=3000
REACT_APP_API_URL=http://localhost:8000

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
ENABLE_GPU=true

# Model Configuration
HF_HOME=/app/cache/huggingface
TORCH_HOME=/app/cache/torch
DIFFSYNTH_CACHE_DIR=/app/cache/diffsynth
CONTROLNET_CACHE_DIR=/app/cache/controlnet

# Feature Flags
ENABLE_DIFFSYNTH=true
ENABLE_CONTROLNET=true
ENABLE_QWEN_EDIT=true

# Performance Settings
MEMORY_OPTIMIZATION=true
TILED_PROCESSING_THRESHOLD=1024
MAX_WORKERS=4

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security
CORS_ORIGINS=http://localhost:3000,http://localhost:80
ALLOWED_HOSTS=localhost,127.0.0.1

# Traefik Configuration
TRAEFIK_DOMAIN=localhost
TRAEFIK_EMAIL=admin@localhost
ENABLE_SSL=false

# Database (if needed)
REDIS_URL=redis://redis:6379/0

# Monitoring
ENABLE_METRICS=true
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
EOF
        print_success "Created .env.docker file"
    else
        print_warning ".env.docker already exists, skipping creation"
    fi
    
    # Create production environment template
    if [[ ! -f ".env.docker.prod" ]]; then
        cat > .env.docker.prod << EOF
# Production Docker Environment Configuration

# Application Settings
APP_NAME=qwen2-app
APP_VERSION=latest

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Frontend Configuration
FRONTEND_PORT=3000
REACT_APP_API_URL=https://api.your-domain.com

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1
ENABLE_GPU=true

# Model Configuration
HF_HOME=/app/cache/huggingface
TORCH_HOME=/app/cache/torch
DIFFSYNTH_CACHE_DIR=/app/cache/diffsynth
CONTROLNET_CACHE_DIR=/app/cache/controlnet

# Feature Flags
ENABLE_DIFFSYNTH=true
ENABLE_CONTROLNET=true
ENABLE_QWEN_EDIT=true

# Performance Settings
MEMORY_OPTIMIZATION=true
TILED_PROCESSING_THRESHOLD=2048
MAX_WORKERS=8

# Logging
LOG_LEVEL=WARNING
LOG_FORMAT=json

# Security
CORS_ORIGINS=https://your-domain.com
ALLOWED_HOSTS=your-domain.com,api.your-domain.com

# Traefik Configuration
TRAEFIK_DOMAIN=your-domain.com
TRAEFIK_EMAIL=admin@your-domain.com
ENABLE_SSL=true

# Database
REDIS_URL=redis://redis:6379/0

# Monitoring
ENABLE_METRICS=true
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
EOF
        print_success "Created .env.docker.prod template"
    fi
}

# Function to create Traefik configuration
create_traefik_config() {
    print_status "Creating Traefik configuration..."
    
    if [[ ! -f "config/traefik/traefik.yml" ]]; then
        cat > config/traefik/traefik.yml << EOF
# Traefik Configuration for Qwen2 Application

global:
  checkNewVersion: false
  sendAnonymousUsage: false

api:
  dashboard: true
  insecure: true

entryPoints:
  web:
    address: ":80"
  websecure:
    address: ":443"

providers:
  docker:
    endpoint: "unix:///var/run/docker.sock"
    exposedByDefault: false
    network: qwen2-network

certificatesResolvers:
  letsencrypt:
    acme:
      email: admin@localhost
      storage: /acme.json
      httpChallenge:
        entryPoint: web

# Redirect HTTP to HTTPS (for production)
# http:
#   redirections:
#     entrypoint:
#       to: websecure
#       scheme: https

log:
  level: INFO
  filePath: "/var/log/traefik/traefik.log"

accessLog:
  filePath: "/var/log/traefik/access.log"
EOF
        print_success "Created Traefik configuration"
    fi
    
    # Create acme.json for SSL certificates
    if [[ ! -f "acme.json" ]]; then
        touch acme.json
        chmod 600 acme.json
        print_success "Created acme.json for SSL certificates"
    fi
}

# Function to create Docker management scripts
create_management_scripts() {
    print_status "Creating Docker management scripts..."
    
    # Create docker-ops.sh script
    cat > scripts/docker-ops.sh << 'EOF'
#!/bin/bash

# Docker Operations Script for Qwen2 Application
# Provides common Docker operations with simplified commands

set -e

COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="qwen2"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_usage() {
    cat << EOF
Docker Operations for Qwen2 Application

Usage: $0 COMMAND [OPTIONS]

COMMANDS:
    start [env]     Start services (dev/prod/staging)
    stop            Stop all services
    restart [env]   Restart services
    logs [service]  Show logs (all services or specific)
    status          Show service status
    build           Build all images
    pull            Pull latest images
    clean           Clean up containers and images
    shell [service] Open shell in service container
    exec [service] [cmd] Execute command in service
    backup          Backup volumes and data
    restore [file]  Restore from backup

EXAMPLES:
    $0 start dev        # Start development environment
    $0 logs api         # Show API service logs
    $0 shell api        # Open shell in API container
    $0 exec api "pip list"  # Execute command in API container

EOF
}

case "${1:-}" in
    start)
        ENV="${2:-dev}"
        case $ENV in
            dev) COMPOSE_FILE="docker-compose.dev.yml" ;;
            prod) COMPOSE_FILE="docker-compose.prod.yml" ;;
            staging) COMPOSE_FILE="docker-compose.yml" ;;
        esac
        print_status "Starting $ENV environment..."
        docker-compose -f "$COMPOSE_FILE" up -d
        print_success "Services started"
        ;;
    
    stop)
        print_status "Stopping all services..."
        docker-compose down
        print_success "Services stopped"
        ;;
    
    restart)
        ENV="${2:-dev}"
        case $ENV in
            dev) COMPOSE_FILE="docker-compose.dev.yml" ;;
            prod) COMPOSE_FILE="docker-compose.prod.yml" ;;
            staging) COMPOSE_FILE="docker-compose.yml" ;;
        esac
        print_status "Restarting $ENV environment..."
        docker-compose -f "$COMPOSE_FILE" restart
        print_success "Services restarted"
        ;;
    
    logs)
        SERVICE="${2:-}"
        if [[ -n "$SERVICE" ]]; then
            docker-compose logs -f "$SERVICE"
        else
            docker-compose logs -f
        fi
        ;;
    
    status)
        docker-compose ps
        ;;
    
    build)
        print_status "Building all images..."
        docker-compose build --no-cache
        print_success "Images built"
        ;;
    
    pull)
        print_status "Pulling latest images..."
        docker-compose pull
        print_success "Images pulled"
        ;;
    
    clean)
        print_warning "This will remove all containers, networks, and unused images"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose down -v --remove-orphans
            docker system prune -f
            print_success "Cleanup completed"
        fi
        ;;
    
    shell)
        SERVICE="${2:-api}"
        docker-compose exec "$SERVICE" /bin/bash
        ;;
    
    exec)
        SERVICE="${2:-api}"
        COMMAND="${3:-/bin/bash}"
        docker-compose exec "$SERVICE" $COMMAND
        ;;
    
    backup)
        BACKUP_FILE="qwen2-backup-$(date +%Y%m%d-%H%M%S).tar.gz"
        print_status "Creating backup: $BACKUP_FILE"
        tar -czf "$BACKUP_FILE" models cache generated_images uploads data
        print_success "Backup created: $BACKUP_FILE"
        ;;
    
    restore)
        BACKUP_FILE="${2:-}"
        if [[ -z "$BACKUP_FILE" || ! -f "$BACKUP_FILE" ]]; then
            print_error "Backup file not found: $BACKUP_FILE"
            exit 1
        fi
        print_warning "This will overwrite existing data"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            tar -xzf "$BACKUP_FILE"
            print_success "Backup restored from: $BACKUP_FILE"
        fi
        ;;
    
    *)
        show_usage
        exit 1
        ;;
esac
EOF
    
    chmod +x scripts/docker-ops.sh
    print_success "Created Docker operations script"
}

# Function to validate setup
validate_setup() {
    print_status "Validating Docker setup..."
    
    local errors=0
    
    # Check required files
    local required_files=(
        "Dockerfile.api"
        "frontend/Dockerfile"
        "docker-compose.yml"
        ".env.docker"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            print_error "Required file missing: $file"
            ((errors++))
        fi
    done
    
    # Check directories
    local required_dirs=(
        "models"
        "cache"
        "generated_images"
        "uploads"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            print_error "Required directory missing: $dir"
            ((errors++))
        fi
    done
    
    if [[ $errors -eq 0 ]]; then
        print_success "Docker environment setup is valid"
    else
        print_error "Setup validation failed with $errors errors"
        exit 1
    fi
}

# Main execution
main() {
    print_status "Setting up Docker environment for Qwen2 Application"
    
    check_docker
    check_docker_compose
    check_gpu_support
    create_directories
    create_env_files
    create_traefik_config
    create_management_scripts
    validate_setup
    
    print_success "Docker environment setup completed!"
    print_status "Next steps:"
    echo "  1. Review and customize .env.docker file"
    echo "  2. Run: ./scripts/deploy-docker.sh dev --build"
    echo "  3. Access application at http://localhost:3000"
    echo ""
    print_status "Available commands:"
    echo "  ./scripts/deploy-docker.sh [env] [options]  - Deploy application"
    echo "  ./scripts/docker-ops.sh [command]          - Manage Docker services"
}

# Run main function
main "$@"