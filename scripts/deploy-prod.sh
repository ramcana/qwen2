#!/bin/bash

# =============================================================================
# Production Environment Deployment Script
# =============================================================================
# This script provides simplified commands for production deployment
# Requirements: 4.1, 4.2 - Simple deployment commands and environment-specific settings
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
PROD_COMPOSE_FILE="$PROJECT_ROOT/docker-compose.prod.yml"

# Function to print colored output
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

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking production prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    
    # Check if compose files exist
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        print_error "Base compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    if [[ ! -f "$PROD_COMPOSE_FILE" ]]; then
        print_error "Production compose file not found: $PROD_COMPOSE_FILE"
        exit 1
    fi
    
    # Check for GPU support in production
    if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        print_warning "GPU support not detected. Production deployment may not work optimally."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    print_success "Prerequisites check passed"
}

# Function to create necessary directories
create_directories() {
    print_status "Creating production directories..."
    
    local dirs=(
        "$PROJECT_ROOT/models"
        "$PROJECT_ROOT/cache/huggingface"
        "$PROJECT_ROOT/cache/torch"
        "$PROJECT_ROOT/cache/diffsynth"
        "$PROJECT_ROOT/cache/controlnet"
        "$PROJECT_ROOT/generated_images"
        "$PROJECT_ROOT/uploads"
        "$PROJECT_ROOT/offload"
        "$PROJECT_ROOT/logs/api"
        "$PROJECT_ROOT/logs/traefik"
        "$PROJECT_ROOT/data/redis"
        "$PROJECT_ROOT/data/prometheus"
        "$PROJECT_ROOT/data/grafana"
        "$PROJECT_ROOT/config/docker"
        "$PROJECT_ROOT/config/redis"
        "$PROJECT_ROOT/config/prometheus"
        "$PROJECT_ROOT/config/grafana"
        "$PROJECT_ROOT/ssl"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        fi
    done
    
    # Set proper permissions for production
    chmod 755 "$PROJECT_ROOT/models"
    chmod 755 "$PROJECT_ROOT/cache"
    chmod 755 "$PROJECT_ROOT/generated_images"
    chmod 755 "$PROJECT_ROOT/uploads"
    chmod 700 "$PROJECT_ROOT/ssl"
    
    print_success "Production directory structure created"
}

# Function to set up production environment
setup_environment() {
    print_status "Setting up production environment variables..."
    
    # Check if production .env exists
    if [[ ! -f "$PROJECT_ROOT/.env.prod" ]]; then
        print_warning "Production .env file not found. Creating template..."
        cat > "$PROJECT_ROOT/.env.prod" << EOF
# Production Environment Configuration
NODE_ENV=production
DEBUG=false
LOG_LEVEL=INFO

# Domain Configuration (REQUIRED - UPDATE THESE)
DOMAIN=yourdomain.com
API_DOMAIN=api.yourdomain.com
DASHBOARD_DOMAIN=dashboard.yourdomain.com
MONITORING_DOMAIN=monitoring.yourdomain.com

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=2

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all

# Cache Configuration
HF_HOME=/app/cache/huggingface
TORCH_HOME=/app/cache/torch
DIFFSYNTH_CACHE_DIR=/app/cache/diffsynth
CONTROLNET_CACHE_DIR=/app/cache/controlnet

# Feature Flags
ENABLE_DIFFSYNTH=true
ENABLE_CONTROLNET=true
ENABLE_QWEN_EDIT=true
ENABLE_QWEN_IMAGE=true

# Production Security Settings
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
CORS_ALLOW_CREDENTIALS=false
SECURE_COOKIES=true
HTTPS_ONLY=true

# Performance Settings
MEMORY_OPTIMIZATION=true
TILED_PROCESSING_THRESHOLD=4096
MAX_BATCH_SIZE=8
ATTENTION_OPTIMIZATION=true

# Cache Settings
CACHE_TTL=3600
MODEL_CACHE_SIZE=8GB
ENABLE_REDIS_CACHE=true
REDIS_URL=redis://redis:6379/0

# SSL Configuration
SSL_EMAIL=admin@yourdomain.com
ACME_CA_SERVER=https://acme-v02.api.letsencrypt.org/directory

# Database Configuration (if using)
POSTGRES_DB=qwen_prod
POSTGRES_USER=qwen
POSTGRES_PASSWORD=CHANGE_THIS_PASSWORD

# Monitoring Configuration
GRAFANA_ADMIN_PASSWORD=CHANGE_THIS_PASSWORD
PROMETHEUS_RETENTION=30d

# Security
TRAEFIK_DASHBOARD_PASSWORD=CHANGE_THIS_PASSWORD
EOF
        print_error "Please edit .env.prod file with your production settings before continuing!"
        print_error "Required changes:"
        print_error "  - Update DOMAIN settings"
        print_error "  - Change all passwords"
        print_error "  - Update SSL_EMAIL"
        exit 1
    fi
    
    # Validate required environment variables
    source "$PROJECT_ROOT/.env.prod"
    
    if [[ "$DOMAIN" == "yourdomain.com" ]]; then
        print_error "Please update DOMAIN in .env.prod file"
        exit 1
    fi
    
    if [[ "$POSTGRES_PASSWORD" == "CHANGE_THIS_PASSWORD" ]]; then
        print_error "Please update passwords in .env.prod file"
        exit 1
    fi
    
    print_success "Production environment validated"
}

# Function to build production images
build_images() {
    print_status "Building production images..."
    
    cd "$PROJECT_ROOT"
    
    # Use docker-compose or docker compose based on availability
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Build with production optimizations
    $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" build --parallel --no-cache
    
    print_success "Production images built successfully"
}

# Function to start production services
start_services() {
    print_status "Starting production services..."
    
    cd "$PROJECT_ROOT"
    
    # Use docker-compose or docker compose based on availability
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Load production environment
    export $(cat .env.prod | grep -v '^#' | xargs)
    
    # Start core services first
    print_status "Starting core services..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" up -d traefik redis
    
    # Wait for core services to be healthy
    print_status "Waiting for core services to be ready..."
    sleep 10
    
    # Start application services
    print_status "Starting application services..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" up -d api frontend
    
    # Start monitoring services if profile is enabled
    if [[ "${ENABLE_MONITORING:-false}" == "true" ]]; then
        print_status "Starting monitoring services..."
        $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" --profile monitoring up -d
    fi
    
    print_success "Production services started"
    
    # Show service status
    print_status "Service status:"
    $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" ps
    
    # Show access URLs
    echo ""
    print_success "Production environment is ready!"
    echo ""
    echo "Access URLs:"
    echo "  Frontend:        https://$DOMAIN"
    echo "  API:             https://$API_DOMAIN"
    echo "  API Docs:        https://$API_DOMAIN/docs"
    if [[ "${ENABLE_MONITORING:-false}" == "true" ]]; then
        echo "  Monitoring:      https://$MONITORING_DOMAIN"
    fi
    echo ""
    echo "Management commands:"
    echo "  View logs:       $0 logs [service]"
    echo "  Scale API:       $0 scale api=3"
    echo "  Update services: $0 update"
    echo "  Stop services:   $0 stop"
    echo ""
}

# Function to stop production services
stop_services() {
    print_status "Stopping production services..."
    
    cd "$PROJECT_ROOT"
    
    # Use docker-compose or docker compose based on availability
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Graceful shutdown
    $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" stop
    
    print_success "Production services stopped"
}

# Function to scale services
scale_services() {
    if [[ -z "$1" ]]; then
        print_error "Please specify service and scale count (e.g., api=3)"
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
    
    # Use docker-compose or docker compose based on availability
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    print_status "Scaling service: $1"
    $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" up -d --scale "$1"
    
    print_success "Service scaled successfully"
}

# Function to update services
update_services() {
    print_status "Updating production services..."
    
    cd "$PROJECT_ROOT"
    
    # Use docker-compose or docker compose based on availability
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Pull latest images
    print_status "Pulling latest images..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" pull
    
    # Rebuild custom images
    print_status "Rebuilding custom images..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" build --parallel
    
    # Rolling update
    print_status "Performing rolling update..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" up -d --remove-orphans
    
    print_success "Production services updated"
}

# Function to show logs
show_logs() {
    cd "$PROJECT_ROOT"
    
    # Use docker-compose or docker compose based on availability
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    if [[ -n "$1" ]]; then
        $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" logs -f --tail=100 "$1"
    else
        $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" logs -f --tail=100
    fi
}

# Function to show service status
show_status() {
    cd "$PROJECT_ROOT"
    
    # Use docker-compose or docker compose based on availability
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    print_status "Production services status:"
    $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" ps
    
    # Show resource usage
    print_status "Resource usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
}

# Function to backup data
backup_data() {
    print_status "Creating production backup..."
    
    local backup_dir="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup volumes
    print_status "Backing up data volumes..."
    tar -czf "$backup_dir/models.tar.gz" -C "$PROJECT_ROOT" models/
    tar -czf "$backup_dir/generated_images.tar.gz" -C "$PROJECT_ROOT" generated_images/
    tar -czf "$backup_dir/uploads.tar.gz" -C "$PROJECT_ROOT" uploads/
    
    # Backup configuration
    print_status "Backing up configuration..."
    tar -czf "$backup_dir/config.tar.gz" -C "$PROJECT_ROOT" config/
    
    # Backup database if exists
    if docker ps --format '{{.Names}}' | grep -q qwen-postgres-prod; then
        print_status "Backing up database..."
        docker exec qwen-postgres-prod pg_dump -U qwen qwen_prod > "$backup_dir/database.sql"
    fi
    
    print_success "Backup created: $backup_dir"
}

# Function to restore data
restore_data() {
    if [[ -z "$1" ]]; then
        print_error "Please specify backup directory"
        exit 1
    fi
    
    local backup_dir="$1"
    
    if [[ ! -d "$backup_dir" ]]; then
        print_error "Backup directory not found: $backup_dir"
        exit 1
    fi
    
    print_warning "This will overwrite existing data. Are you sure?"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    
    print_status "Restoring from backup: $backup_dir"
    
    # Stop services
    stop_services
    
    # Restore data
    if [[ -f "$backup_dir/models.tar.gz" ]]; then
        tar -xzf "$backup_dir/models.tar.gz" -C "$PROJECT_ROOT"
    fi
    
    if [[ -f "$backup_dir/generated_images.tar.gz" ]]; then
        tar -xzf "$backup_dir/generated_images.tar.gz" -C "$PROJECT_ROOT"
    fi
    
    if [[ -f "$backup_dir/uploads.tar.gz" ]]; then
        tar -xzf "$backup_dir/uploads.tar.gz" -C "$PROJECT_ROOT"
    fi
    
    if [[ -f "$backup_dir/config.tar.gz" ]]; then
        tar -xzf "$backup_dir/config.tar.gz" -C "$PROJECT_ROOT"
    fi
    
    print_success "Data restored successfully"
}

# Main function
main() {
    case "${1:-help}" in
        "start"|"up")
            check_prerequisites
            create_directories
            setup_environment
            build_images
            start_services
            ;;
        "stop"|"down")
            stop_services
            ;;
        "restart")
            stop_services
            sleep 5
            start_services
            ;;
        "scale")
            scale_services "$2"
            ;;
        "update")
            update_services
            ;;
        "logs")
            show_logs "$2"
            ;;
        "status"|"ps")
            show_status
            ;;
        "build")
            check_prerequisites
            build_images
            ;;
        "backup")
            backup_data
            ;;
        "restore")
            restore_data "$2"
            ;;
        "help"|"-h"|"--help")
            echo "Production Deployment Script for Qwen2 Image Generation"
            echo ""
            echo "Usage: $0 [COMMAND] [OPTIONS]"
            echo ""
            echo "Commands:"
            echo "  start, up          Start production environment"
            echo "  stop, down         Stop production environment"
            echo "  restart            Restart production environment"
            echo "  scale SERVICE=N    Scale service to N replicas"
            echo "  update             Update services with latest images"
            echo "  logs [service]     Show logs for all services or specific service"
            echo "  status, ps         Show service status and resource usage"
            echo "  build              Build production images"
            echo "  backup             Create backup of production data"
            echo "  restore DIR        Restore from backup directory"
            echo "  help, -h, --help   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 start           # Start production environment"
            echo "  $0 scale api=3     # Scale API service to 3 replicas"
            echo "  $0 logs api        # Show API service logs"
            echo "  $0 backup          # Create backup"
            echo ""
            echo "Prerequisites:"
            echo "  - Docker and Docker Compose installed"
            echo "  - GPU drivers and nvidia-docker runtime"
            echo "  - Configured .env.prod file"
            echo "  - Valid SSL certificates or Let's Encrypt setup"
            echo ""
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Use '$0 help' for usage information."
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"