#!/bin/bash

# =============================================================================
# Development Environment Deployment Script
# =============================================================================
# This script provides simplified commands for development deployment
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
DEV_COMPOSE_FILE="$PROJECT_ROOT/docker-compose.dev.yml"

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
    print_status "Checking prerequisites..."
    
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
    
    if [[ ! -f "$DEV_COMPOSE_FILE" ]]; then
        print_error "Development compose file not found: $DEV_COMPOSE_FILE"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
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
        "$PROJECT_ROOT/config/docker"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        fi
    done
    
    print_success "Directory structure created"
}

# Function to set up environment variables
setup_environment() {
    print_status "Setting up development environment variables..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
        if [[ -f "$PROJECT_ROOT/.env.example" ]]; then
            cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
            print_status "Created .env file from .env.example"
        else
            cat > "$PROJECT_ROOT/.env" << EOF
# Development Environment Configuration
NODE_ENV=development
DEBUG=true
LOG_LEVEL=DEBUG

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

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

# Development Settings
CORS_ORIGINS=*
CORS_ALLOW_CREDENTIALS=true
MEMORY_OPTIMIZATION=false
TILED_PROCESSING_THRESHOLD=1024
MAX_BATCH_SIZE=2
EOF
            print_status "Created default .env file"
        fi
    fi
    
    print_success "Environment setup completed"
}

# Function to build images
build_images() {
    print_status "Building development images..."
    
    cd "$PROJECT_ROOT"
    
    # Use docker-compose or docker compose based on availability
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" build --parallel
    
    print_success "Images built successfully"
}

# Function to start services
start_services() {
    print_status "Starting development services..."
    
    cd "$PROJECT_ROOT"
    
    # Use docker-compose or docker compose based on availability
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Start services with development profiles
    $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" up -d
    
    print_success "Development services started"
    
    # Show service status
    print_status "Service status:"
    $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" ps
    
    # Show access URLs
    echo ""
    print_success "Development environment is ready!"
    echo ""
    echo "Access URLs:"
    echo "  Frontend:        http://localhost:3000"
    echo "  API:             http://localhost:8000"
    echo "  API Docs:        http://localhost:8000/docs"
    echo "  Traefik Dashboard: http://localhost:8080"
    echo ""
    echo "Useful commands:"
    echo "  View logs:       $COMPOSE_CMD -f docker-compose.yml -f docker-compose.dev.yml logs -f"
    echo "  Stop services:   $COMPOSE_CMD -f docker-compose.yml -f docker-compose.dev.yml down"
    echo "  Restart API:     $COMPOSE_CMD -f docker-compose.yml -f docker-compose.dev.yml restart api"
    echo ""
}

# Function to stop services
stop_services() {
    print_status "Stopping development services..."
    
    cd "$PROJECT_ROOT"
    
    # Use docker-compose or docker compose based on availability
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" down
    
    print_success "Development services stopped"
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
        $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" logs -f "$1"
    else
        $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" logs -f
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
    
    print_status "Development services status:"
    $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" ps
}

# Function to restart services
restart_services() {
    cd "$PROJECT_ROOT"
    
    # Use docker-compose or docker compose based on availability
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    if [[ -n "$1" ]]; then
        print_status "Restarting service: $1"
        $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" restart "$1"
    else
        print_status "Restarting all services..."
        $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" restart
    fi
    
    print_success "Services restarted"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up development environment..."
    
    cd "$PROJECT_ROOT"
    
    # Use docker-compose or docker compose based on availability
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Stop and remove containers, networks
    $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" down --remove-orphans
    
    # Remove development images if requested
    if [[ "$1" == "--images" ]]; then
        print_status "Removing development images..."
        docker image rm qwen-api:latest qwen-frontend:latest 2>/dev/null || true
    fi
    
    # Remove volumes if requested
    if [[ "$1" == "--volumes" ]]; then
        print_status "Removing development volumes..."
        $COMPOSE_CMD -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" down --volumes
    fi
    
    print_success "Cleanup completed"
}

# Main function
main() {
    case "${1:-start}" in
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
            restart_services "$2"
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
        "clean")
            cleanup "$2"
            ;;
        "help"|"-h"|"--help")
            echo "Development Deployment Script for Qwen2 Image Generation"
            echo ""
            echo "Usage: $0 [COMMAND] [OPTIONS]"
            echo ""
            echo "Commands:"
            echo "  start, up          Start development environment (default)"
            echo "  stop, down         Stop development environment"
            echo "  restart [service]  Restart all services or specific service"
            echo "  logs [service]     Show logs for all services or specific service"
            echo "  status, ps         Show service status"
            echo "  build              Build development images"
            echo "  clean [--images|--volumes]  Clean up environment"
            echo "  help, -h, --help   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 start           # Start development environment"
            echo "  $0 logs api        # Show API service logs"
            echo "  $0 restart api     # Restart API service"
            echo "  $0 clean --volumes # Clean up including volumes"
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