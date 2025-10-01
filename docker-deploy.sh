#!/bin/bash
# =============================================================================
# Docker Deployment Script for Qwen2 Image Generation
# =============================================================================
# This script provides easy deployment commands for different environments
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
PROJECT_NAME="qwen2"

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
    
    # Check for NVIDIA Docker (for GPU support)
    if command -v nvidia-docker &> /dev/null || docker info | grep -q nvidia; then
        print_success "NVIDIA Docker runtime detected - GPU support available"
    else
        print_warning "NVIDIA Docker runtime not detected - GPU support may not be available"
    fi
    
    print_success "Prerequisites check completed"
}

# Function to setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        print_status "Creating .env file from template..."
        cp .env.example .env
        print_warning "Please edit .env file to customize your configuration"
    fi
    
    # Create necessary directories
    print_status "Creating necessary directories..."
    mkdir -p models cache generated_images uploads offload logs/{api,traefik} data/{redis,prometheus,grafana}
    mkdir -p config/{redis,prometheus,grafana/{dashboards,datasources}} ssl
    
    # Set proper permissions
    chmod 600 acme.json 2>/dev/null || touch acme.json && chmod 600 acme.json
    
    # Create external network if it doesn't exist
    docker network create traefik-public 2>/dev/null || true
    
    print_success "Environment setup completed"
}

# Function to build images
build_images() {
    local environment=${1:-development}
    print_status "Building Docker images for $environment environment..."
    
    case $environment in
        "development"|"dev")
            docker-compose -f docker-compose.yml -f docker-compose.dev.yml build --parallel
            ;;
        "production"|"prod")
            docker-compose -f docker-compose.yml -f docker-compose.prod.yml build --parallel
            ;;
        *)
            docker-compose build --parallel
            ;;
    esac
    
    print_success "Docker images built successfully"
}

# Function to start services
start_services() {
    local environment=${1:-development}
    local profiles=${2:-""}
    
    print_status "Starting services for $environment environment..."
    
    case $environment in
        "development"|"dev")
            if [ -n "$profiles" ]; then
                docker-compose -f docker-compose.yml -f docker-compose.dev.yml --profile "$profiles" up -d
            else
                docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
            fi
            ;;
        "production"|"prod")
            if [ -n "$profiles" ]; then
                docker-compose -f docker-compose.yml -f docker-compose.prod.yml --profile "$profiles" up -d
            else
                docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
            fi
            ;;
        *)
            docker-compose up -d
            ;;
    esac
    
    print_success "Services started successfully"
    show_status
}

# Function to stop services
stop_services() {
    local environment=${1:-development}
    
    print_status "Stopping services..."
    
    case $environment in
        "development"|"dev")
            docker-compose -f docker-compose.yml -f docker-compose.dev.yml down
            ;;
        "production"|"prod")
            docker-compose -f docker-compose.yml -f docker-compose.prod.yml down
            ;;
        *)
            docker-compose down
            ;;
    esac
    
    print_success "Services stopped successfully"
}

# Function to show service status
show_status() {
    print_status "Service Status:"
    docker-compose ps
    
    print_status "Available URLs:"
    echo "  Frontend: http://qwen.localhost"
    echo "  API: http://api.localhost"
    echo "  Traefik Dashboard: http://traefik.localhost:8080"
    
    if docker-compose ps | grep -q redis; then
        echo "  Redis: localhost:6379"
    fi
    
    if docker-compose ps | grep -q grafana; then
        echo "  Grafana: http://monitoring.localhost"
    fi
}

# Function to show logs
show_logs() {
    local service=${1:-""}
    local follow=${2:-false}
    
    if [ -n "$service" ]; then
        if [ "$follow" = "true" ]; then
            docker-compose logs -f "$service"
        else
            docker-compose logs "$service"
        fi
    else
        if [ "$follow" = "true" ]; then
            docker-compose logs -f
        else
            docker-compose logs
        fi
    fi
}

# Function to restart services
restart_services() {
    local environment=${1:-development}
    local service=${2:-""}
    
    if [ -n "$service" ]; then
        print_status "Restarting $service..."
        docker-compose restart "$service"
    else
        print_status "Restarting all services..."
        stop_services "$environment"
        start_services "$environment"
    fi
    
    print_success "Services restarted successfully"
}

# Function to clean up
cleanup() {
    local full_cleanup=${1:-false}
    
    print_status "Cleaning up..."
    
    # Stop and remove containers
    docker-compose down --remove-orphans
    
    if [ "$full_cleanup" = "true" ]; then
        print_warning "Performing full cleanup (removing volumes and images)..."
        
        # Remove volumes
        docker-compose down -v
        
        # Remove images
        docker images | grep "$PROJECT_NAME" | awk '{print $3}' | xargs -r docker rmi -f
        
        # Remove unused networks
        docker network prune -f
        
        print_success "Full cleanup completed"
    else
        print_success "Basic cleanup completed"
    fi
}

# Function to show help
show_help() {
    echo "Qwen2 Docker Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup                    Setup environment and create necessary files"
    echo "  build [env]             Build Docker images (env: dev|prod)"
    echo "  start [env] [profiles]  Start services (env: dev|prod, profiles: monitoring|database|email|tools)"
    echo "  stop [env]              Stop services"
    echo "  restart [env] [service] Restart services or specific service"
    echo "  status                  Show service status and URLs"
    echo "  logs [service] [follow] Show logs (follow: true|false)"
    echo "  cleanup [full]          Clean up containers and optionally volumes/images"
    echo "  help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup                # Initial setup"
    echo "  $0 start dev            # Start development environment"
    echo "  $0 start prod monitoring # Start production with monitoring"
    echo "  $0 logs api true        # Follow API logs"
    echo "  $0 restart dev api      # Restart API service in dev"
    echo "  $0 cleanup full         # Full cleanup including volumes"
    echo ""
    echo "Environment files:"
    echo "  .env                    # Main environment configuration"
    echo "  .env.example           # Environment template"
    echo ""
    echo "Docker Compose files:"
    echo "  docker-compose.yml     # Base configuration"
    echo "  docker-compose.dev.yml # Development overrides"
    echo "  docker-compose.prod.yml # Production overrides"
}

# Main script logic
main() {
    case "${1:-help}" in
        "setup")
            check_prerequisites
            setup_environment
            ;;
        "build")
            check_prerequisites
            build_images "${2:-development}"
            ;;
        "start")
            check_prerequisites
            start_services "${2:-development}" "${3:-}"
            ;;
        "stop")
            stop_services "${2:-development}"
            ;;
        "restart")
            check_prerequisites
            restart_services "${2:-development}" "${3:-}"
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "${2:-}" "${3:-false}"
            ;;
        "cleanup")
            cleanup "${2:-false}"
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"