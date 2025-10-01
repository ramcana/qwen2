#!/bin/bash

# Traefik Manager Script
# Provides common operations for managing Traefik

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE_DEV="docker-compose.traefik.yml"
COMPOSE_FILE_PROD="docker-compose.traefik.prod.yml"
SERVICE_NAME="traefik"

# Function to print colored output
print_msg() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Show usage
show_usage() {
    echo "Traefik Manager - Manage Traefik reverse proxy"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup                 Initialize Traefik configuration"
    echo "  start [dev|prod]      Start Traefik (default: dev)"
    echo "  stop [dev|prod]       Stop Traefik (default: dev)"
    echo "  restart [dev|prod]    Restart Traefik (default: dev)"
    echo "  status [dev|prod]     Show Traefik status (default: dev)"
    echo "  logs [dev|prod]       Show Traefik logs (default: dev)"
    echo "  validate              Validate configuration"
    echo "  dashboard             Open dashboard in browser"
    echo "  cert-info             Show certificate information"
    echo "  cert-renew            Force certificate renewal"
    echo "  cleanup               Clean up containers and networks"
    echo "  help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup              # Initialize configuration"
    echo "  $0 start dev          # Start development environment"
    echo "  $0 start prod         # Start production environment"
    echo "  $0 logs dev -f        # Follow development logs"
    echo "  $0 dashboard          # Open dashboard"
}

# Get compose file based on environment
get_compose_file() {
    local env=${1:-dev}
    case $env in
        "prod"|"production")
            echo "$COMPOSE_FILE_PROD"
            ;;
        "dev"|"development"|*)
            echo "$COMPOSE_FILE_DEV"
            ;;
    esac
}

# Setup Traefik configuration
setup_traefik() {
    print_msg "$BLUE" "🚀 Setting up Traefik configuration..."
    
    if [ -f "scripts/setup_traefik.sh" ]; then
        ./scripts/setup_traefik.sh
    else
        print_msg "$RED" "❌ Setup script not found: scripts/setup_traefik.sh"
        exit 1
    fi
}

# Start Traefik
start_traefik() {
    local env=${1:-dev}
    local compose_file=$(get_compose_file "$env")
    
    print_msg "$GREEN" "🚀 Starting Traefik ($env environment)..."
    
    # Create networks if they don't exist
    docker network create qwen-network 2>/dev/null || true
    docker network create traefik-public 2>/dev/null || true
    
    docker-compose -f "$compose_file" up -d
    
    print_msg "$GREEN" "✅ Traefik started successfully!"
    
    # Show access information
    if [ "$env" = "dev" ] || [ "$env" = "development" ]; then
        echo ""
        print_msg "$BLUE" "📋 Access Information:"
        echo "  Dashboard: https://traefik.localhost:8080 (admin/admin)"
        echo "  API Health: https://api.localhost/health"
        echo "  App Health: https://app.localhost/health"
    fi
}

# Stop Traefik
stop_traefik() {
    local env=${1:-dev}
    local compose_file=$(get_compose_file "$env")
    
    print_msg "$YELLOW" "🛑 Stopping Traefik ($env environment)..."
    
    docker-compose -f "$compose_file" down
    
    print_msg "$GREEN" "✅ Traefik stopped successfully!"
}

# Restart Traefik
restart_traefik() {
    local env=${1:-dev}
    
    print_msg "$BLUE" "🔄 Restarting Traefik ($env environment)..."
    
    stop_traefik "$env"
    sleep 2
    start_traefik "$env"
}

# Show Traefik status
show_status() {
    local env=${1:-dev}
    local compose_file=$(get_compose_file "$env")
    
    print_msg "$BLUE" "📊 Traefik Status ($env environment):"
    echo ""
    
    # Container status
    docker-compose -f "$compose_file" ps
    
    echo ""
    
    # Health check
    if docker-compose -f "$compose_file" ps | grep -q "Up"; then
        print_msg "$GREEN" "✅ Traefik is running"
        
        # Test dashboard
        if curl -s -f "http://localhost:8080/ping" >/dev/null 2>&1; then
            print_msg "$GREEN" "✅ Dashboard is accessible"
        else
            print_msg "$YELLOW" "⚠️  Dashboard is not accessible"
        fi
    else
        print_msg "$RED" "❌ Traefik is not running"
    fi
}

# Show logs
show_logs() {
    local env=${1:-dev}
    local compose_file=$(get_compose_file "$env")
    shift 2>/dev/null || true  # Remove first argument
    
    print_msg "$BLUE" "📋 Traefik Logs ($env environment):"
    
    docker-compose -f "$compose_file" logs "$@" "$SERVICE_NAME"
}

# Validate configuration
validate_config() {
    print_msg "$BLUE" "🔍 Validating Traefik configuration..."
    
    if [ -f "scripts/validate_traefik.sh" ]; then
        ./scripts/validate_traefik.sh
    else
        print_msg "$RED" "❌ Validation script not found: scripts/validate_traefik.sh"
        exit 1
    fi
}

# Open dashboard in browser
open_dashboard() {
    local url="https://traefik.localhost:8080"
    
    print_msg "$BLUE" "🌐 Opening Traefik dashboard..."
    
    # Try to open in browser (cross-platform)
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url"
    elif command -v open >/dev/null 2>&1; then
        open "$url"
    elif command -v start >/dev/null 2>&1; then
        start "$url"
    else
        print_msg "$YELLOW" "⚠️  Could not open browser automatically"
        print_msg "$BLUE" "📋 Dashboard URL: $url"
        print_msg "$BLUE" "📋 Credentials: admin/admin"
    fi
}

# Show certificate information
show_cert_info() {
    print_msg "$BLUE" "🔐 Certificate Information:"
    echo ""
    
    # Check ACME storage
    if [ -f "ssl/acme.json" ]; then
        local size=$(stat -c%s "ssl/acme.json" 2>/dev/null || stat -f%z "ssl/acme.json" 2>/dev/null)
        if [ "$size" -gt 10 ]; then
            print_msg "$GREEN" "✅ ACME certificates found (${size} bytes)"
            
            # Try to show certificate details (requires jq)
            if command -v jq >/dev/null 2>&1; then
                echo ""
                print_msg "$BLUE" "📋 Certificate Details:"
                jq -r '.letsencrypt.Certificates[]? | "Domain: \(.domain.main // "N/A") | Expires: \(.certificate | @base64d | split("\n")[1] // "N/A")"' ssl/acme.json 2>/dev/null || print_msg "$YELLOW" "⚠️  Could not parse certificate details"
            fi
        else
            print_msg "$YELLOW" "⚠️  ACME storage is empty"
        fi
    else
        print_msg "$RED" "❌ ACME storage not found"
    fi
    
    # Check self-signed certificates
    if [ -f "ssl/certs/localhost.crt" ]; then
        print_msg "$GREEN" "✅ Self-signed certificate found"
        
        # Show certificate details
        if command -v openssl >/dev/null 2>&1; then
            echo ""
            print_msg "$BLUE" "📋 Self-signed Certificate Details:"
            openssl x509 -in ssl/certs/localhost.crt -text -noout | grep -E "(Subject:|Not Before|Not After|DNS:)" || true
        fi
    else
        print_msg "$YELLOW" "⚠️  Self-signed certificate not found"
    fi
}

# Force certificate renewal
renew_certificates() {
    print_msg "$BLUE" "🔄 Forcing certificate renewal..."
    
    # Remove ACME storage to force renewal
    if [ -f "ssl/acme.json" ]; then
        cp ssl/acme.json ssl/acme.json.backup
        echo "{}" > ssl/acme.json
        chmod 600 ssl/acme.json
        print_msg "$GREEN" "✅ ACME storage cleared (backup created)"
    fi
    
    # Restart Traefik to trigger renewal
    restart_traefik "dev"
    
    print_msg "$BLUE" "📋 Certificate renewal initiated. Check logs for progress."
}

# Cleanup containers and networks
cleanup() {
    print_msg "$YELLOW" "🧹 Cleaning up Traefik resources..."
    
    # Stop all Traefik containers
    docker-compose -f "$COMPOSE_FILE_DEV" down 2>/dev/null || true
    docker-compose -f "$COMPOSE_FILE_PROD" down 2>/dev/null || true
    
    # Remove networks (only if empty)
    docker network rm qwen-network 2>/dev/null || print_msg "$YELLOW" "⚠️  Could not remove qwen-network (may be in use)"
    docker network rm traefik-public 2>/dev/null || print_msg "$YELLOW" "⚠️  Could not remove traefik-public (may be in use)"
    
    # Clean up unused volumes
    docker volume prune -f
    
    print_msg "$GREEN" "✅ Cleanup completed!"
}

# Main function
main() {
    local command=${1:-help}
    
    case $command in
        "setup")
            setup_traefik
            ;;
        "start")
            start_traefik "$2"
            ;;
        "stop")
            stop_traefik "$2"
            ;;
        "restart")
            restart_traefik "$2"
            ;;
        "status")
            show_status "$2"
            ;;
        "logs")
            show_logs "$2" "${@:3}"
            ;;
        "validate")
            validate_config
            ;;
        "dashboard")
            open_dashboard
            ;;
        "cert-info")
            show_cert_info
            ;;
        "cert-renew")
            renew_certificates
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|*)
            show_usage
            ;;
    esac
}

# Run main function with all arguments
main "$@"