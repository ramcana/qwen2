#!/bin/bash

# Traefik Configuration Validation Script
# This script validates the Traefik setup and configuration

set -e

echo "🔍 Validating Traefik configuration..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "OK")
            echo -e "${GREEN}✅ $message${NC}"
            ;;
        "WARN")
            echo -e "${YELLOW}⚠️  $message${NC}"
            ;;
        "ERROR")
            echo -e "${RED}❌ $message${NC}"
            ;;
        "INFO")
            echo -e "${BLUE}ℹ️  $message${NC}"
            ;;
    esac
}

# Check if Docker is running
check_docker() {
    print_status "INFO" "Checking Docker..."
    if ! docker info >/dev/null 2>&1; then
        print_status "ERROR" "Docker is not running or not accessible"
        exit 1
    fi
    print_status "OK" "Docker is running"
}

# Check required files
check_files() {
    print_status "INFO" "Checking required files..."
    
    local required_files=(
        "config/traefik/traefik.yml"
        "config/traefik/dynamic.yml"
        "docker-compose.traefik.yml"
        "ssl/acme.json"
    )
    
    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            print_status "OK" "Found $file"
        else
            print_status "ERROR" "Missing $file"
            return 1
        fi
    done
}

# Check required directories
check_directories() {
    print_status "INFO" "Checking required directories..."
    
    local required_dirs=(
        "config/traefik"
        "ssl/certs"
        "ssl/private"
        "logs/traefik"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [ -d "$dir" ]; then
            print_status "OK" "Found directory $dir"
        else
            print_status "ERROR" "Missing directory $dir"
            return 1
        fi
    done
}

# Check file permissions
check_permissions() {
    print_status "INFO" "Checking file permissions..."
    
    # Check ACME file permissions
    if [ -f "ssl/acme.json" ]; then
        local perms=$(stat -c "%a" ssl/acme.json 2>/dev/null || stat -f "%A" ssl/acme.json 2>/dev/null)
        if [ "$perms" = "600" ]; then
            print_status "OK" "ACME file permissions are correct (600)"
        else
            print_status "WARN" "ACME file permissions should be 600, found $perms"
        fi
    fi
    
    # Check SSL directory permissions
    if [ -d "ssl" ]; then
        local perms=$(stat -c "%a" ssl 2>/dev/null || stat -f "%A" ssl 2>/dev/null)
        if [ "$perms" = "700" ]; then
            print_status "OK" "SSL directory permissions are correct (700)"
        else
            print_status "WARN" "SSL directory permissions should be 700, found $perms"
        fi
    fi
}

# Validate Traefik configuration syntax
validate_config() {
    print_status "INFO" "Validating Traefik configuration syntax..."
    
    # Check if traefik binary is available for validation
    if command -v traefik >/dev/null 2>&1; then
        if traefik validate --configfile=config/traefik/traefik.yml >/dev/null 2>&1; then
            print_status "OK" "Traefik configuration syntax is valid"
        else
            print_status "ERROR" "Traefik configuration syntax is invalid"
            traefik validate --configfile=config/traefik/traefik.yml
            return 1
        fi
    else
        print_status "WARN" "Traefik binary not found, skipping syntax validation"
    fi
}

# Check Docker networks
check_networks() {
    print_status "INFO" "Checking Docker networks..."
    
    local required_networks=("qwen-network" "traefik-public")
    
    for network in "${required_networks[@]}"; do
        if docker network ls --format "{{.Name}}" | grep -q "^${network}$"; then
            print_status "OK" "Network $network exists"
        else
            print_status "WARN" "Network $network does not exist (will be created on startup)"
        fi
    done
}

# Check environment variables
check_env() {
    print_status "INFO" "Checking environment configuration..."
    
    if [ -f ".env" ]; then
        print_status "OK" "Environment file exists"
        
        # Check for required variables
        local required_vars=("ACME_EMAIL" "TRAEFIK_DOMAIN" "APP_DOMAIN" "API_DOMAIN")
        
        for var in "${required_vars[@]}"; do
            if grep -q "^${var}=" .env; then
                local value=$(grep "^${var}=" .env | cut -d'=' -f2)
                if [ -n "$value" ]; then
                    print_status "OK" "$var is set to: $value"
                else
                    print_status "WARN" "$var is defined but empty"
                fi
            else
                print_status "WARN" "$var is not defined in .env"
            fi
        done
    else
        print_status "WARN" "No .env file found"
    fi
}

# Test Docker Compose configuration
test_compose() {
    print_status "INFO" "Testing Docker Compose configuration..."
    
    if docker-compose -f docker-compose.traefik.yml config >/dev/null 2>&1; then
        print_status "OK" "Docker Compose configuration is valid"
    else
        print_status "ERROR" "Docker Compose configuration is invalid"
        docker-compose -f docker-compose.traefik.yml config
        return 1
    fi
}

# Check if Traefik is running
check_running() {
    print_status "INFO" "Checking if Traefik is running..."
    
    if docker-compose -f docker-compose.traefik.yml ps | grep -q "Up"; then
        print_status "OK" "Traefik container is running"
        
        # Test dashboard access
        local dashboard_url="http://localhost:8080/api/rawdata"
        if curl -s -f "$dashboard_url" >/dev/null 2>&1; then
            print_status "OK" "Traefik dashboard is accessible"
        else
            print_status "WARN" "Traefik dashboard is not accessible at $dashboard_url"
        fi
    else
        print_status "INFO" "Traefik container is not running"
    fi
}

# Main validation function
main() {
    echo "🚀 Starting Traefik validation..."
    echo ""
    
    local failed=0
    
    # Run all checks
    check_docker || failed=1
    echo ""
    
    check_files || failed=1
    echo ""
    
    check_directories || failed=1
    echo ""
    
    check_permissions || failed=1
    echo ""
    
    validate_config || failed=1
    echo ""
    
    check_networks || failed=1
    echo ""
    
    check_env || failed=1
    echo ""
    
    test_compose || failed=1
    echo ""
    
    check_running || failed=1
    echo ""
    
    # Summary
    if [ $failed -eq 0 ]; then
        print_status "OK" "All validation checks passed!"
        echo ""
        echo "🎉 Traefik configuration is ready!"
        echo ""
        echo "Next steps:"
        echo "1. Start Traefik: docker-compose -f docker-compose.traefik.yml up -d"
        echo "2. Check logs: docker-compose -f docker-compose.traefik.yml logs -f traefik"
        echo "3. Access dashboard: https://traefik.localhost:8080"
    else
        print_status "ERROR" "Some validation checks failed!"
        echo ""
        echo "Please fix the issues above before starting Traefik."
        exit 1
    fi
}

# Run main function
main "$@"