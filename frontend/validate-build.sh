#!/bin/bash

# =============================================================================
# Frontend Build Validation Script
# =============================================================================

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

# Function to validate dependencies
validate_dependencies() {
    print_status "Validating dependencies..."
    
    if [ ! -f "package.json" ]; then
        print_error "package.json not found"
        return 1
    fi
    
    if [ ! -f "package-lock.json" ]; then
        print_warning "package-lock.json not found - this may cause dependency issues"
    fi
    
    # Check if TypeScript is available
    if npm list typescript > /dev/null 2>&1; then
        print_success "TypeScript dependency found"
    else
        print_error "TypeScript dependency missing"
        return 1
    fi
    
    # Check if react-scripts is available
    if npm list react-scripts > /dev/null 2>&1; then
        print_success "react-scripts dependency found"
    else
        print_error "react-scripts dependency missing"
        return 1
    fi
    
    return 0
}

# Function to validate source files
validate_source_files() {
    print_status "Validating source files..."
    
    if [ ! -f "src/index.tsx" ]; then
        print_error "src/index.tsx not found"
        return 1
    fi
    
    if [ ! -f "public/index.html" ]; then
        print_error "public/index.html not found"
        return 1
    fi
    
    if [ ! -f "tsconfig.json" ]; then
        print_warning "tsconfig.json not found"
    fi
    
    print_success "Source files validation passed"
    return 0
}

# Function to test TypeScript compilation
test_typescript_compilation() {
    print_status "Testing TypeScript compilation..."
    
    if npx tsc --noEmit --skipLibCheck; then
        print_success "TypeScript compilation test passed"
        return 0
    else
        print_error "TypeScript compilation test failed"
        return 1
    fi
}

# Function to test build process
test_build_process() {
    print_status "Testing build process..."
    
    # Clean any existing build
    if [ -d "build" ]; then
        rm -rf build
        print_status "Cleaned existing build directory"
    fi
    
    # Run build
    if npm run build:production; then
        print_success "Build process completed"
    else
        print_error "Build process failed"
        return 1
    fi
    
    # Validate build output
    if [ ! -d "build" ]; then
        print_error "Build directory not created"
        return 1
    fi
    
    if [ ! -f "build/index.html" ]; then
        print_error "build/index.html not found"
        return 1
    fi
    
    if [ ! -d "build/static" ]; then
        print_warning "build/static directory not found"
    fi
    
    print_success "Build output validation passed"
    return 0
}

# Function to validate nginx configuration
validate_nginx_config() {
    print_status "Validating nginx configuration..."
    
    if [ ! -f "nginx.prod.conf" ]; then
        print_error "nginx.prod.conf not found"
        return 1
    fi
    
    if [ ! -f "nginx-security.conf" ]; then
        print_error "nginx-security.conf not found"
        return 1
    fi
    
    # Test nginx config syntax if nginx is available
    if command -v nginx > /dev/null 2>&1; then
        if nginx -t -c "$(pwd)/nginx.prod.conf"; then
            print_success "Nginx configuration syntax is valid"
        else
            print_error "Nginx configuration syntax is invalid"
            return 1
        fi
    else
        print_warning "nginx not available for configuration testing"
    fi
    
    return 0
}

# Main validation function
main() {
    print_status "Starting frontend build validation..."
    
    local exit_code=0
    
    validate_dependencies || exit_code=1
    validate_source_files || exit_code=1
    test_typescript_compilation || exit_code=1
    validate_nginx_config || exit_code=1
    
    if [ "$1" = "--full" ]; then
        test_build_process || exit_code=1
    fi
    
    if [ $exit_code -eq 0 ]; then
        print_success "All validations passed!"
    else
        print_error "Some validations failed"
    fi
    
    return $exit_code
}

# Show usage
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [--full] [--help]"
    echo ""
    echo "Options:"
    echo "  --full    Run full validation including build test"
    echo "  --help    Show this help message"
    echo ""
    echo "This script validates the frontend build setup and dependencies."
    exit 0
fi

# Run main function
main "$@"