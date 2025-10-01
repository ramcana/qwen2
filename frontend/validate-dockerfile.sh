#!/bin/bash

# =============================================================================
# Dockerfile Validation Script for Qwen2 Frontend
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

# Validate Dockerfile syntax
print_status "Validating Dockerfile syntax..."
if docker build --dry-run -f Dockerfile . > /dev/null 2>&1; then
    print_success "Dockerfile syntax is valid"
else
    print_error "Dockerfile syntax validation failed"
    exit 1
fi

# Check required files exist
print_status "Checking required files..."
required_files=(
    "package.json"
    "nginx.prod.conf"
    "nginx-security.conf"
    ".dockerignore"
    "src/"
    "public/"
)

for file in "${required_files[@]}"; do
    if [[ -e "$file" ]]; then
        print_success "✓ $file exists"
    else
        print_error "✗ $file is missing"
        exit 1
    fi
done

# Validate nginx configuration
print_status "Validating nginx configuration..."
if docker run --rm -v "$(pwd)/nginx.prod.conf:/etc/nginx/nginx.conf:ro" nginx:1.25-alpine nginx -t > /dev/null 2>&1; then
    print_success "Nginx configuration is valid"
else
    print_warning "Nginx configuration validation failed (this might be due to missing includes)"
fi

# Check Docker build stages
print_status "Checking Docker build stages..."
stages=$(grep -E "^FROM .* AS " Dockerfile | awk '{print $4}')
expected_stages=("dependencies" "builder" "production")

for stage in "${expected_stages[@]}"; do
    if echo "$stages" | grep -q "$stage"; then
        print_success "✓ Stage '$stage' found"
    else
        print_error "✗ Stage '$stage' is missing"
        exit 1
    fi
done

# Check build optimizations
print_status "Checking build optimizations..."
optimizations=(
    "multi-stage build"
    "dependency caching"
    "gzip compression"
    "brotli compression"
    "health checks"
    "security headers"
)

if grep -q "mount=type=cache" Dockerfile; then
    print_success "✓ Dependency caching enabled"
else
    print_warning "✗ Dependency caching not found"
fi

if grep -q "gzip -k" Dockerfile; then
    print_success "✓ Gzip compression enabled"
else
    print_warning "✗ Gzip compression not found"
fi

if grep -q "brotli -k" Dockerfile; then
    print_success "✓ Brotli compression enabled"
else
    print_warning "✗ Brotli compression not found"
fi

if grep -q "HEALTHCHECK" Dockerfile; then
    print_success "✓ Health checks configured"
else
    print_error "✗ Health checks not found"
fi

# Check security configurations
print_status "Checking security configurations..."
if grep -q "USER nginx-app" Dockerfile; then
    print_success "✓ Non-root user configured"
else
    print_error "✗ Non-root user not configured"
fi

if grep -q "Content-Security-Policy" nginx-security.conf; then
    print_success "✓ Security headers configured"
else
    print_error "✗ Security headers not found"
fi

print_success "Dockerfile validation completed!"
print_status "To build the image, run: docker build -f Dockerfile -t qwen-frontend ."