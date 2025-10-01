#!/bin/bash
# =============================================================================
# Secret Generation Script for Qwen2 Docker Containerization
# =============================================================================
# This script generates secure secrets for the Qwen2 application including
# passwords, API keys, JWT tokens, and other sensitive configuration data

set -euo pipefail

# Configuration
SECRETS_DIR="./secrets"
SSL_DIR="./ssl"
BACKUP_DIR="./secrets/backup"
LOG_FILE="./logs/secret-generation.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    log "ERROR: $1"
    exit 1
}

# Success message
success() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
    log "SUCCESS: $1"
}

# Warning message
warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
    log "WARNING: $1"
}

# Info message
info() {
    echo -e "${BLUE}INFO: $1${NC}"
    log "INFO: $1"
}

# Generate random password
generate_password() {
    local length=${1:-32}
    local charset=${2:-'A-Za-z0-9!@#$%^&*()_+-=[]{}|;:,.<>?'}
    
    # Use openssl for secure random generation
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -base64 $((length * 3 / 4)) | tr -d "=+/" | cut -c1-${length}
    else
        # Fallback to /dev/urandom
        tr -dc "$charset" < /dev/urandom | head -c${length}
    fi
}

# Generate API key
generate_api_key() {
    local length=${1:-64}
    openssl rand -hex $((length / 2))
}

# Generate JWT secret
generate_jwt_secret() {
    local length=${1:-128}
    openssl rand -base64 $((length * 3 / 4)) | tr -d "=+/" | cut -c1-${length}
}

# Create directories
create_directories() {
    info "Creating directories..."
    
    mkdir -p "$SECRETS_DIR"
    mkdir -p "$SSL_DIR"
    mkdir -p "$BACKUP_DIR"
    mkdir -p "./logs"
    
    # Set proper permissions
    chmod 700 "$SECRETS_DIR"
    chmod 700 "$SSL_DIR"
    chmod 700 "$BACKUP_DIR"
    
    success "Directories created successfully"
}

# Backup existing secrets
backup_secrets() {
    if [ -d "$SECRETS_DIR" ] && [ "$(ls -A $SECRETS_DIR)" ]; then
        info "Backing up existing secrets..."
        
        local backup_timestamp=$(date +%Y%m%d_%H%M%S)
        local backup_path="$BACKUP_DIR/secrets_backup_$backup_timestamp"
        
        mkdir -p "$backup_path"
        cp -r "$SECRETS_DIR"/* "$backup_path/" 2>/dev/null || true
        
        success "Secrets backed up to $backup_path"
    fi
}

# Generate database secrets
generate_database_secrets() {
    info "Generating database secrets..."
    
    # PostgreSQL credentials
    echo "qwen_user" > "$SECRETS_DIR/postgres_user.txt"
    generate_password 32 > "$SECRETS_DIR/postgres_password.txt"
    
    # Redis password
    generate_password 32 > "$SECRETS_DIR/redis_password.txt"
    
    # Set permissions
    chmod 400 "$SECRETS_DIR/postgres_user.txt"
    chmod 400 "$SECRETS_DIR/postgres_password.txt"
    chmod 400 "$SECRETS_DIR/redis_password.txt"
    
    success "Database secrets generated"
}

# Generate API secrets
generate_api_secrets() {
    info "Generating API secrets..."
    
    # API secret key
    generate_api_key 64 > "$SECRETS_DIR/api_secret_key.txt"
    
    # JWT secret
    generate_jwt_secret 128 > "$SECRETS_DIR/jwt_secret_key.txt"
    
    # Set permissions
    chmod 400 "$SECRETS_DIR/api_secret_key.txt"
    chmod 400 "$SECRETS_DIR/jwt_secret_key.txt"
    
    success "API secrets generated"
}

# Generate external API keys (placeholders)
generate_external_api_keys() {
    info "Generating external API key placeholders..."
    
    # HuggingFace token (placeholder)
    echo "hf_placeholder_token_replace_with_real_token" > "$SECRETS_DIR/huggingface_token.txt"
    
    # OpenAI API key (placeholder)
    echo "sk-placeholder_key_replace_with_real_key" > "$SECRETS_DIR/openai_api_key.txt"
    
    # Set permissions
    chmod 400 "$SECRETS_DIR/huggingface_token.txt"
    chmod 400 "$SECRETS_DIR/openai_api_key.txt"
    
    warning "External API keys are placeholders - replace with real tokens"
}

# Generate authentication secrets
generate_auth_secrets() {
    info "Generating authentication secrets..."
    
    # Traefik dashboard auth (admin:admin by default)
    local admin_password=$(generate_password 16)
    local auth_string=$(echo -n "admin:$admin_password" | openssl passwd -apr1 -stdin)
    echo "admin:$auth_string" > "$SECRETS_DIR/traefik_dashboard_auth.txt"
    
    # Prometheus auth
    local prom_password=$(generate_password 16)
    local prom_auth=$(echo -n "prometheus:$prom_password" | openssl passwd -apr1 -stdin)
    echo "prometheus:$prom_auth" > "$SECRETS_DIR/prometheus_auth.txt"
    
    # Grafana admin password
    generate_password 16 > "$SECRETS_DIR/grafana_admin_password.txt"
    
    # Set permissions
    chmod 400 "$SECRETS_DIR/traefik_dashboard_auth.txt"
    chmod 400 "$SECRETS_DIR/prometheus_auth.txt"
    chmod 400 "$SECRETS_DIR/grafana_admin_password.txt"
    
    success "Authentication secrets generated"
    info "Traefik dashboard - admin:$admin_password"
    info "Prometheus - prometheus:$prom_password"
}

# Generate SSL certificates (self-signed for development)
generate_ssl_certificates() {
    info "Generating SSL certificates..."
    
    # Check if openssl is available
    if ! command -v openssl >/dev/null 2>&1; then
        warning "OpenSSL not found, skipping SSL certificate generation"
        return
    fi
    
    # Generate private key
    openssl genrsa -out "$SSL_DIR/private_key.pem" 2048
    
    # Generate certificate signing request
    openssl req -new -key "$SSL_DIR/private_key.pem" -out "$SSL_DIR/certificate.csr" \
        -subj "/C=US/ST=CA/L=San Francisco/O=Qwen2/OU=Development/CN=localhost"
    
    # Generate self-signed certificate
    openssl x509 -req -days 365 -in "$SSL_DIR/certificate.csr" \
        -signkey "$SSL_DIR/private_key.pem" -out "$SSL_DIR/certificate.pem"
    
    # Clean up CSR
    rm "$SSL_DIR/certificate.csr"
    
    # Set permissions
    chmod 400 "$SSL_DIR/private_key.pem"
    chmod 444 "$SSL_DIR/certificate.pem"
    
    success "SSL certificates generated (self-signed for development)"
}

# Validate generated secrets
validate_secrets() {
    info "Validating generated secrets..."
    
    local validation_failed=false
    
    # Check if all required secret files exist
    local required_secrets=(
        "postgres_user.txt"
        "postgres_password.txt"
        "redis_password.txt"
        "api_secret_key.txt"
        "jwt_secret_key.txt"
        "huggingface_token.txt"
        "openai_api_key.txt"
        "traefik_dashboard_auth.txt"
        "prometheus_auth.txt"
        "grafana_admin_password.txt"
    )
    
    for secret in "${required_secrets[@]}"; do
        if [ ! -f "$SECRETS_DIR/$secret" ]; then
            error_exit "Required secret file missing: $secret"
            validation_failed=true
        fi
        
        # Check file permissions
        local perms=$(stat -c "%a" "$SECRETS_DIR/$secret" 2>/dev/null || echo "000")
        if [ "$perms" != "400" ]; then
            warning "Incorrect permissions for $secret: $perms (should be 400)"
        fi
        
        # Check file is not empty
        if [ ! -s "$SECRETS_DIR/$secret" ]; then
            error_exit "Secret file is empty: $secret"
            validation_failed=true
        fi
    done
    
    if [ "$validation_failed" = true ]; then
        error_exit "Secret validation failed"
    fi
    
    success "All secrets validated successfully"
}

# Generate environment file with secret references
generate_env_file() {
    info "Generating environment file with secret references..."
    
    cat > ".env.secrets" << EOF
# =============================================================================
# Environment Variables for Docker Secrets
# =============================================================================
# This file contains environment variable definitions that reference
# Docker secrets for secure configuration management

# Database Configuration
POSTGRES_USER_FILE=/run/secrets/postgres_user
POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
REDIS_PASSWORD_FILE=/run/secrets/redis_password

# API Configuration
API_SECRET_KEY_FILE=/run/secrets/api_secret_key
JWT_SECRET_KEY_FILE=/run/secrets/jwt_secret_key

# External API Keys
HUGGINGFACE_TOKEN_FILE=/run/secrets/huggingface_token
OPENAI_API_KEY_FILE=/run/secrets/openai_api_key

# Authentication
TRAEFIK_DASHBOARD_AUTH_FILE=/run/secrets/traefik_dashboard_auth
PROMETHEUS_AUTH_FILE=/run/secrets/prometheus_auth
GRAFANA_ADMIN_PASSWORD_FILE=/run/secrets/grafana_admin_password

# SSL Configuration
SSL_CERTIFICATE_FILE=/run/secrets/ssl_certificate
SSL_PRIVATE_KEY_FILE=/run/secrets/ssl_private_key

# Security Settings
ENABLE_SECRETS_MANAGEMENT=true
SECRETS_ROTATION_ENABLED=true
AUDIT_SECRETS_ACCESS=true
EOF
    
    chmod 600 ".env.secrets"
    success "Environment file generated: .env.secrets"
}

# Main function
main() {
    echo "============================================================================="
    echo "Qwen2 Docker Secrets Generation Script"
    echo "============================================================================="
    
    # Check if running as root (not recommended)
    if [ "$EUID" -eq 0 ]; then
        warning "Running as root is not recommended for security reasons"
    fi
    
    # Create log file
    touch "$LOG_FILE"
    chmod 600 "$LOG_FILE"
    
    log "Starting secret generation process"
    
    # Execute generation steps
    create_directories
    backup_secrets
    generate_database_secrets
    generate_api_secrets
    generate_external_api_keys
    generate_auth_secrets
    generate_ssl_certificates
    validate_secrets
    generate_env_file
    
    echo "============================================================================="
    success "Secret generation completed successfully!"
    echo "============================================================================="
    
    info "Next steps:"
    echo "1. Review generated secrets in: $SECRETS_DIR"
    echo "2. Replace placeholder API keys with real tokens"
    echo "3. Update .env.secrets file as needed"
    echo "4. Run 'docker-compose up' to start services with secrets"
    
    warning "IMPORTANT: Keep secrets secure and never commit them to version control!"
}

# Run main function
main "$@"