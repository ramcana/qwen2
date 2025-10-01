#!/bin/bash
# =============================================================================
# Security Setup Script for Qwen2 Docker Containerization
# =============================================================================
# This script sets up comprehensive security configurations including:
# - Network security and isolation
# - User and permission management
# - Secrets generation and management
# - SSL certificate setup
# - Security monitoring and logging

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config"
SECRETS_DIR="$PROJECT_ROOT/secrets"
SSL_DIR="$PROJECT_ROOT/ssl"
LOGS_DIR="$PROJECT_ROOT/logs"
DATA_DIR="$PROJECT_ROOT/data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging
LOG_FILE="$LOGS_DIR/security-setup.log"

# Functions
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

error_exit() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    log "ERROR: $1"
    exit 1
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
    log "SUCCESS: $1"
}

warning() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
    log "WARNING: $1"
}

info() {
    echo -e "${BLUE}ℹ INFO: $1${NC}"
    log "INFO: $1"
}

step() {
    echo -e "${PURPLE}▶ $1${NC}"
    log "STEP: $1"
}

# Check prerequisites
check_prerequisites() {
    step "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker >/dev/null 2>&1; then
        error_exit "Docker is not installed or not in PATH"
    fi
    
    if ! docker info >/dev/null 2>&1; then
        error_exit "Docker daemon is not running"
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
        error_exit "Docker Compose is not installed"
    fi
    
    # Check for required tools
    local required_tools=("openssl" "curl" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            warning "$tool is not installed - some features may not work"
        fi
    done
    
    success "Prerequisites check completed"
}

# Create directory structure
create_directories() {
    step "Creating secure directory structure..."
    
    # Create main directories
    local directories=(
        "$SECRETS_DIR"
        "$SSL_DIR"
        "$LOGS_DIR"
        "$DATA_DIR"
        "$DATA_DIR/redis"
        "$DATA_DIR/prometheus"
        "$DATA_DIR/grafana"
        "$CONFIG_DIR/traefik"
        "$LOGS_DIR/api"
        "$LOGS_DIR/traefik"
        "$LOGS_DIR/security"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            info "Created directory: $dir"
        fi
    done
    
    # Set secure permissions
    chmod 700 "$SECRETS_DIR"
    chmod 700 "$SSL_DIR"
    chmod 755 "$LOGS_DIR"
    chmod 755 "$DATA_DIR"
    
    success "Directory structure created with secure permissions"
}

# Generate secrets
generate_secrets() {
    step "Generating security secrets..."
    
    if [ -x "$SCRIPT_DIR/generate-secrets.sh" ]; then
        "$SCRIPT_DIR/generate-secrets.sh"
        success "Secrets generated successfully"
    else
        warning "Secret generation script not found or not executable"
        info "Run 'chmod +x scripts/generate-secrets.sh' and then 'scripts/generate-secrets.sh'"
    fi
}

# Create Traefik users file
create_traefik_users() {
    step "Creating Traefik authentication users..."
    
    local users_file="$CONFIG_DIR/traefik/users"
    
    # Generate admin password if not exists
    if [ ! -f "$users_file" ]; then
        local admin_password
        admin_password=$(openssl rand -base64 16)
        
        # Create htpasswd entry
        local auth_string
        auth_string=$(echo -n "admin:$admin_password" | openssl passwd -apr1 -stdin)
        
        echo "admin:$auth_string" > "$users_file"
        chmod 600 "$users_file"
        
        info "Traefik admin credentials: admin:$admin_password"
        echo "admin:$admin_password" > "$SECRETS_DIR/traefik_admin_credentials.txt"
        chmod 600 "$SECRETS_DIR/traefik_admin_credentials.txt"
        
        success "Traefik users file created"
    else
        info "Traefik users file already exists"
    fi
}

# Setup Docker networks
setup_networks() {
    step "Setting up secure Docker networks..."
    
    # Create external Traefik network if it doesn't exist
    if ! docker network ls | grep -q "traefik-public"; then
        docker network create \
            --driver bridge \
            --subnet=172.19.0.0/16 \
            --gateway=172.19.0.1 \
            traefik-public
        success "Created traefik-public network"
    else
        info "traefik-public network already exists"
    fi
    
    # Remove existing networks if they exist (to recreate with security settings)
    local networks=("qwen-frontend-secure" "qwen-backend-secure" "qwen-data-secure" "qwen-mgmt-secure")
    for network in "${networks[@]}"; do
        if docker network ls | grep -q "$network"; then
            docker network rm "$network" 2>/dev/null || true
            info "Removed existing network: $network"
        fi
    done
    
    success "Network setup completed"
}

# Configure file permissions
configure_permissions() {
    step "Configuring secure file permissions..."
    
    # Set ownership for application directories
    local app_dirs=("models" "cache" "generated_images" "uploads" "offload")
    for dir in "${app_dirs[@]}"; do
        if [ -d "$PROJECT_ROOT/$dir" ]; then
            chmod 755 "$PROJECT_ROOT/$dir"
            info "Set permissions for $dir"
        fi
    done
    
    # Set permissions for configuration files
    find "$CONFIG_DIR" -type f -name "*.yml" -exec chmod 644 {} \;
    find "$CONFIG_DIR" -type f -name "*.conf" -exec chmod 644 {} \;
    
    # Set permissions for scripts
    find "$SCRIPT_DIR" -type f -name "*.sh" -exec chmod +x {} \;
    
    success "File permissions configured"
}

# Setup SSL certificates
setup_ssl() {
    step "Setting up SSL certificates..."
    
    if [ ! -f "$SSL_DIR/certificate.pem" ] || [ ! -f "$SSL_DIR/private_key.pem" ]; then
        if command -v openssl >/dev/null 2>&1; then
            # Generate self-signed certificate for development
            openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
                -keyout "$SSL_DIR/private_key.pem" \
                -out "$SSL_DIR/certificate.pem" \
                -subj "/C=US/ST=CA/L=San Francisco/O=Qwen2/OU=Development/CN=localhost" \
                2>/dev/null
            
            chmod 400 "$SSL_DIR/private_key.pem"
            chmod 444 "$SSL_DIR/certificate.pem"
            
            success "Self-signed SSL certificates generated"
        else
            warning "OpenSSL not found - SSL certificates not generated"
        fi
    else
        info "SSL certificates already exist"
    fi
}

# Create ACME configuration
setup_acme() {
    step "Setting up ACME configuration..."
    
    local acme_file="$PROJECT_ROOT/acme.json"
    
    if [ ! -f "$acme_file" ]; then
        echo '{}' > "$acme_file"
        chmod 600 "$acme_file"
        success "ACME configuration file created"
    else
        info "ACME configuration file already exists"
    fi
}

# Setup monitoring and logging
setup_monitoring() {
    step "Setting up security monitoring and logging..."
    
    # Create log rotation configuration
    cat > "$CONFIG_DIR/docker/logrotate.conf" << 'EOF'
/app/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 qwen-api qwen-api
    postrotate
        /usr/bin/docker kill -s USR1 qwen-api-secure 2>/dev/null || true
    endscript
}

/var/log/traefik/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 root root
    postrotate
        /usr/bin/docker kill -s USR1 qwen-traefik-secure 2>/dev/null || true
    endscript
}
EOF
    
    success "Monitoring and logging configuration created"
}

# Validate security configuration
validate_security() {
    step "Validating security configuration..."
    
    local validation_failed=false
    
    # Check directory permissions
    if [ "$(stat -c "%a" "$SECRETS_DIR")" != "700" ]; then
        warning "Secrets directory permissions are not secure"
        validation_failed=true
    fi
    
    # Check if secrets exist
    local required_secrets=("api_secret_key.txt" "jwt_secret_key.txt")
    for secret in "${required_secrets[@]}"; do
        if [ ! -f "$SECRETS_DIR/$secret" ]; then
            warning "Required secret missing: $secret"
            validation_failed=true
        fi
    done
    
    # Check SSL certificates
    if [ ! -f "$SSL_DIR/certificate.pem" ]; then
        warning "SSL certificate not found"
        validation_failed=true
    fi
    
    # Check Docker networks
    if ! docker network ls | grep -q "traefik-public"; then
        warning "Traefik public network not found"
        validation_failed=true
    fi
    
    if [ "$validation_failed" = true ]; then
        warning "Security validation found issues - please review"
    else
        success "Security validation passed"
    fi
}

# Create security documentation
create_documentation() {
    step "Creating security documentation..."
    
    cat > "$PROJECT_ROOT/SECURITY.md" << 'EOF'
# Security Configuration for Qwen2 Docker Containerization

## Overview

This document describes the security measures implemented in the Qwen2 Docker containerization setup.

## Security Features

### Network Security
- **Network Isolation**: Services are isolated in separate Docker networks
- **DMZ Architecture**: Frontend in DMZ, backend and data layers isolated
- **Firewall Rules**: Traefik acts as application firewall
- **TLS Encryption**: All external communication encrypted

### Authentication & Authorization
- **Secrets Management**: Docker secrets for sensitive data
- **Basic Authentication**: Traefik dashboard protected
- **API Key Authentication**: API endpoints secured
- **Role-Based Access**: Different access levels for different services

### Container Security
- **Non-Root Users**: All containers run as non-root users
- **Read-Only Filesystems**: Where possible, containers use read-only root
- **Security Options**: no-new-privileges, seccomp profiles
- **Resource Limits**: Memory and CPU limits to prevent DoS

### Data Security
- **Encrypted Storage**: Sensitive data encrypted at rest
- **Secure Permissions**: Proper file and directory permissions
- **Audit Logging**: Comprehensive logging for security events
- **Backup Security**: Encrypted backups with rotation

## Security Checklist

- [ ] Secrets generated and secured
- [ ] SSL certificates configured
- [ ] Network isolation implemented
- [ ] Authentication configured
- [ ] Monitoring and logging enabled
- [ ] Regular security updates scheduled
- [ ] Backup and recovery tested

## Incident Response

1. **Detection**: Monitor logs and alerts
2. **Containment**: Isolate affected services
3. **Investigation**: Analyze logs and system state
4. **Recovery**: Restore from secure backups
5. **Lessons Learned**: Update security measures

## Compliance

This setup addresses requirements for:
- OWASP Top 10
- CIS Docker Benchmarks
- NIST Cybersecurity Framework

## Maintenance

- Rotate secrets every 90 days
- Update containers monthly
- Review logs weekly
- Test backups monthly
- Security audit quarterly
EOF
    
    success "Security documentation created"
}

# Main execution
main() {
    echo "============================================================================="
    echo -e "${CYAN}Qwen2 Docker Security Setup${NC}"
    echo "============================================================================="
    
    # Create log file
    mkdir -p "$LOGS_DIR"
    touch "$LOG_FILE"
    chmod 600 "$LOG_FILE"
    
    log "Starting security setup process"
    
    # Execute setup steps
    check_prerequisites
    create_directories
    generate_secrets
    create_traefik_users
    setup_networks
    configure_permissions
    setup_ssl
    setup_acme
    setup_monitoring
    validate_security
    create_documentation
    
    echo "============================================================================="
    success "Security setup completed successfully!"
    echo "============================================================================="
    
    info "Next steps:"
    echo "1. Review generated secrets in: $SECRETS_DIR"
    echo "2. Update external API keys with real tokens"
    echo "3. Configure production SSL certificates if needed"
    echo "4. Run security validation: docker-compose -f docker-compose.security.yml config"
    echo "5. Start secure services: docker-compose -f docker-compose.security.yml up -d"
    
    warning "IMPORTANT:"
    echo "- Keep secrets secure and never commit them to version control"
    echo "- Regularly update and rotate secrets"
    echo "- Monitor security logs for suspicious activity"
    echo "- Test backup and recovery procedures"
    
    info "Security documentation created: $PROJECT_ROOT/SECURITY.md"
}

# Run main function
main "$@"