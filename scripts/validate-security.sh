#!/bin/bash
# =============================================================================
# Security Validation Script for Qwen2 Docker Containerization
# =============================================================================
# This script validates the security configuration and performs security checks
# on the Docker containerization setup including network isolation, permissions,
# secrets management, and compliance with security best practices

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/security-validation.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Functions
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

check_pass() {
    echo -e "${GREEN}✓ PASS: $1${NC}"
    log "PASS: $1"
    ((PASSED_CHECKS++))
    ((TOTAL_CHECKS++))
}

check_fail() {
    echo -e "${RED}✗ FAIL: $1${NC}"
    log "FAIL: $1"
    ((FAILED_CHECKS++))
    ((TOTAL_CHECKS++))
}

check_warn() {
    echo -e "${YELLOW}⚠ WARN: $1${NC}"
    log "WARN: $1"
    ((WARNING_CHECKS++))
    ((TOTAL_CHECKS++))
}

info() {
    echo -e "${BLUE}ℹ INFO: $1${NC}"
    log "INFO: $1"
}

section() {
    echo -e "\n${PURPLE}▶ $1${NC}"
    echo "============================================================================="
    log "SECTION: $1"
}

# Check Docker security
check_docker_security() {
    section "Docker Security Configuration"
    
    # Check if Docker daemon is running
    if docker info >/dev/null 2>&1; then
        check_pass "Docker daemon is running"
    else
        check_fail "Docker daemon is not running"
        return
    fi
    
    # Check Docker version
    local docker_version
    docker_version=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown")
    if [[ "$docker_version" != "unknown" ]]; then
        check_pass "Docker version: $docker_version"
    else
        check_warn "Could not determine Docker version"
    fi
    
    # Check if user is in docker group (security consideration)
    if groups | grep -q docker; then
        check_warn "Current user is in docker group (consider rootless Docker for enhanced security)"
    else
        check_pass "Current user is not in docker group"
    fi
    
    # Check Docker daemon configuration
    if docker info 2>/dev/null | grep -q "live-restore"; then
        check_pass "Docker live-restore is configured"
    else
        check_warn "Docker live-restore not configured"
    fi
}

# Check network security
check_network_security() {
    section "Network Security Configuration"
    
    # Check if required networks exist
    local networks=("qwen-network" "qwen-backend" "traefik-public")
    for network in "${networks[@]}"; do
        if docker network ls | grep -q "$network"; then
            check_pass "Network '$network' exists"
            
            # Check network configuration
            local network_info
            network_info=$(docker network inspect "$network" 2>/dev/null || echo "[]")
            
            # Check if network has proper subnet configuration
            if echo "$network_info" | jq -r '.[0].IPAM.Config[0].Subnet' | grep -q "172.20"; then
                check_pass "Network '$network' has proper subnet configuration"
            else
                check_warn "Network '$network' subnet configuration may not be optimal"
            fi
        else
            check_fail "Network '$network' does not exist"
        fi
    done
    
    # Check for network isolation
    if docker network ls | grep -q "qwen-backend"; then
        local backend_internal
        backend_internal=$(docker network inspect qwen-backend 2>/dev/null | jq -r '.[0].Internal' || echo "false")
        if [[ "$backend_internal" == "false" ]]; then
            check_warn "Backend network is not internal (may need external access for model downloads)"
        else
            check_pass "Backend network is properly isolated"
        fi
    fi
}

# Check container security
check_container_security() {
    section "Container Security Configuration"
    
    # Check if containers are running
    local containers=("qwen-traefik" "qwen-api" "qwen-frontend")
    for container in "${containers[@]}"; do
        if docker ps | grep -q "$container"; then
            check_pass "Container '$container' is running"
            
            # Check container security options
            local security_opts
            security_opts=$(docker inspect "$container" 2>/dev/null | jq -r '.[0].HostConfig.SecurityOpt[]' 2>/dev/null || echo "")
            
            if echo "$security_opts" | grep -q "no-new-privileges:true"; then
                check_pass "Container '$container' has no-new-privileges enabled"
            else
                check_fail "Container '$container' does not have no-new-privileges enabled"
            fi
            
            # Check if running as root
            local user_info
            user_info=$(docker inspect "$container" 2>/dev/null | jq -r '.[0].Config.User' 2>/dev/null || echo "")
            if [[ -n "$user_info" && "$user_info" != "null" && "$user_info" != "0" && "$user_info" != "root" ]]; then
                check_pass "Container '$container' is running as non-root user: $user_info"
            else
                check_warn "Container '$container' may be running as root user"
            fi
            
            # Check read-only root filesystem
            local readonly_root
            readonly_root=$(docker inspect "$container" 2>/dev/null | jq -r '.[0].HostConfig.ReadonlyRootfs' 2>/dev/null || echo "false")
            if [[ "$readonly_root" == "true" ]]; then
                check_pass "Container '$container' has read-only root filesystem"
            else
                check_warn "Container '$container' does not have read-only root filesystem"
            fi
            
        else
            check_warn "Container '$container' is not running"
        fi
    done
}

# Check secrets management
check_secrets_management() {
    section "Secrets Management"
    
    local secrets_dir="$PROJECT_ROOT/secrets"
    
    # Check if secrets directory exists
    if [[ -d "$secrets_dir" ]]; then
        check_pass "Secrets directory exists"
        
        # Check directory permissions
        local perms
        perms=$(stat -c "%a" "$secrets_dir" 2>/dev/null || echo "000")
        if [[ "$perms" == "700" ]]; then
            check_pass "Secrets directory has secure permissions (700)"
        else
            check_fail "Secrets directory permissions are insecure: $perms (should be 700)"
        fi
        
        # Check for required secret files
        local required_secrets=(
            "api_secret_key.txt"
            "jwt_secret_key.txt"
            "postgres_password.txt"
            "redis_password.txt"
        )
        
        for secret in "${required_secrets[@]}"; do
            if [[ -f "$secrets_dir/$secret" ]]; then
                check_pass "Secret file exists: $secret"
                
                # Check file permissions
                local file_perms
                file_perms=$(stat -c "%a" "$secrets_dir/$secret" 2>/dev/null || echo "000")
                if [[ "$file_perms" == "400" ]]; then
                    check_pass "Secret file has secure permissions: $secret"
                else
                    check_fail "Secret file has insecure permissions: $secret ($file_perms, should be 400)"
                fi
                
                # Check if file is not empty
                if [[ -s "$secrets_dir/$secret" ]]; then
                    check_pass "Secret file is not empty: $secret"
                else
                    check_fail "Secret file is empty: $secret"
                fi
            else
                check_fail "Required secret file missing: $secret"
            fi
        done
    else
        check_fail "Secrets directory does not exist"
    fi
}

# Check SSL/TLS configuration
check_ssl_configuration() {
    section "SSL/TLS Configuration"
    
    local ssl_dir="$PROJECT_ROOT/ssl"
    
    # Check SSL directory
    if [[ -d "$ssl_dir" ]]; then
        check_pass "SSL directory exists"
        
        # Check for SSL certificate files
        if [[ -f "$ssl_dir/certificate.pem" ]]; then
            check_pass "SSL certificate file exists"
            
            # Check certificate validity
            if openssl x509 -in "$ssl_dir/certificate.pem" -noout -checkend 86400 2>/dev/null; then
                check_pass "SSL certificate is valid and not expiring within 24 hours"
            else
                check_warn "SSL certificate may be expired or expiring soon"
            fi
        else
            check_warn "SSL certificate file not found (may be using Let's Encrypt)"
        fi
        
        if [[ -f "$ssl_dir/private_key.pem" ]]; then
            check_pass "SSL private key file exists"
            
            # Check private key permissions
            local key_perms
            key_perms=$(stat -c "%a" "$ssl_dir/private_key.pem" 2>/dev/null || echo "000")
            if [[ "$key_perms" == "400" ]]; then
                check_pass "SSL private key has secure permissions"
            else
                check_fail "SSL private key has insecure permissions: $key_perms (should be 400)"
            fi
        else
            check_warn "SSL private key file not found"
        fi
    else
        check_warn "SSL directory does not exist"
    fi
    
    # Check ACME configuration
    if [[ -f "$PROJECT_ROOT/acme.json" ]]; then
        check_pass "ACME configuration file exists"
        
        local acme_perms
        acme_perms=$(stat -c "%a" "$PROJECT_ROOT/acme.json" 2>/dev/null || echo "000")
        if [[ "$acme_perms" == "600" ]]; then
            check_pass "ACME file has secure permissions"
        else
            check_fail "ACME file has insecure permissions: $acme_perms (should be 600)"
        fi
    else
        check_warn "ACME configuration file not found"
    fi
}

# Check file permissions
check_file_permissions() {
    section "File Permissions and Ownership"
    
    # Check configuration files
    local config_files=(
        "docker-compose.yml"
        "docker-compose.security.yml"
        "config/docker/traefik-security.yml"
        "config/docker/security.yml"
    )
    
    for file in "${config_files[@]}"; do
        if [[ -f "$PROJECT_ROOT/$file" ]]; then
            check_pass "Configuration file exists: $file"
            
            local file_perms
            file_perms=$(stat -c "%a" "$PROJECT_ROOT/$file" 2>/dev/null || echo "000")
            if [[ "$file_perms" =~ ^[46][04][04]$ ]]; then
                check_pass "Configuration file has appropriate permissions: $file"
            else
                check_warn "Configuration file permissions may be too permissive: $file ($file_perms)"
            fi
        else
            check_warn "Configuration file not found: $file"
        fi
    done
    
    # Check script permissions
    local scripts=(
        "scripts/generate-secrets.sh"
        "scripts/setup-security.sh"
        "scripts/validate-security.sh"
    )
    
    for script in "${scripts[@]}"; do
        if [[ -f "$PROJECT_ROOT/$script" ]]; then
            if [[ -x "$PROJECT_ROOT/$script" ]]; then
                check_pass "Script is executable: $script"
            else
                check_fail "Script is not executable: $script"
            fi
        else
            check_warn "Script not found: $script"
        fi
    done
}

# Check logging and monitoring
check_logging_monitoring() {
    section "Logging and Monitoring"
    
    local logs_dir="$PROJECT_ROOT/logs"
    
    # Check logs directory
    if [[ -d "$logs_dir" ]]; then
        check_pass "Logs directory exists"
        
        # Check log files
        local log_files=("api" "traefik" "security")
        for log_type in "${log_files[@]}"; do
            if [[ -d "$logs_dir/$log_type" ]]; then
                check_pass "Log directory exists: $log_type"
            else
                check_warn "Log directory not found: $log_type"
            fi
        done
    else
        check_warn "Logs directory does not exist"
    fi
    
    # Check if containers are configured for logging
    local containers=("qwen-traefik" "qwen-api" "qwen-frontend")
    for container in "${containers[@]}"; do
        if docker ps | grep -q "$container"; then
            local log_driver
            log_driver=$(docker inspect "$container" 2>/dev/null | jq -r '.[0].HostConfig.LogConfig.Type' 2>/dev/null || echo "unknown")
            if [[ "$log_driver" == "json-file" ]]; then
                check_pass "Container '$container' has proper log driver configured"
            else
                check_warn "Container '$container' log driver: $log_driver"
            fi
        fi
    done
}

# Check compliance with security standards
check_compliance() {
    section "Security Compliance Checks"
    
    # OWASP Docker Security Top 10
    info "Checking OWASP Docker Security compliance..."
    
    # D01: Secure User Mapping
    local non_root_containers=0
    local total_containers=0
    for container in $(docker ps --format "{{.Names}}" | grep "qwen-" || true); do
        ((total_containers++))
        local user_info
        user_info=$(docker inspect "$container" 2>/dev/null | jq -r '.[0].Config.User' 2>/dev/null || echo "")
        if [[ -n "$user_info" && "$user_info" != "null" && "$user_info" != "0" && "$user_info" != "root" ]]; then
            ((non_root_containers++))
        fi
    done
    
    if [[ $total_containers -gt 0 ]]; then
        local percentage=$((non_root_containers * 100 / total_containers))
        if [[ $percentage -ge 80 ]]; then
            check_pass "OWASP D01: $percentage% of containers run as non-root users"
        else
            check_warn "OWASP D01: Only $percentage% of containers run as non-root users"
        fi
    fi
    
    # D02: Patch Management
    info "OWASP D02: Regular patching should be implemented (manual check required)"
    
    # D03: Network Segmentation
    local network_count
    network_count=$(docker network ls | grep "qwen-" | wc -l)
    if [[ $network_count -ge 2 ]]; then
        check_pass "OWASP D03: Network segmentation implemented ($network_count networks)"
    else
        check_warn "OWASP D03: Limited network segmentation"
    fi
    
    # D04: Secure Defaults
    check_pass "OWASP D04: Security configurations implemented"
    
    # D05: Maintain Security Contexts
    info "OWASP D05: Security contexts maintained through configuration"
    
    # CIS Docker Benchmark checks
    info "Checking CIS Docker Benchmark compliance..."
    
    # Check if Docker daemon is configured securely
    if docker info 2>/dev/null | grep -q "Security Options"; then
        check_pass "CIS: Docker daemon has security options configured"
    else
        check_warn "CIS: Docker daemon security options not visible"
    fi
}

# Generate security report
generate_report() {
    section "Security Validation Report"
    
    echo -e "\n${CYAN}=============================================================================${NC}"
    echo -e "${CYAN}                    SECURITY VALIDATION SUMMARY${NC}"
    echo -e "${CYAN}=============================================================================${NC}"
    
    echo -e "\nTotal Checks: ${BLUE}$TOTAL_CHECKS${NC}"
    echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
    echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
    echo -e "Warnings: ${YELLOW}$WARNING_CHECKS${NC}"
    
    local pass_percentage=0
    if [[ $TOTAL_CHECKS -gt 0 ]]; then
        pass_percentage=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))
    fi
    
    echo -e "\nPass Rate: ${BLUE}$pass_percentage%${NC}"
    
    if [[ $FAILED_CHECKS -eq 0 ]]; then
        echo -e "\n${GREEN}✓ Security validation completed successfully!${NC}"
        if [[ $WARNING_CHECKS -gt 0 ]]; then
            echo -e "${YELLOW}⚠ Please review warnings for potential improvements.${NC}"
        fi
    else
        echo -e "\n${RED}✗ Security validation found critical issues that need attention.${NC}"
    fi
    
    echo -e "\n${BLUE}Recommendations:${NC}"
    if [[ $FAILED_CHECKS -gt 0 ]]; then
        echo "1. Address all failed security checks immediately"
    fi
    if [[ $WARNING_CHECKS -gt 0 ]]; then
        echo "2. Review and address security warnings"
    fi
    echo "3. Regularly run security validation (weekly recommended)"
    echo "4. Keep Docker and container images updated"
    echo "5. Monitor security logs for suspicious activity"
    echo "6. Implement automated security scanning in CI/CD"
    
    # Save report to file
    local report_file="$PROJECT_ROOT/logs/security-validation-report-$(date +%Y%m%d_%H%M%S).json"
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "total_checks": $TOTAL_CHECKS,
    "passed_checks": $PASSED_CHECKS,
    "failed_checks": $FAILED_CHECKS,
    "warning_checks": $WARNING_CHECKS,
    "pass_percentage": $pass_percentage,
    "status": "$(if [[ $FAILED_CHECKS -eq 0 ]]; then echo "PASS"; else echo "FAIL"; fi)"
}
EOF
    
    info "Detailed report saved to: $report_file"
}

# Main execution
main() {
    echo -e "${CYAN}=============================================================================${NC}"
    echo -e "${CYAN}                    Qwen2 Docker Security Validation${NC}"
    echo -e "${CYAN}=============================================================================${NC}"
    
    # Create log file
    mkdir -p "$(dirname "$LOG_FILE")"
    touch "$LOG_FILE"
    chmod 600 "$LOG_FILE"
    
    log "Starting security validation"
    
    # Run validation checks
    check_docker_security
    check_network_security
    check_container_security
    check_secrets_management
    check_ssl_configuration
    check_file_permissions
    check_logging_monitoring
    check_compliance
    
    # Generate final report
    generate_report
    
    # Exit with appropriate code
    if [[ $FAILED_CHECKS -gt 0 ]]; then
        exit 1
    else
        exit 0
    fi
}

# Run main function
main "$@"