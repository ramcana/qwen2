#!/bin/bash
# =============================================================================
# Test Script for Resource Management Implementation
# =============================================================================
# This script tests all resource management components to ensure they work
# correctly and meet the requirements.
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_ROOT}/logs/resource-management-test.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
    
    case $level in
        "ERROR") echo -e "${RED}[ERROR]${NC} $message" >&2 ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} $message" ;;
        "INFO")  echo -e "${GREEN}[INFO]${NC} $message" ;;
        "DEBUG") echo -e "${BLUE}[DEBUG]${NC} $message" ;;
    esac
}

# Test function wrapper
run_test() {
    local test_name=$1
    local test_function=$2
    
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    
    log "INFO" "Running test: $test_name"
    
    if $test_function; then
        log "INFO" "✅ PASS: $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        log "ERROR" "❌ FAIL: $test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Test 1: Check if resource cleanup script exists and is executable
test_cleanup_script_exists() {
    local script_path="${PROJECT_ROOT}/scripts/resource-cleanup.sh"
    
    if [[ -f "$script_path" && -x "$script_path" ]]; then
        return 0
    else
        log "ERROR" "Resource cleanup script not found or not executable: $script_path"
        return 1
    fi
}

# Test 2: Check if GPU memory manager script exists and is executable
test_gpu_manager_script_exists() {
    local script_path="${PROJECT_ROOT}/scripts/gpu-memory-manager.sh"
    
    if [[ -f "$script_path" && -x "$script_path" ]]; then
        return 0
    else
        log "ERROR" "GPU memory manager script not found or not executable: $script_path"
        return 1
    fi
}

# Test 3: Check if disk space monitor script exists and is executable
test_disk_monitor_script_exists() {
    local script_path="${PROJECT_ROOT}/scripts/disk-space-monitor.sh"
    
    if [[ -f "$script_path" && -x "$script_path" ]]; then
        return 0
    else
        log "ERROR" "Disk space monitor script not found or not executable: $script_path"
        return 1
    fi
}

# Test 4: Test resource cleanup script dry run
test_cleanup_script_dry_run() {
    local script_path="${PROJECT_ROOT}/scripts/resource-cleanup.sh"
    
    if "$script_path" --dry-run &> /dev/null; then
        return 0
    else
        log "ERROR" "Resource cleanup script dry run failed"
        return 1
    fi
}

# Test 5: Test GPU memory manager status (if GPU available)
test_gpu_manager_status() {
    local script_path="${PROJECT_ROOT}/scripts/gpu-memory-manager.sh"
    
    # Skip if no GPU available
    if ! command -v nvidia-smi &> /dev/null; then
        log "INFO" "No GPU available, skipping GPU manager test"
        return 0
    fi
    
    if "$script_path" status &> /dev/null; then
        return 0
    else
        log "ERROR" "GPU memory manager status check failed"
        return 1
    fi
}

# Test 6: Test disk space monitor check
test_disk_monitor_check() {
    local script_path="${PROJECT_ROOT}/scripts/disk-space-monitor.sh"
    
    if "$script_path" check &> /dev/null; then
        return 0
    else
        log "ERROR" "Disk space monitor check failed"
        return 1
    fi
}

# Test 7: Check Docker Compose resource-optimized configuration
test_resource_optimized_compose() {
    local base_compose="${PROJECT_ROOT}/docker-compose.yml"
    local resource_compose="${PROJECT_ROOT}/docker-compose.resource-optimized.yml"
    
    if [[ -f "$resource_compose" ]]; then
        # Validate YAML syntax with base file (since it extends base)
        if docker-compose -f "$base_compose" -f "$resource_compose" config &> /dev/null; then
            return 0
        else
            log "ERROR" "Resource-optimized compose file has invalid syntax when combined with base"
            return 1
        fi
    else
        log "ERROR" "Resource-optimized compose file not found: $resource_compose"
        return 1
    fi
}

# Test 8: Check log rotation configuration
test_logrotate_config() {
    local config_file="${PROJECT_ROOT}/config/docker/logrotate.conf"
    
    if [[ -f "$config_file" ]]; then
        # Basic syntax check
        if grep -q "compress" "$config_file" && grep -q "rotate" "$config_file"; then
            return 0
        else
            log "ERROR" "Log rotation configuration appears incomplete"
            return 1
        fi
    else
        log "ERROR" "Log rotation configuration not found: $config_file"
        return 1
    fi
}

# Test 9: Check deployment script exists and is executable
test_deployment_script_exists() {
    local script_path="${PROJECT_ROOT}/scripts/deploy-with-resource-management.sh"
    
    if [[ -f "$script_path" && -x "$script_path" ]]; then
        return 0
    else
        log "ERROR" "Deployment script not found or not executable: $script_path"
        return 1
    fi
}

# Test 10: Check Docker Compose files have resource limits
test_compose_resource_limits() {
    local compose_file="${PROJECT_ROOT}/docker-compose.yml"
    
    if [[ -f "$compose_file" ]]; then
        # Check if resource limits are defined
        if grep -q "resources:" "$compose_file" && grep -q "limits:" "$compose_file"; then
            return 0
        else
            log "ERROR" "Docker Compose file missing resource limits"
            return 1
        fi
    else
        log "ERROR" "Main Docker Compose file not found: $compose_file"
        return 1
    fi
}

# Test 11: Check if required directories exist
test_required_directories() {
    local directories=(
        "logs"
        "cache"
        "generated_images"
        "uploads"
        "config/docker"
        "scripts"
    )
    
    for dir in "${directories[@]}"; do
        local full_path="${PROJECT_ROOT}/${dir}"
        if [[ ! -d "$full_path" ]]; then
            log "ERROR" "Required directory not found: $dir"
            return 1
        fi
    done
    
    return 0
}

# Test 12: Test resource cleanup script help
test_cleanup_script_help() {
    local script_path="${PROJECT_ROOT}/scripts/resource-cleanup.sh"
    
    if "$script_path" --help &> /dev/null; then
        return 0
    else
        log "ERROR" "Resource cleanup script help option failed"
        return 1
    fi
}

# Test 13: Check if Docker Compose has GPU configuration
test_gpu_configuration() {
    local compose_file="${PROJECT_ROOT}/docker-compose.yml"
    
    if [[ -f "$compose_file" ]]; then
        # Check if GPU configuration is present
        if grep -q "nvidia" "$compose_file" && grep -q "gpu" "$compose_file"; then
            return 0
        else
            log "ERROR" "Docker Compose file missing GPU configuration"
            return 1
        fi
    else
        log "ERROR" "Main Docker Compose file not found: $compose_file"
        return 1
    fi
}

# Test 14: Check environment variables in compose files
test_environment_variables() {
    local compose_file="${PROJECT_ROOT}/docker-compose.yml"
    
    if [[ -f "$compose_file" ]]; then
        # Check for resource management environment variables
        if grep -q "MEMORY_OPTIMIZATION" "$compose_file" && grep -q "GPU_MEMORY_FRACTION" "$compose_file"; then
            return 0
        else
            log "ERROR" "Docker Compose file missing resource management environment variables"
            return 1
        fi
    else
        log "ERROR" "Main Docker Compose file not found: $compose_file"
        return 1
    fi
}

# Test 15: Check documentation exists
test_documentation_exists() {
    local doc_file="${PROJECT_ROOT}/docs/RESOURCE_MANAGEMENT_GUIDE.md"
    
    if [[ -f "$doc_file" ]]; then
        # Check if documentation has key sections
        if grep -q "Resource Management" "$doc_file" && grep -q "GPU Memory Management" "$doc_file"; then
            return 0
        else
            log "ERROR" "Resource management documentation appears incomplete"
            return 1
        fi
    else
        log "ERROR" "Resource management documentation not found: $doc_file"
        return 1
    fi
}

# Main test runner
main() {
    log "INFO" "Starting Resource Management Implementation Tests"
    log "INFO" "=============================================="
    
    # Create logs directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Run all tests
    run_test "Resource cleanup script exists" test_cleanup_script_exists
    run_test "GPU memory manager script exists" test_gpu_manager_script_exists
    run_test "Disk space monitor script exists" test_disk_monitor_script_exists
    run_test "Resource cleanup dry run" test_cleanup_script_dry_run
    run_test "GPU memory manager status" test_gpu_manager_status
    run_test "Disk space monitor check" test_disk_monitor_check
    run_test "Resource-optimized compose configuration" test_resource_optimized_compose
    run_test "Log rotation configuration" test_logrotate_config
    run_test "Deployment script exists" test_deployment_script_exists
    run_test "Compose resource limits" test_compose_resource_limits
    run_test "Required directories exist" test_required_directories
    run_test "Cleanup script help" test_cleanup_script_help
    run_test "GPU configuration in compose" test_gpu_configuration
    run_test "Environment variables in compose" test_environment_variables
    run_test "Documentation exists" test_documentation_exists
    
    # Print summary
    log "INFO" ""
    log "INFO" "Test Summary"
    log "INFO" "============"
    log "INFO" "Total tests: $TESTS_TOTAL"
    log "INFO" "Passed: $TESTS_PASSED"
    log "INFO" "Failed: $TESTS_FAILED"
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        log "INFO" "🎉 All tests passed! Resource management implementation is complete."
        return 0
    else
        log "ERROR" "❌ $TESTS_FAILED test(s) failed. Please review the implementation."
        return 1
    fi
}

# Help function
show_help() {
    cat << EOF
Resource Management Implementation Test Script

Usage: $0 [OPTIONS]

Options:
    -h, --help                      Show this help message
    
This script tests the resource management implementation to ensure:
- All required scripts are present and executable
- Configuration files are valid
- Docker Compose files have proper resource limits
- GPU management is configured
- Documentation is complete

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main "$@"