#!/bin/bash
# =============================================================================
# Disk Space Monitoring Script for Qwen2 Docker Environment
# =============================================================================
# This script monitors disk space usage and triggers cleanup when thresholds
# are exceeded to prevent system failures due to disk space issues.
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_ROOT}/logs/disk-monitor.log"
VERBOSE=${VERBOSE:-false}

# Disk space thresholds (in percentage)
DISK_WARNING_THRESHOLD=${DISK_WARNING_THRESHOLD:-80}
DISK_CRITICAL_THRESHOLD=${DISK_CRITICAL_THRESHOLD:-90}
DISK_EMERGENCY_THRESHOLD=${DISK_EMERGENCY_THRESHOLD:-95}

# Directory size thresholds (in MB)
CACHE_SIZE_LIMIT=${CACHE_SIZE_LIMIT:-20480}      # 20GB
LOGS_SIZE_LIMIT=${LOGS_SIZE_LIMIT:-2048}         # 2GB
IMAGES_SIZE_LIMIT=${IMAGES_SIZE_LIMIT:-10240}    # 10GB

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
    
    if [[ "$VERBOSE" == "true" ]]; then
        case $level in
            "ERROR") echo -e "${RED}[ERROR]${NC} $message" >&2 ;;
            "WARN")  echo -e "${YELLOW}[WARN]${NC} $message" ;;
            "INFO")  echo -e "${GREEN}[INFO]${NC} $message" ;;
            "DEBUG") echo -e "${BLUE}[DEBUG]${NC} $message" ;;
        esac
    fi
}

# Get disk usage percentage for a path
get_disk_usage() {
    local path=$1
    df "$path" | awk 'NR==2 {print $5}' | sed 's/%//'
}

# Get directory size in MB
get_directory_size() {
    local dir=$1
    if [[ -d "$dir" ]]; then
        du -sm "$dir" 2>/dev/null | cut -f1 || echo "0"
    else
        echo "0"
    fi
}

# Check overall disk space
check_disk_space() {
    local usage=$(get_disk_usage "$PROJECT_ROOT")
    local available=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    local total=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $2}')
    
    log "INFO" "Disk Usage: ${usage}% (${available} available of ${total} total)"
    
    if [[ $usage -ge $DISK_EMERGENCY_THRESHOLD ]]; then
        log "ERROR" "EMERGENCY: Disk usage at ${usage}% - Immediate cleanup required!"
        return 3
    elif [[ $usage -ge $DISK_CRITICAL_THRESHOLD ]]; then
        log "ERROR" "CRITICAL: Disk usage at ${usage}% - Cleanup required!"
        return 2
    elif [[ $usage -ge $DISK_WARNING_THRESHOLD ]]; then
        log "WARN" "WARNING: Disk usage at ${usage}% - Consider cleanup"
        return 1
    else
        log "INFO" "Disk usage is normal (${usage}%)"
        return 0
    fi
}

# Check directory sizes
check_directory_sizes() {
    local status=0
    
    # Check cache directory
    local cache_size=$(get_directory_size "${PROJECT_ROOT}/cache")
    log "INFO" "Cache directory size: ${cache_size}MB"
    if [[ $cache_size -gt $CACHE_SIZE_LIMIT ]]; then
        log "WARN" "Cache directory exceeds limit (${CACHE_SIZE_LIMIT}MB)"
        status=1
    fi
    
    # Check logs directory
    local logs_size=$(get_directory_size "${PROJECT_ROOT}/logs")
    log "INFO" "Logs directory size: ${logs_size}MB"
    if [[ $logs_size -gt $LOGS_SIZE_LIMIT ]]; then
        log "WARN" "Logs directory exceeds limit (${LOGS_SIZE_LIMIT}MB)"
        status=1
    fi
    
    # Check generated images directory
    local images_size=$(get_directory_size "${PROJECT_ROOT}/generated_images")
    log "INFO" "Generated images directory size: ${images_size}MB"
    if [[ $images_size -gt $IMAGES_SIZE_LIMIT ]]; then
        log "WARN" "Generated images directory exceeds limit (${IMAGES_SIZE_LIMIT}MB)"
        status=1
    fi
    
    # Check Docker system usage
    if command -v docker &> /dev/null; then
        local docker_info=$(docker system df --format "table {{.Type}}\t{{.TotalCount}}\t{{.Size}}" 2>/dev/null || true)
        if [[ -n "$docker_info" ]]; then
            log "INFO" "Docker system usage:"
            echo "$docker_info" | while IFS= read -r line; do
                log "INFO" "  $line"
            done
        fi
    fi
    
    return $status
}

# Trigger automatic cleanup based on severity
trigger_cleanup() {
    local severity=$1
    
    case $severity in
        3) # Emergency
            log "ERROR" "Triggering emergency cleanup..."
            
            # Stop non-essential containers
            log "INFO" "Stopping non-essential containers..."
            docker stop qwen-cleanup-service qwen-resource-monitor 2>/dev/null || true
            
            # Run aggressive cleanup
            "${SCRIPT_DIR}/resource-cleanup.sh" --verbose
            
            # Clean Docker system aggressively
            docker system prune -af --volumes
            
            # Remove old Docker images
            docker image prune -af --filter "until=24h"
            
            log "INFO" "Emergency cleanup completed"
            ;;
            
        2) # Critical
            log "WARN" "Triggering critical cleanup..."
            
            # Run standard cleanup
            "${SCRIPT_DIR}/resource-cleanup.sh" --verbose
            
            # Clean Docker system
            docker system prune -f
            
            log "INFO" "Critical cleanup completed"
            ;;
            
        1) # Warning
            log "INFO" "Triggering warning-level cleanup..."
            
            # Run gentle cleanup
            "${SCRIPT_DIR}/resource-cleanup.sh"
            
            log "INFO" "Warning-level cleanup completed"
            ;;
            
        0) # Normal
            log "INFO" "No cleanup needed - disk usage is normal"
            ;;
    esac
}

# Send alert notification
send_alert() {
    local severity=$1
    local message=$2
    
    # Log the alert
    log "WARN" "ALERT: $message"
    
    # Write alert to file for external monitoring
    local alert_file="${PROJECT_ROOT}/logs/alerts.log"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [SEVERITY:$severity] $message" >> "$alert_file"
    
    # If webhook URL is configured, send notification
    if [[ -n "${ALERT_WEBHOOK_URL:-}" ]]; then
        local payload="{\"text\":\"Qwen2 Disk Alert: $message\",\"severity\":$severity}"
        curl -X POST -H "Content-Type: application/json" -d "$payload" "$ALERT_WEBHOOK_URL" 2>/dev/null || true
    fi
}

# Monitor disk space continuously
monitor_continuous() {
    local interval=${1:-300}  # Check every 5 minutes by default
    
    log "INFO" "Starting continuous disk space monitoring (interval: ${interval}s)"
    
    while true; do
        local disk_status
        check_disk_space
        disk_status=$?
        
        local dir_status
        check_directory_sizes
        dir_status=$?
        
        # Determine overall status
        local overall_status=$((disk_status > dir_status ? disk_status : dir_status))
        
        # Trigger cleanup if needed
        if [[ $overall_status -gt 0 ]]; then
            trigger_cleanup $overall_status
            
            # Send alert for critical situations
            if [[ $overall_status -ge 2 ]]; then
                local usage=$(get_disk_usage "$PROJECT_ROOT")
                send_alert $overall_status "Disk usage critical: ${usage}%"
            fi
        fi
        
        sleep "$interval"
    done
}

# Generate disk usage report
generate_report() {
    log "INFO" "Generating disk usage report..."
    
    local report_file="${PROJECT_ROOT}/logs/disk-usage-report-$(date +%Y%m%d-%H%M%S).txt"
    
    {
        echo "Qwen2 Disk Usage Report"
        echo "======================="
        echo "Generated: $(date)"
        echo ""
        
        echo "Overall Disk Usage:"
        df -h "$PROJECT_ROOT"
        echo ""
        
        echo "Directory Sizes:"
        echo "Cache:            $(get_directory_size "${PROJECT_ROOT}/cache")MB"
        echo "Logs:             $(get_directory_size "${PROJECT_ROOT}/logs")MB"
        echo "Generated Images: $(get_directory_size "${PROJECT_ROOT}/generated_images")MB"
        echo "Models:           $(get_directory_size "${PROJECT_ROOT}/models")MB"
        echo "Uploads:          $(get_directory_size "${PROJECT_ROOT}/uploads")MB"
        echo ""
        
        echo "Docker System Usage:"
        docker system df 2>/dev/null || echo "Docker not available"
        echo ""
        
        echo "Largest Files:"
        find "$PROJECT_ROOT" -type f -size +100M -exec ls -lh {} \; 2>/dev/null | head -20 || true
        echo ""
        
        echo "Oldest Files in Generated Images:"
        find "${PROJECT_ROOT}/generated_images" -type f -name "*.png" -o -name "*.jpg" | \
            xargs ls -lt 2>/dev/null | tail -10 || true
        
    } > "$report_file"
    
    log "INFO" "Report generated: $report_file"
    echo "$report_file"
}

# Main function
main() {
    local action=${1:-"check"}
    
    # Create logs directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    
    log "INFO" "Disk Space Monitor - Action: $action"
    
    case $action in
        "check")
            check_disk_space
            local disk_status=$?
            check_directory_sizes
            local dir_status=$?
            exit $((disk_status > dir_status ? disk_status : dir_status))
            ;;
        "monitor")
            monitor_continuous "${2:-300}"
            ;;
        "cleanup")
            local severity=${2:-1}
            trigger_cleanup "$severity"
            ;;
        "report")
            generate_report
            ;;
        "alert")
            send_alert "${2:-1}" "${3:-Test alert}"
            ;;
        *)
            log "ERROR" "Unknown action: $action"
            show_help
            exit 1
            ;;
    esac
}

# Help function
show_help() {
    cat << EOF
Disk Space Monitor for Qwen2 Docker Environment

Usage: $0 [ACTION] [OPTIONS]

Actions:
    check                           Check disk space once (default)
    monitor [interval]              Monitor continuously (interval in seconds)
    cleanup [severity]              Trigger cleanup (severity: 0-3)
    report                          Generate disk usage report
    alert [severity] [message]      Send test alert
    
Options:
    -v, --verbose                   Enable verbose output
    -h, --help                      Show this help message
    
Environment Variables:
    DISK_WARNING_THRESHOLD          Warning threshold percentage (default: 80)
    DISK_CRITICAL_THRESHOLD         Critical threshold percentage (default: 90)
    DISK_EMERGENCY_THRESHOLD        Emergency threshold percentage (default: 95)
    CACHE_SIZE_LIMIT               Cache size limit in MB (default: 20480)
    LOGS_SIZE_LIMIT                Logs size limit in MB (default: 2048)
    IMAGES_SIZE_LIMIT              Images size limit in MB (default: 10240)
    ALERT_WEBHOOK_URL              Webhook URL for alerts (optional)
    
Exit Codes:
    0 - Normal disk usage
    1 - Warning level
    2 - Critical level
    3 - Emergency level
    
Examples:
    $0                              Check disk space once
    $0 monitor 600                  Monitor every 10 minutes
    $0 cleanup 2                    Trigger critical cleanup
    $0 report                       Generate usage report

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            # Pass remaining arguments to main function
            break
            ;;
    esac
done

# Run main function with remaining arguments
main "$@"