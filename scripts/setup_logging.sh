#!/bin/bash

# Logging Setup Script for DiffSynth Enhanced UI
# Sets up centralized logging and log rotation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Create log directories
setup_log_directories() {
    log_info "Setting up log directories..."
    
    cd "$PROJECT_ROOT"
    
    # Create main log directories
    mkdir -p logs/{api,frontend,traefik,redis,prometheus,grafana}
    mkdir -p logs/api/{access,error,application,diffsynth,controlnet}
    mkdir -p logs/frontend/{access,error}
    mkdir -p logs/traefik/{access,error}
    
    # Set permissions
    chmod -R 755 logs
    
    log_success "Log directories created"
}

# Setup log rotation
setup_log_rotation() {
    log_info "Setting up log rotation..."
    
    # Create logrotate configuration
    cat > "$PROJECT_ROOT/logrotate.conf" << 'EOF'
# Log rotation configuration for DiffSynth Enhanced UI

/app/logs/api/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
    postrotate
        docker-compose exec api kill -USR1 1 2>/dev/null || true
    endscript
}

/app/logs/frontend/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
    postrotate
        docker-compose exec frontend nginx -s reload 2>/dev/null || true
    endscript
}

/app/logs/traefik/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
    postrotate
        docker-compose exec traefik kill -USR1 1 2>/dev/null || true
    endscript
}
EOF
    
    log_success "Log rotation configured"
}

# Setup log monitoring
setup_log_monitoring() {
    log_info "Setting up log monitoring scripts..."
    
    # Create log monitoring script
    cat > "$PROJECT_ROOT/scripts/monitor_logs.sh" << 'EOF'
#!/bin/bash

# Log Monitoring Script for DiffSynth Enhanced UI
# Monitors logs for errors and sends alerts

LOG_DIR="/app/logs"
ALERT_EMAIL="admin@yourdomain.com"
ERROR_THRESHOLD=10

# Function to check for errors in logs
check_errors() {
    local log_file="$1"
    local service="$2"
    
    if [ -f "$log_file" ]; then
        # Count errors in the last hour
        error_count=$(grep -c "ERROR\|CRITICAL\|FATAL" "$log_file" 2>/dev/null || echo 0)
        
        if [ "$error_count" -gt "$ERROR_THRESHOLD" ]; then
            echo "WARNING: $service has $error_count errors in the last hour"
            # Send alert (implement your preferred alerting method)
            # mail -s "DiffSynth Alert: High error rate in $service" "$ALERT_EMAIL" < /dev/null
        fi
    fi
}

# Check all service logs
check_errors "$LOG_DIR/api/application.log" "API Service"
check_errors "$LOG_DIR/api/diffsynth.log" "DiffSynth Service"
check_errors "$LOG_DIR/api/controlnet.log" "ControlNet Service"
check_errors "$LOG_DIR/frontend/error.log" "Frontend Service"
check_errors "$LOG_DIR/traefik/error.log" "Traefik Service"

# Check disk space
disk_usage=$(df "$LOG_DIR" | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$disk_usage" -gt 80 ]; then
    echo "WARNING: Log directory is $disk_usage% full"
fi
EOF
    
    chmod +x "$PROJECT_ROOT/scripts/monitor_logs.sh"
    
    log_success "Log monitoring scripts created"
}

# Setup log aggregation
setup_log_aggregation() {
    log_info "Setting up log aggregation..."
    
    # Create log aggregation script
    cat > "$PROJECT_ROOT/scripts/aggregate_logs.sh" << 'EOF'
#!/bin/bash

# Log Aggregation Script for DiffSynth Enhanced UI
# Aggregates logs from all services for analysis

LOG_DIR="/app/logs"
OUTPUT_DIR="/app/logs/aggregated"
DATE=$(date +%Y%m%d)

mkdir -p "$OUTPUT_DIR"

# Aggregate API logs
echo "Aggregating API logs..."
cat "$LOG_DIR"/api/*.log > "$OUTPUT_DIR/api_$DATE.log" 2>/dev/null || true

# Aggregate frontend logs
echo "Aggregating frontend logs..."
cat "$LOG_DIR"/frontend/*.log > "$OUTPUT_DIR/frontend_$DATE.log" 2>/dev/null || true

# Aggregate traefik logs
echo "Aggregating traefik logs..."
cat "$LOG_DIR"/traefik/*.log > "$OUTPUT_DIR/traefik_$DATE.log" 2>/dev/null || true

# Create summary report
echo "Creating summary report..."
cat > "$OUTPUT_DIR/summary_$DATE.txt" << SUMMARY
DiffSynth Enhanced UI Log Summary - $DATE

API Service:
- Total requests: $(grep -c "GET\|POST\|PUT\|DELETE" "$OUTPUT_DIR/api_$DATE.log" 2>/dev/null || echo 0)
- Errors: $(grep -c "ERROR\|CRITICAL\|FATAL" "$OUTPUT_DIR/api_$DATE.log" 2>/dev/null || echo 0)
- Generation requests: $(grep -c "generate" "$OUTPUT_DIR/api_$DATE.log" 2>/dev/null || echo 0)

Frontend Service:
- Total requests: $(grep -c "GET\|POST" "$OUTPUT_DIR/frontend_$DATE.log" 2>/dev/null || echo 0)
- Errors: $(grep -c "error\|ERROR" "$OUTPUT_DIR/frontend_$DATE.log" 2>/dev/null || echo 0)

Traefik Service:
- Total requests: $(grep -c '"' "$OUTPUT_DIR/traefik_$DATE.log" 2>/dev/null || echo 0)
- 4xx errors: $(grep -c '"4[0-9][0-9]"' "$OUTPUT_DIR/traefik_$DATE.log" 2>/dev/null || echo 0)
- 5xx errors: $(grep -c '"5[0-9][0-9]"' "$OUTPUT_DIR/traefik_$DATE.log" 2>/dev/null || echo 0)

SUMMARY

echo "Log aggregation completed. Summary saved to $OUTPUT_DIR/summary_$DATE.txt"
EOF
    
    chmod +x "$PROJECT_ROOT/scripts/aggregate_logs.sh"
    
    log_success "Log aggregation scripts created"
}

# Main function
main() {
    log_info "Setting up logging for DiffSynth Enhanced UI..."
    
    setup_log_directories
    setup_log_rotation
    setup_log_monitoring
    setup_log_aggregation
    
    log_success "Logging setup completed!"
    
    echo
    log_info "Next steps:"
    echo "1. Add logrotate configuration to system crontab"
    echo "2. Set up log monitoring cron job: */15 * * * * $PROJECT_ROOT/scripts/monitor_logs.sh"
    echo "3. Set up daily log aggregation: 0 1 * * * $PROJECT_ROOT/scripts/aggregate_logs.sh"
    echo "4. Configure alerting system for log monitoring"
}

main "$@"