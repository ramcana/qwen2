#!/bin/bash
# Local Docker Testing Script
# Run Docker tests locally with proper setup and cleanup

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_OUTPUT_DIR="$PROJECT_ROOT/test-results-$(date +%Y%m%d-%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Docker Test Runner - Local Development

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -s, --suite SUITE       Run specific test suite (container_integration, e2e_workflows, performance_validation)
    -c, --no-cleanup        Skip cleanup after tests
    -b, --no-build          Skip building Docker images
    -v, --verbose           Enable verbose output
    -o, --output DIR        Output directory for test results
    --quick                 Run quick tests only (skip performance tests)
    --ci                    Run in CI mode (stricter settings)

EXAMPLES:
    $0                                  # Run all tests
    $0 -s container_integration         # Run only container integration tests
    $0 --quick                          # Run quick tests only
    $0 -v -o ./my-test-results         # Verbose output to custom directory

EOF
}

# Parse command line arguments
SUITE=""
NO_CLEANUP=false
NO_BUILD=false
VERBOSE=false
OUTPUT_DIR=""
QUICK_MODE=false
CI_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--suite)
            SUITE="$2"
            shift 2
            ;;
        -c|--no-cleanup)
            NO_CLEANUP=true
            shift
            ;;
        -b|--no-build)
            NO_BUILD=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --ci)
            CI_MODE=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set output directory
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$TEST_OUTPUT_DIR"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

log_info "Docker Test Runner Starting"
log_info "Project Root: $PROJECT_ROOT"
log_info "Output Directory: $OUTPUT_DIR"

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Setup test environment
setup_environment() {
    log_info "Setting up test environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create required directories
    mkdir -p logs/api logs/traefik
    mkdir -p data/redis data/prometheus data/grafana
    mkdir -p cache/huggingface cache/torch cache/diffsynth cache/controlnet
    mkdir -p models/diffsynth models/controlnet
    mkdir -p generated_images uploads offload
    mkdir -p ssl config/redis config/prometheus config/grafana config/docker
    
    # Create external networks
    docker network create traefik-public 2>/dev/null || true
    
    # Create basic Traefik config for testing
    cat > config/docker/traefik.yml << 'EOF'
api:
  dashboard: true
  insecure: true

entryPoints:
  web:
    address: ":80"
  websecure:
    address: ":443"

providers:
  docker:
    exposedByDefault: false

log:
  level: INFO
EOF
    
    # Create acme.json with correct permissions
    touch acme.json
    chmod 600 acme.json
    
    # Install Python dependencies
    if [[ -f requirements.txt ]]; then
        log_info "Installing Python dependencies..."
        python3 -m pip install -q pytest pytest-json-report requests docker psutil pillow
        python3 -m pip install -q -r requirements.txt
    fi
    
    log_success "Environment setup complete"
}

# Build Docker images
build_images() {
    if [[ "$NO_BUILD" == "true" ]]; then
        log_info "Skipping Docker image build (--no-build specified)"
        return
    fi
    
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build API image
    log_info "Building API image..."
    if [[ "$VERBOSE" == "true" ]]; then
        docker build -t qwen-api:latest -f Dockerfile.api .
    else
        docker build -t qwen-api:latest -f Dockerfile.api . > "$OUTPUT_DIR/build-api.log" 2>&1
    fi
    
    # Build frontend image
    log_info "Building frontend image..."
    if [[ "$VERBOSE" == "true" ]]; then
        docker build -t qwen-frontend:latest -f frontend/Dockerfile frontend/
    else
        docker build -t qwen-frontend:latest -f frontend/Dockerfile frontend/ > "$OUTPUT_DIR/build-frontend.log" 2>&1
    fi
    
    # Verify images
    log_info "Verifying built images..."
    docker images | grep qwen | tee "$OUTPUT_DIR/built-images.txt"
    
    log_success "Docker images built successfully"
}

# Run tests
run_tests() {
    log_info "Running Docker tests..."
    
    cd "$PROJECT_ROOT"
    
    # Prepare test runner arguments
    TEST_ARGS="--output-dir $OUTPUT_DIR"
    
    if [[ -n "$SUITE" ]]; then
        TEST_ARGS="$TEST_ARGS --suite $SUITE"
    fi
    
    if [[ "$NO_CLEANUP" == "true" ]]; then
        TEST_ARGS="$TEST_ARGS --no-cleanup"
    fi
    
    if [[ "$NO_BUILD" == "true" ]]; then
        TEST_ARGS="$TEST_ARGS --no-build"
    fi
    
    # Create test configuration for quick mode
    if [[ "$QUICK_MODE" == "true" ]]; then
        cat > "$OUTPUT_DIR/test-config.json" << 'EOF'
{
    "test_suites": {
        "container_integration": {
            "enabled": true,
            "file": "test_docker_container_integration.py",
            "timeout": 300,
            "required": true,
            "description": "Container integration tests (quick mode)"
        },
        "e2e_workflows": {
            "enabled": true,
            "file": "test_docker_e2e_workflows.py",
            "timeout": 600,
            "required": true,
            "description": "E2E workflow tests (quick mode)"
        },
        "performance_validation": {
            "enabled": false,
            "file": "test_docker_performance_validation.py",
            "timeout": 900,
            "required": false,
            "description": "Performance tests (disabled in quick mode)"
        }
    },
    "docker_setup": {
        "cleanup_before": true,
        "cleanup_after": true,
        "build_images": false,
        "pull_base_images": false
    },
    "thresholds": {
        "min_success_rate": 0.7,
        "max_test_duration": 1200
    }
}
EOF
        TEST_ARGS="$TEST_ARGS --config $OUTPUT_DIR/test-config.json"
    fi
    
    # Run the tests
    log_info "Executing test command: python3 tests/run_docker_tests.py $TEST_ARGS"
    
    if [[ "$VERBOSE" == "true" ]]; then
        python3 tests/run_docker_tests.py $TEST_ARGS
    else
        python3 tests/run_docker_tests.py $TEST_ARGS 2>&1 | tee "$OUTPUT_DIR/test-execution.log"
    fi
    
    TEST_EXIT_CODE=$?
    
    if [[ $TEST_EXIT_CODE -eq 0 ]]; then
        log_success "All tests passed!"
    else
        log_error "Some tests failed (exit code: $TEST_EXIT_CODE)"
    fi
    
    return $TEST_EXIT_CODE
}

# Collect additional diagnostics
collect_diagnostics() {
    log_info "Collecting diagnostic information..."
    
    # Docker system info
    docker system df > "$OUTPUT_DIR/docker-system-df.txt" 2>&1 || true
    docker system info > "$OUTPUT_DIR/docker-system-info.txt" 2>&1 || true
    
    # Container logs
    mkdir -p "$OUTPUT_DIR/container-logs"
    for container in $(docker ps -a --format "{{.Names}}"); do
        log_info "Collecting logs for container: $container"
        docker logs "$container" > "$OUTPUT_DIR/container-logs/${container}.log" 2>&1 || true
    done
    
    # Network information
    docker network ls > "$OUTPUT_DIR/docker-networks.txt" 2>&1 || true
    
    # Volume information
    docker volume ls > "$OUTPUT_DIR/docker-volumes.txt" 2>&1 || true
    
    # Image information
    docker images > "$OUTPUT_DIR/docker-images.txt" 2>&1 || true
    
    log_success "Diagnostics collected"
}

# Cleanup function
cleanup() {
    if [[ "$NO_CLEANUP" == "true" ]]; then
        log_info "Skipping cleanup (--no-cleanup specified)"
        return
    fi
    
    log_info "Cleaning up Docker resources..."
    
    # Stop all test containers
    docker ps -a --filter "name=test-" -q | xargs -r docker stop 2>/dev/null || true
    
    # Remove test containers
    docker ps -a --filter "name=test-" -q | xargs -r docker rm -f 2>/dev/null || true
    
    # Remove test networks
    docker network ls --filter "name=test-" -q | xargs -r docker network rm 2>/dev/null || true
    
    # Remove test volumes
    docker volume ls --filter "name=test-" -q | xargs -r docker volume rm -f 2>/dev/null || true
    
    # Prune system (be careful in CI)
    if [[ "$CI_MODE" != "true" ]]; then
        docker system prune -f 2>/dev/null || true
    fi
    
    log_success "Cleanup complete"
}

# Generate summary report
generate_summary() {
    log_info "Generating test summary..."
    
    SUMMARY_FILE="$OUTPUT_DIR/test-summary.txt"
    
    cat > "$SUMMARY_FILE" << EOF
Docker Test Execution Summary
============================

Execution Time: $(date)
Project Root: $PROJECT_ROOT
Output Directory: $OUTPUT_DIR

Configuration:
- Suite: ${SUITE:-"all"}
- No Cleanup: $NO_CLEANUP
- No Build: $NO_BUILD
- Verbose: $VERBOSE
- Quick Mode: $QUICK_MODE
- CI Mode: $CI_MODE

Test Results:
EOF
    
    # Add test results if available
    if [[ -f "$OUTPUT_DIR/docker_test_report.json" ]]; then
        echo "- Detailed results available in docker_test_report.json" >> "$SUMMARY_FILE"
        echo "- HTML report available in docker_test_report.html" >> "$SUMMARY_FILE"
    fi
    
    # Add file listing
    echo "" >> "$SUMMARY_FILE"
    echo "Generated Files:" >> "$SUMMARY_FILE"
    find "$OUTPUT_DIR" -type f -name "*.log" -o -name "*.json" -o -name "*.html" -o -name "*.txt" | sort >> "$SUMMARY_FILE"
    
    log_success "Summary generated: $SUMMARY_FILE"
}

# Main execution
main() {
    local exit_code=0
    
    # Setup trap for cleanup
    trap cleanup EXIT
    
    # Run all steps
    check_prerequisites
    setup_environment
    build_images
    
    # Run tests and capture exit code
    if run_tests; then
        log_success "Tests completed successfully"
    else
        exit_code=$?
        log_error "Tests failed with exit code: $exit_code"
    fi
    
    # Always collect diagnostics and generate summary
    collect_diagnostics
    generate_summary
    
    # Final message
    echo ""
    log_info "Test execution complete!"
    log_info "Results available in: $OUTPUT_DIR"
    
    if [[ -f "$OUTPUT_DIR/docker_test_report.html" ]]; then
        log_info "Open HTML report: file://$OUTPUT_DIR/docker_test_report.html"
    fi
    
    return $exit_code
}

# Run main function
main "$@"