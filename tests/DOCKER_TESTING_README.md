# Docker Testing Suite

This directory contains comprehensive testing and validation scripts for the Docker containerization of the Qwen2 image generation application.

## Overview

The Docker testing suite validates:

- Container integration and service communication
- End-to-end user workflows in Docker environment
- Performance and resource usage validation
- Automated CI/CD pipeline integration

## Test Files

### Core Test Suites

1. **`test_docker_container_integration.py`**

   - Tests service communication between containers
   - Validates networking and service discovery
   - Tests volume persistence and data sharing
   - Validates network security and isolation

2. **`test_docker_e2e_workflows.py`**

   - End-to-end workflow testing in Docker environment
   - Complete user journey from frontend to backend
   - Multi-step workflows (generate → edit → enhance)
   - Error handling and recovery testing

3. **`test_docker_performance_validation.py`**
   - Resource usage monitoring and validation
   - Performance benchmarks and optimization testing
   - Load testing with concurrent requests
   - Container startup time and efficiency testing

### Test Runners and Utilities

4. **`run_docker_tests.py`**

   - Main test runner for CI/CD pipeline
   - Automated Docker environment setup
   - Comprehensive reporting and artifact collection
   - Configurable test execution

5. **`validate_docker_setup.py`**
   - Pre-test environment validation
   - Docker installation and configuration checks
   - System resource validation
   - Project structure verification

### Configuration and Scripts

6. **`docker_test_config.json`**

   - Comprehensive test configuration
   - Environment-specific settings
   - Resource thresholds and limits
   - Test data and parameters

7. **`scripts/docker-test-local.sh`**

   - Local development test runner
   - Interactive test execution
   - Detailed logging and diagnostics
   - Cleanup and artifact management

8. **`.github/workflows/docker-tests.yml`**
   - GitHub Actions CI/CD workflow
   - Automated testing on push/PR
   - Multi-matrix test execution
   - Security scanning integration

## Quick Start

### Prerequisites

1. **System Requirements:**

   - Docker 20.10+ and Docker Compose 2.0+
   - Python 3.11+ with pip
   - 8GB+ RAM (4GB minimum)
   - 10GB+ free disk space

2. **Validate Setup:**
   ```bash
   python tests/validate_docker_setup.py
   ```

### Running Tests Locally

1. **Run All Tests:**

   ```bash
   ./scripts/docker-test-local.sh
   ```

2. **Run Specific Test Suite:**

   ```bash
   ./scripts/docker-test-local.sh --suite container_integration
   ./scripts/docker-test-local.sh --suite e2e_workflows
   ./scripts/docker-test-local.sh --suite performance_validation
   ```

3. **Quick Tests (Skip Performance):**

   ```bash
   ./scripts/docker-test-local.sh --quick
   ```

4. **Verbose Output:**
   ```bash
   ./scripts/docker-test-local.sh --verbose
   ```

### Running Tests with Python

1. **Using Test Runner:**

   ```bash
   python tests/run_docker_tests.py
   ```

2. **Individual Test Files:**

   ```bash
   pytest tests/test_docker_container_integration.py -v
   pytest tests/test_docker_e2e_workflows.py -v
   pytest tests/test_docker_performance_validation.py -v
   ```

3. **With Custom Configuration:**
   ```bash
   python tests/run_docker_tests.py --config tests/docker_test_config.json
   ```

## Test Configuration

### Environment Variables

- `DOCKER_BUILDKIT=1` - Enable BuildKit for faster builds
- `COMPOSE_DOCKER_CLI_BUILD=1` - Use Docker CLI for compose builds
- `PYTHONPATH` - Include src directory for imports

### Configuration Options

The `docker_test_config.json` file provides extensive configuration:

- **Test Suites:** Enable/disable specific test categories
- **Docker Setup:** Image building and environment preparation
- **Thresholds:** Performance and resource limits
- **Environments:** Development, staging, production settings
- **Monitoring:** Resource usage and metrics collection

### Test Data

Configurable test parameters include:

- Image generation prompts and sizes
- Performance test scenarios
- Resource usage thresholds
- Timeout values

## CI/CD Integration

### GitHub Actions

The workflow automatically:

1. Sets up Docker environment
2. Builds required images
3. Runs test suites in parallel
4. Collects logs and artifacts
5. Generates test reports
6. Comments on PRs with results

### Triggering Tests

Tests run automatically on:

- Push to main/develop branches
- Pull requests
- Changes to Docker-related files
- Manual workflow dispatch

### Artifacts

Generated artifacts include:

- Test reports (JSON/HTML)
- Container logs
- System diagnostics
- Performance metrics
- Security scan results

## Test Categories

### 1. Container Integration Tests

**Purpose:** Validate container communication and networking

**Key Tests:**

- Traefik ↔ API communication
- Frontend ↔ API communication
- Service discovery and load balancing
- Volume persistence across restarts
- Network isolation and security

**Requirements:** 1.1, 1.3, 2.1, 2.2

### 2. End-to-End Workflow Tests

**Purpose:** Validate complete user workflows in Docker

**Key Tests:**

- Text-to-image generation via API
- Image editing with DiffSynth
- Multi-step user journeys
- Error handling and recovery
- Frontend-to-backend integration

**Requirements:** 1.3, 2.3, 4.4

### 3. Performance Validation Tests

**Purpose:** Validate resource usage and performance

**Key Tests:**

- Memory and CPU limit enforcement
- Container startup time benchmarks
- Concurrent request handling
- Resource usage under load
- Performance regression detection

**Requirements:** 1.3, 2.3, 4.4

## Troubleshooting

### Common Issues

1. **Docker Not Available:**

   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

   # Add user to docker group
   sudo usermod -aG docker $USER
   ```

2. **Insufficient Resources:**

   - Increase Docker memory limit in Docker Desktop
   - Free up disk space
   - Close other applications

3. **Port Conflicts:**

   - Stop services using ports 80, 443, 8080, 8000, 3000
   - Use different ports in docker-compose files

4. **Permission Issues:**

   ```bash
   # Fix Docker socket permissions
   sudo chmod 666 /var/run/docker.sock

   # Create required directories
   mkdir -p logs cache generated_images uploads
   ```

### Debug Mode

Enable verbose logging:

```bash
./scripts/docker-test-local.sh --verbose --no-cleanup
```

Check container logs:

```bash
docker-compose logs api
docker-compose logs frontend
docker-compose logs traefik
```

### Getting Help

1. Run validation script: `python tests/validate_docker_setup.py`
2. Check test artifacts in output directory
3. Review container logs and system diagnostics
4. Consult Docker and Docker Compose documentation

## Development

### Adding New Tests

1. Create test file following naming convention: `test_docker_*.py`
2. Use base classes from existing test files
3. Add configuration to `docker_test_config.json`
4. Update test runner and CI workflow

### Test Structure

```python
class TestDockerFeature(DockerTestBase):
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        self.setup_docker_environment()
        yield
        self.cleanup_docker_resources()

    def test_feature_functionality(self):
        # Test implementation
        pass
```

### Best Practices

- Use unique names for test containers/networks/volumes
- Always clean up resources in teardown
- Include proper error handling and timeouts
- Add comprehensive logging for debugging
- Validate both success and failure scenarios

## Reporting

### Test Reports

Generated reports include:

- **JSON Report:** Machine-readable test results
- **HTML Report:** Human-readable dashboard
- **Logs:** Detailed execution logs
- **Metrics:** Performance and resource data

### Metrics Collected

- Test execution times
- Container resource usage
- API response times
- Error rates and types
- System performance data

### Report Locations

- Local: `test-results/` directory
- CI: GitHub Actions artifacts
- Logs: `test-results/container-logs/`
- Reports: `test-results/*.html` and `test-results/*.json`

## Requirements Mapping

This testing suite validates the following requirements from the Docker containerization spec:

- **Requirement 1.1:** Complete Docker stack functionality
- **Requirement 1.3:** Consistent development environment
- **Requirement 2.1:** Optimized Docker images and caching
- **Requirement 2.2:** Resource limits and health checks
- **Requirement 2.3:** Performance monitoring and validation
- **Requirement 4.4:** Health checks and monitoring for all services

The comprehensive test suite ensures that all Docker containerization requirements are thoroughly validated through automated testing.
