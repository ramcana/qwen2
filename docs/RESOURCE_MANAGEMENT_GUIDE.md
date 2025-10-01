# Resource Management Guide for Qwen2 Docker Environment

This guide covers the comprehensive resource management system implemented for the Qwen2 image generation application, including memory limits, CPU constraints, GPU optimization, automatic cleanup, and log rotation.

## Overview

The resource management system provides:

- **Memory and CPU Constraints**: Strict limits to prevent resource exhaustion
- **GPU Memory Management**: Optimization and monitoring for NVIDIA GPUs
- **Automatic Cleanup**: Scheduled cleanup of temporary files, cache, and logs
- **Log Rotation**: Automated log management to prevent disk space issues
- **Disk Space Monitoring**: Continuous monitoring with automatic cleanup triggers
- **Performance Optimization**: System-wide optimizations for better resource utilization

## Components

### 1. Resource Cleanup System

#### Script: `scripts/resource-cleanup.sh`

Provides comprehensive cleanup of:

- Docker system resources (containers, images, volumes)
- HuggingFace and PyTorch cache
- Temporary files and directories
- Old generated images
- Application logs
- GPU memory cache

**Usage:**

```bash
# Run cleanup with default settings
./scripts/resource-cleanup.sh

# Dry run to see what would be cleaned
./scripts/resource-cleanup.sh --dry-run

# Verbose output
./scripts/resource-cleanup.sh --verbose

# Custom thresholds
CACHE_SIZE_THRESHOLD=5120 ./scripts/resource-cleanup.sh
```

**Configuration:**

- `CACHE_SIZE_THRESHOLD`: Cache size threshold in MB (default: 10240)
- `LOG_SIZE_THRESHOLD`: Log size threshold in MB (default: 1024)
- `TEMP_AGE_DAYS`: Age threshold for temp files (default: 7)
- `GENERATED_IMAGES_AGE_DAYS`: Age threshold for images (default: 30)

### 2. GPU Memory Management

#### Script: `scripts/gpu-memory-manager.sh`

Provides GPU-specific resource management:

- Memory usage monitoring
- Cache clearing and optimization
- Process management
- Emergency cleanup procedures

**Usage:**

```bash
# Show GPU status
./scripts/gpu-memory-manager.sh status

# Clear GPU cache
./scripts/gpu-memory-manager.sh clear

# Optimize GPU settings
./scripts/gpu-memory-manager.sh optimize

# Monitor GPU memory
./scripts/gpu-memory-manager.sh monitor 120 10

# Emergency cleanup
./scripts/gpu-memory-manager.sh emergency
```

**Configuration:**

- `GPU_MEMORY_WARNING_THRESHOLD`: Warning threshold in MB (default: 12288)
- `GPU_MEMORY_CRITICAL_THRESHOLD`: Critical threshold in MB (default: 14336)
- `GPU_MEMORY_CLEANUP_THRESHOLD`: Cleanup threshold in MB (default: 15360)

### 3. Disk Space Monitoring

#### Script: `scripts/disk-space-monitor.sh`

Continuous disk space monitoring with automatic cleanup:

- Real-time disk usage monitoring
- Directory size tracking
- Automatic cleanup triggers
- Alert notifications

**Usage:**

```bash
# Check disk space once
./scripts/disk-space-monitor.sh check

# Monitor continuously
./scripts/disk-space-monitor.sh monitor 300

# Generate usage report
./scripts/disk-space-monitor.sh report

# Trigger cleanup
./scripts/disk-space-monitor.sh cleanup 2
```

**Configuration:**

- `DISK_WARNING_THRESHOLD`: Warning percentage (default: 80)
- `DISK_CRITICAL_THRESHOLD`: Critical percentage (default: 90)
- `DISK_EMERGENCY_THRESHOLD`: Emergency percentage (default: 95)

### 4. Enhanced Docker Compose Configurations

#### Resource-Optimized Configuration

The `docker-compose.resource-optimized.yml` file provides:

**API Server Enhancements:**

```yaml
deploy:
  resources:
    limits:
      memory: 12G
      cpus: "6.0"
    reservations:
      memory: 6G
      cpus: "3.0"
```

**GPU Optimization:**

```yaml
environment:
  - GPU_MEMORY_FRACTION=0.85
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
  - ENABLE_MEMORY_EFFICIENT_ATTENTION=true
```

**Automatic Cleanup:**

```yaml
environment:
  - AUTO_CLEANUP_ENABLED=true
  - CLEANUP_TEMP_FILES_INTERVAL=1800
  - CLEANUP_OLD_MODELS_DAYS=30
```

### 5. Log Rotation Configuration

#### Configuration: `config/docker/logrotate.conf`

Automated log rotation for:

- Application logs (daily rotation, 14 days retention)
- API server logs (daily rotation, 10 days retention, 100MB size limit)
- Frontend logs (daily rotation, 7 days retention)
- GPU monitoring logs (daily rotation, 7 days retention, 10MB size limit)

### 6. Enhanced Deployment Script

#### Script: `scripts/deploy-with-resource-management.sh`

Comprehensive deployment with resource management:

- System capability detection
- Automatic resource limit configuration
- Monitoring setup
- Cleanup automation
- Deployment verification

**Usage:**

```bash
# Deploy with default settings
./scripts/deploy-with-resource-management.sh

# Development deployment
ENVIRONMENT=development ./scripts/deploy-with-resource-management.sh

# Production deployment with monitoring
ENVIRONMENT=production ENABLE_MONITORING=true ./scripts/deploy-with-resource-management.sh

# Dry run
DRY_RUN=true ./scripts/deploy-with-resource-management.sh
```

## Resource Limits and Constraints

### Memory Limits

| Service    | Development | Production | Resource-Optimized |
| ---------- | ----------- | ---------- | ------------------ |
| API Server | 8G          | 12G        | 12G                |
| Frontend   | 2G          | 256M       | 256M               |
| Traefik    | 512M        | 256M       | 256M               |
| Redis      | 1G          | 1G         | -                  |

### CPU Limits

| Service    | Development | Production | Resource-Optimized |
| ---------- | ----------- | ---------- | ------------------ |
| API Server | 4.0         | 6.0        | 6.0                |
| Frontend   | 2.0         | 0.5        | 0.5                |
| Traefik    | 1.0         | 0.5        | 0.5                |

### GPU Configuration

**Memory Management:**

- Memory fraction: 85% of available GPU memory
- Allocation configuration: `max_split_size_mb:512`
- Memory efficient attention: Enabled
- Gradient checkpointing: Enabled for large models

**Monitoring Thresholds:**

- Warning: 12GB (75% of 16GB GPU)
- Critical: 14GB (87.5% of 16GB GPU)
- Emergency cleanup: 15GB (93.75% of 16GB GPU)

## Automatic Cleanup Schedules

### Scheduled Cleanup (Cron)

- **Frequency**: Every 6 hours
- **Script**: `resource-cleanup.sh --verbose`
- **Log**: `logs/cleanup-cron.log`

### Continuous Monitoring

- **Disk Space**: Every 5 minutes
- **GPU Memory**: Every 15 seconds (when API is running)
- **Log Rotation**: Daily at midnight

### Cleanup Triggers

#### Warning Level (80% disk usage)

- Clean temporary files older than 7 days
- Compress logs older than 7 days
- Clear Docker build cache

#### Critical Level (90% disk usage)

- Run full resource cleanup
- Remove unused Docker images
- Clean old generated images (>7 days)

#### Emergency Level (95% disk usage)

- Stop non-essential containers
- Aggressive Docker system cleanup
- Force GPU memory cleanup
- Remove old Docker images (>24 hours)

## Monitoring and Alerting

### Metrics Collected

- Disk space usage (overall and per directory)
- Memory usage (system and per container)
- GPU memory usage and temperature
- Container health status
- Log file sizes

### Alert Conditions

- Disk usage > 80% (Warning)
- Disk usage > 90% (Critical)
- GPU memory > 85% (Warning)
- Container unhealthy for > 5 minutes
- Log files > 100MB

### Alert Destinations

- Log files (`logs/alerts.log`)
- Webhook notifications (if configured)
- Email notifications (if configured)

## Configuration Files

### Environment Variables

Create `.env.resource-limits` for custom configuration:

```bash
# System-specific limits
API_MEMORY_LIMIT=12g
API_CPU_LIMIT=6.0
FRONTEND_MEMORY_LIMIT=256m
FRONTEND_CPU_LIMIT=0.5

# Cleanup thresholds
CACHE_SIZE_THRESHOLD=10240
DISK_WARNING_THRESHOLD=80
DISK_CRITICAL_THRESHOLD=90

# GPU settings
GPU_MEMORY_FRACTION=0.85
GPU_MEMORY_WARNING_THRESHOLD=12288
```

### Monitoring Configuration

Create `config/monitoring.yml`:

```yaml
monitoring:
  enabled: true
  interval: 30s

  disk_monitoring:
    enabled: true
    warning_threshold: 80
    critical_threshold: 90

  gpu_monitoring:
    enabled: true
    memory_threshold: 85

  alerts:
    webhook_url: "https://hooks.slack.com/your-webhook"
    email: "admin@yourcompany.com"
```

## Troubleshooting

### High Memory Usage

1. **Check container memory usage:**

   ```bash
   docker stats --no-stream
   ```

2. **Clear GPU memory:**

   ```bash
   ./scripts/gpu-memory-manager.sh clear
   ```

3. **Run cleanup:**
   ```bash
   ./scripts/resource-cleanup.sh --verbose
   ```

### Disk Space Issues

1. **Check disk usage:**

   ```bash
   ./scripts/disk-space-monitor.sh check
   ```

2. **Generate usage report:**

   ```bash
   ./scripts/disk-space-monitor.sh report
   ```

3. **Emergency cleanup:**
   ```bash
   ./scripts/disk-space-monitor.sh cleanup 3
   ```

### GPU Memory Issues

1. **Check GPU status:**

   ```bash
   ./scripts/gpu-memory-manager.sh status
   ```

2. **Emergency GPU cleanup:**

   ```bash
   ./scripts/gpu-memory-manager.sh emergency
   ```

3. **Optimize GPU settings:**
   ```bash
   ./scripts/gpu-memory-manager.sh optimize
   ```

### Container Performance Issues

1. **Check resource limits:**

   ```bash
   docker-compose config | grep -A 10 "resources:"
   ```

2. **Monitor resource usage:**

   ```bash
   docker stats
   ```

3. **Adjust limits in compose file:**
   ```yaml
   deploy:
     resources:
       limits:
         memory: 16G # Increase if needed
         cpus: "8.0"
   ```

## Best Practices

### Resource Allocation

- Reserve 20% of system resources for the host OS
- Allocate 60-70% of memory to the API server
- Use CPU limits to prevent resource starvation
- Monitor GPU memory usage continuously

### Cleanup Strategy

- Run cleanup during low-usage periods
- Keep 7-30 days of generated images
- Rotate logs daily in production
- Monitor cache sizes regularly

### Performance Optimization

- Use tmpfs for temporary files
- Enable GPU memory optimization
- Configure appropriate batch sizes
- Use memory-efficient attention mechanisms

### Monitoring

- Set up alerts for critical thresholds
- Monitor trends over time
- Regular capacity planning reviews
- Automated health checks

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Deploy with Resource Management

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Deploy with resource management
        run: |
          ENVIRONMENT=production \
          ENABLE_MONITORING=true \
          ENABLE_CLEANUP=true \
          ./scripts/deploy-with-resource-management.sh

      - name: Verify deployment
        run: |
          ./scripts/deploy-with-resource-management.sh verify
```

This comprehensive resource management system ensures optimal performance, prevents resource exhaustion, and maintains system stability in production environments.
