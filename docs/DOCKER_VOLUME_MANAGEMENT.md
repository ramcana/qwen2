# Docker Volume Management Guide

This guide covers comprehensive volume management for the Qwen2 Docker deployment, including data persistence, backup strategies, and optimization techniques.

## Table of Contents

- [Volume Overview](#volume-overview)
- [Volume Configuration](#volume-configuration)
- [Data Persistence Strategy](#data-persistence-strategy)
- [Backup and Restore](#backup-and-restore)
- [Volume Optimization](#volume-optimization)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Troubleshooting](#troubleshooting)

## Volume Overview

The Qwen2 application uses several persistent volumes to maintain data across container restarts and updates. Understanding these volumes is crucial for proper deployment and maintenance.

### Volume Types

#### 1. Model Storage Volumes

- **Purpose**: Store pre-trained models and weights
- **Criticality**: High - expensive to re-download
- **Size**: 10-50GB depending on models used
- **Growth**: Moderate - grows as new models are added

#### 2. Cache Volumes

- **Purpose**: Store framework and library caches
- **Criticality**: Medium - improves performance but can be regenerated
- **Size**: 5-20GB
- **Growth**: Moderate - grows with usage

#### 3. User Data Volumes

- **Purpose**: Store user uploads and generated content
- **Criticality**: High - user-generated content
- **Size**: Varies based on usage
- **Growth**: High - grows with user activity

#### 4. Application Data Volumes

- **Purpose**: Store application configuration and logs
- **Criticality**: Medium - important for debugging and configuration
- **Size**: 1-5GB
- **Growth**: Low to moderate

## Volume Configuration

### Docker Compose Volume Definitions

```yaml
volumes:
  # Model storage
  qwen2_models:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ./models

  # Cache storage
  qwen2_cache:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ./cache

  # Generated images
  qwen2_generated:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ./generated_images

  # User uploads
  qwen2_uploads:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ./uploads

  # Application data
  qwen2_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ./data
```

### Service Volume Mounts

```yaml
services:
  api:
    volumes:
      # Model storage
      - qwen2_models:/app/models:rw
      - qwen2_cache:/app/cache:rw

      # User data
      - qwen2_generated:/app/generated_images:rw
      - qwen2_uploads:/app/uploads:rw

      # Application data
      - qwen2_data:/app/data:rw

      # Logs
      - ./logs/api:/app/logs:rw

  frontend:
    volumes:
      # Static assets (if needed)
      - qwen2_uploads:/app/uploads:ro
      - qwen2_generated:/app/generated_images:ro

  traefik:
    volumes:
      # Configuration
      - ./config/traefik:/etc/traefik:ro
      - ./acme.json:/acme.json:rw

      # Logs
      - ./logs/traefik:/var/log/traefik:rw

      # Docker socket
      - /var/run/docker.sock:/var/run/docker.sock:ro
```

## Data Persistence Strategy

### Critical Data Classification

#### Tier 1: Critical Data (Must Backup)

- **Models**: Pre-trained model files
- **User Uploads**: Original user-provided images
- **Generated Images**: User-created content
- **Configuration**: Custom application settings

#### Tier 2: Important Data (Should Backup)

- **Cache**: Framework caches for performance
- **Logs**: Application and access logs
- **Metrics**: Historical performance data

#### Tier 3: Temporary Data (Can Recreate)

- **Temporary Files**: Processing intermediates
- **Build Cache**: Docker build artifacts
- **Session Data**: Temporary user sessions

### Persistence Implementation

#### Host-Mounted Volumes (Recommended)

```bash
# Create directory structure
mkdir -p {models,cache,generated_images,uploads,data,logs}

# Set proper permissions
chmod 755 models cache generated_images uploads data logs
chown -R $USER:$USER models cache generated_images uploads data logs
```

#### Named Docker Volumes (Alternative)

```yaml
volumes:
  qwen2_models:
    driver: local
  qwen2_cache:
    driver: local
  qwen2_data:
    driver: local
```

### Volume Initialization

#### Pre-populate Models

```bash
# Download models before first run
mkdir -p models/Qwen
mkdir -p models/DiffSynth
mkdir -p models/ControlNet

# Set cache directories
mkdir -p cache/huggingface
mkdir -p cache/torch
mkdir -p cache/diffsynth
```

#### Initialize Application Data

```bash
# Create application directories
mkdir -p data/redis
mkdir -p data/prometheus
mkdir -p data/grafana

# Create log directories
mkdir -p logs/api
mkdir -p logs/traefik
mkdir -p logs/nginx
```

## Backup and Restore

### Backup Strategy

#### Full System Backup

```bash
#!/bin/bash
# Full backup script

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups"
BACKUP_FILE="qwen2_full_backup_${BACKUP_DATE}.tar.gz"

mkdir -p "$BACKUP_DIR"

echo "Creating full system backup..."
tar -czf "${BACKUP_DIR}/${BACKUP_FILE}" \
    models/ \
    cache/ \
    generated_images/ \
    uploads/ \
    data/ \
    config/ \
    .env.docker \
    docker-compose*.yml

echo "Backup created: ${BACKUP_DIR}/${BACKUP_FILE}"
echo "Size: $(du -h ${BACKUP_DIR}/${BACKUP_FILE} | cut -f1)"
```

#### Incremental Backup

```bash
#!/bin/bash
# Incremental backup script

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/incremental"
LAST_BACKUP_FILE="backups/.last_backup"

mkdir -p "$BACKUP_DIR"

# Find files changed since last backup
if [[ -f "$LAST_BACKUP_FILE" ]]; then
    SINCE_DATE=$(cat "$LAST_BACKUP_FILE")
    find models cache generated_images uploads data -newer "$SINCE_DATE" > /tmp/changed_files
else
    find models cache generated_images uploads data > /tmp/changed_files
fi

# Create incremental backup
tar -czf "${BACKUP_DIR}/qwen2_incremental_${BACKUP_DATE}.tar.gz" -T /tmp/changed_files

# Update last backup timestamp
date > "$LAST_BACKUP_FILE"

echo "Incremental backup created: ${BACKUP_DIR}/qwen2_incremental_${BACKUP_DATE}.tar.gz"
```

#### Selective Backup Scripts

**Models Only:**

```bash
#!/bin/bash
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "backups/models_backup_${BACKUP_DATE}.tar.gz" models/
echo "Models backup created: backups/models_backup_${BACKUP_DATE}.tar.gz"
```

**User Data Only:**

```bash
#!/bin/bash
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "backups/userdata_backup_${BACKUP_DATE}.tar.gz" generated_images/ uploads/
echo "User data backup created: backups/userdata_backup_${BACKUP_DATE}.tar.gz"
```

### Restore Procedures

#### Full System Restore

```bash
#!/bin/bash
# Full system restore script

BACKUP_FILE="$1"

if [[ -z "$BACKUP_FILE" || ! -f "$BACKUP_FILE" ]]; then
    echo "Usage: $0 <backup_file>"
    echo "Available backups:"
    ls -la backups/qwen2_full_backup_*.tar.gz
    exit 1
fi

echo "WARNING: This will overwrite existing data!"
read -p "Are you sure you want to restore from $BACKUP_FILE? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Stop services
    docker-compose down

    # Create backup of current state
    CURRENT_BACKUP="backups/pre_restore_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
    tar -czf "$CURRENT_BACKUP" models/ cache/ generated_images/ uploads/ data/ 2>/dev/null || true
    echo "Current state backed up to: $CURRENT_BACKUP"

    # Restore from backup
    echo "Restoring from: $BACKUP_FILE"
    tar -xzf "$BACKUP_FILE"

    # Restart services
    docker-compose up -d

    echo "Restore completed successfully!"
else
    echo "Restore cancelled."
fi
```

#### Selective Restore

```bash
#!/bin/bash
# Selective restore script

BACKUP_FILE="$1"
RESTORE_TYPE="$2"

case "$RESTORE_TYPE" in
    models)
        tar -xzf "$BACKUP_FILE" models/
        echo "Models restored from: $BACKUP_FILE"
        ;;
    userdata)
        tar -xzf "$BACKUP_FILE" generated_images/ uploads/
        echo "User data restored from: $BACKUP_FILE"
        ;;
    config)
        tar -xzf "$BACKUP_FILE" config/ .env.docker
        echo "Configuration restored from: $BACKUP_FILE"
        ;;
    *)
        echo "Usage: $0 <backup_file> <models|userdata|config>"
        exit 1
        ;;
esac
```

### Automated Backup Scheduling

#### Cron Job Setup

```bash
# Add to crontab (crontab -e)

# Daily full backup at 2 AM
0 2 * * * /path/to/qwen2/scripts/backup-full.sh

# Hourly incremental backup during business hours
0 9-17 * * 1-5 /path/to/qwen2/scripts/backup-incremental.sh

# Weekly cleanup of old backups (keep last 30 days)
0 3 * * 0 find /path/to/qwen2/backups -name "*.tar.gz" -mtime +30 -delete
```

#### Systemd Timer (Alternative)

```ini
# /etc/systemd/system/qwen2-backup.timer
[Unit]
Description=Qwen2 Daily Backup
Requires=qwen2-backup.service

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

```ini
# /etc/systemd/system/qwen2-backup.service
[Unit]
Description=Qwen2 Backup Service
After=docker.service

[Service]
Type=oneshot
User=qwen2
WorkingDirectory=/path/to/qwen2
ExecStart=/path/to/qwen2/scripts/backup-full.sh
```

## Volume Optimization

### Storage Optimization

#### Model Deduplication

```bash
#!/bin/bash
# Find and deduplicate model files

echo "Scanning for duplicate model files..."
find models/ -type f -name "*.bin" -o -name "*.safetensors" | \
    xargs -I {} sh -c 'echo "$(md5sum "{}" | cut -d" " -f1) {}"' | \
    sort | \
    uniq -d -w32 | \
    cut -d' ' -f2-

echo "Use hardlinks to deduplicate identical files"
```

#### Cache Cleanup

```bash
#!/bin/bash
# Clean up old cache files

# Clean HuggingFace cache (keep last 30 days)
find cache/huggingface/ -type f -mtime +30 -delete

# Clean PyTorch cache
find cache/torch/ -type f -name "*.tmp" -delete

# Clean temporary files
find cache/ -type f -name "*.lock" -mtime +1 -delete
find cache/ -type f -name "*.tmp" -mtime +1 -delete
```

#### Log Rotation

```bash
#!/bin/bash
# Rotate and compress logs

# Rotate API logs
if [[ -f "logs/api/app.log" ]]; then
    mv logs/api/app.log logs/api/app.log.$(date +%Y%m%d)
    gzip logs/api/app.log.$(date +%Y%m%d)
    touch logs/api/app.log
fi

# Clean old logs (keep last 30 days)
find logs/ -name "*.gz" -mtime +30 -delete
```

### Performance Optimization

#### Volume Mount Options

```yaml
# Optimized volume mounts
services:
  api:
    volumes:
      # Read-write with sync for critical data
      - ./models:/app/models:rw,sync

      # Read-write with async for cache (better performance)
      - ./cache:/app/cache:rw,async

      # Read-only for configuration
      - ./config:/app/config:ro
```

#### Filesystem Recommendations

**For Model Storage:**

- **ext4**: Good general performance
- **xfs**: Better for large files
- **zfs**: Advanced features (compression, snapshots)

**For Cache Storage:**

- **tmpfs**: For temporary cache (RAM-based)
- **ext4**: Good balance of performance and reliability

#### SSD Optimization

```bash
# Enable TRIM for SSD
sudo fstrim -av

# Check SSD health
sudo smartctl -a /dev/sda
```

## Monitoring and Maintenance

### Volume Usage Monitoring

#### Disk Usage Script

```bash
#!/bin/bash
# Monitor volume usage

echo "=== Qwen2 Volume Usage Report ==="
echo "Generated: $(date)"
echo

echo "=== Directory Sizes ==="
du -sh models cache generated_images uploads data logs 2>/dev/null | sort -hr

echo
echo "=== Detailed Breakdown ==="
echo "Models:"
du -sh models/*/ 2>/dev/null | sort -hr | head -10

echo
echo "Cache:"
du -sh cache/*/ 2>/dev/null | sort -hr

echo
echo "=== Disk Space ==="
df -h . | tail -1

echo
echo "=== Large Files (>1GB) ==="
find . -type f -size +1G -exec ls -lh {} \; 2>/dev/null | sort -k5 -hr | head -10
```

#### Automated Monitoring

```bash
#!/bin/bash
# Automated volume monitoring with alerts

THRESHOLD=80  # Alert when usage exceeds 80%
USAGE=$(df . | tail -1 | awk '{print $5}' | sed 's/%//')

if [[ $USAGE -gt $THRESHOLD ]]; then
    echo "WARNING: Disk usage is ${USAGE}% (threshold: ${THRESHOLD}%)"

    # Send alert (customize as needed)
    # mail -s "Qwen2 Disk Usage Alert" admin@example.com < /tmp/usage_report
    # curl -X POST "https://hooks.slack.com/..." -d "{'text':'Disk usage alert: ${USAGE}%'}"

    # Suggest cleanup
    echo "Consider running cleanup scripts:"
    echo "  - Clean old generated images"
    echo "  - Clean cache files"
    echo "  - Rotate logs"
fi
```

### Health Checks

#### Volume Health Check

```bash
#!/bin/bash
# Check volume health and accessibility

echo "=== Volume Health Check ==="

# Check if volumes are mounted
VOLUMES=("models" "cache" "generated_images" "uploads" "data")

for vol in "${VOLUMES[@]}"; do
    if [[ -d "$vol" && -w "$vol" ]]; then
        echo "✓ $vol: OK"
    else
        echo "✗ $vol: FAILED (not accessible or not writable)"
    fi
done

# Check for filesystem errors
echo
echo "=== Filesystem Check ==="
if command -v fsck &> /dev/null; then
    fsck -n $(df . | tail -1 | awk '{print $1}') 2>/dev/null || echo "Filesystem check not available"
fi

# Check Docker volume status
echo
echo "=== Docker Volume Status ==="
docker volume ls | grep qwen2 || echo "No Docker named volumes found"
```

### Maintenance Tasks

#### Weekly Maintenance Script

```bash
#!/bin/bash
# Weekly maintenance tasks

echo "=== Qwen2 Weekly Maintenance ==="
echo "Started: $(date)"

# 1. Clean temporary files
echo "Cleaning temporary files..."
find . -name "*.tmp" -type f -mtime +1 -delete
find . -name "*.lock" -type f -mtime +1 -delete

# 2. Rotate logs
echo "Rotating logs..."
./scripts/rotate-logs.sh

# 3. Clean old generated images (optional)
read -p "Clean generated images older than 30 days? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    find generated_images/ -name "*.png" -o -name "*.jpg" -mtime +30 -delete
    echo "Old generated images cleaned"
fi

# 4. Update usage report
echo "Generating usage report..."
./scripts/volume-usage-report.sh > reports/volume_usage_$(date +%Y%m%d).txt

# 5. Check volume health
echo "Checking volume health..."
./scripts/volume-health-check.sh

echo "Maintenance completed: $(date)"
```

## Troubleshooting

### Common Volume Issues

#### Permission Problems

```bash
# Fix ownership issues
sudo chown -R $USER:$USER models cache generated_images uploads data

# Fix permission issues
chmod -R 755 models cache generated_images uploads data
chmod -R 644 models/**/*.bin models/**/*.safetensors 2>/dev/null || true
```

#### Mount Issues

```bash
# Check if volumes are properly mounted
mount | grep $(pwd)

# Remount if necessary
sudo umount ./models 2>/dev/null || true
sudo mount --bind ./models ./models
```

#### Space Issues

```bash
# Find largest files
find . -type f -size +1G -exec ls -lh {} \; | sort -k5 -hr

# Clean Docker system
docker system prune -f
docker volume prune -f

# Clean build cache
docker builder prune -f
```

#### Corruption Issues

```bash
# Check filesystem
sudo fsck -f /dev/sda1  # Replace with your device

# Check file integrity
find models/ -name "*.bin" -exec sha256sum {} \; > model_checksums.txt
# Compare with known good checksums
```

### Recovery Procedures

#### Volume Recovery

```bash
#!/bin/bash
# Volume recovery procedure

echo "=== Volume Recovery Procedure ==="

# 1. Stop services
echo "Stopping services..."
docker-compose down

# 2. Check filesystem
echo "Checking filesystem..."
sudo fsck -y $(df . | tail -1 | awk '{print $1}')

# 3. Restore from backup if needed
if [[ -f "backups/latest_backup.tar.gz" ]]; then
    echo "Backup available. Restore? (y/N)"
    read -n 1 -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tar -xzf backups/latest_backup.tar.gz
        echo "Restored from backup"
    fi
fi

# 4. Restart services
echo "Restarting services..."
docker-compose up -d

echo "Recovery completed"
```

### Monitoring Integration

#### Prometheus Metrics

```yaml
# Add to prometheus.yml
- job_name: "node-exporter"
  static_configs:
    - targets: ["localhost:9100"]

# Monitor disk usage
- alert: DiskSpaceHigh
  expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.2
  for: 5m
  annotations:
    summary: "Disk space is running low"
```

#### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Qwen2 Volume Monitoring",
    "panels": [
      {
        "title": "Disk Usage",
        "type": "stat",
        "targets": [
          {
            "expr": "100 - (node_filesystem_avail_bytes / node_filesystem_size_bytes * 100)"
          }
        ]
      }
    ]
  }
}
```

This comprehensive volume management guide ensures proper data persistence, backup strategies, and maintenance procedures for the Qwen2 Docker deployment.
