#!/usr/bin/env python3
"""
Monitoring and Health Check Configuration for Qwen-Image API
Provides comprehensive monitoring, logging, and health check utilities
"""

import os
import time
import psutil
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum

class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class ServiceStatus(Enum):
    """Service status enumeration"""
    STARTING = "starting"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class HealthCheckResult:
    """Health check result data structure"""
    status: HealthStatus
    timestamp: datetime
    service_name: str
    version: str
    uptime_seconds: float
    checks: Dict[str, bool]
    metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]

@dataclass
class SystemMetrics:
    """System metrics data structure"""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    gpu_memory_percent: Optional[float] = None
    gpu_memory_free_gb: Optional[float] = None
    gpu_temperature: Optional[float] = None

class MonitoringConfig:
    """Configuration for monitoring and health checks"""
    
    def __init__(self):
        self.service_name = "qwen-api"
        self.version = "2.0.0"
        self.startup_time = time.time()
        
        # Health check intervals (seconds)
        self.health_check_interval = 30
        self.metrics_collection_interval = 60
        self.log_rotation_interval = 3600  # 1 hour
        
        # Thresholds for health status
        self.cpu_threshold_warning = 80.0
        self.cpu_threshold_critical = 95.0
        self.memory_threshold_warning = 80.0
        self.memory_threshold_critical = 95.0
        self.disk_threshold_warning = 85.0
        self.disk_threshold_critical = 95.0
        self.gpu_memory_threshold_warning = 90.0
        self.gpu_memory_threshold_critical = 98.0
        
        # Directories to monitor
        self.monitored_directories = [
            "generated_images",
            "uploads", 
            "cache",
            "models",
            "logs"
        ]
        
        # Log configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_format = os.getenv("LOG_FORMAT", "json")
        self.log_file = "logs/monitoring.log"
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup structured logging for monitoring"""
        os.makedirs("logs", exist_ok=True)
        
        # Create formatter based on configuration
        if self.log_format.lower() == "json":
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Setup file handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=100*1024*1024,  # 100MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Configure logger
        self.logger = logging.getLogger("qwen.monitoring")
        self.logger.setLevel(getattr(logging, self.log_level))
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)

class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = config.logger
        self._last_check_time = 0
        self._cached_result: Optional[HealthCheckResult] = None
    
    async def perform_health_check(self, 
                                 generator=None, 
                                 diffsynth_service=None, 
                                 controlnet_service=None,
                                 generation_queue=None,
                                 initialization_status=None) -> HealthCheckResult:
        """Perform comprehensive health check"""
        
        current_time = time.time()
        
        # Use cached result if recent enough (within 10 seconds)
        if (self._cached_result and 
            current_time - self._last_check_time < 10):
            return self._cached_result
        
        errors = []
        warnings = []
        checks = {}
        metrics = {}
        
        try:
            # Basic service checks
            checks["api_server_running"] = True  # If we're here, it's running
            checks["startup_completed"] = initialization_status.get("status") == "ready" if initialization_status else False
            
            # Model availability checks
            checks["qwen_model_loaded"] = generator is not None and getattr(generator, 'pipe', None) is not None
            checks["diffsynth_available"] = diffsynth_service is not None
            checks["controlnet_available"] = controlnet_service is not None
            
            # Directory accessibility checks
            for directory in self.config.monitored_directories:
                check_name = f"directory_{directory}_accessible"
                try:
                    checks[check_name] = os.path.exists(directory) and os.access(directory, os.R_OK | os.W_OK)
                    if not checks[check_name]:
                        warnings.append(f"Directory {directory} is not accessible")
                except Exception as e:
                    checks[check_name] = False
                    errors.append(f"Failed to check directory {directory}: {str(e)}")
            
            # System metrics
            system_metrics = await self._collect_system_metrics()
            metrics.update(asdict(system_metrics))
            
            # GPU checks
            if self._is_gpu_available():
                gpu_metrics = await self._collect_gpu_metrics()
                metrics.update(gpu_metrics)
                checks["gpu_available"] = True
                
                # GPU health checks
                if gpu_metrics.get("memory_usage_percent", 0) > self.config.gpu_memory_threshold_critical:
                    errors.append("GPU memory usage critically high")
                elif gpu_metrics.get("memory_usage_percent", 0) > self.config.gpu_memory_threshold_warning:
                    warnings.append("GPU memory usage high")
            else:
                checks["gpu_available"] = False
                warnings.append("GPU not available")
            
            # Queue health checks
            if generation_queue:
                queue_length = len(generation_queue)
                metrics["queue_length"] = queue_length
                metrics["active_jobs"] = len([job for job in generation_queue.values() 
                                            if job.get("status") == "processing"])
                
                checks["queue_healthy"] = queue_length < 100  # Arbitrary threshold
                if queue_length > 50:
                    warnings.append(f"Generation queue is large: {queue_length} items")
            
            # System resource health checks
            if system_metrics.cpu_percent > self.config.cpu_threshold_critical:
                errors.append(f"CPU usage critically high: {system_metrics.cpu_percent}%")
            elif system_metrics.cpu_percent > self.config.cpu_threshold_warning:
                warnings.append(f"CPU usage high: {system_metrics.cpu_percent}%")
            
            if system_metrics.memory_percent > self.config.memory_threshold_critical:
                errors.append(f"Memory usage critically high: {system_metrics.memory_percent}%")
            elif system_metrics.memory_percent > self.config.memory_threshold_warning:
                warnings.append(f"Memory usage high: {system_metrics.memory_percent}%")
            
            if system_metrics.disk_usage_percent > self.config.disk_threshold_critical:
                errors.append(f"Disk usage critically high: {system_metrics.disk_usage_percent}%")
            elif system_metrics.disk_usage_percent > self.config.disk_threshold_warning:
                warnings.append(f"Disk usage high: {system_metrics.disk_usage_percent}%")
            
            # Determine overall health status
            if errors:
                status = HealthStatus.UNHEALTHY
            elif warnings:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            # Create result
            result = HealthCheckResult(
                status=status,
                timestamp=datetime.now(),
                service_name=self.config.service_name,
                version=self.config.version,
                uptime_seconds=current_time - self.config.startup_time,
                checks=checks,
                metrics=metrics,
                errors=errors,
                warnings=warnings
            )
            
            # Cache result
            self._cached_result = result
            self._last_check_time = current_time
            
            # Log health check result
            self.logger.info("Health check completed", extra={
                "health_status": status.value,
                "checks_passed": sum(checks.values()),
                "total_checks": len(checks),
                "errors_count": len(errors),
                "warnings_count": len(warnings)
            })
            
            return result
            
        except Exception as e:
            error_msg = f"Health check failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                timestamp=datetime.now(),
                service_name=self.config.service_name,
                version=self.config.version,
                uptime_seconds=current_time - self.config.startup_time,
                checks={"health_check_failed": False},
                metrics={},
                errors=[error_msg],
                warnings=[]
            )
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_gb=round(memory_available_gb, 2),
                disk_usage_percent=round(disk_usage_percent, 1),
                disk_free_gb=round(disk_free_gb, 2)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {str(e)}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_gb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0
            )
    
    async def _collect_gpu_metrics(self) -> Dict[str, Any]:
        """Collect GPU metrics"""
        try:
            import torch
            if not torch.cuda.is_available():
                return {}
            
            # Clear cache for accurate measurements
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            device_props = torch.cuda.get_device_properties(0)
            total_memory = device_props.total_memory
            allocated_memory = torch.cuda.memory_allocated()
            free_memory = total_memory - allocated_memory
            
            metrics = {
                "gpu_name": device_props.name,
                "gpu_memory_total_gb": round(total_memory / (1024**3), 2),
                "gpu_memory_allocated_gb": round(allocated_memory / (1024**3), 2),
                "gpu_memory_free_gb": round(free_memory / (1024**3), 2),
                "memory_usage_percent": round((allocated_memory / total_memory) * 100, 1)
            }
            
            # Try to get GPU temperature (requires nvidia-ml-py)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics["gpu_temperature_celsius"] = temp
            except ImportError:
                pass  # nvidia-ml-py not available
            except Exception as e:
                self.logger.debug(f"Could not get GPU temperature: {str(e)}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect GPU metrics: {str(e)}")
            return {}
    
    def _is_gpu_available(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

class MetricsCollector:
    """Metrics collection and aggregation system"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = config.logger
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000  # Keep last 1000 metric points
    
    async def collect_metrics(self, health_result: HealthCheckResult):
        """Collect and store metrics"""
        try:
            metric_point = {
                "timestamp": health_result.timestamp.isoformat(),
                "status": health_result.status.value,
                "uptime_seconds": health_result.uptime_seconds,
                **health_result.metrics
            }
            
            # Add to history
            self.metrics_history.append(metric_point)
            
            # Trim history if too large
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size:]
            
            # Log metrics
            self.logger.info("Metrics collected", extra=metric_point)
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {str(e)}")
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_metrics = [
                m for m in self.metrics_history
                if datetime.fromisoformat(m["timestamp"]) > cutoff_time
            ]
            
            if not recent_metrics:
                return {"message": "No metrics available for the specified period"}
            
            # Calculate averages and extremes
            summary = {
                "period_hours": hours,
                "data_points": len(recent_metrics),
                "time_range": {
                    "start": recent_metrics[0]["timestamp"],
                    "end": recent_metrics[-1]["timestamp"]
                }
            }
            
            # Numeric metrics aggregation
            numeric_fields = ["cpu_percent", "memory_percent", "disk_usage_percent", 
                            "gpu_memory_usage_percent", "queue_length"]
            
            for field in numeric_fields:
                values = [m.get(field) for m in recent_metrics if m.get(field) is not None]
                if values:
                    summary[field] = {
                        "avg": round(sum(values) / len(values), 2),
                        "min": min(values),
                        "max": max(values),
                        "current": values[-1] if values else None
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate metrics summary: {str(e)}")
            return {"error": str(e)}

# Global monitoring instances
monitoring_config = MonitoringConfig()
health_checker = HealthChecker(monitoring_config)
metrics_collector = MetricsCollector(monitoring_config)