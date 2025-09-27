"""
Managed DiffSynth Service Implementation
Integrates DiffSynth service with ServiceManager for lifecycle management
"""

import logging
import time
import asyncio
from typing import Any, Optional
import psutil
import torch

from src.service_manager import ManagedService, ServiceConfig, ServiceStatus, ServiceHealthStatus
from src.diffsynth_service import DiffSynthService, DiffSynthConfig, DiffSynthServiceStatus
from src.diffsynth_models import ImageEditRequest, ImageEditResponse
from src.resource_manager import ServiceType, ResourcePriority

logger = logging.getLogger(__name__)


class ManagedDiffSynthService(ManagedService):
    """
    Managed wrapper for DiffSynth service with lifecycle management
    """
    
    def __init__(self, diffsynth_config: Optional[DiffSynthConfig] = None):
        """Initialize managed DiffSynth service"""
        # Create service config
        service_config = ServiceConfig(
            service_id="managed_diffsynth_service",
            service_type=ServiceType.DIFFSYNTH_SERVICE,
            priority=ResourcePriority.NORMAL,
            auto_start=True,
            auto_shutdown=True,
            health_check_interval=30.0,
            startup_timeout=120.0,
            shutdown_timeout=60.0,
            max_restart_attempts=3,
            restart_delay=5.0,
            memory_limit_gb=diffsynth_config.max_memory_usage_gb if diffsynth_config else 4.0,
            idle_shutdown_delay=300.0  # 5 minutes
        )
        
        super().__init__(service_config)
        
        # Initialize DiffSynth service
        self.diffsynth_config = diffsynth_config or DiffSynthConfig()
        self.diffsynth_service: Optional[DiffSynthService] = None
        
        # Performance tracking
        self.request_times = []
        self.max_request_history = 100
        
        logger.info(f"ManagedDiffSynthService created: {self.config.service_id}")
    
    async def initialize(self) -> bool:
        """Initialize the DiffSynth service"""
        try:
            logger.info("🚀 Initializing managed DiffSynth service...")
            
            # Create DiffSynth service instance
            self.diffsynth_service = DiffSynthService(self.diffsynth_config)
            
            # Initialize in a thread to avoid blocking
            def init_service():
                return self.diffsynth_service.initialize()
            
            # Run initialization in executor
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, init_service)
            
            if success:
                self.status = ServiceStatus.READY
                logger.info("✅ Managed DiffSynth service initialized successfully")
                return True
            else:
                logger.error("❌ DiffSynth service initialization failed")
                self.status = ServiceStatus.ERROR
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize managed DiffSynth service: {e}")
            self.status = ServiceStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the DiffSynth service"""
        try:
            logger.info("🛑 Shutting down managed DiffSynth service...")
            
            if self.diffsynth_service:
                # Cleanup DiffSynth resources
                def cleanup_service():
                    try:
                        self.diffsynth_service._cleanup_resources()
                        return True
                    except Exception as e:
                        logger.error(f"DiffSynth cleanup failed: {e}")
                        return False
                
                # Run cleanup in executor
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(None, cleanup_service)
                
                if success:
                    logger.info("✅ DiffSynth service cleanup completed")
                else:
                    logger.warning("⚠️ DiffSynth service cleanup had issues")
                
                self.diffsynth_service = None
            
            self.status = ServiceStatus.OFFLINE
            logger.info("✅ Managed DiffSynth service shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to shutdown managed DiffSynth service: {e}")
            self.status = ServiceStatus.ERROR
            return False
    
    async def health_check(self) -> ServiceHealthStatus:
        """Perform health check on DiffSynth service"""
        try:
            if not self.diffsynth_service:
                return ServiceHealthStatus.UNHEALTHY
            
            # Check service status
            if self.diffsynth_service.status == DiffSynthServiceStatus.ERROR:
                return ServiceHealthStatus.UNHEALTHY
            elif self.diffsynth_service.status == DiffSynthServiceStatus.NOT_INITIALIZED:
                return ServiceHealthStatus.UNHEALTHY
            elif self.diffsynth_service.status == DiffSynthServiceStatus.READY:
                # Additional health checks
                health_issues = []
                
                # Check memory usage
                if torch.cuda.is_available():
                    try:
                        gpu_memory_used = torch.cuda.memory_allocated() / 1e9
                        self.metrics.memory_usage_gb = gpu_memory_used
                        
                        if gpu_memory_used > self.config.memory_limit_gb * 0.9:
                            health_issues.append("High GPU memory usage")
                    except Exception as e:
                        logger.debug(f"GPU memory check failed: {e}")
                
                # Check CPU usage
                try:
                    process = psutil.Process()
                    cpu_percent = process.cpu_percent()
                    self.metrics.cpu_usage_percent = cpu_percent
                    
                    if cpu_percent > 90:
                        health_issues.append("High CPU usage")
                except Exception as e:
                    logger.debug(f"CPU usage check failed: {e}")
                
                # Check error rate
                if self.metrics.total_requests > 10:
                    error_rate = self.metrics.failed_requests / self.metrics.total_requests
                    if error_rate > 0.5:  # More than 50% errors
                        health_issues.append("High error rate")
                
                # Determine health status
                if len(health_issues) == 0:
                    return ServiceHealthStatus.HEALTHY
                elif len(health_issues) <= 2:
                    logger.warning(f"Service health degraded: {', '.join(health_issues)}")
                    return ServiceHealthStatus.DEGRADED
                else:
                    logger.error(f"Service unhealthy: {', '.join(health_issues)}")
                    return ServiceHealthStatus.UNHEALTHY
            else:
                return ServiceHealthStatus.DEGRADED
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return ServiceHealthStatus.UNKNOWN
    
    async def process_request(self, request: Any) -> Any:
        """Process a service request"""
        try:
            if not self.diffsynth_service:
                raise RuntimeError("DiffSynth service not initialized")
            
            if self.diffsynth_service.status != DiffSynthServiceStatus.READY:
                raise RuntimeError(f"DiffSynth service not ready: {self.diffsynth_service.status}")
            
            # Update activity
            self.update_activity()
            self.status = ServiceStatus.BUSY
            
            # Track request metrics
            start_time = time.time()
            self.metrics.total_requests += 1
            
            try:
                # Process request based on type
                if isinstance(request, ImageEditRequest):
                    # Run in executor to avoid blocking
                    def process_edit():
                        return self.diffsynth_service.edit_image(request)
                    
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(None, process_edit)
                    
                    # Update metrics
                    processing_time = time.time() - start_time
                    self.request_times.append(processing_time)
                    
                    # Keep only recent request times
                    if len(self.request_times) > self.max_request_history:
                        self.request_times = self.request_times[-self.max_request_history:]
                    
                    # Update average response time
                    self.metrics.average_response_time = sum(self.request_times) / len(self.request_times)
                    
                    if response.success:
                        self.metrics.successful_requests += 1
                        logger.debug(f"Request processed successfully in {processing_time:.2f}s")
                    else:
                        self.metrics.failed_requests += 1
                        logger.warning(f"Request failed: {response.message}")
                    
                    return response
                else:
                    raise ValueError(f"Unsupported request type: {type(request)}")
                    
            except Exception as e:
                self.metrics.failed_requests += 1
                logger.error(f"Request processing failed: {e}")
                
                # Return error response
                if isinstance(request, ImageEditRequest):
                    return ImageEditResponse(
                        success=False,
                        message="Service processing failed",
                        error_details=str(e)
                    )
                else:
                    raise e
            
            finally:
                self.status = ServiceStatus.READY
                
        except Exception as e:
            logger.error(f"Failed to process request: {e}")
            self.status = ServiceStatus.READY  # Reset status
            raise e
    
    def get_service_info(self) -> dict:
        """Get detailed service information"""
        info = {
            "service_id": self.config.service_id,
            "service_type": self.config.service_type.value,
            "status": self.status.value,
            "health_status": self.health_status.value,
            "config": {
                "auto_start": self.config.auto_start,
                "auto_shutdown": self.config.auto_shutdown,
                "memory_limit_gb": self.config.memory_limit_gb,
                "idle_shutdown_delay": self.config.idle_shutdown_delay
            },
            "metrics": {
                "startup_time": self.metrics.startup_time,
                "uptime_seconds": self.metrics.uptime_seconds,
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "average_response_time": self.metrics.average_response_time,
                "memory_usage_gb": self.metrics.memory_usage_gb,
                "cpu_usage_percent": self.metrics.cpu_usage_percent,
                "restart_count": self.metrics.restart_count
            },
            "last_activity": self.last_activity,
            "is_idle": self.is_idle()
        }
        
        # Add DiffSynth-specific info if available
        if self.diffsynth_service:
            info["diffsynth_status"] = self.diffsynth_service.status.value
            info["diffsynth_config"] = {
                "model_name": self.diffsynth_service.config.model_name,
                "device": self.diffsynth_service.config.device,
                "enable_eligen": self.diffsynth_service.config.enable_eligen,
                "use_tiled_processing": self.diffsynth_service.config.use_tiled_processing
            }
            info["operation_count"] = self.diffsynth_service.operation_count
            info["error_count"] = self.diffsynth_service.error_count
        
        return info


class ManagedQwenService(ManagedService):
    """
    Managed wrapper for Qwen service (placeholder for future implementation)
    """
    
    def __init__(self):
        """Initialize managed Qwen service"""
        service_config = ServiceConfig(
            service_id="managed_qwen_service",
            service_type=ServiceType.QWEN_GENERATOR,
            priority=ResourcePriority.HIGH,
            auto_start=True,
            auto_shutdown=False,  # Keep Qwen running
            memory_limit_gb=6.0,
            idle_shutdown_delay=600.0  # 10 minutes
        )
        
        super().__init__(service_config)
        logger.info(f"ManagedQwenService created: {self.config.service_id}")
    
    async def initialize(self) -> bool:
        """Initialize the Qwen service (placeholder)"""
        try:
            logger.info("🚀 Initializing managed Qwen service...")
            
            # Simulate initialization
            await asyncio.sleep(2.0)
            
            self.status = ServiceStatus.READY
            logger.info("✅ Managed Qwen service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize managed Qwen service: {e}")
            self.status = ServiceStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the Qwen service (placeholder)"""
        try:
            logger.info("🛑 Shutting down managed Qwen service...")
            
            # Simulate shutdown
            await asyncio.sleep(1.0)
            
            self.status = ServiceStatus.OFFLINE
            logger.info("✅ Managed Qwen service shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to shutdown managed Qwen service: {e}")
            self.status = ServiceStatus.ERROR
            return False
    
    async def health_check(self) -> ServiceHealthStatus:
        """Perform health check on Qwen service (placeholder)"""
        try:
            # Simple health check
            if self.status == ServiceStatus.READY:
                return ServiceHealthStatus.HEALTHY
            else:
                return ServiceHealthStatus.UNHEALTHY
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return ServiceHealthStatus.UNKNOWN
    
    async def process_request(self, request: Any) -> Any:
        """Process a service request (placeholder)"""
        try:
            self.update_activity()
            self.status = ServiceStatus.BUSY
            
            # Simulate processing
            await asyncio.sleep(0.5)
            
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            
            return {"success": True, "message": "Qwen request processed"}
            
        except Exception as e:
            self.metrics.failed_requests += 1
            logger.error(f"Failed to process Qwen request: {e}")
            raise e
        finally:
            self.status = ServiceStatus.READY