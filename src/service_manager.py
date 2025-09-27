"""
Service Manager for AI Service Lifecycle Management
Coordinates multiple AI services with automatic startup, shutdown, and health monitoring
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
from abc import ABC, abstractmethod

try:
    from src.resource_manager import get_resource_manager, ServiceType, ResourcePriority
except ImportError:
    from resource_manager import get_resource_manager, ServiceType, ResourcePriority

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration"""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    OFFLINE = "offline"


class ServiceHealthStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceConfig:
    """Configuration for a managed service"""
    service_id: str
    service_type: ServiceType
    priority: ResourcePriority = ResourcePriority.NORMAL
    auto_start: bool = True
    auto_shutdown: bool = True
    health_check_interval: float = 30.0  # seconds
    startup_timeout: float = 120.0  # seconds
    shutdown_timeout: float = 60.0  # seconds
    max_restart_attempts: int = 3
    restart_delay: float = 5.0  # seconds
    memory_limit_gb: float = 4.0
    idle_shutdown_delay: float = 300.0  # 5 minutes


@dataclass
class ServiceMetrics:
    """Service performance and health metrics"""
    startup_time: float = 0.0
    last_health_check: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    memory_usage_gb: float = 0.0
    cpu_usage_percent: float = 0.0
    uptime_seconds: float = 0.0
    restart_count: int = 0


class ManagedService(ABC):
    """Abstract base class for managed services"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.status = ServiceStatus.NOT_INITIALIZED
        self.health_status = ServiceHealthStatus.UNKNOWN
        self.metrics = ServiceMetrics()
        self.last_activity = time.time()
        self.startup_time: Optional[float] = None
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the service"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the service"""
        pass
    
    @abstractmethod
    async def health_check(self) -> ServiceHealthStatus:
        """Perform health check"""
        pass
    
    @abstractmethod
    async def process_request(self, request: Any) -> Any:
        """Process a service request"""
        pass
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
    
    def is_idle(self) -> bool:
        """Check if service is idle"""
        return (time.time() - self.last_activity) > self.config.idle_shutdown_delay


class ServiceManager:
    """
    Manages lifecycle of multiple AI services
    Provides automatic startup, shutdown, health monitoring, and recovery
    """
    
    def __init__(self):
        """Initialize service manager"""
        self.services: Dict[str, ManagedService] = {}
        self.service_configs: Dict[str, ServiceConfig] = {}
        self.resource_manager = get_resource_manager()
        
        # Management state
        self.is_running = False
        self.management_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Monitoring settings
        self.monitoring_interval = 10.0  # seconds
        self.health_check_interval = 30.0  # seconds
        
        # Callbacks
        self.service_started_callbacks: List[Callable[[str], None]] = []
        self.service_stopped_callbacks: List[Callable[[str], None]] = []
        self.service_error_callbacks: List[Callable[[str, Exception], None]] = []
        
        logger.info("ServiceManager initialized")
    
    def register_service(self, service: ManagedService) -> bool:
        """
        Register a service for management
        
        Args:
            service: ManagedService instance
            
        Returns:
            True if registration successful
        """
        try:
            service_id = service.config.service_id
            
            if service_id in self.services:
                logger.warning(f"Service {service_id} already registered")
                return False
            
            self.services[service_id] = service
            self.service_configs[service_id] = service.config
            
            # Register with resource manager
            self.resource_manager.register_service(
                service_type=service.config.service_type,
                service_id=service_id,
                priority=service.config.priority,
                cleanup_callback=lambda: self._cleanup_service(service_id)
            )
            
            logger.info(f"✅ Registered service: {service_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register service {service.config.service_id}: {e}")
            return False
    
    def unregister_service(self, service_id: str) -> bool:
        """
        Unregister a service
        
        Args:
            service_id: Service identifier
            
        Returns:
            True if unregistration successful
        """
        try:
            if service_id not in self.services:
                logger.warning(f"Service {service_id} not registered")
                return False
            
            # Unregister from resource manager
            self.resource_manager.unregister_service(service_id)
            
            # Remove from tracking
            del self.services[service_id]
            del self.service_configs[service_id]
            
            logger.info(f"✅ Unregistered service: {service_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to unregister service {service_id}: {e}")
            return False
    
    async def start_service(self, service_id: str) -> bool:
        """
        Start a specific service
        
        Args:
            service_id: Service identifier
            
        Returns:
            True if startup successful
        """
        try:
            if service_id not in self.services:
                logger.error(f"Service {service_id} not registered")
                return False
            
            service = self.services[service_id]
            config = self.service_configs[service_id]
            
            if service.status in [ServiceStatus.READY, ServiceStatus.BUSY]:
                logger.info(f"Service {service_id} already running")
                return True
            
            logger.info(f"🚀 Starting service: {service_id}")
            service.status = ServiceStatus.INITIALIZING
            
            # Request memory allocation
            memory_allocated = self.resource_manager.request_memory(
                service_id=service_id,
                requested_memory_gb=config.memory_limit_gb,
                force=False
            )
            
            if not memory_allocated:
                logger.error(f"❌ Failed to allocate memory for service {service_id}")
                service.status = ServiceStatus.ERROR
                return False
            
            # Initialize service with timeout
            start_time = time.time()
            try:
                initialization_task = asyncio.create_task(service.initialize())
                success = await asyncio.wait_for(initialization_task, timeout=config.startup_timeout)
                
                if success:
                    service.status = ServiceStatus.READY
                    service.startup_time = time.time()
                    service.metrics.startup_time = time.time() - start_time
                    service.metrics.uptime_seconds = 0.0
                    
                    logger.info(f"✅ Service {service_id} started successfully in {service.metrics.startup_time:.2f}s")
                    
                    # Notify callbacks
                    for callback in self.service_started_callbacks:
                        try:
                            callback(service_id)
                        except Exception as e:
                            logger.error(f"Service started callback failed: {e}")
                    
                    return True
                else:
                    logger.error(f"❌ Service {service_id} initialization returned False")
                    service.status = ServiceStatus.ERROR
                    self.resource_manager.release_memory(service_id)
                    return False
                    
            except asyncio.TimeoutError:
                logger.error(f"❌ Service {service_id} startup timeout ({config.startup_timeout}s)")
                service.status = ServiceStatus.ERROR
                self.resource_manager.release_memory(service_id)
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to start service {service_id}: {e}")
            if service_id in self.services:
                self.services[service_id].status = ServiceStatus.ERROR
            return False
    
    async def stop_service(self, service_id: str) -> bool:
        """
        Stop a specific service
        
        Args:
            service_id: Service identifier
            
        Returns:
            True if shutdown successful
        """
        try:
            if service_id not in self.services:
                logger.error(f"Service {service_id} not registered")
                return False
            
            service = self.services[service_id]
            config = self.service_configs[service_id]
            
            if service.status == ServiceStatus.OFFLINE:
                logger.info(f"Service {service_id} already offline")
                return True
            
            logger.info(f"🛑 Stopping service: {service_id}")
            service.status = ServiceStatus.SHUTTING_DOWN
            
            # Shutdown service with timeout
            try:
                shutdown_task = asyncio.create_task(service.shutdown())
                success = await asyncio.wait_for(shutdown_task, timeout=config.shutdown_timeout)
                
                if success:
                    service.status = ServiceStatus.OFFLINE
                    
                    # Release memory allocation
                    self.resource_manager.release_memory(service_id)
                    
                    logger.info(f"✅ Service {service_id} stopped successfully")
                    
                    # Notify callbacks
                    for callback in self.service_stopped_callbacks:
                        try:
                            callback(service_id)
                        except Exception as e:
                            logger.error(f"Service stopped callback failed: {e}")
                    
                    return True
                else:
                    logger.error(f"❌ Service {service_id} shutdown returned False")
                    service.status = ServiceStatus.ERROR
                    return False
                    
            except asyncio.TimeoutError:
                logger.error(f"❌ Service {service_id} shutdown timeout ({config.shutdown_timeout}s)")
                service.status = ServiceStatus.ERROR
                # Force release memory even on timeout
                self.resource_manager.release_memory(service_id)
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to stop service {service_id}: {e}")
            if service_id in self.services:
                self.services[service_id].status = ServiceStatus.ERROR
            return False
    
    def _cleanup_service(self, service_id: str) -> None:
        """Cleanup callback for resource manager"""
        try:
            if service_id in self.services:
                service = self.services[service_id]
                if service.status != ServiceStatus.OFFLINE:
                    logger.info(f"🧹 Cleaning up service: {service_id}")
                    # Note: In a real implementation, we'd need to handle async cleanup properly
        except Exception as e:
            logger.error(f"Service cleanup failed for {service_id}: {e}")
    
    def get_service_status(self, service_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of services
        
        Args:
            service_id: Specific service ID, or None for all services
            
        Returns:
            Service status information
        """
        if service_id:
            if service_id not in self.services:
                return {"error": f"Service {service_id} not found"}
            
            service = self.services[service_id]
            return {
                "service_id": service_id,
                "status": service.status.value,
                "health_status": service.health_status.value,
                "metrics": {
                    "startup_time": service.metrics.startup_time,
                    "uptime_seconds": service.metrics.uptime_seconds,
                    "total_requests": service.metrics.total_requests,
                    "successful_requests": service.metrics.successful_requests,
                    "failed_requests": service.metrics.failed_requests,
                    "restart_count": service.metrics.restart_count,
                    "memory_usage_gb": service.metrics.memory_usage_gb
                },
                "last_activity": service.last_activity,
                "is_idle": service.is_idle()
            }
        else:
            # Return status for all services
            all_status = {}
            for sid, service in self.services.items():
                all_status[sid] = {
                    "status": service.status.value,
                    "health_status": service.health_status.value,
                    "uptime_seconds": service.metrics.uptime_seconds,
                    "is_idle": service.is_idle()
                }
            
            return {
                "services": all_status,
                "total_services": len(self.services),
                "running_services": len([s for s in self.services.values() if s.status == ServiceStatus.READY]),
                "management_active": self.is_running
            }
    
    def start_management(self) -> None:
        """Start service management"""
        if self.is_running:
            logger.warning("Service management already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        logger.info("✅ Service management started")
    
    def stop_management(self) -> None:
        """Stop service management"""
        if not self.is_running:
            logger.warning("Service management not running")
            return
        
        logger.info("🛑 Stopping service management...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        logger.info("✅ Service management stopped")
    
    def add_service_started_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for service started events"""
        self.service_started_callbacks.append(callback)
    
    def add_service_stopped_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for service stopped events"""
        self.service_stopped_callbacks.append(callback)
    
    def add_service_error_callback(self, callback: Callable[[str, Exception], None]) -> None:
        """Add callback for service error events"""
        self.service_error_callbacks.append(callback)


# Global service manager instance
_global_service_manager: Optional[ServiceManager] = None


def get_service_manager() -> ServiceManager:
    """Get the global service manager instance"""
    global _global_service_manager
    if _global_service_manager is None:
        _global_service_manager = ServiceManager()
    return _global_service_manager


def initialize_service_manager() -> ServiceManager:
    """Initialize the global service manager"""
    global _global_service_manager
    _global_service_manager = ServiceManager()
    return _global_service_manager