"""
Service Lifecycle Manager
Simplified service management for AI services with resource coordination
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

try:
    from resource_manager import get_resource_manager, ServiceType, ResourcePriority
except ImportError:
    from src.resource_manager import get_resource_manager, ServiceType, ResourcePriority

logger = logging.getLogger(__name__)


class ServiceState(Enum):
    """Service state enumeration"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ServiceInfo:
    """Service information"""
    service_id: str
    service_type: ServiceType
    priority: ResourcePriority
    state: ServiceState = ServiceState.STOPPED
    memory_allocated_gb: float = 0.0
    last_activity: float = 0.0
    start_time: Optional[float] = None
    error_count: int = 0
    restart_count: int = 0


class ServiceLifecycleManager:
    """
    Manages lifecycle of AI services with resource coordination
    Provides startup, shutdown, health monitoring, and automatic recovery
    """
    
    def __init__(self, resource_manager=None):
        """Initialize service lifecycle manager"""
        self.services: Dict[str, ServiceInfo] = {}
        self.resource_manager = resource_manager or get_resource_manager()
        self.is_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Callbacks
        self.service_callbacks: Dict[str, List[Callable]] = {
            "started": [],
            "stopped": [],
            "error": []
        }
        
        logger.info("ServiceLifecycleManager initialized")
    
    def register_service(
        self, 
        service_id: str, 
        service_type: ServiceType, 
        priority: ResourcePriority = ResourcePriority.NORMAL
    ) -> bool:
        """Register a service for lifecycle management"""
        try:
            if service_id in self.services:
                logger.warning(f"Service {service_id} already registered")
                return False
            
            # Create service info
            service_info = ServiceInfo(
                service_id=service_id,
                service_type=service_type,
                priority=priority,
                last_activity=time.time()
            )
            
            self.services[service_id] = service_info
            
            # Register with resource manager
            success = self.resource_manager.register_service(
                service_type=service_type,
                service_id=service_id,
                priority=priority,
                cleanup_callback=lambda: self._cleanup_service(service_id)
            )
            
            if success:
                logger.info(f"✅ Registered service: {service_id}")
                return True
            else:
                # Remove from our tracking if resource manager registration failed
                del self.services[service_id]
                logger.error(f"❌ Failed to register {service_id} with resource manager")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to register service {service_id}: {e}")
            return False
    
    def unregister_service(self, service_id: str) -> bool:
        """Unregister a service"""
        try:
            if service_id not in self.services:
                logger.warning(f"Service {service_id} not registered")
                return False
            
            # Stop service if running
            if self.services[service_id].state == ServiceState.RUNNING:
                self.stop_service(service_id)
            
            # Unregister from resource manager
            self.resource_manager.unregister_service(service_id)
            
            # Remove from tracking
            del self.services[service_id]
            
            logger.info(f"✅ Unregistered service: {service_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to unregister service {service_id}: {e}")
            return False
    
    def start_service(self, service_id: str, memory_gb: float = 4.0) -> bool:
        """Start a service with memory allocation"""
        try:
            if service_id not in self.services:
                logger.error(f"Service {service_id} not registered")
                return False
            
            service_info = self.services[service_id]
            
            if service_info.state == ServiceState.RUNNING:
                logger.info(f"Service {service_id} already running")
                return True
            
            logger.info(f"🚀 Starting service: {service_id}")
            service_info.state = ServiceState.STARTING
            
            # Request memory allocation
            memory_allocated = self.resource_manager.request_memory(
                service_id=service_id,
                requested_memory_gb=memory_gb,
                force=False
            )
            
            if not memory_allocated:
                logger.error(f"❌ Failed to allocate {memory_gb}GB for {service_id}")
                service_info.state = ServiceState.ERROR
                service_info.error_count += 1
                return False
            
            # Simulate service startup (in real implementation, this would initialize the actual service)
            time.sleep(0.1)  # Simulate startup time
            
            # Update service state
            service_info.state = ServiceState.RUNNING
            service_info.memory_allocated_gb = memory_gb
            service_info.start_time = time.time()
            service_info.last_activity = time.time()
            
            logger.info(f"✅ Service {service_id} started successfully")
            
            # Notify callbacks
            for callback in self.service_callbacks["started"]:
                try:
                    callback(service_id)
                except Exception as e:
                    logger.error(f"Service started callback failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start service {service_id}: {e}")
            if service_id in self.services:
                self.services[service_id].state = ServiceState.ERROR
                self.services[service_id].error_count += 1
            return False
    
    def stop_service(self, service_id: str) -> bool:
        """Stop a service and release resources"""
        try:
            if service_id not in self.services:
                logger.error(f"Service {service_id} not registered")
                return False
            
            service_info = self.services[service_id]
            
            if service_info.state == ServiceState.STOPPED:
                logger.info(f"Service {service_id} already stopped")
                return True
            
            logger.info(f"🛑 Stopping service: {service_id}")
            service_info.state = ServiceState.STOPPING
            
            # Simulate service shutdown
            time.sleep(0.1)  # Simulate shutdown time
            
            # Release memory allocation
            self.resource_manager.release_memory(service_id)
            
            # Update service state
            service_info.state = ServiceState.STOPPED
            service_info.memory_allocated_gb = 0.0
            service_info.start_time = None
            
            logger.info(f"✅ Service {service_id} stopped successfully")
            
            # Notify callbacks
            for callback in self.service_callbacks["stopped"]:
                try:
                    callback(service_id)
                except Exception as e:
                    logger.error(f"Service stopped callback failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to stop service {service_id}: {e}")
            if service_id in self.services:
                self.services[service_id].state = ServiceState.ERROR
                self.services[service_id].error_count += 1
            return False
    
    def restart_service(self, service_id: str) -> bool:
        """Restart a service"""
        try:
            if service_id not in self.services:
                logger.error(f"Service {service_id} not registered")
                return False
            
            logger.info(f"🔄 Restarting service: {service_id}")
            
            # Get current memory allocation
            current_memory = self.services[service_id].memory_allocated_gb
            
            # Stop service
            stop_success = self.stop_service(service_id)
            if not stop_success:
                logger.error(f"❌ Failed to stop service {service_id} for restart")
                return False
            
            # Wait a moment
            time.sleep(0.5)
            
            # Start service with same memory allocation
            start_success = self.start_service(service_id, current_memory or 4.0)
            if start_success:
                self.services[service_id].restart_count += 1
                logger.info(f"✅ Service {service_id} restarted successfully")
            else:
                logger.error(f"❌ Failed to start service {service_id} after restart")
            
            return start_success
            
        except Exception as e:
            logger.error(f"❌ Failed to restart service {service_id}: {e}")
            return False
    
    def _cleanup_service(self, service_id: str) -> None:
        """Cleanup callback for resource manager"""
        try:
            if service_id in self.services:
                logger.info(f"🧹 Cleaning up service: {service_id}")
                self.stop_service(service_id)
        except Exception as e:
            logger.error(f"Service cleanup failed for {service_id}: {e}")
    
    def get_service_status(self, service_id: Optional[str] = None) -> Dict[str, Any]:
        """Get service status information"""
        if service_id:
            if service_id not in self.services:
                return {"error": f"Service {service_id} not found"}
            
            service_info = self.services[service_id]
            uptime = time.time() - service_info.start_time if service_info.start_time else 0.0
            
            return {
                "service_id": service_id,
                "service_type": service_info.service_type.value,
                "priority": service_info.priority.value,
                "state": service_info.state.value,
                "memory_allocated_gb": service_info.memory_allocated_gb,
                "uptime_seconds": uptime,
                "last_activity": service_info.last_activity,
                "error_count": service_info.error_count,
                "restart_count": service_info.restart_count
            }
        else:
            # Return status for all services
            all_status = {}
            for sid, service_info in self.services.items():
                uptime = time.time() - service_info.start_time if service_info.start_time else 0.0
                all_status[sid] = {
                    "state": service_info.state.value,
                    "memory_allocated_gb": service_info.memory_allocated_gb,
                    "uptime_seconds": uptime
                }
            
            return {
                "services": all_status,
                "total_services": len(self.services),
                "running_services": len([s for s in self.services.values() if s.state == ServiceState.RUNNING]),
                "management_active": self.is_active
            }
    
    def start_monitoring(self) -> None:
        """Start service monitoring"""
        if self.is_active:
            logger.warning("Service monitoring already active")
            return
        
        self.is_active = True
        self.shutdown_event.clear()
        
        def monitoring_loop():
            logger.info("🔄 Service monitoring started")
            while not self.shutdown_event.is_set():
                try:
                    self._perform_health_checks()
                    time.sleep(10.0)  # Check every 10 seconds
                except Exception as e:
                    logger.error(f"Monitoring loop error: {e}")
                    time.sleep(20.0)  # Back off on error
            logger.info("🛑 Service monitoring stopped")
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("✅ Service monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop service monitoring"""
        if not self.is_active:
            logger.warning("Service monitoring not active")
            return
        
        logger.info("🛑 Stopping service monitoring...")
        
        self.is_active = False
        self.shutdown_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("✅ Service monitoring stopped")
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on services"""
        try:
            current_time = time.time()
            
            for service_id, service_info in self.services.items():
                if service_info.state != ServiceState.RUNNING:
                    continue
                
                # Check if service has been idle too long (5 minutes)
                idle_time = current_time - service_info.last_activity
                if idle_time > 300:  # 5 minutes
                    logger.info(f"💤 Service {service_id} idle for {idle_time:.0f}s, considering shutdown")
                    # In a real implementation, you might auto-shutdown idle services
                
                # Simulate health check (in real implementation, this would ping the actual service)
                # For now, just update activity if service is running
                if service_info.state == ServiceState.RUNNING:
                    service_info.last_activity = current_time
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def add_callback(self, event_type: str, callback: Callable) -> None:
        """Add callback for service events"""
        if event_type in self.service_callbacks:
            self.service_callbacks[event_type].append(callback)
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    def shutdown_all_services(self) -> None:
        """Shutdown all services"""
        logger.info("🛑 Shutting down all services...")
        
        for service_id in list(self.services.keys()):
            if self.services[service_id].state == ServiceState.RUNNING:
                self.stop_service(service_id)
        
        logger.info("✅ All services shutdown complete")


# Global service lifecycle manager instance
_global_service_lifecycle_manager: Optional[ServiceLifecycleManager] = None


def get_service_lifecycle_manager() -> ServiceLifecycleManager:
    """Get the global service lifecycle manager instance"""
    global _global_service_lifecycle_manager
    if _global_service_lifecycle_manager is None:
        _global_service_lifecycle_manager = ServiceLifecycleManager()
    return _global_service_lifecycle_manager


def initialize_service_lifecycle_manager() -> ServiceLifecycleManager:
    """Initialize the global service lifecycle manager"""
    global _global_service_lifecycle_manager
    _global_service_lifecycle_manager = ServiceLifecycleManager()
    return _global_service_lifecycle_manager