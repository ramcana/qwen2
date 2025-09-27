"""
Resource Manager for GPU Memory Sharing
Manages GPU memory allocation between Qwen and DiffSynth services
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import torch
import psutil

logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Service types for resource management"""
    QWEN_GENERATOR = "qwen_generator"
    DIFFSYNTH_SERVICE = "diffsynth_service"
    CONTROLNET_SERVICE = "controlnet_service"


class ResourcePriority(Enum):
    """Resource allocation priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ServiceResource:
    """Resource allocation for a service"""
    service_type: ServiceType
    service_id: str
    allocated_memory_gb: float = 0.0
    reserved_memory_gb: float = 0.0
    priority: ResourcePriority = ResourcePriority.NORMAL
    last_used: float = 0.0
    is_active: bool = False
    cleanup_callback: Optional[Callable] = None


@dataclass
class ResourceLimits:
    """System resource limits"""
    max_gpu_memory_gb: float
    max_cpu_memory_gb: float
    reserved_system_memory_gb: float = 2.0
    memory_warning_threshold: float = 0.8
    memory_critical_threshold: float = 0.95


class ResourceManager:
    """
    Manages GPU memory allocation between multiple AI services
    Ensures efficient resource sharing and prevents memory conflicts
    """
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        """Initialize resource manager with system limits"""
        self.limits = limits or self._detect_system_limits()
        self.services: Dict[str, ServiceResource] = {}
        self.allocation_lock = threading.Lock()
        
        # Resource monitoring
        self.monitoring_enabled = True
        self.monitoring_interval = 5.0  # seconds
        self.last_cleanup = time.time()
        self.cleanup_interval = 30.0  # seconds
        
        # Callbacks for resource events
        self.memory_warning_callbacks: List[Callable[[float], None]] = []
        self.memory_critical_callbacks: List[Callable[[float], None]] = []
        
        logger.info(f"Resource Manager initialized with {self.limits.max_gpu_memory_gb:.1f}GB GPU memory")
    
    def _detect_system_limits(self) -> ResourceLimits:
        """Detect system resource limits"""
        max_gpu_memory_gb = 0.0
        if torch.cuda.is_available():
            max_gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        max_cpu_memory_gb = psutil.virtual_memory().total / 1e9
        
        return ResourceLimits(
            max_gpu_memory_gb=max_gpu_memory_gb,
            max_cpu_memory_gb=max_cpu_memory_gb
        )
    
    def register_service(
        self,
        service_type: ServiceType,
        service_id: str,
        priority: ResourcePriority = ResourcePriority.NORMAL,
        cleanup_callback: Optional[Callable] = None
    ) -> bool:
        """
        Register a service for resource management
        
        Args:
            service_type: Type of service
            service_id: Unique identifier for the service
            priority: Resource allocation priority
            cleanup_callback: Function to call for resource cleanup
            
        Returns:
            True if registration successful
        """
        with self.allocation_lock:
            if service_id in self.services:
                logger.warning(f"Service {service_id} already registered")
                return False
            
            service_resource = ServiceResource(
                service_type=service_type,
                service_id=service_id,
                priority=priority,
                cleanup_callback=cleanup_callback
            )
            
            self.services[service_id] = service_resource
            logger.info(f"✅ Registered service: {service_id} ({service_type.value})")
            return True
    
    def unregister_service(self, service_id: str) -> bool:
        """
        Unregister a service and free its resources
        
        Args:
            service_id: Service identifier
            
        Returns:
            True if unregistration successful
        """
        with self.allocation_lock:
            if service_id not in self.services:
                logger.warning(f"Service {service_id} not registered")
                return False
            
            service = self.services[service_id]
            
            # Call cleanup callback if available
            if service.cleanup_callback:
                try:
                    service.cleanup_callback()
                except Exception as e:
                    logger.error(f"Cleanup callback failed for {service_id}: {e}")
            
            # Free allocated resources
            self._free_service_resources(service_id)
            
            del self.services[service_id]
            logger.info(f"✅ Unregistered service: {service_id}")
            return True
    
    def request_memory(
        self,
        service_id: str,
        requested_memory_gb: float,
        force: bool = False
    ) -> bool:
        """
        Request GPU memory allocation for a service
        
        Args:
            service_id: Service identifier
            requested_memory_gb: Amount of memory requested in GB
            force: Force allocation even if it requires freeing other services
            
        Returns:
            True if allocation successful
        """
        with self.allocation_lock:
            if service_id not in self.services:
                logger.error(f"Service {service_id} not registered")
                return False
            
            service = self.services[service_id]
            current_usage = self._get_current_gpu_usage()
            available_memory = self.limits.max_gpu_memory_gb - current_usage
            
            logger.info(f"Memory request: {service_id} wants {requested_memory_gb:.1f}GB, available: {available_memory:.1f}GB")
            
            # Check if we have enough available memory
            if available_memory >= requested_memory_gb:
                service.allocated_memory_gb = requested_memory_gb
                service.last_used = time.time()
                service.is_active = True
                logger.info(f"✅ Allocated {requested_memory_gb:.1f}GB to {service_id}")
                return True
            
            # Try to free memory from other services if force is enabled
            if force:
                freed_memory = self._free_memory_for_service(service_id, requested_memory_gb)
                if freed_memory >= requested_memory_gb:
                    service.allocated_memory_gb = requested_memory_gb
                    service.last_used = time.time()
                    service.is_active = True
                    logger.info(f"✅ Allocated {requested_memory_gb:.1f}GB to {service_id} (freed {freed_memory:.1f}GB)")
                    return True
            
            logger.warning(f"❌ Cannot allocate {requested_memory_gb:.1f}GB to {service_id}")
            return False
    
    def release_memory(self, service_id: str) -> bool:
        """
        Release GPU memory allocated to a service
        
        Args:
            service_id: Service identifier
            
        Returns:
            True if release successful
        """
        with self.allocation_lock:
            if service_id not in self.services:
                logger.warning(f"Service {service_id} not registered")
                return False
            
            service = self.services[service_id]
            freed_memory = service.allocated_memory_gb
            
            service.allocated_memory_gb = 0.0
            service.is_active = False
            
            # Call cleanup callback
            if service.cleanup_callback:
                try:
                    service.cleanup_callback()
                except Exception as e:
                    logger.error(f"Cleanup callback failed for {service_id}: {e}")
            
            logger.info(f"✅ Released {freed_memory:.1f}GB from {service_id}")
            return True
    
    def _free_memory_for_service(self, requesting_service_id: str, needed_memory_gb: float) -> float:
        """
        Free memory from other services to accommodate a request
        
        Args:
            requesting_service_id: Service making the request
            needed_memory_gb: Amount of memory needed
            
        Returns:
            Amount of memory freed in GB
        """
        requesting_service = self.services[requesting_service_id]
        freed_memory = 0.0
        
        # Sort services by priority (lower priority first) and last used time
        services_to_consider = [
            (sid, service) for sid, service in self.services.items()
            if sid != requesting_service_id and service.is_active
        ]
        
        services_to_consider.sort(key=lambda x: (x[1].priority.value, x[1].last_used))
        
        for service_id, service in services_to_consider:
            if freed_memory >= needed_memory_gb:
                break
            
            # Only free from lower or equal priority services
            if service.priority.value <= requesting_service.priority.value:
                logger.info(f"🔄 Freeing memory from {service_id} for {requesting_service_id}")
                freed_memory += service.allocated_memory_gb
                self.release_memory(service_id)
        
        return freed_memory
    
    def _free_service_resources(self, service_id: str) -> None:
        """Free all resources allocated to a service"""
        if service_id in self.services:
            service = self.services[service_id]
            service.allocated_memory_gb = 0.0
            service.is_active = False
    
    def _get_current_gpu_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            return torch.cuda.memory_allocated() / 1e9
        except Exception as e:
            logger.warning(f"Failed to get GPU usage: {e}")
            return 0.0
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get detailed memory status"""
        current_gpu_usage = self._get_current_gpu_usage()
        current_cpu_usage = psutil.virtual_memory().used / 1e9
        
        # Calculate allocated memory by services
        total_allocated = sum(service.allocated_memory_gb for service in self.services.values())
        
        # Service breakdown
        service_breakdown = {}
        for service_id, service in self.services.items():
            service_breakdown[service_id] = {
                "type": service.service_type.value,
                "allocated_gb": service.allocated_memory_gb,
                "priority": service.priority.value,
                "is_active": service.is_active,
                "last_used": service.last_used
            }
        
        return {
            "gpu_memory": {
                "total_gb": self.limits.max_gpu_memory_gb,
                "used_gb": current_gpu_usage,
                "allocated_by_services_gb": total_allocated,
                "available_gb": self.limits.max_gpu_memory_gb - current_gpu_usage,
                "usage_percent": (current_gpu_usage / self.limits.max_gpu_memory_gb) * 100 if self.limits.max_gpu_memory_gb > 0 else 0
            },
            "cpu_memory": {
                "total_gb": self.limits.max_cpu_memory_gb,
                "used_gb": current_cpu_usage,
                "available_gb": self.limits.max_cpu_memory_gb - current_cpu_usage,
                "usage_percent": (current_cpu_usage / self.limits.max_cpu_memory_gb) * 100
            },
            "services": service_breakdown,
            "thresholds": {
                "warning_percent": self.limits.memory_warning_threshold * 100,
                "critical_percent": self.limits.memory_critical_threshold * 100
            }
        }
    
    def monitor_resources(self) -> None:
        """Monitor resource usage and trigger callbacks if needed"""
        if not self.monitoring_enabled:
            return
        
        try:
            status = self.get_memory_status()
            gpu_usage_percent = status["gpu_memory"]["usage_percent"] / 100
            
            # Check thresholds
            if gpu_usage_percent >= self.limits.memory_critical_threshold:
                logger.warning(f"🚨 Critical GPU memory usage: {gpu_usage_percent*100:.1f}%")
                for callback in self.memory_critical_callbacks:
                    try:
                        callback(gpu_usage_percent)
                    except Exception as e:
                        logger.error(f"Memory critical callback failed: {e}")
            
            elif gpu_usage_percent >= self.limits.memory_warning_threshold:
                logger.warning(f"⚠️ High GPU memory usage: {gpu_usage_percent*100:.1f}%")
                for callback in self.memory_warning_callbacks:
                    try:
                        callback(gpu_usage_percent)
                    except Exception as e:
                        logger.error(f"Memory warning callback failed: {e}")
            
            # Periodic cleanup
            current_time = time.time()
            if current_time - self.last_cleanup > self.cleanup_interval:
                self._periodic_cleanup()
                self.last_cleanup = current_time
        
        except Exception as e:
            logger.error(f"Resource monitoring failed: {e}")
    
    def _periodic_cleanup(self) -> None:
        """Perform periodic cleanup of unused resources"""
        logger.debug("🧹 Performing periodic resource cleanup")
        
        current_time = time.time()
        inactive_threshold = 300.0  # 5 minutes
        
        with self.allocation_lock:
            for service_id, service in list(self.services.items()):
                # Release memory from services that haven't been used recently
                if (service.is_active and 
                    current_time - service.last_used > inactive_threshold and
                    service.priority != ResourcePriority.CRITICAL):
                    
                    logger.info(f"🔄 Auto-releasing memory from inactive service: {service_id}")
                    self.release_memory(service_id)
        
        # Force GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def add_memory_warning_callback(self, callback: Callable[[float], None]) -> None:
        """Add callback for memory warning events"""
        self.memory_warning_callbacks.append(callback)
    
    def add_memory_critical_callback(self, callback: Callable[[float], None]) -> None:
        """Add callback for memory critical events"""
        self.memory_critical_callbacks.append(callback)
    
    def force_cleanup_all(self) -> None:
        """Force cleanup of all services (emergency function)"""
        logger.warning("🚨 Force cleanup of all services")
        
        with self.allocation_lock:
            for service_id in list(self.services.keys()):
                self.release_memory(service_id)
        
        # Aggressive GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Multiple cleanup passes
            for _ in range(3):
                torch.cuda.empty_cache()
                time.sleep(0.1)
    
    def get_service_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for service resource optimization"""
        status = self.get_memory_status()
        recommendations = []
        
        gpu_usage = status["gpu_memory"]["usage_percent"]
        
        if gpu_usage > 90:
            recommendations.append("Critical: GPU memory usage very high. Consider releasing unused services.")
        elif gpu_usage > 70:
            recommendations.append("Warning: GPU memory usage high. Monitor active services.")
        
        # Check for inactive services
        current_time = time.time()
        inactive_services = []
        for service_id, service in self.services.items():
            if service.is_active and current_time - service.last_used > 300:
                inactive_services.append(service_id)
        
        if inactive_services:
            recommendations.append(f"Consider releasing inactive services: {', '.join(inactive_services)}")
        
        # Check for priority conflicts
        active_services = [s for s in self.services.values() if s.is_active]
        if len(active_services) > 1:
            high_priority_count = sum(1 for s in active_services if s.priority == ResourcePriority.HIGH)
            if high_priority_count > 1:
                recommendations.append("Multiple high-priority services active. Consider prioritization.")
        
        return {
            "recommendations": recommendations,
            "memory_status": status,
            "optimization_suggestions": [
                "Use CPU offloading for inactive models",
                "Enable tiled processing for large images",
                "Consider model quantization for memory savings"
            ]
        }
    
    def optimize_memory_allocation(self) -> Dict[str, Any]:
        """
        Optimize memory allocation across all services
        Returns optimization results and actions taken
        """
        optimization_results = {
            "actions_taken": [],
            "memory_freed_gb": 0.0,
            "services_optimized": [],
            "recommendations": []
        }
        
        current_time = time.time()
        
        with self.allocation_lock:
            # Find services that can be optimized
            for service_id, service in self.services.items():
                if not service.is_active:
                    continue
                
                # Check if service is inactive for too long
                if current_time - service.last_used > 600:  # 10 minutes
                    if service.priority != ResourcePriority.CRITICAL:
                        freed_memory = service.allocated_memory_gb
                        self.release_memory(service_id)
                        optimization_results["actions_taken"].append(f"Released {freed_memory:.1f}GB from inactive service {service_id}")
                        optimization_results["memory_freed_gb"] += freed_memory
                        optimization_results["services_optimized"].append(service_id)
                
                # Suggest CPU offloading for low priority services
                elif service.priority == ResourcePriority.LOW and service.allocated_memory_gb > 2.0:
                    optimization_results["recommendations"].append(f"Consider CPU offloading for {service_id}")
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return optimization_results
    
    def get_workload_analysis(self) -> Dict[str, Any]:
        """
        Analyze current workload and resource usage patterns
        """
        current_time = time.time()
        
        workload_analysis = {
            "total_services": len(self.services),
            "active_services": 0,
            "service_breakdown": {},
            "memory_efficiency": 0.0,
            "usage_patterns": {},
            "bottlenecks": []
        }
        
        total_allocated = 0.0
        actual_usage = self._get_current_gpu_usage()
        
        for service_id, service in self.services.items():
            service_info = {
                "type": service.service_type.value,
                "priority": service.priority.value,
                "allocated_gb": service.allocated_memory_gb,
                "is_active": service.is_active,
                "idle_time": current_time - service.last_used if service.last_used > 0 else 0
            }
            
            workload_analysis["service_breakdown"][service_id] = service_info
            
            if service.is_active:
                workload_analysis["active_services"] += 1
                total_allocated += service.allocated_memory_gb
                
                # Analyze usage patterns
                if service_info["idle_time"] > 300:  # 5 minutes
                    workload_analysis["bottlenecks"].append(f"{service_id} idle for {service_info['idle_time']:.0f}s")
        
        # Calculate memory efficiency
        if total_allocated > 0:
            workload_analysis["memory_efficiency"] = actual_usage / total_allocated
        
        return workload_analysis
    
    def start_monitoring(self) -> None:
        """Start background resource monitoring"""
        if self.monitoring_enabled:
            import threading
            
            def monitoring_loop():
                while self.monitoring_enabled:
                    try:
                        self.monitor_resources()
                        time.sleep(self.monitoring_interval)
                    except Exception as e:
                        logger.error(f"Resource monitoring error: {e}")
                        time.sleep(self.monitoring_interval * 2)  # Back off on error
            
            monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
            monitor_thread.start()
            logger.info("✅ Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background resource monitoring"""
        self.monitoring_enabled = False
        logger.info("🛑 Resource monitoring stopped")


# Global resource manager instance
_global_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager instance"""
    global _global_resource_manager
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager()
    return _global_resource_manager


def initialize_resource_manager(limits: Optional[ResourceLimits] = None) -> ResourceManager:
    """Initialize the global resource manager with custom limits"""
    global _global_resource_manager
    _global_resource_manager = ResourceManager(limits)
    return _global_resource_manager