"""
Unit tests for Resource Manager
Tests GPU memory sharing and resource allocation between services
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import threading

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from resource_manager import (
    ResourceManager, ServiceType, ResourcePriority, ServiceResource,
    ResourceLimits, get_resource_manager, initialize_resource_manager
)


class TestResourceLimits(unittest.TestCase):
    """Test ResourceLimits dataclass"""
    
    def test_resource_limits_creation(self):
        """Test ResourceLimits creation"""
        limits = ResourceLimits(
            max_gpu_memory_gb=16.0,
            max_cpu_memory_gb=32.0,
            reserved_system_memory_gb=4.0
        )
        
        self.assertEqual(limits.max_gpu_memory_gb, 16.0)
        self.assertEqual(limits.max_cpu_memory_gb, 32.0)
        self.assertEqual(limits.reserved_system_memory_gb, 4.0)
        self.assertEqual(limits.memory_warning_threshold, 0.8)
        self.assertEqual(limits.memory_critical_threshold, 0.95)


class TestServiceResource(unittest.TestCase):
    """Test ServiceResource dataclass"""
    
    def test_service_resource_creation(self):
        """Test ServiceResource creation"""
        cleanup_callback = Mock()
        
        resource = ServiceResource(
            service_type=ServiceType.QWEN_GENERATOR,
            service_id="test_service",
            allocated_memory_gb=2.0,
            priority=ResourcePriority.HIGH,
            cleanup_callback=cleanup_callback
        )
        
        self.assertEqual(resource.service_type, ServiceType.QWEN_GENERATOR)
        self.assertEqual(resource.service_id, "test_service")
        self.assertEqual(resource.allocated_memory_gb, 2.0)
        self.assertEqual(resource.priority, ResourcePriority.HIGH)
        self.assertEqual(resource.cleanup_callback, cleanup_callback)
        self.assertFalse(resource.is_active)
        self.assertEqual(resource.reserved_memory_gb, 0.0)


class TestResourceManager(unittest.TestCase):
    """Test ResourceManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.limits = ResourceLimits(
            max_gpu_memory_gb=8.0,
            max_cpu_memory_gb=16.0
        )
        self.manager = ResourceManager(self.limits)
    
    def tearDown(self):
        """Clean up after tests"""
        # Clear any registered services
        with self.manager.allocation_lock:
            self.manager.services.clear()
    
    @patch('resource_manager.torch.cuda.is_available')
    @patch('resource_manager.torch.cuda.get_device_properties')
    @patch('resource_manager.psutil.virtual_memory')
    def test_detect_system_limits(self, mock_virtual_memory, mock_device_props, mock_cuda_available):
        """Test system limits detection"""
        mock_cuda_available.return_value = True
        
        mock_props = Mock()
        mock_props.total_memory = 16 * 1e9  # 16GB
        mock_device_props.return_value = mock_props
        
        mock_memory = Mock()
        mock_memory.total = 32 * 1e9  # 32GB
        mock_virtual_memory.return_value = mock_memory
        
        manager = ResourceManager()
        
        self.assertEqual(manager.limits.max_gpu_memory_gb, 16.0)
        self.assertEqual(manager.limits.max_cpu_memory_gb, 32.0)
    
    def test_register_service(self):
        """Test service registration"""
        cleanup_callback = Mock()
        
        result = self.manager.register_service(
            ServiceType.QWEN_GENERATOR,
            "qwen_service",
            ResourcePriority.HIGH,
            cleanup_callback
        )
        
        self.assertTrue(result)
        self.assertIn("qwen_service", self.manager.services)
        
        service = self.manager.services["qwen_service"]
        self.assertEqual(service.service_type, ServiceType.QWEN_GENERATOR)
        self.assertEqual(service.priority, ResourcePriority.HIGH)
        self.assertEqual(service.cleanup_callback, cleanup_callback)
    
    def test_register_duplicate_service(self):
        """Test registering duplicate service"""
        self.manager.register_service(ServiceType.QWEN_GENERATOR, "test_service")
        
        # Try to register again
        result = self.manager.register_service(ServiceType.DIFFSYNTH_SERVICE, "test_service")
        
        self.assertFalse(result)
        # Should still be the original service
        self.assertEqual(self.manager.services["test_service"].service_type, ServiceType.QWEN_GENERATOR)
    
    def test_unregister_service(self):
        """Test service unregistration"""
        cleanup_callback = Mock()
        
        self.manager.register_service(
            ServiceType.QWEN_GENERATOR,
            "test_service",
            cleanup_callback=cleanup_callback
        )
        
        result = self.manager.unregister_service("test_service")
        
        self.assertTrue(result)
        self.assertNotIn("test_service", self.manager.services)
        cleanup_callback.assert_called_once()
    
    def test_unregister_nonexistent_service(self):
        """Test unregistering nonexistent service"""
        result = self.manager.unregister_service("nonexistent_service")
        
        self.assertFalse(result)
    
    @patch('resource_manager.ResourceManager._get_current_gpu_usage')
    def test_request_memory_success(self, mock_gpu_usage):
        """Test successful memory allocation"""
        mock_gpu_usage.return_value = 2.0  # 2GB currently used
        
        self.manager.register_service(ServiceType.QWEN_GENERATOR, "test_service")
        
        result = self.manager.request_memory("test_service", 4.0)  # Request 4GB
        
        self.assertTrue(result)
        service = self.manager.services["test_service"]
        self.assertEqual(service.allocated_memory_gb, 4.0)
        self.assertTrue(service.is_active)
        self.assertGreater(service.last_used, 0)
    
    @patch('resource_manager.ResourceManager._get_current_gpu_usage')
    def test_request_memory_insufficient(self, mock_gpu_usage):
        """Test memory allocation with insufficient memory"""
        mock_gpu_usage.return_value = 6.0  # 6GB currently used, only 2GB available
        
        self.manager.register_service(ServiceType.QWEN_GENERATOR, "test_service")
        
        result = self.manager.request_memory("test_service", 4.0)  # Request 4GB
        
        self.assertFalse(result)
        service = self.manager.services["test_service"]
        self.assertEqual(service.allocated_memory_gb, 0.0)
        self.assertFalse(service.is_active)
    
    @patch('resource_manager.ResourceManager._get_current_gpu_usage')
    def test_request_memory_force_allocation(self, mock_gpu_usage):
        """Test forced memory allocation"""
        mock_gpu_usage.return_value = 6.0  # 6GB currently used
        
        # Register two services
        self.manager.register_service(ServiceType.QWEN_GENERATOR, "service1", ResourcePriority.LOW)
        self.manager.register_service(ServiceType.DIFFSYNTH_SERVICE, "service2", ResourcePriority.HIGH)
        
        # Allocate memory to low priority service
        self.manager.request_memory("service1", 2.0)
        
        # Force allocation for high priority service
        with patch.object(self.manager, '_free_memory_for_service', return_value=4.0):
            result = self.manager.request_memory("service2", 4.0, force=True)
        
        self.assertTrue(result)
        service2 = self.manager.services["service2"]
        self.assertEqual(service2.allocated_memory_gb, 4.0)
        self.assertTrue(service2.is_active)
    
    def test_request_memory_unregistered_service(self):
        """Test memory request for unregistered service"""
        result = self.manager.request_memory("nonexistent_service", 2.0)
        
        self.assertFalse(result)
    
    def test_release_memory(self):
        """Test memory release"""
        cleanup_callback = Mock()
        
        self.manager.register_service(
            ServiceType.QWEN_GENERATOR,
            "test_service",
            cleanup_callback=cleanup_callback
        )
        
        # Allocate memory first
        with patch.object(self.manager, '_get_current_gpu_usage', return_value=0.0):
            self.manager.request_memory("test_service", 2.0)
        
        # Release memory
        result = self.manager.release_memory("test_service")
        
        self.assertTrue(result)
        service = self.manager.services["test_service"]
        self.assertEqual(service.allocated_memory_gb, 0.0)
        self.assertFalse(service.is_active)
        cleanup_callback.assert_called_once()
    
    def test_free_memory_for_service(self):
        """Test freeing memory from other services"""
        # Register services with different priorities
        self.manager.register_service(ServiceType.QWEN_GENERATOR, "low_priority", ResourcePriority.LOW)
        self.manager.register_service(ServiceType.DIFFSYNTH_SERVICE, "normal_priority", ResourcePriority.NORMAL)
        self.manager.register_service(ServiceType.CONTROLNET_SERVICE, "high_priority", ResourcePriority.HIGH)
        
        # Allocate memory to services
        with patch.object(self.manager, '_get_current_gpu_usage', return_value=0.0):
            self.manager.request_memory("low_priority", 2.0)
            self.manager.request_memory("normal_priority", 1.5)
        
        # Free memory for high priority service
        freed_memory = self.manager._free_memory_for_service("high_priority", 3.0)
        
        self.assertGreaterEqual(freed_memory, 3.0)
        
        # Check that lower priority services were freed
        low_service = self.manager.services["low_priority"]
        normal_service = self.manager.services["normal_priority"]
        
        self.assertFalse(low_service.is_active)
        self.assertFalse(normal_service.is_active)
    
    @patch('resource_manager.torch.cuda.is_available')
    @patch('resource_manager.torch.cuda.memory_allocated')
    def test_get_current_gpu_usage(self, mock_memory_allocated, mock_cuda_available):
        """Test GPU usage detection"""
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 2 * 1e9  # 2GB
        
        usage = self.manager._get_current_gpu_usage()
        
        self.assertEqual(usage, 2.0)
    
    @patch('resource_manager.torch.cuda.is_available')
    def test_get_current_gpu_usage_no_cuda(self, mock_cuda_available):
        """Test GPU usage when CUDA not available"""
        mock_cuda_available.return_value = False
        
        usage = self.manager._get_current_gpu_usage()
        
        self.assertEqual(usage, 0.0)
    
    @patch('resource_manager.psutil.virtual_memory')
    @patch('resource_manager.torch.cuda.memory_allocated')
    @patch('resource_manager.torch.cuda.is_available')
    def test_get_memory_status(self, mock_cuda_available, mock_memory_allocated, mock_virtual_memory):
        """Test memory status reporting"""
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 2 * 1e9  # 2GB
        
        mock_memory = Mock()
        mock_memory.used = 8 * 1e9  # 8GB
        mock_virtual_memory.return_value = mock_memory
        
        # Register and allocate to a service
        self.manager.register_service(ServiceType.QWEN_GENERATOR, "test_service")
        with patch.object(self.manager, '_get_current_gpu_usage', return_value=2.0):
            self.manager.request_memory("test_service", 1.5)
        
        status = self.manager.get_memory_status()
        
        self.assertIn("gpu_memory", status)
        self.assertIn("cpu_memory", status)
        self.assertIn("services", status)
        self.assertIn("thresholds", status)
        
        # Check GPU memory info
        gpu_info = status["gpu_memory"]
        self.assertEqual(gpu_info["total_gb"], 8.0)
        self.assertEqual(gpu_info["used_gb"], 2.0)
        self.assertEqual(gpu_info["allocated_by_services_gb"], 1.5)
        
        # Check service info
        self.assertIn("test_service", status["services"])
        service_info = status["services"]["test_service"]
        self.assertEqual(service_info["type"], "qwen_generator")
        self.assertEqual(service_info["allocated_gb"], 1.5)
        self.assertTrue(service_info["is_active"])
    
    def test_monitor_resources_warning(self):
        """Test resource monitoring with warning threshold"""
        warning_callback = Mock()
        self.manager.add_memory_warning_callback(warning_callback)
        
        # Mock high memory usage
        with patch.object(self.manager, 'get_memory_status') as mock_status:
            mock_status.return_value = {
                "gpu_memory": {"usage_percent": 85.0},  # Above warning threshold
                "cpu_memory": {"usage_percent": 50.0},
                "services": {},
                "thresholds": {"warning_percent": 80.0, "critical_percent": 95.0}
            }
            
            self.manager.monitor_resources()
        
        warning_callback.assert_called_once_with(0.85)
    
    def test_monitor_resources_critical(self):
        """Test resource monitoring with critical threshold"""
        critical_callback = Mock()
        self.manager.add_memory_critical_callback(critical_callback)
        
        # Mock critical memory usage
        with patch.object(self.manager, 'get_memory_status') as mock_status:
            mock_status.return_value = {
                "gpu_memory": {"usage_percent": 97.0},  # Above critical threshold
                "cpu_memory": {"usage_percent": 50.0},
                "services": {},
                "thresholds": {"warning_percent": 80.0, "critical_percent": 95.0}
            }
            
            self.manager.monitor_resources()
        
        critical_callback.assert_called_once_with(0.97)
    
    def test_periodic_cleanup(self):
        """Test periodic cleanup of inactive services"""
        # Register service and allocate memory
        self.manager.register_service(ServiceType.QWEN_GENERATOR, "test_service")
        
        with patch.object(self.manager, '_get_current_gpu_usage', return_value=0.0):
            self.manager.request_memory("test_service", 2.0)
        
        # Set last_used to old time
        service = self.manager.services["test_service"]
        service.last_used = time.time() - 400  # 400 seconds ago
        
        # Run cleanup
        self.manager._periodic_cleanup()
        
        # Service should be released
        self.assertFalse(service.is_active)
        self.assertEqual(service.allocated_memory_gb, 0.0)
    
    def test_force_cleanup_all(self):
        """Test force cleanup of all services"""
        cleanup_callback1 = Mock()
        cleanup_callback2 = Mock()
        
        # Register multiple services
        self.manager.register_service(ServiceType.QWEN_GENERATOR, "service1", cleanup_callback=cleanup_callback1)
        self.manager.register_service(ServiceType.DIFFSYNTH_SERVICE, "service2", cleanup_callback=cleanup_callback2)
        
        # Allocate memory to services
        with patch.object(self.manager, '_get_current_gpu_usage', return_value=0.0):
            self.manager.request_memory("service1", 2.0)
            self.manager.request_memory("service2", 1.5)
        
        # Force cleanup
        self.manager.force_cleanup_all()
        
        # All services should be released
        for service in self.manager.services.values():
            self.assertFalse(service.is_active)
            self.assertEqual(service.allocated_memory_gb, 0.0)
        
        cleanup_callback1.assert_called_once()
        cleanup_callback2.assert_called_once()
    
    def test_get_service_recommendations(self):
        """Test service optimization recommendations"""
        # Register services with different usage patterns
        self.manager.register_service(ServiceType.QWEN_GENERATOR, "active_service")
        self.manager.register_service(ServiceType.DIFFSYNTH_SERVICE, "inactive_service")
        
        with patch.object(self.manager, '_get_current_gpu_usage', return_value=0.0):
            self.manager.request_memory("active_service", 2.0)
            self.manager.request_memory("inactive_service", 1.5)
        
        # Make one service inactive
        inactive_service = self.manager.services["inactive_service"]
        inactive_service.last_used = time.time() - 400  # Old timestamp
        
        # Mock high memory usage
        with patch.object(self.manager, 'get_memory_status') as mock_status:
            mock_status.return_value = {
                "gpu_memory": {"usage_percent": 85.0},
                "cpu_memory": {"usage_percent": 50.0},
                "services": {
                    "active_service": {"is_active": True},
                    "inactive_service": {"is_active": True}
                },
                "thresholds": {"warning_percent": 80.0, "critical_percent": 95.0}
            }
            
            recommendations = self.manager.get_service_recommendations()
        
        self.assertIn("recommendations", recommendations)
        self.assertIn("memory_status", recommendations)
        self.assertIn("optimization_suggestions", recommendations)
        
        # Should recommend releasing inactive services
        recommendations_text = " ".join(recommendations["recommendations"])
        self.assertIn("inactive_service", recommendations_text)


class TestGlobalResourceManager(unittest.TestCase):
    """Test global resource manager functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Clear global instance
        import resource_manager
        resource_manager._global_resource_manager = None
    
    def test_get_resource_manager_singleton(self):
        """Test global resource manager singleton"""
        manager1 = get_resource_manager()
        manager2 = get_resource_manager()
        
        self.assertIs(manager1, manager2)
        self.assertIsInstance(manager1, ResourceManager)
    
    def test_initialize_resource_manager(self):
        """Test initializing global resource manager with custom limits"""
        limits = ResourceLimits(max_gpu_memory_gb=16.0, max_cpu_memory_gb=32.0)
        
        manager = initialize_resource_manager(limits)
        
        self.assertIsInstance(manager, ResourceManager)
        self.assertEqual(manager.limits.max_gpu_memory_gb, 16.0)
        self.assertEqual(manager.limits.max_cpu_memory_gb, 32.0)
        
        # Should be the same as get_resource_manager()
        self.assertIs(manager, get_resource_manager())


class TestResourceManagerEnhancements(unittest.TestCase):
    """Test enhanced ResourceManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.limits = ResourceLimits(max_gpu_memory_gb=8.0, max_cpu_memory_gb=16.0)
        self.manager = ResourceManager(self.limits)
    
    def tearDown(self):
        """Clean up after tests"""
        with self.manager.allocation_lock:
            self.manager.services.clear()
    
    def test_optimize_memory_allocation(self):
        """Test memory allocation optimization"""
        # Register services with different usage patterns
        self.manager.register_service(ServiceType.QWEN_GENERATOR, "active_service", ResourcePriority.HIGH)
        self.manager.register_service(ServiceType.DIFFSYNTH_SERVICE, "inactive_service", ResourcePriority.LOW)
        
        # Allocate memory to services
        with patch.object(self.manager, '_get_current_gpu_usage', return_value=0.0):
            self.manager.request_memory("active_service", 2.0)
            self.manager.request_memory("inactive_service", 1.5)
        
        # Make one service inactive (old timestamp)
        inactive_service = self.manager.services["inactive_service"]
        inactive_service.last_used = time.time() - 700  # 700 seconds ago
        
        # Run optimization
        results = self.manager.optimize_memory_allocation()
        
        self.assertIn("actions_taken", results)
        self.assertIn("memory_freed_gb", results)
        self.assertIn("services_optimized", results)
        self.assertGreater(results["memory_freed_gb"], 0)
        self.assertIn("inactive_service", results["services_optimized"])
        
        # Inactive service should be released
        self.assertFalse(inactive_service.is_active)
    
    def test_get_workload_analysis(self):
        """Test workload analysis functionality"""
        # Register services with different patterns
        self.manager.register_service(ServiceType.QWEN_GENERATOR, "qwen_service", ResourcePriority.HIGH)
        self.manager.register_service(ServiceType.DIFFSYNTH_SERVICE, "diffsynth_service", ResourcePriority.NORMAL)
        self.manager.register_service(ServiceType.CONTROLNET_SERVICE, "controlnet_service", ResourcePriority.LOW)
        
        # Allocate memory to some services
        with patch.object(self.manager, '_get_current_gpu_usage', return_value=2.5):
            self.manager.request_memory("qwen_service", 2.0)
            self.manager.request_memory("diffsynth_service", 1.5)
        
        # Make one service idle
        idle_service = self.manager.services["diffsynth_service"]
        idle_service.last_used = time.time() - 400  # 400 seconds ago
        
        analysis = self.manager.get_workload_analysis()
        
        self.assertEqual(analysis["total_services"], 3)
        self.assertEqual(analysis["active_services"], 2)
        self.assertIn("service_breakdown", analysis)
        self.assertIn("memory_efficiency", analysis)
        self.assertIn("bottlenecks", analysis)
        
        # Check service breakdown
        self.assertIn("qwen_service", analysis["service_breakdown"])
        self.assertIn("diffsynth_service", analysis["service_breakdown"])
        self.assertIn("controlnet_service", analysis["service_breakdown"])
        
        # Should detect idle service as bottleneck
        bottlenecks_text = " ".join(analysis["bottlenecks"])
        self.assertIn("diffsynth_service", bottlenecks_text)
        
        # Memory efficiency should be calculated
        self.assertGreater(analysis["memory_efficiency"], 0)
    
    def test_workload_analysis_with_various_workloads(self):
        """Test workload analysis under various conditions"""
        # Test with no active services
        analysis = self.manager.get_workload_analysis()
        self.assertEqual(analysis["active_services"], 0)
        self.assertEqual(analysis["memory_efficiency"], 0.0)
        
        # Test with all services active
        for i in range(3):
            service_id = f"service_{i}"
            self.manager.register_service(ServiceType.QWEN_GENERATOR, service_id)
            with patch.object(self.manager, '_get_current_gpu_usage', return_value=1.0):
                self.manager.request_memory(service_id, 1.0)
        
        analysis = self.manager.get_workload_analysis()
        self.assertEqual(analysis["active_services"], 3)
        self.assertEqual(len(analysis["service_breakdown"]), 3)


class TestResourceManagerThreadSafety(unittest.TestCase):
    """Test resource manager thread safety"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.limits = ResourceLimits(max_gpu_memory_gb=8.0, max_cpu_memory_gb=16.0)
        self.manager = ResourceManager(self.limits)
        self.results = []
    
    def test_concurrent_service_registration(self):
        """Test concurrent service registration"""
        def register_service(service_id):
            result = self.manager.register_service(ServiceType.QWEN_GENERATOR, f"service_{service_id}")
            self.results.append(result)
        
        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_service, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All registrations should succeed
        self.assertEqual(len(self.results), 10)
        self.assertTrue(all(self.results))
        self.assertEqual(len(self.manager.services), 10)
    
    def test_concurrent_memory_requests(self):
        """Test concurrent memory allocation requests"""
        # Register services first
        for i in range(5):
            self.manager.register_service(ServiceType.QWEN_GENERATOR, f"service_{i}")
        
        def request_memory(service_id):
            with patch.object(self.manager, '_get_current_gpu_usage', return_value=0.0):
                result = self.manager.request_memory(f"service_{service_id}", 1.0)
                self.results.append(result)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=request_memory, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Some requests should succeed (depending on available memory)
        self.assertEqual(len(self.results), 5)
        successful_requests = sum(self.results)
        self.assertGreater(successful_requests, 0)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)