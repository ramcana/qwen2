"""
Integration tests for ResourceManager under various workloads
Tests memory management scenarios with multiple AI services
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


class TestResourceManagerWorkloads(unittest.TestCase):
    """Test ResourceManager under various workload scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.limits = ResourceLimits(
            max_gpu_memory_gb=12.0,
            max_cpu_memory_gb=32.0,
            memory_warning_threshold=0.7,
            memory_critical_threshold=0.9
        )
        self.manager = ResourceManager(self.limits)
    
    def tearDown(self):
        """Clean up after tests"""
        with self.manager.allocation_lock:
            self.manager.services.clear()
    
    def test_mixed_service_workload(self):
        """Test mixed workload with Qwen, DiffSynth, and ControlNet services"""
        # Register services with different priorities
        services = [
            ("qwen_main", ServiceType.QWEN_GENERATOR, ResourcePriority.HIGH),
            ("diffsynth_edit", ServiceType.DIFFSYNTH_SERVICE, ResourcePriority.NORMAL),
            ("controlnet_guide", ServiceType.CONTROLNET_SERVICE, ResourcePriority.NORMAL),
            ("qwen_backup", ServiceType.QWEN_GENERATOR, ResourcePriority.LOW)
        ]
        
        for service_id, service_type, priority in services:
            success = self.manager.register_service(service_type, service_id, priority)
            self.assertTrue(success, f"Failed to register {service_id}")
        
        # Simulate memory allocation requests
        with patch.object(self.manager, '_get_current_gpu_usage', return_value=2.0):
            # High priority service should get memory
            self.assertTrue(self.manager.request_memory("qwen_main", 4.0))
            
            # Normal priority services should get memory if available
            self.assertTrue(self.manager.request_memory("diffsynth_edit", 3.0))
            
            # This should fail due to insufficient memory (2 + 4 + 3 = 9GB > 8GB available)
            self.assertFalse(self.manager.request_memory("controlnet_guide", 2.0))
            
            # Force allocation should work by freeing lower priority services
            self.assertTrue(self.manager.request_memory("controlnet_guide", 2.0, force=True))
        
        # Check final allocation state
        status = self.manager.get_memory_status()
        active_services = [s for s in status["services"].values() if s["is_active"]]
        self.assertGreaterEqual(len(active_services), 2)
    
    def test_memory_pressure_scenarios(self):
        """Test behavior under memory pressure"""
        # Register multiple services
        for i in range(5):
            self.manager.register_service(
                ServiceType.DIFFSYNTH_SERVICE, 
                f"service_{i}", 
                ResourcePriority.NORMAL
            )
        
        # Simulate high memory usage
        with patch.object(self.manager, '_get_current_gpu_usage', return_value=8.0):
            # Try to allocate memory when system is near capacity
            results = []
            for i in range(5):
                result = self.manager.request_memory(f"service_{i}", 1.0)
                results.append(result)
            
            # Not all should succeed due to memory pressure
            successful_allocations = sum(results)
            self.assertLess(successful_allocations, 5)
            self.assertGreater(successful_allocations, 0)
    
    def test_automatic_cleanup_under_load(self):
        """Test automatic cleanup when system is under load"""
        # Register services
        services = ["service_a", "service_b", "service_c"]
        for service_id in services:
            self.manager.register_service(ServiceType.DIFFSYNTH_SERVICE, service_id)
        
        # Allocate memory to all services
        with patch.object(self.manager, '_get_current_gpu_usage', return_value=1.0):
            for service_id in services:
                self.manager.request_memory(service_id, 2.0)
        
        # Make some services inactive (simulate old usage)
        current_time = time.time()
        self.manager.services["service_a"].last_used = current_time - 700  # 11+ minutes ago
        self.manager.services["service_b"].last_used = current_time - 400  # 6+ minutes ago
        
        # Run optimization
        optimization_results = self.manager.optimize_memory_allocation()
        
        # Should have freed memory from inactive services
        self.assertGreater(optimization_results["memory_freed_gb"], 0)
        self.assertIn("service_a", optimization_results["services_optimized"])
        
        # Verify services were actually released
        self.assertFalse(self.manager.services["service_a"].is_active)
    
    def test_priority_based_allocation(self):
        """Test that higher priority services get preference"""
        # Register services with different priorities
        self.manager.register_service(ServiceType.QWEN_GENERATOR, "critical_service", ResourcePriority.CRITICAL)
        self.manager.register_service(ServiceType.DIFFSYNTH_SERVICE, "high_service", ResourcePriority.HIGH)
        self.manager.register_service(ServiceType.CONTROLNET_SERVICE, "normal_service", ResourcePriority.NORMAL)
        self.manager.register_service(ServiceType.DIFFSYNTH_SERVICE, "low_service", ResourcePriority.LOW)
        
        # Fill up memory with low priority service
        with patch.object(self.manager, '_get_current_gpu_usage', return_value=2.0):
            self.assertTrue(self.manager.request_memory("low_service", 6.0))  # Uses most available memory
        
        # High priority service should be able to force allocation
        with patch.object(self.manager, '_get_current_gpu_usage', return_value=8.0):
            self.assertTrue(self.manager.request_memory("critical_service", 4.0, force=True))
        
        # Low priority service should have been evicted
        self.assertFalse(self.manager.services["low_service"].is_active)
        self.assertTrue(self.manager.services["critical_service"].is_active)
    
    def test_workload_analysis_accuracy(self):
        """Test accuracy of workload analysis under different conditions"""
        # Test empty system
        analysis = self.manager.get_workload_analysis()
        self.assertEqual(analysis["total_services"], 0)
        self.assertEqual(analysis["active_services"], 0)
        self.assertEqual(analysis["memory_efficiency"], 0.0)
        
        # Add services with different usage patterns
        self.manager.register_service(ServiceType.QWEN_GENERATOR, "active_qwen")
        self.manager.register_service(ServiceType.DIFFSYNTH_SERVICE, "idle_diffsynth")
        self.manager.register_service(ServiceType.CONTROLNET_SERVICE, "unused_controlnet")
        
        # Allocate memory and set usage patterns
        with patch.object(self.manager, '_get_current_gpu_usage', return_value=3.0):
            self.manager.request_memory("active_qwen", 2.0)
            self.manager.request_memory("idle_diffsynth", 1.0)
        
        # Make one service idle
        current_time = time.time()
        self.manager.services["idle_diffsynth"].last_used = current_time - 400
        
        # Analyze workload
        analysis = self.manager.get_workload_analysis()
        
        self.assertEqual(analysis["total_services"], 3)
        self.assertEqual(analysis["active_services"], 2)
        self.assertGreater(analysis["memory_efficiency"], 0)
        
        # Should detect idle service
        self.assertGreater(len(analysis["bottlenecks"]), 0)
        bottleneck_text = " ".join(analysis["bottlenecks"])
        self.assertIn("idle_diffsynth", bottleneck_text)
    
    def test_resource_monitoring_callbacks(self):
        """Test resource monitoring with warning and critical callbacks"""
        warning_called = []
        critical_called = []
        
        def warning_callback(usage_percent):
            warning_called.append(usage_percent)
        
        def critical_callback(usage_percent):
            critical_called.append(usage_percent)
        
        self.manager.add_memory_warning_callback(warning_callback)
        self.manager.add_memory_critical_callback(critical_callback)
        
        # Simulate warning level usage
        with patch.object(self.manager, 'get_memory_status') as mock_status:
            mock_status.return_value = {
                "gpu_memory": {"usage_percent": 75.0},  # Above warning threshold (70%)
                "cpu_memory": {"usage_percent": 50.0},
                "services": {},
                "thresholds": {"warning_percent": 70.0, "critical_percent": 90.0}
            }
            
            self.manager.monitor_resources()
            
            self.assertEqual(len(warning_called), 1)
            self.assertEqual(warning_called[0], 0.75)
            self.assertEqual(len(critical_called), 0)
        
        # Simulate critical level usage
        with patch.object(self.manager, 'get_memory_status') as mock_status:
            mock_status.return_value = {
                "gpu_memory": {"usage_percent": 95.0},  # Above critical threshold (90%)
                "cpu_memory": {"usage_percent": 50.0},
                "services": {},
                "thresholds": {"warning_percent": 70.0, "critical_percent": 90.0}
            }
            
            self.manager.monitor_resources()
            
            self.assertEqual(len(critical_called), 1)
            self.assertEqual(critical_called[0], 0.95)
    
    def test_concurrent_operations_under_load(self):
        """Test concurrent operations under heavy load"""
        # Register multiple services
        service_count = 8
        for i in range(service_count):
            self.manager.register_service(
                ServiceType.DIFFSYNTH_SERVICE, 
                f"concurrent_service_{i}",
                ResourcePriority.NORMAL
            )
        
        results = []
        
        def allocate_memory(service_id):
            with patch.object(self.manager, '_get_current_gpu_usage', return_value=2.0):
                result = self.manager.request_memory(f"concurrent_service_{service_id}", 1.0)
                results.append(result)
        
        def release_memory(service_id):
            self.manager.release_memory(f"concurrent_service_{service_id}")
        
        # Start concurrent allocation threads
        allocation_threads = []
        for i in range(service_count):
            thread = threading.Thread(target=allocate_memory, args=(i,))
            allocation_threads.append(thread)
            thread.start()
        
        # Wait for allocations to complete
        for thread in allocation_threads:
            thread.join()
        
        # Some allocations should succeed
        successful_allocations = sum(results)
        self.assertGreater(successful_allocations, 0)
        self.assertLessEqual(successful_allocations, service_count)
        
        # Start concurrent release threads
        release_threads = []
        for i in range(service_count):
            thread = threading.Thread(target=release_memory, args=(i,))
            release_threads.append(thread)
            thread.start()
        
        # Wait for releases to complete
        for thread in release_threads:
            thread.join()
        
        # All services should be released
        active_services = [s for s in self.manager.services.values() if s.is_active]
        self.assertEqual(len(active_services), 0)
    
    def test_emergency_cleanup_scenario(self):
        """Test emergency cleanup when system runs out of memory"""
        # Fill system with services
        for i in range(4):
            service_id = f"memory_hog_{i}"
            self.manager.register_service(ServiceType.DIFFSYNTH_SERVICE, service_id)
            
            with patch.object(self.manager, '_get_current_gpu_usage', return_value=i * 2.0):
                self.manager.request_memory(service_id, 2.0)
        
        # Verify services are active
        active_count = sum(1 for s in self.manager.services.values() if s.is_active)
        self.assertGreater(active_count, 0)
        
        # Force emergency cleanup
        self.manager.force_cleanup_all()
        
        # All services should be released
        active_count = sum(1 for s in self.manager.services.values() if s.is_active)
        self.assertEqual(active_count, 0)
        
        # All allocated memory should be zero
        total_allocated = sum(s.allocated_memory_gb for s in self.manager.services.values())
        self.assertEqual(total_allocated, 0.0)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)