"""
Integration tests for ResourceManager and ServiceLifecycleManager
Tests the complete resource management and service lifecycle coordination
"""

import unittest
from unittest.mock import Mock, patch
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from resource_manager import ResourceManager, ResourceLimits, ServiceType, ResourcePriority
from service_lifecycle_manager import ServiceLifecycleManager, ServiceState


class TestResourceServiceIntegration(unittest.TestCase):
    """Test integration between ResourceManager and ServiceLifecycleManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create resource manager with limited memory for testing
        self.limits = ResourceLimits(
            max_gpu_memory_gb=8.0,
            max_cpu_memory_gb=16.0,
            memory_warning_threshold=0.7,
            memory_critical_threshold=0.9
        )
        self.resource_manager = ResourceManager(self.limits)
        
        # Create service lifecycle manager with the same resource manager instance
        self.service_manager = ServiceLifecycleManager(self.resource_manager)
    
    def tearDown(self):
        """Clean up after tests"""
        # Stop monitoring if running
        if self.service_manager.is_active:
            self.service_manager.stop_monitoring()
        
        # Clear services
        self.service_manager.services.clear()
        
        # Clear resource manager services
        with self.resource_manager.allocation_lock:
            self.resource_manager.services.clear()
    
    def test_service_registration_with_resource_manager(self):
        """Test that service registration works with resource manager"""
        # Register service through service manager
        success = self.service_manager.register_service(
            "integrated_qwen", 
            ServiceType.QWEN_GENERATOR, 
            ResourcePriority.HIGH
        )
        
        self.assertTrue(success)
        
        # Verify service is registered in both managers
        self.assertIn("integrated_qwen", self.service_manager.services)
        self.assertIn("integrated_qwen", self.resource_manager.services)
        
        # Verify resource manager has correct service info
        rm_service = self.resource_manager.services["integrated_qwen"]
        self.assertEqual(rm_service.service_type, ServiceType.QWEN_GENERATOR)
        self.assertEqual(rm_service.priority, ResourcePriority.HIGH)
    
    def test_memory_allocation_coordination(self):
        """Test memory allocation coordination between managers"""
        # Register multiple services
        services = [
            ("qwen_main", ServiceType.QWEN_GENERATOR, ResourcePriority.HIGH, 4.0),
            ("diffsynth_edit", ServiceType.DIFFSYNTH_SERVICE, ResourcePriority.NORMAL, 3.0),
            ("controlnet_guide", ServiceType.CONTROLNET_SERVICE, ResourcePriority.NORMAL, 2.0)
        ]
        
        for service_id, service_type, priority, memory_gb in services:
            self.service_manager.register_service(service_id, service_type, priority)
        
        # Mock GPU usage to simulate available memory
        with patch.object(self.resource_manager, '_get_current_gpu_usage', return_value=1.0):
            # Start services - should succeed based on priority and available memory
            qwen_started = self.service_manager.start_service("qwen_main", 4.0)
            diffsynth_started = self.service_manager.start_service("diffsynth_edit", 3.0)
            
            # This should fail due to insufficient memory (1 + 4 + 3 + 2 = 10GB > 8GB total)
            controlnet_started = self.service_manager.start_service("controlnet_guide", 2.0)
        
        # Verify results
        self.assertTrue(qwen_started)
        self.assertTrue(diffsynth_started)
        self.assertFalse(controlnet_started)  # Should fail due to memory constraints
        
        # Check resource manager allocations
        rm_status = self.resource_manager.get_memory_status()
        self.assertEqual(rm_status["gpu_memory"]["allocated_by_services_gb"], 7.0)  # 4 + 3
    
    def test_priority_based_resource_allocation(self):
        """Test that higher priority services can evict lower priority ones"""
        # Register services with different priorities
        self.service_manager.register_service("low_priority", ServiceType.DIFFSYNTH_SERVICE, ResourcePriority.LOW)
        self.service_manager.register_service("high_priority", ServiceType.QWEN_GENERATOR, ResourcePriority.HIGH)
        
        # Mock GPU usage
        with patch.object(self.resource_manager, '_get_current_gpu_usage', return_value=2.0):
            # Start low priority service first (uses most available memory)
            low_started = self.service_manager.start_service("low_priority", 5.0)
            self.assertTrue(low_started)
            
            # High priority service should be able to force allocation
            # This will trigger resource manager to free memory from low priority service
            high_started = self.service_manager.start_service("high_priority", 4.0)
        
        # Check final states
        low_service = self.service_manager.services["low_priority"]
        high_service = self.service_manager.services["high_priority"]
        
        # Low priority service should have been stopped to make room
        self.assertEqual(low_service.state, ServiceState.STOPPED)
        self.assertEqual(high_service.state, ServiceState.RUNNING)
    
    def test_service_cleanup_coordination(self):
        """Test that service cleanup works through resource manager"""
        # Register and start a service
        self.service_manager.register_service("cleanup_test", ServiceType.DIFFSYNTH_SERVICE)
        
        with patch.object(self.resource_manager, '_get_current_gpu_usage', return_value=1.0):
            self.service_manager.start_service("cleanup_test", 3.0)
        
        # Verify service is running
        self.assertEqual(self.service_manager.services["cleanup_test"].state, ServiceState.RUNNING)
        
        # Trigger cleanup through resource manager
        self.resource_manager.force_cleanup_all()
        
        # Service should be stopped
        self.assertEqual(self.service_manager.services["cleanup_test"].state, ServiceState.STOPPED)
        self.assertEqual(self.service_manager.services["cleanup_test"].memory_allocated_gb, 0.0)
    
    def test_memory_optimization_with_services(self):
        """Test memory optimization with active services"""
        # Register multiple services
        services = ["service_a", "service_b", "service_c"]
        for service_id in services:
            self.service_manager.register_service(service_id, ServiceType.DIFFSYNTH_SERVICE)
        
        # Start all services
        with patch.object(self.resource_manager, '_get_current_gpu_usage', return_value=1.0):
            for service_id in services:
                self.service_manager.start_service(service_id, 2.0)
        
        # Make some services inactive (simulate old usage)
        current_time = time.time()
        self.service_manager.services["service_a"].last_activity = current_time - 700  # 11+ minutes ago
        
        # Also update resource manager's service tracking
        if "service_a" in self.resource_manager.services:
            self.resource_manager.services["service_a"].last_used = current_time - 700
        
        # Run optimization
        optimization_results = self.resource_manager.optimize_memory_allocation()
        
        # Should have freed memory from inactive service
        self.assertGreater(optimization_results["memory_freed_gb"], 0)
        self.assertIn("service_a", optimization_results["services_optimized"])
        
        # Verify service was actually stopped
        self.assertEqual(self.service_manager.services["service_a"].state, ServiceState.STOPPED)
    
    def test_workload_analysis_with_services(self):
        """Test workload analysis with running services"""
        # Register services with different patterns
        self.service_manager.register_service("active_qwen", ServiceType.QWEN_GENERATOR, ResourcePriority.HIGH)
        self.service_manager.register_service("idle_diffsynth", ServiceType.DIFFSYNTH_SERVICE, ResourcePriority.NORMAL)
        self.service_manager.register_service("stopped_controlnet", ServiceType.CONTROLNET_SERVICE, ResourcePriority.LOW)
        
        # Start some services
        with patch.object(self.resource_manager, '_get_current_gpu_usage', return_value=2.0):
            self.service_manager.start_service("active_qwen", 3.0)
            self.service_manager.start_service("idle_diffsynth", 2.0)
            # Leave stopped_controlnet stopped
        
        # Make one service idle
        current_time = time.time()
        self.service_manager.services["idle_diffsynth"].last_activity = current_time - 400
        
        # Get workload analysis from resource manager
        analysis = self.resource_manager.get_workload_analysis()
        
        self.assertEqual(analysis["total_services"], 3)
        self.assertEqual(analysis["active_services"], 2)  # Only running services
        
        # Should detect idle service as bottleneck
        self.assertGreater(len(analysis["bottlenecks"]), 0)
        bottleneck_text = " ".join(analysis["bottlenecks"])
        self.assertIn("idle_diffsynth", bottleneck_text)
    
    def test_service_monitoring_with_resource_management(self):
        """Test service monitoring integration with resource management"""
        # Register services
        self.service_manager.register_service("monitored_service", ServiceType.DIFFSYNTH_SERVICE)
        
        # Start service
        with patch.object(self.resource_manager, '_get_current_gpu_usage', return_value=1.0):
            self.service_manager.start_service("monitored_service", 3.0)
        
        # Start monitoring
        self.service_manager.start_monitoring()
        
        # Let monitoring run briefly
        time.sleep(0.5)
        
        # Check that service is being monitored
        service_info = self.service_manager.services["monitored_service"]
        self.assertEqual(service_info.state, ServiceState.RUNNING)
        
        # Stop monitoring
        self.service_manager.stop_monitoring()
        
        # Verify monitoring stopped
        self.assertFalse(self.service_manager.is_active)
    
    def test_concurrent_service_operations(self):
        """Test concurrent service operations with resource coordination"""
        import threading
        
        # Register multiple services
        service_count = 5
        for i in range(service_count):
            self.service_manager.register_service(
                f"concurrent_service_{i}", 
                ServiceType.DIFFSYNTH_SERVICE, 
                ResourcePriority.NORMAL
            )
        
        results = []
        
        def start_service(service_id):
            with patch.object(self.resource_manager, '_get_current_gpu_usage', return_value=1.0):
                result = self.service_manager.start_service(f"concurrent_service_{service_id}", 1.5)
                results.append(result)
        
        # Start services concurrently
        threads = []
        for i in range(service_count):
            thread = threading.Thread(target=start_service, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Some services should start successfully (limited by memory)
        successful_starts = sum(results)
        self.assertGreater(successful_starts, 0)
        self.assertLessEqual(successful_starts, service_count)
        
        # Verify resource manager consistency
        rm_status = self.resource_manager.get_memory_status()
        sm_status = self.service_manager.get_service_status()
        
        # Running services should match between managers
        rm_active_services = len([s for s in rm_status["services"].values() if s["is_active"]])
        sm_running_services = sm_status["running_services"]
        
        # They should be consistent (allowing for some timing differences)
        self.assertLessEqual(abs(rm_active_services - sm_running_services), 1)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)