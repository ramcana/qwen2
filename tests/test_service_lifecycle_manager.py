"""
Unit tests for ServiceLifecycleManager
Tests service lifecycle management, health monitoring, and resource coordination
"""

import unittest
from unittest.mock import Mock, patch
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from service_lifecycle_manager import (
    ServiceLifecycleManager, ServiceState, ServiceInfo,
    get_service_lifecycle_manager, initialize_service_lifecycle_manager
)
from resource_manager import ServiceType, ResourcePriority


class TestServiceLifecycleManager(unittest.TestCase):
    """Test ServiceLifecycleManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = ServiceLifecycleManager()
    
    def tearDown(self):
        """Clean up after tests"""
        # Stop monitoring if running
        if self.manager.is_active:
            self.manager.stop_monitoring()
        
        # Clear services
        self.manager.services.clear()
    
    def test_service_registration(self):
        """Test service registration"""
        result = self.manager.register_service(
            "test_service", 
            ServiceType.DIFFSYNTH_SERVICE, 
            ResourcePriority.NORMAL
        )
        
        self.assertTrue(result)
        self.assertIn("test_service", self.manager.services)
        
        service_info = self.manager.services["test_service"]
        self.assertEqual(service_info.service_id, "test_service")
        self.assertEqual(service_info.service_type, ServiceType.DIFFSYNTH_SERVICE)
        self.assertEqual(service_info.priority, ResourcePriority.NORMAL)
        self.assertEqual(service_info.state, ServiceState.STOPPED)
    
    def test_duplicate_service_registration(self):
        """Test registering duplicate service"""
        self.manager.register_service("duplicate_service", ServiceType.QWEN_GENERATOR)
        
        # Try to register again
        result = self.manager.register_service("duplicate_service", ServiceType.DIFFSYNTH_SERVICE)
        
        self.assertFalse(result)
        # Should still be the original service
        self.assertEqual(self.manager.services["duplicate_service"].service_type, ServiceType.QWEN_GENERATOR)
    
    def test_service_unregistration(self):
        """Test service unregistration"""
        self.manager.register_service("test_service", ServiceType.DIFFSYNTH_SERVICE)
        
        result = self.manager.unregister_service("test_service")
        
        self.assertTrue(result)
        self.assertNotIn("test_service", self.manager.services)
    
    def test_unregister_nonexistent_service(self):
        """Test unregistering nonexistent service"""
        result = self.manager.unregister_service("nonexistent_service")
        
        self.assertFalse(result)
    
    def test_start_service_success(self):
        """Test successful service startup"""
        self.manager.register_service("start_test", ServiceType.QWEN_GENERATOR)
        
        # Mock resource manager to succeed
        with patch.object(self.manager.resource_manager, 'request_memory', return_value=True):
            result = self.manager.start_service("start_test", 4.0)
        
        self.assertTrue(result)
        service_info = self.manager.services["start_test"]
        self.assertEqual(service_info.state, ServiceState.RUNNING)
        self.assertEqual(service_info.memory_allocated_gb, 4.0)
        self.assertIsNotNone(service_info.start_time)
    
    def test_start_service_memory_failure(self):
        """Test service startup with memory allocation failure"""
        self.manager.register_service("memory_fail_test", ServiceType.DIFFSYNTH_SERVICE)
        
        # Mock resource manager to fail
        with patch.object(self.manager.resource_manager, 'request_memory', return_value=False):
            result = self.manager.start_service("memory_fail_test", 4.0)
        
        self.assertFalse(result)
        service_info = self.manager.services["memory_fail_test"]
        self.assertEqual(service_info.state, ServiceState.ERROR)
        self.assertEqual(service_info.error_count, 1)
    
    def test_start_nonexistent_service(self):
        """Test starting nonexistent service"""
        result = self.manager.start_service("nonexistent_service")
        
        self.assertFalse(result)
    
    def test_stop_service_success(self):
        """Test successful service shutdown"""
        self.manager.register_service("stop_test", ServiceType.QWEN_GENERATOR)
        
        # Start service first
        with patch.object(self.manager.resource_manager, 'request_memory', return_value=True):
            self.manager.start_service("stop_test", 4.0)
        
        # Mock resource manager release
        with patch.object(self.manager.resource_manager, 'release_memory', return_value=True):
            result = self.manager.stop_service("stop_test")
        
        self.assertTrue(result)
        service_info = self.manager.services["stop_test"]
        self.assertEqual(service_info.state, ServiceState.STOPPED)
        self.assertEqual(service_info.memory_allocated_gb, 0.0)
        self.assertIsNone(service_info.start_time)
    
    def test_restart_service(self):
        """Test service restart"""
        self.manager.register_service("restart_test", ServiceType.DIFFSYNTH_SERVICE)
        
        # Mock resource manager
        with patch.object(self.manager.resource_manager, 'request_memory', return_value=True), \
             patch.object(self.manager.resource_manager, 'release_memory', return_value=True):
            
            # Start service first
            self.manager.start_service("restart_test", 4.0)
            
            # Restart service
            result = self.manager.restart_service("restart_test")
        
        self.assertTrue(result)
        service_info = self.manager.services["restart_test"]
        self.assertEqual(service_info.state, ServiceState.RUNNING)
        self.assertEqual(service_info.restart_count, 1)
    
    def test_get_service_status_single(self):
        """Test getting status of single service"""
        self.manager.register_service("status_test", ServiceType.CONTROLNET_SERVICE)
        
        status = self.manager.get_service_status("status_test")
        
        self.assertIn("service_id", status)
        self.assertIn("service_type", status)
        self.assertIn("state", status)
        self.assertIn("memory_allocated_gb", status)
        self.assertEqual(status["service_id"], "status_test")
        self.assertEqual(status["service_type"], "controlnet_service")
    
    def test_get_service_status_all(self):
        """Test getting status of all services"""
        self.manager.register_service("service_1", ServiceType.QWEN_GENERATOR)
        self.manager.register_service("service_2", ServiceType.DIFFSYNTH_SERVICE)
        
        status = self.manager.get_service_status()
        
        self.assertIn("services", status)
        self.assertIn("total_services", status)
        self.assertIn("running_services", status)
        self.assertIn("management_active", status)
        self.assertEqual(status["total_services"], 2)
        self.assertIn("service_1", status["services"])
        self.assertIn("service_2", status["services"])
    
    def test_get_service_status_nonexistent(self):
        """Test getting status of nonexistent service"""
        status = self.manager.get_service_status("nonexistent_service")
        
        self.assertIn("error", status)
    
    def test_monitoring_lifecycle(self):
        """Test monitoring start and stop"""
        self.assertFalse(self.manager.is_active)
        
        self.manager.start_monitoring()
        self.assertTrue(self.manager.is_active)
        
        # Let monitoring run briefly
        time.sleep(0.5)
        
        self.manager.stop_monitoring()
        self.assertFalse(self.manager.is_active)
    
    def test_health_checks(self):
        """Test health check functionality"""
        self.manager.register_service("health_test", ServiceType.DIFFSYNTH_SERVICE)
        
        # Start service
        with patch.object(self.manager.resource_manager, 'request_memory', return_value=True):
            self.manager.start_service("health_test", 4.0)
        
        # Record initial activity time
        initial_activity = self.manager.services["health_test"].last_activity
        
        # Perform health check
        self.manager._perform_health_checks()
        
        # Activity time should be updated
        updated_activity = self.manager.services["health_test"].last_activity
        self.assertGreaterEqual(updated_activity, initial_activity)
    
    def test_callback_functionality(self):
        """Test service event callbacks"""
        started_callback = Mock()
        stopped_callback = Mock()
        
        self.manager.add_callback("started", started_callback)
        self.manager.add_callback("stopped", stopped_callback)
        
        self.manager.register_service("callback_test", ServiceType.QWEN_GENERATOR)
        
        # Mock resource manager
        with patch.object(self.manager.resource_manager, 'request_memory', return_value=True), \
             patch.object(self.manager.resource_manager, 'release_memory', return_value=True):
            
            # Start service (should trigger started callback)
            self.manager.start_service("callback_test", 4.0)
            started_callback.assert_called_once_with("callback_test")
            
            # Stop service (should trigger stopped callback)
            self.manager.stop_service("callback_test")
            stopped_callback.assert_called_once_with("callback_test")
    
    def test_shutdown_all_services(self):
        """Test shutting down all services"""
        # Register and start multiple services
        services = ["service_1", "service_2", "service_3"]
        
        for service_id in services:
            self.manager.register_service(service_id, ServiceType.DIFFSYNTH_SERVICE)
        
        # Start all services
        with patch.object(self.manager.resource_manager, 'request_memory', return_value=True):
            for service_id in services:
                self.manager.start_service(service_id, 2.0)
        
        # Verify all are running
        status = self.manager.get_service_status()
        self.assertEqual(status["running_services"], 3)
        
        # Shutdown all
        with patch.object(self.manager.resource_manager, 'release_memory', return_value=True):
            self.manager.shutdown_all_services()
        
        # Verify all are stopped
        status = self.manager.get_service_status()
        self.assertEqual(status["running_services"], 0)
        
        # Check individual service states
        for service_id in services:
            service_info = self.manager.services[service_id]
            self.assertEqual(service_info.state, ServiceState.STOPPED)
    
    def test_service_priority_handling(self):
        """Test handling of different service priorities"""
        # Register services with different priorities
        self.manager.register_service("critical_service", ServiceType.QWEN_GENERATOR, ResourcePriority.CRITICAL)
        self.manager.register_service("high_service", ServiceType.DIFFSYNTH_SERVICE, ResourcePriority.HIGH)
        self.manager.register_service("normal_service", ServiceType.CONTROLNET_SERVICE, ResourcePriority.NORMAL)
        self.manager.register_service("low_service", ServiceType.DIFFSYNTH_SERVICE, ResourcePriority.LOW)
        
        # Verify priorities are set correctly
        self.assertEqual(self.manager.services["critical_service"].priority, ResourcePriority.CRITICAL)
        self.assertEqual(self.manager.services["high_service"].priority, ResourcePriority.HIGH)
        self.assertEqual(self.manager.services["normal_service"].priority, ResourcePriority.NORMAL)
        self.assertEqual(self.manager.services["low_service"].priority, ResourcePriority.LOW)


class TestGlobalServiceLifecycleManager(unittest.TestCase):
    """Test global service lifecycle manager functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Clear global instance
        import service_lifecycle_manager
        service_lifecycle_manager._global_service_lifecycle_manager = None
    
    def test_get_service_lifecycle_manager_singleton(self):
        """Test global service lifecycle manager singleton"""
        manager1 = get_service_lifecycle_manager()
        manager2 = get_service_lifecycle_manager()
        
        self.assertIs(manager1, manager2)
        self.assertIsInstance(manager1, ServiceLifecycleManager)
    
    def test_initialize_service_lifecycle_manager(self):
        """Test initializing global service lifecycle manager"""
        manager = initialize_service_lifecycle_manager()
        
        self.assertIsInstance(manager, ServiceLifecycleManager)
        
        # Should be the same as get_service_lifecycle_manager()
        self.assertIs(manager, get_service_lifecycle_manager())


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)