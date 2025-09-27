"""
Unit tests for ServiceManager
Tests service lifecycle management, health monitoring, and automatic recovery
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from service_manager import (
    ServiceManager, ManagedService, ServiceConfig, ServiceStatus, 
    ServiceHealthStatus, ServiceMetrics, get_service_manager, initialize_service_manager
)
from resource_manager import ServiceType, ResourcePriority


class MockManagedService(ManagedService):
    """Mock managed service for testing"""
    
    def __init__(self, service_id: str = "test_service", fail_init: bool = False, fail_health: bool = False):
        config = ServiceConfig(
            service_id=service_id,
            service_type=ServiceType.DIFFSYNTH_SERVICE,
            priority=ResourcePriority.NORMAL,
            startup_timeout=5.0,
            shutdown_timeout=5.0
        )
        super().__init__(config)
        
        self.fail_init = fail_init
        self.fail_health = fail_health
        self.init_called = False
        self.shutdown_called = False
        self.health_check_called = False
        self.process_request_called = False
    
    async def initialize(self) -> bool:
        """Mock initialization"""
        self.init_called = True
        if self.fail_init:
            return False
        await asyncio.sleep(0.1)  # Simulate initialization time
        return True
    
    async def shutdown(self) -> bool:
        """Mock shutdown"""
        self.shutdown_called = True
        await asyncio.sleep(0.1)  # Simulate shutdown time
        return True
    
    async def health_check(self) -> ServiceHealthStatus:
        """Mock health check"""
        self.health_check_called = True
        if self.fail_health:
            return ServiceHealthStatus.UNHEALTHY
        return ServiceHealthStatus.HEALTHY
    
    async def process_request(self, request) -> dict:
        """Mock request processing"""
        self.process_request_called = True
        self.update_activity()
        await asyncio.sleep(0.1)  # Simulate processing time
        return {"success": True, "result": "processed"}


class TestServiceManager(unittest.TestCase):
    """Test ServiceManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = ServiceManager()
    
    def tearDown(self):
        """Clean up after tests"""
        # Stop management if running
        if self.manager.is_running:
            self.manager.stop_management()
        
        # Clear services
        self.manager.services.clear()
        self.manager.service_configs.clear()
    
    def test_service_registration(self):
        """Test service registration"""
        service = MockManagedService("test_service_1")
        
        result = self.manager.register_service(service)
        
        self.assertTrue(result)
        self.assertIn("test_service_1", self.manager.services)
        self.assertIn("test_service_1", self.manager.service_configs)
    
    def test_duplicate_service_registration(self):
        """Test registering duplicate service"""
        service1 = MockManagedService("duplicate_service")
        service2 = MockManagedService("duplicate_service")
        
        result1 = self.manager.register_service(service1)
        result2 = self.manager.register_service(service2)
        
        self.assertTrue(result1)
        self.assertFalse(result2)
        self.assertEqual(len(self.manager.services), 1)
    
    def test_service_unregistration(self):
        """Test service unregistration"""
        service = MockManagedService("test_service_unreg")
        
        self.manager.register_service(service)
        result = self.manager.unregister_service("test_service_unreg")
        
        self.assertTrue(result)
        self.assertNotIn("test_service_unreg", self.manager.services)
        self.assertNotIn("test_service_unreg", self.manager.service_configs)
    
    def test_unregister_nonexistent_service(self):
        """Test unregistering nonexistent service"""
        result = self.manager.unregister_service("nonexistent_service")
        
        self.assertFalse(result)
    
    async def test_start_service_success(self):
        """Test successful service startup"""
        service = MockManagedService("start_test_service")
        self.manager.register_service(service)
        
        # Mock resource manager
        with patch.object(self.manager.resource_manager, 'request_memory', return_value=True):
            result = await self.manager.start_service("start_test_service")
        
        self.assertTrue(result)
        self.assertTrue(service.init_called)
        self.assertEqual(service.status, ServiceStatus.READY)
    
    async def test_start_service_failure(self):
        """Test service startup failure"""
        service = MockManagedService("fail_start_service", fail_init=True)
        self.manager.register_service(service)
        
        # Mock resource manager
        with patch.object(self.manager.resource_manager, 'request_memory', return_value=True):
            result = await self.manager.start_service("fail_start_service")
        
        self.assertFalse(result)
        self.assertTrue(service.init_called)
        self.assertEqual(service.status, ServiceStatus.ERROR)
    
    async def test_start_service_memory_allocation_failure(self):
        """Test service startup with memory allocation failure"""
        service = MockManagedService("memory_fail_service")
        self.manager.register_service(service)
        
        # Mock resource manager to fail memory allocation
        with patch.object(self.manager.resource_manager, 'request_memory', return_value=False):
            result = await self.manager.start_service("memory_fail_service")
        
        self.assertFalse(result)
        self.assertFalse(service.init_called)  # Should not even try to initialize
        self.assertEqual(service.status, ServiceStatus.ERROR)
    
    async def test_stop_service_success(self):
        """Test successful service shutdown"""
        service = MockManagedService("stop_test_service")
        self.manager.register_service(service)
        
        # Start service first
        with patch.object(self.manager.resource_manager, 'request_memory', return_value=True):
            await self.manager.start_service("stop_test_service")
        
        # Mock resource manager release
        with patch.object(self.manager.resource_manager, 'release_memory', return_value=True):
            result = await self.manager.stop_service("stop_test_service")
        
        self.assertTrue(result)
        self.assertTrue(service.shutdown_called)
        self.assertEqual(service.status, ServiceStatus.OFFLINE)
    
    async def test_restart_service(self):
        """Test service restart"""
        service = MockManagedService("restart_test_service")
        self.manager.register_service(service)
        
        # Mock resource manager
        with patch.object(self.manager.resource_manager, 'request_memory', return_value=True), \
             patch.object(self.manager.resource_manager, 'release_memory', return_value=True):
            
            # Start service first
            await self.manager.start_service("restart_test_service")
            
            # Reset call flags
            service.init_called = False
            service.shutdown_called = False
            
            # Restart service
            result = await self.manager.restart_service("restart_test_service")
        
        self.assertTrue(result)
        self.assertTrue(service.shutdown_called)
        self.assertTrue(service.init_called)
        self.assertEqual(service.status, ServiceStatus.READY)
        self.assertEqual(service.metrics.restart_count, 1)
    
    def test_get_service_status_single(self):
        """Test getting status of single service"""
        service = MockManagedService("status_test_service")
        self.manager.register_service(service)
        
        status = self.manager.get_service_status("status_test_service")
        
        self.assertIn("service_id", status)
        self.assertIn("status", status)
        self.assertIn("health_status", status)
        self.assertIn("metrics", status)
        self.assertEqual(status["service_id"], "status_test_service")
    
    def test_get_service_status_all(self):
        """Test getting status of all services"""
        service1 = MockManagedService("service_1")
        service2 = MockManagedService("service_2")
        
        self.manager.register_service(service1)
        self.manager.register_service(service2)
        
        status = self.manager.get_service_status()
        
        self.assertIn("services", status)
        self.assertIn("total_services", status)
        self.assertIn("running_services", status)
        self.assertEqual(status["total_services"], 2)
        self.assertIn("service_1", status["services"])
        self.assertIn("service_2", status["services"])
    
    def test_get_service_status_nonexistent(self):
        """Test getting status of nonexistent service"""
        status = self.manager.get_service_status("nonexistent_service")
        
        self.assertIn("error", status)
    
    def test_callback_registration(self):
        """Test callback registration"""
        started_callback = Mock()
        stopped_callback = Mock()
        error_callback = Mock()
        
        self.manager.add_service_started_callback(started_callback)
        self.manager.add_service_stopped_callback(stopped_callback)
        self.manager.add_service_error_callback(error_callback)
        
        self.assertIn(started_callback, self.manager.service_started_callbacks)
        self.assertIn(stopped_callback, self.manager.service_stopped_callbacks)
        self.assertIn(error_callback, self.manager.service_error_callbacks)
    
    def test_management_lifecycle(self):
        """Test management start and stop"""
        self.assertFalse(self.manager.is_running)
        
        self.manager.start_management()
        self.assertTrue(self.manager.is_running)
        
        self.manager.stop_management()
        self.assertFalse(self.manager.is_running)
    
    async def test_health_check_integration(self):
        """Test health check integration"""
        service = MockManagedService("health_test_service")
        self.manager.register_service(service)
        
        # Start service
        with patch.object(self.manager.resource_manager, 'request_memory', return_value=True):
            await self.manager.start_service("health_test_service")
        
        # Perform health check
        await self.manager._perform_health_checks()
        
        self.assertTrue(service.health_check_called)
        self.assertEqual(service.health_status, ServiceHealthStatus.HEALTHY)
    
    async def test_unhealthy_service_restart(self):
        """Test automatic restart of unhealthy service"""
        service = MockManagedService("unhealthy_service", fail_health=True)
        self.manager.register_service(service)
        
        # Start service
        with patch.object(self.manager.resource_manager, 'request_memory', return_value=True), \
             patch.object(self.manager.resource_manager, 'release_memory', return_value=True):
            
            await self.manager.start_service("unhealthy_service")
            
            # Reset flags
            service.init_called = False
            service.shutdown_called = False
            
            # Perform health check (should trigger restart)
            await self.manager._perform_health_checks()
        
        # Service should have been restarted
        self.assertTrue(service.shutdown_called)
        self.assertTrue(service.init_called)
    
    async def test_idle_service_management(self):
        """Test idle service shutdown"""
        service = MockManagedService("idle_service")
        service.config.idle_shutdown_delay = 0.1  # Very short delay for testing
        service.config.auto_shutdown = True
        
        self.manager.register_service(service)
        
        # Start service
        with patch.object(self.manager.resource_manager, 'request_memory', return_value=True), \
             patch.object(self.manager.resource_manager, 'release_memory', return_value=True):
            
            await self.manager.start_service("idle_service")
            
            # Make service idle
            service.last_activity = time.time() - 1.0  # 1 second ago
            
            # Manage idle services
            await self.manager._manage_idle_services()
        
        # Service should be shut down
        self.assertTrue(service.shutdown_called)
        self.assertEqual(service.status, ServiceStatus.OFFLINE)
    
    async def test_shutdown_all_services(self):
        """Test shutting down all services"""
        service1 = MockManagedService("service_1")
        service2 = MockManagedService("service_2")
        
        self.manager.register_service(service1)
        self.manager.register_service(service2)
        
        # Start services
        with patch.object(self.manager.resource_manager, 'request_memory', return_value=True):
            await self.manager.start_service("service_1")
            await self.manager.start_service("service_2")
        
        # Shutdown all
        with patch.object(self.manager.resource_manager, 'release_memory', return_value=True):
            await self.manager.shutdown_all_services()
        
        # Both services should be shut down
        self.assertTrue(service1.shutdown_called)
        self.assertTrue(service2.shutdown_called)
        self.assertEqual(service1.status, ServiceStatus.OFFLINE)
        self.assertEqual(service2.status, ServiceStatus.OFFLINE)


class TestGlobalServiceManager(unittest.TestCase):
    """Test global service manager functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Clear global instance
        import service_manager
        service_manager._global_service_manager = None
    
    def test_get_service_manager_singleton(self):
        """Test global service manager singleton"""
        manager1 = get_service_manager()
        manager2 = get_service_manager()
        
        self.assertIs(manager1, manager2)
        self.assertIsInstance(manager1, ServiceManager)
    
    def test_initialize_service_manager(self):
        """Test initializing global service manager"""
        manager = initialize_service_manager()
        
        self.assertIsInstance(manager, ServiceManager)
        
        # Should be the same as get_service_manager()
        self.assertIs(manager, get_service_manager())


class TestAsyncServiceManager(unittest.TestCase):
    """Test ServiceManager async functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = ServiceManager()
    
    def tearDown(self):
        """Clean up after tests"""
        if self.manager.is_running:
            self.manager.stop_management()
        self.manager.services.clear()
        self.manager.service_configs.clear()
    
    def test_async_service_operations(self):
        """Test async service operations"""
        async def run_test():
            service = MockManagedService("async_test_service")
            self.manager.register_service(service)
            
            # Mock resource manager
            with patch.object(self.manager.resource_manager, 'request_memory', return_value=True), \
                 patch.object(self.manager.resource_manager, 'release_memory', return_value=True):
                
                # Start service
                start_result = await self.manager.start_service("async_test_service")
                self.assertTrue(start_result)
                
                # Process request
                request_result = await service.process_request({"test": "data"})
                self.assertTrue(request_result["success"])
                
                # Stop service
                stop_result = await self.manager.stop_service("async_test_service")
                self.assertTrue(stop_result)
        
        # Run async test
        asyncio.run(run_test())


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)