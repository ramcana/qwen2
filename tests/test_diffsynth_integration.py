"""
Integration tests for DiffSynth Service Foundation
Tests the complete service setup and resource management integration
"""

import unittest
import time
from unittest.mock import Mock, patch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from diffsynth_service import DiffSynthService, DiffSynthConfig, create_diffsynth_service
from resource_manager import ResourceManager, ServiceType, ResourcePriority, get_resource_manager


class TestDiffSynthServiceIntegration(unittest.TestCase):
    """Integration tests for DiffSynth service with resource management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.resource_manager = ResourceManager()
        self.config = DiffSynthConfig(device="cpu")
        self.service = DiffSynthService(self.config)
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self.service, 'shutdown'):
            self.service.shutdown()
        
        # Clear resource manager
        with self.resource_manager.allocation_lock:
            self.resource_manager.services.clear()
    
    def test_service_resource_manager_integration(self):
        """Test integration between service and resource manager"""
        # Register service with resource manager
        success = self.resource_manager.register_service(
            ServiceType.DIFFSYNTH_SERVICE,
            "test_diffsynth_service",
            ResourcePriority.NORMAL,
            cleanup_callback=self.service._cleanup_resources
        )
        
        self.assertTrue(success)
        self.assertIn("test_diffsynth_service", self.resource_manager.services)
        
        # Request memory allocation
        with patch.object(self.resource_manager, '_get_current_gpu_usage', return_value=0.0):
            memory_allocated = self.resource_manager.request_memory("test_diffsynth_service", 2.0)
        
        self.assertTrue(memory_allocated)
        
        service_resource = self.resource_manager.services["test_diffsynth_service"]
        self.assertEqual(service_resource.allocated_memory_gb, 2.0)
        self.assertTrue(service_resource.is_active)
    
    def test_service_lifecycle_with_resource_management(self):
        """Test complete service lifecycle with resource management"""
        # Create service using factory function
        service = create_diffsynth_service(device="cpu")
        
        # Register with resource manager
        self.resource_manager.register_service(
            ServiceType.DIFFSYNTH_SERVICE,
            "lifecycle_test_service",
            ResourcePriority.HIGH
        )
        
        # Check initial status
        status = service.get_status()
        self.assertEqual(status["status"], "not_initialized")
        self.assertFalse(status["initialized"])
        
        # Verify resource usage tracking
        self.assertIn("resource_usage", status)
        self.assertIn("config", status)
        
        # Test shutdown
        service.shutdown()
        self.assertEqual(service.status.value, "offline")
    
    def test_memory_optimization_patterns(self):
        """Test that memory optimization patterns are applied correctly"""
        service = DiffSynthService(self.config)
        
        # Test memory optimization setup
        service._setup_memory_optimizations()
        
        # Verify service configuration includes optimization settings
        status = service.get_status()
        config_info = status["config"]
        
        self.assertIn("enable_vram_management", config_info)
        self.assertIn("enable_cpu_offload", config_info)
        self.assertIn("use_tiled_processing", config_info)
        
        # Test resource usage tracking
        service._update_resource_usage()
        
        resource_usage = status["resource_usage"]
        self.assertIn("cpu_memory_used_gb", resource_usage)
        self.assertGreaterEqual(resource_usage["cpu_memory_used_gb"], 0)
    
    def test_error_handling_integration(self):
        """Test error handling integration with resource management"""
        service = DiffSynthService(self.config)
        
        # Test system requirements check
        requirements_ok = service._check_system_requirements()
        self.assertTrue(requirements_ok)  # Should pass on CPU
        
        # Test error handling for invalid image
        result_image, message = service.edit_image(
            prompt="test prompt",
            image="/nonexistent/path.png"
        )
        
        self.assertIsNone(result_image)
        # Service is not initialized, so it returns "not available" message
        self.assertIn("not available", message)
        
        # Verify error count is tracked
        status = service.get_status()
        self.assertEqual(status["error_count"], 0)  # No errors in service itself
    
    def test_resource_sharing_between_services(self):
        """Test resource sharing between multiple services"""
        # Create multiple services
        service1 = create_diffsynth_service(device="cpu")
        service2 = create_diffsynth_service(device="cpu")
        
        # Register both services
        self.resource_manager.register_service(
            ServiceType.DIFFSYNTH_SERVICE,
            "service1",
            ResourcePriority.LOW
        )
        
        self.resource_manager.register_service(
            ServiceType.QWEN_GENERATOR,
            "service2", 
            ResourcePriority.HIGH
        )
        
        # Allocate memory to low priority service
        with patch.object(self.resource_manager, '_get_current_gpu_usage', return_value=0.0):
            success1 = self.resource_manager.request_memory("service1", 4.0)
            self.assertTrue(success1)
        
        # Try to allocate memory to high priority service (should succeed without forcing since we have enough memory)
        with patch.object(self.resource_manager, '_get_current_gpu_usage', return_value=4.0):
            success2 = self.resource_manager.request_memory("service2", 3.0)
            self.assertTrue(success2)
        
        # Verify resource allocation - both should be active since we have enough total memory
        service1_resource = self.resource_manager.services["service1"]
        service2_resource = self.resource_manager.services["service2"]
        
        self.assertTrue(service1_resource.is_active)   # Should still be active
        self.assertTrue(service2_resource.is_active)   # Should be allocated
    
    def test_configuration_validation(self):
        """Test configuration validation and defaults"""
        # Test default configuration
        default_config = DiffSynthConfig()
        
        self.assertEqual(default_config.model_name, "Qwen/Qwen-Image-Edit")
        self.assertTrue(default_config.enable_vram_management)
        self.assertTrue(default_config.enable_cpu_offload)
        self.assertEqual(default_config.max_memory_usage_gb, 4.0)
        
        # Test custom configuration
        custom_config = DiffSynthConfig(
            model_name="custom/model",
            enable_vram_management=False,
            max_memory_usage_gb=8.0
        )
        
        self.assertEqual(custom_config.model_name, "custom/model")
        self.assertFalse(custom_config.enable_vram_management)
        self.assertEqual(custom_config.max_memory_usage_gb, 8.0)
    
    def test_service_status_reporting(self):
        """Test comprehensive service status reporting"""
        service = DiffSynthService(self.config)
        
        # Get initial status
        status = service.get_status()
        
        # Verify all required fields are present
        required_fields = [
            "status", "initialized", "model_name", "device",
            "operation_count", "error_count", "resource_usage", "config"
        ]
        
        for field in required_fields:
            self.assertIn(field, status)
        
        # Verify resource usage fields
        resource_usage = status["resource_usage"]
        resource_fields = [
            "gpu_memory_allocated_gb", "gpu_memory_reserved_gb",
            "cpu_memory_used_gb", "last_updated"
        ]
        
        for field in resource_fields:
            self.assertIn(field, resource_usage)
        
        # Verify config fields
        config_info = status["config"]
        config_fields = [
            "enable_vram_management", "enable_cpu_offload",
            "use_tiled_processing", "max_memory_usage_gb"
        ]
        
        for field in config_fields:
            self.assertIn(field, config_info)
    
    def test_global_resource_manager_integration(self):
        """Test integration with global resource manager"""
        # Get global resource manager
        global_manager = get_resource_manager()
        
        self.assertIsInstance(global_manager, ResourceManager)
        
        # Register service with global manager
        success = global_manager.register_service(
            ServiceType.DIFFSYNTH_SERVICE,
            "global_test_service"
        )
        
        self.assertTrue(success)
        
        # Verify service is registered
        self.assertIn("global_test_service", global_manager.services)
        
        # Clean up
        global_manager.unregister_service("global_test_service")


class TestDiffSynthServicePerformance(unittest.TestCase):
    """Performance tests for DiffSynth service"""
    
    def test_service_creation_performance(self):
        """Test service creation performance"""
        start_time = time.time()
        
        service = create_diffsynth_service(device="cpu")
        
        creation_time = time.time() - start_time
        
        # Service creation should be fast (< 1 second)
        self.assertLess(creation_time, 1.0)
        
        # Verify service is properly initialized
        self.assertIsNotNone(service)
        self.assertEqual(service.config.device, "cpu")
    
    def test_resource_manager_performance(self):
        """Test resource manager performance with multiple services"""
        manager = ResourceManager()
        
        start_time = time.time()
        
        # Register multiple services
        for i in range(10):
            manager.register_service(
                ServiceType.DIFFSYNTH_SERVICE,
                f"perf_test_service_{i}"
            )
        
        registration_time = time.time() - start_time
        
        # Registration should be fast
        self.assertLess(registration_time, 0.1)
        
        # Verify all services are registered
        self.assertEqual(len(manager.services), 10)
        
        # Test memory allocation performance
        start_time = time.time()
        
        with patch.object(manager, '_get_current_gpu_usage', return_value=0.0):
            for i in range(5):
                manager.request_memory(f"perf_test_service_{i}", 1.0)
        
        allocation_time = time.time() - start_time
        
        # Memory allocation should be fast
        self.assertLess(allocation_time, 0.1)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)