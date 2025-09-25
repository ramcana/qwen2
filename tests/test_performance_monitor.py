"""
Unit tests for PerformanceMonitor class
Tests timing accuracy, metrics collection, and MMDiT-specific performance monitoring
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import torch
import sys
import os
import tempfile
import json
from contextlib import contextmanager

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.performance_monitor import (
    PerformanceMonitor, 
    PerformanceMetrics,
    create_mmdit_performance_monitor,
    monitor_generation_performance
)


class TestPerformanceMetrics(unittest.TestCase):
    """Test PerformanceMetrics dataclass"""
    
    def test_default_metrics(self):
        """Test default metric values"""
        metrics = PerformanceMetrics()
        
        # Test timing defaults
        self.assertEqual(metrics.model_load_time, 0.0)
        self.assertEqual(metrics.generation_time_per_step, 0.0)
        self.assertEqual(metrics.total_generation_time, 0.0)
        
        # Test memory defaults
        self.assertEqual(metrics.gpu_memory_used_gb, 0.0)
        self.assertEqual(metrics.gpu_memory_utilization_percent, 0.0)
        
        # Test generation defaults
        self.assertEqual(metrics.num_inference_steps, 0)
        self.assertEqual(metrics.image_resolution, (0, 0))
        self.assertEqual(metrics.batch_size, 1)
        
        # Test validation defaults
        self.assertFalse(metrics.target_met)
        self.assertEqual(metrics.target_generation_time, 5.0)
        self.assertEqual(metrics.performance_score, 0.0)
        
        # Test architecture defaults
        self.assertEqual(metrics.architecture_type, "MMDiT")
        self.assertEqual(metrics.torch_dtype, "float16")
        self.assertEqual(metrics.device, "cuda")
    
    def test_custom_metrics(self):
        """Test custom metric values"""
        metrics = PerformanceMetrics(
            total_generation_time=3.5,
            target_generation_time=5.0,
            architecture_type="UNet",
            image_resolution=(512, 512),
            num_inference_steps=25
        )
        
        self.assertEqual(metrics.total_generation_time, 3.5)
        self.assertEqual(metrics.target_generation_time, 5.0)
        self.assertEqual(metrics.architecture_type, "UNet")
        self.assertEqual(metrics.image_resolution, (512, 512))
        self.assertEqual(metrics.num_inference_steps, 25)


class TestPerformanceMonitor(unittest.TestCase):
    """Test PerformanceMonitor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = PerformanceMonitor(target_generation_time=5.0)
    
    def test_initialization(self):
        """Test monitor initialization"""
        self.assertEqual(self.monitor.target_generation_time, 5.0)
        self.assertIsInstance(self.monitor.current_metrics, PerformanceMetrics)
        self.assertEqual(self.monitor.current_metrics.target_generation_time, 5.0)
        self.assertIsNone(self.monitor._start_time)
        self.assertEqual(len(self.monitor._step_times), 0)
    
    def test_start_end_timing(self):
        """Test basic timing functionality"""
        # Test start timing
        self.monitor.start_timing()
        self.assertIsNotNone(self.monitor._start_time)
        
        # Simulate some work
        time.sleep(0.1)
        
        # Test end timing
        total_time = self.monitor.end_timing()
        self.assertGreater(total_time, 0.05)  # Should be at least 50ms
        self.assertLess(total_time, 0.5)      # Should be less than 500ms
        self.assertEqual(self.monitor.current_metrics.total_generation_time, total_time)
    
    def test_step_timing(self):
        """Test step-by-step timing"""
        # Start overall timing
        self.monitor.start_timing()
        
        # Simulate multiple steps
        step_times = []
        for i in range(3):
            self.monitor.start_step_timing()
            time.sleep(0.02)  # 20ms per step
            step_time = self.monitor.end_step_timing()
            step_times.append(step_time)
            self.assertGreater(step_time, 0.01)  # At least 10ms
            self.assertLess(step_time, 0.1)      # Less than 100ms
        
        # End overall timing
        total_time = self.monitor.end_timing()
        
        # Check step times were recorded
        self.assertEqual(len(self.monitor._step_times), 3)
        self.assertGreater(self.monitor.current_metrics.generation_time_per_step, 0)
        self.assertAlmostEqual(
            self.monitor.current_metrics.generation_time_per_step,
            sum(step_times) / len(step_times),
            places=3
        )
    
    def test_model_load_timing(self):
        """Test model loading time measurement"""
        def mock_load_function(model_path):
            time.sleep(0.05)  # Simulate 50ms load time
            return f"loaded_{model_path}"
        
        result, load_time = self.monitor.measure_model_load_time(
            mock_load_function, "test_model"
        )
        
        self.assertEqual(result, "loaded_test_model")
        self.assertGreater(load_time, 0.04)  # At least 40ms
        self.assertLess(load_time, 0.1)      # Less than 100ms
        self.assertEqual(self.monitor.current_metrics.model_load_time, load_time)
    
    def test_model_load_timing_with_exception(self):
        """Test model loading time measurement with exception"""
        def failing_load_function():
            time.sleep(0.02)
            raise ValueError("Load failed")
        
        with self.assertRaises(ValueError):
            self.monitor.measure_model_load_time(failing_load_function)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    @patch('psutil.virtual_memory')
    def test_memory_metrics_capture(self, mock_virtual_memory, mock_memory_reserved, 
                                   mock_memory_allocated, mock_cuda_available):
        """Test memory metrics capture"""
        # Mock CUDA availability and memory
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 2 * 1e9  # 2GB
        mock_memory_reserved.return_value = 3 * 1e9   # 3GB
        
        # Mock system memory
        mock_memory_info = Mock()
        mock_memory_info.used = 8 * 1e9  # 8GB
        mock_memory_info.percent = 75.0
        mock_virtual_memory.return_value = mock_memory_info
        
        # Set total GPU memory for calculation
        self.monitor.current_metrics.gpu_memory_total_gb = 16.0
        
        # Capture metrics
        self.monitor._capture_memory_metrics()
        
        # Verify GPU memory metrics
        self.assertEqual(self.monitor.current_metrics.gpu_memory_used_gb, 2.0)
        self.assertEqual(self.monitor.current_metrics.gpu_memory_utilization_percent, 12.5)  # 2/16 * 100
        
        # Verify system memory metrics
        self.assertEqual(self.monitor.current_metrics.system_memory_used_gb, 8.0)
        self.assertEqual(self.monitor.current_metrics.system_memory_utilization_percent, 75.0)
    
    def test_performance_validation(self):
        """Test performance target validation"""
        # Test target met
        self.assertTrue(self.monitor.validate_performance_target(3.0))  # 3s < 5s target
        
        # Test target missed
        self.assertFalse(self.monitor.validate_performance_target(7.0))  # 7s > 5s target
        
        # Test exact target
        self.assertTrue(self.monitor.validate_performance_target(5.0))  # 5s = 5s target
    
    def test_performance_score_calculation(self):
        """Test performance score calculation"""
        # Set up metrics for score calculation
        self.monitor.current_metrics.total_generation_time = 2.5  # Half the target
        self.monitor.current_metrics.target_generation_time = 5.0
        
        # Finalize metrics to trigger score calculation
        self.monitor._finalize_generation_metrics()
        
        # Score should be 200 (capped at 100)
        self.assertEqual(self.monitor.current_metrics.performance_score, 100.0)
        self.assertTrue(self.monitor.current_metrics.target_met)
        
        # Test slower performance
        self.monitor.current_metrics.total_generation_time = 10.0  # Double the target
        self.monitor._finalize_generation_metrics()
        
        # Score should be 50 (5/10 * 100)
        self.assertEqual(self.monitor.current_metrics.performance_score, 50.0)
        self.assertFalse(self.monitor.current_metrics.target_met)
    
    def test_monitor_generation_context_manager(self):
        """Test generation monitoring context manager"""
        with self.monitor.monitor_generation(
            model_name="Qwen-Image",
            architecture_type="MMDiT",
            num_steps=20,
            resolution=(1024, 1024)
        ) as monitor:
            # Verify monitor is the same instance
            self.assertIs(monitor, self.monitor)
            
            # Verify metrics were initialized
            self.assertEqual(self.monitor.current_metrics.model_name, "Qwen-Image")
            self.assertEqual(self.monitor.current_metrics.architecture_type, "MMDiT")
            self.assertEqual(self.monitor.current_metrics.num_inference_steps, 20)
            self.assertEqual(self.monitor.current_metrics.image_resolution, (1024, 1024))
            
            # Simulate some generation work
            time.sleep(0.05)
        
        # Verify timing was captured
        self.assertGreater(self.monitor.current_metrics.total_generation_time, 0.04)
        
        # Verify metrics were finalized
        self.assertIsNotNone(self.monitor.current_metrics.performance_score)
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        # Set up some metrics
        self.monitor.current_metrics.total_generation_time = 3.5
        self.monitor.current_metrics.generation_time_per_step = 0.175
        self.monitor.current_metrics.target_met = True
        self.monitor.current_metrics.performance_score = 85.7
        self.monitor.current_metrics.model_name = "Qwen-Image"
        self.monitor.current_metrics.architecture_type = "MMDiT"
        self.monitor.current_metrics.num_inference_steps = 20
        self.monitor.current_metrics.image_resolution = (1024, 1024)
        
        summary = self.monitor.get_performance_summary()
        
        # Verify current generation data
        self.assertEqual(summary["current_generation"]["total_time"], 3.5)
        self.assertEqual(summary["current_generation"]["per_step_time"], 0.175)
        self.assertTrue(summary["current_generation"]["target_met"])
        self.assertEqual(summary["current_generation"]["performance_score"], 85.7)
        self.assertEqual(summary["current_generation"]["model_name"], "Qwen-Image")
        self.assertEqual(summary["current_generation"]["architecture"], "MMDiT")
        self.assertEqual(summary["current_generation"]["steps"], 20)
        self.assertEqual(summary["current_generation"]["resolution"], "1024x1024")
        
        # Verify target configuration
        self.assertEqual(summary["target_configuration"]["target_time"], 5.0)
    
    def test_before_after_comparison(self):
        """Test before/after performance comparison"""
        # Create "before" metrics
        before_metrics = PerformanceMetrics(
            total_generation_time=10.0,
            generation_time_per_step=0.5,
            target_met=False,
            performance_score=50.0,
            model_name="Qwen-Image-Edit"
        )
        
        # Set up "after" metrics
        self.monitor.current_metrics.total_generation_time = 2.5
        self.monitor.current_metrics.generation_time_per_step = 0.125
        self.monitor.current_metrics.target_met = True
        self.monitor.current_metrics.performance_score = 100.0
        self.monitor.current_metrics.model_name = "Qwen-Image"
        
        comparison = self.monitor.get_before_after_comparison(before_metrics)
        
        # Verify before/after data
        self.assertEqual(comparison["before"]["total_time"], 10.0)
        self.assertEqual(comparison["after"]["total_time"], 2.5)
        
        # Verify improvements
        self.assertEqual(comparison["improvement"]["total_time_percent"], 75.0)  # (10-2.5)/10 * 100
        self.assertEqual(comparison["improvement"]["per_step_time_percent"], 75.0)  # (0.5-0.125)/0.5 * 100
        self.assertEqual(comparison["improvement"]["performance_score_change"], 50.0)  # 100-50
        self.assertTrue(comparison["improvement"]["target_status_change"])
        
        # Verify summary
        self.assertTrue(comparison["summary"]["faster"])
        self.assertEqual(comparison["summary"]["improvement_factor"], 4.0)  # 10/2.5
        self.assertTrue(comparison["summary"]["meets_target_now"])
        self.assertTrue(comparison["summary"]["significant_improvement"])
    
    def test_export_metrics_to_json(self):
        """Test JSON export functionality"""
        # Set up some metrics
        self.monitor.current_metrics.total_generation_time = 3.5
        self.monitor.current_metrics.model_name = "Qwen-Image"
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.monitor.export_metrics_to_json(temp_path)
            
            # Verify file was created and contains expected data
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            self.assertIn("current_metrics", data)
            self.assertIn("performance_summary", data)
            self.assertIn("export_timestamp", data)
            
            # Verify specific metrics
            self.assertEqual(data["current_metrics"]["total_generation_time"], 3.5)
            self.assertEqual(data["current_metrics"]["model_name"], "Qwen-Image")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_reset_metrics(self):
        """Test metrics reset functionality"""
        # Set up some state
        self.monitor.start_timing()
        self.monitor._step_times = [0.1, 0.2, 0.3]
        self.monitor.current_metrics.total_generation_time = 5.0
        
        # Reset
        self.monitor.reset_metrics()
        
        # Verify reset state
        self.assertIsNone(self.monitor._start_time)
        self.assertEqual(len(self.monitor._step_times), 0)
        self.assertEqual(self.monitor.current_metrics.total_generation_time, 0.0)
        self.assertEqual(self.monitor.current_metrics.target_generation_time, 5.0)  # Should preserve target
    
    def test_clear_history(self):
        """Test history clearing"""
        # Add some history (simulate by directly adding to collections)
        self.monitor._generation_history.append(PerformanceMetrics())
        self.monitor._step_time_history.append(0.1)
        
        # Verify history exists
        self.assertEqual(len(self.monitor._generation_history), 1)
        self.assertEqual(len(self.monitor._step_time_history), 1)
        
        # Clear history
        self.monitor.clear_history()
        
        # Verify history is cleared
        self.assertEqual(len(self.monitor._generation_history), 0)
        self.assertEqual(len(self.monitor._step_time_history), 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_create_mmdit_performance_monitor(self):
        """Test MMDiT performance monitor creation"""
        monitor = create_mmdit_performance_monitor(target_time=3.0)
        
        self.assertIsInstance(monitor, PerformanceMonitor)
        self.assertEqual(monitor.target_generation_time, 3.0)
    
    def test_monitor_generation_performance_context_manager(self):
        """Test convenience context manager"""
        with monitor_generation_performance(
            model_name="Test-Model",
            target_time=4.0,
            architecture_type="UNet",
            num_steps=15,
            resolution=(512, 512)
        ) as monitor:
            self.assertIsInstance(monitor, PerformanceMonitor)
            self.assertEqual(monitor.target_generation_time, 4.0)
            self.assertEqual(monitor.current_metrics.model_name, "Test-Model")
            self.assertEqual(monitor.current_metrics.architecture_type, "UNet")
            self.assertEqual(monitor.current_metrics.num_inference_steps, 15)
            self.assertEqual(monitor.current_metrics.image_resolution, (512, 512))
            
            # Simulate work
            time.sleep(0.02)
        
        # Verify timing was captured
        self.assertGreater(monitor.current_metrics.total_generation_time, 0.01)


class TestMMDiTSpecificFeatures(unittest.TestCase):
    """Test MMDiT architecture-specific features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = PerformanceMonitor(target_generation_time=5.0)
    
    def test_mmdit_architecture_detection(self):
        """Test MMDiT architecture-specific handling"""
        with self.monitor.monitor_generation(
            model_name="Qwen-Image",
            architecture_type="MMDiT",
            num_steps=20,
            resolution=(1024, 1024)
        ):
            # Simulate MMDiT generation
            time.sleep(0.03)
        
        # Verify MMDiT-specific metrics
        self.assertEqual(self.monitor.current_metrics.architecture_type, "MMDiT")
        self.assertEqual(self.monitor.current_metrics.model_name, "Qwen-Image")
        self.assertEqual(self.monitor.current_metrics.image_resolution, (1024, 1024))
    
    def test_performance_warnings_for_mmdit(self):
        """Test performance warnings specific to MMDiT"""
        # Set up slow MMDiT performance
        self.monitor.current_metrics.architecture_type = "MMDiT"
        self.monitor.current_metrics.total_generation_time = 15.0  # Very slow
        self.monitor.current_metrics.generation_time_per_step = 7.5  # Very slow per step
        self.monitor.current_metrics.target_generation_time = 5.0
        self.monitor.current_metrics.gpu_memory_utilization_percent = 30.0  # Low utilization
        
        # This should trigger MMDiT-specific warnings in the logs
        # We can't easily test log output, but we can verify the method runs without error
        self.monitor._log_performance_warnings()
        
        # Verify target was not met
        self.monitor._finalize_generation_metrics()
        self.assertFalse(self.monitor.current_metrics.target_met)
    
    def test_mmdit_vs_unet_comparison(self):
        """Test comparison between MMDiT and UNet architectures"""
        # Test MMDiT metrics
        mmdit_metrics = PerformanceMetrics(
            architecture_type="MMDiT",
            model_name="Qwen-Image",
            total_generation_time=3.0,
            generation_time_per_step=0.15,
            target_met=True
        )
        
        # Test UNet metrics
        unet_metrics = PerformanceMetrics(
            architecture_type="UNet",
            model_name="Stable-Diffusion",
            total_generation_time=4.0,
            generation_time_per_step=0.2,
            target_met=True
        )
        
        # Both should be valid but have different characteristics
        self.assertEqual(mmdit_metrics.architecture_type, "MMDiT")
        self.assertEqual(unet_metrics.architecture_type, "UNet")
        self.assertLess(mmdit_metrics.total_generation_time, unet_metrics.total_generation_time)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)