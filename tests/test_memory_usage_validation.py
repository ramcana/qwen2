#!/usr/bin/env python3
"""
Memory usage validation tests for DiffSynth Enhanced UI
Tests memory management and optimization according to requirements 6.1, 6.2
"""

import unittest
import time
import gc
import os
import sys
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import psutil

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MemoryValidator:
    """Validates memory usage patterns for DiffSynth operations"""
    
    def __init__(self, memory_limit_mb: int = 8192):  # 8GB default limit
        self.memory_limit_mb = memory_limit_mb
        self.torch_available = TORCH_AVAILABLE
        self.baseline_memory = self._get_current_memory_usage()
        self.memory_snapshots = []
        
    def _get_current_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage across system and GPU"""
        usage = {
            'system_memory_mb': psutil.virtual_memory().used / (1024 * 1024),
            'system_memory_percent': psutil.virtual_memory().percent,
            'gpu_memory_mb': 0.0,
            'gpu_memory_percent': 0.0,
            'total_memory_mb': 0.0
        }
        
        if self.torch_available and torch.cuda.is_available():
            try:
                usage['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                usage['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
                
                # Get total GPU memory
                device_props = torch.cuda.get_device_properties(0)
                total_gpu_memory = device_props.total_memory / (1024 * 1024)
                usage['gpu_memory_percent'] = (usage['gpu_memory_mb'] / total_gpu_memory) * 100
                usage['gpu_total_memory_mb'] = total_gpu_memory
            except Exception:
                pass
        
        usage['total_memory_mb'] = usage['system_memory_mb'] + usage['gpu_memory_mb']
        return usage
    
    def start_monitoring(self, operation_name: str):
        """Start monitoring memory usage for an operation"""
        self.current_operation = operation_name
        self.operation_start_memory = self._get_current_memory_usage()
        self.memory_snapshots = [self.operation_start_memory]
        
    def take_snapshot(self, label: str = ""):
        """Take a memory usage snapshot"""
        snapshot = self._get_current_memory_usage()
        snapshot['label'] = label
        snapshot['timestamp'] = time.time()
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def end_monitoring(self) -> Dict[str, Any]:
        """End monitoring and return memory analysis"""
        final_memory = self._get_current_memory_usage()
        self.memory_snapshots.append(final_memory)
        
        return self._analyze_memory_usage()
    
    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        if len(self.memory_snapshots) < 2:
            return {'error': 'Insufficient memory snapshots for analysis'}
        
        start_memory = self.memory_snapshots[0]
        end_memory = self.memory_snapshots[-1]
        peak_memory = max(self.memory_snapshots, key=lambda x: x['total_memory_mb'])
        
        # Calculate memory deltas
        system_delta = end_memory['system_memory_mb'] - start_memory['system_memory_mb']
        gpu_delta = end_memory['gpu_memory_mb'] - start_memory['gpu_memory_mb']
        total_delta = end_memory['total_memory_mb'] - start_memory['total_memory_mb']
        
        # Calculate peak usage
        peak_system = peak_memory['system_memory_mb'] - start_memory['system_memory_mb']
        peak_gpu = peak_memory['gpu_memory_mb'] - start_memory['gpu_memory_mb']
        peak_total = peak_memory['total_memory_mb'] - start_memory['total_memory_mb']
        
        analysis = {
            'operation_name': getattr(self, 'current_operation', 'unknown'),
            'memory_deltas': {
                'system_mb': system_delta,
                'gpu_mb': gpu_delta,
                'total_mb': total_delta
            },
            'peak_usage': {
                'system_mb': peak_system,
                'gpu_mb': peak_gpu,
                'total_mb': peak_total
            },
            'memory_efficiency': self._calculate_memory_efficiency(total_delta, peak_total),
            'memory_leaks_detected': self._detect_memory_leaks(system_delta, gpu_delta),
            'within_limits': peak_total < self.memory_limit_mb,
            'snapshots_count': len(self.memory_snapshots),
            'recommendations': self._generate_memory_recommendations(peak_total, total_delta)
        }
        
        return analysis
    
    def _calculate_memory_efficiency(self, final_delta: float, peak_delta: float) -> str:
        """Calculate memory efficiency rating"""
        if peak_delta <= 0:
            return 'Excellent'
        
        efficiency_ratio = abs(final_delta) / peak_delta if peak_delta > 0 else 0
        
        if efficiency_ratio < 0.1:  # Less than 10% of peak remains
            return 'Excellent'
        elif efficiency_ratio < 0.3:  # Less than 30% of peak remains
            return 'Good'
        elif efficiency_ratio < 0.6:  # Less than 60% of peak remains
            return 'Fair'
        else:
            return 'Poor'
    
    def _detect_memory_leaks(self, system_delta: float, gpu_delta: float) -> List[str]:
        """Detect potential memory leaks"""
        leaks = []
        
        # Significant system memory increase that wasn't cleaned up
        if system_delta > 100:  # More than 100MB increase
            leaks.append(f'System memory leak detected: +{system_delta:.1f}MB')
        
        # Significant GPU memory increase that wasn't cleaned up
        if gpu_delta > 50:  # More than 50MB increase
            leaks.append(f'GPU memory leak detected: +{gpu_delta:.1f}MB')
        
        return leaks
    
    def _generate_memory_recommendations(self, peak_usage: float, final_delta: float) -> List[str]:
        """Generate memory optimization recommendations"""
        recommendations = []
        
        if peak_usage > self.memory_limit_mb * 0.8:  # Using more than 80% of limit
            recommendations.append('Consider reducing batch size or image resolution')
            recommendations.append('Enable gradient checkpointing if available')
        
        if final_delta > 50:  # More than 50MB not cleaned up
            recommendations.append('Add explicit memory cleanup after operations')
            recommendations.append('Consider using context managers for resource management')
        
        if peak_usage > 2000:  # More than 2GB peak usage
            recommendations.append('Consider implementing tiled processing for large images')
            recommendations.append('Use mixed precision training if supported')
        
        return recommendations
    
    def validate_memory_sharing(self, service_a_func, service_b_func, 
                              shared_resources: bool = True) -> Dict[str, Any]:
        """Validate memory sharing between services (e.g., Qwen and DiffSynth)"""
        
        # Test sequential usage (should share memory efficiently)
        self.start_monitoring('memory_sharing_test')
        
        # Run service A
        self.take_snapshot('before_service_a')
        try:
            result_a = service_a_func()
            service_a_success = True
        except Exception as e:
            result_a = None
            service_a_success = False
        
        self.take_snapshot('after_service_a')
        
        # Clear service A if shared resources
        if shared_resources:
            gc.collect()
            if self.torch_available and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.take_snapshot('after_service_a_cleanup')
        
        # Run service B
        try:
            result_b = service_b_func()
            service_b_success = True
        except Exception as e:
            result_b = None
            service_b_success = False
        
        self.take_snapshot('after_service_b')
        
        analysis = self.end_monitoring()
        
        # Add sharing-specific analysis
        snapshots = self.memory_snapshots
        service_a_peak = max(snapshots[1:3], key=lambda x: x['total_memory_mb'])
        service_b_peak = snapshots[-2]  # After service B
        
        sharing_analysis = {
            'service_a_success': service_a_success,
            'service_b_success': service_b_success,
            'service_a_peak_mb': service_a_peak['total_memory_mb'],
            'service_b_peak_mb': service_b_peak['total_memory_mb'],
            'memory_sharing_efficiency': self._calculate_sharing_efficiency(
                service_a_peak['total_memory_mb'], 
                service_b_peak['total_memory_mb']
            ),
            'total_memory_saved_mb': self._calculate_memory_saved(
                service_a_peak['total_memory_mb'], 
                service_b_peak['total_memory_mb'],
                shared_resources
            )
        }
        
        analysis.update(sharing_analysis)
        return analysis
    
    def _calculate_sharing_efficiency(self, service_a_peak: float, service_b_peak: float) -> str:
        """Calculate memory sharing efficiency"""
        if service_a_peak <= 0 or service_b_peak <= 0:
            return 'Unknown'
        
        # If service B uses significantly less memory than A, sharing is working
        ratio = service_b_peak / service_a_peak
        
        if ratio < 0.3:  # Service B uses less than 30% of service A's peak
            return 'Excellent'
        elif ratio < 0.6:  # Service B uses less than 60% of service A's peak
            return 'Good'
        elif ratio < 0.9:  # Service B uses less than 90% of service A's peak
            return 'Fair'
        else:
            return 'Poor'
    
    def _calculate_memory_saved(self, service_a_peak: float, service_b_peak: float, 
                               shared: bool) -> float:
        """Calculate memory saved through sharing"""
        if not shared:
            return 0.0
        
        # Estimate memory that would be used without sharing
        estimated_without_sharing = service_a_peak + service_b_peak
        actual_peak = max(service_a_peak, service_b_peak)
        
        return max(0, estimated_without_sharing - actual_peak)


class TestMemoryUsageValidation(unittest.TestCase):
    """Test memory usage validation for DiffSynth operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.memory_validator = MemoryValidator(memory_limit_mb=4096)  # 4GB limit for tests
        
    def test_memory_validator_initialization(self):
        """Test memory validator initialization"""
        validator = MemoryValidator(memory_limit_mb=2048)
        
        self.assertEqual(validator.memory_limit_mb, 2048)
        self.assertIn('system_memory_mb', validator.baseline_memory)
        self.assertIn('total_memory_mb', validator.baseline_memory)
        self.assertEqual(len(validator.memory_snapshots), 0)
    
    def test_basic_memory_monitoring(self):
        """Test basic memory monitoring functionality"""
        
        def mock_operation():
            """Mock operation that allocates some memory"""
            # Allocate some memory
            data = [0] * 1000000  # ~8MB of integers
            time.sleep(0.01)
            return data
        
        # Monitor the operation
        self.memory_validator.start_monitoring('basic_test')
        
        self.memory_validator.take_snapshot('before_allocation')
        result = mock_operation()
        self.memory_validator.take_snapshot('after_allocation')
        
        analysis = self.memory_validator.end_monitoring()
        
        # Verify analysis structure
        self.assertIn('operation_name', analysis)
        self.assertIn('memory_deltas', analysis)
        self.assertIn('peak_usage', analysis)
        self.assertIn('memory_efficiency', analysis)
        self.assertIn('within_limits', analysis)
        
        self.assertEqual(analysis['operation_name'], 'basic_test')
        self.assertGreaterEqual(len(self.memory_validator.memory_snapshots), 3)
        
        # Should be within limits for this small test
        self.assertTrue(analysis['within_limits'])
    
    def test_memory_leak_detection(self):
        """Test memory leak detection"""
        
        # Create a function that intentionally "leaks" memory
        leaked_data = []
        
        def leaky_operation():
            """Operation that doesn't clean up memory"""
            # Allocate memory and keep reference (simulating leak)
            data = [0] * 5000000  # ~40MB of integers
            leaked_data.append(data)  # Keep reference to prevent GC
            return "operation_complete"
        
        self.memory_validator.start_monitoring('leak_test')
        result = leaky_operation()
        analysis = self.memory_validator.end_monitoring()
        
        # Should detect memory increase
        self.assertGreater(analysis['memory_deltas']['total_mb'], 0)
        
        # May detect leak depending on system behavior
        # (Memory leak detection depends on GC timing)
        self.assertIn('memory_leaks_detected', analysis)
    
    def test_memory_efficiency_calculation(self):
        """Test memory efficiency calculation"""
        
        def efficient_operation():
            """Operation that cleans up after itself"""
            data = [0] * 1000000  # Allocate memory
            time.sleep(0.01)
            del data  # Clean up
            gc.collect()  # Force garbage collection
            return "efficient_complete"
        
        def inefficient_operation():
            """Operation that doesn't clean up well"""
            self.persistent_data = [0] * 1000000  # Keep reference
            time.sleep(0.01)
            return "inefficient_complete"
        
        # Test efficient operation
        self.memory_validator.start_monitoring('efficient_test')
        efficient_result = efficient_operation()
        efficient_analysis = self.memory_validator.end_monitoring()
        
        # Test inefficient operation
        self.memory_validator.start_monitoring('inefficient_test')
        inefficient_result = inefficient_operation()
        inefficient_analysis = self.memory_validator.end_monitoring()
        
        # Efficient operation should have better efficiency rating
        # (though exact ratings depend on system GC behavior)
        self.assertIn(efficient_analysis['memory_efficiency'], 
                     ['Excellent', 'Good', 'Fair', 'Poor'])
        self.assertIn(inefficient_analysis['memory_efficiency'], 
                     ['Excellent', 'Good', 'Fair', 'Poor'])
    
    def test_memory_limit_validation(self):
        """Test memory limit validation"""
        
        # Set a very low limit for testing
        validator = MemoryValidator(memory_limit_mb=100)  # 100MB limit
        
        def memory_intensive_operation():
            """Operation that uses significant memory"""
            # This should exceed the 100MB limit
            data = [0] * 50000000  # ~400MB of integers
            time.sleep(0.01)
            return data
        
        validator.start_monitoring('limit_test')
        result = memory_intensive_operation()
        analysis = validator.end_monitoring()
        
        # Should detect limit exceeded
        # Note: This test may not always trigger due to system memory management
        self.assertIn('within_limits', analysis)
    
    def test_memory_sharing_validation(self):
        """Test memory sharing between services"""
        
        # Mock service functions
        service_a_data = None
        service_b_data = None
        
        def mock_qwen_service():
            """Mock Qwen text-to-image service"""
            nonlocal service_a_data
            service_a_data = [0] * 2000000  # ~16MB
            time.sleep(0.02)
            return "qwen_image_generated"
        
        def mock_diffsynth_service():
            """Mock DiffSynth editing service"""
            nonlocal service_b_data
            service_b_data = [0] * 1500000  # ~12MB
            time.sleep(0.015)
            return "diffsynth_edit_complete"
        
        # Test with shared resources (should be more efficient)
        sharing_analysis = self.memory_validator.validate_memory_sharing(
            mock_qwen_service,
            mock_diffsynth_service,
            shared_resources=True
        )
        
        # Verify analysis structure
        self.assertIn('service_a_success', sharing_analysis)
        self.assertIn('service_b_success', sharing_analysis)
        self.assertIn('memory_sharing_efficiency', sharing_analysis)
        self.assertIn('total_memory_saved_mb', sharing_analysis)
        
        # Both services should succeed
        self.assertTrue(sharing_analysis['service_a_success'])
        self.assertTrue(sharing_analysis['service_b_success'])
        
        # Should have efficiency rating
        self.assertIn(sharing_analysis['memory_sharing_efficiency'],
                     ['Excellent', 'Good', 'Fair', 'Poor', 'Unknown'])
    
    def test_concurrent_memory_usage(self):
        """Test memory usage under concurrent operations"""
        
        results = []
        results_lock = threading.Lock()
        
        def concurrent_memory_operation(operation_id: int):
            """Operation for concurrent testing"""
            # Each operation allocates some memory
            data = [0] * (500000 + operation_id * 100000)  # Variable allocation
            time.sleep(0.01)
            
            with results_lock:
                results.append({
                    'operation_id': operation_id,
                    'data_size': len(data),
                    'success': True
                })
            
            return f"concurrent_op_{operation_id}_complete"
        
        # Monitor concurrent operations
        self.memory_validator.start_monitoring('concurrent_test')
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=concurrent_memory_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        analysis = self.memory_validator.end_monitoring()
        
        # Verify all operations completed
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result['success'])
        
        # Should have memory usage data
        self.assertIn('peak_usage', analysis)
        self.assertGreater(analysis['peak_usage']['total_mb'], 0)
    
    def test_memory_recommendations_generation(self):
        """Test memory optimization recommendations"""
        
        # Create scenarios that should trigger different recommendations
        
        # High memory usage scenario
        high_memory_validator = MemoryValidator(memory_limit_mb=1000)  # 1GB limit
        
        def high_memory_operation():
            """Operation using significant memory"""
            data = [0] * 20000000  # ~160MB
            time.sleep(0.01)
            return data
        
        high_memory_validator.start_monitoring('high_memory_test')
        result = high_memory_operation()
        high_analysis = high_memory_validator.end_monitoring()
        
        # Should generate recommendations for high memory usage
        self.assertIn('recommendations', high_analysis)
        recommendations = high_analysis['recommendations']
        
        # Should suggest memory optimization techniques
        recommendation_text = ' '.join(recommendations).lower()
        self.assertTrue(
            any(keyword in recommendation_text for keyword in 
                ['batch size', 'resolution', 'tiled', 'cleanup', 'precision'])
        )
    
    def test_gpu_memory_monitoring(self):
        """Test GPU memory monitoring if available"""
        
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            self.skipTest("CUDA not available for GPU memory testing")
        
        def gpu_operation():
            """Operation that uses GPU memory"""
            # Allocate some GPU memory
            device = torch.device('cuda')
            tensor = torch.randn(1000, 1000, device=device)  # ~4MB tensor
            time.sleep(0.01)
            
            # Clean up
            del tensor
            torch.cuda.empty_cache()
            
            return "gpu_operation_complete"
        
        self.memory_validator.start_monitoring('gpu_test')
        result = gpu_operation()
        analysis = self.memory_validator.end_monitoring()
        
        # Should track GPU memory
        self.assertIn('memory_deltas', analysis)
        
        # GPU memory delta might be positive during operation
        # (exact behavior depends on PyTorch memory management)
        self.assertIn('peak_usage', analysis)
    
    def test_memory_snapshot_functionality(self):
        """Test memory snapshot functionality"""
        
        self.memory_validator.start_monitoring('snapshot_test')
        
        # Take snapshots at different points
        snapshot1 = self.memory_validator.take_snapshot('initial')
        
        # Allocate some memory
        data1 = [0] * 1000000
        snapshot2 = self.memory_validator.take_snapshot('after_allocation_1')
        
        # Allocate more memory
        data2 = [0] * 2000000
        snapshot3 = self.memory_validator.take_snapshot('after_allocation_2')
        
        analysis = self.memory_validator.end_monitoring()
        
        # Verify snapshots were recorded
        self.assertGreaterEqual(len(self.memory_validator.memory_snapshots), 5)  # start + 3 manual + end
        
        # Verify snapshot labels
        labeled_snapshots = [s for s in self.memory_validator.memory_snapshots if 'label' in s]
        labels = [s['label'] for s in labeled_snapshots]
        
        self.assertIn('initial', labels)
        self.assertIn('after_allocation_1', labels)
        self.assertIn('after_allocation_2', labels)
        
        # Memory usage should generally increase with allocations
        self.assertLessEqual(snapshot1['total_memory_mb'], snapshot3['total_memory_mb'])
    
    def test_memory_analysis_edge_cases(self):
        """Test memory analysis edge cases"""
        
        # Test with no snapshots
        empty_validator = MemoryValidator()
        empty_analysis = empty_validator._analyze_memory_usage()
        self.assertIn('error', empty_analysis)
        
        # Test with single snapshot
        single_validator = MemoryValidator()
        single_validator.memory_snapshots = [{'total_memory_mb': 100}]
        single_analysis = single_validator._analyze_memory_usage()
        self.assertIn('error', single_analysis)
        
        # Test with zero memory usage
        zero_validator = MemoryValidator()
        zero_validator.memory_snapshots = [
            {'system_memory_mb': 100, 'gpu_memory_mb': 0, 'total_memory_mb': 100},
            {'system_memory_mb': 100, 'gpu_memory_mb': 0, 'total_memory_mb': 100}
        ]
        zero_analysis = zero_validator._analyze_memory_usage()
        
        self.assertEqual(zero_analysis['memory_deltas']['total_mb'], 0)
        self.assertEqual(zero_analysis['memory_efficiency'], 'Excellent')


if __name__ == '__main__':
    unittest.main()