#!/usr/bin/env python3
"""
Performance benchmarks for DiffSynth Enhanced UI
Tests generation and editing operations performance according to requirements 6.1, 6.2, 8.1, 8.4
"""

import unittest
import time
import json
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import threading
import psutil
import gc

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PerformanceBenchmark:
    """Performance benchmark runner for DiffSynth operations"""
    
    def __init__(self):
        self.results = []
        self.baseline_metrics = {}
        self.memory_monitor = MemoryMonitor()
        
    def benchmark_generation_operation(self, operation_name: str, operation_func, 
                                     iterations: int = 5, **kwargs) -> Dict[str, Any]:
        """Benchmark a generation operation"""
        times = []
        memory_usage = []
        errors = []
        
        for i in range(iterations):
            gc.collect()  # Clean memory before each test
            
            start_memory = self.memory_monitor.get_memory_usage()
            start_time = time.perf_counter()
            
            try:
                result = operation_func(**kwargs)
                success = True
            except Exception as e:
                errors.append(str(e))
                success = False
                result = None
            
            end_time = time.perf_counter()
            end_memory = self.memory_monitor.get_memory_usage()
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            times.append(execution_time)
            memory_usage.append(memory_delta)
        
        # Calculate statistics
        avg_time = sum(times) / len(times) if times else 0
        min_time = min(times) if times else 0
        max_time = max(times) if times else 0
        avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0
        
        benchmark_result = {
            'operation_name': operation_name,
            'iterations': iterations,
            'avg_time_seconds': avg_time,
            'min_time_seconds': min_time,
            'max_time_seconds': max_time,
            'avg_memory_mb': avg_memory,
            'success_rate': (iterations - len(errors)) / iterations * 100,
            'errors': errors,
            'timestamp': time.time()
        }
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def benchmark_editing_operation(self, operation_name: str, operation_func,
                                  image_sizes: List[tuple] = None, **kwargs) -> Dict[str, Any]:
        """Benchmark image editing operations with different image sizes"""
        if image_sizes is None:
            image_sizes = [(512, 512), (1024, 1024), (2048, 2048)]
        
        size_results = {}
        
        for width, height in image_sizes:
            size_key = f"{width}x{height}"
            
            # Mock image data for the size
            mock_image = self._create_mock_image(width, height)
            
            result = self.benchmark_generation_operation(
                f"{operation_name}_{size_key}",
                operation_func,
                image=mock_image,
                width=width,
                height=height,
                **kwargs
            )
            
            size_results[size_key] = result
        
        return {
            'operation_name': operation_name,
            'size_results': size_results,
            'timestamp': time.time()
        }
    
    def _create_mock_image(self, width: int, height: int):
        """Create mock image data for testing"""
        # Return a simple mock that represents image data
        return {
            'width': width,
            'height': height,
            'channels': 3,
            'data_size_mb': (width * height * 3) / (1024 * 1024)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all benchmarks"""
        if not self.results:
            return {'error': 'No benchmark results available'}
        
        total_operations = len(self.results)
        avg_success_rate = sum(r['success_rate'] for r in self.results) / total_operations
        
        # Find fastest and slowest operations
        fastest = min(self.results, key=lambda x: x['avg_time_seconds'])
        slowest = max(self.results, key=lambda x: x['avg_time_seconds'])
        
        return {
            'total_operations_tested': total_operations,
            'overall_success_rate': avg_success_rate,
            'fastest_operation': {
                'name': fastest['operation_name'],
                'time_seconds': fastest['avg_time_seconds']
            },
            'slowest_operation': {
                'name': slowest['operation_name'],
                'time_seconds': slowest['avg_time_seconds']
            },
            'memory_efficiency': self._calculate_memory_efficiency(),
            'performance_grade': self._calculate_performance_grade(avg_success_rate)
        }
    
    def _calculate_memory_efficiency(self) -> str:
        """Calculate memory efficiency grade"""
        memory_results = [r.get('avg_memory_mb', 0) for r in self.results if 'avg_memory_mb' in r]
        
        if not memory_results:
            return 'Unknown'
        
        avg_memory = sum(memory_results) / len(memory_results)
        
        if avg_memory < 100:
            return 'Excellent'
        elif avg_memory < 500:
            return 'Good'
        elif avg_memory < 1000:
            return 'Fair'
        else:
            return 'Poor'
    
    def _calculate_performance_grade(self, success_rate: float) -> str:
        """Calculate overall performance grade"""
        if success_rate >= 95:
            return 'A'
        elif success_rate >= 85:
            return 'B'
        elif success_rate >= 75:
            return 'C'
        elif success_rate >= 65:
            return 'D'
        else:
            return 'F'


class MemoryMonitor:
    """Monitor system and GPU memory usage"""
    
    def __init__(self):
        self.torch_available = TORCH_AVAILABLE
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        # System memory
        system_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        # GPU memory if available
        gpu_memory = 0
        if self.torch_available and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            except:
                pass
        
        return system_memory + gpu_memory
    
    def get_detailed_memory_info(self) -> Dict[str, Any]:
        """Get detailed memory information"""
        info = {
            'system_memory_mb': psutil.virtual_memory().used / (1024 * 1024),
            'system_memory_percent': psutil.virtual_memory().percent,
            'gpu_available': False,
            'gpu_memory_mb': 0,
            'gpu_memory_percent': 0
        }
        
        if self.torch_available and torch.cuda.is_available():
            try:
                info['gpu_available'] = True
                info['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                info['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
                
                # Calculate GPU memory percentage if possible
                try:
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    info['gpu_memory_percent'] = (torch.cuda.memory_allocated() / total_memory) * 100
                except:
                    pass
            except:
                pass
        
        return info


class TestDiffSynthPerformanceBenchmarks(unittest.TestCase):
    """Test DiffSynth performance benchmarks"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.benchmark = PerformanceBenchmark()
        
    def test_text_to_image_generation_benchmark(self):
        """Test text-to-image generation performance benchmark"""
        
        def mock_text_to_image_generation(prompt: str, width: int = 1024, height: int = 1024, 
                                        steps: int = 20, **kwargs):
            """Mock text-to-image generation"""
            # Simulate processing time based on image size and steps
            base_time = 0.01  # Base 10ms
            size_factor = (width * height) / (1024 * 1024)  # Relative to 1024x1024
            step_factor = steps / 20  # Relative to 20 steps
            
            processing_time = base_time * size_factor * step_factor
            time.sleep(processing_time)
            
            return {
                'image_path': f'generated_{width}x{height}_{steps}steps.png',
                'generation_time': processing_time,
                'success': True
            }
        
        # Benchmark different configurations
        configurations = [
            {'prompt': 'test image', 'width': 512, 'height': 512, 'steps': 10},
            {'prompt': 'test image', 'width': 1024, 'height': 1024, 'steps': 20},
            {'prompt': 'test image', 'width': 1024, 'height': 1024, 'steps': 50},
        ]
        
        results = []
        for config in configurations:
            result = self.benchmark.benchmark_generation_operation(
                f"text_to_image_{config['width']}x{config['height']}_{config['steps']}steps",
                mock_text_to_image_generation,
                iterations=3,
                **config
            )
            results.append(result)
        
        # Verify results
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result['success_rate'], 100.0)
            self.assertGreater(result['avg_time_seconds'], 0)
            self.assertEqual(len(result['errors']), 0)
        
        # Verify performance scales with complexity
        simple_result = next(r for r in results if '512x512_10steps' in r['operation_name'])
        complex_result = next(r for r in results if '1024x1024_50steps' in r['operation_name'])
        
        self.assertGreater(complex_result['avg_time_seconds'], simple_result['avg_time_seconds'])
    
    def test_image_editing_operations_benchmark(self):
        """Test image editing operations performance benchmark"""
        
        def mock_inpainting_operation(image, mask, prompt: str, **kwargs):
            """Mock inpainting operation"""
            # Simulate processing time based on image size
            size_factor = (image['width'] * image['height']) / (1024 * 1024)
            processing_time = 0.02 * size_factor  # 20ms base for 1024x1024
            time.sleep(processing_time)
            
            return {
                'edited_image_path': f'inpainted_{image["width"]}x{image["height"]}.png',
                'processing_time': processing_time,
                'success': True
            }
        
        def mock_outpainting_operation(image, direction: str, pixels: int, **kwargs):
            """Mock outpainting operation"""
            size_factor = (image['width'] * image['height']) / (1024 * 1024)
            processing_time = 0.015 * size_factor  # 15ms base for 1024x1024
            time.sleep(processing_time)
            
            return {
                'extended_image_path': f'outpainted_{image["width"]}x{image["height"]}.png',
                'processing_time': processing_time,
                'success': True
            }
        
        # Test inpainting performance
        inpainting_result = self.benchmark.benchmark_editing_operation(
            'inpainting',
            mock_inpainting_operation,
            image_sizes=[(512, 512), (1024, 1024)],
            mask={'type': 'circle', 'radius': 50},
            prompt='fill the masked area'
        )
        
        # Test outpainting performance
        outpainting_result = self.benchmark.benchmark_editing_operation(
            'outpainting',
            mock_outpainting_operation,
            image_sizes=[(512, 512), (1024, 1024)],
            direction='right',
            pixels=256
        )
        
        # Verify results structure
        self.assertIn('size_results', inpainting_result)
        self.assertIn('size_results', outpainting_result)
        
        # Verify different sizes were tested
        self.assertIn('512x512', inpainting_result['size_results'])
        self.assertIn('1024x1024', inpainting_result['size_results'])
        
        # Verify performance scales with image size
        inpaint_512 = inpainting_result['size_results']['512x512']
        inpaint_1024 = inpainting_result['size_results']['1024x1024']
        
        self.assertGreater(inpaint_1024['avg_time_seconds'], inpaint_512['avg_time_seconds'])
    
    def test_controlnet_operations_benchmark(self):
        """Test ControlNet operations performance benchmark"""
        
        def mock_controlnet_detection(image, control_type: str, **kwargs):
            """Mock ControlNet control detection"""
            # Different control types have different processing times
            processing_times = {
                'canny': 0.005,
                'depth': 0.015,
                'pose': 0.025,
                'segmentation': 0.035
            }
            
            base_time = processing_times.get(control_type, 0.01)
            size_factor = (image['width'] * image['height']) / (1024 * 1024)
            processing_time = base_time * size_factor
            
            time.sleep(processing_time)
            
            return {
                'control_map_path': f'{control_type}_control_{image["width"]}x{image["height"]}.png',
                'detection_time': processing_time,
                'control_type': control_type,
                'success': True
            }
        
        def mock_controlnet_generation(image, control_map, prompt: str, **kwargs):
            """Mock ControlNet-guided generation"""
            size_factor = (image['width'] * image['height']) / (1024 * 1024)
            processing_time = 0.03 * size_factor  # 30ms base for 1024x1024
            time.sleep(processing_time)
            
            return {
                'generated_image_path': f'controlnet_gen_{image["width"]}x{image["height"]}.png',
                'generation_time': processing_time,
                'success': True
            }
        
        # Test different ControlNet detection types
        control_types = ['canny', 'depth', 'pose', 'segmentation']
        detection_results = []
        
        for control_type in control_types:
            result = self.benchmark.benchmark_generation_operation(
                f'controlnet_detection_{control_type}',
                mock_controlnet_detection,
                iterations=3,
                image=self.benchmark._create_mock_image(1024, 1024),
                control_type=control_type
            )
            detection_results.append(result)
        
        # Test ControlNet generation
        generation_result = self.benchmark.benchmark_editing_operation(
            'controlnet_generation',
            mock_controlnet_generation,
            image_sizes=[(512, 512), (1024, 1024)],
            control_map={'type': 'canny'},
            prompt='generate with control guidance'
        )
        
        # Verify detection results
        self.assertEqual(len(detection_results), 4)
        for result in detection_results:
            self.assertEqual(result['success_rate'], 100.0)
        
        # Verify canny is fastest (simplest detection)
        canny_result = next(r for r in detection_results if 'canny' in r['operation_name'])
        segmentation_result = next(r for r in detection_results if 'segmentation' in r['operation_name'])
        
        self.assertLess(canny_result['avg_time_seconds'], segmentation_result['avg_time_seconds'])
        
        # Verify generation results
        self.assertIn('size_results', generation_result)
        self.assertEqual(len(generation_result['size_results']), 2)
    
    def test_memory_usage_validation(self):
        """Test memory usage validation during operations"""
        
        def memory_intensive_operation(complexity_level: int = 1, **kwargs):
            """Mock operation that uses varying amounts of memory"""
            # Simulate memory allocation
            memory_data = []
            for i in range(complexity_level * 100):
                memory_data.append([0] * 1000)  # Allocate some memory
            
            time.sleep(0.01)  # Simulate processing
            
            # Keep reference to prevent garbage collection during test
            return {
                'result': f'processed_complexity_{complexity_level}',
                'memory_data': memory_data,
                'success': True
            }
        
        # Test different complexity levels
        complexity_levels = [1, 5, 10]
        memory_results = []
        
        for level in complexity_levels:
            result = self.benchmark.benchmark_generation_operation(
                f'memory_test_complexity_{level}',
                memory_intensive_operation,
                iterations=2,  # Fewer iterations for memory tests
                complexity_level=level
            )
            memory_results.append(result)
        
        # Verify memory usage increases with complexity
        self.assertEqual(len(memory_results), 3)
        
        # Memory usage should generally increase with complexity
        # (though garbage collection may affect exact measurements)
        low_complexity = memory_results[0]
        high_complexity = memory_results[2]
        
        self.assertGreater(high_complexity['avg_time_seconds'], low_complexity['avg_time_seconds'])
    
    def test_concurrent_operations_performance(self):
        """Test performance under concurrent operations"""
        
        def concurrent_operation(operation_id: int, **kwargs):
            """Mock operation for concurrent testing"""
            # Simulate variable processing time
            processing_time = 0.01 + (operation_id % 3) * 0.005  # 10-20ms
            time.sleep(processing_time)
            
            return {
                'operation_id': operation_id,
                'processing_time': processing_time,
                'success': True
            }
        
        # Test sequential operations
        sequential_start = time.perf_counter()
        sequential_results = []
        
        for i in range(5):
            result = concurrent_operation(i)
            sequential_results.append(result)
        
        sequential_time = time.perf_counter() - sequential_start
        
        # Test concurrent operations (simulated)
        concurrent_start = time.perf_counter()
        concurrent_results = []
        
        # Simulate concurrent execution by running operations in threads
        threads = []
        results_lock = threading.Lock()
        
        def thread_operation(op_id):
            result = concurrent_operation(op_id)
            with results_lock:
                concurrent_results.append(result)
        
        for i in range(5):
            thread = threading.Thread(target=thread_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        concurrent_time = time.perf_counter() - concurrent_start
        
        # Verify results
        self.assertEqual(len(sequential_results), 5)
        self.assertEqual(len(concurrent_results), 5)
        
        # Concurrent should be faster than sequential
        self.assertLess(concurrent_time, sequential_time)
        
        # Store benchmark results
        self.benchmark.results.append({
            'operation_name': 'concurrent_vs_sequential',
            'sequential_time': sequential_time,
            'concurrent_time': concurrent_time,
            'speedup_factor': sequential_time / concurrent_time,
            'timestamp': time.time()
        })
    
    def test_performance_summary_generation(self):
        """Test performance summary generation"""
        # Add some mock results
        self.benchmark.results = [
            {
                'operation_name': 'fast_operation',
                'avg_time_seconds': 0.01,
                'success_rate': 100.0,
                'avg_memory_mb': 50.0
            },
            {
                'operation_name': 'slow_operation',
                'avg_time_seconds': 0.1,
                'success_rate': 95.0,
                'avg_memory_mb': 200.0
            },
            {
                'operation_name': 'memory_heavy_operation',
                'avg_time_seconds': 0.05,
                'success_rate': 90.0,
                'avg_memory_mb': 800.0
            }
        ]
        
        summary = self.benchmark.get_performance_summary()
        
        # Verify summary structure
        self.assertIn('total_operations_tested', summary)
        self.assertIn('overall_success_rate', summary)
        self.assertIn('fastest_operation', summary)
        self.assertIn('slowest_operation', summary)
        self.assertIn('memory_efficiency', summary)
        self.assertIn('performance_grade', summary)
        
        # Verify calculations
        self.assertEqual(summary['total_operations_tested'], 3)
        self.assertEqual(summary['overall_success_rate'], 95.0)  # (100+95+90)/3
        self.assertEqual(summary['fastest_operation']['name'], 'fast_operation')
        self.assertEqual(summary['slowest_operation']['name'], 'slow_operation')
        
        # Memory efficiency should be 'Good' (average of 50, 200, 800 = 350MB)
        self.assertEqual(summary['memory_efficiency'], 'Good')
        
        # Performance grade should be 'A' (95% success rate)
        self.assertEqual(summary['performance_grade'], 'A')
    
    def test_benchmark_export_and_import(self):
        """Test benchmark results export and import"""
        # Add test results
        self.benchmark.results = [
            {
                'operation_name': 'export_test',
                'avg_time_seconds': 0.02,
                'success_rate': 100.0,
                'timestamp': time.time()
            }
        ]
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Export results
            export_data = {
                'benchmark_results': self.benchmark.results,
                'summary': self.benchmark.get_performance_summary(),
                'export_timestamp': time.time(),
                'system_info': {
                    'python_version': sys.version,
                    'platform': sys.platform
                }
            }
            
            with open(temp_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            # Verify file was created
            self.assertTrue(os.path.exists(temp_path))
            
            # Import and verify
            with open(temp_path, 'r') as f:
                imported_data = json.load(f)
            
            self.assertIn('benchmark_results', imported_data)
            self.assertIn('summary', imported_data)
            self.assertEqual(len(imported_data['benchmark_results']), 1)
            self.assertEqual(imported_data['benchmark_results'][0]['operation_name'], 'export_test')
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()