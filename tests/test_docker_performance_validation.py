#!/usr/bin/env python3
"""
Docker Performance Validation Tests
Tests resource usage, performance metrics, and optimization in Docker environment.
"""

import asyncio
import json
import os
import sys
import time
import uuid
import subprocess
import requests
import pytest
import psutil
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import docker
from docker.errors import DockerException
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_limit_mb: float
    memory_usage_percent: float
    network_rx_bytes: int
    network_tx_bytes: int
    disk_read_bytes: int
    disk_write_bytes: int
    gpu_usage_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    timestamp: float = 0.0


@dataclass
class LoadTestResult:
    """Container for load test results"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    error_rate: float
    peak_memory_usage: float
    peak_cpu_usage: float


class DockerPerformanceValidator:
    """Docker performance validation and monitoring"""
    
    def __init__(self):
        self.docker_client = None
        self.monitoring_active = False
        self.metrics_history = []
        self.compose_project = f"qwen-perf-test-{uuid.uuid4().hex[:8]}"
        
    def setup_docker_client(self):
        """Initialize Docker client"""
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            return True
        except DockerException as e:
            pytest.skip(f"Docker not available: {e}")
            return False
    
    def get_container_metrics(self, container_name: str) -> Optional[PerformanceMetrics]:
        """Get performance metrics for a specific container"""
        if not self.docker_client:
            return None
        
        try:
            container = self.docker_client.containers.get(container_name)
            stats = container.stats(stream=False)
            
            # Calculate CPU usage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            cpu_usage_percent = 0.0
            if system_delta > 0:
                cpu_usage_percent = (cpu_delta / system_delta) * 100.0
            
            # Calculate memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_usage_mb = memory_usage / (1024 * 1024)
            memory_limit_mb = memory_limit / (1024 * 1024)
            memory_usage_percent = (memory_usage / memory_limit) * 100.0
            
            # Network stats
            networks = stats.get('networks', {})
            network_rx = sum(net['rx_bytes'] for net in networks.values())
            network_tx = sum(net['tx_bytes'] for net in networks.values())
            
            # Disk I/O stats
            blkio_stats = stats.get('blkio_stats', {})
            io_service_bytes = blkio_stats.get('io_service_bytes_recursive', [])
            
            disk_read = sum(entry['value'] for entry in io_service_bytes 
                           if entry['op'] == 'Read')
            disk_write = sum(entry['value'] for entry in io_service_bytes 
                            if entry['op'] == 'Write')
            
            return PerformanceMetrics(
                cpu_usage_percent=cpu_usage_percent,
                memory_usage_mb=memory_usage_mb,
                memory_limit_mb=memory_limit_mb,
                memory_usage_percent=memory_usage_percent,
                network_rx_bytes=network_rx,
                network_tx_bytes=network_tx,
                disk_read_bytes=disk_read,
                disk_write_bytes=disk_write,
                timestamp=time.time()
            )
            
        except Exception as e:
            print(f"Error getting metrics for {container_name}: {e}")
            return None
    
    def monitor_containers(self, container_names: List[str], duration: int = 60) -> List[Dict[str, Any]]:
        """Monitor containers for specified duration"""
        self.monitoring_active = True
        self.metrics_history = []
        
        def collect_metrics():
            while self.monitoring_active:
                timestamp = time.time()
                metrics_snapshot = {'timestamp': timestamp}
                
                for container_name in container_names:
                    metrics = self.get_container_metrics(container_name)
                    if metrics:
                        metrics_snapshot[container_name] = metrics
                
                self.metrics_history.append(metrics_snapshot)
                time.sleep(5)  # Collect metrics every 5 seconds
        
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=collect_metrics)
        monitor_thread.start()
        
        # Wait for specified duration
        time.sleep(duration)
        
        # Stop monitoring
        self.monitoring_active = False
        monitor_thread.join(timeout=10)
        
        return self.metrics_history
    
    def analyze_performance_metrics(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze collected performance metrics"""
        if not metrics_history:
            return {}
        
        analysis = {}
        
        # Get all container names
        container_names = set()
        for snapshot in metrics_history:
            container_names.update(k for k in snapshot.keys() if k != 'timestamp')
        
        for container_name in container_names:
            container_metrics = []
            
            for snapshot in metrics_history:
                if container_name in snapshot:
                    container_metrics.append(snapshot[container_name])
            
            if not container_metrics:
                continue
            
            # Calculate statistics
            cpu_values = [m.cpu_usage_percent for m in container_metrics]
            memory_values = [m.memory_usage_mb for m in container_metrics]
            memory_percent_values = [m.memory_usage_percent for m in container_metrics]
            
            analysis[container_name] = {
                'cpu_usage': {
                    'avg': sum(cpu_values) / len(cpu_values),
                    'min': min(cpu_values),
                    'max': max(cpu_values),
                    'samples': len(cpu_values)
                },
                'memory_usage_mb': {
                    'avg': sum(memory_values) / len(memory_values),
                    'min': min(memory_values),
                    'max': max(memory_values),
                    'samples': len(memory_values)
                },
                'memory_usage_percent': {
                    'avg': sum(memory_percent_values) / len(memory_percent_values),
                    'min': min(memory_percent_values),
                    'max': max(memory_percent_values),
                    'samples': len(memory_percent_values)
                },
                'total_samples': len(container_metrics)
            }
        
        return analysis
    
    def start_compose_stack_for_testing(self):
        """Start Docker Compose stack for performance testing"""
        compose_path = Path(__file__).parent.parent / "docker-compose.yml"
        
        cmd = [
            "docker-compose",
            "-f", str(compose_path),
            "-p", self.compose_project,
            "up", "-d"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                pytest.fail(f"Failed to start compose stack: {result.stderr}")
            
            # Wait for services to be ready
            time.sleep(30)
            return True
            
        except Exception as e:
            pytest.fail(f"Error starting compose stack: {e}")
    
    def stop_compose_stack(self):
        """Stop Docker Compose stack"""
        cmd = ["docker-compose", "-p", self.compose_project, "down", "-v"]
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
        except Exception as e:
            print(f"Warning: Error stopping compose stack: {e}")


class TestDockerResourceUsage(DockerPerformanceValidator):
    """Test Docker resource usage and limits"""
    
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup for each test"""
        self.setup_docker_client()
        yield
        self.stop_compose_stack()
    
    def test_container_memory_limits(self):
        """Test that containers respect memory limits"""
        if not self.docker_client:
            pytest.skip("Docker not available")
        
        # Start a container with memory limit
        memory_limit = "512m"
        container = self.docker_client.containers.run(
            "alpine:latest",
            name=f"test-memory-limit-{uuid.uuid4().hex[:8]}",
            command=["sleep", "60"],
            mem_limit=memory_limit,
            detach=True,
            remove=False
        )
        
        try:
            # Wait for container to start
            time.sleep(5)
            
            # Get container metrics
            metrics = self.get_container_metrics(container.name)
            assert metrics is not None
            
            # Verify memory limit is respected
            expected_limit_mb = 512
            assert abs(metrics.memory_limit_mb - expected_limit_mb) < 50  # Allow some variance
            
            # Memory usage should be well below limit for idle container
            assert metrics.memory_usage_percent < 50, f"Memory usage too high: {metrics.memory_usage_percent}%"
            
        finally:
            container.stop(timeout=10)
            container.remove(force=True)
    
    def test_container_cpu_limits(self):
        """Test that containers respect CPU limits"""
        if not self.docker_client:
            pytest.skip("Docker not available")
        
        # Start container with CPU limit (0.5 CPU cores)
        container = self.docker_client.containers.run(
            "alpine:latest",
            name=f"test-cpu-limit-{uuid.uuid4().hex[:8]}",
            command=["sh", "-c", "while true; do echo test; sleep 0.1; done"],
            nano_cpus=int(0.5 * 1e9),  # 0.5 CPU cores in nanoseconds
            detach=True,
            remove=False
        )
        
        try:
            # Monitor CPU usage for 30 seconds
            metrics_history = []
            for _ in range(6):  # 6 samples over 30 seconds
                time.sleep(5)
                metrics = self.get_container_metrics(container.name)
                if metrics:
                    metrics_history.append(metrics.cpu_usage_percent)
            
            # Calculate average CPU usage
            if metrics_history:
                avg_cpu = sum(metrics_history) / len(metrics_history)
                
                # CPU usage should be limited (allowing some variance)
                # With 0.5 CPU limit, usage should not exceed ~60% consistently
                assert avg_cpu < 80, f"CPU usage too high for limited container: {avg_cpu}%"
            
        finally:
            container.stop(timeout=10)
            container.remove(force=True)
    
    def test_api_container_resource_usage_under_load(self):
        """Test API container resource usage under load"""
        if not self.docker_client:
            pytest.skip("Docker not available")
        
        # Start compose stack
        self.start_compose_stack_for_testing()
        
        # Get API container name
        api_container_name = f"{self.compose_project}_api_1"
        
        # Start monitoring
        def monitor_resources():
            return self.monitor_containers([api_container_name], duration=60)
        
        # Start monitoring in background
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
        
        # Generate load on API
        def make_request():
            try:
                response = requests.post(
                    "http://localhost:8080/api/generate/text-to-image",
                    json={
                        "prompt": "Test image for load testing",
                        "width": 256,
                        "height": 256,
                        "num_inference_steps": 5
                    },
                    timeout=30
                )
                return response.status_code == 200
            except:
                return False
        
        # Make concurrent requests
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in as_completed(futures)]
        
        # Wait for monitoring to complete
        monitor_thread.join()
        
        # Analyze metrics
        analysis = self.analyze_performance_metrics(self.metrics_history)
        
        if api_container_name in analysis:
            api_metrics = analysis[api_container_name]
            
            # Verify resource usage is reasonable
            assert api_metrics['memory_usage_percent']['max'] < 90, \
                f"Memory usage too high: {api_metrics['memory_usage_percent']['max']}%"
            
            assert api_metrics['cpu_usage']['max'] < 200, \
                f"CPU usage too high: {api_metrics['cpu_usage']['max']}%"
            
            print(f"API container performance under load:")
            print(f"  Max memory usage: {api_metrics['memory_usage_percent']['max']:.1f}%")
            print(f"  Max CPU usage: {api_metrics['cpu_usage']['max']:.1f}%")
            print(f"  Successful requests: {sum(results)}/{len(results)}")


class TestDockerPerformanceBenchmarks(DockerPerformanceValidator):
    """Test Docker performance benchmarks and optimization"""
    
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup for each test"""
        self.setup_docker_client()
        yield
        self.stop_compose_stack()
    
    def test_container_startup_time(self):
        """Test container startup time performance"""
        if not self.docker_client:
            pytest.skip("Docker not available")
        
        startup_times = []
        
        # Test startup time for multiple containers
        for i in range(3):
            start_time = time.time()
            
            container = self.docker_client.containers.run(
                "qwen-api:latest",
                name=f"test-startup-{i}-{uuid.uuid4().hex[:8]}",
                environment={"QUICK_START": "true"},
                detach=True,
                remove=False
            )
            
            # Wait for container to be ready
            ready = False
            timeout = 120  # 2 minutes timeout
            
            while time.time() - start_time < timeout and not ready:
                try:
                    container.reload()
                    if container.status == 'running':
                        # Try to connect to health endpoint
                        networks = container.attrs['NetworkSettings']['Networks']
                        if networks:
                            network_name = list(networks.keys())[0]
                            ip_address = networks[network_name]['IPAddress']
                            
                            response = requests.get(f"http://{ip_address}:8000/health", timeout=5)
                            if response.status_code == 200:
                                ready = True
                except:
                    pass
                
                if not ready:
                    time.sleep(2)
            
            startup_time = time.time() - start_time
            startup_times.append(startup_time)
            
            # Clean up
            container.stop(timeout=10)
            container.remove(force=True)
            
            if not ready:
                pytest.fail(f"Container {i} did not start within {timeout} seconds")
        
        # Analyze startup times
        avg_startup_time = sum(startup_times) / len(startup_times)
        max_startup_time = max(startup_times)
        
        print(f"Container startup times:")
        print(f"  Average: {avg_startup_time:.1f}s")
        print(f"  Maximum: {max_startup_time:.1f}s")
        print(f"  All times: {[f'{t:.1f}s' for t in startup_times]}")
        
        # Verify startup times are reasonable
        assert avg_startup_time < 180, f"Average startup time too slow: {avg_startup_time:.1f}s"
        assert max_startup_time < 300, f"Maximum startup time too slow: {max_startup_time:.1f}s"
    
    def test_image_generation_performance_in_docker(self):
        """Test image generation performance in Docker vs expected benchmarks"""
        if not self.docker_client:
            pytest.skip("Docker not available")
        
        # Start compose stack
        self.start_compose_stack_for_testing()
        
        # Wait for services to be ready
        time.sleep(30)
        
        # Test different image generation scenarios
        test_cases = [
            {
                "name": "small_fast",
                "params": {"prompt": "A cat", "width": 256, "height": 256, "num_inference_steps": 10},
                "expected_max_time": 30
            },
            {
                "name": "medium_quality",
                "params": {"prompt": "A landscape", "width": 512, "height": 512, "num_inference_steps": 20},
                "expected_max_time": 60
            },
            {
                "name": "large_detailed",
                "params": {"prompt": "A detailed portrait", "width": 768, "height": 768, "num_inference_steps": 30},
                "expected_max_time": 120
            }
        ]
        
        performance_results = {}
        
        for test_case in test_cases:
            print(f"Testing {test_case['name']}...")
            
            start_time = time.time()
            
            try:
                response = requests.post(
                    "http://localhost:8080/api/generate/text-to-image",
                    json=test_case['params'],
                    timeout=test_case['expected_max_time'] + 30
                )
                
                generation_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        reported_time = data.get('generation_time', generation_time)
                        
                        performance_results[test_case['name']] = {
                            'success': True,
                            'total_time': generation_time,
                            'generation_time': reported_time,
                            'expected_max': test_case['expected_max_time']
                        }
                        
                        # Verify performance meets expectations
                        assert generation_time <= test_case['expected_max_time'], \
                            f"{test_case['name']} took {generation_time:.1f}s, expected max {test_case['expected_max_time']}s"
                    else:
                        performance_results[test_case['name']] = {'success': False, 'error': 'Generation failed'}
                else:
                    performance_results[test_case['name']] = {'success': False, 'error': f'HTTP {response.status_code}'}
                    
            except requests.Timeout:
                performance_results[test_case['name']] = {'success': False, 'error': 'Timeout'}
            except Exception as e:
                performance_results[test_case['name']] = {'success': False, 'error': str(e)}
        
        # Print performance summary
        print("\nPerformance Results:")
        for name, result in performance_results.items():
            if result['success']:
                print(f"  {name}: {result['total_time']:.1f}s (expected max: {result['expected_max']}s)")
            else:
                print(f"  {name}: FAILED - {result['error']}")
        
        # Verify at least some tests passed
        successful_tests = sum(1 for r in performance_results.values() if r['success'])
        assert successful_tests > 0, "No performance tests passed"
        
        return performance_results
    
    def test_concurrent_request_performance(self):
        """Test performance under concurrent requests"""
        if not self.docker_client:
            pytest.skip("Docker not available")
        
        # Start compose stack
        self.start_compose_stack_for_testing()
        time.sleep(30)
        
        # Test concurrent requests
        def make_concurrent_request(request_id: int) -> Tuple[int, bool, float]:
            start_time = time.time()
            
            try:
                response = requests.post(
                    "http://localhost:8080/api/generate/text-to-image",
                    json={
                        "prompt": f"Test image {request_id}",
                        "width": 256,
                        "height": 256,
                        "num_inference_steps": 8
                    },
                    timeout=60
                )
                
                duration = time.time() - start_time
                success = response.status_code == 200 and response.json().get('success', False)
                
                return request_id, success, duration
                
            except Exception as e:
                duration = time.time() - start_time
                return request_id, False, duration
        
        # Run concurrent requests
        num_concurrent = 3
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_concurrent_request, i) for i in range(num_concurrent)]
            results = [future.result() for future in as_completed(futures)]
        
        # Analyze results
        successful_requests = sum(1 for _, success, _ in results if success)
        total_requests = len(results)
        success_rate = successful_requests / total_requests
        
        response_times = [duration for _, success, duration in results if success]
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
        else:
            avg_response_time = max_response_time = min_response_time = 0
        
        load_test_result = LoadTestResult(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=total_requests - successful_requests,
            average_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            requests_per_second=successful_requests / max(max_response_time, 1),
            error_rate=(total_requests - successful_requests) / total_requests,
            peak_memory_usage=0.0,  # Would need monitoring data
            peak_cpu_usage=0.0      # Would need monitoring data
        )
        
        print(f"\nConcurrent Request Performance:")
        print(f"  Total requests: {load_test_result.total_requests}")
        print(f"  Successful: {load_test_result.successful_requests}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Average response time: {load_test_result.average_response_time:.1f}s")
        print(f"  Response time range: {load_test_result.min_response_time:.1f}s - {load_test_result.max_response_time:.1f}s")
        
        # Verify performance criteria
        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.1%}"
        assert load_test_result.average_response_time < 90, f"Average response time too high: {avg_response_time:.1f}s"
        
        return load_test_result


if __name__ == "__main__":
    # Run Docker performance validation tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--maxfail=3",
        "-s"
    ])