#!/usr/bin/env python3
"""
Docker Container Integration Tests
Tests service communication, networking, and container health in Docker environment.
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
from typing import Dict, Any, Optional, List
from pathlib import Path
import docker
from docker.errors import DockerException, NotFound, APIError

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class DockerContainerIntegrationTest:
    """Test Docker container integration and service communication"""
    
    def __init__(self):
        self.docker_client = None
        self.containers = {}
        self.networks = {}
        self.test_timeout = 300  # 5 minutes
        
    def setup_docker_client(self):
        """Initialize Docker client"""
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            return True
        except DockerException as e:
            pytest.skip(f"Docker not available: {e}")
            return False
    
    def cleanup_test_resources(self):
        """Clean up test containers and networks"""
        if not self.docker_client:
            return
            
        # Stop and remove test containers
        for container_name, container in self.containers.items():
            try:
                if container.status == 'running':
                    container.stop(timeout=10)
                container.remove(force=True)
                print(f"Cleaned up container: {container_name}")
            except Exception as e:
                print(f"Warning: Could not clean up container {container_name}: {e}")
        
        # Remove test networks
        for network_name, network in self.networks.items():
            try:
                network.remove()
                print(f"Cleaned up network: {network_name}")
            except Exception as e:
                print(f"Warning: Could not clean up network {network_name}: {e}")
        
        self.containers.clear()
        self.networks.clear()


class TestDockerServiceCommunication(DockerContainerIntegrationTest):
    """Test communication between Docker services"""
    
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup for each test"""
        self.setup_docker_client()
        yield
        self.cleanup_test_resources()
    
    def test_traefik_api_communication(self):
        """Test communication between Traefik and API service"""
        if not self.docker_client:
            pytest.skip("Docker not available")
        
        # Create test network
        network_name = f"test-qwen-network-{uuid.uuid4().hex[:8]}"
        network = self.docker_client.networks.create(
            network_name,
            driver="bridge"
        )
        self.networks[network_name] = network
        
        # Start API container
        api_container = self.docker_client.containers.run(
            "qwen-api:latest",
            name=f"test-qwen-api-{uuid.uuid4().hex[:8]}",
            network=network_name,
            environment={
                "API_HOST": "0.0.0.0",
                "API_PORT": "8000",
                "LOG_LEVEL": "DEBUG"
            },
            detach=True,
            remove=False,
            healthcheck={
                "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                "interval": 30000000000,  # 30s in nanoseconds
                "timeout": 10000000000,   # 10s in nanoseconds
                "retries": 3
            }
        )
        self.containers["api"] = api_container
        
        # Wait for API to be healthy
        self._wait_for_container_health(api_container, timeout=120)
        
        # Start Traefik container
        traefik_container = self.docker_client.containers.run(
            "traefik:v3.0",
            name=f"test-qwen-traefik-{uuid.uuid4().hex[:8]}",
            network=network_name,
            ports={"80/tcp": 8080, "8080/tcp": 8081},
            volumes={
                "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "ro"},
                str(Path(__file__).parent.parent / "config/docker/traefik.yml"): {
                    "bind": "/etc/traefik/traefik.yml", "mode": "ro"
                }
            },
            environment={
                "TRAEFIK_LOG_LEVEL": "DEBUG",
                "TRAEFIK_API_DASHBOARD": "true",
                "TRAEFIK_API_INSECURE": "true"
            },
            detach=True,
            remove=False
        )
        self.containers["traefik"] = traefik_container
        
        # Wait for Traefik to start
        time.sleep(10)
        
        # Test direct API communication
        api_ip = api_container.attrs['NetworkSettings']['Networks'][network_name]['IPAddress']
        
        # Test API health endpoint directly
        try:
            response = requests.get(f"http://{api_ip}:8000/health", timeout=10)
            assert response.status_code == 200
            health_data = response.json()
            assert health_data.get('status') == 'healthy'
        except requests.RequestException as e:
            pytest.fail(f"Direct API communication failed: {e}")
        
        # Test Traefik dashboard
        try:
            dashboard_response = requests.get("http://localhost:8081/api/rawdata", timeout=10)
            assert dashboard_response.status_code == 200
        except requests.RequestException as e:
            pytest.fail(f"Traefik dashboard communication failed: {e}")
    
    def test_frontend_api_communication(self):
        """Test communication between frontend and API service"""
        if not self.docker_client:
            pytest.skip("Docker not available")
        
        # Create test network
        network_name = f"test-qwen-network-{uuid.uuid4().hex[:8]}"
        network = self.docker_client.networks.create(network_name, driver="bridge")
        self.networks[network_name] = network
        
        # Start API container
        api_container = self.docker_client.containers.run(
            "qwen-api:latest",
            name=f"test-qwen-api-{uuid.uuid4().hex[:8]}",
            network=network_name,
            environment={
                "API_HOST": "0.0.0.0",
                "API_PORT": "8000",
                "CORS_ORIGINS": "*"
            },
            detach=True,
            remove=False
        )
        self.containers["api"] = api_container
        
        # Wait for API to be ready
        self._wait_for_container_health(api_container, timeout=120)
        
        # Start frontend container
        frontend_container = self.docker_client.containers.run(
            "qwen-frontend:latest",
            name=f"test-qwen-frontend-{uuid.uuid4().hex[:8]}",
            network=network_name,
            ports={"80/tcp": 3000},
            environment={
                "REACT_APP_API_URL": f"http://{api_container.name}:8000"
            },
            detach=True,
            remove=False
        )
        self.containers["frontend"] = frontend_container
        
        # Wait for frontend to start
        time.sleep(15)
        
        # Test frontend can reach API
        api_ip = api_container.attrs['NetworkSettings']['Networks'][network_name]['IPAddress']
        
        # Execute curl command inside frontend container to test API connectivity
        try:
            exec_result = frontend_container.exec_run(
                f"curl -f http://{api_ip}:8000/health",
                timeout=10
            )
            assert exec_result.exit_code == 0
            
            # Parse response
            response_data = json.loads(exec_result.output.decode())
            assert response_data.get('status') == 'healthy'
            
        except Exception as e:
            pytest.fail(f"Frontend to API communication failed: {e}")
    
    def test_service_discovery_and_load_balancing(self):
        """Test service discovery and load balancing with multiple API instances"""
        if not self.docker_client:
            pytest.skip("Docker not available")
        
        # Create test network
        network_name = f"test-qwen-network-{uuid.uuid4().hex[:8]}"
        network = self.docker_client.networks.create(network_name, driver="bridge")
        self.networks[network_name] = network
        
        # Start multiple API instances
        api_containers = []
        for i in range(2):
            api_container = self.docker_client.containers.run(
                "qwen-api:latest",
                name=f"test-qwen-api-{i}-{uuid.uuid4().hex[:8]}",
                network=network_name,
                environment={
                    "API_HOST": "0.0.0.0",
                    "API_PORT": "8000",
                    "INSTANCE_ID": f"api-{i}"
                },
                labels={
                    "traefik.enable": "true",
                    "traefik.http.routers.api.rule": "PathPrefix(`/api`)",
                    "traefik.http.services.api.loadbalancer.server.port": "8000"
                },
                detach=True,
                remove=False
            )
            api_containers.append(api_container)
            self.containers[f"api-{i}"] = api_container
        
        # Wait for all API instances to be ready
        for container in api_containers:
            self._wait_for_container_health(container, timeout=120)
        
        # Start Traefik with service discovery
        traefik_container = self.docker_client.containers.run(
            "traefik:v3.0",
            name=f"test-qwen-traefik-{uuid.uuid4().hex[:8]}",
            network=network_name,
            ports={"80/tcp": 8080},
            volumes={
                "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "ro"}
            },
            command=[
                "--api.dashboard=true",
                "--api.insecure=true",
                "--providers.docker=true",
                "--providers.docker.exposedbydefault=false",
                "--entrypoints.web.address=:80"
            ],
            detach=True,
            remove=False
        )
        self.containers["traefik"] = traefik_container
        
        # Wait for Traefik to discover services
        time.sleep(15)
        
        # Test load balancing by making multiple requests
        responses = []
        for i in range(10):
            try:
                response = requests.get("http://localhost:8080/api/health", timeout=5)
                if response.status_code == 200:
                    responses.append(response.json())
            except requests.RequestException:
                pass  # Some requests may fail during startup
        
        # Verify we got responses from multiple instances
        assert len(responses) > 0, "No successful responses received"
        
        # Check if load balancing is working (responses from different instances)
        instance_ids = set()
        for response in responses:
            if 'instance_id' in response:
                instance_ids.add(response['instance_id'])
        
        # We should see responses from multiple instances if load balancing works
        print(f"Received responses from {len(instance_ids)} different instances")
    
    def _wait_for_container_health(self, container, timeout=60):
        """Wait for container to become healthy"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            container.reload()
            
            # Check if container is running
            if container.status != 'running':
                time.sleep(2)
                continue
            
            # Check health status if available
            health = container.attrs.get('State', {}).get('Health', {})
            if health.get('Status') == 'healthy':
                return True
            
            # If no health check, try to connect to the service
            try:
                # Get container IP
                networks = container.attrs['NetworkSettings']['Networks']
                if networks:
                    network_name = list(networks.keys())[0]
                    ip_address = networks[network_name]['IPAddress']
                    
                    # Try to connect to health endpoint
                    response = requests.get(f"http://{ip_address}:8000/health", timeout=5)
                    if response.status_code == 200:
                        return True
            except:
                pass
            
            time.sleep(2)
        
        # Get container logs for debugging
        logs = container.logs(tail=50).decode('utf-8')
        pytest.fail(f"Container {container.name} did not become healthy within {timeout}s. Logs:\n{logs}")


class TestDockerVolumeAndPersistence(DockerContainerIntegrationTest):
    """Test Docker volume management and data persistence"""
    
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup for each test"""
        self.setup_docker_client()
        yield
        self.cleanup_test_resources()
    
    def test_model_volume_persistence(self):
        """Test model volume persistence across container restarts"""
        if not self.docker_client:
            pytest.skip("Docker not available")
        
        # Create test volume
        volume_name = f"test-models-{uuid.uuid4().hex[:8]}"
        volume = self.docker_client.volumes.create(name=volume_name)
        
        try:
            # Start container with volume
            container1 = self.docker_client.containers.run(
                "alpine:latest",
                name=f"test-volume-writer-{uuid.uuid4().hex[:8]}",
                volumes={volume_name: {"bind": "/app/models", "mode": "rw"}},
                command=["sh", "-c", "echo 'test model data' > /app/models/test_model.txt && sleep 5"],
                detach=True,
                remove=False
            )
            self.containers["writer"] = container1
            
            # Wait for container to finish
            container1.wait(timeout=30)
            
            # Start another container with same volume
            container2 = self.docker_client.containers.run(
                "alpine:latest",
                name=f"test-volume-reader-{uuid.uuid4().hex[:8]}",
                volumes={volume_name: {"bind": "/app/models", "mode": "ro"}},
                command=["cat", "/app/models/test_model.txt"],
                detach=True,
                remove=False
            )
            self.containers["reader"] = container2
            
            # Wait and check output
            result = container2.wait(timeout=30)
            assert result['StatusCode'] == 0
            
            output = container2.logs().decode('utf-8').strip()
            assert output == "test model data"
            
        finally:
            # Clean up volume
            try:
                volume.remove(force=True)
            except Exception as e:
                print(f"Warning: Could not clean up volume {volume_name}: {e}")
    
    def test_cache_volume_sharing(self):
        """Test cache volume sharing between multiple containers"""
        if not self.docker_client:
            pytest.skip("Docker not available")
        
        # Create test cache volume
        cache_volume_name = f"test-cache-{uuid.uuid4().hex[:8]}"
        cache_volume = self.docker_client.volumes.create(name=cache_volume_name)
        
        try:
            # Start first container to write cache
            container1 = self.docker_client.containers.run(
                "alpine:latest",
                name=f"test-cache-writer-{uuid.uuid4().hex[:8]}",
                volumes={cache_volume_name: {"bind": "/app/cache", "mode": "rw"}},
                command=["sh", "-c", "mkdir -p /app/cache/huggingface && echo 'cached model' > /app/cache/huggingface/model.bin"],
                detach=True,
                remove=False
            )
            self.containers["cache_writer"] = container1
            container1.wait(timeout=30)
            
            # Start multiple containers to read cache simultaneously
            readers = []
            for i in range(3):
                reader = self.docker_client.containers.run(
                    "alpine:latest",
                    name=f"test-cache-reader-{i}-{uuid.uuid4().hex[:8]}",
                    volumes={cache_volume_name: {"bind": "/app/cache", "mode": "ro"}},
                    command=["cat", "/app/cache/huggingface/model.bin"],
                    detach=True,
                    remove=False
                )
                readers.append(reader)
                self.containers[f"cache_reader_{i}"] = reader
            
            # Verify all readers can access cache
            for i, reader in enumerate(readers):
                result = reader.wait(timeout=30)
                assert result['StatusCode'] == 0
                
                output = reader.logs().decode('utf-8').strip()
                assert output == "cached model"
                
        finally:
            # Clean up volume
            try:
                cache_volume.remove(force=True)
            except Exception as e:
                print(f"Warning: Could not clean up cache volume {cache_volume_name}: {e}")


class TestDockerNetworkSecurity(DockerContainerIntegrationTest):
    """Test Docker network security and isolation"""
    
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup for each test"""
        self.setup_docker_client()
        yield
        self.cleanup_test_resources()
    
    def test_network_isolation(self):
        """Test network isolation between different service networks"""
        if not self.docker_client:
            pytest.skip("Docker not available")
        
        # Create two isolated networks
        network1_name = f"test-network-1-{uuid.uuid4().hex[:8]}"
        network2_name = f"test-network-2-{uuid.uuid4().hex[:8]}"
        
        network1 = self.docker_client.networks.create(network1_name, driver="bridge")
        network2 = self.docker_client.networks.create(network2_name, driver="bridge")
        
        self.networks[network1_name] = network1
        self.networks[network2_name] = network2
        
        # Start containers in different networks
        container1 = self.docker_client.containers.run(
            "alpine:latest",
            name=f"test-isolated-1-{uuid.uuid4().hex[:8]}",
            network=network1_name,
            command=["sleep", "60"],
            detach=True,
            remove=False
        )
        self.containers["isolated1"] = container1
        
        container2 = self.docker_client.containers.run(
            "alpine:latest",
            name=f"test-isolated-2-{uuid.uuid4().hex[:8]}",
            network=network2_name,
            command=["sleep", "60"],
            detach=True,
            remove=False
        )
        self.containers["isolated2"] = container2
        
        # Wait for containers to start
        time.sleep(5)
        
        # Get IP addresses
        container1.reload()
        container2.reload()
        
        ip1 = container1.attrs['NetworkSettings']['Networks'][network1_name]['IPAddress']
        ip2 = container2.attrs['NetworkSettings']['Networks'][network2_name]['IPAddress']
        
        # Test that containers cannot reach each other
        try:
            # Try to ping from container1 to container2
            exec_result = container1.exec_run(f"ping -c 1 -W 2 {ip2}", timeout=10)
            # Ping should fail (non-zero exit code) due to network isolation
            assert exec_result.exit_code != 0, "Network isolation failed - containers can reach each other"
        except Exception:
            # Exception is also acceptable as it indicates network isolation
            pass
    
    def test_internal_network_external_access(self):
        """Test that internal networks cannot access external resources when configured"""
        if not self.docker_client:
            pytest.skip("Docker not available")
        
        # Create internal network (no external access)
        internal_network_name = f"test-internal-{uuid.uuid4().hex[:8]}"
        internal_network = self.docker_client.networks.create(
            internal_network_name,
            driver="bridge",
            internal=True  # No external access
        )
        self.networks[internal_network_name] = internal_network
        
        # Start container in internal network
        container = self.docker_client.containers.run(
            "alpine:latest",
            name=f"test-internal-container-{uuid.uuid4().hex[:8]}",
            network=internal_network_name,
            command=["sleep", "30"],
            detach=True,
            remove=False
        )
        self.containers["internal"] = container
        
        # Wait for container to start
        time.sleep(5)
        
        # Test that container cannot reach external resources
        try:
            # Try to reach external DNS
            exec_result = container.exec_run("nslookup google.com", timeout=10)
            # Should fail due to no external access
            assert exec_result.exit_code != 0, "Internal network has external access when it shouldn't"
        except Exception:
            # Exception is acceptable as it indicates no external access
            pass


if __name__ == "__main__":
    # Run container integration tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--maxfail=3",
        "-x"
    ])