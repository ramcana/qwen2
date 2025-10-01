#!/usr/bin/env python3
"""
Frontend-Backend Integration Test Suite

This test suite validates complete frontend-backend integration including:
- Container communication
- API endpoint connectivity
- CORS configuration
- Health checks
- Data flow validation
"""

import asyncio
import json
import os
import sys
import time
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import docker
import pytest

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class FrontendBackendIntegrationTester:
    """Comprehensive frontend-backend integration tester"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.test_results = {
            "container_status": {},
            "api_connectivity": {},
            "cors_validation": {},
            "health_checks": {},
            "data_flow": {},
            "performance": {},
            "errors": []
        }
        
        # Configuration
        self.backend_container_name = "qwen-api"
        self.frontend_container_name = "qwen-frontend"
        self.traefik_container_name = "qwen-traefik"
        
        # API endpoints to test
        self.api_endpoints = [
            "/health",
            "/health/detailed",
            "/health/ready",
            "/health/live",
            "/status",
            "/aspect-ratios",
            "/queue",
            "/memory/status"
        ]
        
        # Frontend URLs to test
        self.frontend_urls = [
            "/",
            "/health",
            "/api/health",
            "/api/status"
        ]
        
    def log_result(self, category: str, test_name: str, success: bool, 
                   message: str, details: Optional[Dict] = None):
        """Log test result"""
        result = {
            "success": success,
            "message": message,
            "timestamp": time.time(),
            "details": details or {}
        }
        
        if category not in self.test_results:
            self.test_results[category] = {}
        
        self.test_results[category][test_name] = result
        
        status = "✅" if success else "❌"
        print(f"{status} {category}.{test_name}: {message}")
        
        if not success:
            self.test_results["errors"].append({
                "category": category,
                "test": test_name,
                "message": message,
                "details": details
            })
    
    def check_container_status(self) -> bool:
        """Check if all required containers are running"""
        print("\n🔍 Checking container status...")
        
        required_containers = [
            self.backend_container_name,
            self.frontend_container_name,
            self.traefik_container_name
        ]
        
        all_running = True
        
        for container_name in required_containers:
            try:
                container = self.docker_client.containers.get(container_name)
                is_running = container.status == "running"
                
                self.log_result(
                    "container_status", 
                    container_name,
                    is_running,
                    f"Container {container.status}",
                    {
                        "id": container.id[:12],
                        "image": container.image.tags[0] if container.image.tags else "unknown",
                        "status": container.status,
                        "ports": container.ports
                    }
                )
                
                if not is_running:
                    all_running = False
                    
            except docker.errors.NotFound:
                self.log_result(
                    "container_status",
                    container_name,
                    False,
                    "Container not found"
                )
                all_running = False
            except Exception as e:
                self.log_result(
                    "container_status",
                    container_name,
                    False,
                    f"Error checking container: {str(e)}"
                )
                all_running = False
        
        return all_running
    
    def test_backend_api_connectivity(self) -> bool:
        """Test direct backend API connectivity"""
        print("\n🔍 Testing backend API connectivity...")
        
        # Try different backend URLs
        backend_urls = [
            "http://localhost:8000",  # Direct access
            "http://qwen-api:8000",   # Container name
            "http://127.0.0.1:8000"   # Localhost IP
        ]
        
        backend_accessible = False
        
        for base_url in backend_urls:
            try:
                response = requests.get(f"{base_url}/health", timeout=10)
                if response.status_code == 200:
                    backend_accessible = True
                    self.log_result(
                        "api_connectivity",
                        f"backend_direct_{base_url.replace(':', '_').replace('/', '_')}",
                        True,
                        f"Backend accessible at {base_url}",
                        {"status_code": response.status_code, "response": response.json()}
                    )
                    break
                else:
                    self.log_result(
                        "api_connectivity",
                        f"backend_direct_{base_url.replace(':', '_').replace('/', '_')}",
                        False,
                        f"Backend returned status {response.status_code}"
                    )
            except requests.exceptions.RequestException as e:
                self.log_result(
                    "api_connectivity",
                    f"backend_direct_{base_url.replace(':', '_').replace('/', '_')}",
                    False,
                    f"Connection failed: {str(e)}"
                )
        
        # Test individual API endpoints
        if backend_accessible:
            working_backend_url = None
            for url in backend_urls:
                try:
                    response = requests.get(f"{url}/health", timeout=5)
                    if response.status_code == 200:
                        working_backend_url = url
                        break
                except:
                    continue
            
            if working_backend_url:
                for endpoint in self.api_endpoints:
                    try:
                        response = requests.get(f"{working_backend_url}{endpoint}", timeout=10)
                        success = response.status_code in [200, 201, 202]
                        
                        self.log_result(
                            "api_connectivity",
                            f"endpoint_{endpoint.replace('/', '_')}",
                            success,
                            f"Endpoint {endpoint}: {response.status_code}",
                            {
                                "status_code": response.status_code,
                                "response_size": len(response.content),
                                "content_type": response.headers.get("content-type")
                            }
                        )
                    except Exception as e:
                        self.log_result(
                            "api_connectivity",
                            f"endpoint_{endpoint.replace('/', '_')}",
                            False,
                            f"Endpoint {endpoint} failed: {str(e)}"
                        )
        
        return backend_accessible
    
    def test_frontend_connectivity(self) -> bool:
        """Test frontend container connectivity"""
        print("\n🔍 Testing frontend connectivity...")
        
        # Try different frontend URLs
        frontend_urls = [
            "http://localhost:80",     # Direct nginx access
            "http://localhost",        # Default port
            "http://qwen-frontend:80", # Container name
            "http://127.0.0.1:80"      # Localhost IP
        ]
        
        frontend_accessible = False
        
        for base_url in frontend_urls:
            try:
                response = requests.get(f"{base_url}/health", timeout=10)
                if response.status_code == 200:
                    frontend_accessible = True
                    self.log_result(
                        "api_connectivity",
                        f"frontend_direct_{base_url.replace(':', '_').replace('/', '_')}",
                        True,
                        f"Frontend accessible at {base_url}",
                        {"status_code": response.status_code}
                    )
                    break
                else:
                    self.log_result(
                        "api_connectivity",
                        f"frontend_direct_{base_url.replace(':', '_').replace('/', '_')}",
                        False,
                        f"Frontend returned status {response.status_code}"
                    )
            except requests.exceptions.RequestException as e:
                self.log_result(
                    "api_connectivity",
                    f"frontend_direct_{base_url.replace(':', '_').replace('/', '_')}",
                    False,
                    f"Connection failed: {str(e)}"
                )
        
        return frontend_accessible
    
    def test_traefik_routing(self) -> bool:
        """Test Traefik reverse proxy routing"""
        print("\n🔍 Testing Traefik routing...")
        
        # Test Traefik dashboard
        try:
            response = requests.get("http://localhost:9090/api/rawdata", timeout=10)
            traefik_working = response.status_code == 200
            
            self.log_result(
                "api_connectivity",
                "traefik_dashboard",
                traefik_working,
                f"Traefik dashboard: {response.status_code}",
                {"status_code": response.status_code}
            )
        except Exception as e:
            self.log_result(
                "api_connectivity",
                "traefik_dashboard",
                False,
                f"Traefik dashboard failed: {str(e)}"
            )
            traefik_working = False
        
        # Test routing through Traefik
        traefik_routes = [
            ("http://localhost/health", "frontend_health"),
            ("http://localhost/api/health", "api_health"),
            ("http://qwen.localhost/health", "qwen_frontend_health"),
            ("http://api.localhost/health", "qwen_api_health")
        ]
        
        for url, test_name in traefik_routes:
            try:
                response = requests.get(url, timeout=10, headers={"Host": url.split("//")[1].split("/")[0]})
                success = response.status_code in [200, 201, 202]
                
                self.log_result(
                    "api_connectivity",
                    f"traefik_{test_name}",
                    success,
                    f"Traefik route {url}: {response.status_code}",
                    {"status_code": response.status_code}
                )
            except Exception as e:
                self.log_result(
                    "api_connectivity",
                    f"traefik_{test_name}",
                    False,
                    f"Traefik route {url} failed: {str(e)}"
                )
        
        return traefik_working
    
    def test_cors_configuration(self) -> bool:
        """Test CORS configuration for frontend-backend communication"""
        print("\n🔍 Testing CORS configuration...")
        
        # Test CORS preflight requests
        cors_tests = [
            {
                "url": "http://localhost/api/health",
                "origin": "http://localhost:3000",
                "method": "GET"
            },
            {
                "url": "http://localhost/api/status",
                "origin": "http://localhost",
                "method": "GET"
            }
        ]
        
        cors_working = True
        
        for test in cors_tests:
            try:
                # OPTIONS preflight request
                headers = {
                    "Origin": test["origin"],
                    "Access-Control-Request-Method": test["method"],
                    "Access-Control-Request-Headers": "Content-Type"
                }
                
                response = requests.options(test["url"], headers=headers, timeout=10)
                
                # Check CORS headers
                cors_headers = {
                    "access-control-allow-origin": response.headers.get("access-control-allow-origin"),
                    "access-control-allow-methods": response.headers.get("access-control-allow-methods"),
                    "access-control-allow-headers": response.headers.get("access-control-allow-headers"),
                    "access-control-allow-credentials": response.headers.get("access-control-allow-credentials")
                }
                
                cors_valid = (
                    response.status_code in [200, 204] and
                    cors_headers["access-control-allow-origin"] is not None
                )
                
                self.log_result(
                    "cors_validation",
                    f"preflight_{test['url'].replace('/', '_').replace(':', '_')}",
                    cors_valid,
                    f"CORS preflight for {test['url']}: {response.status_code}",
                    {
                        "status_code": response.status_code,
                        "cors_headers": cors_headers
                    }
                )
                
                if not cors_valid:
                    cors_working = False
                    
            except Exception as e:
                self.log_result(
                    "cors_validation",
                    f"preflight_{test['url'].replace('/', '_').replace(':', '_')}",
                    False,
                    f"CORS preflight failed: {str(e)}"
                )
                cors_working = False
        
        return cors_working
    
    def test_health_checks(self) -> bool:
        """Test comprehensive health check endpoints"""
        print("\n🔍 Testing health check endpoints...")
        
        health_endpoints = [
            ("http://localhost/health", "frontend_health"),
            ("http://localhost/api/health", "backend_health"),
            ("http://localhost/api/health/detailed", "backend_detailed_health"),
            ("http://localhost/api/health/ready", "backend_readiness"),
            ("http://localhost/api/health/live", "backend_liveness")
        ]
        
        all_healthy = True
        
        for url, test_name in health_endpoints:
            try:
                response = requests.get(url, timeout=15)
                is_healthy = response.status_code == 200
                
                response_data = {}
                try:
                    response_data = response.json()
                except:
                    response_data = {"text": response.text[:200]}
                
                self.log_result(
                    "health_checks",
                    test_name,
                    is_healthy,
                    f"Health check {url}: {response.status_code}",
                    {
                        "status_code": response.status_code,
                        "response": response_data
                    }
                )
                
                if not is_healthy:
                    all_healthy = False
                    
            except Exception as e:
                self.log_result(
                    "health_checks",
                    test_name,
                    False,
                    f"Health check {url} failed: {str(e)}"
                )
                all_healthy = False
        
        return all_healthy
    
    def test_data_flow(self) -> bool:
        """Test complete data flow from frontend to backend"""
        print("\n🔍 Testing data flow...")
        
        # Test API status endpoint through frontend proxy
        try:
            response = requests.get("http://localhost/api/status", timeout=15)
            if response.status_code == 200:
                status_data = response.json()
                
                # Validate status response structure
                required_fields = ["model_loaded", "device", "memory_info", "queue_length"]
                has_required_fields = all(field in status_data for field in required_fields)
                
                self.log_result(
                    "data_flow",
                    "status_api_structure",
                    has_required_fields,
                    f"Status API structure validation: {'passed' if has_required_fields else 'failed'}",
                    {
                        "response_fields": list(status_data.keys()),
                        "required_fields": required_fields,
                        "missing_fields": [f for f in required_fields if f not in status_data]
                    }
                )
                
                # Test memory info structure
                memory_info = status_data.get("memory_info", {})
                memory_valid = isinstance(memory_info, dict) and len(memory_info) > 0
                
                self.log_result(
                    "data_flow",
                    "memory_info_structure",
                    memory_valid,
                    f"Memory info structure: {'valid' if memory_valid else 'invalid'}",
                    {"memory_info": memory_info}
                )
                
                return has_required_fields and memory_valid
            else:
                self.log_result(
                    "data_flow",
                    "status_api_access",
                    False,
                    f"Status API returned {response.status_code}"
                )
                return False
                
        except Exception as e:
            self.log_result(
                "data_flow",
                "status_api_access",
                False,
                f"Status API failed: {str(e)}"
            )
            return False
    
    def test_performance_metrics(self) -> bool:
        """Test performance metrics and response times"""
        print("\n🔍 Testing performance metrics...")
        
        performance_tests = [
            ("http://localhost/health", "frontend_health_response_time"),
            ("http://localhost/api/health", "backend_health_response_time"),
            ("http://localhost/api/status", "backend_status_response_time")
        ]
        
        all_performant = True
        
        for url, test_name in performance_tests:
            try:
                start_time = time.time()
                response = requests.get(url, timeout=10)
                response_time = time.time() - start_time
                
                # Performance thresholds
                is_fast = response_time < 5.0  # 5 seconds max
                is_responsive = response_time < 2.0  # 2 seconds preferred
                
                self.log_result(
                    "performance",
                    test_name,
                    is_fast,
                    f"Response time: {response_time:.2f}s ({'fast' if is_responsive else 'acceptable' if is_fast else 'slow'})",
                    {
                        "response_time_seconds": response_time,
                        "status_code": response.status_code,
                        "is_fast": is_fast,
                        "is_responsive": is_responsive
                    }
                )
                
                if not is_fast:
                    all_performant = False
                    
            except Exception as e:
                self.log_result(
                    "performance",
                    test_name,
                    False,
                    f"Performance test failed: {str(e)}"
                )
                all_performant = False
        
        return all_performant
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive integration test suite"""
        print("🚀 Starting Frontend-Backend Integration Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all test categories
        test_results = {
            "containers_running": self.check_container_status(),
            "backend_connectivity": self.test_backend_api_connectivity(),
            "frontend_connectivity": self.test_frontend_connectivity(),
            "traefik_routing": self.test_traefik_routing(),
            "cors_configuration": self.test_cors_configuration(),
            "health_checks": self.test_health_checks(),
            "data_flow": self.test_data_flow(),
            "performance": self.test_performance_metrics()
        }
        
        total_time = time.time() - start_time
        
        # Calculate overall success
        overall_success = all(test_results.values())
        
        # Generate summary
        print("\n" + "=" * 60)
        print("🏁 Integration Test Summary")
        print("=" * 60)
        
        for test_name, success in test_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {test_name.replace('_', ' ').title()}")
        
        print(f"\n⏱️  Total test time: {total_time:.2f} seconds")
        print(f"🎯 Overall result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        
        if self.test_results["errors"]:
            print(f"\n❌ {len(self.test_results['errors'])} errors found:")
            for error in self.test_results["errors"][:5]:  # Show first 5 errors
                print(f"   • {error['category']}.{error['test']}: {error['message']}")
            if len(self.test_results["errors"]) > 5:
                print(f"   ... and {len(self.test_results['errors']) - 5} more errors")
        
        # Save detailed results
        self.test_results["summary"] = {
            "overall_success": overall_success,
            "total_time_seconds": total_time,
            "test_categories": test_results,
            "timestamp": time.time()
        }
        
        return self.test_results
    
    def save_results(self, filename: str = "frontend_backend_integration_test_results.json"):
        """Save test results to file"""
        results_path = Path(__file__).parent / filename
        with open(results_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {results_path}")


def main():
    """Main test execution"""
    tester = FrontendBackendIntegrationTester()
    
    try:
        results = tester.run_comprehensive_test()
        tester.save_results()
        
        # Exit with appropriate code
        overall_success = results["summary"]["overall_success"]
        sys.exit(0 if overall_success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()