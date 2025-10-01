#!/usr/bin/env python3
"""
Simple API Connectivity Test

Tests basic frontend-backend connectivity without Docker dependencies.
This can be run to verify the integration is working.
"""

import requests
import time
import json
import sys
from typing import Dict, List, Tuple

class APIConnectivityTester:
    """Simple API connectivity tester"""
    
    def __init__(self):
        self.results = []
        
    def test_endpoint(self, url: str, description: str, timeout: int = 10) -> bool:
        """Test a single endpoint"""
        try:
            start_time = time.time()
            response = requests.get(url, timeout=timeout)
            response_time = time.time() - start_time
            
            success = response.status_code in [200, 201, 202]
            
            result = {
                "url": url,
                "description": description,
                "success": success,
                "status_code": response.status_code,
                "response_time": response_time,
                "error": None
            }
            
            if success:
                try:
                    result["response_data"] = response.json()
                except:
                    result["response_data"] = response.text[:200]
            
            self.results.append(result)
            
            status = "✅" if success else "❌"
            print(f"{status} {description}: {response.status_code} ({response_time:.2f}s)")
            
            return success
            
        except requests.exceptions.RequestException as e:
            result = {
                "url": url,
                "description": description,
                "success": False,
                "status_code": None,
                "response_time": None,
                "error": str(e)
            }
            
            self.results.append(result)
            print(f"❌ {description}: Connection failed - {str(e)}")
            return False
    
    def run_tests(self) -> bool:
        """Run all connectivity tests"""
        print("🚀 Testing Frontend-Backend Integration")
        print("=" * 50)
        
        # Test endpoints
        tests = [
            # Frontend health checks
            ("http://localhost/health", "Frontend Health Check"),
            ("http://localhost:80/health", "Frontend Health Check (Port 80)"),
            
            # Backend health checks through proxy
            ("http://localhost/api/health", "Backend Health (via Proxy)"),
            ("http://localhost/api/health/live", "Backend Liveness (via Proxy)"),
            ("http://localhost/api/status", "Backend Status (via Proxy)"),
            
            # Direct backend access (if available)
            ("http://localhost:8000/health", "Backend Health (Direct)"),
            ("http://localhost:8000/status", "Backend Status (Direct)"),
            
            # Traefik dashboard
            ("http://localhost:9090/ping", "Traefik Health"),
        ]
        
        successful_tests = 0
        total_tests = len(tests)
        
        for url, description in tests:
            if self.test_endpoint(url, description):
                successful_tests += 1
        
        print("\n" + "=" * 50)
        print(f"🏁 Test Results: {successful_tests}/{total_tests} passed")
        
        # Test CORS if backend is accessible
        if any(r["success"] and "/api/" in r["url"] for r in self.results):
            print("\n🔍 Testing CORS Configuration...")
            self.test_cors()
        
        # Test data flow if API is accessible
        if any(r["success"] and "status" in r["url"] for r in self.results):
            print("\n🔍 Testing Data Flow...")
            self.test_data_flow()
        
        overall_success = successful_tests >= total_tests * 0.6  # 60% success rate
        
        print(f"\n🎯 Overall Result: {'✅ INTEGRATION WORKING' if overall_success else '❌ INTEGRATION ISSUES'}")
        
        return overall_success
    
    def test_cors(self):
        """Test CORS configuration"""
        cors_test_url = "http://localhost/api/health"
        
        try:
            headers = {
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type"
            }
            
            response = requests.options(cors_test_url, headers=headers, timeout=10)
            
            cors_headers = {
                "access-control-allow-origin": response.headers.get("access-control-allow-origin"),
                "access-control-allow-methods": response.headers.get("access-control-allow-methods"),
                "access-control-allow-credentials": response.headers.get("access-control-allow-credentials")
            }
            
            cors_working = (
                response.status_code in [200, 204] and
                cors_headers["access-control-allow-origin"] is not None
            )
            
            status = "✅" if cors_working else "❌"
            print(f"{status} CORS Configuration: {'Working' if cors_working else 'Issues detected'}")
            
            if cors_working:
                print(f"   • Origin: {cors_headers['access-control-allow-origin']}")
                print(f"   • Methods: {cors_headers['access-control-allow-methods']}")
                print(f"   • Credentials: {cors_headers['access-control-allow-credentials']}")
            
        except Exception as e:
            print(f"❌ CORS Test Failed: {str(e)}")
    
    def test_data_flow(self):
        """Test data flow through the API"""
        status_urls = [
            "http://localhost/api/status",
            "http://localhost:8000/status"
        ]
        
        for url in status_urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for expected fields
                    expected_fields = ["model_loaded", "device", "memory_info"]
                    has_fields = all(field in data for field in expected_fields)
                    
                    status = "✅" if has_fields else "⚠️"
                    print(f"{status} Data Flow ({url}): {'Complete' if has_fields else 'Partial'}")
                    
                    if has_fields:
                        print(f"   • Model Loaded: {data.get('model_loaded', 'unknown')}")
                        print(f"   • Device: {data.get('device', 'unknown')}")
                        print(f"   • Queue Length: {data.get('queue_length', 'unknown')}")
                    
                    break
                    
            except Exception as e:
                continue
        else:
            print("❌ Data Flow Test: No accessible status endpoint")
    
    def save_results(self, filename: str = "api_connectivity_results.json"):
        """Save results to file"""
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "results": self.results,
                "summary": {
                    "total_tests": len(self.results),
                    "successful_tests": sum(1 for r in self.results if r["success"]),
                    "failed_tests": sum(1 for r in self.results if not r["success"])
                }
            }, f, indent=2)
        print(f"\n💾 Results saved to: {filename}")


def main():
    """Main test execution"""
    tester = APIConnectivityTester()
    
    try:
        success = tester.run_tests()
        tester.save_results()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()