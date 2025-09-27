#!/usr/bin/env python3
"""
Comprehensive Integration and Performance Test Suite
Tests the complete FastAPI + React architecture with memory optimization
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

try:
    from ui_config_manager import UIConfigManager
except ImportError:
    print("⚠️ UI Config Manager not available")
    UIConfigManager = None

class IntegrationTestSuite:
    """Comprehensive integration test suite"""
    
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.react_url = "http://localhost:3001"
        self.gradio_url = "http://localhost:7860"
        self.test_results = {}
        self.performance_metrics = {}
        
    def run_all_tests(self):
        """Run complete test suite"""
        print("🧪 Qwen-Image Integration & Performance Test Suite")
        print("=" * 60)
        
        tests = [
            ("Service Health Check", self.test_service_health),
            ("API Endpoint Integration", self.test_api_integration),
            ("Memory Optimization", self.test_memory_optimization),
            ("Performance Validation", self.test_performance),
            ("Feature Flag System", self.test_feature_flags),
            ("Concurrent Request Handling", self.test_concurrent_requests),
            ("Error Recovery", self.test_error_recovery),
            ("React Frontend Integration", self.test_react_integration)
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running: {test_name}")
            print("-" * 40)
            
            try:
                start_time = time.time()
                result = test_func()
                duration = time.time() - start_time
                
                self.test_results[test_name] = {
                    "success": result,
                    "duration": duration
                }
                
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{status} {test_name} ({duration:.1f}s)")
                
            except Exception as e:
                print(f"❌ ERROR {test_name}: {e}")
                self.test_results[test_name] = {
                    "success": False,
                    "duration": 0,
                    "error": str(e)
                }
        
        self.print_summary()
        return self.calculate_overall_success()
    
    def test_service_health(self) -> bool:
        """Test health of all services"""
        services = [
            ("FastAPI", self.api_url + "/health"),
            ("React Dev Server", self.react_url),
            ("Gradio UI", self.gradio_url)
        ]
        
        results = []
        for name, url in services:
            try:
                response = requests.get(url, timeout=5)
                success = response.status_code == 200
                print(f"   {name}: {'✅' if success else '❌'} ({response.status_code})")
                results.append(success)
            except requests.exceptions.ConnectionError:
                print(f"   {name}: ❌ (Connection refused)")
                results.append(False)
            except Exception as e:
                print(f"   {name}: ❌ ({e})")
                results.append(False)
        
        # At least API should be running for integration tests
        return results[0]  # FastAPI is critical
    
    def test_api_integration(self) -> bool:
        """Test API endpoint integration"""
        endpoints = [
            ("GET", "/", "Root endpoint"),
            ("GET", "/status", "Status check"),
            ("GET", "/aspect-ratios", "Aspect ratios"),
            ("GET", "/queue", "Queue status"),
            ("GET", "/memory/clear", "Memory clear")
        ]
        
        success_count = 0
        for method, endpoint, description in endpoints:
            try:
                if method == "GET":
                    response = requests.get(self.api_url + endpoint, timeout=10)
                elif method == "POST":
                    response = requests.post(self.api_url + endpoint, timeout=10)
                
                success = response.status_code in [200, 201]
                print(f"   {description}: {'✅' if success else '❌'} ({response.status_code})")
                
                if success:
                    success_count += 1
                    
            except Exception as e:
                print(f"   {description}: ❌ ({e})")
        
        return success_count >= len(endpoints) * 0.8  # 80% success rate
    
    def test_memory_optimization(self) -> bool:
        """Test memory optimization features"""
        print("   Testing memory management...")
        
        try:
            # Clear memory
            response = requests.get(self.api_url + "/memory/clear")
            if response.status_code != 200:
                print("   ❌ Memory clear failed")
                return False
            
            clear_data = response.json()
            print("   ✅ Memory cleared successfully")
            
            # Check status for memory info
            response = requests.get(self.api_url + "/status")
            if response.status_code != 200:
                print("   ❌ Status check failed")
                return False
            
            status_data = response.json()
            memory_info = status_data.get('memory_info')
            
            if memory_info:
                print(f"   📊 VRAM: {memory_info['allocated_gb']}GB / {memory_info['total_gb']}GB")
                print(f"   📊 Usage: {memory_info['usage_percent']}%")
                
                # Memory should be reasonable after clear
                if memory_info['usage_percent'] < 90:
                    print("   ✅ Memory usage within acceptable range")
                    return True
                else:
                    print("   ⚠️ High memory usage detected")
                    return False
            else:
                print("   ⚠️ No memory info available (CPU mode?)")
                return True  # Not a failure if GPU not available
                
        except Exception as e:
            print(f"   ❌ Memory test error: {e}")
            return False
    
    def test_performance(self) -> bool:
        """Test performance characteristics"""
        print("   Running performance validation...")
        
        # Test small image generation for performance
        payload = {
            "prompt": "Performance test: a simple geometric shape",
            "width": 512,
            "height": 512,
            "num_inference_steps": 10,  # Fast for testing
            "cfg_scale": 3.0,
            "seed": 42,
            "language": "en",
            "enhance_prompt": False
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                self.api_url + "/generate/text-to-image",
                json=payload,
                timeout=60
            )
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    generation_time = data.get('generation_time', total_time)
                    print("   ✅ Generation successful")
                    print(f"   ⏱️  Generation time: {generation_time:.1f}s")
                    print(f"   ⏱️  Total request time: {total_time:.1f}s")
                    
                    # Store metrics
                    self.performance_metrics['generation_time'] = generation_time
                    self.performance_metrics['total_request_time'] = total_time
                    
                    # Performance thresholds
                    if generation_time < 120:  # 2 minutes max for small image
                        print("   ✅ Performance within acceptable range")
                        return True
                    else:
                        print("   ⚠️ Performance slower than expected")
                        return False
                else:
                    print(f"   ❌ Generation failed: {data.get('message')}")
                    return False
            else:
                print(f"   ❌ Request failed: {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            print("   ❌ Performance test timed out")
            return False
        except Exception as e:
            print(f"   ❌ Performance test error: {e}")
            return False
    
    def test_feature_flags(self) -> bool:
        """Test feature flag system"""
        if not UIConfigManager:
            print("   ⚠️ Feature flag system not available")
            return True
        
        try:
            config_manager = UIConfigManager()
            
            # Test basic operations
            current_mode = config_manager.get_ui_mode()
            print(f"   📄 Current UI mode: {current_mode}")
            
            # Test feature checks
            features = config_manager.get_feature_flags()
            print(f"   🚩 Active features: {sum(features.values())}/{len(features)}")
            
            for feature, enabled in features.items():
                status = "✅" if enabled else "❌"
                print(f"      {status} {feature.replace('_', ' ').title()}")
            
            print("   ✅ Feature flag system operational")
            return True
            
        except Exception as e:
            print(f"   ❌ Feature flag test error: {e}")
            return False
    
    def test_concurrent_requests(self) -> bool:
        """Test concurrent request handling"""
        print("   Testing concurrent request handling...")
        
        def make_request(request_id):
            payload = {
                "prompt": f"Concurrent test {request_id}: a simple shape",
                "width": 512,
                "height": 512,
                "num_inference_steps": 5,  # Very fast
                "cfg_scale": 2.0,
                "seed": request_id,
                "language": "en",
                "enhance_prompt": False
            }
            
            try:
                response = requests.post(
                    self.api_url + "/generate/text-to-image",
                    json=payload,
                    timeout=30
                )
                return {
                    "id": request_id,
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "data": response.json() if response.status_code == 200 else None
                }
            except Exception as e:
                return {
                    "id": request_id,
                    "success": False,
                    "error": str(e)
                }
        
        # Test with 3 concurrent requests (queue should handle this)
        num_requests = 3
        
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            results = [future.result() for future in futures]
        
        successful = sum(1 for r in results if r['success'])
        queued = sum(1 for r in results if r.get('data', {}).get('job_id'))
        
        print(f"   📊 Requests: {num_requests}, Successful: {successful}, Queued: {queued}")
        
        # At least one should succeed or be queued properly
        if successful >= 1 or queued >= 1:
            print("   ✅ Concurrent request handling working")
            return True
        else:
            print("   ❌ Concurrent request handling failed")
            return False
    
    def test_error_recovery(self) -> bool:
        """Test error recovery mechanisms"""
        print("   Testing error recovery...")
        
        # Test invalid request
        invalid_payload = {
            "prompt": "",  # Empty prompt should fail
            "width": -1,   # Invalid width
            "height": -1,  # Invalid height
        }
        
        try:
            response = requests.post(
                self.api_url + "/generate/text-to-image",
                json=invalid_payload,
                timeout=10
            )
            
            # Should fail gracefully with proper error message
            if response.status_code in [400, 422]:  # Validation errors
                print("   ✅ Invalid request handled gracefully")
                
                # Check if error message is informative
                if response.headers.get('content-type', '').startswith('application/json'):
                    error_data = response.json()
                    if 'detail' in error_data:
                        print(f"   ✅ Error message provided: {error_data['detail'][:50]}...")
                        return True
                
                return True
            else:
                print(f"   ⚠️ Unexpected response to invalid request: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error recovery test failed: {e}")
            return False
    
    def test_react_integration(self) -> bool:
        """Test React frontend integration"""
        try:
            response = requests.get(self.react_url, timeout=5)
            if response.status_code == 200:
                print("   ✅ React development server accessible")
                
                # Check if it's actually React (look for typical React patterns)
                content = response.text
                if 'react' in content.lower() or 'root' in content:
                    print("   ✅ React application detected")
                    return True
                else:
                    print("   ⚠️ React content not confirmed")
                    return False
            else:
                print(f"   ❌ React server returned: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("   ⚠️ React development server not running")
            print("   💡 Start with: cd frontend && npm start")
            return False
        except Exception as e:
            print(f"   ❌ React integration test error: {e}")
            return False
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 Integration Test Results Summary")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r['success'])
        total_time = sum(r['duration'] for r in self.test_results.values())
        
        print(f"Tests Run: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {100 * passed_tests / total_tests:.1f}%")
        print(f"Total Time: {total_time:.1f}s")
        
        if self.performance_metrics:
            print("\n📈 Performance Metrics:")
            for metric, value in self.performance_metrics.items():
                print(f"   {metric.replace('_', ' ').title()}: {value:.1f}s")
        
        print("\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            duration = result['duration']
            print(f"   {status} {test_name} ({duration:.1f}s)")
            
            if 'error' in result:
                print(f"       Error: {result['error']}")
        
        # Recommendations
        print("\n💡 Recommendations:")
        
        if passed_tests == total_tests:
            print("   🎉 All tests passed! System is ready for production.")
            print("   📝 Consider running performance tests under load.")
        elif passed_tests >= total_tests * 0.8:
            print("   ✅ Most tests passed. Review failed tests.")
            print("   🔧 Address any critical failures before deployment.")
        else:
            print("   ⚠️ Multiple test failures detected.")
            print("   🚨 System requires significant fixes before use.")
        
        # Service-specific recommendations
        failed_tests = [name for name, result in self.test_results.items() if not result['success']]
        
        if 'Service Health Check' in failed_tests:
            print("   🔧 Start missing services:")
            print("      - FastAPI: python src/api/main.py")
            print("      - React: cd frontend && npm start")
            print("      - Gradio: python src/qwen_image_ui.py")
        
        if 'React Frontend Integration' in failed_tests:
            print("   🔧 React setup required:")
            print("      - Install deps: cd frontend && npm install")
            print("      - Start server: npm start")
        
        if 'Performance Validation' in failed_tests:
            print("   🔧 Performance issues detected:")
            print("      - Check GPU memory and availability")
            print("      - Verify model is properly loaded")
            print("      - Monitor system resources")
    
    def calculate_overall_success(self) -> bool:
        """Calculate overall test success"""
        if not self.test_results:
            return False
        
        passed = sum(1 for r in self.test_results.values() if r['success'])
        total = len(self.test_results)
        
        # Require 80% success rate
        return passed >= total * 0.8

def main():
    """Main test execution"""
    print("🔧 Pre-flight checks...")
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Please run from the project root directory")
        return False
    
    # Check if API server is likely running
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        print("✅ FastAPI server detected")
    except:
        print("⚠️ FastAPI server not detected. Starting integration tests anyway...")
    
    # Run the test suite
    test_suite = IntegrationTestSuite()
    success = test_suite.run_all_tests()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)