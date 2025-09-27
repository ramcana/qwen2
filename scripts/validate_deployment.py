#!/usr/bin/env python3
"""
Deployment Validation Script for DiffSynth Enhanced UI
Tests all endpoints and services to ensure proper deployment
"""

import requests
import time
import json
import sys
import argparse
from typing import Dict, List, Tuple
from urllib.parse import urljoin

class DeploymentValidator:
    def __init__(self, base_url: str = "http://localhost:8000", frontend_url: str = "http://localhost:3000"):
        self.base_url = base_url.rstrip('/')
        self.frontend_url = frontend_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 30
        
        # Test results
        self.results = {
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'tests': []
        }
    
    def log(self, level: str, message: str):
        """Log a message with level"""
        colors = {
            'INFO': '\033[0;34m',
            'SUCCESS': '\033[0;32m',
            'WARNING': '\033[1;33m',
            'ERROR': '\033[0;31m',
            'NC': '\033[0m'
        }
        
        color = colors.get(level, colors['NC'])
        print(f"{color}[{level}]{colors['NC']} {message}")
    
    def test_endpoint(self, name: str, method: str, endpoint: str, expected_status: int = 200, 
                     data: Dict = None, timeout: int = 30) -> Tuple[bool, str]:
        """Test a single endpoint"""
        try:
            url = urljoin(self.base_url, endpoint)
            
            if method.upper() == 'GET':
                response = self.session.get(url, timeout=timeout)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, timeout=timeout)
            else:
                return False, f"Unsupported method: {method}"
            
            if response.status_code == expected_status:
                return True, f"Status: {response.status_code}"
            else:
                return False, f"Expected {expected_status}, got {response.status_code}"
                
        except requests.exceptions.Timeout:
            return False, "Request timeout"
        except requests.exceptions.ConnectionError:
            return False, "Connection error"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def run_test(self, name: str, test_func, critical: bool = True):
        """Run a test and record results"""
        self.log('INFO', f"Testing: {name}")
        
        try:
            success, message = test_func()
            
            test_result = {
                'name': name,
                'success': success,
                'message': message,
                'critical': critical
            }
            
            self.results['tests'].append(test_result)
            
            if success:
                self.results['passed'] += 1
                self.log('SUCCESS', f"✓ {name}: {message}")
            else:
                if critical:
                    self.results['failed'] += 1
                    self.log('ERROR', f"✗ {name}: {message}")
                else:
                    self.results['warnings'] += 1
                    self.log('WARNING', f"⚠ {name}: {message}")
                    
        except Exception as e:
            self.results['failed'] += 1
            self.log('ERROR', f"✗ {name}: Exception - {str(e)}")
    
    def test_basic_health(self):
        """Test basic health endpoint"""
        return self.test_endpoint("Health Check", "GET", "/health")
    
    def test_system_status(self):
        """Test system status endpoint"""
        success, message = self.test_endpoint("System Status", "GET", "/status")
        
        if success:
            try:
                response = self.session.get(urljoin(self.base_url, "/status"))
                data = response.json()
                
                # Check if model is loaded
                if data.get('model_loaded'):
                    return True, "System ready with model loaded"
                else:
                    return True, "System ready but model not loaded"
                    
            except Exception as e:
                return True, f"Status endpoint works but couldn't parse response: {e}"
        
        return success, message
    
    def test_memory_status(self):
        """Test memory status endpoint"""
        return self.test_endpoint("Memory Status", "GET", "/memory/status")
    
    def test_diffsynth_endpoints(self):
        """Test DiffSynth endpoints"""
        endpoints = [
            ("/diffsynth/status", "DiffSynth Status"),
            ("/controlnet/types", "ControlNet Types"),
        ]
        
        results = []
        for endpoint, name in endpoints:
            success, message = self.test_endpoint(name, "GET", endpoint)
            results.append((success, f"{name}: {message}"))
        
        # Return overall result
        successes = sum(1 for success, _ in results if success)
        if successes == len(results):
            return True, f"All DiffSynth endpoints working ({successes}/{len(results)})"
        elif successes > 0:
            return True, f"Some DiffSynth endpoints working ({successes}/{len(results)})"
        else:
            return False, "No DiffSynth endpoints working"
    
    def test_queue_management(self):
        """Test queue management endpoints"""
        return self.test_endpoint("Queue Status", "GET", "/queue")
    
    def test_frontend_accessibility(self):
        """Test frontend accessibility"""
        try:
            response = self.session.get(self.frontend_url, timeout=10)
            if response.status_code == 200:
                return True, f"Frontend accessible (Status: {response.status_code})"
            else:
                return False, f"Frontend returned status: {response.status_code}"
        except Exception as e:
            return False, f"Frontend not accessible: {str(e)}"
    
    def test_api_cors(self):
        """Test CORS headers"""
        try:
            response = self.session.options(urljoin(self.base_url, "/health"))
            cors_headers = [
                'Access-Control-Allow-Origin',
                'Access-Control-Allow-Methods',
                'Access-Control-Allow-Headers'
            ]
            
            found_headers = [h for h in cors_headers if h in response.headers]
            
            if found_headers:
                return True, f"CORS headers present: {', '.join(found_headers)}"
            else:
                return False, "No CORS headers found"
                
        except Exception as e:
            return False, f"CORS test failed: {str(e)}"
    
    def test_file_upload_endpoint(self):
        """Test file upload endpoint availability"""
        # We don't actually upload a file, just check if the endpoint exists
        try:
            response = self.session.post(urljoin(self.base_url, "/upload"))
            # We expect a 400 (bad request) since we're not sending a file
            if response.status_code in [400, 422]:
                return True, "Upload endpoint available"
            else:
                return False, f"Upload endpoint returned unexpected status: {response.status_code}"
        except Exception as e:
            return False, f"Upload endpoint test failed: {str(e)}"
    
    def test_model_initialization(self):
        """Test model initialization"""
        try:
            # Try to initialize the model
            response = self.session.post(urljoin(self.base_url, "/initialize"), timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return True, "Model initialization successful"
                else:
                    return False, f"Model initialization failed: {data.get('message', 'Unknown error')}"
            else:
                return False, f"Model initialization returned status: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return False, "Model initialization timeout (this is normal for first run)"
        except Exception as e:
            return False, f"Model initialization test failed: {str(e)}"
    
    def test_service_switching(self):
        """Test service switching endpoint"""
        return self.test_endpoint("Service Status", "GET", "/services/status")
    
    def run_all_tests(self):
        """Run all validation tests"""
        self.log('INFO', "Starting deployment validation...")
        self.log('INFO', f"API Base URL: {self.base_url}")
        self.log('INFO', f"Frontend URL: {self.frontend_url}")
        print()
        
        # Critical tests (must pass)
        self.run_test("Basic Health Check", self.test_basic_health, critical=True)
        self.run_test("System Status", self.test_system_status, critical=True)
        self.run_test("Memory Status", self.test_memory_status, critical=True)
        self.run_test("Queue Management", self.test_queue_management, critical=True)
        self.run_test("Frontend Accessibility", self.test_frontend_accessibility, critical=True)
        
        # Important tests (should pass)
        self.run_test("CORS Configuration", self.test_api_cors, critical=False)
        self.run_test("File Upload Endpoint", self.test_file_upload_endpoint, critical=False)
        self.run_test("Service Management", self.test_service_switching, critical=False)
        
        # Feature tests (may fail if services not ready)
        self.run_test("DiffSynth Endpoints", self.test_diffsynth_endpoints, critical=False)
        self.run_test("Model Initialization", self.test_model_initialization, critical=False)
        
        # Print results
        print()
        self.print_results()
        
        return self.results['failed'] == 0
    
    def print_results(self):
        """Print test results summary"""
        total_tests = len(self.results['tests'])
        passed = self.results['passed']
        failed = self.results['failed']
        warnings = self.results['warnings']
        
        self.log('INFO', "Validation Results:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Warnings: {warnings}")
        print()
        
        if failed == 0:
            self.log('SUCCESS', "All critical tests passed! Deployment appears successful.")
        else:
            self.log('ERROR', f"{failed} critical tests failed. Deployment may have issues.")
        
        if warnings > 0:
            self.log('WARNING', f"{warnings} non-critical tests had issues. Some features may not be available.")
        
        # Print failed tests
        failed_tests = [t for t in self.results['tests'] if not t['success'] and t['critical']]
        if failed_tests:
            print()
            self.log('ERROR', "Failed Critical Tests:")
            for test in failed_tests:
                print(f"  - {test['name']}: {test['message']}")
        
        # Print warnings
        warning_tests = [t for t in self.results['tests'] if not t['success'] and not t['critical']]
        if warning_tests:
            print()
            self.log('WARNING', "Non-Critical Issues:")
            for test in warning_tests:
                print(f"  - {test['name']}: {test['message']}")
    
    def save_results(self, filename: str):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.log('INFO', f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Validate DiffSynth Enhanced UI deployment')
    parser.add_argument('--api-url', default='http://localhost:8000', 
                       help='API base URL (default: http://localhost:8000)')
    parser.add_argument('--frontend-url', default='http://localhost:3000',
                       help='Frontend URL (default: http://localhost:3000)')
    parser.add_argument('--output', help='Save results to JSON file')
    parser.add_argument('--wait', type=int, default=0,
                       help='Wait N seconds before starting tests')
    
    args = parser.parse_args()
    
    if args.wait > 0:
        print(f"Waiting {args.wait} seconds before starting tests...")
        time.sleep(args.wait)
    
    validator = DeploymentValidator(args.api_url, args.frontend_url)
    success = validator.run_all_tests()
    
    if args.output:
        validator.save_results(args.output)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()