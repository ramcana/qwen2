#!/usr/bin/env python3
"""
Complete Frontend-Backend Integration Test Runner

This script runs comprehensive integration tests to validate:
1. Container status and health
2. API connectivity and routing
3. Frontend-backend communication
4. CORS configuration
5. Data flow validation
6. Performance metrics

Usage:
    python test-integration-complete.py [--quick] [--save-results]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

class IntegrationTestRunner:
    """Orchestrates complete integration testing"""
    
    def __init__(self, quick_mode: bool = False, save_results: bool = True):
        self.quick_mode = quick_mode
        self.save_results = save_results
        self.test_results = {
            "timestamp": time.time(),
            "quick_mode": quick_mode,
            "tests": {},
            "summary": {},
            "errors": []
        }
        
        # Test configuration
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / "tests"
        self.frontend_dir = self.project_root / "frontend"
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def run_command(self, command: List[str], cwd: Optional[Path] = None, 
                   timeout: int = 60) -> Dict:
        """Run a command and return results"""
        try:
            self.log(f"Running: {' '.join(command)}")
            
            result = subprocess.run(
                command,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": ' '.join(command)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "command": ' '.join(command)
            }
        except Exception as e:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "command": ' '.join(command)
            }
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are available"""
        self.log("Checking prerequisites...")
        
        prerequisites = [
            ("python3", ["python3", "--version"]),
            ("node", ["node", "--version"]),
            ("docker", ["docker", "--version"]),
            ("requests", ["python3", "-c", "import requests; print('requests available')"]),
        ]
        
        all_available = True
        
        for name, command in prerequisites:
            result = self.run_command(command, timeout=10)
            if result["success"]:
                self.log(f"✅ {name}: Available")
            else:
                self.log(f"❌ {name}: Not available - {result['stderr']}", "ERROR")
                all_available = False
        
        return all_available
    
    def test_api_connectivity(self) -> Dict:
        """Run API connectivity tests"""
        self.log("Running API connectivity tests...")
        
        test_script = self.tests_dir / "test_api_connectivity.py"
        if not test_script.exists():
            return {
                "success": False,
                "error": "API connectivity test script not found"
            }
        
        result = self.run_command(
            ["python3", str(test_script)],
            timeout=120
        )
        
        # Try to load results file
        results_file = self.project_root / "api_connectivity_results.json"
        detailed_results = {}
        if results_file.exists():
            try:
                with open(results_file) as f:
                    detailed_results = json.load(f)
            except Exception as e:
                self.log(f"Could not load API connectivity results: {e}", "WARNING")
        
        return {
            "success": result["success"],
            "returncode": result["returncode"],
            "output": result["stdout"],
            "error": result["stderr"],
            "detailed_results": detailed_results
        }
    
    def test_frontend_integration(self) -> Dict:
        """Run frontend integration tests"""
        self.log("Running frontend integration tests...")
        
        test_script = self.frontend_dir / "test-integration.js"
        if not test_script.exists():
            return {
                "success": False,
                "error": "Frontend integration test script not found"
            }
        
        # Check if node_modules exists
        node_modules = self.frontend_dir / "node_modules"
        if not node_modules.exists():
            self.log("Installing frontend dependencies...", "WARNING")
            npm_install = self.run_command(
                ["npm", "install"],
                cwd=self.frontend_dir,
                timeout=300
            )
            if not npm_install["success"]:
                return {
                    "success": False,
                    "error": f"Failed to install frontend dependencies: {npm_install['stderr']}"
                }
        
        result = self.run_command(
            ["node", str(test_script)],
            cwd=self.frontend_dir,
            timeout=120
        )
        
        # Try to load results file
        results_file = self.frontend_dir / "frontend-integration-test-results.json"
        detailed_results = {}
        if results_file.exists():
            try:
                with open(results_file) as f:
                    detailed_results = json.load(f)
            except Exception as e:
                self.log(f"Could not load frontend integration results: {e}", "WARNING")
        
        return {
            "success": result["success"],
            "returncode": result["returncode"],
            "output": result["stdout"],
            "error": result["stderr"],
            "detailed_results": detailed_results
        }
    
    def test_docker_integration(self) -> Dict:
        """Run Docker-based integration tests (if not in quick mode)"""
        if self.quick_mode:
            self.log("Skipping Docker integration tests (quick mode)")
            return {"success": True, "skipped": True, "reason": "quick_mode"}
        
        self.log("Running Docker integration tests...")
        
        test_script = self.tests_dir / "test_frontend_backend_integration.py"
        if not test_script.exists():
            return {
                "success": False,
                "error": "Docker integration test script not found"
            }
        
        # Check if Docker is available and containers are running
        docker_check = self.run_command(["docker", "ps"], timeout=10)
        if not docker_check["success"]:
            return {
                "success": False,
                "error": "Docker not available or not running"
            }
        
        result = self.run_command(
            ["python3", str(test_script)],
            timeout=300
        )
        
        # Try to load results file
        results_file = self.tests_dir / "frontend_backend_integration_test_results.json"
        detailed_results = {}
        if results_file.exists():
            try:
                with open(results_file) as f:
                    detailed_results = json.load(f)
            except Exception as e:
                self.log(f"Could not load Docker integration results: {e}", "WARNING")
        
        return {
            "success": result["success"],
            "returncode": result["returncode"],
            "output": result["stdout"],
            "error": result["stderr"],
            "detailed_results": detailed_results
        }
    
    def validate_application_functionality(self) -> Dict:
        """Validate basic application functionality"""
        self.log("Validating application functionality...")
        
        import requests
        
        functionality_tests = [
            {
                "name": "frontend_health",
                "url": "http://localhost/health",
                "expected_status": 200
            },
            {
                "name": "backend_health",
                "url": "http://localhost/api/health",
                "expected_status": 200
            },
            {
                "name": "backend_status",
                "url": "http://localhost/api/status",
                "expected_status": 200,
                "validate_json": True,
                "required_fields": ["model_loaded", "device", "memory_info"]
            },
            {
                "name": "aspect_ratios",
                "url": "http://localhost/api/aspect-ratios",
                "expected_status": 200,
                "validate_json": True
            }
        ]
        
        results = {}
        all_passed = True
        
        for test in functionality_tests:
            try:
                response = requests.get(test["url"], timeout=15)
                
                # Check status code
                status_ok = response.status_code == test["expected_status"]
                
                # Validate JSON if required
                json_ok = True
                json_data = {}
                if test.get("validate_json"):
                    try:
                        json_data = response.json()
                        if test.get("required_fields"):
                            json_ok = all(field in json_data for field in test["required_fields"])
                    except Exception:
                        json_ok = False
                
                test_passed = status_ok and json_ok
                
                results[test["name"]] = {
                    "success": test_passed,
                    "status_code": response.status_code,
                    "expected_status": test["expected_status"],
                    "json_valid": json_ok,
                    "response_size": len(response.content),
                    "json_data": json_data if test.get("validate_json") else None
                }
                
                if test_passed:
                    self.log(f"✅ {test['name']}: OK")
                else:
                    self.log(f"❌ {test['name']}: Failed", "ERROR")
                    all_passed = False
                
            except Exception as e:
                results[test["name"]] = {
                    "success": False,
                    "error": str(e)
                }
                self.log(f"❌ {test['name']}: {str(e)}", "ERROR")
                all_passed = False
        
        return {
            "success": all_passed,
            "tests": results
        }
    
    def run_all_tests(self) -> Dict:
        """Run all integration tests"""
        self.log("🚀 Starting Complete Integration Test Suite")
        self.log("=" * 60)
        
        start_time = time.time()
        
        # Check prerequisites
        if not self.check_prerequisites():
            self.log("Prerequisites check failed", "ERROR")
            return {
                "success": False,
                "error": "Prerequisites not met"
            }
        
        # Run test suites
        test_suites = [
            ("api_connectivity", self.test_api_connectivity),
            ("frontend_integration", self.test_frontend_integration),
            ("docker_integration", self.test_docker_integration),
            ("application_functionality", self.validate_application_functionality)
        ]
        
        for suite_name, test_function in test_suites:
            self.log(f"\n📋 Running {suite_name.replace('_', ' ').title()} Tests...")
            
            try:
                result = test_function()
                self.test_results["tests"][suite_name] = result
                
                if result.get("skipped"):
                    self.log(f"⏭️  {suite_name}: Skipped - {result.get('reason', 'unknown')}")
                elif result["success"]:
                    self.log(f"✅ {suite_name}: PASSED")
                else:
                    self.log(f"❌ {suite_name}: FAILED", "ERROR")
                    if result.get("error"):
                        self.test_results["errors"].append({
                            "suite": suite_name,
                            "error": result["error"]
                        })
                
            except Exception as e:
                self.log(f"💥 {suite_name}: Exception - {str(e)}", "ERROR")
                self.test_results["tests"][suite_name] = {
                    "success": False,
                    "error": str(e)
                }
                self.test_results["errors"].append({
                    "suite": suite_name,
                    "error": str(e)
                })
        
        # Calculate summary
        total_time = time.time() - start_time
        
        test_results = self.test_results["tests"]
        passed_tests = sum(1 for result in test_results.values() 
                          if result.get("success") and not result.get("skipped"))
        failed_tests = sum(1 for result in test_results.values() 
                          if not result.get("success") and not result.get("skipped"))
        skipped_tests = sum(1 for result in test_results.values() 
                           if result.get("skipped"))
        
        overall_success = failed_tests == 0 and passed_tests > 0
        
        self.test_results["summary"] = {
            "overall_success": overall_success,
            "total_time_seconds": total_time,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "skipped_tests": skipped_tests,
            "total_tests": len(test_results)
        }
        
        # Print summary
        self.log("\n" + "=" * 60)
        self.log("🏁 Integration Test Summary")
        self.log("=" * 60)
        
        for suite_name, result in test_results.items():
            if result.get("skipped"):
                status = "⏭️  SKIP"
            elif result["success"]:
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            
            display_name = suite_name.replace('_', ' ').title()
            self.log(f"{status} {display_name}")
        
        self.log(f"\n⏱️  Total test time: {total_time:.2f} seconds")
        self.log(f"📊 Results: {passed_tests} passed, {failed_tests} failed, {skipped_tests} skipped")
        self.log(f"🎯 Overall result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        
        if self.test_results["errors"]:
            self.log(f"\n❌ {len(self.test_results['errors'])} errors found:")
            for error in self.test_results["errors"][:3]:
                self.log(f"   • {error['suite']}: {error['error']}")
            if len(self.test_results["errors"]) > 3:
                self.log(f"   ... and {len(self.test_results['errors']) - 3} more errors")
        
        return self.test_results
    
    def save_test_results(self, filename: str = "complete_integration_test_results.json"):
        """Save test results to file"""
        if not self.save_results:
            return
        
        results_path = self.project_root / filename
        with open(results_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        self.log(f"\n💾 Complete test results saved to: {results_path}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Complete Frontend-Backend Integration Test Runner")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick tests only (skip Docker integration)")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save detailed results to file")
    
    args = parser.parse_args()
    
    runner = IntegrationTestRunner(
        quick_mode=args.quick,
        save_results=not args.no_save
    )
    
    try:
        results = runner.run_all_tests()
        runner.save_test_results()
        
        # Exit with appropriate code
        overall_success = results["summary"]["overall_success"]
        sys.exit(0 if overall_success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()