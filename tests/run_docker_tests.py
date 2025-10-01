#!/usr/bin/env python3
"""
Docker Testing Suite Runner
Automated testing script for CI/CD pipeline to run all Docker-related tests.
"""

import os
import sys
import subprocess
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class DockerTestRunner:
    """Automated Docker test runner for CI/CD pipeline"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_default_config()
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        self.temp_dir = None
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default test configuration"""
        return {
            'test_suites': {
                'container_integration': {
                    'enabled': True,
                    'file': 'test_docker_container_integration.py',
                    'timeout': 600,  # 10 minutes
                    'required': True,
                    'description': 'Container integration and service communication tests'
                },
                'e2e_workflows': {
                    'enabled': True,
                    'file': 'test_docker_e2e_workflows.py',
                    'timeout': 1200,  # 20 minutes
                    'required': True,
                    'description': 'End-to-end workflow tests in Docker environment'
                },
                'performance_validation': {
                    'enabled': True,
                    'file': 'test_docker_performance_validation.py',
                    'timeout': 900,  # 15 minutes
                    'required': False,
                    'description': 'Performance and resource usage validation tests'
                }
            },
            'docker_setup': {
                'cleanup_before': True,
                'cleanup_after': True,
                'build_images': True,
                'pull_base_images': True
            },
            'reporting': {
                'generate_html_report': True,
                'generate_json_report': True,
                'save_logs': True,
                'upload_artifacts': False
            },
            'thresholds': {
                'min_success_rate': 0.8,
                'max_test_duration': 2400,  # 40 minutes total
                'max_memory_usage_mb': 8192,
                'max_cpu_usage_percent': 200
            }
        }
    
    def setup_test_environment(self) -> bool:
        """Setup test environment"""
        print("Setting up Docker test environment...")
        
        # Create temporary directory for test artifacts
        self.temp_dir = tempfile.mkdtemp(prefix="docker_tests_")
        print(f"Test artifacts directory: {self.temp_dir}")
        
        # Check Docker availability
        if not self._check_docker_available():
            return False
        
        # Cleanup existing test resources if configured
        if self.config['docker_setup']['cleanup_before']:
            self._cleanup_docker_resources()
        
        # Pull base images if configured
        if self.config['docker_setup']['pull_base_images']:
            self._pull_base_images()
        
        # Build test images if configured
        if self.config['docker_setup']['build_images']:
            if not self._build_test_images():
                return False
        
        return True
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all configured test suites"""
        print("Starting Docker test suite execution...")
        self.start_time = time.time()
        
        # Setup environment
        if not self.setup_test_environment():
            return self._generate_failure_report("Environment setup failed")
        
        # Run each test suite
        for suite_name, suite_config in self.config['test_suites'].items():
            if not suite_config['enabled']:
                print(f"Skipping disabled test suite: {suite_name}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Running test suite: {suite_name}")
            print(f"Description: {suite_config['description']}")
            print(f"{'='*60}")
            
            result = self._run_test_suite(suite_name, suite_config)
            self.test_results[suite_name] = result
            
            # Check if required test failed
            if suite_config['required'] and not result['success']:
                print(f"Required test suite {suite_name} failed, stopping execution")
                break
        
        self.end_time = time.time()
        
        # Generate final report
        return self._generate_final_report()
    
    def _check_docker_available(self) -> bool:
        """Check if Docker is available and running"""
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"Docker not available: {result.stderr}")
                return False
            
            # Check Docker Compose
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"Docker Compose not available: {result.stderr}")
                return False
            
            print("Docker and Docker Compose are available")
            return True
            
        except subprocess.TimeoutExpired:
            print("Docker check timed out")
            return False
        except FileNotFoundError:
            print("Docker command not found")
            return False
        except Exception as e:
            print(f"Error checking Docker: {e}")
            return False
    
    def _cleanup_docker_resources(self):
        """Cleanup existing Docker test resources"""
        print("Cleaning up existing Docker test resources...")
        
        cleanup_commands = [
            # Stop and remove test containers
            ["docker", "ps", "-a", "--filter", "name=test-", "-q"],
            # Remove test networks
            ["docker", "network", "ls", "--filter", "name=test-", "-q"],
            # Remove test volumes
            ["docker", "volume", "ls", "--filter", "name=test-", "-q"],
            # Prune unused resources
            ["docker", "system", "prune", "-f"]
        ]
        
        for cmd in cleanup_commands:
            try:
                if "ps" in cmd or "ls" in cmd:
                    # Get list of resources to remove
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0 and result.stdout.strip():
                        resource_ids = result.stdout.strip().split('\n')
                        
                        # Remove resources
                        if "ps" in cmd:
                            subprocess.run(["docker", "rm", "-f"] + resource_ids, 
                                         capture_output=True, timeout=60)
                        elif "network" in cmd:
                            for net_id in resource_ids:
                                subprocess.run(["docker", "network", "rm", net_id], 
                                             capture_output=True, timeout=30)
                        elif "volume" in cmd:
                            subprocess.run(["docker", "volume", "rm", "-f"] + resource_ids, 
                                         capture_output=True, timeout=60)
                else:
                    subprocess.run(cmd, capture_output=True, timeout=60)
                    
            except Exception as e:
                print(f"Warning: Cleanup command failed: {e}")
    
    def _pull_base_images(self):
        """Pull required base images"""
        print("Pulling base Docker images...")
        
        base_images = [
            "python:3.11-slim",
            "node:18-alpine",
            "nginx:alpine",
            "traefik:v3.0",
            "alpine:latest",
            "redis:7-alpine"
        ]
        
        for image in base_images:
            try:
                print(f"Pulling {image}...")
                result = subprocess.run(
                    ["docker", "pull", image],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes per image
                )
                
                if result.returncode != 0:
                    print(f"Warning: Failed to pull {image}: {result.stderr}")
                else:
                    print(f"Successfully pulled {image}")
                    
            except subprocess.TimeoutExpired:
                print(f"Warning: Timeout pulling {image}")
            except Exception as e:
                print(f"Warning: Error pulling {image}: {e}")
    
    def _build_test_images(self) -> bool:
        """Build Docker images for testing"""
        print("Building Docker images for testing...")
        
        project_root = Path(__file__).parent.parent
        
        build_configs = [
            {
                'name': 'qwen-api',
                'dockerfile': 'Dockerfile.api',
                'context': str(project_root),
                'timeout': 600
            },
            {
                'name': 'qwen-frontend',
                'dockerfile': 'Dockerfile',
                'context': str(project_root / 'frontend'),
                'timeout': 300
            }
        ]
        
        for config in build_configs:
            try:
                print(f"Building {config['name']}...")
                
                cmd = [
                    "docker", "build",
                    "-t", config['name'] + ":latest",
                    "-f", config['dockerfile'],
                    config['context']
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=config['timeout']
                )
                
                if result.returncode != 0:
                    print(f"Failed to build {config['name']}: {result.stderr}")
                    return False
                else:
                    print(f"Successfully built {config['name']}")
                    
            except subprocess.TimeoutExpired:
                print(f"Timeout building {config['name']}")
                return False
            except Exception as e:
                print(f"Error building {config['name']}: {e}")
                return False
        
        return True
    
    def _run_test_suite(self, suite_name: str, suite_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test suite"""
        start_time = time.time()
        
        # Prepare pytest command
        test_file = Path(__file__).parent / suite_config['file']
        
        if not test_file.exists():
            return {
                'success': False,
                'duration': 0,
                'error': f"Test file not found: {test_file}",
                'stdout': '',
                'stderr': '',
                'test_count': 0,
                'passed': 0,
                'failed': 0,
                'skipped': 0
            }
        
        # Create output files
        json_report_file = Path(self.temp_dir) / f"{suite_name}_report.json"
        log_file = Path(self.temp_dir) / f"{suite_name}_output.log"
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_file),
            "-v",
            "--tb=short",
            "--json-report",
            f"--json-report-file={json_report_file}",
            "--maxfail=5",
            "-x"  # Stop on first failure for CI
        ]
        
        try:
            # Run the test suite
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=suite_config['timeout'],
                cwd=Path(__file__).parent
            )
            
            duration = time.time() - start_time
            
            # Save output to log file
            with open(log_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"Duration: {duration:.2f}s\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
            
            # Parse JSON report if available
            test_details = self._parse_json_report(json_report_file)
            
            return {
                'success': result.returncode == 0,
                'duration': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'log_file': str(log_file),
                **test_details
            }
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                'success': False,
                'duration': duration,
                'error': f"Test suite timed out after {suite_config['timeout']} seconds",
                'stdout': '',
                'stderr': '',
                'test_count': 0,
                'passed': 0,
                'failed': 0,
                'skipped': 0
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                'success': False,
                'duration': duration,
                'error': f"Error running test suite: {e}",
                'stdout': '',
                'stderr': '',
                'test_count': 0,
                'passed': 0,
                'failed': 0,
                'skipped': 0
            }
    
    def _parse_json_report(self, json_file: Path) -> Dict[str, Any]:
        """Parse pytest JSON report"""
        try:
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                summary = data.get('summary', {})
                return {
                    'test_count': summary.get('total', 0),
                    'passed': summary.get('passed', 0),
                    'failed': summary.get('failed', 0),
                    'skipped': summary.get('skipped', 0),
                    'errors': summary.get('error', 0),
                    'json_report': str(json_file)
                }
        except Exception as e:
            print(f"Warning: Could not parse JSON report {json_file}: {e}")
        
        return {
            'test_count': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0
        }
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final test report"""
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Calculate summary statistics
        total_tests = sum(r.get('test_count', 0) for r in self.test_results.values())
        total_passed = sum(r.get('passed', 0) for r in self.test_results.values())
        total_failed = sum(r.get('failed', 0) for r in self.test_results.values())
        total_skipped = sum(r.get('skipped', 0) for r in self.test_results.values())
        
        successful_suites = sum(1 for r in self.test_results.values() if r.get('success', False))
        total_suites = len(self.test_results)
        
        success_rate = total_passed / total_tests if total_tests > 0 else 0
        suite_success_rate = successful_suites / total_suites if total_suites > 0 else 0
        
        report = {
            'summary': {
                'total_duration': total_duration,
                'total_suites': total_suites,
                'successful_suites': successful_suites,
                'suite_success_rate': suite_success_rate,
                'total_tests': total_tests,
                'total_passed': total_passed,
                'total_failed': total_failed,
                'total_skipped': total_skipped,
                'test_success_rate': success_rate,
                'overall_success': suite_success_rate >= self.config['thresholds']['min_success_rate']
            },
            'suite_results': self.test_results,
            'configuration': self.config,
            'artifacts_directory': self.temp_dir,
            'timestamp': time.time()
        }
        
        # Save reports
        if self.config['reporting']['generate_json_report']:
            self._save_json_report(report)
        
        if self.config['reporting']['generate_html_report']:
            self._generate_html_report(report)
        
        return report
    
    def _generate_failure_report(self, error_message: str) -> Dict[str, Any]:
        """Generate failure report when setup fails"""
        return {
            'summary': {
                'total_duration': 0,
                'total_suites': 0,
                'successful_suites': 0,
                'suite_success_rate': 0,
                'total_tests': 0,
                'total_passed': 0,
                'total_failed': 0,
                'total_skipped': 0,
                'test_success_rate': 0,
                'overall_success': False,
                'setup_error': error_message
            },
            'suite_results': {},
            'configuration': self.config,
            'timestamp': time.time()
        }
    
    def _save_json_report(self, report: Dict[str, Any]):
        """Save JSON report"""
        try:
            report_file = Path(self.temp_dir) / "docker_test_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"JSON report saved: {report_file}")
        except Exception as e:
            print(f"Warning: Could not save JSON report: {e}")
    
    def _generate_html_report(self, report: Dict[str, Any]):
        """Generate HTML report"""
        try:
            html_content = self._create_html_report_content(report)
            report_file = Path(self.temp_dir) / "docker_test_report.html"
            
            with open(report_file, 'w') as f:
                f.write(html_content)
            
            print(f"HTML report saved: {report_file}")
        except Exception as e:
            print(f"Warning: Could not generate HTML report: {e}")
    
    def _create_html_report_content(self, report: Dict[str, Any]) -> str:
        """Create HTML report content"""
        summary = report['summary']
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Docker Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
        .warning {{ color: orange; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .suite-success {{ background-color: #d4edda; }}
        .suite-failure {{ background-color: #f8d7da; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Docker Test Report</h1>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report['timestamp']))}</p>
        <p>Duration: {summary['total_duration']:.1f} seconds</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Overall Result:</strong> 
            <span class="{'success' if summary['overall_success'] else 'failure'}">
                {'PASSED' if summary['overall_success'] else 'FAILED'}
            </span>
        </p>
        <p><strong>Test Suites:</strong> {summary['successful_suites']}/{summary['total_suites']} passed 
           ({summary['suite_success_rate']:.1%})</p>
        <p><strong>Individual Tests:</strong> {summary['total_passed']}/{summary['total_tests']} passed 
           ({summary['test_success_rate']:.1%})</p>
        <p><strong>Failed:</strong> {summary['total_failed']}, 
           <strong>Skipped:</strong> {summary['total_skipped']}</p>
    </div>
    
    <h2>Test Suite Results</h2>
    <table>
        <tr>
            <th>Suite</th>
            <th>Status</th>
            <th>Duration</th>
            <th>Tests</th>
            <th>Passed</th>
            <th>Failed</th>
            <th>Skipped</th>
        </tr>
"""
        
        for suite_name, result in report['suite_results'].items():
            status_class = 'suite-success' if result.get('success', False) else 'suite-failure'
            status_text = 'PASSED' if result.get('success', False) else 'FAILED'
            
            html += f"""
        <tr class="{status_class}">
            <td>{suite_name}</td>
            <td>{status_text}</td>
            <td>{result.get('duration', 0):.1f}s</td>
            <td>{result.get('test_count', 0)}</td>
            <td>{result.get('passed', 0)}</td>
            <td>{result.get('failed', 0)}</td>
            <td>{result.get('skipped', 0)}</td>
        </tr>
"""
        
        html += """
    </table>
</body>
</html>
"""
        return html
    
    def cleanup(self):
        """Cleanup test environment"""
        if self.config['docker_setup']['cleanup_after']:
            self._cleanup_docker_resources()
        
        # Keep temp directory for artifact collection in CI
        if self.temp_dir and not self.config['reporting']['save_logs']:
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning: Could not cleanup temp directory: {e}")
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted test report"""
        print("\n" + "="*80)
        print("DOCKER TEST SUITE REPORT")
        print("="*80)
        
        summary = report['summary']
        
        print(f"\nOVERALL RESULT: {'PASSED' if summary['overall_success'] else 'FAILED'}")
        print(f"Total Duration: {summary['total_duration']:.1f} seconds")
        print(f"Test Suites: {summary['successful_suites']}/{summary['total_suites']} passed ({summary['suite_success_rate']:.1%})")
        print(f"Individual Tests: {summary['total_passed']}/{summary['total_tests']} passed ({summary['test_success_rate']:.1%})")
        
        if summary['total_failed'] > 0 or summary['total_skipped'] > 0:
            print(f"Failed: {summary['total_failed']}, Skipped: {summary['total_skipped']}")
        
        print(f"\nSUITE RESULTS:")
        for suite_name, result in report['suite_results'].items():
            status = "PASSED" if result.get('success', False) else "FAILED"
            duration = result.get('duration', 0)
            test_info = f"{result.get('passed', 0)}/{result.get('test_count', 0)} tests"
            
            print(f"  {suite_name}: {status} ({duration:.1f}s, {test_info})")
            
            if not result.get('success', False) and result.get('error'):
                print(f"    Error: {result['error']}")
        
        if report.get('artifacts_directory'):
            print(f"\nArtifacts saved to: {report['artifacts_directory']}")
        
        print("\n" + "="*80)


def main():
    """Main entry point for Docker test runner"""
    parser = argparse.ArgumentParser(description="Run Docker test suite")
    parser.add_argument("--config", help="Path to test configuration file")
    parser.add_argument("--suite", help="Run specific test suite only")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip cleanup after tests")
    parser.add_argument("--no-build", action="store_true", help="Skip building Docker images")
    parser.add_argument("--output-dir", help="Directory for test artifacts")
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    
    # Create test runner
    runner = DockerTestRunner(config)
    
    # Apply command line overrides
    if args.no_cleanup:
        runner.config['docker_setup']['cleanup_after'] = False
    
    if args.no_build:
        runner.config['docker_setup']['build_images'] = False
    
    if args.suite:
        # Run only specified suite
        for suite_name in list(runner.config['test_suites'].keys()):
            if suite_name != args.suite:
                runner.config['test_suites'][suite_name]['enabled'] = False
    
    if args.output_dir:
        runner.temp_dir = args.output_dir
        os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Run tests
        report = runner.run_all_tests()
        
        # Print report
        runner.print_report(report)
        
        # Exit with appropriate code
        if report['summary']['overall_success']:
            print("\n✅ All Docker tests passed!")
            exit_code = 0
        else:
            print("\n❌ Docker tests failed!")
            exit_code = 1
        
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user")
        exit_code = 130
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        exit_code = 1
    finally:
        # Cleanup
        runner.cleanup()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()