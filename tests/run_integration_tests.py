#!/usr/bin/env python3
"""
Integration Test Runner
Runs all integration tests for complete workflows and generates a comprehensive report.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class IntegrationTestRunner:
    """Runner for integration tests with reporting"""
    
    def __init__(self):
        self.test_files = [
            'test_complete_workflow_integration_e2e.py',
            'test_api_integration_workflows.py',
            'test_complete_workflow_integration.py',  # Existing file
            'test_api_workflow_integration.py',       # Existing file
            'test_frontend_workflow_integration.py'   # Existing file
        ]
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_test_file(self, test_file: str) -> Dict[str, Any]:
        """Run a single test file and return results"""
        print(f"\n{'='*60}")
        print(f"Running {test_file}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Run pytest with verbose output and JSON report
            cmd = [
                sys.executable, '-m', 'pytest',
                test_file,
                '-v',
                '--tb=short',
                '--json-report',
                f'--json-report-file=test_results_{test_file.replace(".py", "")}.json',
                '--maxfail=5'
            ]
            
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(__file__),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per test file
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse JSON report if available
            json_report_file = f'test_results_{test_file.replace(".py", "")}.json'
            json_report_path = os.path.join(os.path.dirname(__file__), json_report_file)
            
            test_details = {}
            if os.path.exists(json_report_path):
                try:
                    with open(json_report_path, 'r') as f:
                        json_data = json.load(f)
                        test_details = {
                            'total_tests': json_data.get('summary', {}).get('total', 0),
                            'passed': json_data.get('summary', {}).get('passed', 0),
                            'failed': json_data.get('summary', {}).get('failed', 0),
                            'skipped': json_data.get('summary', {}).get('skipped', 0),
                            'errors': json_data.get('summary', {}).get('error', 0)
                        }
                except Exception as e:
                    print(f"Warning: Could not parse JSON report: {e}")
            
            return {
                'file': test_file,
                'success': result.returncode == 0,
                'duration': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'test_details': test_details
            }
            
        except subprocess.TimeoutExpired:
            return {
                'file': test_file,
                'success': False,
                'duration': 300,
                'return_code': -1,
                'stdout': '',
                'stderr': 'Test execution timed out after 5 minutes',
                'test_details': {}
            }
        except Exception as e:
            return {
                'file': test_file,
                'success': False,
                'duration': time.time() - start_time,
                'return_code': -1,
                'stdout': '',
                'stderr': f'Error running test: {str(e)}',
                'test_details': {}
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        print("Starting Integration Test Suite")
        print(f"Running {len(self.test_files)} test files")
        
        self.start_time = time.time()
        
        for test_file in self.test_files:
            test_path = os.path.join(os.path.dirname(__file__), test_file)
            
            if not os.path.exists(test_path):
                print(f"Warning: Test file {test_file} not found, skipping...")
                self.results[test_file] = {
                    'file': test_file,
                    'success': False,
                    'duration': 0,
                    'return_code': -1,
                    'stdout': '',
                    'stderr': f'Test file {test_file} not found',
                    'test_details': {}
                }
                continue
            
            result = self.run_test_file(test_file)
            self.results[test_file] = result
            
            # Print immediate feedback
            status = "PASSED" if result['success'] else "FAILED"
            print(f"\n{test_file}: {status} ({result['duration']:.2f}s)")
            
            if not result['success']:
                print(f"Error output: {result['stderr'][:500]}...")
        
        self.end_time = time.time()
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Calculate summary statistics
        total_files = len(self.results)
        passed_files = sum(1 for r in self.results.values() if r['success'])
        failed_files = total_files - passed_files
        
        total_tests = sum(r['test_details'].get('total_tests', 0) for r in self.results.values())
        total_passed = sum(r['test_details'].get('passed', 0) for r in self.results.values())
        total_failed = sum(r['test_details'].get('failed', 0) for r in self.results.values())
        total_skipped = sum(r['test_details'].get('skipped', 0) for r in self.results.values())
        total_errors = sum(r['test_details'].get('errors', 0) for r in self.results.values())
        
        report = {
            'summary': {
                'total_duration': total_duration,
                'total_files': total_files,
                'passed_files': passed_files,
                'failed_files': failed_files,
                'success_rate': (passed_files / total_files * 100) if total_files > 0 else 0,
                'total_tests': total_tests,
                'total_passed': total_passed,
                'total_failed': total_failed,
                'total_skipped': total_skipped,
                'total_errors': total_errors,
                'test_success_rate': (total_passed / total_tests * 100) if total_tests > 0 else 0
            },
            'file_results': self.results,
            'workflow_coverage': self.analyze_workflow_coverage(),
            'recommendations': self.generate_recommendations()
        }
        
        return report
    
    def analyze_workflow_coverage(self) -> Dict[str, Any]:
        """Analyze which workflows are covered by the tests"""
        coverage = {
            'text_to_image_workflow': False,
            'diffsynth_editing_workflow': False,
            'controlnet_workflow': False,
            'service_switching_workflow': False,
            'resource_sharing_workflow': False,
            'error_recovery_workflow': False,
            'api_integration_workflow': False,
            'complete_end_to_end_workflow': False
        }
        
        # Analyze test file names and results to determine coverage
        for test_file, result in self.results.items():
            if result['success']:
                if 'text_to_image' in test_file.lower() or 'qwen' in result['stdout'].lower():
                    coverage['text_to_image_workflow'] = True
                
                if 'diffsynth' in test_file.lower() or 'edit' in result['stdout'].lower():
                    coverage['diffsynth_editing_workflow'] = True
                
                if 'controlnet' in test_file.lower():
                    coverage['controlnet_workflow'] = True
                
                if 'service' in test_file.lower() and 'switch' in result['stdout'].lower():
                    coverage['service_switching_workflow'] = True
                
                if 'resource' in result['stdout'].lower():
                    coverage['resource_sharing_workflow'] = True
                
                if 'error' in result['stdout'].lower() or 'recovery' in result['stdout'].lower():
                    coverage['error_recovery_workflow'] = True
                
                if 'api' in test_file.lower():
                    coverage['api_integration_workflow'] = True
                
                if 'complete' in test_file.lower() or 'e2e' in test_file.lower():
                    coverage['complete_end_to_end_workflow'] = True
        
        coverage_percentage = sum(coverage.values()) / len(coverage) * 100
        
        return {
            'workflows': coverage,
            'coverage_percentage': coverage_percentage,
            'covered_workflows': [k for k, v in coverage.items() if v],
            'missing_workflows': [k for k, v in coverage.items() if not v]
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check for failed tests
        failed_files = [f for f, r in self.results.items() if not r['success']]
        if failed_files:
            recommendations.append(
                f"Fix failing test files: {', '.join(failed_files)}"
            )
        
        # Check for missing workflow coverage
        coverage = self.analyze_workflow_coverage()
        if coverage['missing_workflows']:
            recommendations.append(
                f"Add tests for missing workflows: {', '.join(coverage['missing_workflows'])}"
            )
        
        # Check for performance issues
        slow_tests = [f for f, r in self.results.items() if r['duration'] > 60]
        if slow_tests:
            recommendations.append(
                f"Optimize slow-running tests: {', '.join(slow_tests)}"
            )
        
        # Check for skipped tests
        total_skipped = sum(r['test_details'].get('skipped', 0) for r in self.results.values())
        if total_skipped > 0:
            recommendations.append(
                f"Investigate {total_skipped} skipped tests - they may indicate missing dependencies"
            )
        
        return recommendations
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted test report"""
        print("\n" + "="*80)
        print("INTEGRATION TEST REPORT")
        print("="*80)
        
        summary = report['summary']
        print(f"\nSUMMARY:")
        print(f"  Total Duration: {summary['total_duration']:.2f} seconds")
        print(f"  Test Files: {summary['passed_files']}/{summary['total_files']} passed ({summary['success_rate']:.1f}%)")
        print(f"  Individual Tests: {summary['total_passed']}/{summary['total_tests']} passed ({summary['test_success_rate']:.1f}%)")
        print(f"  Failed: {summary['total_failed']}, Skipped: {summary['total_skipped']}, Errors: {summary['total_errors']}")
        
        print(f"\nWORKFLOW COVERAGE:")
        coverage = report['workflow_coverage']
        print(f"  Overall Coverage: {coverage['coverage_percentage']:.1f}%")
        print(f"  Covered Workflows: {len(coverage['covered_workflows'])}/{len(coverage['workflows'])}")
        
        for workflow, covered in coverage['workflows'].items():
            status = "✓" if covered else "✗"
            print(f"    {status} {workflow.replace('_', ' ').title()}")
        
        print(f"\nFILE RESULTS:")
        for test_file, result in report['file_results'].items():
            status = "PASSED" if result['success'] else "FAILED"
            duration = result['duration']
            details = result['test_details']
            
            print(f"  {test_file}: {status} ({duration:.2f}s)")
            if details:
                print(f"    Tests: {details.get('passed', 0)}/{details.get('total_tests', 0)} passed")
            
            if not result['success'] and result['stderr']:
                print(f"    Error: {result['stderr'][:100]}...")
        
        if report['recommendations']:
            print(f"\nRECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*80)
    
    def save_report(self, report: Dict[str, Any], filename: str = "integration_test_report.json"):
        """Save report to JSON file"""
        report_path = os.path.join(os.path.dirname(__file__), filename)
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nDetailed report saved to: {report_path}")
        except Exception as e:
            print(f"Warning: Could not save report to {report_path}: {e}")


def main():
    """Main entry point"""
    runner = IntegrationTestRunner()
    
    try:
        report = runner.run_all_tests()
        runner.print_report(report)
        runner.save_report(report)
        
        # Exit with appropriate code
        if report['summary']['failed_files'] == 0:
            print("\n🎉 All integration tests passed!")
            sys.exit(0)
        else:
            print(f"\n❌ {report['summary']['failed_files']} test file(s) failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()