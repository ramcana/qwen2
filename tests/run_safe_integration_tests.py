#!/usr/bin/env python3
"""
Safe Integration Test Runner
Runs integration tests that work in the current environment without problematic dependencies.
"""

import os
import sys
import subprocess
import time
from typing import Dict, List, Any


class SafeIntegrationTestRunner:
    """Runner for safe integration tests"""
    
    def __init__(self):
        self.safe_test_files = [
            'test_integration_safe.py'
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
            # Run pytest with basic options
            cmd = [
                sys.executable, '-m', 'pytest',
                test_file,
                '-v',
                '--tb=short',
                '--maxfail=10'
            ]
            
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(__file__),
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per test file
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse output for test counts
            stdout = result.stdout
            test_counts = self.parse_test_output(stdout)
            
            return {
                'file': test_file,
                'success': result.returncode == 0,
                'duration': duration,
                'return_code': result.returncode,
                'stdout': stdout,
                'stderr': result.stderr,
                'test_counts': test_counts
            }
            
        except subprocess.TimeoutExpired:
            return {
                'file': test_file,
                'success': False,
                'duration': 120,
                'return_code': -1,
                'stdout': '',
                'stderr': 'Test execution timed out after 2 minutes',
                'test_counts': {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0}
            }
        except Exception as e:
            return {
                'file': test_file,
                'success': False,
                'duration': time.time() - start_time,
                'return_code': -1,
                'stdout': '',
                'stderr': f'Error running test: {str(e)}',
                'test_counts': {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0}
            }
    
    def parse_test_output(self, output: str) -> Dict[str, int]:
        """Parse pytest output to extract test counts"""
        counts = {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0}
        
        lines = output.split('\n')
        for line in lines:
            if 'passed' in line and ('failed' in line or 'error' in line or 'skipped' in line):
                # Look for summary line like "14 passed in 0.09s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed' and i > 0:
                        try:
                            counts['passed'] = int(parts[i-1])
                            counts['total'] += counts['passed']
                        except (ValueError, IndexError):
                            pass
                    elif part == 'failed' and i > 0:
                        try:
                            counts['failed'] = int(parts[i-1])
                            counts['total'] += counts['failed']
                        except (ValueError, IndexError):
                            pass
                    elif part == 'skipped' and i > 0:
                        try:
                            counts['skipped'] = int(parts[i-1])
                            counts['total'] += counts['skipped']
                        except (ValueError, IndexError):
                            pass
            elif 'collected' in line and 'items' in line:
                # Look for collection line like "collected 14 items"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'collected' and i + 1 < len(parts):
                        try:
                            counts['total'] = int(parts[i+1])
                        except (ValueError, IndexError):
                            pass
        
        return counts
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all safe integration tests"""
        print("Starting Safe Integration Test Suite")
        print(f"Running {len(self.safe_test_files)} test file(s)")
        
        self.start_time = time.time()
        
        for test_file in self.safe_test_files:
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
                    'test_counts': {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0}
                }
                continue
            
            result = self.run_test_file(test_file)
            self.results[test_file] = result
            
            # Print immediate feedback
            status = "PASSED" if result['success'] else "FAILED"
            counts = result['test_counts']
            print(f"\n{test_file}: {status} ({result['duration']:.2f}s)")
            print(f"  Tests: {counts['passed']}/{counts['total']} passed")
            
            if not result['success']:
                print(f"  Error: {result['stderr'][:200]}...")
        
        self.end_time = time.time()
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Calculate summary statistics
        total_files = len(self.results)
        passed_files = sum(1 for r in self.results.values() if r['success'])
        failed_files = total_files - passed_files
        
        total_tests = sum(r['test_counts']['total'] for r in self.results.values())
        total_passed = sum(r['test_counts']['passed'] for r in self.results.values())
        total_failed = sum(r['test_counts']['failed'] for r in self.results.values())
        total_skipped = sum(r['test_counts']['skipped'] for r in self.results.values())
        
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
                'test_success_rate': (total_passed / total_tests * 100) if total_tests > 0 else 0
            },
            'file_results': self.results,
            'workflow_coverage': self.analyze_workflow_coverage(),
            'recommendations': self.generate_recommendations()
        }
        
        return report
    
    def analyze_workflow_coverage(self) -> Dict[str, Any]:
        """Analyze which workflows are covered by the safe tests"""
        coverage = {
            'text_to_image_workflow': False,
            'diffsynth_editing_workflow': False,
            'controlnet_workflow': False,
            'service_switching_workflow': False,
            'resource_sharing_workflow': False,
            'error_recovery_workflow': False,
            'complete_end_to_end_workflow': False
        }
        
        # Analyze test results to determine coverage
        for test_file, result in self.results.items():
            if result['success']:
                stdout = result['stdout'].lower()
                
                if 'text_to_image' in stdout or 'generation' in stdout:
                    coverage['text_to_image_workflow'] = True
                
                if 'diffsynth' in stdout or 'editing' in stdout or 'inpaint' in stdout:
                    coverage['diffsynth_editing_workflow'] = True
                
                if 'controlnet' in stdout or 'control' in stdout:
                    coverage['controlnet_workflow'] = True
                
                if 'service' in stdout and 'switch' in stdout:
                    coverage['service_switching_workflow'] = True
                
                if 'resource' in stdout or 'memory' in stdout:
                    coverage['resource_sharing_workflow'] = True
                
                if 'error' in stdout or 'recovery' in stdout:
                    coverage['error_recovery_workflow'] = True
                
                if 'complete' in stdout or 'workflow' in stdout:
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
        slow_tests = [f for f, r in self.results.items() if r['duration'] > 30]
        if slow_tests:
            recommendations.append(
                f"Optimize slow-running tests: {', '.join(slow_tests)}"
            )
        
        return recommendations
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted test report"""
        print("\n" + "="*80)
        print("SAFE INTEGRATION TEST REPORT")
        print("="*80)
        
        summary = report['summary']
        print(f"\nSUMMARY:")
        print(f"  Total Duration: {summary['total_duration']:.2f} seconds")
        print(f"  Test Files: {summary['passed_files']}/{summary['total_files']} passed ({summary['success_rate']:.1f}%)")
        print(f"  Individual Tests: {summary['total_passed']}/{summary['total_tests']} passed ({summary['test_success_rate']:.1f}%)")
        print(f"  Failed: {summary['total_failed']}, Skipped: {summary['total_skipped']}")
        
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
            counts = result['test_counts']
            
            print(f"  {test_file}: {status} ({duration:.2f}s)")
            print(f"    Tests: {counts['passed']}/{counts['total']} passed")
            
            if not result['success'] and result['stderr']:
                print(f"    Error: {result['stderr'][:100]}...")
        
        if report['recommendations']:
            print(f"\nRECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*80)


def main():
    """Main entry point"""
    runner = SafeIntegrationTestRunner()
    
    try:
        report = runner.run_all_tests()
        runner.print_report(report)
        
        # Exit with appropriate code
        if report['summary']['failed_files'] == 0:
            print("\n🎉 All safe integration tests passed!")
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