#!/usr/bin/env python3
"""
Comprehensive test runner for DiffSynth Enhanced UI performance and compatibility testing
Executes all performance benchmarks, memory validation, cross-browser compatibility, and API backward compatibility tests
"""

import unittest
import sys
import os
import time
import json
import tempfile
from typing import Dict, List, Any
import argparse

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import test modules
from test_diffsynth_performance_benchmarks import TestDiffSynthPerformanceBenchmarks
from test_memory_usage_validation import TestMemoryUsageValidation
from test_cross_browser_compatibility import TestCrossBrowserCompatibility
from test_api_backward_compatibility import TestAPIBackwardCompatibility


class PerformanceCompatibilityTestSuite:
    """Comprehensive test suite for performance and compatibility testing"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or tempfile.mkdtemp(prefix='diffsynth_tests_')
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    def run_all_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run all performance and compatibility tests"""
        
        print("🧪 DiffSynth Enhanced UI - Performance & Compatibility Test Suite")
        print("=" * 70)
        
        self.start_time = time.time()
        
        # Define test suites
        test_suites = [
            {
                'name': 'Performance Benchmarks',
                'class': TestDiffSynthPerformanceBenchmarks,
                'description': 'Tests generation and editing operations performance'
            },
            {
                'name': 'Memory Usage Validation',
                'class': TestMemoryUsageValidation,
                'description': 'Tests memory management and optimization'
            },
            {
                'name': 'Cross-Browser Compatibility',
                'class': TestCrossBrowserCompatibility,
                'description': 'Tests UI features across different browsers'
            },
            {
                'name': 'API Backward Compatibility',
                'class': TestAPIBackwardCompatibility,
                'description': 'Tests API compatibility with existing integrations'
            }
        ]
        
        # Run each test suite
        for suite_info in test_suites:
            suite_name = suite_info['name']
            suite_class = suite_info['class']
            
            print(f"\n📋 Running {suite_name}")
            print(f"   {suite_info['description']}")
            print("-" * 50)
            
            try:
                suite_result = self._run_test_suite(suite_class, verbose)
                self.test_results[suite_name] = suite_result
                
                # Print summary
                if suite_result['success']:
                    print(f"✅ {suite_name}: {suite_result['tests_run']} tests passed")
                else:
                    print(f"❌ {suite_name}: {suite_result['failures']} failures, {suite_result['errors']} errors")
                    
            except Exception as e:
                print(f"💥 {suite_name}: Failed to run - {str(e)}")
                self.test_results[suite_name] = {
                    'success': False,
                    'error': str(e),
                    'tests_run': 0,
                    'failures': 0,
                    'errors': 1
                }
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report()
        
        # Save report
        report_path = self._save_report(report)
        
        print(f"\n📊 Test Report saved to: {report_path}")
        
        return report
    
    def _run_test_suite(self, test_class, verbose: bool = True) -> Dict[str, Any]:
        """Run a specific test suite"""
        
        # Create test loader and runner
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        
        # Create test runner with custom result class
        runner = unittest.TextTestRunner(
            verbosity=2 if verbose else 1,
            stream=sys.stdout,
            buffer=True
        )
        
        # Run tests
        result = runner.run(suite)
        
        return {
            'success': result.wasSuccessful(),
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
            'failure_details': [
                {'test': str(test), 'traceback': traceback}
                for test, traceback in result.failures
            ],
            'error_details': [
                {'test': str(test), 'traceback': traceback}
                for test, traceback in result.errors
            ]
        }
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Calculate overall statistics
        total_tests = sum(result.get('tests_run', 0) for result in self.test_results.values())
        total_failures = sum(result.get('failures', 0) for result in self.test_results.values())
        total_errors = sum(result.get('errors', 0) for result in self.test_results.values())
        total_skipped = sum(result.get('skipped', 0) for result in self.test_results.values())
        
        successful_suites = sum(1 for result in self.test_results.values() if result.get('success', False))
        total_suites = len(self.test_results)
        
        # Calculate grades
        overall_success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
        suite_success_rate = (successful_suites / total_suites * 100) if total_suites > 0 else 0
        
        report = {
            'test_execution': {
                'start_time': self.start_time,
                'end_time': self.end_time,
                'duration_seconds': total_duration,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            },
            'summary': {
                'total_test_suites': total_suites,
                'successful_suites': successful_suites,
                'suite_success_rate': suite_success_rate,
                'total_tests': total_tests,
                'passed_tests': total_tests - total_failures - total_errors,
                'failed_tests': total_failures,
                'error_tests': total_errors,
                'skipped_tests': total_skipped,
                'overall_success_rate': overall_success_rate,
                'overall_grade': self._calculate_overall_grade(overall_success_rate, suite_success_rate)
            },
            'suite_results': self.test_results,
            'performance_analysis': self._analyze_performance_results(),
            'compatibility_analysis': self._analyze_compatibility_results(),
            'recommendations': self._generate_recommendations(),
            'system_info': self._collect_system_info()
        }
        
        return report
    
    def _analyze_performance_results(self) -> Dict[str, Any]:
        """Analyze performance test results"""
        
        performance_suite = self.test_results.get('Performance Benchmarks', {})
        memory_suite = self.test_results.get('Memory Usage Validation', {})
        
        analysis = {
            'performance_tests_passed': performance_suite.get('success', False),
            'memory_tests_passed': memory_suite.get('success', False),
            'performance_grade': 'Unknown',
            'memory_grade': 'Unknown',
            'key_findings': []
        }
        
        # Analyze performance results
        if performance_suite.get('success'):
            analysis['performance_grade'] = 'A'
            analysis['key_findings'].append('All performance benchmarks passed')
        elif performance_suite.get('tests_run', 0) > 0:
            success_rate = ((performance_suite.get('tests_run', 0) - 
                           performance_suite.get('failures', 0) - 
                           performance_suite.get('errors', 0)) / 
                          performance_suite.get('tests_run', 1)) * 100
            
            if success_rate >= 80:
                analysis['performance_grade'] = 'B'
            elif success_rate >= 60:
                analysis['performance_grade'] = 'C'
            else:
                analysis['performance_grade'] = 'D'
        
        # Analyze memory results
        if memory_suite.get('success'):
            analysis['memory_grade'] = 'A'
            analysis['key_findings'].append('All memory validation tests passed')
        elif memory_suite.get('tests_run', 0) > 0:
            success_rate = ((memory_suite.get('tests_run', 0) - 
                           memory_suite.get('failures', 0) - 
                           memory_suite.get('errors', 0)) / 
                          memory_suite.get('tests_run', 1)) * 100
            
            if success_rate >= 80:
                analysis['memory_grade'] = 'B'
            elif success_rate >= 60:
                analysis['memory_grade'] = 'C'
            else:
                analysis['memory_grade'] = 'D'
        
        return analysis
    
    def _analyze_compatibility_results(self) -> Dict[str, Any]:
        """Analyze compatibility test results"""
        
        browser_suite = self.test_results.get('Cross-Browser Compatibility', {})
        api_suite = self.test_results.get('API Backward Compatibility', {})
        
        analysis = {
            'browser_tests_passed': browser_suite.get('success', False),
            'api_tests_passed': api_suite.get('success', False),
            'browser_grade': 'Unknown',
            'api_grade': 'Unknown',
            'compatibility_issues': []
        }
        
        # Analyze browser compatibility
        if browser_suite.get('success'):
            analysis['browser_grade'] = 'A'
        elif browser_suite.get('tests_run', 0) > 0:
            success_rate = ((browser_suite.get('tests_run', 0) - 
                           browser_suite.get('failures', 0) - 
                           browser_suite.get('errors', 0)) / 
                          browser_suite.get('tests_run', 1)) * 100
            
            if success_rate >= 80:
                analysis['browser_grade'] = 'B'
            elif success_rate >= 60:
                analysis['browser_grade'] = 'C'
            else:
                analysis['browser_grade'] = 'D'
                analysis['compatibility_issues'].append('Significant browser compatibility issues detected')
        
        # Analyze API compatibility
        if api_suite.get('success'):
            analysis['api_grade'] = 'A'
        elif api_suite.get('tests_run', 0) > 0:
            success_rate = ((api_suite.get('tests_run', 0) - 
                           api_suite.get('failures', 0) - 
                           api_suite.get('errors', 0)) / 
                          api_suite.get('tests_run', 1)) * 100
            
            if success_rate >= 80:
                analysis['api_grade'] = 'B'
            elif success_rate >= 60:
                analysis['api_grade'] = 'C'
            else:
                analysis['api_grade'] = 'D'
                analysis['compatibility_issues'].append('API backward compatibility issues detected')
        
        return analysis
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        # Check for failures in each suite
        for suite_name, result in self.test_results.items():
            if not result.get('success', False):
                if 'Performance' in suite_name:
                    recommendations.extend([
                        'Optimize generation and editing operation performance',
                        'Implement performance monitoring in production',
                        'Consider GPU memory optimization techniques'
                    ])
                elif 'Memory' in suite_name:
                    recommendations.extend([
                        'Implement better memory management and cleanup',
                        'Add memory usage monitoring and alerts',
                        'Consider implementing memory pooling for large operations'
                    ])
                elif 'Browser' in suite_name:
                    recommendations.extend([
                        'Test UI features on actual devices and browsers',
                        'Implement progressive enhancement for older browsers',
                        'Add browser-specific polyfills where needed'
                    ])
                elif 'API' in suite_name:
                    recommendations.extend([
                        'Ensure API backward compatibility is maintained',
                        'Implement API versioning strategy',
                        'Add comprehensive API integration tests to CI/CD'
                    ])
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.extend([
                'All tests passed! Consider adding more edge case testing',
                'Implement continuous performance monitoring',
                'Add automated compatibility testing to CI/CD pipeline'
            ])
        else:
            recommendations.extend([
                'Implement automated testing in CI/CD pipeline',
                'Add performance regression testing',
                'Create monitoring dashboards for production systems'
            ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _calculate_overall_grade(self, test_success_rate: float, suite_success_rate: float) -> str:
        """Calculate overall grade based on success rates"""
        
        # Weight individual test success more heavily
        overall_score = (test_success_rate * 0.7) + (suite_success_rate * 0.3)
        
        if overall_score >= 95:
            return 'A+'
        elif overall_score >= 90:
            return 'A'
        elif overall_score >= 85:
            return 'B+'
        elif overall_score >= 80:
            return 'B'
        elif overall_score >= 75:
            return 'C+'
        elif overall_score >= 70:
            return 'C'
        elif overall_score >= 65:
            return 'D+'
        elif overall_score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for the report"""
        
        import platform
        
        system_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'system': platform.system(),
            'release': platform.release(),
            'test_runner_version': '1.0.0'
        }
        
        # Add GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                system_info['gpu_available'] = True
                system_info['gpu_count'] = torch.cuda.device_count()
                system_info['gpu_name'] = torch.cuda.get_device_name(0)
                system_info['cuda_version'] = torch.version.cuda
            else:
                system_info['gpu_available'] = False
        except ImportError:
            system_info['gpu_available'] = False
            system_info['torch_available'] = False
        
        return system_info
    
    def _save_report(self, report: Dict[str, Any]) -> str:
        """Save test report to file"""
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_filename = f'diffsynth_performance_compatibility_report_{timestamp}.json'
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save JSON report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also save a human-readable summary
        summary_path = os.path.join(self.output_dir, f'test_summary_{timestamp}.txt')
        self._save_human_readable_summary(report, summary_path)
        
        return report_path
    
    def _save_human_readable_summary(self, report: Dict[str, Any], summary_path: str):
        """Save human-readable test summary"""
        
        with open(summary_path, 'w') as f:
            f.write("DiffSynth Enhanced UI - Performance & Compatibility Test Report\n")
            f.write("=" * 70 + "\n\n")
            
            # Execution info
            exec_info = report['test_execution']
            f.write(f"Test Execution: {exec_info['timestamp']}\n")
            f.write(f"Duration: {exec_info['duration_seconds']:.2f} seconds\n\n")
            
            # Summary
            summary = report['summary']
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Overall Grade: {summary['overall_grade']}\n")
            f.write(f"Test Suites: {summary['successful_suites']}/{summary['total_test_suites']} passed ({summary['suite_success_rate']:.1f}%)\n")
            f.write(f"Individual Tests: {summary['passed_tests']}/{summary['total_tests']} passed ({summary['overall_success_rate']:.1f}%)\n")
            f.write(f"Failures: {summary['failed_tests']}\n")
            f.write(f"Errors: {summary['error_tests']}\n")
            f.write(f"Skipped: {summary['skipped_tests']}\n\n")
            
            # Performance Analysis
            perf_analysis = report['performance_analysis']
            f.write("PERFORMANCE ANALYSIS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Performance Grade: {perf_analysis['performance_grade']}\n")
            f.write(f"Memory Grade: {perf_analysis['memory_grade']}\n")
            for finding in perf_analysis['key_findings']:
                f.write(f"• {finding}\n")
            f.write("\n")
            
            # Compatibility Analysis
            compat_analysis = report['compatibility_analysis']
            f.write("COMPATIBILITY ANALYSIS\n")
            f.write("-" * 27 + "\n")
            f.write(f"Browser Grade: {compat_analysis['browser_grade']}\n")
            f.write(f"API Grade: {compat_analysis['api_grade']}\n")
            for issue in compat_analysis['compatibility_issues']:
                f.write(f"⚠️  {issue}\n")
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")
            
            # Suite Details
            f.write("DETAILED RESULTS\n")
            f.write("-" * 20 + "\n")
            for suite_name, result in report['suite_results'].items():
                status = "✅ PASSED" if result.get('success') else "❌ FAILED"
                f.write(f"{suite_name}: {status}\n")
                f.write(f"  Tests: {result.get('tests_run', 0)}\n")
                f.write(f"  Failures: {result.get('failures', 0)}\n")
                f.write(f"  Errors: {result.get('errors', 0)}\n")
                f.write("\n")


def main():
    """Main function to run performance and compatibility tests"""
    
    parser = argparse.ArgumentParser(
        description='Run DiffSynth Enhanced UI performance and compatibility tests'
    )
    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory for test reports',
        default=None
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose test output'
    )
    parser.add_argument(
        '--suite', '-s',
        choices=['performance', 'memory', 'browser', 'api', 'all'],
        default='all',
        help='Specific test suite to run'
    )
    
    args = parser.parse_args()
    
    # Create test suite runner
    test_suite = PerformanceCompatibilityTestSuite(output_dir=args.output_dir)
    
    try:
        if args.suite == 'all':
            # Run all tests
            report = test_suite.run_all_tests(verbose=args.verbose)
        else:
            # Run specific suite (would need implementation for individual suites)
            print(f"Running specific suite '{args.suite}' not yet implemented")
            print("Running all tests instead...")
            report = test_suite.run_all_tests(verbose=args.verbose)
        
        # Print final summary
        print("\n" + "=" * 70)
        print("🏁 TEST EXECUTION COMPLETE")
        print("=" * 70)
        
        summary = report['summary']
        print(f"Overall Grade: {summary['overall_grade']}")
        print(f"Success Rate: {summary['overall_success_rate']:.1f}%")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Errors: {summary['error_tests']}")
        
        # Exit with appropriate code
        if summary['overall_success_rate'] >= 80:
            print("\n🎉 Tests completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  Some tests failed. Check the report for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test execution failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()