#!/usr/bin/env python3
"""
Validate Integration Test Structure
Validates the structure and syntax of integration tests without importing dependencies.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Any


class IntegrationTestValidator:
    """Validator for integration test files"""
    
    def __init__(self):
        self.test_files = [
            'test_complete_workflow_integration_e2e.py',
            'test_api_integration_workflows.py'
        ]
        self.validation_results = {}
    
    def validate_syntax(self, file_path: str) -> Dict[str, Any]:
        """Validate Python syntax of a test file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST to check syntax
            ast.parse(content)
            
            return {
                'valid': True,
                'error': None,
                'line_count': len(content.splitlines())
            }
        except SyntaxError as e:
            return {
                'valid': False,
                'error': f"Syntax error at line {e.lineno}: {e.msg}",
                'line_count': 0
            }
        except Exception as e:
            return {
                'valid': False,
                'error': f"Error reading file: {str(e)}",
                'line_count': 0
            }
    
    def analyze_test_structure(self, file_path: str) -> Dict[str, Any]:
        """Analyze the structure of a test file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            classes = []
            functions = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    classes.append({
                        'name': node.name,
                        'methods': class_methods,
                        'test_methods': [m for m in class_methods if m.startswith('test_')]
                    })
                elif isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    functions.append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        imports.extend([alias.name for alias in node.names])
                    else:
                        imports.append(node.module or 'relative_import')
            
            return {
                'classes': classes,
                'standalone_functions': functions,
                'imports': imports,
                'total_test_methods': sum(len(c['test_methods']) for c in classes) + len(functions)
            }
        except Exception as e:
            return {
                'error': f"Error analyzing structure: {str(e)}",
                'classes': [],
                'standalone_functions': [],
                'imports': [],
                'total_test_methods': 0
            }
    
    def check_workflow_coverage(self, file_path: str) -> Dict[str, Any]:
        """Check which workflows are covered by the test file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            
            workflows = {
                'text_to_image': any(term in content for term in ['text_to_image', 'qwen', 'generation']),
                'diffsynth_editing': any(term in content for term in ['diffsynth', 'edit_image', 'inpaint', 'outpaint']),
                'controlnet': any(term in content for term in ['controlnet', 'control_net', 'canny', 'depth']),
                'service_switching': any(term in content for term in ['switch_service', 'service_switching']),
                'resource_sharing': any(term in content for term in ['resource', 'memory', 'allocation']),
                'error_recovery': any(term in content for term in ['error', 'recovery', 'fallback']),
                'api_integration': any(term in content for term in ['api', 'endpoint', 'client.post', 'client.get']),
                'complete_workflow': any(term in content for term in ['complete', 'end_to_end', 'e2e', 'workflow'])
            }
            
            coverage_count = sum(workflows.values())
            coverage_percentage = (coverage_count / len(workflows)) * 100
            
            return {
                'workflows': workflows,
                'coverage_count': coverage_count,
                'coverage_percentage': coverage_percentage
            }
        except Exception as e:
            return {
                'error': f"Error checking coverage: {str(e)}",
                'workflows': {},
                'coverage_count': 0,
                'coverage_percentage': 0
            }
    
    def validate_all_files(self) -> Dict[str, Any]:
        """Validate all integration test files"""
        results = {}
        
        for test_file in self.test_files:
            file_path = os.path.join(os.path.dirname(__file__), test_file)
            
            if not os.path.exists(file_path):
                results[test_file] = {
                    'exists': False,
                    'syntax': {'valid': False, 'error': 'File not found'},
                    'structure': {'error': 'File not found'},
                    'coverage': {'error': 'File not found'}
                }
                continue
            
            print(f"Validating {test_file}...")
            
            results[test_file] = {
                'exists': True,
                'syntax': self.validate_syntax(file_path),
                'structure': self.analyze_test_structure(file_path),
                'coverage': self.check_workflow_coverage(file_path)
            }
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation report"""
        total_files = len(results)
        valid_files = sum(1 for r in results.values() if r.get('syntax', {}).get('valid', False))
        
        total_test_methods = sum(
            r.get('structure', {}).get('total_test_methods', 0) 
            for r in results.values()
        )
        
        all_workflows = set()
        covered_workflows = set()
        
        for file_result in results.values():
            coverage = file_result.get('coverage', {})
            workflows = coverage.get('workflows', {})
            
            all_workflows.update(workflows.keys())
            covered_workflows.update(k for k, v in workflows.items() if v)
        
        overall_coverage = (len(covered_workflows) / len(all_workflows) * 100) if all_workflows else 0
        
        return {
            'summary': {
                'total_files': total_files,
                'valid_files': valid_files,
                'invalid_files': total_files - valid_files,
                'total_test_methods': total_test_methods,
                'overall_coverage_percentage': overall_coverage,
                'covered_workflows': list(covered_workflows),
                'missing_workflows': list(all_workflows - covered_workflows)
            },
            'file_results': results
        }
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted validation report"""
        print("\n" + "="*80)
        print("INTEGRATION TEST VALIDATION REPORT")
        print("="*80)
        
        summary = report['summary']
        print(f"\nSUMMARY:")
        print(f"  Files: {summary['valid_files']}/{summary['total_files']} valid")
        print(f"  Total Test Methods: {summary['total_test_methods']}")
        print(f"  Workflow Coverage: {summary['overall_coverage_percentage']:.1f}%")
        
        print(f"\nCOVERED WORKFLOWS:")
        for workflow in summary['covered_workflows']:
            print(f"  ✓ {workflow.replace('_', ' ').title()}")
        
        if summary['missing_workflows']:
            print(f"\nMISSING WORKFLOWS:")
            for workflow in summary['missing_workflows']:
                print(f"  ✗ {workflow.replace('_', ' ').title()}")
        
        print(f"\nFILE DETAILS:")
        for file_name, result in report['file_results'].items():
            print(f"\n  {file_name}:")
            
            if not result['exists']:
                print(f"    ❌ File not found")
                continue
            
            syntax = result['syntax']
            if syntax['valid']:
                print(f"    ✅ Syntax valid ({syntax['line_count']} lines)")
            else:
                print(f"    ❌ Syntax error: {syntax['error']}")
                continue
            
            structure = result['structure']
            if 'error' not in structure:
                print(f"    📊 Structure: {len(structure['classes'])} classes, {structure['total_test_methods']} test methods")
                
                for cls in structure['classes']:
                    print(f"      - {cls['name']}: {len(cls['test_methods'])} tests")
            else:
                print(f"    ❌ Structure error: {structure['error']}")
            
            coverage = result['coverage']
            if 'error' not in coverage:
                print(f"    🎯 Coverage: {coverage['coverage_percentage']:.1f}% ({coverage['coverage_count']}/8 workflows)")
            else:
                print(f"    ❌ Coverage error: {coverage['error']}")
        
        print("\n" + "="*80)


def main():
    """Main entry point"""
    validator = IntegrationTestValidator()
    
    print("Validating integration test files...")
    results = validator.validate_all_files()
    report = validator.generate_report(results)
    validator.print_report(report)
    
    # Exit with appropriate code
    if report['summary']['invalid_files'] == 0:
        print("\n🎉 All integration test files are valid!")
        return 0
    else:
        print(f"\n❌ {report['summary']['invalid_files']} test file(s) have issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())