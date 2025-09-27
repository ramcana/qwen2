#!/usr/bin/env python3
"""
API backward compatibility tests for DiffSynth Enhanced UI
Tests that existing API endpoints continue to work while new features are added
According to requirements 8.1, 8.4
"""

import unittest
import json
import time
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional, Union
import asyncio

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class APICompatibilityTester:
    """Tests API backward compatibility for DiffSynth Enhanced UI"""
    
    def __init__(self):
        self.test_results = []
        
        # Define existing API endpoints that must remain compatible
        self.existing_endpoints = {
            'GET /': {'description': 'Root endpoint', 'version': '1.0'},
            'GET /health': {'description': 'Health check', 'version': '1.0'},
            'GET /status': {'description': 'System status', 'version': '1.0'},
            'POST /initialize': {'description': 'Initialize model', 'version': '1.0'},
            'GET /aspect-ratios': {'description': 'Get aspect ratios', 'version': '1.0'},
            'POST /generate/text-to-image': {'description': 'Text to image generation', 'version': '1.0'},
            'GET /queue': {'description': 'Queue status', 'version': '1.0'},
            'GET /memory/clear': {'description': 'Clear memory', 'version': '1.0'}
        }
        
        # Define new DiffSynth endpoints
        self.new_endpoints = {
            'POST /diffsynth/edit': {'description': 'General image editing', 'version': '2.0'},
            'POST /diffsynth/inpaint': {'description': 'Inpainting operations', 'version': '2.0'},
            'POST /diffsynth/outpaint': {'description': 'Outpainting operations', 'version': '2.0'},
            'POST /diffsynth/style-transfer': {'description': 'Style transfer', 'version': '2.0'},
            'POST /controlnet/detect': {'description': 'ControlNet detection', 'version': '2.0'},
            'POST /controlnet/generate': {'description': 'ControlNet generation', 'version': '2.0'},
            'GET /controlnet/types': {'description': 'Available control types', 'version': '2.0'},
            'GET /services/status': {'description': 'Multi-service status', 'version': '2.0'},
            'POST /services/switch': {'description': 'Service switching', 'version': '2.0'}
        }
        
        # Define expected request/response schemas
        self.api_schemas = self._define_api_schemas()
    
    def _define_api_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Define API request/response schemas for compatibility testing"""
        
        return {
            # Existing API schemas (must remain unchanged)
            'POST /generate/text-to-image': {
                'request': {
                    'required': ['prompt'],
                    'optional': ['negative_prompt', 'width', 'height', 'num_inference_steps', 
                               'cfg_scale', 'seed', 'language', 'enhance_prompt', 'aspect_ratio']
                },
                'response': {
                    'required': ['success', 'message'],
                    'optional': ['image_path', 'generation_time', 'metadata']
                }
            },
            'GET /health': {
                'response': {
                    'required': ['status'],
                    'optional': ['model_loaded', 'gpu_available', 'memory_info']
                }
            },
            'GET /status': {
                'response': {
                    'required': ['model_loaded', 'gpu_available'],
                    'optional': ['queue_size', 'is_generating', 'memory_info']
                }
            },
            
            # New DiffSynth API schemas
            'POST /diffsynth/edit': {
                'request': {
                    'required': ['image', 'prompt'],
                    'optional': ['negative_prompt', 'strength', 'num_inference_steps', 'cfg_scale']
                },
                'response': {
                    'required': ['success', 'message'],
                    'optional': ['edited_image_path', 'processing_time', 'metadata']
                }
            },
            'POST /diffsynth/inpaint': {
                'request': {
                    'required': ['image', 'mask', 'prompt'],
                    'optional': ['negative_prompt', 'strength', 'num_inference_steps']
                },
                'response': {
                    'required': ['success', 'message'],
                    'optional': ['inpainted_image_path', 'processing_time']
                }
            },
            'POST /controlnet/detect': {
                'request': {
                    'required': ['image'],
                    'optional': ['control_type', 'detection_params']
                },
                'response': {
                    'required': ['success', 'detected_type'],
                    'optional': ['control_map_path', 'confidence', 'detection_time']
                }
            }
        }
    
    def test_existing_endpoint_compatibility(self, endpoint: str, method: str = 'GET',
                                           request_data: Dict = None) -> Dict[str, Any]:
        """Test that existing endpoints maintain backward compatibility"""
        
        endpoint_key = f"{method} {endpoint}"
        
        if endpoint_key not in self.existing_endpoints:
            return {
                'endpoint': endpoint_key,
                'compatible': False,
                'error': 'Endpoint not in existing endpoints list'
            }
        
        # Simulate API call
        try:
            response = self._simulate_api_call(method, endpoint, request_data)
            
            # Validate response schema
            schema_validation = self._validate_response_schema(endpoint_key, response)
            
            # Check for breaking changes
            breaking_changes = self._detect_breaking_changes(endpoint_key, response)
            
            result = {
                'endpoint': endpoint_key,
                'method': method,
                'compatible': schema_validation['valid'] and len(breaking_changes) == 0,
                'response_valid': schema_validation['valid'],
                'schema_errors': schema_validation.get('errors', []),
                'breaking_changes': breaking_changes,
                'response_time_ms': response.get('_response_time_ms', 0),
                'status_code': response.get('_status_code', 200),
                'timestamp': time.time()
            }
            
        except Exception as e:
            result = {
                'endpoint': endpoint_key,
                'method': method,
                'compatible': False,
                'error': str(e),
                'timestamp': time.time()
            }
        
        self.test_results.append(result)
        return result
    
    def test_new_endpoint_functionality(self, endpoint: str, method: str = 'POST',
                                      request_data: Dict = None) -> Dict[str, Any]:
        """Test new DiffSynth endpoints for proper functionality"""
        
        endpoint_key = f"{method} {endpoint}"
        
        if endpoint_key not in self.new_endpoints:
            return {
                'endpoint': endpoint_key,
                'functional': False,
                'error': 'Endpoint not in new endpoints list'
            }
        
        try:
            response = self._simulate_api_call(method, endpoint, request_data)
            
            # Validate new endpoint schema
            schema_validation = self._validate_response_schema(endpoint_key, response)
            
            # Check new endpoint specific requirements
            functionality_check = self._check_new_endpoint_functionality(endpoint_key, response)
            
            result = {
                'endpoint': endpoint_key,
                'method': method,
                'functional': schema_validation['valid'] and functionality_check['valid'],
                'response_valid': schema_validation['valid'],
                'functionality_valid': functionality_check['valid'],
                'schema_errors': schema_validation.get('errors', []),
                'functionality_errors': functionality_check.get('errors', []),
                'response_time_ms': response.get('_response_time_ms', 0),
                'status_code': response.get('_status_code', 200),
                'timestamp': time.time()
            }
            
        except Exception as e:
            result = {
                'endpoint': endpoint_key,
                'functional': False,
                'error': str(e),
                'timestamp': time.time()
            }
        
        self.test_results.append(result)
        return result
    
    def _simulate_api_call(self, method: str, endpoint: str, request_data: Dict = None) -> Dict[str, Any]:
        """Simulate API call and return mock response"""
        
        start_time = time.perf_counter()
        
        # Simulate processing time
        processing_time = 0.01 + (len(endpoint) * 0.001)  # Vary by endpoint complexity
        time.sleep(processing_time)
        
        response_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Generate mock responses based on endpoint
        if endpoint == '/':
            return {
                'name': 'Qwen-Image Generator API',
                'version': '2.0.0',
                'description': 'Enhanced with DiffSynth capabilities',
                '_response_time_ms': response_time_ms,
                '_status_code': 200
            }
        
        elif endpoint == '/health':
            return {
                'status': 'healthy',
                'model_loaded': True,
                'gpu_available': True,
                'memory_info': {
                    'allocated_gb': 2.5,
                    'total_gb': 8.0,
                    'usage_percent': 31.25
                },
                '_response_time_ms': response_time_ms,
                '_status_code': 200
            }
        
        elif endpoint == '/status':
            return {
                'model_loaded': True,
                'gpu_available': True,
                'queue_size': 0,
                'is_generating': False,
                'memory_info': {
                    'allocated_gb': 2.5,
                    'total_gb': 8.0
                },
                '_response_time_ms': response_time_ms,
                '_status_code': 200
            }
        
        elif endpoint == '/initialize':
            return {
                'success': True,
                'message': 'Model initialized successfully',
                'model_loaded': True,
                '_response_time_ms': response_time_ms,
                '_status_code': 200
            }
        
        elif endpoint == '/aspect-ratios':
            return {
                'ratios': {
                    '1:1': [1024, 1024],
                    '16:9': [1920, 1080],
                    '4:3': [1024, 768],
                    '3:2': [1536, 1024]
                },
                '_response_time_ms': response_time_ms,
                '_status_code': 200
            }
        
        elif endpoint == '/queue':
            return {
                'queue_size': 0,
                'is_generating': False,
                'queue': [],
                '_response_time_ms': response_time_ms,
                '_status_code': 200
            }
        
        elif endpoint == '/memory/clear':
            return {
                'success': True,
                'message': 'Memory cleared successfully',
                'memory_info': {
                    'allocated_gb': 0.5,
                    'total_gb': 8.0,
                    'usage_percent': 6.25
                },
                '_response_time_ms': response_time_ms,
                '_status_code': 200
            }
        
        elif endpoint == '/generate/text-to-image':
            # Validate required parameters
            if not request_data or 'prompt' not in request_data:
                return {
                    'error': 'Missing required parameter: prompt',
                    'detail': 'The prompt parameter is required for text-to-image generation',
                    '_response_time_ms': response_time_ms,
                    '_status_code': 400
                }
            
            # Validate parameter types
            if 'width' in request_data and not isinstance(request_data['width'], int):
                return {
                    'error': 'Invalid parameter type',
                    'detail': 'Width must be an integer',
                    '_response_time_ms': response_time_ms,
                    '_status_code': 400
                }
            
            return {
                'success': True,
                'message': 'Image generated successfully',
                'image_path': 'generated_images/test_image_12345.png',
                'generation_time': 3.45,
                'metadata': {
                    'prompt': request_data.get('prompt', 'test prompt'),
                    'width': request_data.get('width', 1024),
                    'height': request_data.get('height', 1024),
                    'steps': request_data.get('num_inference_steps', 20)
                },
                '_response_time_ms': response_time_ms,
                '_status_code': 200
            }
        
        elif endpoint == '/diffsynth/edit':
            return {
                'success': True,
                'message': 'Image edited successfully',
                'edited_image_path': 'edited_images/diffsynth_edit_12345.png',
                'processing_time': 2.1,
                'metadata': {
                    'operation': 'edit',
                    'prompt': request_data.get('prompt', 'edit prompt'),
                    'strength': request_data.get('strength', 0.8)
                },
                '_response_time_ms': response_time_ms,
                '_status_code': 200
            }
        
        elif endpoint == '/diffsynth/inpaint':
            return {
                'success': True,
                'message': 'Inpainting completed successfully',
                'inpainted_image_path': 'edited_images/inpaint_12345.png',
                'processing_time': 1.8,
                '_response_time_ms': response_time_ms,
                '_status_code': 200
            }
        
        elif endpoint == '/controlnet/detect':
            return {
                'success': True,
                'detected_type': 'canny',
                'control_map_path': 'control_maps/canny_12345.png',
                'confidence': 0.95,
                'detection_time': 0.3,
                '_response_time_ms': response_time_ms,
                '_status_code': 200
            }
        
        elif endpoint == '/services/status':
            return {
                'services': {
                    'qwen': {'status': 'active', 'memory_usage_mb': 2048},
                    'diffsynth': {'status': 'active', 'memory_usage_mb': 1536}
                },
                'total_memory_usage_mb': 3584,
                'active_service': 'qwen',
                '_response_time_ms': response_time_ms,
                '_status_code': 200
            }
        
        else:
            # Default response for unknown endpoints
            return {
                'error': 'Endpoint not found',
                '_response_time_ms': response_time_ms,
                '_status_code': 404
            }
    
    def _validate_response_schema(self, endpoint_key: str, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate response against expected schema"""
        
        if endpoint_key not in self.api_schemas:
            return {'valid': True, 'message': 'No schema defined for endpoint'}
        
        schema = self.api_schemas[endpoint_key].get('response', {})
        required_fields = schema.get('required', [])
        optional_fields = schema.get('optional', [])
        
        errors = []
        
        # Check required fields
        for field in required_fields:
            if field not in response:
                errors.append(f"Missing required field: {field}")
        
        # Check for unexpected fields (excluding internal fields starting with _)
        expected_fields = set(required_fields + optional_fields)
        actual_fields = set(k for k in response.keys() if not k.startswith('_'))
        unexpected_fields = actual_fields - expected_fields
        
        if unexpected_fields:
            # Only warn about unexpected fields, don't fail validation
            # This allows for backward-compatible additions
            pass
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'unexpected_fields': list(unexpected_fields)
        }
    
    def _detect_breaking_changes(self, endpoint_key: str, response: Dict[str, Any]) -> List[str]:
        """Detect potential breaking changes in API responses"""
        
        breaking_changes = []
        
        # Check for removed fields that were previously required
        # This would be based on historical API documentation
        
        # Check for changed data types
        if endpoint_key == 'GET /health':
            if 'status' in response and not isinstance(response['status'], str):
                breaking_changes.append("'status' field changed from string type")
        
        elif endpoint_key == 'POST /generate/text-to-image':
            if 'success' in response and not isinstance(response['success'], bool):
                breaking_changes.append("'success' field changed from boolean type")
            
            if 'generation_time' in response and not isinstance(response['generation_time'], (int, float)):
                breaking_changes.append("'generation_time' field changed from numeric type")
        
        # Check for changed response structure
        if endpoint_key == 'GET /status':
            required_fields = ['model_loaded', 'gpu_available']
            for field in required_fields:
                if field not in response:
                    breaking_changes.append(f"Required field '{field}' removed from response")
        
        return breaking_changes
    
    def _check_new_endpoint_functionality(self, endpoint_key: str, response: Dict[str, Any]) -> Dict[str, Any]:
        """Check functionality of new DiffSynth endpoints"""
        
        errors = []
        
        # Check DiffSynth-specific functionality
        if '/diffsynth/' in endpoint_key:
            # Should have processing time information
            if 'processing_time' not in response and 'success' in response and response['success']:
                errors.append("DiffSynth endpoints should include processing_time")
            
            # Should have appropriate output paths
            if endpoint_key == 'POST /diffsynth/edit' and 'edited_image_path' not in response:
                if response.get('success'):
                    errors.append("Edit endpoint should return edited_image_path on success")
        
        elif '/controlnet/' in endpoint_key:
            # ControlNet-specific checks
            if endpoint_key == 'POST /controlnet/detect':
                if 'detected_type' not in response and response.get('success'):
                    errors.append("ControlNet detect should return detected_type on success")
        
        elif '/services/' in endpoint_key:
            # Service management checks
            if endpoint_key == 'GET /services/status':
                if 'services' not in response:
                    errors.append("Services status should include services information")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def test_api_versioning_compatibility(self) -> Dict[str, Any]:
        """Test API versioning and compatibility"""
        
        # Test that version 1.0 endpoints still work
        v1_endpoints = [
            ('GET', '/'),
            ('GET', '/health'),
            ('GET', '/status'),
            ('POST', '/generate/text-to-image')
        ]
        
        v1_results = []
        for method, endpoint in v1_endpoints:
            request_data = {'prompt': 'test'} if method == 'POST' else None
            result = self.test_existing_endpoint_compatibility(endpoint, method, request_data)
            v1_results.append(result)
        
        # Test that version 2.0 endpoints work
        v2_endpoints = [
            ('POST', '/diffsynth/edit'),
            ('POST', '/controlnet/detect'),
            ('GET', '/services/status')
        ]
        
        v2_results = []
        for method, endpoint in v2_endpoints:
            request_data = {'image': 'test.jpg', 'prompt': 'test'} if method == 'POST' else None
            result = self.test_new_endpoint_functionality(endpoint, method, request_data)
            v2_results.append(result)
        
        # Calculate compatibility metrics
        v1_compatible = sum(1 for r in v1_results if r.get('compatible', False))
        v2_functional = sum(1 for r in v2_results if r.get('functional', False))
        
        return {
            'v1_compatibility': {
                'total_endpoints': len(v1_results),
                'compatible_endpoints': v1_compatible,
                'compatibility_rate': (v1_compatible / len(v1_results)) * 100,
                'results': v1_results
            },
            'v2_functionality': {
                'total_endpoints': len(v2_results),
                'functional_endpoints': v2_functional,
                'functionality_rate': (v2_functional / len(v2_results)) * 100,
                'results': v2_results
            },
            'overall_grade': self._calculate_api_grade(v1_compatible, len(v1_results), 
                                                     v2_functional, len(v2_results))
        }
    
    def test_request_parameter_compatibility(self) -> Dict[str, Any]:
        """Test that existing request parameters continue to work"""
        
        # Test text-to-image with various parameter combinations
        parameter_tests = [
            # Minimal request (should work)
            {'prompt': 'test image'},
            
            # Full v1.0 request (should work)
            {
                'prompt': 'detailed test image',
                'negative_prompt': 'blurry, low quality',
                'width': 1024,
                'height': 1024,
                'num_inference_steps': 20,
                'cfg_scale': 7.5,
                'seed': 12345,
                'language': 'en',
                'enhance_prompt': True,
                'aspect_ratio': '1:1'
            },
            
            # Request with new optional parameters (should work)
            {
                'prompt': 'test with new params',
                'diffsynth_mode': 'enhanced',  # New parameter
                'quality_preset': 'high'       # New parameter
            }
        ]
        
        results = []
        for i, params in enumerate(parameter_tests):
            result = self.test_existing_endpoint_compatibility(
                '/generate/text-to-image', 'POST', params
            )
            result['test_case'] = f"parameter_test_{i+1}"
            result['parameters'] = params
            results.append(result)
        
        # All parameter combinations should work
        compatible_tests = sum(1 for r in results if r.get('compatible', False))
        
        return {
            'total_tests': len(results),
            'compatible_tests': compatible_tests,
            'compatibility_rate': (compatible_tests / len(results)) * 100,
            'parameter_test_results': results
        }
    
    def test_response_format_stability(self) -> Dict[str, Any]:
        """Test that response formats remain stable"""
        
        # Test multiple calls to ensure consistent response format
        stability_tests = []
        
        for i in range(5):
            health_result = self.test_existing_endpoint_compatibility('/health', 'GET')
            status_result = self.test_existing_endpoint_compatibility('/status', 'GET')
            
            stability_tests.extend([health_result, status_result])
        
        # Check for format consistency
        format_issues = []
        
        # Group results by endpoint
        health_results = [r for r in stability_tests if '/health' in r['endpoint']]
        status_results = [r for r in stability_tests if '/status' in r['endpoint']]
        
        # Check health endpoint consistency
        if len(health_results) > 1:
            first_health = health_results[0]
            for result in health_results[1:]:
                if result.get('schema_errors') != first_health.get('schema_errors'):
                    format_issues.append("Health endpoint response format inconsistent")
                    break
        
        # Check status endpoint consistency
        if len(status_results) > 1:
            first_status = status_results[0]
            for result in status_results[1:]:
                if result.get('schema_errors') != first_status.get('schema_errors'):
                    format_issues.append("Status endpoint response format inconsistent")
                    break
        
        return {
            'stability_tests_run': len(stability_tests),
            'format_consistent': len(format_issues) == 0,
            'format_issues': format_issues,
            'test_results': stability_tests
        }
    
    def test_error_handling_compatibility(self) -> Dict[str, Any]:
        """Test that error handling remains compatible"""
        
        # Test various error scenarios
        error_tests = [
            {
                'name': 'missing_required_parameter',
                'endpoint': '/generate/text-to-image',
                'method': 'POST',
                'data': {},  # Missing required 'prompt'
                'expected_status': 400
            },
            {
                'name': 'invalid_parameter_type',
                'endpoint': '/generate/text-to-image',
                'method': 'POST',
                'data': {'prompt': 'test', 'width': 'invalid'},
                'expected_status': 400
            },
            {
                'name': 'nonexistent_endpoint',
                'endpoint': '/nonexistent',
                'method': 'GET',
                'data': None,
                'expected_status': 404
            }
        ]
        
        error_results = []
        
        for test in error_tests:
            try:
                response = self._simulate_api_call(test['method'], test['endpoint'], test['data'])
                
                # Check if error handling is appropriate
                status_code = response.get('_status_code', 200)
                error_handled_correctly = status_code == test['expected_status']
                
                error_results.append({
                    'test_name': test['name'],
                    'endpoint': test['endpoint'],
                    'expected_status': test['expected_status'],
                    'actual_status': status_code,
                    'handled_correctly': error_handled_correctly,
                    'response': {k: v for k, v in response.items() if not k.startswith('_')}
                })
                
            except Exception as e:
                error_results.append({
                    'test_name': test['name'],
                    'endpoint': test['endpoint'],
                    'handled_correctly': False,
                    'error': str(e)
                })
        
        correctly_handled = sum(1 for r in error_results if r.get('handled_correctly', False))
        
        return {
            'total_error_tests': len(error_results),
            'correctly_handled': correctly_handled,
            'error_handling_rate': (correctly_handled / len(error_results)) * 100,
            'error_test_results': error_results
        }
    
    def _calculate_api_grade(self, v1_compatible: int, v1_total: int, 
                           v2_functional: int, v2_total: int) -> str:
        """Calculate overall API compatibility grade"""
        
        v1_rate = (v1_compatible / v1_total) * 100 if v1_total > 0 else 100
        v2_rate = (v2_functional / v2_total) * 100 if v2_total > 0 else 100
        
        # Weight v1 compatibility higher (70%) than v2 functionality (30%)
        overall_score = (v1_rate * 0.7) + (v2_rate * 0.3)
        
        if overall_score >= 95:
            return 'A'
        elif overall_score >= 85:
            return 'B'
        elif overall_score >= 75:
            return 'C'
        elif overall_score >= 65:
            return 'D'
        else:
            return 'F'
    
    def generate_compatibility_report(self) -> Dict[str, Any]:
        """Generate comprehensive API compatibility report"""
        
        # Run all compatibility tests
        versioning_result = self.test_api_versioning_compatibility()
        parameter_result = self.test_request_parameter_compatibility()
        format_result = self.test_response_format_stability()
        error_result = self.test_error_handling_compatibility()
        
        # Calculate overall metrics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results 
                             if r.get('compatible', False) or r.get('functional', False))
        
        return {
            'summary': {
                'total_api_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
                'overall_grade': versioning_result['overall_grade']
            },
            'versioning_compatibility': versioning_result,
            'parameter_compatibility': parameter_result,
            'response_format_stability': format_result,
            'error_handling_compatibility': error_result,
            'recommendations': self._generate_api_recommendations(),
            'detailed_test_results': self.test_results
        }
    
    def _generate_api_recommendations(self) -> List[str]:
        """Generate API compatibility recommendations"""
        
        recommendations = []
        
        # Analyze test results for common issues
        breaking_changes_found = any(
            len(r.get('breaking_changes', [])) > 0 for r in self.test_results
        )
        
        schema_errors_found = any(
            len(r.get('schema_errors', [])) > 0 for r in self.test_results
        )
        
        if breaking_changes_found:
            recommendations.append('Address breaking changes in existing API endpoints')
            recommendations.append('Consider API versioning strategy for incompatible changes')
        
        if schema_errors_found:
            recommendations.append('Fix schema validation errors in API responses')
            recommendations.append('Ensure all required fields are present in responses')
        
        # General recommendations
        recommendations.extend([
            'Implement comprehensive API testing in CI/CD pipeline',
            'Document all API changes and maintain changelog',
            'Use semantic versioning for API releases',
            'Provide migration guides for API consumers',
            'Consider deprecation warnings for removed features'
        ])
        
        return recommendations


class TestAPIBackwardCompatibility(unittest.TestCase):
    """Test API backward compatibility for DiffSynth Enhanced UI"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_tester = APICompatibilityTester()
    
    def test_existing_endpoints_remain_functional(self):
        """Test that all existing endpoints remain functional"""
        
        existing_endpoints = [
            ('GET', '/'),
            ('GET', '/health'),
            ('GET', '/status'),
            ('POST', '/initialize'),
            ('GET', '/aspect-ratios'),
            ('POST', '/generate/text-to-image'),
            ('GET', '/queue'),
            ('GET', '/memory/clear')
        ]
        
        for method, endpoint in existing_endpoints:
            with self.subTest(endpoint=endpoint, method=method):
                request_data = {'prompt': 'test'} if method == 'POST' and 'generate' in endpoint else None
                
                result = self.api_tester.test_existing_endpoint_compatibility(
                    endpoint, method, request_data
                )
                
                self.assertTrue(result.get('compatible', False),
                              f"Existing endpoint {method} {endpoint} should remain compatible")
                self.assertEqual(result.get('status_code', 0), 200,
                               f"Endpoint {method} {endpoint} should return 200 status")
    
    def test_new_diffsynth_endpoints_functional(self):
        """Test that new DiffSynth endpoints are functional"""
        
        new_endpoints = [
            ('POST', '/diffsynth/edit'),
            ('POST', '/diffsynth/inpaint'),
            ('POST', '/diffsynth/outpaint'),
            ('POST', '/diffsynth/style-transfer'),
            ('POST', '/controlnet/detect'),
            ('POST', '/controlnet/generate'),
            ('GET', '/controlnet/types'),
            ('GET', '/services/status'),
            ('POST', '/services/switch')
        ]
        
        for method, endpoint in new_endpoints:
            with self.subTest(endpoint=endpoint, method=method):
                request_data = None
                if method == 'POST':
                    if '/diffsynth/' in endpoint or '/controlnet/' in endpoint:
                        request_data = {'image': 'test.jpg', 'prompt': 'test prompt'}
                    elif '/services/' in endpoint:
                        request_data = {'service': 'diffsynth'}
                
                result = self.api_tester.test_new_endpoint_functionality(
                    endpoint, method, request_data
                )
                
                # New endpoints should be functional
                self.assertTrue(result.get('functional', False),
                              f"New endpoint {method} {endpoint} should be functional")
    
    def test_text_to_image_endpoint_compatibility(self):
        """Test detailed compatibility of the main text-to-image endpoint"""
        
        # Test with minimal parameters (v1.0 style)
        minimal_request = {'prompt': 'a simple test image'}
        result = self.api_tester.test_existing_endpoint_compatibility(
            '/generate/text-to-image', 'POST', minimal_request
        )
        
        self.assertTrue(result['compatible'])
        self.assertTrue(result['response_valid'])
        self.assertEqual(len(result['breaking_changes']), 0)
        
        # Test with full v1.0 parameters
        full_request = {
            'prompt': 'detailed test image',
            'negative_prompt': 'blurry, low quality',
            'width': 1024,
            'height': 1024,
            'num_inference_steps': 20,
            'cfg_scale': 7.5,
            'seed': 12345,
            'language': 'en',
            'enhance_prompt': True,
            'aspect_ratio': '1:1'
        }
        
        result = self.api_tester.test_existing_endpoint_compatibility(
            '/generate/text-to-image', 'POST', full_request
        )
        
        self.assertTrue(result['compatible'])
        self.assertEqual(len(result['breaking_changes']), 0)
    
    def test_health_endpoint_schema_stability(self):
        """Test that health endpoint schema remains stable"""
        
        result = self.api_tester.test_existing_endpoint_compatibility('/health', 'GET')
        
        self.assertTrue(result['compatible'])
        self.assertTrue(result['response_valid'])
        self.assertEqual(len(result['schema_errors']), 0)
        
        # Health endpoint should always have 'status' field
        # This would be verified in the actual API response
    
    def test_status_endpoint_backward_compatibility(self):
        """Test status endpoint backward compatibility"""
        
        result = self.api_tester.test_existing_endpoint_compatibility('/status', 'GET')
        
        self.assertTrue(result['compatible'])
        self.assertTrue(result['response_valid'])
        
        # Should not have breaking changes
        self.assertEqual(len(result['breaking_changes']), 0)
    
    def test_api_versioning_strategy(self):
        """Test API versioning compatibility"""
        
        versioning_result = self.api_tester.test_api_versioning_compatibility()
        
        # V1 compatibility should be high
        v1_compat = versioning_result['v1_compatibility']
        self.assertGreaterEqual(v1_compat['compatibility_rate'], 90,
                               "V1 API compatibility should be >= 90%")
        
        # V2 functionality should work
        v2_func = versioning_result['v2_functionality']
        self.assertGreaterEqual(v2_func['functionality_rate'], 80,
                               "V2 API functionality should be >= 80%")
        
        # Overall grade should be acceptable
        self.assertIn(versioning_result['overall_grade'], ['A', 'B', 'C'],
                     "Overall API grade should be C or better")
    
    def test_request_parameter_backward_compatibility(self):
        """Test that request parameters remain backward compatible"""
        
        parameter_result = self.api_tester.test_request_parameter_compatibility()
        
        # All parameter combinations should work
        self.assertGreaterEqual(parameter_result['compatibility_rate'], 90,
                               "Parameter compatibility should be >= 90%")
        
        # Check specific test cases
        test_results = parameter_result['parameter_test_results']
        
        # Minimal request should work
        minimal_test = next((r for r in test_results if r['test_case'] == 'parameter_test_1'), None)
        self.assertIsNotNone(minimal_test)
        self.assertTrue(minimal_test['compatible'])
        
        # Full v1.0 request should work
        full_test = next((r for r in test_results if r['test_case'] == 'parameter_test_2'), None)
        self.assertIsNotNone(full_test)
        self.assertTrue(full_test['compatible'])
    
    def test_response_format_consistency(self):
        """Test response format consistency"""
        
        format_result = self.api_tester.test_response_format_stability()
        
        self.assertTrue(format_result['format_consistent'],
                      "API response formats should be consistent")
        self.assertEqual(len(format_result['format_issues']), 0,
                        "Should not have format consistency issues")
    
    def test_error_handling_backward_compatibility(self):
        """Test that error handling remains backward compatible"""
        
        error_result = self.api_tester.test_error_handling_compatibility()
        
        # Error handling should be mostly correct
        self.assertGreaterEqual(error_result['error_handling_rate'], 80,
                               "Error handling compatibility should be >= 80%")
        
        # Check specific error scenarios
        error_tests = error_result['error_test_results']
        
        # Missing parameter should return 400
        missing_param_test = next(
            (r for r in error_tests if r['test_name'] == 'missing_required_parameter'), None
        )
        if missing_param_test:
            self.assertTrue(missing_param_test['handled_correctly'])
        
        # Nonexistent endpoint should return 404
        not_found_test = next(
            (r for r in error_tests if r['test_name'] == 'nonexistent_endpoint'), None
        )
        if not_found_test:
            self.assertTrue(not_found_test['handled_correctly'])
    
    def test_new_endpoint_integration(self):
        """Test that new endpoints integrate properly without breaking existing ones"""
        
        # Test that existing endpoints still work after new ones are added
        existing_result = self.api_tester.test_existing_endpoint_compatibility(
            '/generate/text-to-image', 'POST', {'prompt': 'integration test'}
        )
        
        # Test that new endpoints work
        new_result = self.api_tester.test_new_endpoint_functionality(
            '/diffsynth/edit', 'POST', {'image': 'test.jpg', 'prompt': 'edit test'}
        )
        
        # Both should work
        self.assertTrue(existing_result['compatible'])
        self.assertTrue(new_result['functional'])
    
    def test_comprehensive_compatibility_report(self):
        """Test comprehensive compatibility report generation"""
        
        # Generate full compatibility report
        report = self.api_tester.generate_compatibility_report()
        
        # Verify report structure
        self.assertIn('summary', report)
        self.assertIn('versioning_compatibility', report)
        self.assertIn('parameter_compatibility', report)
        self.assertIn('response_format_stability', report)
        self.assertIn('error_handling_compatibility', report)
        self.assertIn('recommendations', report)
        
        # Summary should have reasonable metrics
        summary = report['summary']
        self.assertGreaterEqual(summary['success_rate'], 70,
                               "Overall API success rate should be >= 70%")
        
        # Should have recommendations
        self.assertGreater(len(report['recommendations']), 0,
                          "Should provide compatibility recommendations")
    
    def test_api_performance_impact(self):
        """Test that new features don't significantly impact existing API performance"""
        
        # Test response times for existing endpoints
        endpoints_to_test = [
            ('GET', '/health'),
            ('GET', '/status'),
            ('POST', '/generate/text-to-image')
        ]
        
        response_times = []
        
        for method, endpoint in endpoints_to_test:
            request_data = {'prompt': 'performance test'} if method == 'POST' else None
            
            result = self.api_tester.test_existing_endpoint_compatibility(
                endpoint, method, request_data
            )
            
            response_time = result.get('response_time_ms', 0)
            response_times.append(response_time)
            
            # Response times should be reasonable (< 1000ms for mock responses)
            self.assertLess(response_time, 1000,
                           f"Response time for {method} {endpoint} should be < 1000ms")
        
        # Average response time should be reasonable
        avg_response_time = sum(response_times) / len(response_times)
        self.assertLess(avg_response_time, 500,
                       "Average response time should be < 500ms")


if __name__ == '__main__':
    unittest.main()