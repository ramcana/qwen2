"""
API Workflow Integration Tests
Tests the FastAPI endpoints for complete workflows including DiffSynth and ControlNet
"""

import asyncio
import json
import os
import sys
import time
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock
import pytest
from fastapi.testclient import TestClient
import base64
from PIL import Image
import io

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from api_server import app
    from qwen_generator import QwenImageGenerator
    from diffsynth_service import DiffSynthService, DiffSynthConfig
    from controlnet_service import ControlNetService, ControlNetType
    from service_manager import ServiceManager
    from resource_manager import ResourceManager
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestAPITextToImageWorkflow:
    """Test text-to-image generation through API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_qwen_generator(self):
        """Mock Qwen generator for API testing"""
        with patch('api_server.QwenImageGenerator') as mock_gen:
            generator_instance = Mock()
            mock_gen.return_value = generator_instance
            
            # Create test image
            img = Image.new('RGB', (512, 512), color='red')
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            test_image_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            generator_instance.generate_image.return_value = {
                'success': True,
                'image_base64': test_image_base64,
                'generation_time': 5.2,
                'seed': 42,
                'metadata': {
                    'prompt': 'test prompt',
                    'width': 512,
                    'height': 512
                }
            }
            
            yield generator_instance
    
    def test_text_to_image_api_endpoint(self, client, mock_qwen_generator):
        """Test /generate/text-to-image endpoint"""
        payload = {
            "prompt": "A beautiful landscape with mountains",
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,
            "cfg_scale": 7.5,
            "seed": 42,
            "language": "en"
        }
        
        response = client.post("/generate/text-to-image", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['success'] is True
        assert 'image_base64' in data
        assert data['generation_time'] > 0
        assert data['metadata']['prompt'] == payload['prompt']
        
        # Verify generator was called
        mock_qwen_generator.generate_image.assert_called_once()
    
    def test_text_to_image_with_invalid_parameters(self, client):
        """Test text-to-image endpoint with invalid parameters"""
        payload = {
            "prompt": "",  # Empty prompt
            "width": -1,   # Invalid width
            "height": 513  # Invalid height (not multiple of 8)
        }
        
        response = client.post("/generate/text-to-image", json=payload)
        
        # Should return validation error
        assert response.status_code == 422
        error_data = response.json()
        assert 'detail' in error_data
    
    def test_text_to_image_queue_endpoint(self, client, mock_qwen_generator):
        """Test queued text-to-image generation"""
        with patch('api_server.QueueManager') as mock_queue:
            queue_instance = Mock()
            mock_queue.return_value = queue_instance
            queue_instance.add_job.return_value = "job_123"
            
            payload = {
                "prompt": "Queued generation test",
                "width": 512,
                "height": 512
            }
            
            response = client.post("/generate/text-to-image/queue", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data['job_id'] == "job_123"
            assert data['status'] == 'queued'
    
    def test_job_status_endpoint(self, client):
        """Test job status checking endpoint"""
        with patch('api_server.QueueManager') as mock_queue:
            queue_instance = Mock()
            mock_queue.return_value = queue_instance
            queue_instance.get_job_status.return_value = {
                'status': 'completed',
                'result': {
                    'success': True,
                    'image_base64': 'test_image_data'
                }
            }
            
            response = client.get("/jobs/job_123/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'completed'
            assert data['result']['success'] is True


class TestAPIDiffSynthWorkflow:
    """Test DiffSynth editing through API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_diffsynth_service(self):
        """Mock DiffSynth service for API testing"""
        with patch('api_server.DiffSynthService') as mock_service:
            service_instance = Mock()
            mock_service.return_value = service_instance
            
            # Create test image
            img = Image.new('RGB', (512, 512), color='blue')
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            test_image_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            service_instance.edit_image.return_value = (test_image_base64, "Edit successful")
            service_instance.inpaint.return_value = (test_image_base64, "Inpainting successful")
            service_instance.outpaint.return_value = (test_image_base64, "Outpainting successful")
            service_instance.style_transfer.return_value = (test_image_base64, "Style transfer successful")
            
            yield service_instance
    
    @pytest.fixture
    def test_image_base64(self):
        """Create test image in base64 format"""
        img = Image.new('RGB', (512, 512), color='green')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        return base64.b64encode(img_buffer.getvalue()).decode()
    
    def test_diffsynth_edit_endpoint(self, client, mock_diffsynth_service, test_image_base64):
        """Test /diffsynth/edit endpoint"""
        payload = {
            "prompt": "Add a sunset to this landscape",
            "image_base64": test_image_base64,
            "strength": 0.7,
            "guidance_scale": 7.5
        }
        
        response = client.post("/diffsynth/edit", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['success'] is True
        assert 'image_base64' in data
        assert data['message'] == "Edit successful"
        
        # Verify service was called
        mock_diffsynth_service.edit_image.assert_called_once()
    
    def test_diffsynth_inpaint_endpoint(self, client, mock_diffsynth_service, test_image_base64):
        """Test /diffsynth/inpaint endpoint"""
        # Create mask
        mask_img = Image.new('L', (512, 512), color=0)
        mask_buffer = io.BytesIO()
        mask_img.save(mask_buffer, format='PNG')
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
        
        payload = {
            "prompt": "A beautiful flower",
            "image_base64": test_image_base64,
            "mask_base64": mask_base64,
            "strength": 0.8
        }
        
        response = client.post("/diffsynth/inpaint", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['success'] is True
        assert 'image_base64' in data
        assert data['message'] == "Inpainting successful"
        
        mock_diffsynth_service.inpaint.assert_called_once()
    
    def test_diffsynth_outpaint_endpoint(self, client, mock_diffsynth_service, test_image_base64):
        """Test /diffsynth/outpaint endpoint"""
        payload = {
            "prompt": "Extend the landscape",
            "image_base64": test_image_base64,
            "direction": "right",
            "pixels": 256
        }
        
        response = client.post("/diffsynth/outpaint", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['success'] is True
        assert 'image_base64' in data
        assert data['message'] == "Outpainting successful"
        
        mock_diffsynth_service.outpaint.assert_called_once()
    
    def test_diffsynth_style_transfer_endpoint(self, client, mock_diffsynth_service, test_image_base64):
        """Test /diffsynth/style-transfer endpoint"""
        # Create style image
        style_img = Image.new('RGB', (512, 512), color='purple')
        style_buffer = io.BytesIO()
        style_img.save(style_buffer, format='PNG')
        style_base64 = base64.b64encode(style_buffer.getvalue()).decode()
        
        payload = {
            "prompt": "Apply artistic style",
            "content_image_base64": test_image_base64,
            "style_image_base64": style_base64,
            "style_strength": 0.6
        }
        
        response = client.post("/diffsynth/style-transfer", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['success'] is True
        assert 'image_base64' in data
        assert data['message'] == "Style transfer successful"
        
        mock_diffsynth_service.style_transfer.assert_called_once()


class TestAPIControlNetWorkflow:
    """Test ControlNet functionality through API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_controlnet_service(self):
        """Mock ControlNet service for API testing"""
        with patch('api_server.ControlNetService') as mock_service:
            service_instance = Mock()
            mock_service.return_value = service_instance
            
            # Create test image
            img = Image.new('RGB', (512, 512), color='yellow')
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            test_image_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            service_instance.detect_control_type.return_value = {
                'detected_type': 'canny',
                'confidence': 0.95,
                'preview_base64': test_image_base64
            }
            
            service_instance.generate_control_map.return_value = {
                'control_map_base64': test_image_base64,
                'control_type': 'canny',
                'processing_time': 1.2
            }
            
            service_instance.process_with_control.return_value = {
                'success': True,
                'image_base64': test_image_base64,
                'generation_time': 8.5,
                'control_influence': 0.8
            }
            
            yield service_instance
    
    @pytest.fixture
    def test_control_image(self):
        """Create test control image"""
        img = Image.new('RGB', (512, 512), color='white')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        return base64.b64encode(img_buffer.getvalue()).decode()
    
    def test_controlnet_detect_endpoint(self, client, mock_controlnet_service, test_control_image):
        """Test /controlnet/detect endpoint"""
        payload = {
            "image_base64": test_control_image
        }
        
        response = client.post("/controlnet/detect", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['detected_type'] == 'canny'
        assert data['confidence'] > 0.9
        assert 'preview_base64' in data
        
        mock_controlnet_service.detect_control_type.assert_called_once()
    
    def test_controlnet_generate_endpoint(self, client, mock_controlnet_service, test_control_image):
        """Test /controlnet/generate endpoint"""
        payload = {
            "prompt": "A beautiful portrait",
            "control_image_base64": test_control_image,
            "control_type": "canny",
            "controlnet_conditioning_scale": 1.0,
            "width": 512,
            "height": 512
        }
        
        response = client.post("/controlnet/generate", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['success'] is True
        assert 'image_base64' in data
        assert data['generation_time'] > 0
        assert data['control_influence'] > 0
        
        mock_controlnet_service.process_with_control.assert_called_once()
    
    def test_controlnet_types_endpoint(self, client):
        """Test /controlnet/types endpoint"""
        response = client.get("/controlnet/types")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'available_types' in data
        assert isinstance(data['available_types'], list)
        assert 'canny' in data['available_types']
        assert 'depth' in data['available_types']


class TestAPIServiceManagement:
    """Test service management endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_service_manager(self):
        """Mock service manager for API testing"""
        with patch('api_server.ServiceManager') as mock_manager:
            manager_instance = Mock()
            mock_manager.return_value = manager_instance
            
            manager_instance.get_service_status.return_value = {
                'qwen_generator': {
                    'status': 'healthy',
                    'response_time': 0.1,
                    'memory_usage': 2.5
                },
                'diffsynth_service': {
                    'status': 'healthy',
                    'response_time': 0.2,
                    'memory_usage': 3.2
                },
                'controlnet_service': {
                    'status': 'healthy',
                    'response_time': 0.15,
                    'memory_usage': 1.8
                }
            }
            
            manager_instance.switch_service.return_value = {
                'success': True,
                'message': 'Service switched successfully',
                'active_service': 'diffsynth_service'
            }
            
            yield manager_instance
    
    def test_services_status_endpoint(self, client, mock_service_manager):
        """Test /services/status endpoint"""
        response = client.get("/services/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'qwen_generator' in data
        assert 'diffsynth_service' in data
        assert 'controlnet_service' in data
        
        # Check service details
        qwen_status = data['qwen_generator']
        assert qwen_status['status'] == 'healthy'
        assert qwen_status['response_time'] > 0
        assert qwen_status['memory_usage'] > 0
        
        mock_service_manager.get_service_status.assert_called_once()
    
    def test_services_switch_endpoint(self, client, mock_service_manager):
        """Test /services/switch endpoint"""
        payload = {
            "target_service": "diffsynth_service",
            "priority": "high"
        }
        
        response = client.post("/services/switch", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['success'] is True
        assert data['message'] == 'Service switched successfully'
        assert data['active_service'] == 'diffsynth_service'
        
        mock_service_manager.switch_service.assert_called_once()
    
    def test_memory_status_endpoint(self, client):
        """Test /memory/status endpoint"""
        with patch('api_server.get_resource_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager
            
            mock_manager.get_memory_status.return_value = {
                'total_memory_gb': 16.0,
                'allocated_memory_gb': 8.5,
                'available_memory_gb': 7.5,
                'usage_percentage': 53.1,
                'services': {
                    'qwen_generator': 2.5,
                    'diffsynth_service': 3.2,
                    'controlnet_service': 1.8
                }
            }
            
            response = client.get("/memory/status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data['total_memory_gb'] == 16.0
            assert data['allocated_memory_gb'] == 8.5
            assert data['usage_percentage'] > 50
            assert 'services' in data
    
    def test_memory_clear_endpoint(self, client):
        """Test /memory/clear endpoint"""
        with patch('api_server.get_resource_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager
            
            mock_manager.clear_unused_memory.return_value = {
                'success': True,
                'freed_memory_gb': 2.3,
                'message': 'Memory cleared successfully'
            }
            
            response = client.post("/memory/clear")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data['success'] is True
            assert data['freed_memory_gb'] > 0
            assert 'message' in data


class TestAPIErrorHandling:
    """Test API error handling and fallback mechanisms"""
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app"""
        return TestClient(app)
    
    def test_service_unavailable_fallback(self, client):
        """Test fallback when DiffSynth service is unavailable"""
        with patch('api_server.DiffSynthService') as mock_service:
            service_instance = Mock()
            mock_service.return_value = service_instance
            service_instance.edit_image.return_value = (None, "DiffSynth service unavailable")
            
            payload = {
                "prompt": "Edit this image",
                "image_base64": "test_image_data"
            }
            
            response = client.post("/diffsynth/edit", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data['success'] is False
            assert 'fallback' in data['message'].lower() or 'unavailable' in data['message'].lower()
    
    def test_invalid_image_data_handling(self, client):
        """Test handling of invalid image data"""
        payload = {
            "prompt": "Edit this image",
            "image_base64": "invalid_base64_data"
        }
        
        response = client.post("/diffsynth/edit", json=payload)
        
        # Should handle gracefully
        assert response.status_code in [400, 422, 200]  # Various valid error responses
    
    def test_memory_exhaustion_handling(self, client):
        """Test handling of memory exhaustion scenarios"""
        with patch('api_server.QwenImageGenerator') as mock_gen:
            generator_instance = Mock()
            mock_gen.return_value = generator_instance
            generator_instance.generate_image.return_value = {
                'success': False,
                'message': 'GPU memory insufficient',
                'error_type': 'memory_error'
            }
            
            payload = {
                "prompt": "Generate large image",
                "width": 2048,
                "height": 2048
            }
            
            response = client.post("/generate/text-to-image", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data['success'] is False
            assert 'memory' in data['message'].lower()
    
    def test_concurrent_request_handling(self, client):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            payload = {
                "prompt": "Concurrent test",
                "width": 512,
                "height": 512
            }
            
            response = client.post("/generate/text-to-image", json=payload)
            results.append({
                'status_code': response.status_code,
                'success': response.status_code == 200
            })
        
        # Start multiple concurrent requests
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all requests were handled
        assert len(results) == 3
        # At least some should succeed or be properly queued
        handled_properly = sum(1 for r in results if r['status_code'] in [200, 202])
        assert handled_properly >= 1


if __name__ == "__main__":
    # Run the API integration tests
    pytest.main([__file__, "-v", "--tb=short"])