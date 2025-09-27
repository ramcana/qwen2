"""
API Integration Workflow Tests
Tests complete workflows through API endpoints including text-to-image generation,
DiffSynth editing, ControlNet-guided generation, and service management.
"""

import asyncio
import json
import os
import sys
import time
import uuid
import base64
import tempfile
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import threading

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from api_server import app
    from qwen_generator import QwenImageGenerator
    from diffsynth_service import DiffSynthService, DiffSynthConfig, DiffSynthServiceStatus
    from controlnet_service import ControlNetService, ControlNetType
    from service_manager import ServiceManager
    from resource_manager import ResourceManager
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestAPITextToImageWorkflowIntegration:
    """Test complete text-to-image workflow through API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_services(self):
        """Mock all services for API testing"""
        # Create test image
        img = Image.new('RGB', (512, 512), color='red')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        test_image_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        with patch('api_server.QwenImageGenerator') as mock_qwen, \
             patch('api_server.DiffSynthService') as mock_diffsynth, \
             patch('api_server.ControlNetService') as mock_controlnet:
            
            # Setup Qwen generator mock
            qwen_instance = Mock()
            mock_qwen.return_value = qwen_instance
            qwen_instance.generate_image.return_value = {
                'success': True,
                'image_base64': test_image_base64,
                'generation_time': 5.2,
                'seed': 42,
                'metadata': {
                    'prompt': 'test prompt',
                    'width': 512,
                    'height': 512,
                    'steps': 20
                }
            }
            
            # Setup DiffSynth service mock
            diffsynth_instance = Mock()
            mock_diffsynth.return_value = diffsynth_instance
            diffsynth_instance.status = DiffSynthServiceStatus.READY
            diffsynth_instance.edit_image.return_value = (test_image_base64, "Edit successful")
            diffsynth_instance.inpaint.return_value = (test_image_base64, "Inpainting successful")
            diffsynth_instance.outpaint.return_value = (test_image_base64, "Outpainting successful")
            diffsynth_instance.style_transfer.return_value = (test_image_base64, "Style transfer successful")
            
            # Setup ControlNet service mock
            controlnet_instance = Mock()
            mock_controlnet.return_value = controlnet_instance
            controlnet_instance.initialized = True
            controlnet_instance.detect_control_type.return_value = {
                'detected_type': 'canny',
                'confidence': 0.95,
                'preview_base64': test_image_base64
            }
            controlnet_instance.process_with_control.return_value = {
                'success': True,
                'image_base64': test_image_base64,
                'generation_time': 8.5,
                'control_influence': 0.8
            }
            
            yield {
                'qwen': qwen_instance,
                'diffsynth': diffsynth_instance,
                'controlnet': controlnet_instance,
                'test_image_base64': test_image_base64
            }
    
    def test_complete_text_to_image_api_workflow(self, client, mock_services):
        """Test complete text-to-image generation workflow through API"""
        # Step 1: Generate image
        generation_payload = {
            "prompt": "A beautiful landscape with mountains",
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,
            "cfg_scale": 7.5,
            "seed": 42,
            "language": "en"
        }
        
        response = client.post("/generate/text-to-image", json=generation_payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['success'] is True
        assert 'image_base64' in data
        assert data['generation_time'] > 0
        assert data['metadata']['prompt'] == generation_payload['prompt']
        
        # Step 2: Verify service was called with correct parameters
        mock_services['qwen'].generate_image.assert_called_once()
        
        # Step 3: Test image metadata retrieval
        metadata_response = client.get(f"/images/metadata/{data.get('image_id', 'test_id')}")
        # This endpoint might not exist yet, but shows the workflow concept
        
        return data['image_base64']  # Return for chaining tests
    
    def test_text_to_image_with_queue_api_workflow(self, client, mock_services):
        """Test queued text-to-image generation workflow through API"""
        with patch('api_server.QueueManager') as mock_queue_class:
            queue_manager = Mock()
            mock_queue_class.return_value = queue_manager
            
            # Step 1: Queue generation job
            job_id = "job_" + str(uuid.uuid4())
            queue_manager.add_job.return_value = job_id
            
            queue_payload = {
                "prompt": "Queued generation test",
                "width": 512,
                "height": 512,
                "priority": "normal"
            }
            
            response = client.post("/generate/text-to-image/queue", json=queue_payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data['job_id'] == job_id
            assert data['status'] == 'queued'
            
            # Step 2: Check job status
            queue_manager.get_job_status.return_value = {
                'status': 'processing',
                'progress': 0.5,
                'estimated_time_remaining': 15
            }
            
            status_response = client.get(f"/jobs/{job_id}/status")
            
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data['status'] == 'processing'
            assert status_data['progress'] == 0.5
            
            # Step 3: Simulate job completion
            queue_manager.get_job_status.return_value = {
                'status': 'completed',
                'result': mock_services['qwen'].generate_image.return_value
            }
            
            final_status_response = client.get(f"/jobs/{job_id}/status")
            final_status_data = final_status_response.json()
            
            assert final_status_data['status'] == 'completed'
            assert final_status_data['result']['success'] is True
    
    def test_text_to_image_with_parameter_validation_api(self, client):
        """Test text-to-image API with parameter validation"""
        # Test with invalid parameters
        invalid_payload = {
            "prompt": "",  # Empty prompt
            "width": -1,   # Invalid width
            "height": 513, # Invalid height (not multiple of 8)
            "cfg_scale": -1  # Invalid cfg_scale
        }
        
        response = client.post("/generate/text-to-image", json=invalid_payload)
        
        # Should return validation error
        assert response.status_code == 422
        error_data = response.json()
        assert 'detail' in error_data
        
        # Verify specific validation errors
        validation_errors = error_data['detail']
        error_fields = [error['loc'][-1] for error in validation_errors]
        
        assert 'prompt' in error_fields or 'width' in error_fields


class TestAPIDiffSynthWorkflowIntegration:
    """Test complete DiffSynth editing workflow through API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app"""
        return TestClient(app)
    
    @pytest.fixture
    def test_images(self):
        """Create test images for editing workflows"""
        # Original image
        original_img = Image.new('RGB', (512, 512), color='blue')
        original_buffer = io.BytesIO()
        original_img.save(original_buffer, format='PNG')
        original_base64 = base64.b64encode(original_buffer.getvalue()).decode()
        
        # Mask image
        mask_img = Image.new('L', (512, 512), color=0)
        mask_img.paste(255, (100, 100, 400, 400))  # White region to inpaint
        mask_buffer = io.BytesIO()
        mask_img.save(mask_buffer, format='PNG')
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
        
        # Style image
        style_img = Image.new('RGB', (512, 512), color='purple')
        style_buffer = io.BytesIO()
        style_img.save(style_buffer, format='PNG')
        style_base64 = base64.b64encode(style_buffer.getvalue()).decode()
        
        return {
            'original': original_base64,
            'mask': mask_base64,
            'style': style_base64
        }
    
    def test_complete_image_editing_api_workflow(self, client, test_images):
        """Test complete image editing workflow through API"""
        with patch('api_server.DiffSynthService') as mock_service_class:
            service_instance = Mock()
            mock_service_class.return_value = service_instance
            service_instance.status = DiffSynthServiceStatus.READY
            service_instance.edit_image.return_value = (test_images['original'], "Edit successful")
            
            # Step 1: Basic image editing
            edit_payload = {
                "prompt": "Add a beautiful sunset to this landscape",
                "image_base64": test_images['original'],
                "strength": 0.7,
                "guidance_scale": 7.5,
                "num_inference_steps": 20
            }
            
            response = client.post("/diffsynth/edit", json=edit_payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data['success'] is True
            assert 'image_base64' in data
            assert data['message'] == "Edit successful"
            assert 'processing_time' in data
            
            # Verify service was called
            service_instance.edit_image.assert_called_once()
    
    def test_complete_inpainting_api_workflow(self, client, test_images):
        """Test complete inpainting workflow through API"""
        with patch('api_server.DiffSynthService') as mock_service_class:
            service_instance = Mock()
            mock_service_class.return_value = service_instance
            service_instance.status = DiffSynthServiceStatus.READY
            service_instance.inpaint.return_value = (test_images['original'], "Inpainting successful")
            
            # Step 1: Inpainting request
            inpaint_payload = {
                "prompt": "A beautiful flower garden",
                "image_base64": test_images['original'],
                "mask_base64": test_images['mask'],
                "strength": 0.9,
                "guidance_scale": 8.0
            }
            
            response = client.post("/diffsynth/inpaint", json=inpaint_payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data['success'] is True
            assert 'image_base64' in data
            assert data['message'] == "Inpainting successful"
            
            # Verify service was called
            service_instance.inpaint.assert_called_once()
    
    def test_complete_outpainting_api_workflow(self, client, test_images):
        """Test complete outpainting workflow through API"""
        with patch('api_server.DiffSynthService') as mock_service_class:
            service_instance = Mock()
            mock_service_class.return_value = service_instance
            service_instance.status = DiffSynthServiceStatus.READY
            service_instance.outpaint.return_value = (test_images['original'], "Outpainting successful")
            
            # Step 1: Outpainting request
            outpaint_payload = {
                "prompt": "Extend this landscape to show more of the horizon",
                "image_base64": test_images['original'],
                "direction": "right",
                "pixels": 256,
                "guidance_scale": 7.0
            }
            
            response = client.post("/diffsynth/outpaint", json=outpaint_payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data['success'] is True
            assert 'image_base64' in data
            assert data['message'] == "Outpainting successful"
            
            # Verify service was called
            service_instance.outpaint.assert_called_once()
    
    def test_complete_style_transfer_api_workflow(self, client, test_images):
        """Test complete style transfer workflow through API"""
        with patch('api_server.DiffSynthService') as mock_service_class:
            service_instance = Mock()
            mock_service_class.return_value = service_instance
            service_instance.status = DiffSynthServiceStatus.READY
            service_instance.style_transfer.return_value = (test_images['original'], "Style transfer successful")
            
            # Step 1: Style transfer request
            style_payload = {
                "prompt": "Apply impressionist painting style",
                "content_image_base64": test_images['original'],
                "style_image_base64": test_images['style'],
                "style_strength": 0.8,
                "content_strength": 0.6
            }
            
            response = client.post("/diffsynth/style-transfer", json=style_payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data['success'] is True
            assert 'image_base64' in data
            assert data['message'] == "Style transfer successful"
            
            # Verify service was called
            service_instance.style_transfer.assert_called_once()


class TestAPIControlNetWorkflowIntegration:
    """Test complete ControlNet workflow through API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app"""
        return TestClient(app)
    
    @pytest.fixture
    def control_image(self):
        """Create test control image"""
        img = Image.new('RGB', (512, 512), color='white')
        # Add edge-like features
        for i in range(100, 400):
            img.putpixel((i, 100), (0, 0, 0))  # Top edge
            img.putpixel((i, 400), (0, 0, 0))  # Bottom edge
            img.putpixel((100, i), (0, 0, 0))  # Left edge
            img.putpixel((400, i), (0, 0, 0))  # Right edge
        
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    
    def test_complete_controlnet_detection_api_workflow(self, client, control_image):
        """Test complete ControlNet detection workflow through API"""
        with patch('api_server.ControlNetService') as mock_service_class:
            service_instance = Mock()
            mock_service_class.return_value = service_instance
            service_instance.initialized = True
            service_instance.detect_control_type.return_value = {
                'detected_type': 'canny',
                'confidence': 0.95,
                'preview_base64': control_image,
                'detection_time': 1.2
            }
            
            # Step 1: Control type detection
            detection_payload = {
                "image_base64": control_image
            }
            
            response = client.post("/controlnet/detect", json=detection_payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data['detected_type'] == 'canny'
            assert data['confidence'] > 0.9
            assert 'preview_base64' in data
            assert data['detection_time'] > 0
            
            # Verify service was called
            service_instance.detect_control_type.assert_called_once()
    
    def test_complete_controlnet_generation_api_workflow(self, client, control_image):
        """Test complete ControlNet generation workflow through API"""
        with patch('api_server.ControlNetService') as mock_service_class:
            service_instance = Mock()
            mock_service_class.return_value = service_instance
            service_instance.initialized = True
            service_instance.process_with_control.return_value = {
                'success': True,
                'image_base64': control_image,
                'generation_time': 12.5,
                'control_influence': 0.85,
                'parameters': {
                    'controlnet_conditioning_scale': 1.0,
                    'control_guidance_start': 0.0,
                    'control_guidance_end': 1.0
                }
            }
            
            # Step 1: ControlNet generation
            generation_payload = {
                "prompt": "A beautiful architectural drawing",
                "control_image_base64": control_image,
                "control_type": "canny",
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "width": 512,
                "height": 512,
                "num_inference_steps": 20,
                "guidance_scale": 7.5
            }
            
            response = client.post("/controlnet/generate", json=generation_payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data['success'] is True
            assert 'image_base64' in data
            assert data['generation_time'] > 0
            assert data['control_influence'] > 0.8
            assert 'parameters' in data
            
            # Verify service was called
            service_instance.process_with_control.assert_called_once()
    
    def test_controlnet_types_api_endpoint(self, client):
        """Test ControlNet types listing endpoint"""
        response = client.get("/controlnet/types")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'available_types' in data
        assert isinstance(data['available_types'], list)
        assert 'canny' in data['available_types']
        assert 'depth' in data['available_types']
        assert 'pose' in data['available_types']


class TestAPIServiceManagementWorkflowIntegration:
    """Test service management workflow through API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app"""
        return TestClient(app)
    
    def test_complete_service_status_api_workflow(self, client):
        """Test complete service status monitoring workflow through API"""
        with patch('api_server.ServiceManager') as mock_manager_class:
            manager_instance = Mock()
            mock_manager_class.return_value = manager_instance
            
            manager_instance.get_service_status.return_value = {
                'qwen_generator': {
                    'status': 'healthy',
                    'response_time': 0.1,
                    'memory_usage': 2.5,
                    'last_activity': time.time()
                },
                'diffsynth_service': {
                    'status': 'healthy',
                    'response_time': 0.2,
                    'memory_usage': 3.2,
                    'last_activity': time.time()
                },
                'controlnet_service': {
                    'status': 'healthy',
                    'response_time': 0.15,
                    'memory_usage': 1.8,
                    'last_activity': time.time()
                }
            }
            
            # Step 1: Get service status
            response = client.get("/services/status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert 'qwen_generator' in data
            assert 'diffsynth_service' in data
            assert 'controlnet_service' in data
            
            # Verify service details
            for service_name, service_data in data.items():
                assert service_data['status'] == 'healthy'
                assert service_data['response_time'] > 0
                assert service_data['memory_usage'] > 0
                assert 'last_activity' in service_data
            
            # Verify manager was called
            manager_instance.get_service_status.assert_called_once()
    
    def test_complete_service_switching_api_workflow(self, client):
        """Test complete service switching workflow through API"""
        with patch('api_server.ServiceManager') as mock_manager_class:
            manager_instance = Mock()
            mock_manager_class.return_value = manager_instance
            
            manager_instance.switch_service.return_value = {
                'success': True,
                'previous_service': 'qwen_generator',
                'new_service': 'diffsynth_service',
                'switch_time': 2.3,
                'resource_reallocation': {
                    'freed_memory': 2.5,
                    'allocated_memory': 3.2
                }
            }
            
            # Step 1: Switch service
            switch_payload = {
                "target_service": "diffsynth_service",
                "priority": "high",
                "force_switch": False
            }
            
            response = client.post("/services/switch", json=switch_payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data['success'] is True
            assert data['new_service'] == 'diffsynth_service'
            assert data['switch_time'] > 0
            assert 'resource_reallocation' in data
            
            # Verify manager was called
            manager_instance.switch_service.assert_called_once()
    
    def test_memory_management_api_workflow(self, client):
        """Test memory management workflow through API"""
        with patch('api_server.get_resource_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager
            
            # Step 1: Get memory status
            mock_manager.get_memory_status.return_value = {
                'total_memory_gb': 16.0,
                'allocated_memory_gb': 8.5,
                'available_memory_gb': 7.5,
                'usage_percentage': 53.1,
                'services': {
                    'qwen_generator': 2.5,
                    'diffsynth_service': 3.2,
                    'controlnet_service': 1.8
                },
                'fragmentation_level': 0.15
            }
            
            response = client.get("/memory/status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data['total_memory_gb'] == 16.0
            assert data['allocated_memory_gb'] == 8.5
            assert data['usage_percentage'] > 50
            assert 'services' in data
            assert 'fragmentation_level' in data
            
            # Step 2: Clear unused memory
            mock_manager.clear_unused_memory.return_value = {
                'success': True,
                'freed_memory_gb': 2.3,
                'cleanup_actions': ['cache_clear', 'garbage_collect'],
                'cleanup_time': 1.8
            }
            
            clear_response = client.post("/memory/clear")
            
            assert clear_response.status_code == 200
            clear_data = clear_response.json()
            
            assert clear_data['success'] is True
            assert clear_data['freed_memory_gb'] > 0
            assert 'cleanup_actions' in clear_data
            
            # Verify manager methods were called
            mock_manager.get_memory_status.assert_called_once()
            mock_manager.clear_unused_memory.assert_called_once()


class TestAPICompleteWorkflowIntegration:
    """Test complete workflows combining multiple API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_all_services(self):
        """Mock all services for complete workflow testing"""
        # Create test image
        img = Image.new('RGB', (512, 512), color='red')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        test_image_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        with patch('api_server.QwenImageGenerator') as mock_qwen, \
             patch('api_server.DiffSynthService') as mock_diffsynth, \
             patch('api_server.ControlNetService') as mock_controlnet, \
             patch('api_server.ServiceManager') as mock_service_manager:
            
            # Setup all service mocks
            qwen_instance = Mock()
            mock_qwen.return_value = qwen_instance
            qwen_instance.generate_image.return_value = {
                'success': True,
                'image_base64': test_image_base64,
                'generation_time': 5.0
            }
            
            diffsynth_instance = Mock()
            mock_diffsynth.return_value = diffsynth_instance
            diffsynth_instance.status = DiffSynthServiceStatus.READY
            diffsynth_instance.edit_image.return_value = (test_image_base64, "Edit successful")
            
            controlnet_instance = Mock()
            mock_controlnet.return_value = controlnet_instance
            controlnet_instance.initialized = True
            controlnet_instance.process_with_control.return_value = {
                'success': True,
                'image_base64': test_image_base64,
                'generation_time': 8.0
            }
            
            service_manager_instance = Mock()
            mock_service_manager.return_value = service_manager_instance
            service_manager_instance.switch_service.return_value = {
                'success': True,
                'new_service': 'target_service'
            }
            
            yield {
                'qwen': qwen_instance,
                'diffsynth': diffsynth_instance,
                'controlnet': controlnet_instance,
                'service_manager': service_manager_instance,
                'test_image_base64': test_image_base64
            }
    
    def test_complete_generation_to_editing_api_workflow(self, client, mock_all_services):
        """Test complete workflow: Generate -> Edit -> Enhance through API"""
        # Step 1: Generate initial image
        generation_payload = {
            "prompt": "A beautiful mountain landscape",
            "width": 512,
            "height": 512,
            "num_inference_steps": 20
        }
        
        gen_response = client.post("/generate/text-to-image", json=generation_payload)
        assert gen_response.status_code == 200
        gen_data = gen_response.json()
        assert gen_data['success'] is True
        
        generated_image = gen_data['image_base64']
        
        # Step 2: Edit the generated image
        edit_payload = {
            "prompt": "Add a sunset to the mountain landscape",
            "image_base64": generated_image,
            "strength": 0.7
        }
        
        edit_response = client.post("/diffsynth/edit", json=edit_payload)
        assert edit_response.status_code == 200
        edit_data = edit_response.json()
        assert edit_data['success'] is True
        
        edited_image = edit_data['image_base64']
        
        # Step 3: Enhance with ControlNet
        controlnet_payload = {
            "prompt": "Enhance the composition and lighting",
            "control_image_base64": edited_image,
            "control_type": "canny",
            "controlnet_conditioning_scale": 0.8
        }
        
        controlnet_response = client.post("/controlnet/generate", json=controlnet_payload)
        assert controlnet_response.status_code == 200
        controlnet_data = controlnet_response.json()
        assert controlnet_data['success'] is True
        
        # Step 4: Verify complete workflow
        workflow_summary = {
            'steps_completed': 3,
            'services_used': ['qwen_generator', 'diffsynth_service', 'controlnet_service'],
            'final_image': controlnet_data['image_base64'],
            'total_processing_time': (
                gen_data.get('generation_time', 0) +
                edit_data.get('processing_time', 0) +
                controlnet_data.get('generation_time', 0)
            )
        }
        
        assert workflow_summary['steps_completed'] == 3
        assert len(workflow_summary['services_used']) == 3
        assert workflow_summary['final_image'] is not None
        
        # Verify all services were called
        mock_all_services['qwen'].generate_image.assert_called_once()
        mock_all_services['diffsynth'].edit_image.assert_called_once()
        mock_all_services['controlnet'].process_with_control.assert_called_once()
    
    def test_api_workflow_with_service_switching(self, client, mock_all_services):
        """Test API workflow with automatic service switching"""
        # Step 1: Check initial service status
        status_response = client.get("/services/status")
        assert status_response.status_code == 200
        
        # Step 2: Switch to DiffSynth service
        switch_payload = {
            "target_service": "diffsynth_service",
            "priority": "high"
        }
        
        switch_response = client.post("/services/switch", json=switch_payload)
        assert switch_response.status_code == 200
        switch_data = switch_response.json()
        assert switch_data['success'] is True
        
        # Step 3: Perform editing operation
        edit_payload = {
            "prompt": "Edit with switched service",
            "image_base64": mock_all_services['test_image_base64']
        }
        
        edit_response = client.post("/diffsynth/edit", json=edit_payload)
        assert edit_response.status_code == 200
        edit_data = edit_response.json()
        assert edit_data['success'] is True
        
        # Verify service switching occurred
        mock_all_services['service_manager'].switch_service.assert_called_once()
        mock_all_services['diffsynth'].edit_image.assert_called_once()
    
    def test_api_workflow_with_error_handling(self, client, mock_all_services):
        """Test API workflow with error handling and recovery"""
        # Step 1: Simulate service failure
        mock_all_services['diffsynth'].edit_image.return_value = (None, "Service temporarily unavailable")
        
        edit_payload = {
            "prompt": "Test error handling",
            "image_base64": mock_all_services['test_image_base64']
        }
        
        edit_response = client.post("/diffsynth/edit", json=edit_payload)
        assert edit_response.status_code == 200
        edit_data = edit_response.json()
        
        # Should handle error gracefully
        assert edit_data['success'] is False
        assert 'unavailable' in edit_data['message'].lower()
        
        # Step 2: Check if fallback mechanisms are suggested
        if 'suggested_actions' in edit_data:
            assert isinstance(edit_data['suggested_actions'], list)
            assert len(edit_data['suggested_actions']) > 0


if __name__ == "__main__":
    # Configure test execution
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--maxfail=3"
    ])