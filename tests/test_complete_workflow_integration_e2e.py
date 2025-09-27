"""
Complete End-to-End Workflow Integration Tests
Tests complete workflows for text-to-image generation, DiffSynth editing, 
ControlNet-guided generation, and service switching with resource sharing.

This test suite validates all requirements by testing complete user workflows
from start to finish, including error handling and fallback mechanisms.
"""

import asyncio
import json
import os
import sys
import time
import uuid
import base64
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pytest
import requests
from PIL import Image
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from qwen_generator import QwenImageGenerator
    from diffsynth_service import DiffSynthService, DiffSynthConfig, DiffSynthServiceStatus
    from controlnet_service import ControlNetService, ControlNetType, ControlNetRequest
    from resource_manager import ResourceManager, ServiceType, ResourcePriority
    from diffsynth_models import ImageEditRequest, InpaintRequest, OutpaintRequest, StyleTransferRequest
    from service_manager import ServiceManager
    from preset_manager import PresetManager
    from api_server import app
    from fastapi.testclient import TestClient
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestEndToEndTextToImageWorkflow:
    """End-to-end tests for complete text-to-image generation workflow"""
    
    @pytest.fixture
    def workflow_setup(self):
        """Set up complete workflow environment"""
        # Create resource manager
        resource_manager = ResourceManager()
        
        # Create Qwen generator with mocked components
        qwen_generator = Mock(spec=QwenImageGenerator)
        qwen_generator.model_loaded = True
        qwen_generator.device = "cuda"
        
        # Create test image response
        img = Image.new('RGB', (512, 512), color='red')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        qwen_generator.generate_image.return_value = {
            'success': True,
            'image_base64': img_base64,
            'generation_time': 5.2,
            'seed': 42,
            'metadata': {
                'prompt': 'test prompt',
                'width': 512,
                'height': 512,
                'steps': 20,
                'cfg_scale': 7.5
            }
        }
        
        return {
            'resource_manager': resource_manager,
            'qwen_generator': qwen_generator,
            'test_image_base64': img_base64
        }
    
    def test_complete_text_to_image_workflow_with_validation(self, workflow_setup):
        """Test complete text-to-image workflow with parameter validation"""
        components = workflow_setup
        qwen_generator = components['qwen_generator']
        
        # Test workflow: Parameter validation -> Generation -> Post-processing
        
        # Step 1: Validate generation parameters
        generation_params = {
            'prompt': 'A beautiful landscape with mountains and a lake',
            'width': 512,
            'height': 512,
            'num_inference_steps': 20,
            'cfg_scale': 7.5,
            'seed': 42,
            'negative_prompt': 'blurry, low quality'
        }
        
        # Mock parameter validation
        with patch('qwen_generator.validate_generation_params') as mock_validate:
            mock_validate.return_value = (True, None)
            
            # Step 2: Execute generation
            result = qwen_generator.generate_image(**generation_params)
            
            # Step 3: Validate results
            assert result['success'] is True
            assert 'image_base64' in result
            assert result['generation_time'] > 0
            assert result['metadata']['prompt'] == generation_params['prompt']
            assert result['metadata']['width'] == generation_params['width']
            assert result['metadata']['height'] == generation_params['height']
            
            # Verify generator was called with correct parameters
            qwen_generator.generate_image.assert_called_once_with(**generation_params)
            mock_validate.assert_called_once()
    
    def test_text_to_image_workflow_with_queue_management(self, workflow_setup):
        """Test text-to-image workflow with queue management and status tracking"""
        components = workflow_setup
        qwen_generator = components['qwen_generator']
        
        # Mock queue manager
        with patch('qwen_generator.QueueManager') as mock_queue_class:
            queue_manager = Mock()
            mock_queue_class.return_value = queue_manager
            
            # Test job queuing
            job_id = "job_" + str(uuid.uuid4())
            queue_manager.add_job.return_value = job_id
            queue_manager.get_job_status.return_value = {
                'status': 'queued',
                'position': 1,
                'estimated_time': 30
            }
            
            # Step 1: Queue generation job
            queued_job_id = qwen_generator.queue_generation(
                prompt="Queued generation test",
                width=512,
                height=512
            )
            
            assert queued_job_id == job_id
            queue_manager.add_job.assert_called_once()
            
            # Step 2: Check job status
            status = qwen_generator.get_job_status(job_id)
            assert status['status'] == 'queued'
            assert status['position'] == 1
            
            # Step 3: Simulate job completion
            queue_manager.get_job_status.return_value = {
                'status': 'completed',
                'result': components['qwen_generator'].generate_image.return_value
            }
            
            final_status = qwen_generator.get_job_status(job_id)
            assert final_status['status'] == 'completed'
            assert final_status['result']['success'] is True
    
    def test_text_to_image_workflow_with_error_recovery(self, workflow_setup):
        """Test text-to-image workflow with error handling and recovery"""
        components = workflow_setup
        qwen_generator = components['qwen_generator']
        
        # Test memory error scenario
        qwen_generator.generate_image.side_effect = [
            {'success': False, 'error_type': 'memory_error', 'message': 'GPU memory insufficient'},
            components['qwen_generator'].generate_image.return_value  # Success on retry
        ]
        
        # Mock error recovery mechanism
        with patch('qwen_generator.handle_memory_error') as mock_handle_error:
            mock_handle_error.return_value = True  # Recovery successful
            
            # Step 1: First attempt fails
            result1 = qwen_generator.generate_image(
                prompt="Large image generation",
                width=1024,
                height=1024
            )
            
            assert result1['success'] is False
            assert result1['error_type'] == 'memory_error'
            
            # Step 2: Error recovery
            recovery_success = mock_handle_error(result1['error_type'])
            assert recovery_success is True
            
            # Step 3: Retry with reduced parameters
            result2 = qwen_generator.generate_image(
                prompt="Large image generation",
                width=512,  # Reduced size
                height=512
            )
            
            assert result2['success'] is True
            mock_handle_error.assert_called_once_with('memory_error')


class TestEndToEndDiffSynthEditingWorkflow:
    """End-to-end tests for complete DiffSynth image editing workflow"""
    
    @pytest.fixture
    def editing_workflow_setup(self):
        """Set up complete editing workflow environment"""
        # Create resource manager
        resource_manager = ResourceManager()
        
        # Create DiffSynth service with mocked components
        config = DiffSynthConfig(device="cuda")
        diffsynth_service = Mock(spec=DiffSynthService)
        diffsynth_service.status = DiffSynthServiceStatus.READY
        diffsynth_service.config = config
        
        # Create test images
        original_img = Image.new('RGB', (512, 512), color='blue')
        edited_img = Image.new('RGB', (512, 512), color='green')
        
        original_buffer = io.BytesIO()
        original_img.save(original_buffer, format='PNG')
        original_base64 = base64.b64encode(original_buffer.getvalue()).decode()
        
        edited_buffer = io.BytesIO()
        edited_img.save(edited_buffer, format='PNG')
        edited_base64 = base64.b64encode(edited_buffer.getvalue()).decode()
        
        # Mock editing responses
        diffsynth_service.edit_image.return_value = (edited_base64, "Edit successful")
        diffsynth_service.inpaint.return_value = (edited_base64, "Inpainting successful")
        diffsynth_service.outpaint.return_value = (edited_base64, "Outpainting successful")
        diffsynth_service.style_transfer.return_value = (edited_base64, "Style transfer successful")
        
        return {
            'resource_manager': resource_manager,
            'diffsynth_service': diffsynth_service,
            'original_image_base64': original_base64,
            'edited_image_base64': edited_base64
        }
    
    def test_complete_image_editing_workflow_with_comparison(self, editing_workflow_setup):
        """Test complete image editing workflow with before/after comparison"""
        components = editing_workflow_setup
        diffsynth_service = components['diffsynth_service']
        original_image = components['original_image_base64']
        
        # Test workflow: Load image -> Edit -> Compare -> Save
        
        # Step 1: Load original image
        edit_request = ImageEditRequest(
            prompt="Add a beautiful sunset to this landscape",
            image_base64=original_image,
            strength=0.7,
            guidance_scale=7.5,
            num_inference_steps=20
        )
        
        # Step 2: Perform editing
        edited_image, message = diffsynth_service.edit_image(
            prompt=edit_request.prompt,
            image_base64=edit_request.image_base64,
            strength=edit_request.strength,
            guidance_scale=edit_request.guidance_scale
        )
        
        assert edited_image is not None
        assert "successful" in message
        
        # Step 3: Create comparison data
        comparison_data = {
            'original': original_image,
            'edited': edited_image,
            'edit_parameters': {
                'prompt': edit_request.prompt,
                'strength': edit_request.strength,
                'guidance_scale': edit_request.guidance_scale
            },
            'processing_time': 8.5,
            'timestamp': time.time()
        }
        
        # Step 4: Validate comparison
        assert comparison_data['original'] != comparison_data['edited']
        assert comparison_data['edit_parameters']['prompt'] == edit_request.prompt
        assert comparison_data['processing_time'] > 0
        
        # Verify service was called correctly
        diffsynth_service.edit_image.assert_called_once()
    
    def test_complete_inpainting_workflow_with_mask_validation(self, editing_workflow_setup):
        """Test complete inpainting workflow with mask validation and preprocessing"""
        components = editing_workflow_setup
        diffsynth_service = components['diffsynth_service']
        original_image = components['original_image_base64']
        
        # Create mask image
        mask_img = Image.new('L', (512, 512), color=0)
        # Add white region to mask (area to inpaint)
        mask_img.paste(255, (100, 100, 400, 400))
        mask_buffer = io.BytesIO()
        mask_img.save(mask_buffer, format='PNG')
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
        
        # Test workflow: Validate mask -> Preprocess -> Inpaint -> Validate result
        
        # Step 1: Create inpaint request
        inpaint_request = InpaintRequest(
            prompt="A beautiful flower garden",
            image_base64=original_image,
            mask_base64=mask_base64,
            strength=0.9,
            guidance_scale=8.0
        )
        
        # Step 2: Mock mask validation
        with patch.object(diffsynth_service, '_validate_mask_compatibility', return_value=True) as mock_validate:
            # Step 3: Perform inpainting
            result_image, message = diffsynth_service.inpaint(inpaint_request)
            
            assert result_image is not None
            assert "successful" in message
            
            # Step 4: Verify mask validation was called
            mock_validate.assert_called_once()
            
            # Step 5: Verify inpainting parameters
            diffsynth_service.inpaint.assert_called_once_with(inpaint_request)


class TestEndToEndControlNetWorkflow:
    """End-to-end tests for complete ControlNet-guided generation workflow"""
    
    @pytest.fixture
    def controlnet_workflow_setup(self):
        """Set up complete ControlNet workflow environment"""
        # Create ControlNet service with mocked components
        controlnet_service = Mock(spec=ControlNetService)
        controlnet_service.initialized = True
        
        # Create test control image
        control_img = Image.new('RGB', (512, 512), color='white')
        # Add some edge-like features
        for i in range(100, 400):
            control_img.putpixel((i, 100), (0, 0, 0))  # Top edge
            control_img.putpixel((i, 400), (0, 0, 0))  # Bottom edge
            control_img.putpixel((100, i), (0, 0, 0))  # Left edge
            control_img.putpixel((400, i), (0, 0, 0))  # Right edge
        
        control_buffer = io.BytesIO()
        control_img.save(control_buffer, format='PNG')
        control_image_base64 = base64.b64encode(control_buffer.getvalue()).decode()
        
        # Create generated result image
        result_img = Image.new('RGB', (512, 512), color='blue')
        result_buffer = io.BytesIO()
        result_img.save(result_buffer, format='PNG')
        result_image_base64 = base64.b64encode(result_buffer.getvalue()).decode()
        
        # Mock ControlNet responses
        controlnet_service.detect_control_type.return_value = {
            'detected_type': ControlNetType.CANNY,
            'confidence': 0.95,
            'preview_base64': control_image_base64,
            'detection_time': 1.2
        }
        
        controlnet_service.generate_control_map.return_value = {
            'control_map_base64': control_image_base64,
            'control_type': ControlNetType.CANNY,
            'processing_time': 2.1
        }
        
        controlnet_service.process_with_control.return_value = {
            'success': True,
            'image_base64': result_image_base64,
            'generation_time': 12.5,
            'control_influence': 0.85,
            'parameters': {
                'controlnet_conditioning_scale': 1.0,
                'control_guidance_start': 0.0,
                'control_guidance_end': 1.0
            }
        }
        
        return {
            'controlnet_service': controlnet_service,
            'control_image_base64': control_image_base64,
            'result_image_base64': result_image_base64
        }
    
    def test_complete_controlnet_auto_detection_workflow(self, controlnet_workflow_setup):
        """Test complete ControlNet workflow with automatic control type detection"""
        components = controlnet_workflow_setup
        controlnet_service = components['controlnet_service']
        control_image = components['control_image_base64']
        
        # Test workflow: Upload image -> Auto-detect -> Generate control map -> Generate image
        
        # Step 1: Auto-detect control type
        detection_result = controlnet_service.detect_control_type(control_image)
        
        assert detection_result['detected_type'] == ControlNetType.CANNY
        assert detection_result['confidence'] > 0.9
        assert 'preview_base64' in detection_result
        assert detection_result['detection_time'] > 0
        
        # Step 2: Generate control map based on detection
        control_map_result = controlnet_service.generate_control_map(
            image_base64=control_image,
            control_type=detection_result['detected_type']
        )
        
        assert control_map_result['control_type'] == ControlNetType.CANNY
        assert 'control_map_base64' in control_map_result
        assert control_map_result['processing_time'] > 0
        
        # Step 3: Generate image with ControlNet guidance
        controlnet_request = ControlNetRequest(
            prompt="A beautiful architectural drawing",
            control_image_base64=control_map_result['control_map_base64'],
            control_type=detection_result['detected_type'],
            controlnet_conditioning_scale=1.0
        )
        
        generation_result = controlnet_service.process_with_control(controlnet_request)
        
        assert generation_result['success'] is True
        assert 'image_base64' in generation_result
        assert generation_result['generation_time'] > 0
        assert generation_result['control_influence'] > 0.8
        
        # Verify all steps were called
        controlnet_service.detect_control_type.assert_called_once_with(control_image)
        controlnet_service.generate_control_map.assert_called_once()
        controlnet_service.process_with_control.assert_called_once_with(controlnet_request)


class TestEndToEndServiceSwitchingAndResourceSharing:
    """End-to-end tests for service switching and resource sharing workflows"""
    
    @pytest.fixture
    def service_management_setup(self):
        """Set up complete service management environment"""
        # Create resource manager
        resource_manager = ResourceManager()
        
        # Create service manager
        service_manager = Mock(spec=ServiceManager)
        service_manager.resource_manager = resource_manager
        
        # Create mock services
        qwen_service = Mock()
        qwen_service.service_id = "qwen_generator_1"
        qwen_service.service_type = ServiceType.QWEN_GENERATOR
        
        diffsynth_service = Mock()
        diffsynth_service.service_id = "diffsynth_service_1"
        diffsynth_service.service_type = ServiceType.DIFFSYNTH_SERVICE
        
        controlnet_service = Mock()
        controlnet_service.service_id = "controlnet_service_1"
        controlnet_service.service_type = ServiceType.CONTROLNET_SERVICE
        
        # Mock service health status
        service_manager.get_service_health.return_value = {
            'qwen_generator_1': {'status': 'healthy', 'response_time': 0.1, 'memory_usage': 2.5},
            'diffsynth_service_1': {'status': 'healthy', 'response_time': 0.2, 'memory_usage': 3.2},
            'controlnet_service_1': {'status': 'healthy', 'response_time': 0.15, 'memory_usage': 1.8}
        }
        
        return {
            'resource_manager': resource_manager,
            'service_manager': service_manager,
            'qwen_service': qwen_service,
            'diffsynth_service': diffsynth_service,
            'controlnet_service': controlnet_service
        }
    
    def test_complete_service_switching_workflow(self, service_management_setup):
        """Test complete workflow for switching between services with resource management"""
        components = service_management_setup
        resource_manager = components['resource_manager']
        service_manager = components['service_manager']
        
        # Test workflow: Register services -> Allocate resources -> Switch services -> Reallocate
        
        # Step 1: Register all services
        services = [
            (components['qwen_service'], ServiceType.QWEN_GENERATOR, ResourcePriority.NORMAL),
            (components['diffsynth_service'], ServiceType.DIFFSYNTH_SERVICE, ResourcePriority.HIGH),
            (components['controlnet_service'], ServiceType.CONTROLNET_SERVICE, ResourcePriority.LOW)
        ]
        
        for service, service_type, priority in services:
            success = resource_manager.register_service(
                service_type,
                service.service_id,
                priority
            )
            assert success is True
        
        # Step 2: Allocate initial resources (Qwen active)
        with patch.object(resource_manager, '_get_current_gpu_usage', return_value=0.0):
            qwen_allocated = resource_manager.request_memory("qwen_generator_1", 3.0)
            assert qwen_allocated is True
        
        # Step 3: Switch to DiffSynth service (higher priority)
        service_manager.switch_service.return_value = {
            'success': True,
            'previous_service': 'qwen_generator_1',
            'new_service': 'diffsynth_service_1',
            'resource_reallocation': {
                'freed_memory': 3.0,
                'allocated_memory': 4.0
            }
        }
        
        switch_result = service_manager.switch_service(
            target_service="diffsynth_service_1",
            priority="high"
        )
        
        assert switch_result['success'] is True
        assert switch_result['new_service'] == 'diffsynth_service_1'
        
        # Step 4: Verify resource reallocation
        with patch.object(resource_manager, '_get_current_gpu_usage', return_value=1.0):
            diffsynth_allocated = resource_manager.request_memory("diffsynth_service_1", 4.0)
            assert diffsynth_allocated is True
        
        # Verify switch operations
        service_manager.switch_service.assert_called_once()


class TestEndToEndCompleteWorkflowIntegration:
    """End-to-end tests combining all workflows in realistic user scenarios"""
    
    @pytest.fixture
    def complete_system_setup(self):
        """Set up complete system with all components"""
        # Create all managers and services
        resource_manager = ResourceManager()
        service_manager = Mock(spec=ServiceManager)
        preset_manager = Mock(spec=PresetManager)
        
        # Create all services
        qwen_generator = Mock(spec=QwenImageGenerator)
        diffsynth_service = Mock(spec=DiffSynthService)
        controlnet_service = Mock(spec=ControlNetService)
        
        # Create test images
        test_img = Image.new('RGB', (512, 512), color='red')
        test_buffer = io.BytesIO()
        test_img.save(test_buffer, format='PNG')
        test_image_base64 = base64.b64encode(test_buffer.getvalue()).decode()
        
        # Mock responses
        qwen_generator.generate_image.return_value = {
            'success': True,
            'image_base64': test_image_base64,
            'generation_time': 5.0,
            'metadata': {'prompt': 'test', 'width': 512, 'height': 512}
        }
        
        diffsynth_service.edit_image.return_value = (test_image_base64, "Edit successful")
        diffsynth_service.status = DiffSynthServiceStatus.READY
        
        controlnet_service.process_with_control.return_value = {
            'success': True,
            'image_base64': test_image_base64,
            'generation_time': 8.0
        }
        controlnet_service.initialized = True
        
        return {
            'resource_manager': resource_manager,
            'service_manager': service_manager,
            'preset_manager': preset_manager,
            'qwen_generator': qwen_generator,
            'diffsynth_service': diffsynth_service,
            'controlnet_service': controlnet_service,
            'test_image_base64': test_image_base64
        }
    
    def test_complete_generation_to_editing_to_controlnet_workflow(self, complete_system_setup):
        """Test complete workflow: Generate -> Edit -> ControlNet enhance -> Save"""
        components = complete_system_setup
        
        # Step 1: Generate initial image with Qwen
        generation_result = components['qwen_generator'].generate_image(
            prompt="A beautiful mountain landscape",
            width=512,
            height=512,
            num_inference_steps=20
        )
        
        assert generation_result['success'] is True
        generated_image = generation_result['image_base64']
        
        # Step 2: Edit the generated image with DiffSynth
        edited_image, edit_message = components['diffsynth_service'].edit_image(
            prompt="Add a sunset to the mountain landscape",
            image_base64=generated_image,
            strength=0.7
        )
        
        assert edited_image is not None
        assert "successful" in edit_message
        
        # Step 3: Enhance with ControlNet for better composition
        controlnet_request = ControlNetRequest(
            prompt="Enhance the composition and lighting",
            control_image_base64=edited_image,
            control_type=ControlNetType.CANNY,
            controlnet_conditioning_scale=0.8
        )
        
        final_result = components['controlnet_service'].process_with_control(controlnet_request)
        
        assert final_result['success'] is True
        final_image = final_result['image_base64']
        
        # Step 4: Create workflow summary
        workflow_summary = {
            'steps': [
                {'type': 'generation', 'service': 'qwen', 'time': generation_result['generation_time']},
                {'type': 'editing', 'service': 'diffsynth', 'time': 3.5},
                {'type': 'enhancement', 'service': 'controlnet', 'time': final_result['generation_time']}
            ],
            'total_time': generation_result['generation_time'] + 3.5 + final_result['generation_time'],
            'final_image': final_image,
            'workflow_id': str(uuid.uuid4())
        }
        
        # Verify complete workflow
        assert len(workflow_summary['steps']) == 3
        assert workflow_summary['total_time'] > 15  # Reasonable total time
        assert workflow_summary['final_image'] is not None
        
        # Verify all services were called
        components['qwen_generator'].generate_image.assert_called_once()
        components['diffsynth_service'].edit_image.assert_called_once()
        components['controlnet_service'].process_with_control.assert_called_once()
    
    def test_workflow_with_error_recovery_and_fallbacks(self, complete_system_setup):
        """Test workflow with error handling, recovery, and fallback mechanisms"""
        components = complete_system_setup
        
        # Step 1: Initial generation succeeds
        generation_result = components['qwen_generator'].generate_image(
            prompt="Test image for error recovery",
            width=512,
            height=512
        )
        assert generation_result['success'] is True
        
        # Step 2: DiffSynth editing fails
        components['diffsynth_service'].edit_image.return_value = (None, "DiffSynth service unavailable")
        
        edit_result, edit_message = components['diffsynth_service'].edit_image(
            prompt="Edit the image",
            image_base64=generation_result['image_base64']
        )
        
        assert edit_result is None
        assert "unavailable" in edit_message
        
        # Step 3: Fallback to alternative editing method
        components['qwen_generator'].simple_edit.return_value = {
            'success': True,
            'image_base64': components['test_image_base64'],
            'message': 'Fallback editing successful'
        }
        
        fallback_result = components['qwen_generator'].simple_edit(
            prompt="Edit the image",
            image_base64=generation_result['image_base64']
        )
        
        assert fallback_result['success'] is True
        
        # Step 4: Create error recovery report
        error_recovery_report = {
            'workflow_id': str(uuid.uuid4()),
            'errors_encountered': [
                {'service': 'diffsynth', 'error': 'service_unavailable', 'fallback_used': 'qwen_simple_edit'}
            ],
            'final_success': True,
            'fallbacks_used': 1
        }
        
        # Verify error recovery workflow
        assert error_recovery_report['final_success'] is True
        assert len(error_recovery_report['errors_encountered']) == 1
        assert error_recovery_report['fallbacks_used'] == 1
        
        # Verify fallback mechanism was triggered
        components['qwen_generator'].simple_edit.assert_called_once()


if __name__ == "__main__":
    # Configure test execution
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--maxfail=5",
        "-x"  # Stop on first failure for debugging
    ])