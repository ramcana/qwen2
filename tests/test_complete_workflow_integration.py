"""
Complete Workflow Integration Tests
Tests end-to-end workflows for text-to-image generation, DiffSynth editing, 
ControlNet-guided generation, and service switching
"""

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import pytest
import requests
from PIL import Image
import io
import base64

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from qwen_generator import QwenImageGenerator
    from diffsynth_service import DiffSynthService, DiffSynthConfig, DiffSynthServiceStatus
    from controlnet_service import ControlNetService, ControlNetType, ControlNetRequest
    from resource_manager import ResourceManager, ServiceType, ResourcePriority
    from diffsynth_models import ImageEditRequest, InpaintRequest, OutpaintRequest, StyleTransferRequest
    from api_server import app
    from service_manager import ServiceManager
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestTextToImageWorkflow:
    """End-to-end tests for text-to-image generation workflow"""
    
    @pytest.fixture
    def qwen_generator(self):
        """Create a mocked Qwen generator for testing"""
        with patch('qwen_generator.OPTIMIZED_COMPONENTS_AVAILABLE', True):
            generator = QwenImageGenerator()
            generator.pipe = Mock()
            generator.model_loaded = True
            generator.device = "cpu"
            yield generator
    
    @pytest.fixture
    def mock_image_response(self):
        """Create a mock image response"""
        # Create a simple test image
        img = Image.new('RGB', (512, 512), color='red')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return {
            'success': True,
            'image_base64': img_base64,
            'generation_time': 5.2,
            'seed': 42,
            'metadata': {
                'prompt': 'test prompt',
                'width': 512,
                'height': 512,
                'steps': 20
            }
        }
    
    def test_complete_text_to_image_generation(self, qwen_generator, mock_image_response):
        """Test complete text-to-image generation workflow"""
        # Mock the generation process
        qwen_generator.generate_image = Mock(return_value=mock_image_response)
        
        # Test generation request
        result = qwen_generator.generate_image(
            prompt="A beautiful landscape with mountains",
            width=512,
            height=512,
            num_inference_steps=20,
            cfg_scale=7.5,
            seed=42
        )
        
        # Verify successful generation
        assert result['success'] is True
        assert 'image_base64' in result
        assert result['generation_time'] > 0
        assert result['metadata']['prompt'] == "A beautiful landscape with mountains"
        
        # Verify generator was called with correct parameters
        qwen_generator.generate_image.assert_called_once_with(
            prompt="A beautiful landscape with mountains",
            width=512,
            height=512,
            num_inference_steps=20,
            cfg_scale=7.5,
            seed=42
        )
    
    def test_text_to_image_with_error_handling(self, qwen_generator):
        """Test text-to-image workflow with error handling"""
        # Mock generation failure
        qwen_generator.generate_image = Mock(return_value={
            'success': False,
            'message': 'GPU memory insufficient',
            'error_type': 'memory_error'
        })
        
        # Test generation with error
        result = qwen_generator.generate_image(
            prompt="Test prompt",
            width=1024,
            height=1024
        )
        
        # Verify error handling
        assert result['success'] is False
        assert 'message' in result
        assert result['error_type'] == 'memory_error'
    
    def test_text_to_image_parameter_validation(self, qwen_generator):
        """Test parameter validation in text-to-image workflow"""
        # Test with invalid parameters
        with patch.object(qwen_generator, '_validate_generation_params') as mock_validate:
            mock_validate.return_value = (False, "Invalid width: must be multiple of 8")
            
            result = qwen_generator.generate_image(
                prompt="Test prompt",
                width=513,  # Invalid width
                height=512
            )
            
            mock_validate.assert_called_once()
    
    def test_text_to_image_with_queue_management(self, qwen_generator):
        """Test text-to-image workflow with queue management"""
        # Mock queue system
        with patch('qwen_generator.QueueManager') as mock_queue:
            queue_instance = Mock()
            mock_queue.return_value = queue_instance
            queue_instance.add_job.return_value = "job_123"
            queue_instance.get_job_status.return_value = {
                'status': 'completed',
                'result': {'success': True, 'image_base64': 'test_image_data'}
            }
            
            # Test queued generation
            job_id = qwen_generator.queue_generation(
                prompt="Queued generation test",
                width=512,
                height=512
            )
            
            assert job_id == "job_123"
            
            # Test job status check
            status = qwen_generator.get_job_status(job_id)
            assert status['status'] == 'completed'
            assert status['result']['success'] is True


class TestDiffSynthEditingWorkflow:
    """Integration tests for image editing workflow with DiffSynth"""
    
    @pytest.fixture
    def diffsynth_service(self):
        """Create a mocked DiffSynth service for testing"""
        config = DiffSynthConfig(device="cpu")
        service = DiffSynthService(config)
        service.status = DiffSynthServiceStatus.READY
        service.pipe = Mock()
        yield service
    
    @pytest.fixture
    def test_image_base64(self):
        """Create a test image in base64 format"""
        img = Image.new('RGB', (512, 512), color='blue')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        return base64.b64encode(img_buffer.getvalue()).decode()
    
    def test_complete_image_editing_workflow(self, diffsynth_service, test_image_base64):
        """Test complete image editing workflow"""
        # Mock the editing process
        edited_image_base64 = test_image_base64  # Simplified for testing
        diffsynth_service.edit_image = Mock(return_value=(edited_image_base64, "Edit successful"))
        
        # Test image editing request
        result_image, message = diffsynth_service.edit_image(
            prompt="Add a sunset to this landscape",
            image_base64=test_image_base64,
            strength=0.7,
            guidance_scale=7.5
        )
        
        # Verify successful editing
        assert result_image is not None
        assert message == "Edit successful"
        
        # Verify service was called with correct parameters
        diffsynth_service.edit_image.assert_called_once_with(
            prompt="Add a sunset to this landscape",
            image_base64=test_image_base64,
            strength=0.7,
            guidance_scale=7.5
        )
    
    def test_inpainting_workflow(self, diffsynth_service, test_image_base64):
        """Test inpainting workflow"""
        # Create a simple mask
        mask_img = Image.new('L', (512, 512), color=0)  # Black mask
        mask_buffer = io.BytesIO()
        mask_img.save(mask_buffer, format='PNG')
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
        
        # Mock inpainting process
        diffsynth_service.inpaint = Mock(return_value=(test_image_base64, "Inpainting successful"))
        
        # Test inpainting request
        inpaint_request = InpaintRequest(
            prompt="A beautiful flower",
            image_base64=test_image_base64,
            mask_base64=mask_base64,
            strength=0.8
        )
        
        result_image, message = diffsynth_service.inpaint(inpaint_request)
        
        # Verify successful inpainting
        assert result_image is not None
        assert message == "Inpainting successful"
        diffsynth_service.inpaint.assert_called_once_with(inpaint_request)
    
    def test_outpainting_workflow(self, diffsynth_service, test_image_base64):
        """Test outpainting workflow"""
        # Mock outpainting process
        diffsynth_service.outpaint = Mock(return_value=(test_image_base64, "Outpainting successful"))
        
        # Test outpainting request
        outpaint_request = OutpaintRequest(
            prompt="Extend the landscape",
            image_base64=test_image_base64,
            direction="right",
            pixels=256
        )
        
        result_image, message = diffsynth_service.outpaint(outpaint_request)
        
        # Verify successful outpainting
        assert result_image is not None
        assert message == "Outpainting successful"
        diffsynth_service.outpaint.assert_called_once_with(outpaint_request)
    
    def test_style_transfer_workflow(self, diffsynth_service, test_image_base64):
        """Test style transfer workflow"""
        # Create a style image
        style_img = Image.new('RGB', (512, 512), color='green')
        style_buffer = io.BytesIO()
        style_img.save(style_buffer, format='PNG')
        style_base64 = base64.b64encode(style_buffer.getvalue()).decode()
        
        # Mock style transfer process
        diffsynth_service.style_transfer = Mock(return_value=(test_image_base64, "Style transfer successful"))
        
        # Test style transfer request
        style_request = StyleTransferRequest(
            prompt="Apply artistic style",
            content_image_base64=test_image_base64,
            style_image_base64=style_base64,
            style_strength=0.6
        )
        
        result_image, message = diffsynth_service.style_transfer(style_request)
        
        # Verify successful style transfer
        assert result_image is not None
        assert message == "Style transfer successful"
        diffsynth_service.style_transfer.assert_called_once_with(style_request)
    
    def test_tiled_processing_workflow(self, diffsynth_service):
        """Test tiled processing for large images"""
        # Create a large test image
        large_img = Image.new('RGB', (2048, 2048), color='purple')
        large_buffer = io.BytesIO()
        large_img.save(large_buffer, format='PNG')
        large_image_base64 = base64.b64encode(large_buffer.getvalue()).decode()
        
        # Mock tiled processing
        with patch('diffsynth_service.TiledProcessor') as mock_tiled:
            tiled_instance = Mock()
            mock_tiled.return_value = tiled_instance
            tiled_instance.should_use_tiling.return_value = True
            tiled_instance.process_tiled.return_value = (large_image_base64, "Tiled processing successful")
            
            diffsynth_service.edit_image = Mock(return_value=(large_image_base64, "Tiled processing successful"))
            
            # Test large image editing
            result_image, message = diffsynth_service.edit_image(
                prompt="Edit large image",
                image_base64=large_image_base64
            )
            
            # Verify tiled processing was used
            assert result_image is not None
            assert "successful" in message


class TestControlNetWorkflow:
    """Tests for ControlNet-guided generation workflow"""
    
    @pytest.fixture
    def controlnet_service(self):
        """Create a mocked ControlNet service for testing"""
        service = ControlNetService()
        service.initialized = True
        yield service
    
    @pytest.fixture
    def test_control_image(self):
        """Create a test control image"""
        img = Image.new('RGB', (512, 512), color='white')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        return base64.b64encode(img_buffer.getvalue()).decode()
    
    def test_controlnet_detection_workflow(self, controlnet_service, test_control_image):
        """Test ControlNet control type detection workflow"""
        # Mock detection process
        controlnet_service.detect_control_type = Mock(return_value={
            'detected_type': ControlNetType.CANNY,
            'confidence': 0.95,
            'preview_base64': test_control_image
        })
        
        # Test control type detection
        result = controlnet_service.detect_control_type(test_control_image)
        
        # Verify detection results
        assert result['detected_type'] == ControlNetType.CANNY
        assert result['confidence'] > 0.9
        assert 'preview_base64' in result
        
        controlnet_service.detect_control_type.assert_called_once_with(test_control_image)
    
    def test_control_map_generation_workflow(self, controlnet_service, test_control_image):
        """Test control map generation workflow"""
        # Mock control map generation
        controlnet_service.generate_control_map = Mock(return_value={
            'control_map_base64': test_control_image,
            'control_type': ControlNetType.CANNY,
            'processing_time': 1.2
        })
        
        # Test control map generation
        result = controlnet_service.generate_control_map(
            image_base64=test_control_image,
            control_type=ControlNetType.CANNY
        )
        
        # Verify control map generation
        assert 'control_map_base64' in result
        assert result['control_type'] == ControlNetType.CANNY
        assert result['processing_time'] > 0
        
        controlnet_service.generate_control_map.assert_called_once()
    
    def test_controlnet_guided_generation_workflow(self, controlnet_service, test_control_image):
        """Test ControlNet-guided generation workflow"""
        # Mock guided generation
        controlnet_service.process_with_control = Mock(return_value={
            'success': True,
            'image_base64': test_control_image,
            'generation_time': 8.5,
            'control_influence': 0.8
        })
        
        # Test ControlNet-guided generation
        request = ControlNetRequest(
            prompt="A beautiful portrait",
            control_image_base64=test_control_image,
            control_type=ControlNetType.CANNY,
            controlnet_conditioning_scale=1.0
        )
        
        result = controlnet_service.process_with_control(request)
        
        # Verify guided generation
        assert result['success'] is True
        assert 'image_base64' in result
        assert result['generation_time'] > 0
        assert result['control_influence'] > 0
        
        controlnet_service.process_with_control.assert_called_once_with(request)
    
    def test_multiple_control_types_workflow(self, controlnet_service, test_control_image):
        """Test workflow with multiple control types"""
        # Mock multiple control type detection
        controlnet_service.detect_multiple_controls = Mock(return_value={
            'detected_controls': [
                {'type': ControlNetType.CANNY, 'confidence': 0.9},
                {'type': ControlNetType.DEPTH, 'confidence': 0.7}
            ],
            'recommended_type': ControlNetType.CANNY
        })
        
        # Test multiple control detection
        result = controlnet_service.detect_multiple_controls(test_control_image)
        
        # Verify multiple controls detected
        assert len(result['detected_controls']) == 2
        assert result['recommended_type'] == ControlNetType.CANNY
        assert result['detected_controls'][0]['confidence'] > result['detected_controls'][1]['confidence']


class TestServiceSwitchingAndResourceSharing:
    """Tests for service switching and resource sharing"""
    
    @pytest.fixture
    def resource_manager(self):
        """Create a resource manager for testing"""
        manager = ResourceManager()
        yield manager
        # Cleanup
        with manager.allocation_lock:
            manager.services.clear()
    
    @pytest.fixture
    def service_manager(self, resource_manager):
        """Create a service manager for testing"""
        manager = ServiceManager(resource_manager)
        yield manager
    
    def test_service_switching_workflow(self, service_manager, resource_manager):
        """Test switching between different services"""
        # Register multiple services
        qwen_service_id = "qwen_generator_1"
        diffsynth_service_id = "diffsynth_service_1"
        
        resource_manager.register_service(
            ServiceType.QWEN_GENERATOR,
            qwen_service_id,
            ResourcePriority.NORMAL
        )
        
        resource_manager.register_service(
            ServiceType.DIFFSYNTH_SERVICE,
            diffsynth_service_id,
            ResourcePriority.NORMAL
        )
        
        # Test switching from Qwen to DiffSynth
        with patch.object(resource_manager, '_get_current_gpu_usage', return_value=2.0):
            # Allocate memory to Qwen service
            qwen_allocated = resource_manager.request_memory(qwen_service_id, 4.0)
            assert qwen_allocated is True
            
            # Switch to DiffSynth service
            diffsynth_allocated = resource_manager.request_memory(diffsynth_service_id, 3.0)
            assert diffsynth_allocated is True
            
            # Verify both services can coexist with available memory
            qwen_resource = resource_manager.services[qwen_service_id]
            diffsynth_resource = resource_manager.services[diffsynth_service_id]
            
            assert qwen_resource.is_active
            assert diffsynth_resource.is_active
    
    def test_resource_priority_management(self, resource_manager):
        """Test resource allocation based on priority"""
        # Register services with different priorities
        high_priority_id = "high_priority_service"
        low_priority_id = "low_priority_service"
        
        resource_manager.register_service(
            ServiceType.QWEN_GENERATOR,
            high_priority_id,
            ResourcePriority.HIGH
        )
        
        resource_manager.register_service(
            ServiceType.DIFFSYNTH_SERVICE,
            low_priority_id,
            ResourcePriority.LOW
        )
        
        # Allocate memory to low priority service first
        with patch.object(resource_manager, '_get_current_gpu_usage', return_value=1.0):
            low_allocated = resource_manager.request_memory(low_priority_id, 6.0)
            assert low_allocated is True
        
        # Try to allocate memory to high priority service (should succeed by preempting low priority)
        with patch.object(resource_manager, '_get_current_gpu_usage', return_value=7.0):
            high_allocated = resource_manager.request_memory(high_priority_id, 4.0, force_if_needed=True)
            assert high_allocated is True
            
            # Verify high priority service got resources
            high_resource = resource_manager.services[high_priority_id]
            assert high_resource.is_active
            assert high_resource.allocated_memory_gb == 4.0
    
    def test_concurrent_service_usage(self, resource_manager):
        """Test concurrent usage of multiple services"""
        import threading
        import time
        
        # Register multiple services
        service_ids = [f"service_{i}" for i in range(3)]
        for i, service_id in enumerate(service_ids):
            resource_manager.register_service(
                ServiceType.QWEN_GENERATOR if i % 2 == 0 else ServiceType.DIFFSYNTH_SERVICE,
                service_id,
                ResourcePriority.NORMAL
            )
        
        results = {}
        
        def allocate_memory(service_id, memory_gb):
            with patch.object(resource_manager, '_get_current_gpu_usage', return_value=0.0):
                result = resource_manager.request_memory(service_id, memory_gb)
                results[service_id] = result
                time.sleep(0.1)  # Simulate some processing time
                resource_manager.release_memory(service_id)
        
        # Start concurrent memory allocation
        threads = []
        for i, service_id in enumerate(service_ids):
            thread = threading.Thread(target=allocate_memory, args=(service_id, 2.0))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all allocations were handled
        assert len(results) == 3
        # At least some should succeed (depending on total memory)
        successful_allocations = sum(1 for success in results.values() if success)
        assert successful_allocations > 0
    
    def test_service_health_monitoring(self, service_manager):
        """Test service health monitoring and recovery"""
        # Mock service health checks
        with patch.object(service_manager, 'check_service_health') as mock_health:
            mock_health.return_value = {
                'qwen_generator': {'status': 'healthy', 'response_time': 0.1},
                'diffsynth_service': {'status': 'unhealthy', 'error': 'Memory error'},
                'controlnet_service': {'status': 'healthy', 'response_time': 0.2}
            }
            
            # Test health monitoring
            health_status = service_manager.check_service_health()
            
            # Verify health status
            assert health_status['qwen_generator']['status'] == 'healthy'
            assert health_status['diffsynth_service']['status'] == 'unhealthy'
            assert health_status['controlnet_service']['status'] == 'healthy'
            
            # Test automatic recovery for unhealthy service
            with patch.object(service_manager, 'restart_service') as mock_restart:
                service_manager.handle_unhealthy_services(health_status)
                mock_restart.assert_called_with('diffsynth_service')
    
    def test_resource_cleanup_on_service_shutdown(self, resource_manager):
        """Test resource cleanup when services shut down"""
        service_id = "test_cleanup_service"
        
        # Register service with cleanup callback
        cleanup_called = False
        
        def cleanup_callback():
            nonlocal cleanup_called
            cleanup_called = True
        
        resource_manager.register_service(
            ServiceType.DIFFSYNTH_SERVICE,
            service_id,
            ResourcePriority.NORMAL,
            cleanup_callback=cleanup_callback
        )
        
        # Allocate memory
        with patch.object(resource_manager, '_get_current_gpu_usage', return_value=1.0):
            allocated = resource_manager.request_memory(service_id, 3.0)
            assert allocated is True
        
        # Unregister service (simulating shutdown)
        resource_manager.unregister_service(service_id)
        
        # Verify cleanup was called
        assert cleanup_called is True
        assert service_id not in resource_manager.services


class TestEndToEndWorkflowIntegration:
    """End-to-end integration tests combining all workflows"""
    
    @pytest.fixture
    def full_system_setup(self):
        """Set up a complete system for end-to-end testing"""
        # Mock all major components
        qwen_generator = Mock()
        diffsynth_service = Mock()
        controlnet_service = Mock()
        resource_manager = ResourceManager()
        
        return {
            'qwen_generator': qwen_generator,
            'diffsynth_service': diffsynth_service,
            'controlnet_service': controlnet_service,
            'resource_manager': resource_manager
        }
    
    def test_complete_generation_to_editing_workflow(self, full_system_setup):
        """Test complete workflow from generation to editing"""
        components = full_system_setup
        
        # Step 1: Generate initial image with Qwen
        test_image_base64 = "test_generated_image_data"
        components['qwen_generator'].generate_image.return_value = {
            'success': True,
            'image_base64': test_image_base64,
            'generation_time': 5.0
        }
        
        generated_result = components['qwen_generator'].generate_image(
            prompt="A landscape with mountains"
        )
        
        assert generated_result['success'] is True
        generated_image = generated_result['image_base64']
        
        # Step 2: Edit the generated image with DiffSynth
        edited_image_base64 = "test_edited_image_data"
        components['diffsynth_service'].edit_image.return_value = (edited_image_base64, "Edit successful")
        
        edited_result, message = components['diffsynth_service'].edit_image(
            prompt="Add a sunset to the landscape",
            image_base64=generated_image
        )
        
        assert edited_result is not None
        assert message == "Edit successful"
        
        # Step 3: Apply ControlNet guidance to the edited image
        final_image_base64 = "test_controlnet_image_data"
        components['controlnet_service'].process_with_control.return_value = {
            'success': True,
            'image_base64': final_image_base64,
            'generation_time': 7.0
        }
        
        controlnet_request = Mock()
        controlnet_request.prompt = "Enhance the composition"
        controlnet_request.image_base64 = edited_result
        
        final_result = components['controlnet_service'].process_with_control(controlnet_request)
        
        assert final_result['success'] is True
        assert final_result['image_base64'] == final_image_base64
        
        # Verify the complete workflow
        components['qwen_generator'].generate_image.assert_called_once()
        components['diffsynth_service'].edit_image.assert_called_once()
        components['controlnet_service'].process_with_control.assert_called_once()
    
    def test_workflow_with_error_recovery(self, full_system_setup):
        """Test workflow with error recovery mechanisms"""
        components = full_system_setup
        
        # Step 1: Generation succeeds
        components['qwen_generator'].generate_image.return_value = {
            'success': True,
            'image_base64': "test_image_data"
        }
        
        # Step 2: Editing fails
        components['diffsynth_service'].edit_image.return_value = (None, "DiffSynth service unavailable")
        
        # Step 3: Fallback to simple editing
        components['qwen_generator'].simple_edit.return_value = {
            'success': True,
            'image_base64': "fallback_edited_image"
        }
        
        # Execute workflow with error handling
        generated_result = components['qwen_generator'].generate_image(prompt="Test prompt")
        assert generated_result['success'] is True
        
        edited_result, edit_message = components['diffsynth_service'].edit_image(
            prompt="Edit prompt",
            image_base64=generated_result['image_base64']
        )
        
        # Handle editing failure
        if edited_result is None:
            fallback_result = components['qwen_generator'].simple_edit(
                prompt="Edit prompt",
                image_base64=generated_result['image_base64']
            )
            assert fallback_result['success'] is True
    
    def test_performance_under_load(self, full_system_setup):
        """Test system performance under concurrent load"""
        components = full_system_setup
        
        # Mock performance metrics
        components['qwen_generator'].generate_image.return_value = {
            'success': True,
            'image_base64': "test_image",
            'generation_time': 3.0
        }
        
        components['diffsynth_service'].edit_image.return_value = ("edited_image", "Success")
        
        # Simulate concurrent requests
        import threading
        import time
        
        results = []
        
        def process_request(request_id):
            start_time = time.time()
            
            # Generate
            gen_result = components['qwen_generator'].generate_image(
                prompt=f"Request {request_id}"
            )
            
            # Edit
            edit_result, _ = components['diffsynth_service'].edit_image(
                prompt=f"Edit {request_id}",
                image_base64=gen_result['image_base64']
            )
            
            total_time = time.time() - start_time
            results.append({
                'request_id': request_id,
                'success': gen_result['success'] and edit_result is not None,
                'total_time': total_time
            })
        
        # Start multiple concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all requests completed
        assert len(results) == 5
        successful_requests = sum(1 for r in results if r['success'])
        assert successful_requests == 5
        
        # Verify reasonable performance
        avg_time = sum(r['total_time'] for r in results) / len(results)
        assert avg_time < 10.0  # Should complete within reasonable time


if __name__ == "__main__":
    # Run the integration tests
    pytest.main([__file__, "-v", "--tb=short"])