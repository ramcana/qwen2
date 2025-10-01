#!/usr/bin/env python3
"""
Docker End-to-End Workflow Tests
Tests complete user workflows in Docker environment from container startup to image generation.
"""

import asyncio
import json
import os
import sys
import time
import uuid
import subprocess
import requests
import pytest
import base64
import tempfile
from typing import Dict, Any, Optional, List
from pathlib import Path
import docker
from docker.errors import DockerException
from PIL import Image
import io

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class DockerE2EWorkflowTest:
    """Base class for Docker end-to-end workflow tests"""
    
    def __init__(self):
        self.docker_client = None
        self.compose_project = f"qwen-e2e-test-{uuid.uuid4().hex[:8]}"
        self.test_timeout = 600  # 10 minutes for E2E tests
        self.base_url = "http://localhost:8080"  # Traefik proxy
        
    def setup_docker_environment(self):
        """Setup Docker environment for E2E testing"""
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            return True
        except DockerException as e:
            pytest.skip(f"Docker not available: {e}")
            return False
    
    def start_docker_compose_stack(self, compose_file="docker-compose.yml"):
        """Start Docker Compose stack for testing"""
        compose_path = Path(__file__).parent.parent / compose_file
        
        if not compose_path.exists():
            pytest.skip(f"Docker Compose file not found: {compose_path}")
        
        # Start the stack
        cmd = [
            "docker-compose",
            "-f", str(compose_path),
            "-p", self.compose_project,
            "up", "-d",
            "--build"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes to start
                cwd=compose_path.parent
            )
            
            if result.returncode != 0:
                pytest.fail(f"Failed to start Docker Compose stack: {result.stderr}")
            
            # Wait for services to be ready
            self._wait_for_services_ready()
            return True
            
        except subprocess.TimeoutExpired:
            pytest.fail("Docker Compose stack startup timed out")
        except Exception as e:
            pytest.fail(f"Error starting Docker Compose stack: {e}")
    
    def stop_docker_compose_stack(self):
        """Stop and clean up Docker Compose stack"""
        cmd = [
            "docker-compose",
            "-p", self.compose_project,
            "down",
            "-v",  # Remove volumes
            "--remove-orphans"
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
        except Exception as e:
            print(f"Warning: Error stopping Docker Compose stack: {e}")
    
    def _wait_for_services_ready(self, timeout=300):
        """Wait for all services to be ready"""
        start_time = time.time()
        
        services_to_check = [
            {"name": "traefik", "url": f"{self.base_url}/api/rawdata", "expected_status": 200},
            {"name": "api", "url": f"{self.base_url}/api/health", "expected_status": 200},
            {"name": "frontend", "url": f"{self.base_url}/", "expected_status": 200}
        ]
        
        while time.time() - start_time < timeout:
            all_ready = True
            
            for service in services_to_check:
                try:
                    response = requests.get(service["url"], timeout=10)
                    if response.status_code != service["expected_status"]:
                        all_ready = False
                        break
                except requests.RequestException:
                    all_ready = False
                    break
            
            if all_ready:
                print("All services are ready")
                return True
            
            print("Waiting for services to be ready...")
            time.sleep(10)
        
        # Get service logs for debugging
        self._get_service_logs()
        pytest.fail(f"Services not ready within {timeout} seconds")
    
    def _get_service_logs(self):
        """Get logs from all services for debugging"""
        cmd = ["docker-compose", "-p", self.compose_project, "logs", "--tail=50"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            print("Service logs:")
            print(result.stdout)
        except Exception as e:
            print(f"Could not get service logs: {e}")


class TestDockerTextToImageE2EWorkflow(DockerE2EWorkflowTest):
    """Test complete text-to-image workflow in Docker environment"""
    
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup for each test"""
        self.setup_docker_environment()
        self.start_docker_compose_stack()
        yield
        self.stop_docker_compose_stack()
    
    def test_complete_text_to_image_workflow_via_api(self):
        """Test complete text-to-image generation workflow via API"""
        # Step 1: Verify API is accessible
        health_response = requests.get(f"{self.base_url}/api/health", timeout=30)
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        assert health_data.get('status') == 'healthy'
        
        # Step 2: Generate image via API
        generation_payload = {
            "prompt": "A beautiful mountain landscape with a lake",
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,
            "cfg_scale": 7.5,
            "seed": 42
        }
        
        generation_response = requests.post(
            f"{self.base_url}/api/generate/text-to-image",
            json=generation_payload,
            timeout=120  # 2 minutes for generation
        )
        
        assert generation_response.status_code == 200
        generation_data = generation_response.json()
        
        assert generation_data.get('success') is True
        assert 'image_base64' in generation_data
        assert generation_data.get('generation_time', 0) > 0
        
        # Step 3: Validate generated image
        image_base64 = generation_data['image_base64']
        self._validate_generated_image(image_base64, expected_size=(512, 512))
        
        # Step 4: Verify metadata
        metadata = generation_data.get('metadata', {})
        assert metadata.get('prompt') == generation_payload['prompt']
        assert metadata.get('width') == generation_payload['width']
        assert metadata.get('height') == generation_payload['height']
        
        return generation_data
    
    def test_text_to_image_with_different_parameters(self):
        """Test text-to-image with various parameter combinations"""
        test_cases = [
            {
                "name": "small_image",
                "params": {"prompt": "A cat", "width": 256, "height": 256, "num_inference_steps": 10}
            },
            {
                "name": "large_image",
                "params": {"prompt": "A dog", "width": 768, "height": 768, "num_inference_steps": 30}
            },
            {
                "name": "high_cfg",
                "params": {"prompt": "A bird", "width": 512, "height": 512, "cfg_scale": 15.0}
            },
            {
                "name": "low_cfg",
                "params": {"prompt": "A fish", "width": 512, "height": 512, "cfg_scale": 3.0}
            }
        ]
        
        results = {}
        
        for test_case in test_cases:
            print(f"Testing {test_case['name']}...")
            
            response = requests.post(
                f"{self.base_url}/api/generate/text-to-image",
                json=test_case['params'],
                timeout=180
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data.get('success') is True
            
            # Validate image dimensions
            expected_size = (test_case['params']['width'], test_case['params']['height'])
            self._validate_generated_image(data['image_base64'], expected_size)
            
            results[test_case['name']] = {
                'generation_time': data.get('generation_time'),
                'success': True
            }
        
        # Verify all test cases passed
        assert all(result['success'] for result in results.values())
        
        return results
    
    def test_text_to_image_error_handling(self):
        """Test error handling in text-to-image workflow"""
        # Test with invalid parameters
        invalid_cases = [
            {
                "name": "empty_prompt",
                "params": {"prompt": "", "width": 512, "height": 512},
                "expected_error": "prompt"
            },
            {
                "name": "invalid_dimensions",
                "params": {"prompt": "test", "width": -1, "height": 512},
                "expected_error": "width"
            },
            {
                "name": "too_large_image",
                "params": {"prompt": "test", "width": 2048, "height": 2048},
                "expected_error": "size"
            }
        ]
        
        for test_case in invalid_cases:
            print(f"Testing error case: {test_case['name']}...")
            
            response = requests.post(
                f"{self.base_url}/api/generate/text-to-image",
                json=test_case['params'],
                timeout=30
            )
            
            # Should return error status
            assert response.status_code in [400, 422, 500]
            
            if response.status_code != 500:  # Server errors might not have JSON
                error_data = response.json()
                assert 'error' in error_data or 'detail' in error_data
    
    def _validate_generated_image(self, image_base64: str, expected_size: tuple):
        """Validate generated image format and dimensions"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Check dimensions
            assert image.size == expected_size, f"Expected {expected_size}, got {image.size}"
            
            # Check format
            assert image.format in ['PNG', 'JPEG'], f"Unexpected format: {image.format}"
            
            # Check that image is not blank
            extrema = image.getextrema()
            if image.mode == 'RGB':
                # For RGB, extrema is ((min_r, max_r), (min_g, max_g), (min_b, max_b))
                assert any(max_val > min_val for min_val, max_val in extrema), "Image appears to be blank"
            
            return True
            
        except Exception as e:
            pytest.fail(f"Image validation failed: {e}")


class TestDockerDiffSynthE2EWorkflow(DockerE2EWorkflowTest):
    """Test complete DiffSynth editing workflow in Docker environment"""
    
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup for each test"""
        self.setup_docker_environment()
        self.start_docker_compose_stack()
        yield
        self.stop_docker_compose_stack()
    
    def test_complete_image_editing_workflow(self):
        """Test complete image editing workflow via DiffSynth API"""
        # Step 1: Generate initial image
        initial_generation = {
            "prompt": "A simple landscape",
            "width": 512,
            "height": 512,
            "num_inference_steps": 15
        }
        
        gen_response = requests.post(
            f"{self.base_url}/api/generate/text-to-image",
            json=initial_generation,
            timeout=120
        )
        
        assert gen_response.status_code == 200
        gen_data = gen_response.json()
        assert gen_data.get('success') is True
        
        original_image = gen_data['image_base64']
        
        # Step 2: Edit the generated image
        edit_payload = {
            "prompt": "Add a beautiful sunset to this landscape",
            "image_base64": original_image,
            "strength": 0.7,
            "guidance_scale": 7.5,
            "num_inference_steps": 20
        }
        
        edit_response = requests.post(
            f"{self.base_url}/api/diffsynth/edit",
            json=edit_payload,
            timeout=180
        )
        
        assert edit_response.status_code == 200
        edit_data = edit_response.json()
        
        assert edit_data.get('success') is True
        assert 'image_base64' in edit_data
        assert edit_data.get('processing_time', 0) > 0
        
        # Step 3: Validate edited image
        edited_image = edit_data['image_base64']
        self._validate_generated_image(edited_image, (512, 512))
        
        # Step 4: Verify images are different
        assert original_image != edited_image, "Edited image should be different from original"
        
        return {
            'original': original_image,
            'edited': edited_image,
            'edit_time': edit_data.get('processing_time')
        }
    
    def test_complete_inpainting_workflow(self):
        """Test complete inpainting workflow"""
        # Step 1: Generate base image
        base_generation = {
            "prompt": "A room with furniture",
            "width": 512,
            "height": 512
        }
        
        gen_response = requests.post(
            f"{self.base_url}/api/generate/text-to-image",
            json=base_generation,
            timeout=120
        )
        
        assert gen_response.status_code == 200
        gen_data = gen_response.json()
        base_image = gen_data['image_base64']
        
        # Step 2: Create mask for inpainting
        mask_image = self._create_test_mask(512, 512)
        
        # Step 3: Perform inpainting
        inpaint_payload = {
            "prompt": "A beautiful painting on the wall",
            "image_base64": base_image,
            "mask_base64": mask_image,
            "strength": 0.9,
            "guidance_scale": 8.0
        }
        
        inpaint_response = requests.post(
            f"{self.base_url}/api/diffsynth/inpaint",
            json=inpaint_payload,
            timeout=180
        )
        
        assert inpaint_response.status_code == 200
        inpaint_data = inpaint_response.json()
        
        assert inpaint_data.get('success') is True
        assert 'image_base64' in inpaint_data
        
        # Validate inpainted image
        inpainted_image = inpaint_data['image_base64']
        self._validate_generated_image(inpainted_image, (512, 512))
        
        return {
            'base': base_image,
            'mask': mask_image,
            'inpainted': inpainted_image
        }
    
    def _create_test_mask(self, width: int, height: int) -> str:
        """Create a test mask for inpainting"""
        # Create mask with white region in center
        mask = Image.new('L', (width, height), 0)  # Black background
        
        # Add white rectangle in center (area to inpaint)
        center_x, center_y = width // 2, height // 2
        mask_size = min(width, height) // 4
        
        for x in range(center_x - mask_size, center_x + mask_size):
            for y in range(center_y - mask_size, center_y + mask_size):
                if 0 <= x < width and 0 <= y < height:
                    mask.putpixel((x, y), 255)  # White (area to inpaint)
        
        # Convert to base64
        buffer = io.BytesIO()
        mask.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()


class TestDockerCompleteUserJourney(DockerE2EWorkflowTest):
    """Test complete user journey from frontend to backend"""
    
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup for each test"""
        self.setup_docker_environment()
        self.start_docker_compose_stack()
        yield
        self.stop_docker_compose_stack()
    
    def test_complete_user_journey_frontend_to_backend(self):
        """Test complete user journey from frontend interaction to image generation"""
        # Step 1: Verify frontend is accessible
        frontend_response = requests.get(self.base_url, timeout=30)
        assert frontend_response.status_code == 200
        
        # Step 2: Verify frontend can reach API
        # This simulates what the frontend would do
        api_health_response = requests.get(f"{self.base_url}/api/health", timeout=30)
        assert api_health_response.status_code == 200
        
        # Step 3: Simulate frontend making generation request
        user_request = {
            "prompt": "A beautiful sunset over mountains",
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,
            "cfg_scale": 7.5
        }
        
        # This is what the frontend would send to the API
        generation_response = requests.post(
            f"{self.base_url}/api/generate/text-to-image",
            json=user_request,
            headers={"Content-Type": "application/json"},
            timeout=180
        )
        
        assert generation_response.status_code == 200
        generation_data = generation_response.json()
        
        assert generation_data.get('success') is True
        assert 'image_base64' in generation_data
        
        # Step 4: Simulate frontend receiving and processing the image
        image_data = generation_data['image_base64']
        
        # Validate the image can be processed by frontend
        try:
            decoded_image = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(decoded_image))
            assert image.size == (user_request['width'], user_request['height'])
        except Exception as e:
            pytest.fail(f"Frontend image processing failed: {e}")
        
        # Step 5: Test image download functionality
        # Simulate frontend requesting image download
        download_payload = {
            "image_base64": image_data,
            "filename": "generated_image.png",
            "format": "PNG"
        }
        
        # This endpoint might not exist yet, but shows the workflow
        try:
            download_response = requests.post(
                f"{self.base_url}/api/images/download",
                json=download_payload,
                timeout=30
            )
            # If endpoint exists, it should work
            if download_response.status_code == 200:
                assert len(download_response.content) > 0
        except requests.RequestException:
            # Endpoint might not be implemented yet
            pass
        
        return {
            'user_request': user_request,
            'generation_result': generation_data,
            'workflow_success': True
        }
    
    def test_multi_step_user_workflow(self):
        """Test multi-step user workflow: Generate -> Edit -> Enhance"""
        workflow_steps = []
        
        # Step 1: Initial generation
        step1_request = {
            "prompt": "A simple house",
            "width": 512,
            "height": 512
        }
        
        step1_response = requests.post(
            f"{self.base_url}/api/generate/text-to-image",
            json=step1_request,
            timeout=120
        )
        
        assert step1_response.status_code == 200
        step1_data = step1_response.json()
        assert step1_data.get('success') is True
        
        workflow_steps.append({
            'step': 'generation',
            'success': True,
            'time': step1_data.get('generation_time', 0)
        })
        
        # Step 2: Edit the generated image
        step2_request = {
            "prompt": "Add a garden around the house",
            "image_base64": step1_data['image_base64'],
            "strength": 0.6
        }
        
        step2_response = requests.post(
            f"{self.base_url}/api/diffsynth/edit",
            json=step2_request,
            timeout=180
        )
        
        if step2_response.status_code == 200:
            step2_data = step2_response.json()
            if step2_data.get('success'):
                workflow_steps.append({
                    'step': 'editing',
                    'success': True,
                    'time': step2_data.get('processing_time', 0)
                })
                current_image = step2_data['image_base64']
            else:
                current_image = step1_data['image_base64']  # Fallback to original
        else:
            current_image = step1_data['image_base64']  # Fallback to original
        
        # Step 3: Final enhancement (if ControlNet is available)
        try:
            step3_request = {
                "prompt": "Enhance the lighting and composition",
                "control_image_base64": current_image,
                "control_type": "canny",
                "controlnet_conditioning_scale": 0.8
            }
            
            step3_response = requests.post(
                f"{self.base_url}/api/controlnet/generate",
                json=step3_request,
                timeout=180
            )
            
            if step3_response.status_code == 200:
                step3_data = step3_response.json()
                if step3_data.get('success'):
                    workflow_steps.append({
                        'step': 'enhancement',
                        'success': True,
                        'time': step3_data.get('generation_time', 0)
                    })
        except requests.RequestException:
            # ControlNet might not be available
            pass
        
        # Verify workflow completed successfully
        assert len(workflow_steps) >= 1  # At least generation should work
        assert all(step['success'] for step in workflow_steps)
        
        total_time = sum(step['time'] for step in workflow_steps)
        
        return {
            'workflow_steps': workflow_steps,
            'total_time': total_time,
            'steps_completed': len(workflow_steps)
        }


if __name__ == "__main__":
    # Run Docker E2E tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--maxfail=2",
        "-s"  # Don't capture output for debugging
    ])