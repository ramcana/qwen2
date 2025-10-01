#!/usr/bin/env python3
"""
Test script for image-to-image API functionality
"""

import requests
import time
import os
from PIL import Image, ImageDraw

def create_test_image():
    """Create a simple test image"""
    # Create a simple test image
    img = Image.new('RGB', (512, 512), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple pattern
    draw.rectangle([100, 100, 400, 400], fill='white', outline='black', width=3)
    draw.text((200, 250), "TEST IMAGE", fill='black')
    
    # Save test image
    test_image_path = "test_input.png"
    img.save(test_image_path)
    return test_image_path

def test_img2img_api():
    """Test the image-to-image API endpoint"""
    base_url = "http://localhost:8000"
    
    # Create test image
    test_image_path = create_test_image()
    print(f"Created test image: {test_image_path}")
    
    try:
        # Check if API is running
        response = requests.get(f"{base_url}/health")
        if response.status_code != 200:
            print("❌ API server is not running or not healthy")
            return
        
        print("✅ API server is running")
        
        # Check model status
        status_response = requests.get(f"{base_url}/status")
        status_data = status_response.json()
        
        if not status_data.get("model_loaded"):
            print("⚠️ Model not loaded. Attempting to initialize...")
            init_response = requests.post(f"{base_url}/initialize")
            if init_response.status_code != 200:
                print("❌ Failed to initialize model")
                return
            
            # Wait for initialization
            print("⏳ Waiting for model initialization...")
            time.sleep(10)
        
        # Test image-to-image generation
        print("🎨 Testing image-to-image generation...")
        
        with open(test_image_path, 'rb') as f:
            files = {'init_image': f}
            data = {
                'prompt': 'A beautiful landscape with mountains and a lake, photorealistic',
                'negative_prompt': 'blurry, low quality, distorted',
                'width': 512,
                'height': 512,
                'num_inference_steps': 20,
                'cfg_scale': 7.0,
                'seed': 42,
                'strength': 0.7,
                'enhance_prompt': True
            }
            
            response = requests.post(f"{base_url}/generate/image-to-image", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('job_id')
            print(f"✅ Generation started. Job ID: {job_id}")
            
            # Poll for completion
            print("⏳ Waiting for generation to complete...")
            max_wait = 300  # 5 minutes
            wait_time = 0
            
            while wait_time < max_wait:
                status_response = requests.get(f"{base_url}/queue/{job_id}")
                if status_response.status_code == 200:
                    job_status = status_response.json()
                    status = job_status.get('status')
                    
                    if status == 'completed':
                        image_path = job_status.get('image_path')
                        generation_time = job_status.get('generation_time')
                        print(f"✅ Generation completed in {generation_time:.2f}s")
                        print(f"📸 Generated image: {image_path}")
                        
                        # Try to download the image
                        if image_path:
                            filename = os.path.basename(image_path)
                            img_response = requests.get(f"{base_url}/images/{filename}")
                            if img_response.status_code == 200:
                                with open(f"downloaded_{filename}", 'wb') as f:
                                    f.write(img_response.content)
                                print(f"✅ Downloaded image as: downloaded_{filename}")
                            else:
                                print("⚠️ Could not download generated image")
                        
                        break
                    elif status == 'failed':
                        error = job_status.get('error', 'Unknown error')
                        print(f"❌ Generation failed: {error}")
                        break
                    elif status == 'processing':
                        print("⏳ Still processing...")
                    
                time.sleep(5)
                wait_time += 5
            
            if wait_time >= max_wait:
                print("⏰ Generation timed out")
        
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"Response: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        # Clean up test image
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

if __name__ == "__main__":
    test_img2img_api()