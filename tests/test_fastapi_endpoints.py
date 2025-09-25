#!/usr/bin/env python3
"""
Test script for FastAPI endpoints with memory optimization
Tests all API endpoints and verifies memory management
"""

import os
import sys
import time

import requests

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

API_BASE_URL = "http://localhost:8000"

def test_api_connection():
    """Test basic API connection"""
    print("🔗 Testing API connection...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            print("✅ API connection successful")
            data = response.json()
            print(f"   API Name: {data.get('name')}")
            print(f"   Version: {data.get('version')}")
            return True
        else:
            print(f"❌ API connection failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is the server running?")
        print("💡 Start the API server with: python src/api/main.py")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_health_check():
    """Test health check endpoint"""
    print("\n🩺 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            print(f"   GPU available: {data.get('gpu_available')}")
            if data.get('memory_info'):
                mem = data['memory_info']
                print(f"   VRAM: {mem.get('allocated_gb')}GB / {mem.get('total_gb')}GB ({mem.get('usage_percent')}%)")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_status_endpoint():
    """Test status endpoint"""
    print("\n📊 Testing status endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ Status endpoint working")
            print(f"   Model loaded: {data.get('model_loaded')}")
            print(f"   GPU available: {data.get('gpu_available')}")
            print(f"   Queue size: {data.get('queue_size')}")
            print(f"   Is generating: {data.get('is_generating')}")
            return data
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Status endpoint error: {e}")
        return None

def test_model_initialization():
    """Test model initialization"""
    print("\n🤖 Testing model initialization...")
    try:
        response = requests.post(f"{API_BASE_URL}/initialize")
        if response.status_code == 200:
            data = response.json()
            print("✅ Model initialization successful")
            print(f"   Message: {data.get('message')}")
            return True
        else:
            print(f"❌ Model initialization failed: {response.status_code}")
            if response.headers.get('content-type', '').startswith('application/json'):
                error_data = response.json()
                print(f"   Error: {error_data.get('detail')}")
            return False
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        return False

def test_aspect_ratios():
    """Test aspect ratios endpoint"""
    print("\n📏 Testing aspect ratios endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/aspect-ratios")
        if response.status_code == 200:
            data = response.json()
            ratios = data.get('ratios', {})
            print("✅ Aspect ratios endpoint working")
            print(f"   Available ratios: {len(ratios)}")
            for ratio, dimensions in list(ratios.items())[:3]:
                print(f"   {ratio}: {dimensions[0]}×{dimensions[1]}")
            if len(ratios) > 3:
                print(f"   ... and {len(ratios) - 3} more")
            return ratios
        else:
            print(f"❌ Aspect ratios endpoint failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Aspect ratios error: {e}")
        return None

def test_memory_clear():
    """Test memory clear endpoint"""
    print("\n🧹 Testing memory clear endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/memory/clear")
        if response.status_code == 200:
            data = response.json()
            print("✅ Memory clear successful")
            print(f"   Message: {data.get('message')}")
            if data.get('memory_info'):
                mem = data['memory_info']
                print(f"   Memory after clear: {mem.get('allocated_gb')}GB ({mem.get('usage_percent')}%)")
            return True
        else:
            print(f"❌ Memory clear failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Memory clear error: {e}")
        return False

def test_text_to_image_generation():
    """Test text-to-image generation endpoint"""
    print("\n🎨 Testing text-to-image generation...")
    
    # Test payload
    payload = {
        "prompt": "A test image for API validation: a simple coffee cup with steam",
        "negative_prompt": "blurry, low quality",
        "width": 832,
        "height": 832,
        "num_inference_steps": 20,  # Fast for testing
        "cfg_scale": 4.0,
        "seed": 12345,
        "language": "en",
        "enhance_prompt": True,
        "aspect_ratio": "1:1"
    }
    
    try:
        print("   Sending generation request...")
        print(f"   Prompt: {payload['prompt']}")
        print(f"   Settings: {payload['width']}×{payload['height']}, {payload['num_inference_steps']} steps")
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/generate/text-to-image",
            json=payload,
            timeout=120  # 2 minute timeout
        )
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Text-to-image generation successful")
            print(f"   Success: {data.get('success')}")
            print(f"   Message: {data.get('message')}")
            print(f"   Generation time: {data.get('generation_time', 'N/A')}s")
            print(f"   Request time: {request_time:.1f}s")
            
            if data.get('image_path'):
                image_path = data['image_path']
                print(f"   Image saved: {image_path}")
                
                # Check if image file exists
                if os.path.exists(image_path):
                    file_size = os.path.getsize(image_path) / 1024 / 1024  # MB
                    print(f"   File size: {file_size:.1f}MB")
                    return True
                else:
                    print("   ⚠️ Image file not found on disk")
                    return False
            else:
                print("   ⚠️ No image path in response")
                return False
        else:
            print(f"❌ Text-to-image generation failed: {response.status_code}")
            if response.headers.get('content-type', '').startswith('application/json'):
                error_data = response.json()
                print(f"   Error: {error_data.get('detail')}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Generation request timed out (>2 minutes)")
        return False
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False

def test_queue_management():
    """Test queue management endpoints"""
    print("\n📋 Testing queue management...")
    try:
        # Get queue status
        response = requests.get(f"{API_BASE_URL}/queue")
        if response.status_code == 200:
            data = response.json()
            print("✅ Queue endpoint working")
            print(f"   Queue size: {data.get('queue_size')}")
            print(f"   Is generating: {data.get('is_generating')}")
            
            if data.get('queue'):
                print(f"   Queue items: {len(data['queue'])}")
            
            return True
        else:
            print(f"❌ Queue endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Queue error: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive API test suite"""
    print("🧪 Qwen-Image FastAPI Comprehensive Test Suite")
    print("=" * 55)
    
    tests = [
        ("API Connection", test_api_connection),
        ("Health Check", test_health_check),
        ("Status Endpoint", test_status_endpoint),
        ("Model Initialization", test_model_initialization),
        ("Aspect Ratios", test_aspect_ratios),
        ("Memory Clear", test_memory_clear),
        ("Queue Management", test_queue_management),
        ("Text-to-Image Generation", test_text_to_image_generation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 55)
    print("📊 Test Results Summary")
    print("=" * 55)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print("\n" + "=" * 55)
    print(f"🏁 Tests Completed: {passed}/{total} passed ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! FastAPI endpoints are working correctly.")
        print("\n💡 Next steps:")
        print("   1. Test React frontend integration")
        print("   2. Run performance validation")
        print("   3. Deploy to production")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("\n💡 Troubleshooting:")
        print("   1. Ensure the API server is running: python src/api/main.py")
        print("   2. Check GPU memory availability")
        print("   3. Verify model files are downloaded")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)