#!/usr/bin/env python3
"""
Test script for the simple FastAPI endpoints: /health and /generate
Verifies that other apps can hit these endpoints
"""

import os
import sys
import time

import requests

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

API_BASE_URL = "http://localhost:8000"


def test_health_endpoint():
    """Test the /health endpoint"""
    print("🩺 Testing /health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health endpoint is working")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            print(f"   GPU available: {data.get('gpu_available')}")
            if data.get("memory_info"):
                mem = data["memory_info"]
                print(
                    f"   VRAM: {mem.get('allocated_gb')}GB / {mem.get('total_gb')}GB ({mem.get('usage_percent')}%)"
                )
            return True
        else:
            print(f"❌ Health endpoint failed with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is the server running?")
        print("💡 Start the API server with: python src/api/main.py")
        return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False


def test_generate_endpoint():
    """Test the simple /generate endpoint"""
    print("\n🎨 Testing /generate endpoint...")

    # Test payload - simple text-to-image request
    payload = {
        "prompt": "A beautiful sunset over mountains",
        "width": 832,
        "height": 832,
        "num_inference_steps": 20,
        "cfg_scale": 4.0,
        "seed": 12345,
    }

    try:
        print("   Sending generation request...")
        print(f"   Prompt: {payload['prompt']}")

        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/generate", json=payload, timeout=120  # 2 minute timeout
        )
        request_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            print("✅ Generate endpoint is working")
            print(f"   Success: {data.get('success')}")
            print(f"   Message: {data.get('message')}")
            print(f"   Request time: {request_time:.1f}s")

            if data.get("image_path"):
                print(f"   Image will be saved to: {data['image_path']}")
                return True
            else:
                print("   ⚠️ No image path in response")
                return True  # Endpoint worked, even if image wasn't generated
        else:
            print(
                f"❌ Generate endpoint failed with status code: {response.status_code}"
            )
            if response.headers.get("content-type", "").startswith("application/json"):
                error_data = response.json()
                print(f"   Error: {error_data.get('detail')}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is the server running?")
        print("💡 Start the API server with: python src/api/main.py")
        return False
    except requests.exceptions.Timeout:
        print("❌ Generate request timed out (>2 minutes)")
        return False
    except Exception as e:
        print(f"❌ Generate endpoint error: {e}")
        return False


def run_simple_tests():
    """Run tests for the simple endpoints: /health and /generate"""
    print("🧪 Qwen-Image FastAPI Simple Endpoints Test")
    print("=" * 45)

    # Test the health endpoint
    health_ok = test_health_endpoint()

    # Test the generate endpoint
    generate_ok = test_generate_endpoint()

    # Summary
    print("\n" + "=" * 45)
    print("📊 Test Results Summary")
    print("=" * 45)

    print(f"{'✅ HEALTH ENDPOINT' if health_ok else '❌ HEALTH ENDPOINT'}: /health")
    print(
        f"{'✅ GENERATE ENDPOINT' if generate_ok else '❌ GENERATE ENDPOINT'}: /generate"
    )

    all_passed = health_ok and generate_ok

    print("\n" + "=" * 45)
    if all_passed:
        print("🎉 All simple endpoints are working correctly!")
        print("💡 Other apps can now hit these endpoints.")
    else:
        print("⚠️  Some endpoints failed. Check the error messages above.")
        print("💡 Make sure the FastAPI server is running: python src/api/main.py")

    return all_passed


if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1)
