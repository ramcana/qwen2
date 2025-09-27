#!/usr/bin/env python3
"""
Test script for DiffSynth Enhanced API
"""

import requests
import json
import time

def test_generation():
    """Test text-to-image generation"""
    
    # Test health endpoint
    print("🔍 Testing health endpoint...")
    health_response = requests.get("http://localhost:8000/health")
    print(f"Health Status: {health_response.status_code}")
    if health_response.status_code == 200:
        health_data = health_response.json()
        print(f"GPU Available: {health_data['gpu']['available']}")
        print(f"DiffSynth Available: {health_data['services']['diffsynth']['available']}")
    
    # Test generation
    print("\n🎨 Testing text-to-image generation...")
    
    generation_request = {
        "prompt": "a beautiful sunset over mountains, digital art",
        "negative_prompt": "blurry, low quality",
        "width": 512,
        "height": 512,
        "steps": 10,
        "cfg_scale": 7.0,
        "seed": 42
    }
    
    # Submit generation job
    response = requests.post(
        "http://localhost:8000/api/generate/text-to-image",
        json=generation_request,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code != 200:
        print(f"❌ Generation request failed: {response.status_code}")
        print(response.text)
        return
    
    job_data = response.json()
    job_id = job_data["job_id"]
    print(f"✅ Job submitted: {job_id}")
    
    # Poll job status
    print("⏳ Waiting for generation to complete...")
    max_attempts = 60  # 5 minutes max
    attempt = 0
    
    while attempt < max_attempts:
        status_response = requests.get(f"http://localhost:8000/api/jobs/{job_id}")
        
        if status_response.status_code != 200:
            print(f"❌ Status check failed: {status_response.status_code}")
            break
        
        status_data = status_response.json()
        status = status_data["status"]
        progress = status_data["progress"]
        message = status_data["message"]
        
        print(f"Status: {status} | Progress: {progress:.1%} | {message}")
        
        if status == "completed":
            result = status_data["result"]
            print(f"🎉 Generation completed!")
            print(f"Image URL: {result['image_url']}")
            print(f"Filename: {result['filename']}")
            print(f"Seed used: {result['seed']}")
            return True
        elif status == "failed":
            error = status_data.get("error", "Unknown error")
            print(f"❌ Generation failed: {error}")
            return False
        
        time.sleep(5)
        attempt += 1
    
    print("⏰ Generation timed out")
    return False

if __name__ == "__main__":
    print("🚀 Testing DiffSynth Enhanced API")
    print("=" * 40)
    
    success = test_generation()
    
    if success:
        print("\n✅ API test completed successfully!")
    else:
        print("\n❌ API test failed")