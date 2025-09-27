#!/usr/bin/env python3
"""
Quick system test for DiffSynth Enhanced API
"""

import requests
import time
import json

def test_api_connection():
    """Test if API is responding"""
    try:
        print("🔍 Testing API connection...")
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API is responding!")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   DiffSynth: {health_data.get('services', {}).get('diffsynth', {}).get('status', 'unknown')}")
            print(f"   GPU: {'Available' if health_data.get('gpu', {}).get('available') else 'Not Available'}")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API - is the backend running?")
        return False
    except requests.exceptions.Timeout:
        print("❌ API connection timed out")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_frontend():
    """Test if frontend is responding"""
    try:
        print("🔍 Testing frontend connection...")
        response = requests.get("http://localhost:3001", timeout=5)
        
        if response.status_code == 200:
            print("✅ Frontend is responding!")
            return True
        else:
            print(f"❌ Frontend returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to frontend - is the HTTP server running?")
        return False
    except Exception as e:
        print(f"❌ Frontend test failed: {e}")
        return False

def main():
    print("🚀 DiffSynth Enhanced System Test")
    print("=" * 40)
    
    # Test API
    api_ok = test_api_connection()
    print()
    
    # Test Frontend
    frontend_ok = test_frontend()
    print()
    
    # Summary
    print("📊 Test Results:")
    print(f"   Backend API: {'✅ OK' if api_ok else '❌ FAILED'}")
    print(f"   Frontend: {'✅ OK' if frontend_ok else '❌ FAILED'}")
    
    if api_ok and frontend_ok:
        print("\n🎉 System is ready!")
        print("   • Enhanced Frontend: http://localhost:3001/frontend/html/enhanced_frontend.html")
        print("   • Test Connection: http://localhost:3001/tests/frontend/test_connection.html")
        print("   • API Docs: http://localhost:8000/docs")
    else:
        print("\n🔧 System needs attention:")
        if not api_ok:
            print("   • Start backend: python src/api_server_diffsynth.py &")
        if not frontend_ok:
            print("   • Start frontend: python serve_frontend.py &")

if __name__ == "__main__":
    main()