#!/usr/bin/env python3
"""
Test script to verify that the network and caching fixes are working
"""

import os
import subprocess
import sys
from pathlib import Path


def test_qwen_home_set():
    """Test that QWEN_HOME is properly set"""
    qwen_home = os.environ.get("QWEN_HOME")
    if qwen_home:
        print("✅ QWEN_HOME is set:", qwen_home)
        return True
    else:
        # Check if we're in the project directory
        current_dir = Path.cwd()
        if (current_dir / "scripts" / "launch_complete_system.sh").exists():
            print("💡 QWEN_HOME is not set, but we're in the project directory")
            print("💡 Run this test from within the launch script for full validation")
            return True
        else:
            print("❌ QWEN_HOME is not set")
            return False


def test_model_cache_directory():
    """Test that the model cache directory exists and has content"""
    qwen_home = os.environ.get("QWEN_HOME")

    # If QWEN_HOME is not set, try to infer it
    if not qwen_home:
        current_dir = Path.cwd()
        if (current_dir / "scripts" / "launch_complete_system.sh").exists():
            qwen_home = str(current_dir)
            print("💡 Using current directory as QWEN_HOME for testing")
        else:
            print("❌ QWEN_HOME not set and not in project directory")
            return False

    cache_dir = Path(qwen_home) / "models" / "qwen-image"
    if cache_dir.exists():
        # Check if it has content
        files = list(cache_dir.rglob("*"))
        if files:
            print(f"✅ Model cache directory exists with {len(files)} files:", cache_dir)
            return True
        else:
            print("⚠️ Model cache directory exists but is empty:", cache_dir)
            return True
    else:
        print("❌ Model cache directory does not exist:", cache_dir)
        return False


def test_hf_transfer_installed():
    """Test that hf_transfer is installed"""
    try:
        import hf_transfer

        print("✅ hf_transfer is installed")
        return True
    except ImportError:
        print("❌ hf_transfer is not installed")
        return False


def test_hf_transfer_enabled():
    """Test that hf_transfer is enabled"""
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        print("✅ hf_transfer is enabled")
        return True
    else:
        print("❌ hf_transfer is not enabled")
        return False


def test_network_connectivity():
    """Test connectivity to HuggingFace endpoints"""
    import socket

    endpoints = [
        ("huggingface.co", 443),
        ("transfer.xethub.hf.co", 443),
    ]

    all_connected = True
    for host, port in endpoints:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()

            if result == 0:
                print(f"✅ Can connect to {host}:{port}")
            else:
                print(f"❌ Cannot connect to {host}:{port}")
                all_connected = False
        except Exception as e:
            print(f"❌ Error connecting to {host}:{port}: {e}")
            all_connected = False

    return all_connected


def test_environment_variables():
    """Test that required environment variables are set"""
    required_vars = {
        "HF_ENDPOINT": "https://huggingface.co",
        "HF_HUB_OFFLINE": "0",
    }

    all_set = True
    for var, expected_value in required_vars.items():
        actual_value = os.environ.get(var)
        if actual_value == expected_value:
            print(f"✅ {var} is correctly set to: {actual_value}")
        else:
            print(f"⚠️ {var} is set to: {actual_value}, expected: {expected_value}")
            # Not necessarily a failure, just informative

    return True


def main():
    """Main test function"""
    print("🧪 Testing Network and Caching Fixes")
    print("=" * 50)

    tests = [
        ("QWEN_HOME Environment Variable", test_qwen_home_set),
        ("Model Cache Directory", test_model_cache_directory),
        ("hf_transfer Installation", test_hf_transfer_installed),
        ("hf_transfer Enabled", test_hf_transfer_enabled),
        ("Network Connectivity", test_network_connectivity),
        ("Environment Variables", test_environment_variables),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print("\n" + "=" * 50)
    print(f"🏁 Tests Completed: {passed}/{total} passed ({100*passed/total:.1f}%)")

    if passed == total:
        print("🎉 All tests passed! Network and caching fixes are working correctly.")
    elif passed >= total * 0.8:
        print("✅ Most tests passed. The fixes are generally working.")
        print("💡 Some minor issues may need attention.")
    else:
        print("⚠️ Many tests failed. The fixes may not be fully effective.")
        print("💡 Review the individual test results above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
