#!/usr/bin/env python3
"""
Verification script to test that the backend server starts properly
"""

import os
import subprocess
import sys
import time

import requests


def start_server():
    """Start the backend server in a subprocess"""
    try:
        # Change to project directory
        os.chdir("/home/ramji_t/projects/Qwen2")

        # Start server process
        process = subprocess.Popen(
            ["python", "src/api/robust_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        return process
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None


def check_server_status(process, timeout=30):
    """Check if server is running by making HTTP requests"""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            # Try to connect to the server
            response = requests.get("http://localhost:8000", timeout=5)
            if response.status_code == 200:
                print("✅ Server is running and responding")
                return True
        except requests.exceptions.RequestException:
            # Server not ready yet, wait a bit
            time.sleep(2)

        # Check if process is still running
        if process.poll() is not None:
            print("❌ Server process has exited")
            stdout, stderr = process.communicate()
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False

    print("⚠️ Server started but not responding within timeout")
    return False


def main():
    """Main verification function"""
    print("🔍 Verifying backend server startup...")
    print("=" * 50)

    # Start server
    print("🔄 Starting backend server...")
    server_process = start_server()

    if server_process is None:
        print("❌ Failed to start server process")
        return 1

    print(f"✅ Server process started (PID: {server_process.pid})")

    # Give server time to initialize
    print("⏳ Waiting for server to initialize...")
    time.sleep(10)

    # Check server status
    if check_server_status(server_process, timeout=30):
        print("✅ Backend server verification successful!")
        print("💡 You can now run the complete system with:")
        print("   ./scripts/launch_ui.sh")
        print("   Then select option 3 (Complete System)")
    else:
        print("❌ Backend server verification failed")
        return 1

    # Terminate server process
    print("🛑 Terminating server process...")
    server_process.terminate()
    try:
        server_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_process.kill()

    return 0


if __name__ == "__main__":
    sys.exit(main())
