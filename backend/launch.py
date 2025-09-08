#!/usr/bin/env python3
"""
Full-Stack Qwen-Image Generator Launcher
Launches the FastAPI backend and the React frontend concurrently.
"""

import subprocess
import sys
import os

def launch_app():
    """
    Launch the backend and frontend servers in parallel.
    """
    # Get the root directory of the project
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    backend_dir = os.path.join(root_dir, 'backend')
    frontend_dir = os.path.join(root_dir, 'frontend')

    print("🎨 Launching Full-Stack Qwen Image Generator")
    print("="*50)
    print(f"Project Root: {root_dir}")
    print(f"Backend Dir: {backend_dir}")
    print(f"Frontend Dir: {frontend_dir}")
    print("="*50)

    # Command to launch the FastAPI backend
    # We need to be in the backend/src directory to run this
    backend_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]

    # Command to launch the React frontend
    frontend_command = ["npm", "run", "dev", "--", "--port", "3000"]

    backend_process = None
    frontend_process = None

    try:
        print("🚀 Starting FastAPI backend...")
        # The working directory for the backend is backend/src
        backend_process = subprocess.Popen(backend_command, cwd=os.path.join(backend_dir, 'src'))
        print(f"Backend process started with PID: {backend_process.pid}")
        print("Backend available at: http://localhost:8000")

        print("\n🚀 Starting React frontend...")
        frontend_process = subprocess.Popen(frontend_command, cwd=frontend_dir)
        print(f"Frontend process started with PID: {frontend_process.pid}")
        print("Frontend available at: http://localhost:3000")

        print("\n✅ Both servers are running. Press Ctrl+C to stop.")

        # Wait for processes to complete
        frontend_process.wait()

    except KeyboardInterrupt:
        print("\n👋 Stopping servers...")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
    finally:
        if backend_process:
            print("Terminating backend process...")
            backend_process.terminate()
            backend_process.wait()
        if frontend_process:
            print("Terminating frontend process...")
            frontend_process.terminate()
            frontend_process.wait()
        print("All servers stopped. Goodbye!")

if __name__ == "__main__":
    launch_app()
