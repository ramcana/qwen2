#!/usr/bin/env python3
"""
Robust Qwen-Image API Server Launcher
Handles long model loading times and provides better error handling
"""

import os
import signal
import sys

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)


def signal_handler(sig, frame):
    """Handle graceful shutdown"""
    print("\n🛑 Received shutdown signal...")
    print("🧹 Cleaning up resources...")
    sys.exit(0)


def main():
    """Main server launcher with robust error handling"""

    print("🚀 QWEN-IMAGE API SERVER LAUNCHER")
    print("=" * 50)
    print("High-Performance Configuration for RTX 4080")
    print("Expected generation time: 15-60 seconds")
    print("=" * 50)

    # Print current working directory and Python path for debugging
    print(f"📂 Current working directory: {os.getcwd()}")
    print(f"🏠 Project root: {project_root}")

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set high-performance environment variables
    env_vars = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:512",
        "CUDA_LAUNCH_BLOCKING": "0",
        "OMP_NUM_THREADS": "32",
        "MKL_NUM_THREADS": "32",
        "NUMBA_NUM_THREADS": "32",
    }

    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✅ {key}={value}")

    try:
        print("\n🔄 Starting FastAPI server...")
        print("💡 Model will be loaded on first request")
        print("⏰ No timeout applied - server will wait for model loading")

        # Change to project root directory first
        os.chdir(project_root)
        print(f"📂 Changed to project directory: {os.getcwd()}")

        # Now import the app after changing directory
        import uvicorn

        from src.api.main import app

        # Run with more robust settings (only valid parameters)
        uvicorn.run(
            "src.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload for production stability
            log_level="info",
            # timeout_keep_alive=300,  # 5 minutes keep-alive
            # timeout_graceful_shutdown=60,  # 1 minute graceful shutdown
        )

    except KeyboardInterrupt:
        print("\n🛑 Server interrupted by user")
        return 0
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("💡 Troubleshooting tips:")
        print("   1. Make sure you're running from the project root directory")
        print("   2. Check that all required packages are installed")
        print("   3. Verify the src/api/main.py file exists")
        print("   4. Try: python -m src.api.robust_server")
        print("\n📄 Current directory structure:")
        os.system("ls -la src/api/")
        return 0
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        print("💡 Troubleshooting tips:")
        print("   1. Check GPU memory: nvidia-smi")
        print("   2. Run performance optimizer: python tools/performance_optimizer.py")
        print("   3. Check model files in ./models/ directory")
        print("   4. Verify internet connection for model downloads")
        # Don't exit with error code to prevent script from exiting
        return 0

    print("\n✅ Server shutdown complete")
    return 0


if __name__ == "__main__":
    # Run main and handle any uncaught exceptions gracefully
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n💥 Uncaught exception: {e}")
        sys.exit(0)  # Exit gracefully to prevent abrupt termination
