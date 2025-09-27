#!/usr/bin/env python3
"""
Backend Server Launcher
Starts the Qwen-Image API server with performance optimizations
"""

import os
import sys
import subprocess

def main():
    print("🚀 Starting Qwen-Image Backend Server")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("src/api_server_diffsynth.py"):
        print("❌ Error: src/api_server_diffsynth.py not found")
        print("💡 Make sure you're in the project root directory")
        sys.exit(1)
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️ Warning: Virtual environment not detected")
        print("💡 Consider activating venv: source venv/bin/activate")
    
    print("📋 Server Configuration:")
    print("   • Direct Access: http://localhost:8000")
    print("   • Traefik Access: http://api.localhost (if using Docker)")
    print("   • API Docs: http://localhost:8000/docs")
    print("   • Performance: RTX 4080 optimized")
    print("   • Services: DiffSynth + ControlNet + Qwen-Image")
    print("   • Models: Load on first request")
    print("")
    
    print("🎯 Starting server...")
    print("   Press Ctrl+C to stop")
    print("")
    
    try:
        # Start the DiffSynth API server
        subprocess.run([
            sys.executable, 
            "src/api_server_diffsynth.py"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()