#!/usr/bin/env python3
"""
Frontend Server Launcher
Starts the React frontend for Qwen-Image
"""

import os
import sys
import subprocess
import time

def check_node():
    """Check if Node.js is available"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            return True, result.stdout.strip()
    except FileNotFoundError:
        pass
    return False, None

def check_npm():
    """Check if npm is available"""
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            return True, result.stdout.strip()
    except FileNotFoundError:
        pass
    return False, None

def main():
    print("🎨 Starting Qwen-Image React Frontend")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("frontend/package.json"):
        print("❌ Error: frontend/package.json not found")
        print("💡 Make sure you're in the project root directory")
        sys.exit(1)
    
    # Check Node.js
    node_available, node_version = check_node()
    if not node_available:
        print("❌ Error: Node.js not found")
        print("💡 Please install Node.js: https://nodejs.org/")
        sys.exit(1)
    
    # Check npm
    npm_available, npm_version = check_npm()
    if not npm_available:
        print("❌ Error: npm not found")
        print("💡 npm should come with Node.js installation")
        sys.exit(1)
    
    print(f"✅ Node.js: {node_version}")
    print(f"✅ npm: {npm_version}")
    print("")
    
    # Change to frontend directory
    os.chdir("frontend")
    
    # Check if node_modules exists
    if not os.path.exists("node_modules"):
        print("📦 Installing dependencies...")
        try:
            subprocess.run(['npm', 'install', '--legacy-peer-deps'], check=True)
            print("✅ Dependencies installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            print("💡 Try manually: cd frontend && npm install --legacy-peer-deps")
            sys.exit(1)
    
    print("📋 Frontend Configuration:")
    print("   • Development Server: http://localhost:3001")
    print("   • Traefik Access: http://qwen.localhost (if using Docker)")
    print("   • API Backend: http://localhost:8000")
    print("   • Hot Reload: Enabled")
    print("")
    
    print("🎯 Starting React development server...")
    print("   Press Ctrl+C to stop")
    print("")
    
    try:
        # Start the React development server
        subprocess.run(['npm', 'start'])
    except KeyboardInterrupt:
        print("\n👋 Frontend server stopped")
    except Exception as e:
        print(f"\n❌ Error starting frontend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()