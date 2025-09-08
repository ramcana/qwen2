#!/usr/bin/env python3
"""
Unified launcher for Qwen-Image Generator
Supports both Gradio and React UIs with FastAPI backend
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

try:
    from ui_config_manager import UIConfigManager
except ImportError:
    print("⚠️ UI Config Manager not found. Using fallback configuration.")
    UIConfigManager = None


def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    missing_deps = []
    
    # Check Python packages
    try:
        import fastapi
        import gradio
        import uvicorn
        print("✅ FastAPI, Uvicorn, and Gradio found")
    except ImportError as e:
        missing_deps.append(f"Python packages: {e}")
    
    # Check Node.js for React
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js found: {result.stdout.strip()}")
        else:
            missing_deps.append("Node.js not available")
    except FileNotFoundError:
        missing_deps.append("Node.js not installed")
    
    # Check if frontend directory exists
    frontend_dir = Path("frontend")
    if frontend_dir.exists():
        package_json = frontend_dir / "package.json"
        if package_json.exists():
            print("✅ React frontend found")
        else:
            missing_deps.append("React frontend package.json missing")
    else:
        print("⚠️ React frontend directory not found")
    
    if missing_deps:
        print("\n❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\n💡 Installation guide:")
        print("  1. Install Python packages: pip install -r requirements.txt")
        print("  2. Install Node.js: https://nodejs.org/")
        print("  3. Setup React frontend: cd frontend && npm install")
        return False
    
    print("✅ All dependencies satisfied")
    return True


def install_react_dependencies():
    """Install React frontend dependencies"""
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    package_json = frontend_dir / "package.json"
    if not package_json.exists():
        print("❌ package.json not found in frontend directory")
        return False
    
    print("📦 Installing React dependencies...")
    try:
        result = subprocess.run(
            ["npm", "install"],
            cwd=frontend_dir,
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ React dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install React dependencies: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False


def launch_api(config_manager=None, background=True):
    """Launch FastAPI backend"""
    if config_manager and not config_manager.is_api_enabled():
        print("⚠️ API is disabled in configuration")
        return None
    
    api_config = config_manager.get_api_config() if config_manager else {"port": 8000, "host": "0.0.0.0"}
    
    print(f"📡 Starting FastAPI server on http://{api_config['host']}:{api_config['port']}")
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.api.main:app",
        "--host", api_config["host"],
        "--port", str(api_config["port"]),
        "--reload"
    ]
    
    if background:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"✅ API started in background (PID: {process.pid})")
        return process
    else:
        print("Running API in foreground...")
        subprocess.run(cmd)
        return None


def launch_gradio(config_manager=None, background=True):
    """Launch Gradio UI"""
    if config_manager and not config_manager.is_gradio_enabled():
        print("⚠️ Gradio UI is disabled in configuration")
        return None
    
    gradio_config = config_manager.get_gradio_config() if config_manager else {"port": 7860}
    
    print(f"🎨 Starting Gradio UI on http://localhost:{gradio_config['port']}")
    
    cmd = [sys.executable, "src/qwen_image_ui.py"]
    
    if background:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"✅ Gradio started in background (PID: {process.pid})")
        return process
    else:
        print("Running Gradio in foreground...")
        subprocess.run(cmd)
        return None


def launch_react(config_manager=None, background=True):
    """Launch React development server"""
    if config_manager and not config_manager.is_react_enabled():
        print("⚠️ React UI is disabled in configuration")
        return None
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return None
    
    react_config = config_manager.get_react_config() if config_manager else {"dev_port": 3000}
    
    print(f"⚛️  Starting React development server on http://localhost:{react_config['dev_port']}")
    
    # Set environment variables for React
    env = os.environ.copy()
    env["PORT"] = str(react_config["dev_port"])
    env["BROWSER"] = "none"  # Don't auto-open browser
    
    cmd = ["npm", "start"]
    
    if background:
        process = subprocess.Popen(
            cmd,
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        print(f"✅ React started in background (PID: {process.pid})")
        return process
    else:
        print("Running React in foreground...")
        subprocess.run(cmd, cwd=frontend_dir, env=env)
        return None


def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Qwen-Image Generator Unified Launcher")
    
    parser.add_argument("--mode", choices=["gradio", "react", "both", "api-only"], 
                       default="both", help="UI mode to launch")
    parser.add_argument("--foreground", action="store_true", 
                       help="Run in foreground (default: background)")
    parser.add_argument("--no-api", action="store_true", 
                       help="Don't start API server")
    parser.add_argument("--install-deps", action="store_true", 
                       help="Install React dependencies")
    parser.add_argument("--check-deps", action="store_true", 
                       help="Check dependencies and exit")
    parser.add_argument("--config", action="store_true", 
                       help="Show current configuration")
    
    args = parser.parse_args()
    
    # Load configuration if available
    config_manager = None
    if UIConfigManager:
        try:
            config_manager = UIConfigManager()
        except Exception as e:
            print(f"⚠️ Error loading config manager: {e}")
    
    # Handle special commands
    if args.check_deps:
        check_dependencies()
        return
    
    if args.install_deps:
        install_react_dependencies()
        return
    
    if args.config:
        if config_manager:
            config_manager.print_status()
        else:
            print("❌ Config manager not available")
        return
    
    print("🎨 Qwen-Image Generator Unified Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n💡 Run with --install-deps to install React dependencies")
        return
    
    background = not args.foreground
    processes = []
    
    try:
        # Launch API (unless disabled)
        if not args.no_api and args.mode != "gradio":
            api_process = launch_api(config_manager, background)
            if api_process:
                processes.append(("API", api_process))
                time.sleep(2)  # Give API time to start
        
        # Launch based on mode
        if args.mode in ["gradio", "both"]:
            gradio_process = launch_gradio(config_manager, background)
            if gradio_process:
                processes.append(("Gradio", gradio_process))
        
        if args.mode in ["react", "both"]:
            react_process = launch_react(config_manager, background)
            if react_process:
                processes.append(("React", react_process))
        
        if args.mode == "api-only":
            if args.no_api:
                print("❌ Cannot run API-only mode with --no-api flag")
                return
            # API already started above
        
        if background and processes:
            print(f"\n✅ Started {len(processes)} background processes")
            print("\n🌐 Access URLs:")
            
            if not args.no_api:
                api_port = config_manager.get_api_config()["port"] if config_manager else 8000
                print(f"📡 API: http://localhost:{api_port}")
                print(f"📚 API Docs: http://localhost:{api_port}/docs")
            
            if args.mode in ["gradio", "both"]:
                gradio_port = config_manager.get_gradio_config()["port"] if config_manager else 7860
                print(f"🎨 Gradio: http://localhost:{gradio_port}")
            
            if args.mode in ["react", "both"]:
                react_port = config_manager.get_react_config()["dev_port"] if config_manager else 3000
                print(f"⚛️  React: http://localhost:{react_port}")
            
            print("\n💡 Tips:")
            print("  - Use Ctrl+C to stop all processes")
            print("  - Check logs with: python launch_unified.py --check-deps")
            print("  - Configure UIs with: python src/ui_config_manager.py status")
            
            # Wait for processes or user interrupt
            try:
                while any(proc.poll() is None for _, proc in processes):
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping all processes...")
                for name, proc in processes:
                    proc.terminate()
                    print(f"Stopped {name}")
                    
                # Wait a bit for graceful shutdown
                time.sleep(2)
                
                # Force kill if needed
                for name, proc in processes:
                    if proc.poll() is None:
                        proc.kill()
                        print(f"Force killed {name}")
        
        elif not background:
            print("✅ All services started in foreground mode")
    
    except Exception as e:
        print(f"❌ Error during launch: {e}")
        # Cleanup on error
        for name, proc in processes:
            try:
                proc.terminate()
            except:
                pass


if __name__ == "__main__":
    main()