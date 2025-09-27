#!/usr/bin/env python3
"""
Feature Flag System for UI Selection (Gradio vs React)
Allows seamless switching between UI interfaces
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, cast

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_FILE = "config/ui_config.json"
DEFAULT_CONFIG: Dict[str, Any] = {
    "ui_mode": "gradio",  # "gradio" or "react" or "both"
    "gradio": {
        "enabled": True,
        "port": 7860,
        "host": "0.0.0.0",
        "share": False,
        "in_browser": False
    },
    "react": {
        "enabled": False,
        "dev_port": 3001,
        "build_dir": "frontend/build"
    },
    "api": {
        "enabled": True,
        "port": 8000,
        "host": "0.0.0.0",
        "auto_start": True
    },
    "features": {
        "memory_optimization": True,
        "request_queuing": True,
        "performance_monitoring": True,
        "background_cleanup": True
    }
}


class UIConfigManager:
    """Manages UI configuration and feature flags"""
    
    def __init__(self, config_path: str = CONFIG_FILE):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                    return config_data if isinstance(config_data, dict) else DEFAULT_CONFIG.copy()
            except Exception as e:
                print(f"⚠️ Error loading config: {e}")
                print("🔄 Using default configuration")
        
        # Create default config
        self._save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    
    def _save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Error saving config: {e}")
            return False
    
    def get_ui_mode(self) -> str:
        """Get current UI mode"""
        mode = self.config.get("ui_mode", "gradio")
        return str(mode)
    
    def set_ui_mode(self, mode: str) -> bool:
        """Set UI mode (gradio, react, or both)"""
        if mode not in ["gradio", "react", "both"]:
            print(f"❌ Invalid UI mode: {mode}")
            return False
        
        self.config["ui_mode"] = mode
        
        # Update component states based on mode
        if mode == "gradio":
            self.config["gradio"]["enabled"] = True
            self.config["react"]["enabled"] = False
        elif mode == "react":
            self.config["gradio"]["enabled"] = False
            self.config["react"]["enabled"] = True
        elif mode == "both":
            self.config["gradio"]["enabled"] = True
            self.config["react"]["enabled"] = True
        
        return self._save_config(self.config)
    
    def is_gradio_enabled(self) -> bool:
        """Check if Gradio UI is enabled"""
        gradio_config = self.config.get("gradio", {})
        if isinstance(gradio_config, dict):
            return bool(gradio_config.get("enabled", True))
        return True
    
    def is_react_enabled(self) -> bool:
        """Check if React UI is enabled"""
        react_config = self.config.get("react", {})
        if isinstance(react_config, dict):
            return bool(react_config.get("enabled", False))
        return False
    
    def is_api_enabled(self) -> bool:
        """Check if API is enabled"""
        api_config = self.config.get("api", {})
        if isinstance(api_config, dict):
            return bool(api_config.get("enabled", True))
        return True
    
    def get_gradio_config(self) -> Dict[str, Any]:
        """Get Gradio configuration"""
        gradio_config = self.config.get("gradio")
        if isinstance(gradio_config, dict):
            return gradio_config
        return cast(Dict[str, Any], DEFAULT_CONFIG["gradio"]).copy()
    
    def get_react_config(self) -> Dict[str, Any]:
        """Get React configuration"""
        react_config = self.config.get("react")
        if isinstance(react_config, dict):
            return react_config
        return cast(Dict[str, Any], DEFAULT_CONFIG["react"]).copy()
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        api_config = self.config.get("api")
        if isinstance(api_config, dict):
            return api_config
        return cast(Dict[str, Any], DEFAULT_CONFIG["api"]).copy()
    
    def get_feature_flags(self) -> Dict[str, Any]:
        """Get feature flags"""
        features = self.config.get("features")
        if isinstance(features, dict):
            return features
        return cast(Dict[str, Any], DEFAULT_CONFIG["features"]).copy()
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a specific feature is enabled"""
        feature_flags = self.get_feature_flags()
        return bool(feature_flags.get(feature, False))
    
    def set_feature(self, feature: str, enabled: bool) -> bool:
        """Enable or disable a feature"""
        if "features" not in self.config:
            self.config["features"] = {}
        
        self.config["features"][feature] = enabled
        return self._save_config(self.config)
    
    def print_status(self):
        """Print current configuration status"""
        print("🎛️  UI Configuration Status")
        print("=" * 40)
        print(f"UI Mode: {self.get_ui_mode()}")
        print(f"Gradio: {'✅ Enabled' if self.is_gradio_enabled() else '❌ Disabled'}")
        print(f"React: {'✅ Enabled' if self.is_react_enabled() else '❌ Disabled'}")
        print(f"API: {'✅ Enabled' if self.is_api_enabled() else '❌ Disabled'}")
        
        print("\n🚀 Ports:")
        if self.is_gradio_enabled():
            gradio_config = self.get_gradio_config()
            print(f"Gradio: http://localhost:{gradio_config['port']}")
        if self.is_react_enabled():
            react_config = self.get_react_config()
            print(f"React: http://localhost:{react_config['dev_port']}")
        if self.is_api_enabled():
            api_config = self.get_api_config()
            print(f"API: http://localhost:{api_config['port']}")
        
        print("\n⚡ Features:")
        for feature, enabled in self.get_feature_flags().items():
            status = "✅" if enabled else "❌"
            print(f"{status} {feature.replace('_', ' ').title()}")


def main():
    """Command-line interface for UI configuration"""
    parser = argparse.ArgumentParser(description="Qwen-Image UI Configuration Manager")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Status command
    subparsers.add_parser("status", help="Show current configuration")
    
    # Set UI mode
    mode_parser = subparsers.add_parser("set-mode", help="Set UI mode")
    mode_parser.add_argument("mode", choices=["gradio", "react", "both"], help="UI mode to set")
    
    # Feature management
    feature_parser = subparsers.add_parser("feature", help="Manage features")
    feature_parser.add_argument("name", help="Feature name")
    feature_parser.add_argument("action", choices=["enable", "disable"], help="Action to perform")
    
    # Port configuration
    port_parser = subparsers.add_parser("set-port", help="Set port for service")
    port_parser.add_argument("service", choices=["gradio", "react", "api"], help="Service to configure")
    port_parser.add_argument("port", type=int, help="Port number")
    
    # Launch command
    launch_parser = subparsers.add_parser("launch", help="Launch configured UIs")
    launch_parser.add_argument("--background", action="store_true", help="Launch in background")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    config_manager = UIConfigManager()
    
    if args.command == "status":
        config_manager.print_status()
    
    elif args.command == "set-mode":
        if config_manager.set_ui_mode(args.mode):
            print(f"✅ UI mode set to: {args.mode}")
            config_manager.print_status()
        else:
            print("❌ Failed to set UI mode")
    
    elif args.command == "feature":
        enabled = args.action == "enable"
        if config_manager.set_feature(args.name, enabled):
            status = "enabled" if enabled else "disabled"
            print(f"✅ Feature '{args.name}' {status}")
        else:
            print(f"❌ Failed to {args.action} feature '{args.name}'")
    
    elif args.command == "set-port":
        service_key = args.service
        if service_key in config_manager.config:
            port_key = "port" if args.service == "api" else ("port" if args.service == "gradio" else "dev_port")
            config_manager.config[service_key][port_key] = args.port
            
            if config_manager._save_config(config_manager.config):
                print(f"✅ {args.service.title()} port set to: {args.port}")
            else:
                print(f"❌ Failed to set {args.service} port")
    
    elif args.command == "launch":
        launch_configured_uis(config_manager, background=args.background)


def launch_configured_uis(config_manager: UIConfigManager, background: bool = False):
    """Launch UIs based on current configuration"""
    import subprocess
    import time
    
    print("🚀 Launching configured UIs...")
    
    processes = []
    
    # Launch API if enabled
    if config_manager.is_api_enabled():
        api_config = config_manager.get_api_config()
        print(f"📡 Starting API server on port {api_config['port']}...")
        
        api_cmd = [
            sys.executable, "-m", "uvicorn",
            "src.api.main:app",
            "--host", api_config["host"],
            "--port", str(api_config["port"]),
            "--reload"
        ]
        
        if background:
            process = subprocess.Popen(api_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append(("API", process))
        else:
            print("Running API in foreground...")
            subprocess.run(api_cmd)
    
    # Launch Gradio if enabled
    if config_manager.is_gradio_enabled():
        gradio_config = config_manager.get_gradio_config()
        print(f"🎨 Starting Gradio UI on port {gradio_config['port']}...")
        
        gradio_cmd = [sys.executable, "src/qwen_image_ui.py"]
        
        if background:
            process = subprocess.Popen(gradio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append(("Gradio", process))
        else:
            print("Running Gradio in foreground...")
            subprocess.run(gradio_cmd)
    
    # Launch React if enabled (in development mode)
    if config_manager.is_react_enabled():
        react_config = config_manager.get_react_config()
        print(f"⚛️  Starting React UI on port {react_config['dev_port']}...")
        
        # Check if Node.js is available
        try:
            subprocess.run(["node", "--version"], check=True, capture_output=True)
            
            react_cmd = ["npm", "start"]
            
            if background:
                process = subprocess.Popen(
                    react_cmd, 
                    cwd="frontend",
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                processes.append(("React", process))
            else:
                print("Running React in foreground...")
                subprocess.run(react_cmd, cwd="frontend")
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ Node.js not found. Please install Node.js to run React UI.")
    
    if background and processes:
        print(f"\n✅ Started {len(processes)} background processes")
        print("\n🌐 Access URLs:")
        
        if config_manager.is_api_enabled():
            api_port = config_manager.get_api_config()["port"]
            print(f"📡 API: http://localhost:{api_port}")
            print(f"📚 API Docs: http://localhost:{api_port}/docs")
        
        if config_manager.is_gradio_enabled():
            gradio_port = config_manager.get_gradio_config()["port"]
            print(f"🎨 Gradio: http://localhost:{gradio_port}")
        
        if config_manager.is_react_enabled():
            react_port = config_manager.get_react_config()["dev_port"]
            print(f"⚛️  React: http://localhost:{react_port}")
        
        print("\n💡 To stop all processes, press Ctrl+C")
        
        try:
            # Wait for processes
            while any(proc.poll() is None for _, proc in processes):
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping all processes...")
            for name, proc in processes:
                proc.terminate()
                print(f"Stopped {name}")


if __name__ == "__main__":
    main()