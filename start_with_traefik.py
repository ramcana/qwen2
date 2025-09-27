#!/usr/bin/env python3
"""
Full Stack Launcher with Traefik
Starts the complete Qwen-Image stack using Docker Compose + Traefik
"""

import os
import sys
import subprocess
import time

def check_docker():
    """Check if Docker and Docker Compose are available"""
    try:
        # Check Docker
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            return False, "Docker not found"
        
        # Check Docker Compose
        result = subprocess.run(['docker', 'compose', 'version'], capture_output=True, text=True)
        if result.returncode != 0:
            # Try old docker-compose command
            result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                return False, "Docker Compose not found"
        
        return True, "Docker and Docker Compose available"
    except FileNotFoundError:
        return False, "Docker not found"

def setup_hosts():
    """Setup local hosts for Traefik routing"""
    hosts_entries = [
        "127.0.0.1 api.localhost",
        "127.0.0.1 qwen.localhost", 
        "127.0.0.1 frontend.localhost",
        "127.0.0.1 traefik.localhost"
    ]
    
    print("🔧 Setting up local hosts...")
    
    try:
        # Check if entries already exist
        with open('/etc/hosts', 'r') as f:
            hosts_content = f.read()
        
        missing_entries = []
        for entry in hosts_entries:
            if entry.split()[1] not in hosts_content:
                missing_entries.append(entry)
        
        if missing_entries:
            print("   Adding missing host entries (requires sudo)...")
            for entry in missing_entries:
                subprocess.run(['sudo', 'sh', '-c', f'echo "{entry}" >> /etc/hosts'], check=True)
            print("✅ Host entries added")
        else:
            print("✅ Host entries already exist")
            
    except Exception as e:
        print(f"⚠️ Could not setup hosts automatically: {e}")
        print("💡 Please add these entries to /etc/hosts manually:")
        for entry in hosts_entries:
            print(f"   {entry}")

def main():
    print("🐳 Starting Qwen-Image with Traefik")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("docker-compose.yml"):
        print("❌ Error: docker-compose.yml not found")
        print("💡 Make sure you're in the project root directory")
        sys.exit(1)
    
    # Check Docker
    docker_available, docker_message = check_docker()
    if not docker_available:
        print(f"❌ Error: {docker_message}")
        print("💡 Please install Docker and Docker Compose")
        sys.exit(1)
    
    print(f"✅ {docker_message}")
    
    # Setup hosts
    setup_hosts()
    
    # Ensure acme.json has correct permissions
    if os.path.exists("acme.json"):
        os.chmod("acme.json", 0o600)
    
    print("\n📋 Stack Configuration:")
    print("   • Traefik Dashboard: http://traefik.localhost:8080")
    print("   • API Server: http://api.localhost")
    print("   • Frontend: http://qwen.localhost")
    print("   • Direct API: http://localhost:8000 (if port exposed)")
    print("")
    
    print("🎯 Starting Docker Compose stack...")
    print("   Press Ctrl+C to stop")
    print("")
    
    try:
        # Start with Docker Compose
        subprocess.run(['docker', 'compose', 'up', '--build'])
    except KeyboardInterrupt:
        print("\n🛑 Stopping stack...")
        subprocess.run(['docker', 'compose', 'down'])
        print("👋 Stack stopped")
    except Exception as e:
        print(f"\n❌ Error starting stack: {e}")
        print("💡 Try manually: docker compose up --build")
        sys.exit(1)

if __name__ == "__main__":
    main()