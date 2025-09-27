#!/usr/bin/env python3
"""
Simple HTTP server to serve frontend files
This avoids CORS issues with file:// protocol
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

PORT = 3001

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.getcwd(), **kwargs)

def serve_frontend():
    """Serve frontend files on localhost:3000"""
    
    # Check if frontend files exist
    frontend_files = [
        "clean_frontend.html",
        "simple_frontend.html", 
        "enhanced_frontend.html",
        "docker_frontend.html"
    ]
    
    available_files = [f for f in frontend_files if Path(f).exists()]
    
    if not available_files:
        print("❌ No frontend files found!")
        return
    
    print(f"🌐 Starting frontend server on http://localhost:{PORT}")
    print(f"📁 Available frontend files:")
    for file in available_files:
        print(f"   • http://localhost:{PORT}/{file}")
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"\n🚀 Server running at http://localhost:{PORT}")
            print("Press Ctrl+C to stop")
            
            # Open browser to clean frontend
            if "clean_frontend.html" in available_files:
                webbrowser.open(f"http://localhost:{PORT}/clean_frontend.html")
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {PORT} is already in use. Try a different port.")
        else:
            print(f"❌ Server error: {e}")

if __name__ == "__main__":
    serve_frontend()