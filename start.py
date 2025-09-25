#!/usr/bin/env python3
"""
Qwen-Image Launcher
Simple launcher with options for different deployment modes
"""

import sys
import subprocess

def show_menu():
    print("🎨 Qwen-Image Generator")
    print("=" * 40)
    print("")
    print("Choose how to start:")
    print("")
    print("1️⃣  Backend Only")
    print("   • API server on http://localhost:8000")
    print("   • For API testing or custom frontends")
    print("")
    print("2️⃣  Frontend Only") 
    print("   • React app on http://localhost:3000")
    print("   • Requires backend running separately")
    print("")
    print("3️⃣  Full Stack (Local)")
    print("   • Start backend and frontend separately")
    print("   • Backend: http://localhost:8000")
    print("   • Frontend: http://localhost:3000")
    print("")
    print("4️⃣  Full Stack (Traefik)")
    print("   • Complete Docker setup with Traefik")
    print("   • API: http://api.localhost")
    print("   • Frontend: http://qwen.localhost")
    print("   • Dashboard: http://traefik.localhost:8080")
    print("")
    print("0️⃣  Exit")
    print("")

def main():
    while True:
        show_menu()
        try:
            choice = input("🎯 Enter your choice (1-4, 0 to exit): ").strip()
            
            if choice == '1':
                print("\n🚀 Starting Backend Server...")
                subprocess.run([sys.executable, 'start_backend.py'])
                break
                
            elif choice == '2':
                print("\n🎨 Starting Frontend Server...")
                subprocess.run([sys.executable, 'start_frontend.py'])
                break
                
            elif choice == '3':
                print("\n🔄 Starting Full Stack (Local)...")
                print("💡 This will start backend first, then frontend")
                print("   You can also run them separately in different terminals:")
                print("   Terminal 1: python start_backend.py")
                print("   Terminal 2: python start_frontend.py")
                print("")
                
                response = input("Continue? (y/n): ").lower().strip()
                if response == 'y':
                    print("\n🚀 Starting backend server...")
                    print("💡 After backend starts, open another terminal and run:")
                    print("   python start_frontend.py")
                    subprocess.run([sys.executable, 'start_backend.py'])
                break
                
            elif choice == '4':
                print("\n🐳 Starting Full Stack with Traefik...")
                subprocess.run([sys.executable, 'start_with_traefik.py'])
                break
                
            elif choice == '0':
                print("\n👋 Goodbye!")
                sys.exit(0)
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 0.")
                input("Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)
        except EOFError:
            print("\n\n👋 Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    main()