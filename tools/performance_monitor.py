#!/usr/bin/env python3
"""
Real-time performance monitoring for Qwen-Image generation
"""
import time
import torch
import psutil
import os
from pathlib import Path

def monitor_gpu_usage():
    """Monitor GPU memory and utilization"""
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    try:
        # Get GPU memory info
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        return {
            "gpu_available": True,
            "memory_allocated_gb": memory_allocated,
            "memory_reserved_gb": memory_reserved,
            "memory_total_gb": memory_total,
            "memory_usage_percent": (memory_allocated / memory_total) * 100
        }
    except Exception as e:
        return {"gpu_available": True, "error": str(e)}

def monitor_system_resources():
    """Monitor CPU and RAM usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    return {
        "cpu_percent": cpu_percent,
        "ram_total_gb": memory.total / (1024**3),
        "ram_used_gb": memory.used / (1024**3),
        "ram_available_gb": memory.available / (1024**3),
        "ram_usage_percent": memory.percent
    }

def check_generation_progress():
    """Check if generation is running by looking for output files"""
    output_files = [
        "generated_smoke_test.jpg",
        "edited_smoke_test.jpg"
    ]
    
    for output_file in output_files:
        if Path(output_file).exists():
            stat = Path(output_file).stat()
            return {
                "output_found": True,
                "file": output_file,
                "size_kb": stat.st_size / 1024,
                "modified": time.ctime(stat.st_mtime)
            }
    
    return {"output_found": False}

def main():
    """Main monitoring loop"""
    print("🔍 Qwen-Image Performance Monitor")
    print("=" * 50)
    
    start_time = time.time()
    
    while True:
        try:
            # Clear screen (optional)
            # os.system('clear' if os.name == 'posix' else 'cls')
            
            current_time = time.time()
            elapsed = current_time - start_time
            
            print(f"\n⏱️  Elapsed Time: {elapsed:.1f}s")
            print("-" * 30)
            
            # GPU monitoring
            gpu_info = monitor_gpu_usage()
            if gpu_info["gpu_available"]:
                if "error" not in gpu_info:
                    print(f"🎮 GPU Memory: {gpu_info['memory_allocated_gb']:.2f}GB / {gpu_info['memory_total_gb']:.2f}GB ({gpu_info['memory_usage_percent']:.1f}%)")
                    print(f"🎮 GPU Reserved: {gpu_info['memory_reserved_gb']:.2f}GB")
                else:
                    print(f"🎮 GPU Error: {gpu_info['error']}")
            else:
                print("🎮 GPU: Not available")
            
            # System monitoring
            system_info = monitor_system_resources()
            print(f"💻 CPU Usage: {system_info['cpu_percent']:.1f}%")
            print(f"💾 RAM Usage: {system_info['ram_used_gb']:.2f}GB / {system_info['ram_total_gb']:.2f}GB ({system_info['ram_usage_percent']:.1f}%)")
            
            # Check for output
            progress_info = check_generation_progress()
            if progress_info["output_found"]:
                print(f"✅ Output Generated: {progress_info['file']} ({progress_info['size_kb']:.1f}KB)")
                print(f"📅 Modified: {progress_info['modified']}")
                print("\n🎉 Generation Complete!")
                break
            else:
                print("⏳ Generation in progress...")
            
            # Performance tips
            if elapsed > 60:  # After 1 minute
                print("\n💡 Performance Tips:")
                if gpu_info.get("memory_usage_percent", 0) < 50:
                    print("   - GPU memory usage is low - consider increasing batch size")
                if system_info["cpu_percent"] < 30:
                    print("   - CPU usage is low - model is likely GPU-bound (normal)")
                if elapsed > 300:  # 5 minutes
                    print("   - Generation is taking longer than expected")
                    print("   - First run is slower due to compilation")
            
            time.sleep(5)  # Update every 5 seconds
            
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()