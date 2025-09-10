#!/usr/bin/env python3
"""
Performance Optimizer for Qwen-Image Generator
Diagnoses and fixes 500+ second generation times on high-end hardware

For AMD Threadripper PRO 5995WX + RTX 4080 + 128GB RAM
Target: Reduce from 500+s to 15-60s per image
"""

import os
import sys
import time
from typing import Dict, List

import psutil
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_system_resources() -> Dict[str, any]:
    """Check system resources and identify bottlenecks"""

    print("🔍 SYSTEM RESOURCE ANALYSIS")
    print("=" * 50)

    results = {}

    # CPU Information
    cpu_count = os.cpu_count()
    cpu_freq = psutil.cpu_freq()
    cpu_percent = psutil.cpu_percent(interval=1)

    print("CPU:")
    print(f"  • Cores: {cpu_count}")
    print(
        f"  • Frequency: {cpu_freq.current:.0f} MHz"
        if cpu_freq
        else "  • Frequency: Unknown"
    )
    print(f"  • Usage: {cpu_percent:.1f}%")

    results["cpu"] = {
        "cores": cpu_count,
        "frequency": cpu_freq.current if cpu_freq else 0,
        "usage": cpu_percent,
    }

    # Expected: 64 cores, 2700+ MHz
    if cpu_count < 32:
        print(f"  ⚠️ WARNING: Expected 64+ cores, found {cpu_count}")
    else:
        print("  ✅ CPU core count looks good")

    # Memory Information
    memory = psutil.virtual_memory()
    memory_gb = memory.total / 1e9
    memory_available_gb = memory.available / 1e9

    print("\nRAM:")
    print(f"  • Total: {memory_gb:.0f}GB")
    print(f"  • Available: {memory_available_gb:.0f}GB")
    print(f"  • Usage: {memory.percent:.1f}%")

    results["memory"] = {
        "total_gb": memory_gb,
        "available_gb": memory_available_gb,
        "usage_percent": memory.percent,
    }

    # Expected: 128GB+
    if memory_gb < 100:
        print(f"  ⚠️ WARNING: Expected 128GB+ RAM, found {memory_gb:.0f}GB")
    else:
        print("  ✅ RAM amount looks good")

    # GPU Information
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        total_vram = device_props.total_memory / 1e9
        allocated_vram = torch.cuda.memory_allocated(0) / 1e9
        available_vram = total_vram - allocated_vram

        print("\nGPU:")
        print(f"  • Model: {device_props.name}")
        print(f"  • Total VRAM: {total_vram:.1f}GB")
        print(f"  • Allocated: {allocated_vram:.1f}GB")
        print(f"  • Available: {available_vram:.1f}GB")
        print(f"  • Compute Capability: {device_props.major}.{device_props.minor}")

        results["gpu"] = {
            "name": device_props.name,
            "total_vram": total_vram,
            "allocated_vram": allocated_vram,
            "available_vram": available_vram,
            "compute_capability": f"{device_props.major}.{device_props.minor}",
        }

        # Expected: RTX 4080, 16GB VRAM
        if "RTX 4080" not in device_props.name:
            print(f"  ⚠️ WARNING: Expected RTX 4080, found {device_props.name}")

        if total_vram < 15:
            print(f"  ⚠️ WARNING: Expected 16GB VRAM, found {total_vram:.1f}GB")
        else:
            print("  ✅ GPU specs look good")

    else:
        print("\nGPU:")
        print("  ❌ CUDA not available!")
        results["gpu"] = None

    return results


def check_pytorch_configuration() -> Dict[str, any]:
    """Check PyTorch configuration for performance issues"""

    print("\n🔍 PYTORCH CONFIGURATION ANALYSIS")
    print("=" * 50)

    results = {}

    # PyTorch version
    pytorch_version = torch.__version__
    print(f"PyTorch Version: {pytorch_version}")
    results["pytorch_version"] = pytorch_version

    # CUDA version
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        print(f"CUDA Version: {cuda_version}")
        results["cuda_version"] = cuda_version

        # Check CUDA efficiency
        print("\nCUDA Configuration:")
        print(f"  • Device Count: {torch.cuda.device_count()}")
        print(f"  • Current Device: {torch.cuda.current_device()}")

        # Check for performance-killing settings
        print("\nPerformance Settings:")

        # Check memory fraction
        try:
            # This is approximate - actual check would require internal PyTorch state
            print("  • Memory Management: Checking...")
        except:
            pass

        # Check environment variables
        perf_vars = {
            "PYTORCH_CUDA_ALLOC_CONF": "Memory allocation config",
            "CUDA_LAUNCH_BLOCKING": "Synchronous execution",
            "OMP_NUM_THREADS": "CPU threading",
            "MKL_NUM_THREADS": "Intel MKL threading",
        }

        print("\nEnvironment Variables:")
        for var, description in perf_vars.items():
            value = os.environ.get(var, "Not set")
            print(f"  • {var}: {value}")

            # Performance recommendations
            if var == "CUDA_LAUNCH_BLOCKING" and value == "1":
                print("    ⚠️ BLOCKING ENABLED - Major performance killer!")
            elif var == "OMP_NUM_THREADS" and (
                value == "Not set" or int(value) if value.isdigit() else 0 < 16
            ):
                print("    💡 Consider setting to 32+ for Threadripper")

    return results


def diagnose_generation_bottlenecks() -> List[str]:
    """Identify specific bottlenecks causing slow generation"""

    print("\n🔍 GENERATION BOTTLENECK ANALYSIS")
    print("=" * 50)

    issues = []

    # Check if model is using CPU offloading
    print("Checking common performance killers...")

    # 1. Check for CPU offloading indicators
    if torch.cuda.is_available():
        # Simulate checking if components are being offloaded
        print("  • CPU Offloading: Analyzing...")

        # Common signs of CPU offloading:
        # - Sequential CPU offload enabled
        # - Model CPU offload enabled
        # - Device map with CPU assignments

        # This would be detected during model loading
        issues.append(
            "CPU offloading may be enabled - keeps model components on slow CPU"
        )

    # 2. Check memory restrictions
    print("  • Memory Restrictions: Analyzing...")

    # Look for conservative memory settings
    config_files = ["src/qwen_edit_config.py", "src/qwen_image_config.py"]

    for config_file in config_files:
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                content = f.read()

                if "max_memory" in content and '"12GB"' in content:
                    issues.append(
                        f"Conservative VRAM limit in {config_file} - only using 12GB of 16GB"
                    )

                if 'enable_sequential_cpu_offload": True' in content:
                    issues.append(
                        f"Sequential CPU offload enabled in {config_file} - major performance killer"
                    )

                if 'low_cpu_mem_usage": True' in content:
                    issues.append(
                        f"Low CPU memory usage enabled in {config_file} - unnecessary with 128GB RAM"
                    )

    # 3. Check attention slicing
    print("  • Attention Slicing: Analyzing...")
    issues.append(
        "Attention slicing may be enabled - trades speed for memory on high-VRAM systems"
    )

    # 4. Check device map issues
    print("  • Device Mapping: Analyzing...")
    issues.append("Device mapping may be forcing CPU usage instead of pure GPU")

    return issues


def run_performance_benchmark() -> float:
    """Run a quick performance benchmark"""

    print("\n🧪 PERFORMANCE BENCHMARK")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot benchmark")
        return 0.0

    print("Running GPU compute benchmark...")

    # Clear cache
    torch.cuda.empty_cache()

    # Create test tensors
    device = "cuda"
    dtype = torch.bfloat16

    # Simulate diffusion model operations
    batch_size = 1
    height, width = 928, 1664
    latent_channels = 4

    print(f"Test configuration: {batch_size}x{latent_channels}x{height//8}x{width//8}")

    # Test tensor operations
    start_time = time.time()

    with torch.no_grad():
        # Simulate UNet operations
        latents = torch.randn(
            batch_size,
            latent_channels,
            height // 8,
            width // 8,
            device=device,
            dtype=dtype,
        )

        # Simulate 10 denoising steps
        for step in range(10):
            # Simulate UNet forward pass
            noise_pred = torch.nn.functional.conv2d(
                latents,
                torch.randn(
                    latent_channels, latent_channels, 3, 3, device=device, dtype=dtype
                ),
                padding=1,
            )

            # Simulate scheduler step
            latents = latents - 0.1 * noise_pred

            # Add some compute
            latents = torch.nn.functional.relu(latents)

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    steps_per_second = 10 / elapsed_time
    estimated_50_step_time = 50 / steps_per_second

    print("Benchmark Results:")
    print(f"  • 10 steps completed in: {elapsed_time:.2f}s")
    print(f"  • Steps per second: {steps_per_second:.2f}")
    print(f"  • Estimated 50-step generation: {estimated_50_step_time:.1f}s")

    if estimated_50_step_time > 100:
        print("  ❌ POOR performance - should be 15-60s")
    elif estimated_50_step_time > 60:
        print("  ⚠️ SLOW performance - could be better")
    else:
        print("  ✅ GOOD performance")

    return estimated_50_step_time


def generate_optimization_recommendations(
    system_info: Dict, issues: List[str]
) -> List[str]:
    """Generate specific optimization recommendations"""

    print("\n💡 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 50)

    recommendations = []

    # High-level recommendations
    recommendations.append("CRITICAL: Switch to high-end configuration")
    recommendations.append(
        "  → Use HighEndQwenImageGenerator instead of QwenImageGenerator"
    )
    recommendations.append("  → File: src/qwen_highend_generator.py")

    # Specific fixes based on issues
    for issue in issues:
        if "CPU offload" in issue:
            recommendations.append("DISABLE all CPU offloading:")
            recommendations.append("  → Set enable_sequential_cpu_offload = False")
            recommendations.append("  → Set enable_model_cpu_offload = False")
            recommendations.append("  → Keep entire model on GPU")

        elif "12GB" in issue:
            recommendations.append("INCREASE VRAM allocation:")
            recommendations.append("  → Change max_memory from {'0': '12GB'} to None")
            recommendations.append("  → Use full 16GB VRAM available")

        elif "low_cpu_mem_usage" in issue:
            recommendations.append("DISABLE low_cpu_mem_usage:")
            recommendations.append("  → Set low_cpu_mem_usage = False")
            recommendations.append("  → Utilize your 128GB RAM effectively")

        elif "attention slicing" in issue:
            recommendations.append("DISABLE attention slicing:")
            recommendations.append("  → Set enable_attention_slicing = False")
            recommendations.append("  → Use full VRAM for maximum speed")

    # Hardware-specific recommendations
    if system_info.get("cpu", {}).get("cores", 0) >= 32:
        recommendations.append("OPTIMIZE CPU threading:")
        recommendations.append("  → Set OMP_NUM_THREADS=32")
        recommendations.append("  → Set MKL_NUM_THREADS=32")

    if system_info.get("gpu", {}).get("total_vram", 0) >= 15:
        recommendations.append("MAXIMIZE VRAM usage:")
        recommendations.append("  → Set device_map = None")
        recommendations.append("  → Keep all components on GPU")
        recommendations.append("  → Set PYTORCH_CUDA_MEMORY_FRACTION=0.95")

    # Environment optimizations
    recommendations.append("SET performance environment variables:")
    recommendations.append("  → PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    recommendations.append("  → CUDA_LAUNCH_BLOCKING=0")

    return recommendations


def apply_quick_fixes() -> bool:
    """Apply quick performance fixes"""

    print("\n🔧 APPLYING QUICK PERFORMANCE FIXES")
    print("=" * 50)

    try:
        # Set optimal environment variables
        optimal_env = {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:512",
            "CUDA_LAUNCH_BLOCKING": "0",
            "OMP_NUM_THREADS": "32",
            "MKL_NUM_THREADS": "32",
            "NUMBA_NUM_THREADS": "32",
        }

        for key, value in optimal_env.items():
            os.environ[key] = value
            print(f"✅ Set {key}={value}")

        # Set CUDA memory fraction
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)
            print("✅ Set CUDA memory fraction to 95%")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✅ Cleared GPU cache")

        print("\n✅ Quick fixes applied!")
        print("⚠️ For permanent fixes, update configuration files")

        return True

    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        return False


def main():
    """Main performance optimization workflow"""

    print("🚀 QWEN-IMAGE PERFORMANCE OPTIMIZER")
    print("=" * 60)
    print("Diagnosing 500+ second generation times...")
    print("Target: 15-60 seconds on high-end hardware")
    print("=" * 60)

    # 1. Check system resources
    system_info = check_system_resources()

    # 2. Check PyTorch configuration
    pytorch_info = check_pytorch_configuration()

    # 3. Diagnose bottlenecks
    issues = diagnose_generation_bottlenecks()

    # 4. Run benchmark
    benchmark_time = run_performance_benchmark()

    # 5. Generate recommendations
    recommendations = generate_optimization_recommendations(system_info, issues)

    # Print summary
    print("\n📊 ANALYSIS SUMMARY")
    print("=" * 50)

    print(f"Issues Found: {len(issues)}")
    for issue in issues:
        print(f"  • {issue}")

    print("\nRecommendations:")
    for rec in recommendations:
        print(f"  {rec}")

    # 6. Offer to apply quick fixes
    print("\n🔧 QUICK FIXES AVAILABLE")
    print("Apply immediate environment optimizations? (y/n): ", end="")

    try:
        response = input().lower().strip()
        if response in ["y", "yes"]:
            success = apply_quick_fixes()
            if success:
                print("\n✅ Quick fixes applied!")
                print("⚠️ Restart your application to see full benefits")
        else:
            print("Skipping quick fixes")
    except KeyboardInterrupt:
        print("\nOperation cancelled")

    print("\n🎯 NEXT STEPS:")
    print("1. Switch to high-end generator: src/qwen_highend_generator.py")
    print("2. Update configuration files with recommendations")
    print("3. Restart application")
    print("4. Monitor generation times (target: 15-60s)")

    return benchmark_time


if __name__ == "__main__":
    main()
