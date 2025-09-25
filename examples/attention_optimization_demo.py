"""
Modern Attention and Memory Optimization Demo
Demonstrates scaled dot-product attention, Flash Attention 2, memory-efficient patterns,
dynamic batch sizing, and torch.compile optimizations for Qwen-Image generation
"""

import os
import sys
import time
import torch
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from attention_optimizer import (
    AttentionOptimizer,
    AttentionConfig,
    MemoryMonitor,
    DynamicBatchSizer,
    create_attention_config,
    create_attention_optimizer,
    PYTORCH_2_AVAILABLE,
    FLASH_ATTENTION_AVAILABLE,
    XFORMERS_AVAILABLE
)

from pipeline_optimizer import PipelineOptimizer, OptimizationConfig


def print_system_info():
    """Print system and optimization capability information"""
    print("🔍 System Information")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {total_memory:.1f}GB")
    
    print(f"PyTorch 2.0+ SDPA: {PYTORCH_2_AVAILABLE}")
    print(f"Flash Attention 2: {FLASH_ATTENTION_AVAILABLE}")
    print(f"xformers: {XFORMERS_AVAILABLE}")
    print()


def demo_attention_config_creation():
    """Demonstrate attention configuration creation"""
    print("🔧 Attention Configuration Demo")
    print("=" * 50)
    
    # Test different optimization levels
    levels = ["ultra_fast", "balanced", "quality", "experimental"]
    
    for level in levels:
        print(f"\n{level.upper()} Configuration:")
        config = create_attention_config(
            architecture="MMDiT",
            optimization_level=level
        )
        
        print(f"  SDPA: {config.use_scaled_dot_product_attention}")
        print(f"  Flash Attention 2: {config.use_flash_attention_2}")
        print(f"  Memory Efficient: {config.use_memory_efficient_attention}")
        print(f"  Dynamic Batching: {config.enable_dynamic_batch_sizing}")
        print(f"  torch.compile: {config.enable_torch_compile}")
        print(f"  Gradient Checkpointing: {config.enable_gradient_checkpointing}")
    
    print()


def demo_memory_monitoring():
    """Demonstrate memory monitoring capabilities"""
    print("💾 Memory Monitoring Demo")
    print("=" * 50)
    
    monitor = MemoryMonitor()
    memory_info = monitor.get_memory_info()
    
    print(f"Total GPU Memory: {memory_info['total']:.1f}GB")
    print(f"Allocated Memory: {memory_info['allocated']:.1f}GB")
    print(f"Cached Memory: {memory_info['cached']:.1f}GB")
    print(f"Free Memory: {memory_info['free']:.1f}GB")
    
    # Test memory availability
    print(f"\nMemory availability tests:")
    print(f"  Can allocate 4GB: {monitor.is_memory_available(4.0)}")
    print(f"  Can allocate 8GB: {monitor.is_memory_available(8.0)}")
    print(f"  Can allocate 16GB: {monitor.is_memory_available(16.0)}")
    
    print()


def demo_dynamic_batch_sizing():
    """Demonstrate dynamic batch sizing"""
    print("📊 Dynamic Batch Sizing Demo")
    print("=" * 50)
    
    config = AttentionConfig(
        enable_dynamic_batch_sizing=True,
        max_batch_size=8,
        memory_threshold_gb=0.8
    )
    
    sizer = DynamicBatchSizer(config)
    
    # Test different resolutions
    resolutions = [(512, 512), (1024, 1024), (1536, 1536), (2048, 2048)]
    
    print("Optimal batch sizes for different resolutions:")
    for width, height in resolutions:
        batch_size = sizer.get_optimal_batch_size(width, height, 1)
        memory_estimate = sizer._estimate_memory_usage(width * height, batch_size)
        print(f"  {width}x{height}: batch_size={batch_size}, estimated_memory={memory_estimate:.2f}GB")
    
    print()


def demo_attention_benchmarking():
    """Demonstrate attention method benchmarking"""
    print("⚡ Attention Benchmarking Demo")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available - skipping benchmarks")
        return
    
    optimizer = create_attention_optimizer(
        architecture="MMDiT",
        optimization_level="balanced"
    )
    
    # Benchmark different tensor sizes
    test_configs = [
        (1, 512, 512),    # Small
        (1, 1024, 1024),  # Medium
        (1, 2048, 1024),  # Large
    ]
    
    for batch_size, seq_len, embed_dim in test_configs:
        print(f"\nBenchmarking {batch_size}x{seq_len}x{embed_dim}:")
        
        try:
            results = optimizer.benchmark_attention_methods(batch_size, seq_len, embed_dim)
            
            if results:
                fastest_method = min(results.items(), key=lambda x: x[1])
                print(f"  Fastest: {fastest_method[0]} ({fastest_method[1]:.2f}ms)")
                
                for method, time_ms in sorted(results.items(), key=lambda x: x[1]):
                    speedup = fastest_method[1] / time_ms if time_ms > 0 else 1.0
                    print(f"  {method}: {time_ms:.2f}ms (speedup: {speedup:.2f}x)")
            else:
                print("  No benchmark results available")
                
        except Exception as e:
            print(f"  Benchmark failed: {e}")
    
    print()


def demo_memory_efficient_attention():
    """Demonstrate memory-efficient attention function"""
    print("🧠 Memory-Efficient Attention Demo")
    print("=" * 50)
    
    optimizer = create_attention_optimizer(
        architecture="MMDiT",
        optimization_level="quality"
    )
    
    # Create memory-efficient attention function
    attention_fn = optimizer.create_memory_efficient_attention_function(64, 16)
    
    if torch.cuda.is_available():
        # Test with different tensor sizes
        test_sizes = [(1, 256, 1024), (1, 512, 1024), (1, 1024, 1024)]
        
        for batch_size, seq_len, embed_dim in test_sizes:
            print(f"\nTesting {batch_size}x{seq_len}x{embed_dim}:")
            
            try:
                # Create test tensors
                query = torch.randn(batch_size, seq_len, embed_dim, device="cuda", dtype=torch.float16)
                key = torch.randn(batch_size, seq_len, embed_dim, device="cuda", dtype=torch.float16)
                value = torch.randn(batch_size, seq_len, embed_dim, device="cuda", dtype=torch.float16)
                
                # Measure memory before
                memory_before = torch.cuda.memory_allocated() / 1e9
                
                # Time the attention computation
                torch.cuda.synchronize()
                start_time = time.time()
                
                with optimizer.optimized_inference_context():
                    result = attention_fn(query, key, value)
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                # Measure memory after
                memory_after = torch.cuda.memory_allocated() / 1e9
                memory_used = memory_after - memory_before
                
                print(f"  Time: {(end_time - start_time) * 1000:.2f}ms")
                print(f"  Memory used: {memory_used:.3f}GB")
                print(f"  Output shape: {result.shape}")
                
                # Clean up
                del query, key, value, result
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  Test failed: {e}")
    else:
        print("CUDA not available - skipping memory tests")
    
    print()


def demo_pipeline_integration():
    """Demonstrate integration with pipeline optimizer"""
    print("🔗 Pipeline Integration Demo")
    print("=" * 50)
    
    # Create pipeline optimizer with attention optimizations
    pipeline_config = OptimizationConfig(
        architecture_type="MMDiT",
        enable_attention_optimizations=True,
        attention_optimization_level="balanced",
        enable_dynamic_batch_sizing=True,
        enable_memory_efficient_attention=True,
        optimal_width=1024,
        optimal_height=1024,
        optimal_steps=25
    )
    
    pipeline_optimizer = PipelineOptimizer(pipeline_config)
    
    # Show configuration
    print("Pipeline Configuration:")
    print(f"  Architecture: {pipeline_config.architecture_type}")
    print(f"  Attention Optimizations: {pipeline_config.enable_attention_optimizations}")
    print(f"  Optimization Level: {pipeline_config.attention_optimization_level}")
    print(f"  Dynamic Batching: {pipeline_config.enable_dynamic_batch_sizing}")
    print(f"  Memory Efficient: {pipeline_config.enable_memory_efficient_attention}")
    
    # Generate optimal settings
    generation_settings = pipeline_optimizer.configure_generation_settings()
    print(f"\nGeneration Settings:")
    for key, value in generation_settings.items():
        print(f"  {key}: {value}")
    
    # Show attention optimizer status
    if pipeline_optimizer.attention_optimizer:
        print(f"\nAttention Optimizer Status: ✅ Active")
        print(f"  Architecture: {pipeline_optimizer.attention_optimizer.config.architecture_type}")
        print(f"  SDPA: {pipeline_optimizer.attention_optimizer.config.use_scaled_dot_product_attention}")
        print(f"  Flash Attention: {pipeline_optimizer.attention_optimizer.config.use_flash_attention_2}")
        print(f"  Dynamic Batching: {pipeline_optimizer.attention_optimizer.config.enable_dynamic_batch_sizing}")
    else:
        print(f"\nAttention Optimizer Status: ❌ Not Available")
    
    print()


def demo_optimization_context():
    """Demonstrate optimized inference context"""
    print("⚙️ Optimization Context Demo")
    print("=" * 50)
    
    optimizer = create_attention_optimizer()
    
    print("Testing optimization context manager...")
    
    # Show settings before context
    print(f"Before context:")
    print(f"  Gradients enabled: {torch.is_grad_enabled()}")
    print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    
    # Use optimization context
    with optimizer.optimized_inference_context():
        print(f"Inside context:")
        print(f"  Gradients enabled: {torch.is_grad_enabled()}")
        print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        
        # Simulate some computation
        if torch.cuda.is_available():
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.matmul(x, x.T)
            print(f"  Computation result shape: {y.shape}")
    
    # Show settings after context
    print(f"After context:")
    print(f"  Gradients enabled: {torch.is_grad_enabled()}")
    print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    
    print()


def main():
    """Run all attention optimization demos"""
    print("🚀 Modern Attention and Memory Optimization Demo")
    print("=" * 60)
    print()
    
    try:
        # System information
        print_system_info()
        
        # Configuration demos
        demo_attention_config_creation()
        
        # Memory monitoring
        demo_memory_monitoring()
        
        # Dynamic batch sizing
        demo_dynamic_batch_sizing()
        
        # Attention benchmarking (if CUDA available)
        demo_attention_benchmarking()
        
        # Memory-efficient attention
        demo_memory_efficient_attention()
        
        # Pipeline integration
        demo_pipeline_integration()
        
        # Optimization context
        demo_optimization_context()
        
        print("✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()