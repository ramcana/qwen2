"""
Modern Attention and Memory Optimizations for Qwen-Image Generation
Implements scaled dot-product attention, Flash Attention 2, memory-efficient patterns,
dynamic batch sizing, and torch.compile optimizations for MMDiT architecture
"""

import logging
import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass
from contextlib import contextmanager
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for PyTorch 2.0+ features
PYTORCH_2_AVAILABLE = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
FLASH_ATTENTION_AVAILABLE = False
XFORMERS_AVAILABLE = False

# Check for Flash Attention 2
try:
    import flash_attn
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
    logger.info("✅ Flash Attention 2 available")
except ImportError:
    logger.info("⚠️ Flash Attention 2 not available")

# Check for xformers
try:
    import xformers
    import xformers.ops
    XFORMERS_AVAILABLE = True
    logger.info("✅ xformers available")
except ImportError:
    logger.info("⚠️ xformers not available")


@dataclass
class AttentionConfig:
    """Configuration for attention optimizations"""
    
    # Attention backend selection
    use_scaled_dot_product_attention: bool = True
    use_flash_attention_2: bool = False  # Disabled by default for compatibility
    use_xformers: bool = False
    use_memory_efficient_attention: bool = True
    
    # Memory optimization settings
    enable_attention_slicing: bool = False
    attention_slice_size: Optional[int] = None
    enable_gradient_checkpointing: bool = False
    
    # Dynamic batch sizing
    enable_dynamic_batch_sizing: bool = True
    max_batch_size: int = 4
    min_batch_size: int = 1
    memory_threshold_gb: float = 0.8  # Use 80% of available VRAM
    
    # torch.compile settings
    enable_torch_compile: bool = False
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    compile_dynamic: bool = False
    
    # Architecture-specific settings
    architecture_type: str = "MMDiT"  # "MMDiT" or "UNet"
    head_dim: int = 64
    num_heads: int = 16
    
    # Performance tuning
    enable_math_sdp: bool = True
    enable_flash_sdp: bool = False  # Disabled for MMDiT compatibility
    enable_mem_efficient_sdp: bool = True
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        try:
            # Check PyTorch version for SDPA
            if self.use_scaled_dot_product_attention and not PYTORCH_2_AVAILABLE:
                logger.warning("Scaled dot-product attention requires PyTorch 2.0+")
                self.use_scaled_dot_product_attention = False
            
            # Check Flash Attention availability
            if self.use_flash_attention_2 and not FLASH_ATTENTION_AVAILABLE:
                logger.warning("Flash Attention 2 not available, falling back to SDPA")
                self.use_flash_attention_2 = False
                self.use_scaled_dot_product_attention = True
            
            # Check xformers availability
            if self.use_xformers and not XFORMERS_AVAILABLE:
                logger.warning("xformers not available, falling back to SDPA")
                self.use_xformers = False
                self.use_scaled_dot_product_attention = True
            
            # Validate batch size settings
            if self.max_batch_size < self.min_batch_size:
                raise ValueError("max_batch_size must be >= min_batch_size")
            
            # Architecture-specific validation
            if self.architecture_type == "MMDiT":
                if self.use_flash_attention_2:
                    logger.warning("Flash Attention 2 may have compatibility issues with MMDiT")
                if self.enable_attention_slicing:
                    logger.warning("Attention slicing may reduce MMDiT performance")
            
            return True
            
        except Exception as e:
            logger.error(f"Attention configuration validation failed: {e}")
            return False


class AttentionOptimizer:
    """Optimizes attention mechanisms for modern GPU performance"""
    
    def __init__(self, config: Optional[AttentionConfig] = None):
        self.config = config or AttentionConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Validate configuration
        if not self.config.validate():
            logger.warning("Using fallback attention configuration")
            self.config = AttentionConfig()
        
        # Initialize memory monitoring
        self.memory_monitor = MemoryMonitor() if self.device == "cuda" else None
        
        # Cache for compiled functions
        self._compiled_functions = {}
        
        logger.info(f"AttentionOptimizer initialized for {self.config.architecture_type} architecture")
        self._log_available_optimizations()
    
    def _log_available_optimizations(self):
        """Log available optimization features"""
        optimizations = []
        
        if self.config.use_scaled_dot_product_attention and PYTORCH_2_AVAILABLE:
            optimizations.append("SDPA")
        if self.config.use_flash_attention_2 and FLASH_ATTENTION_AVAILABLE:
            optimizations.append("Flash Attention 2")
        if self.config.use_xformers and XFORMERS_AVAILABLE:
            optimizations.append("xformers")
        if self.config.enable_dynamic_batch_sizing:
            optimizations.append("Dynamic Batching")
        if self.config.enable_torch_compile:
            optimizations.append("torch.compile")
        
        logger.info(f"Available optimizations: {', '.join(optimizations) if optimizations else 'None'}")
    
    def optimize_pipeline_attention(self, pipeline) -> bool:
        """
        Optimize attention mechanisms in a diffusion pipeline
        
        Args:
            pipeline: Diffusion pipeline to optimize
            
        Returns:
            True if optimization was successful
        """
        try:
            logger.info("🚀 Optimizing pipeline attention mechanisms...")
            
            # Configure PyTorch SDPA backends
            self._configure_sdpa_backends()
            
            # Optimize transformer/UNet attention
            if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                self._optimize_transformer_attention(pipeline.transformer)
            elif hasattr(pipeline, 'unet') and pipeline.unet is not None:
                self._optimize_unet_attention(pipeline.unet)
            
            # Apply memory optimizations
            self._apply_memory_optimizations(pipeline)
            
            # Apply torch.compile if enabled
            if self.config.enable_torch_compile:
                self._apply_torch_compile(pipeline)
            
            logger.info("✅ Pipeline attention optimization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline attention optimization failed: {e}")
            return False
    
    def _configure_sdpa_backends(self):
        """Configure scaled dot-product attention backends"""
        if not PYTORCH_2_AVAILABLE:
            return
        
        try:
            # Configure SDPA backends based on configuration
            torch.backends.cuda.enable_math_sdp(self.config.enable_math_sdp)
            torch.backends.cuda.enable_flash_sdp(self.config.enable_flash_sdp)
            torch.backends.cuda.enable_mem_efficient_sdp(self.config.enable_mem_efficient_sdp)
            
            logger.info(f"SDPA backends configured: math={self.config.enable_math_sdp}, "
                       f"flash={self.config.enable_flash_sdp}, mem_efficient={self.config.enable_mem_efficient_sdp}")
            
        except Exception as e:
            logger.warning(f"Failed to configure SDPA backends: {e}")
    
    def _optimize_transformer_attention(self, transformer):
        """Optimize attention in MMDiT transformer"""
        try:
            logger.info("🔧 Optimizing MMDiT transformer attention...")
            
            # MMDiT transformers (like Qwen-Image) are incompatible with custom attention processors
            # They use their own optimized attention mechanism internally
            logger.info("⚠️ MMDiT transformer uses built-in optimized attention")
            logger.info("   Custom attention processors (AttnProcessor2_0, Flash Attention) are not compatible")
            logger.info("   Using default transformer attention mechanism")
            
            # Do NOT apply custom attention processors to MMDiT transformers
            # This prevents the "not enough values to unpack" error
            
            # Apply gradient checkpointing if enabled
            if self.config.enable_gradient_checkpointing:
                if hasattr(transformer, 'enable_gradient_checkpointing'):
                    transformer.enable_gradient_checkpointing()
                    logger.info("✅ Gradient checkpointing enabled for transformer")
            
        except Exception as e:
            logger.warning(f"Transformer attention optimization failed: {e}")
    
    def _optimize_unet_attention(self, unet):
        """Optimize attention in UNet architecture"""
        try:
            logger.info("🔧 Optimizing UNet attention...")
            
            # Apply attention processor based on configuration
            if self.config.use_flash_attention_2 and FLASH_ATTENTION_AVAILABLE:
                self._apply_flash_attention_processor(unet)
            elif self.config.use_scaled_dot_product_attention and PYTORCH_2_AVAILABLE:
                self._apply_sdpa_processor(unet)
            elif self.config.use_xformers and XFORMERS_AVAILABLE:
                self._apply_xformers_processor(unet)
            
            # Apply attention slicing if enabled (for memory efficiency)
            if self.config.enable_attention_slicing:
                if hasattr(unet, 'set_attention_slice'):
                    unet.set_attention_slice(self.config.attention_slice_size)
                    logger.info("✅ Attention slicing enabled for UNet")
            
            # Apply gradient checkpointing if enabled
            if self.config.enable_gradient_checkpointing:
                if hasattr(unet, 'enable_gradient_checkpointing'):
                    unet.enable_gradient_checkpointing()
                    logger.info("✅ Gradient checkpointing enabled for UNet")
            
        except Exception as e:
            logger.warning(f"UNet attention optimization failed: {e}")
    
    def _apply_flash_attention_processor(self, model):
        """Apply Flash Attention 2 processor"""
        try:
            # Check if this is an MMDiT transformer (incompatible with Flash Attention)
            model_class_name = model.__class__.__name__
            if "Transformer" in model_class_name and "Qwen" in model_class_name:
                logger.warning("⚠️ Skipping Flash Attention for MMDiT transformer (incompatible)")
                logger.info("   MMDiT transformers use built-in optimized attention")
                return
            
            from diffusers.models.attention_processor import FlashAttnProcessor2_0
            
            if hasattr(model, 'set_attn_processor'):
                model.set_attn_processor(FlashAttnProcessor2_0())
                logger.info("✅ Flash Attention 2 processor applied")
            else:
                logger.warning("Model does not support attention processor setting")
                
        except ImportError:
            logger.warning("FlashAttnProcessor2_0 not available")
        except Exception as e:
            logger.warning(f"Failed to apply Flash Attention processor: {e}")
    
    def _apply_sdpa_processor(self, model):
        """Apply scaled dot-product attention processor"""
        try:
            # Check if this is an MMDiT transformer (incompatible with AttnProcessor2_0)
            model_class_name = model.__class__.__name__
            if "Transformer" in model_class_name and "Qwen" in model_class_name:
                logger.warning("⚠️ Skipping AttnProcessor2_0 for MMDiT transformer (incompatible)")
                logger.info("   MMDiT transformers use built-in optimized attention")
                return
            
            from diffusers.models.attention_processor import AttnProcessor2_0
            
            if hasattr(model, 'set_attn_processor'):
                model.set_attn_processor(AttnProcessor2_0())
                logger.info("✅ SDPA processor applied")
            else:
                logger.warning("Model does not support attention processor setting")
                
        except ImportError:
            logger.warning("AttnProcessor2_0 not available")
        except Exception as e:
            logger.warning(f"Failed to apply SDPA processor: {e}")
    
    def _apply_xformers_processor(self, model):
        """Apply xformers attention processor"""
        try:
            from diffusers.models.attention_processor import XFormersAttnProcessor
            
            if hasattr(model, 'set_attn_processor'):
                model.set_attn_processor(XFormersAttnProcessor())
                logger.info("✅ xformers processor applied")
            else:
                logger.warning("Model does not support attention processor setting")
                
        except ImportError:
            logger.warning("XFormersAttnProcessor not available")
        except Exception as e:
            logger.warning(f"Failed to apply xformers processor: {e}")
    
    def _apply_memory_optimizations(self, pipeline):
        """Apply memory optimization settings"""
        try:
            logger.info("🔧 Applying memory optimizations...")
            
            # Disable memory-saving features for performance (high-VRAM GPUs)
            if not self.config.enable_attention_slicing:
                if hasattr(pipeline, 'disable_attention_slicing'):
                    pipeline.disable_attention_slicing()
                    logger.info("✅ Attention slicing disabled for performance")
            
            # Configure VAE optimizations
            if hasattr(pipeline, 'vae'):
                if hasattr(pipeline, 'disable_vae_slicing'):
                    pipeline.disable_vae_slicing()
                    logger.info("✅ VAE slicing disabled for performance")
                
                if hasattr(pipeline, 'disable_vae_tiling'):
                    pipeline.disable_vae_tiling()
                    logger.info("✅ VAE tiling disabled for performance")
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
    
    def _apply_torch_compile(self, pipeline):
        """Apply torch.compile optimizations"""
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile not available (requires PyTorch 2.0+)")
            return
        
        try:
            logger.info("🔧 Applying torch.compile optimizations...")
            
            # Compile transformer/UNet (most compute-intensive component)
            if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                if self.config.architecture_type == "MMDiT":
                    logger.warning("torch.compile disabled for MMDiT due to tensor unpacking issues")
                    return
                
                compiled_transformer = torch.compile(
                    pipeline.transformer,
                    mode=self.config.compile_mode,
                    dynamic=self.config.compile_dynamic
                )
                pipeline.transformer = compiled_transformer
                logger.info("✅ Transformer compiled with torch.compile")
                
            elif hasattr(pipeline, 'unet') and pipeline.unet is not None:
                compiled_unet = torch.compile(
                    pipeline.unet,
                    mode=self.config.compile_mode,
                    dynamic=self.config.compile_dynamic
                )
                pipeline.unet = compiled_unet
                logger.info("✅ UNet compiled with torch.compile")
            
        except Exception as e:
            logger.warning(f"torch.compile optimization failed: {e}")
    
    def calculate_optimal_batch_size(self, 
                                   width: int, 
                                   height: int, 
                                   num_inference_steps: int) -> int:
        """
        Calculate optimal batch size based on available VRAM and generation parameters
        
        Args:
            width: Image width
            height: Image height
            num_inference_steps: Number of inference steps
            
        Returns:
            Optimal batch size
        """
        if not self.config.enable_dynamic_batch_sizing or self.device != "cuda":
            return self.config.min_batch_size
        
        try:
            # Get available GPU memory
            if self.memory_monitor:
                available_memory_gb = self.memory_monitor.get_available_memory_gb()
            else:
                available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # Estimate memory usage per image
            # This is a rough estimation based on typical MMDiT memory usage
            pixels = width * height
            memory_per_image_gb = (
                pixels * 4 * 2 / 1e9 +  # Feature maps (float16)
                pixels * 8 / 1e9 +       # Attention maps
                0.5                       # Base overhead
            )
            
            # Calculate maximum batch size within memory threshold
            usable_memory_gb = available_memory_gb * self.config.memory_threshold_gb
            max_batch_size = max(1, int(usable_memory_gb / memory_per_image_gb))
            
            # Clamp to configured limits
            optimal_batch_size = min(max_batch_size, self.config.max_batch_size)
            optimal_batch_size = max(optimal_batch_size, self.config.min_batch_size)
            
            logger.info(f"Calculated optimal batch size: {optimal_batch_size} "
                       f"(available memory: {available_memory_gb:.1f}GB, "
                       f"estimated per image: {memory_per_image_gb:.2f}GB)")
            
            return optimal_batch_size
            
        except Exception as e:
            logger.warning(f"Failed to calculate optimal batch size: {e}")
            return self.config.min_batch_size
    
    def create_memory_efficient_attention_function(self, 
                                                  head_dim: int, 
                                                  num_heads: int) -> callable:
        """
        Create a memory-efficient attention function for high-resolution generation
        
        Args:
            head_dim: Dimension of each attention head
            num_heads: Number of attention heads
            
        Returns:
            Optimized attention function
        """
        def memory_efficient_attention(query, key, value, attn_mask=None, dropout_p=0.0):
            """Memory-efficient attention implementation"""
            
            batch_size, seq_len, embed_dim = query.shape
            
            # Use scaled dot-product attention if available
            if PYTORCH_2_AVAILABLE and self.config.use_scaled_dot_product_attention:
                # Reshape for multi-head attention
                q = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                k = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                v = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                
                # Apply SDPA
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=self.config.enable_flash_sdp,
                    enable_math=self.config.enable_math_sdp,
                    enable_mem_efficient=self.config.enable_mem_efficient_sdp
                ):
                    attn_output = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
                    )
                
                # Reshape back
                attn_output = attn_output.transpose(1, 2).contiguous().view(
                    batch_size, seq_len, embed_dim
                )
                
                return attn_output
            
            # Fallback to manual implementation with memory optimization
            else:
                # Chunked attention for memory efficiency
                chunk_size = min(seq_len, 1024)  # Process in chunks
                attn_outputs = []
                
                for i in range(0, seq_len, chunk_size):
                    end_idx = min(i + chunk_size, seq_len)
                    
                    q_chunk = query[:, i:end_idx]
                    k_chunk = key[:, i:end_idx]
                    v_chunk = value[:, i:end_idx]
                    
                    # Compute attention for chunk
                    attn_weights = torch.matmul(q_chunk, k_chunk.transpose(-2, -1))
                    attn_weights = attn_weights / math.sqrt(head_dim)
                    
                    if attn_mask is not None:
                        attn_weights += attn_mask[:, i:end_idx, i:end_idx]
                    
                    attn_weights = F.softmax(attn_weights, dim=-1)
                    
                    if dropout_p > 0.0:
                        attn_weights = F.dropout(attn_weights, p=dropout_p)
                    
                    attn_output_chunk = torch.matmul(attn_weights, v_chunk)
                    attn_outputs.append(attn_output_chunk)
                
                return torch.cat(attn_outputs, dim=1)
        
        return memory_efficient_attention
    
    def benchmark_attention_methods(self, 
                                  batch_size: int = 1, 
                                  seq_len: int = 1024, 
                                  embed_dim: int = 1024) -> Dict[str, float]:
        """
        Benchmark different attention methods to find the fastest
        
        Args:
            batch_size: Batch size for benchmarking
            seq_len: Sequence length
            embed_dim: Embedding dimension
            
        Returns:
            Dictionary of method names and their execution times
        """
        if self.device != "cuda":
            logger.warning("Attention benchmarking requires CUDA")
            return {}
        
        results = {}
        
        # Create test tensors
        query = torch.randn(batch_size, seq_len, embed_dim, device=self.device, dtype=torch.float16)
        key = torch.randn(batch_size, seq_len, embed_dim, device=self.device, dtype=torch.float16)
        value = torch.randn(batch_size, seq_len, embed_dim, device=self.device, dtype=torch.float16)
        
        # Warm up GPU
        for _ in range(5):
            _ = torch.matmul(query, key.transpose(-2, -1))
        
        torch.cuda.synchronize()
        
        # Benchmark SDPA
        if PYTORCH_2_AVAILABLE:
            try:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                for _ in range(10):
                    _ = F.scaled_dot_product_attention(query, key, value)
                end_time.record()
                
                torch.cuda.synchronize()
                results["SDPA"] = start_time.elapsed_time(end_time) / 10
                
            except Exception as e:
                logger.warning(f"SDPA benchmark failed: {e}")
        
        # Benchmark Flash Attention
        if FLASH_ATTENTION_AVAILABLE:
            try:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                for _ in range(10):
                    _ = flash_attn_func(query, key, value)
                end_time.record()
                
                torch.cuda.synchronize()
                results["Flash Attention"] = start_time.elapsed_time(end_time) / 10
                
            except Exception as e:
                logger.warning(f"Flash Attention benchmark failed: {e}")
        
        # Benchmark manual attention
        try:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            for _ in range(10):
                attn_weights = torch.matmul(query, key.transpose(-2, -1))
                attn_weights = F.softmax(attn_weights / math.sqrt(embed_dim), dim=-1)
                _ = torch.matmul(attn_weights, value)
            end_time.record()
            
            torch.cuda.synchronize()
            results["Manual"] = start_time.elapsed_time(end_time) / 10
            
        except Exception as e:
            logger.warning(f"Manual attention benchmark failed: {e}")
        
        # Log results
        if results:
            logger.info("Attention benchmark results (ms):")
            for method, time_ms in sorted(results.items(), key=lambda x: x[1]):
                logger.info(f"  {method}: {time_ms:.2f}ms")
        
        return results
    
    @contextmanager
    def optimized_inference_context(self):
        """Context manager for optimized inference"""
        # Store original settings
        original_grad_enabled = torch.is_grad_enabled()
        original_cudnn_benchmark = torch.backends.cudnn.benchmark
        original_cudnn_deterministic = torch.backends.cudnn.deterministic
        
        try:
            # Apply optimizations
            torch.set_grad_enabled(False)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Clear cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            yield
            
        finally:
            # Restore original settings
            torch.set_grad_enabled(original_grad_enabled)
            torch.backends.cudnn.benchmark = original_cudnn_benchmark
            torch.backends.cudnn.deterministic = original_cudnn_deterministic


class MemoryMonitor:
    """Monitor GPU memory usage for dynamic optimization"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information"""
        if self.device != "cuda":
            return {"total": 0, "allocated": 0, "cached": 0, "free": 0}
        
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            cached_memory = torch.cuda.memory_reserved()
            free_memory = total_memory - cached_memory
            
            return {
                "total": total_memory / 1e9,
                "allocated": allocated_memory / 1e9,
                "cached": cached_memory / 1e9,
                "free": free_memory / 1e9
            }
            
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            return {"total": 0, "allocated": 0, "cached": 0, "free": 0}
    
    def get_available_memory_gb(self) -> float:
        """Get available GPU memory in GB"""
        memory_info = self.get_memory_info()
        return memory_info["free"]
    
    def is_memory_available(self, required_gb: float) -> bool:
        """Check if required memory is available"""
        available_gb = self.get_available_memory_gb()
        return available_gb >= required_gb
    
    def clear_cache(self):
        """Clear GPU memory cache"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class DynamicBatchSizer:
    """Dynamically adjust batch size based on available memory"""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.batch_size_history = []
        
    def get_optimal_batch_size(self, 
                              width: int, 
                              height: int, 
                              current_batch_size: int = 1) -> int:
        """
        Get optimal batch size for current generation parameters
        
        Args:
            width: Image width
            height: Image height
            current_batch_size: Current batch size
            
        Returns:
            Optimal batch size
        """
        if not self.config.enable_dynamic_batch_sizing:
            return current_batch_size
        
        try:
            # Estimate memory requirement
            pixels = width * height
            estimated_memory_gb = self._estimate_memory_usage(pixels, current_batch_size)
            
            # Check if current batch size fits
            available_memory_gb = self.memory_monitor.get_available_memory_gb()
            
            if estimated_memory_gb <= available_memory_gb * self.config.memory_threshold_gb:
                # Try to increase batch size
                max_possible = int(
                    (available_memory_gb * self.config.memory_threshold_gb) / 
                    self._estimate_memory_usage(pixels, 1)
                )
                optimal_batch_size = min(max_possible, self.config.max_batch_size)
            else:
                # Decrease batch size
                optimal_batch_size = max(
                    int(current_batch_size * 0.8), 
                    self.config.min_batch_size
                )
            
            # Clamp to configured limits
            optimal_batch_size = max(optimal_batch_size, self.config.min_batch_size)
            optimal_batch_size = min(optimal_batch_size, self.config.max_batch_size)
            
            # Store in history
            self.batch_size_history.append(optimal_batch_size)
            if len(self.batch_size_history) > 10:
                self.batch_size_history.pop(0)
            
            return optimal_batch_size
            
        except Exception as e:
            logger.warning(f"Failed to calculate optimal batch size: {e}")
            return current_batch_size
    
    def _estimate_memory_usage(self, pixels: int, batch_size: int) -> float:
        """Estimate memory usage for given parameters"""
        # Rough estimation based on typical diffusion model memory usage
        base_memory_gb = 2.0  # Base model memory
        feature_memory_gb = pixels * batch_size * 4 * 2 / 1e9  # Feature maps (float16)
        attention_memory_gb = pixels * batch_size * 8 / 1e9     # Attention maps
        
        return base_memory_gb + feature_memory_gb + attention_memory_gb


# Factory functions for easy configuration
def create_attention_config(
    architecture: str = "MMDiT",
    optimization_level: str = "balanced",
    **kwargs
) -> AttentionConfig:
    """Create attention configuration for specific architecture and optimization level"""
    
    config = AttentionConfig(**kwargs)
    config.architecture_type = architecture
    
    if optimization_level == "ultra_fast":
        config.use_scaled_dot_product_attention = True
        config.use_flash_attention_2 = False  # Compatibility
        config.enable_dynamic_batch_sizing = True
        config.enable_torch_compile = False   # Stability
        
    elif optimization_level == "balanced":
        config.use_scaled_dot_product_attention = True
        config.use_memory_efficient_attention = True
        config.enable_dynamic_batch_sizing = True
        config.enable_torch_compile = False
        
    elif optimization_level == "quality":
        config.use_scaled_dot_product_attention = True
        config.use_memory_efficient_attention = True
        config.enable_gradient_checkpointing = True
        config.enable_torch_compile = False
        
    elif optimization_level == "experimental":
        config.use_flash_attention_2 = FLASH_ATTENTION_AVAILABLE
        config.enable_torch_compile = True
        config.compile_mode = "max-autotune"
        config.enable_dynamic_batch_sizing = True
    
    # Architecture-specific adjustments
    if architecture == "MMDiT":
        config.enable_flash_sdp = False  # Compatibility issues
        config.enable_attention_slicing = False  # Performance killer
        
    elif architecture == "UNet":
        config.use_xformers = XFORMERS_AVAILABLE
        config.enable_attention_slicing = True  # Memory efficiency
    
    return config


def create_attention_optimizer(
    architecture: str = "MMDiT",
    optimization_level: str = "balanced",
    **kwargs
) -> AttentionOptimizer:
    """Create attention optimizer with specified configuration"""
    
    config = create_attention_config(architecture, optimization_level, **kwargs)
    return AttentionOptimizer(config)