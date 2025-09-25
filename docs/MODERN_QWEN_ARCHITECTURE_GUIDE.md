# Modern Qwen Architecture Guide

## Overview

This guide covers the modern Qwen architecture optimizations implemented in this project, including MMDiT (Multimodal Diffusion Transformer) support, Qwen2-VL integration, and performance optimizations for modern GPUs.

## Architecture Comparison

### MMDiT vs UNet

| Feature                       | MMDiT (Qwen-Image)             | UNet (Traditional)               |
| ----------------------------- | ------------------------------ | -------------------------------- |
| **Architecture**              | Modern transformer-based       | Convolutional neural network     |
| **Text-to-Image Performance** | Optimized for T2I              | General purpose                  |
| **Model Size**                | 8GB (Qwen-Image)               | Varies (typically 4-7GB)         |
| **Generation Speed**          | 2-5s per step (optimized)      | 3-8s per step                    |
| **Text Rendering**            | Excellent multilingual support | Basic text support               |
| **Memory Efficiency**         | Good with optimizations        | Good with memory-saving features |
| **Compatibility**             | Qwen-specific optimizations    | Wide ecosystem support           |

### Model Variants

#### Qwen-Image (Recommended for Text-to-Image)

- **Size**: ~8GB
- **Architecture**: MMDiT
- **Use Case**: Fast text-to-image generation
- **Performance**: 2-5 seconds per step on RTX 4080
- **Features**: Excellent text rendering, multilingual support

#### Qwen-Image-Edit (Not Recommended for T2I)

- **Size**: ~54GB
- **Architecture**: MMDiT
- **Use Case**: Image editing and manipulation
- **Performance**: 30-180+ seconds per step for T2I
- **Issue**: Wrong model type for text-to-image tasks

#### Qwen2-VL-7B-Instruct (Multimodal Enhancement)

- **Size**: ~15GB
- **Architecture**: Transformer
- **Use Case**: Text understanding and image analysis
- **Integration**: Enhances prompt understanding for better generation

## Installation and Setup

### 1. Download Models

```bash
# Download optimal setup (recommended)
python tools/download_qwen_image.py --optimal

# Or download individually
python tools/download_qwen_image.py --model qwen-image
python tools/download_qwen_image.py --model qwen2-vl-7b
```

### 2. Verify Installation

```bash
# Check current setup
python tools/download_qwen_image.py --detect

# Run quick performance test
python examples/performance_comparison_demo.py
```

## Usage Examples

### Basic Optimized Generation

```python
from src.pipeline_optimizer import PipelineOptimizer, OptimizationConfig
from src.model_detection_service import ModelDetectionService

# Detect and load optimal model
detector = ModelDetectionService()
current_model = detector.detect_current_model()

# Create optimized configuration
config = OptimizationConfig(
    architecture_type="MMDiT",
    enable_attention_slicing=False,  # Disabled for performance
    enable_vae_slicing=False,        # Disabled for performance
    enable_tf32=True,                # Enabled for RTX GPUs
    enable_cudnn_benchmark=True      # Enabled for consistent inputs
)

# Create optimized pipeline
optimizer = PipelineOptimizer(config)
pipeline = optimizer.create_optimized_pipeline(
    current_model.path,
    "MMDiT"
)

# Generate image
image = pipeline(
    prompt="A beautiful landscape with mountains and lakes",
    width=1024,
    height=1024,
    num_inference_steps=20,
    true_cfg_scale=3.5  # MMDiT uses true_cfg_scale
).images[0]
```

### Multimodal Integration

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Load Qwen2-VL for prompt enhancement
qwen2vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Enhance prompt
enhanced_prompt = enhance_prompt_with_qwen2vl(
    "A cat sitting on a chair",
    context="Create a photorealistic image with professional lighting"
)

# Generate with enhanced prompt
image = pipeline(prompt=enhanced_prompt, ...).images[0]
```

## Performance Optimization

### GPU Optimizations

#### RTX 4080/4090 (16GB+ VRAM)

```python
config = OptimizationConfig(
    enable_attention_slicing=False,  # Disabled for performance
    enable_vae_slicing=False,        # Disabled for performance
    enable_tf32=True,                # Enabled for Tensor Cores
    enable_cudnn_benchmark=True,     # Enabled for consistent inputs
    optimal_steps=20,
    optimal_cfg_scale=3.5
)
```

#### RTX 3080/4070 (12GB VRAM)

```python
config = OptimizationConfig(
    enable_attention_slicing=False,  # Can be disabled
    enable_vae_slicing=False,        # Can be disabled
    enable_tf32=True,
    enable_cudnn_benchmark=True,
    optimal_steps=25,
    optimal_cfg_scale=3.5
)
```

#### RTX 3070/4060 (8GB VRAM)

```python
config = OptimizationConfig(
    enable_attention_slicing=True,   # Enabled for memory efficiency
    enable_vae_slicing=True,         # Enabled for memory efficiency
    enable_tf32=True,
    enable_cudnn_benchmark=True,
    optimal_steps=20,
    optimal_cfg_scale=3.5
)
```

### Performance Targets

| Hardware | Target Performance | Typical Results |
| -------- | ------------------ | --------------- |
| RTX 4090 | 1-3s per step      | 2-4s per step   |
| RTX 4080 | 2-5s per step      | 3-6s per step   |
| RTX 3080 | 3-7s per step      | 4-8s per step   |
| RTX 3070 | 5-10s per step     | 6-12s per step  |

## Architecture-Specific Considerations

### MMDiT (Qwen-Image) Specifics

#### Compatible Features

- ✅ TF32 optimization
- ✅ cuDNN benchmark
- ✅ Memory-efficient attention
- ✅ Custom attention processors
- ✅ `true_cfg_scale` parameter

#### Incompatible Features

- ❌ AttnProcessor2_0 (causes tensor unpacking errors)
- ❌ torch.compile (tensor format differences)
- ❌ Flash Attention 2 (architecture incompatibility)
- ❌ `guidance_scale` parameter (use `true_cfg_scale`)

#### Optimization Tips

1. **Disable memory-saving features** on high-VRAM GPUs for better performance
2. **Use appropriate CFG parameter**: `true_cfg_scale` instead of `guidance_scale`
3. **Optimal step count**: 15-25 steps for good quality/speed balance
4. **Resolution**: 1024x1024 is optimal, higher resolutions scale well

### UNet Architecture Specifics

#### Compatible Features

- ✅ AttnProcessor2_0 (Flash Attention)
- ✅ torch.compile optimization
- ✅ All memory-saving features
- ✅ `guidance_scale` parameter

#### Optimization Tips

1. **Enable Flash Attention** with AttnProcessor2_0
2. **Use torch.compile** for additional speed improvements
3. **Memory-saving features** can be enabled without major performance loss

## Troubleshooting

### Common Issues

#### "Tensor unpacking error" with MMDiT

```
RuntimeError: too many values to unpack (expected 2)
```

**Solution**: Disable AttnProcessor2_0 and use default attention processor.

#### Poor performance (>10s per step)

**Possible causes**:

1. Using Qwen-Image-Edit instead of Qwen-Image
2. Memory-saving features enabled on high-VRAM GPU
3. CPU inference instead of GPU
4. Suboptimal generation parameters

**Solution**: Run detection and optimization:

```bash
python tools/download_qwen_image.py --detect
python examples/performance_comparison_demo.py
```

#### Out of memory errors

**Solutions**:

1. Enable attention slicing: `enable_attention_slicing=True`
2. Enable VAE slicing: `enable_vae_slicing=True`
3. Reduce resolution: `width=768, height=768`
4. Enable CPU offloading: `enable_cpu_offload=True`

### Performance Debugging

#### Check Current Configuration

```python
from src.model_detection_service import ModelDetectionService

detector = ModelDetectionService()
current_model = detector.detect_current_model()
architecture = detector.detect_model_architecture(current_model)
perf_chars = detector.analyze_performance_characteristics(current_model)

print(f"Model: {current_model.name}")
print(f"Architecture: {architecture}")
print(f"Expected performance: {perf_chars['expected_generation_time']}")
print(f"Bottlenecks: {perf_chars['bottlenecks']}")
```

#### Validate Optimizations

```python
from src.pipeline_optimizer import PipelineOptimizer

optimizer = PipelineOptimizer(config)
validation = optimizer.validate_optimization(pipeline)
print(f"Optimization status: {validation}")
```

## Advanced Features

### Multimodal Integration Benefits

1. **Enhanced Prompt Understanding**: Qwen2-VL analyzes and improves prompts
2. **Context-Aware Generation**: Better understanding of complex instructions
3. **Style Analysis**: Can analyze reference images for style matching
4. **Multilingual Support**: Better handling of non-English prompts

### Custom Optimization Configurations

#### Ultra Performance (High-VRAM GPUs)

```python
ultra_config = OptimizationConfig(
    optimal_steps=15,
    optimal_cfg_scale=3.0,
    enable_attention_slicing=False,
    enable_vae_slicing=False,
    enable_tf32=True,
    enable_cudnn_benchmark=True
)
```

#### Memory Efficient (Low-VRAM GPUs)

```python
memory_config = OptimizationConfig(
    optimal_steps=20,
    optimal_cfg_scale=3.5,
    enable_attention_slicing=True,
    enable_vae_slicing=True,
    enable_cpu_offload=True,
    optimal_width=768,
    optimal_height=768
)
```

## Migration Guide

### From Old Setup to Modern Architecture

1. **Backup existing models** (optional)
2. **Download new models**:
   ```bash
   python tools/download_qwen_image.py --optimal
   ```
3. **Update code** to use new pipeline classes:

   ```python
   # Old
   from diffusers import DiffusionPipeline
   pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image-Edit")

   # New
   from src.pipeline_optimizer import PipelineOptimizer
   optimizer = PipelineOptimizer()
   pipe = optimizer.create_optimized_pipeline("Qwen/Qwen-Image", "MMDiT")
   ```

4. **Update generation parameters**:

   ```python
   # Old
   image = pipe(prompt="...", guidance_scale=7.5)

   # New
   image = pipe(prompt="...", true_cfg_scale=3.5)
   ```

### API Compatibility

The optimization system maintains backward compatibility with existing code through the compatibility layer. Existing scripts should continue to work with improved performance.

## Best Practices

### Model Selection

1. **Use Qwen-Image** for text-to-image generation
2. **Avoid Qwen-Image-Edit** for text-to-image tasks
3. **Add Qwen2-VL** for enhanced prompt understanding

### Performance Optimization

1. **Disable memory-saving features** on high-VRAM GPUs
2. **Use optimal step counts** (15-25 steps)
3. **Enable GPU optimizations** (TF32, cuDNN benchmark)
4. **Monitor GPU utilization** and memory usage

### Quality vs Speed

1. **Ultra Fast**: 10-15 steps, CFG 2.5-3.0
2. **Balanced**: 20-25 steps, CFG 3.5
3. **High Quality**: 30-40 steps, CFG 4.0-4.5

## Examples and Demos

### Available Demo Scripts

1. **`examples/optimized_text_to_image_demo.py`**: Comprehensive optimization showcase
2. **`examples/multimodal_integration_demo.py`**: Qwen2-VL integration examples
3. **`examples/performance_comparison_demo.py`**: Benchmarking different configurations
4. **`examples/official_qwen_example.py`**: Updated official examples with optimizations

### Running Demos

```bash
# Quick optimization demo
python examples/optimized_text_to_image_demo.py

# Multimodal features
python examples/multimodal_integration_demo.py

# Performance benchmarking
python examples/performance_comparison_demo.py

# Official examples with optimizations
python examples/official_qwen_example.py
```

## Support and Resources

### Documentation

- [Official Qwen-Image Documentation](https://huggingface.co/Qwen/Qwen-Image)
- [Qwen2-VL Documentation](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)

### Performance Monitoring

- Use `examples/performance_comparison_demo.py` for benchmarking
- Monitor GPU utilization with `nvidia-smi`
- Check memory usage with `torch.cuda.memory_summary()`

### Community Resources

- [Qwen GitHub Repository](https://github.com/QwenLM/Qwen)
- [Diffusers GitHub Repository](https://github.com/huggingface/diffusers)
- [PyTorch Optimization Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
