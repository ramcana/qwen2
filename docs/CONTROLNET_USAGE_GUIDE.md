# ControlNet Usage Guide

## Introduction

ControlNet is a powerful technique that allows you to guide AI image generation using structural information from reference images. Instead of relying solely on text prompts, ControlNet uses visual cues like edges, depth maps, poses, or segmentation masks to control the composition and structure of generated images.

## Understanding ControlNet

### What is ControlNet?

ControlNet adds spatial conditioning to diffusion models, allowing you to:

- Control the exact pose of characters
- Maintain specific compositions and layouts
- Preserve architectural structures
- Guide artistic style while maintaining structure
- Create consistent character positions across multiple images

### How It Works

1. **Control Image**: You provide a reference image
2. **Feature Extraction**: The system extracts structural features (edges, depth, pose, etc.)
3. **Conditioning**: These features guide the generation process
4. **Text Prompt**: Combined with your text description for the final result

## Available ControlNet Types

### 1. Canny Edge Detection

**Best For**: Preserving outlines and structural boundaries

**Use Cases**:

- Architectural drawings to photorealistic buildings
- Sketch to detailed artwork
- Maintaining object boundaries
- Line art to colored illustrations

**Example Workflow**:

```
Control Image: Simple line drawing of a house
Text Prompt: "Victorian house with detailed architecture, sunset lighting"
Result: Photorealistic Victorian house following the line drawing structure
```

**Tips**:

- Use high-contrast images with clear edges
- Simple line drawings work better than complex sketches
- Adjust conditioning scale (0.8-1.2) based on how strictly you want to follow edges

### 2. Depth Control

**Best For**: 3D-aware generation and spatial relationships

**Use Cases**:

- Converting 2D images to different styles while preserving depth
- Creating images with specific spatial layouts
- Maintaining foreground/background relationships
- Architectural visualization

**Example Workflow**:

```
Control Image: Photo of a room interior
Text Prompt: "Futuristic sci-fi interior, neon lighting, cyberpunk style"
Result: Sci-fi room with the same spatial layout and depth relationships
```

**Tips**:

- Works well with images that have clear depth variation
- Indoor scenes and landscapes are ideal
- Lower conditioning scale (0.6-0.9) for more creative freedom

### 3. Pose Detection

**Best For**: Human figure control and character consistency

**Use Cases**:

- Character design with specific poses
- Fashion photography with controlled poses
- Animation reference frames
- Consistent character positioning

**Example Workflow**:

```
Control Image: Photo of person in dancing pose
Text Prompt: "Elegant ballet dancer in flowing dress, stage lighting"
Result: Ballet dancer in the exact same pose as the reference
```

**Tips**:

- Works best with clear, unobstructed human figures
- Single person poses are more reliable than group poses
- High conditioning scale (1.0-1.5) for accurate pose matching

### 4. Normal Map Control

**Best For**: Surface detail and texture control

**Use Cases**:

- Detailed surface textures
- Fabric and material rendering
- Architectural surface details
- Product visualization

**Example Workflow**:

```
Control Image: Fabric texture with normal map
Text Prompt: "Luxury silk dress with intricate patterns"
Result: Dress with realistic fabric surface details
```

**Tips**:

- Requires good quality normal maps
- Best for close-up detail work
- Combine with other control types for full scene control

### 5. Segmentation Control

**Best For**: Region-specific control and composition

**Use Cases**:

- Landscape composition control
- Object placement and sizing
- Color region control
- Complex scene layouts

**Example Workflow**:

```
Control Image: Segmentation map with sky, mountains, water regions
Text Prompt: "Fantasy landscape with floating islands and magical lighting"
Result: Fantasy scene following the exact regional layout
```

**Tips**:

- Create clear, distinct regions
- Use different colors for different semantic areas
- Works well for landscape and architectural scenes

## Practical Examples

### Example 1: Architectural Visualization

**Objective**: Convert a simple floor plan to a photorealistic interior

**Steps**:

1. **Prepare Control Image**: Clean floor plan or architectural drawing
2. **Choose Control Type**: Canny edge detection
3. **Set Parameters**:
   - Conditioning Scale: 1.0
   - Guidance Start: 0.0
   - Guidance End: 1.0
4. **Text Prompt**: "Modern minimalist living room, natural lighting, hardwood floors, large windows"
5. **Generate**: Process with ControlNet

**Expected Result**: Photorealistic interior following the floor plan layout

### Example 2: Character Design

**Objective**: Create a fantasy character with a specific pose

**Steps**:

1. **Prepare Control Image**: Photo or drawing of desired pose
2. **Choose Control Type**: Pose detection
3. **Set Parameters**:
   - Conditioning Scale: 1.2
   - Guidance Start: 0.0
   - Guidance End: 0.8
4. **Text Prompt**: "Fantasy warrior in ornate armor, magical sword, epic lighting"
5. **Generate**: Process with ControlNet

**Expected Result**: Fantasy warrior in the exact pose from the reference

### Example 3: Style Transfer with Structure

**Objective**: Apply artistic style while maintaining composition

**Steps**:

1. **Prepare Control Image**: Photo with good composition
2. **Choose Control Type**: Depth or Canny (depending on content)
3. **Set Parameters**:
   - Conditioning Scale: 0.8
   - Guidance Start: 0.0
   - Guidance End: 1.0
4. **Text Prompt**: "Oil painting in impressionist style, vibrant colors, artistic brushstrokes"
5. **Generate**: Process with ControlNet

**Expected Result**: Impressionist painting with original composition

## Advanced Techniques

### Multi-ControlNet

Combine multiple control types for enhanced control:

1. **Primary Control**: Use the most important structural guidance (e.g., pose)
2. **Secondary Control**: Add additional guidance (e.g., depth)
3. **Balance Scales**: Adjust conditioning scales to balance influences
4. **Test Combinations**: Experiment with different control type combinations

### Control Strength Scheduling

Vary control strength during generation:

- **Strong Start**: High conditioning at the beginning for structure
- **Gradual Release**: Reduce conditioning for creative freedom
- **Fine-tune End**: Adjust final conditioning for detail refinement

### Preprocessing Tips

**For Canny Edge**:

- Use image editing software to clean up edges
- Adjust contrast for clearer edge detection
- Remove unnecessary details that might confuse the model

**For Depth Maps**:

- Ensure good depth variation in the source image
- Use depth estimation tools for better depth maps
- Consider manual depth map creation for precise control

**For Pose Detection**:

- Use clear, well-lit photos of poses
- Ensure the full figure is visible
- Avoid complex backgrounds that might interfere with detection

## Parameter Guidelines

### Conditioning Scale

- **0.5-0.7**: Loose guidance, more creative freedom
- **0.8-1.0**: Balanced control and creativity
- **1.1-1.5**: Strong guidance, strict adherence to control
- **1.6+**: Very strict control, may reduce quality

### Guidance Start/End

- **Start 0.0, End 1.0**: Control throughout entire generation
- **Start 0.0, End 0.8**: Release control in final steps for refinement
- **Start 0.2, End 1.0**: Allow initial creative freedom, then apply control
- **Start 0.0, End 0.5**: Early control only, creative finish

### Inference Steps

- **20-30 steps**: Fast generation, good for testing
- **40-50 steps**: Balanced quality and speed
- **60+ steps**: High quality, slower generation

## Troubleshooting

### Common Issues

**Control Not Applied**:

- Increase conditioning scale
- Check control image quality
- Verify control type matches image content
- Ensure guidance start/end values are appropriate

**Over-controlled Results**:

- Reduce conditioning scale
- Adjust guidance end to release control earlier
- Try different control types
- Simplify the control image

**Poor Quality Output**:

- Increase inference steps
- Improve control image quality
- Adjust CFG scale
- Refine text prompt

**Inconsistent Results**:

- Use fixed seeds for reproducibility
- Check control image preprocessing
- Verify parameter consistency
- Consider control image resolution

### Best Practices

**Control Image Preparation**:

- Use high-resolution source images
- Ensure good contrast and clarity
- Remove unnecessary noise or artifacts
- Match aspect ratio to desired output

**Prompt Engineering**:

- Be specific about desired style and content
- Use negative prompts to avoid unwanted elements
- Include lighting and mood descriptions
- Specify artistic style or technique

**Parameter Tuning**:

- Start with default values and adjust gradually
- Test different conditioning scales
- Experiment with guidance scheduling
- Save successful parameter combinations

## Integration with Other Features

### Combining with Inpainting

1. Generate base image with ControlNet
2. Use inpainting to refine specific areas
3. Maintain overall structure while improving details

### Style Transfer Enhancement

1. Use ControlNet for structural guidance
2. Apply style transfer for artistic effects
3. Combine multiple techniques for complex results

### Outpainting with Control

1. Generate core image with ControlNet
2. Use outpainting to extend the composition
3. Maintain structural consistency across extensions

## API Usage Examples

### Basic ControlNet Request

```python
import requests

# Prepare request
request_data = {
    "prompt": "Fantasy castle on a hilltop, dramatic lighting",
    "control_image_path": "path/to/control/image.jpg",
    "control_type": "canny",
    "controlnet_conditioning_scale": 1.0,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "width": 768,
    "height": 768
}

# Send request
response = requests.post("http://localhost:8000/controlnet/generate", json=request_data)
result = response.json()
```

### Auto-Detection Request

```python
# Let the system detect the best control type
request_data = {
    "prompt": "Modern architecture, glass and steel",
    "control_image_path": "path/to/reference.jpg",
    "control_type": "auto",  # Automatic detection
    "controlnet_conditioning_scale": 0.8
}

response = requests.post("http://localhost:8000/controlnet/generate", json=request_data)
```

### Control Type Detection

```python
# Detect what control type works best for an image
detection_request = {
    "image_path": "path/to/image.jpg"
}

response = requests.post("http://localhost:8000/controlnet/detect", json=detection_request)
detection_result = response.json()

print(f"Best control type: {detection_result['detected_type']}")
print(f"Confidence: {detection_result['confidence']}")
```

## Conclusion

ControlNet provides powerful tools for precise image generation control. By understanding the different control types and their applications, you can create more consistent and controlled AI-generated images. Experiment with different combinations and parameters to find what works best for your specific use cases.

Remember that ControlNet is most effective when combined with well-crafted text prompts and appropriate parameter settings. Start with simple examples and gradually explore more complex applications as you become familiar with the system.
