# DiffSynth Enhanced UI User Guide

## Overview

The DiffSynth Enhanced UI provides powerful image editing capabilities alongside the existing Qwen text-to-image generation. This comprehensive guide will help you navigate the enhanced interface and make the most of the advanced editing features.

## Getting Started

### Accessing the Enhanced UI

1. **Launch the Application**: Start the system using your preferred method (Docker, direct Python, or scripts)
2. **Open the Web Interface**: Navigate to `http://localhost:3000` in your browser
3. **Initialize Services**: The system will automatically initialize both Qwen and DiffSynth services

### Interface Overview

The enhanced UI features three main modes accessible via tabs:

- **Generate**: Traditional text-to-image generation using Qwen
- **Edit**: Advanced image editing using DiffSynth
- **ControlNet**: Structural guidance for precise image control

## Text-to-Image Generation (Generate Mode)

### Basic Generation

1. **Enter Your Prompt**: Describe the image you want to create
2. **Adjust Parameters**:

   - **Aspect Ratio**: Choose from preset ratios (16:9, 1:1, 4:3, etc.)
   - **Steps**: Higher values (50-100) for better quality, lower (20-30) for speed
   - **CFG Scale**: Controls prompt adherence (4.0-8.0 recommended)
   - **Seed**: Use -1 for random, or specific numbers for reproducible results

3. **Generate**: Click "Generate Image" and wait for processing

### Advanced Options

- **Negative Prompt**: Specify what you don't want in the image
- **Language**: Choose prompt language (English, Chinese, etc.)
- **Enhance Prompt**: Automatically improve your prompt description

## Image Editing (Edit Mode)

### Getting Started with Editing

1. **Switch to Edit Mode**: Click the "Edit" tab
2. **Upload Base Image**: Click "Upload Image" or drag and drop your image
3. **Choose Edit Operation**: Select from Inpaint, Outpaint, or Style Transfer

### Inpainting

**Purpose**: Fill in or modify specific areas of an image

**Steps**:

1. Upload your base image
2. Select "Inpaint" mode
3. Upload or create a mask image (white areas will be edited, black areas preserved)
4. Enter a prompt describing what should appear in the masked area
5. Adjust strength (0.1-1.0, higher values make bigger changes)
6. Click "Process"

**Tips**:

- Use high-contrast masks for best results
- Start with lower strength values and increase if needed
- Detailed prompts work better for inpainting

### Outpainting

**Purpose**: Extend images beyond their original boundaries

**Steps**:

1. Upload your base image
2. Select "Outpaint" mode
3. Choose extension direction (up, down, left, right, or multiple)
4. Set extension pixels (128-512 recommended)
5. Enter a prompt describing the extended area
6. Click "Process"

**Tips**:

- Consider the image's context when writing prompts
- Smaller extensions (128-256px) often look more natural
- Multiple directions can be processed simultaneously

### Style Transfer

**Purpose**: Apply the artistic style of one image to another

**Steps**:

1. Upload your content image (the image to be stylized)
2. Select "Style Transfer" mode
3. Upload your style reference image
4. Adjust style strength (0.1-1.0)
5. Optionally add a text prompt for additional guidance
6. Click "Process"

**Tips**:

- Strong artistic styles work best as references
- Lower strength values preserve more original content
- Experiment with different style-content combinations

## ControlNet Mode

### Understanding ControlNet

ControlNet allows you to guide image generation using structural information like edges, depth, or poses. This provides precise control over composition and layout.

### Available Control Types

- **Canny Edge**: Uses edge detection for structural guidance
- **Depth**: Uses depth information for 3D-aware generation
- **Pose**: Uses human pose detection for character control
- **Normal**: Uses surface normal maps for detailed control
- **Segmentation**: Uses semantic segmentation for region control

### Using ControlNet

1. **Switch to ControlNet Mode**: Click the "ControlNet" tab
2. **Upload Control Image**: This provides the structural guidance
3. **Choose Control Type**:
   - Select "Auto" for automatic detection
   - Choose specific type if you know what works best
4. **Preview Control Map**: Review the detected features
5. **Enter Generation Prompt**: Describe what you want to generate
6. **Adjust Parameters**:
   - **Conditioning Scale**: How strongly to follow the control (0.5-1.5)
   - **Guidance Start/End**: When to apply control during generation
7. **Generate**: Click "Generate with Control"

### ControlNet Tips

- **High-contrast control images** work better
- **Conditioning scale of 1.0** is usually a good starting point
- **Combine with detailed prompts** for best results
- **Preview the control map** to ensure proper detection

## Advanced Features

### Tiled Processing

For large images (>2048px), the system automatically uses tiled processing:

- **Automatic Detection**: System determines when tiling is beneficial
- **Memory Efficient**: Processes large images without memory overflow
- **Progress Tracking**: Shows progress for tiled operations
- **Quality Preservation**: Maintains quality across tile boundaries

### EliGen Integration

Enhanced generation quality through EliGen:

- **Quality Presets**: Choose from optimized quality settings
- **Automatic Enhancement**: Improves generation quality automatically
- **Resource Aware**: Adapts to available system resources

### Preset Management

Save and load your favorite configurations:

1. **Create Preset**: After configuring parameters, click "Save Preset"
2. **Name Your Preset**: Give it a descriptive name
3. **Choose Category**: Photo editing, artistic creation, or technical illustration
4. **Load Preset**: Select from saved presets to quickly apply settings

## Image Workspace Features

### Comparison View

- **Before/After**: Toggle between original and edited images
- **Side-by-Side**: View both images simultaneously
- **Overlay Mode**: Blend images to see differences

### Version History

- **Track Changes**: All edits are saved as versions
- **Revert Changes**: Go back to any previous version
- **Branch Edits**: Create multiple edit paths from the same base

### Export Options

- **High Quality**: Export at full resolution
- **Format Options**: PNG, JPEG, WebP
- **Metadata**: Include generation parameters
- **Batch Export**: Export multiple versions at once

## Performance Optimization

### Memory Management

- **Automatic Cleanup**: System manages GPU memory automatically
- **Service Switching**: Resources shared between Qwen and DiffSynth
- **Memory Monitoring**: Real-time memory usage display

### Processing Tips

- **Start Small**: Test with smaller images first
- **Batch Processing**: Process multiple images efficiently
- **Resource Monitoring**: Watch memory usage during processing
- **Queue Management**: Monitor processing queue status

## Troubleshooting

### Common Issues

**Image Upload Fails**:

- Check file format (PNG, JPEG, WebP supported)
- Ensure file size is reasonable (<50MB)
- Verify file is not corrupted

**Processing Takes Too Long**:

- Reduce image size for testing
- Lower inference steps for faster processing
- Check system resources and close other applications

**Out of Memory Errors**:

- Use tiled processing for large images
- Clear GPU memory using the memory management tools
- Reduce batch size or image dimensions

**Poor Quality Results**:

- Increase inference steps
- Adjust CFG scale
- Try different prompts or negative prompts
- Check control image quality for ControlNet

### Getting Help

- **Status Panel**: Check system status and initialization
- **Error Messages**: Read detailed error descriptions
- **Memory Monitor**: Check GPU memory usage
- **Queue Status**: Monitor processing queue

## Best Practices

### Prompt Writing

- **Be Specific**: Detailed descriptions work better
- **Use Style Keywords**: Include artistic styles, techniques
- **Negative Prompts**: Specify what to avoid
- **Language Consistency**: Use consistent language throughout

### Image Preparation

- **Good Quality Sources**: Start with high-quality images
- **Appropriate Resolution**: Match resolution to your needs
- **Clean Masks**: Use clean, high-contrast masks for inpainting
- **Relevant Control Images**: Ensure control images match your intent

### Workflow Optimization

- **Test Parameters**: Start with default settings and adjust
- **Save Presets**: Save successful configurations
- **Batch Similar Tasks**: Process similar images together
- **Monitor Resources**: Keep an eye on system performance

## Advanced Workflows

### Photo Enhancement

1. Start with Generate mode for base image
2. Switch to Edit mode for refinements
3. Use inpainting for specific area improvements
4. Apply style transfer for artistic effects

### Artistic Creation

1. Use ControlNet for structural guidance
2. Apply style transfer for artistic effects
3. Use outpainting to expand compositions
4. Combine multiple techniques for complex results

### Technical Illustration

1. Use ControlNet with edge detection
2. Apply precise prompts for technical accuracy
3. Use inpainting for detail corrections
4. Export at high resolution for professional use

This guide provides a comprehensive overview of the DiffSynth Enhanced UI capabilities. Experiment with different features and combinations to discover what works best for your creative projects.
