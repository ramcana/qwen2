"""
DiffSynth Utilities
Image preprocessing and postprocessing utilities for DiffSynth operations
"""

import logging
import os
import time
from typing import Optional, Tuple, Union, Dict, Any, List
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Image preprocessing utilities for DiffSynth operations"""
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        self.max_dimension = 2048
        self.min_dimension = 256
    
    def load_and_validate_image(self, image_input: Union[str, Image.Image]) -> Optional[Image.Image]:
        """
        Load and validate input image
        
        Args:
            image_input: Image path or PIL Image object
            
        Returns:
            Validated PIL Image or None if invalid
        """
        try:
            # Load image
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    logger.error(f"Image file not found: {image_input}")
                    return None
                
                # Check file extension
                ext = Path(image_input).suffix.lower()
                if ext not in self.supported_formats:
                    logger.error(f"Unsupported image format: {ext}")
                    return None
                
                image = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                logger.error(f"Unsupported image input type: {type(image_input)}")
                return None
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Validate dimensions
            width, height = image.size
            if width < self.min_dimension or height < self.min_dimension:
                logger.error(f"Image too small: {width}x{height}, minimum: {self.min_dimension}x{self.min_dimension}")
                return None
            
            if width > self.max_dimension or height > self.max_dimension:
                logger.warning(f"Image large: {width}x{height}, may require tiled processing")
            
            logger.debug(f"✅ Image loaded and validated: {width}x{height}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to load/validate image: {e}")
            return None
    
    def prepare_image_for_editing(
        self,
        image: Image.Image,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        maintain_aspect_ratio: bool = True
    ) -> Image.Image:
        """
        Prepare image for editing operations
        
        Args:
            image: Input PIL Image
            target_width: Target width (optional)
            target_height: Target height (optional)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            processed_image = image.copy()
            
            # Resize if target dimensions specified
            if target_width or target_height:
                processed_image = self._resize_image(
                    processed_image, target_width, target_height, maintain_aspect_ratio
                )
            
            # Ensure dimensions are multiples of 8 (required for many models)
            processed_image = self._ensure_multiple_of_8(processed_image)
            
            # Normalize image quality
            processed_image = self._normalize_image_quality(processed_image)
            
            logger.debug(f"✅ Image prepared for editing: {processed_image.size}")
            return processed_image
            
        except Exception as e:
            logger.error(f"Failed to prepare image for editing: {e}")
            return image
    
    def prepare_mask(self, mask_input: Union[str, Image.Image], blur_radius: int = 4) -> Optional[Image.Image]:
        """
        Prepare mask for inpainting operations
        
        Args:
            mask_input: Mask image path or PIL Image
            blur_radius: Blur radius for mask edges
            
        Returns:
            Processed mask image or None if invalid
        """
        try:
            # Load mask
            mask = self.load_and_validate_image(mask_input)
            if mask is None:
                return None
            
            # Convert to grayscale
            if mask.mode != 'L':
                mask = mask.convert('L')
            
            # Apply blur to soften edges
            if blur_radius > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Ensure binary mask (0 or 255)
            mask_array = np.array(mask)
            mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
            mask = Image.fromarray(mask_array, mode='L')
            
            logger.debug(f"✅ Mask prepared: {mask.size}")
            return mask
            
        except Exception as e:
            logger.error(f"Failed to prepare mask: {e}")
            return None
    
    def _resize_image(
        self,
        image: Image.Image,
        target_width: Optional[int],
        target_height: Optional[int],
        maintain_aspect_ratio: bool
    ) -> Image.Image:
        """Resize image to target dimensions"""
        current_width, current_height = image.size
        
        if not target_width and not target_height:
            return image
        
        if maintain_aspect_ratio:
            # Calculate new dimensions maintaining aspect ratio
            aspect_ratio = current_width / current_height
            
            if target_width and target_height:
                # Use the dimension that results in smaller scaling
                scale_w = target_width / current_width
                scale_h = target_height / current_height
                scale = min(scale_w, scale_h)
                new_width = int(current_width * scale)
                new_height = int(current_height * scale)
            elif target_width:
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:  # target_height
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
        else:
            new_width = target_width or current_width
            new_height = target_height or current_height
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _ensure_multiple_of_8(self, image: Image.Image) -> Image.Image:
        """Ensure image dimensions are multiples of 8"""
        width, height = image.size
        new_width = (width // 8) * 8
        new_height = (height // 8) * 8
        
        if new_width != width or new_height != height:
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.debug(f"Adjusted dimensions to multiples of 8: {new_width}x{new_height}")
        
        return image
    
    def _normalize_image_quality(self, image: Image.Image) -> Image.Image:
        """Normalize image quality and properties"""
        # Enhance image slightly for better processing
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)  # Slight sharpening
        
        return image


class ImagePostprocessor:
    """Image postprocessing utilities for DiffSynth operations"""
    
    def __init__(self):
        self.default_quality = 95
        self.default_format = 'JPEG'
    
    def postprocess_edited_image(
        self,
        image: Image.Image,
        original_image: Optional[Image.Image] = None,
        enhance_quality: bool = True,
        target_format: str = 'JPEG'
    ) -> Image.Image:
        """
        Postprocess edited image for optimal quality
        
        Args:
            image: Edited image
            original_image: Original image for reference (optional)
            enhance_quality: Whether to apply quality enhancements
            target_format: Target image format
            
        Returns:
            Postprocessed image
        """
        try:
            processed_image = image.copy()
            
            # Apply quality enhancements
            if enhance_quality:
                processed_image = self._enhance_image_quality(processed_image)
            
            # Color correction if original image provided
            if original_image:
                processed_image = self._apply_color_correction(processed_image, original_image)
            
            # Final cleanup
            processed_image = self._final_cleanup(processed_image)
            
            logger.debug(f"✅ Image postprocessed: {processed_image.size}")
            return processed_image
            
        except Exception as e:
            logger.error(f"Failed to postprocess image: {e}")
            return image
    
    def save_image(
        self,
        image: Image.Image,
        output_path: str,
        quality: int = None,
        format: str = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save image with optimal settings
        
        Args:
            image: PIL Image to save
            output_path: Output file path
            quality: JPEG quality (1-100)
            format: Image format
            metadata: Additional metadata to embed
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine format from extension if not specified
            if format is None:
                ext = Path(output_path).suffix.lower()
                format_map = {
                    '.jpg': 'JPEG',
                    '.jpeg': 'JPEG',
                    '.png': 'PNG',
                    '.webp': 'WEBP'
                }
                format = format_map.get(ext, self.default_format)
            
            # Set quality
            if quality is None:
                quality = self.default_quality
            
            # Prepare save parameters
            save_kwargs = {}
            if format == 'JPEG':
                save_kwargs.update({
                    'quality': quality,
                    'optimize': True,
                    'progressive': True
                })
            elif format == 'PNG':
                save_kwargs.update({
                    'optimize': True,
                    'compress_level': 6
                })
            elif format == 'WEBP':
                save_kwargs.update({
                    'quality': quality,
                    'method': 6
                })
            
            # Add metadata if provided
            if metadata:
                # Convert metadata to appropriate format
                if format == 'PNG':
                    from PIL.PngImagePlugin import PngInfo
                    pnginfo = PngInfo()
                    for key, value in metadata.items():
                        pnginfo.add_text(key, str(value))
                    save_kwargs['pnginfo'] = pnginfo
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save image
            image.save(output_path, format=format, **save_kwargs)
            
            logger.debug(f"✅ Image saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False
    
    def _enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """Apply quality enhancements to image"""
        # Slight contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)
        
        # Slight color enhancement
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.02)
        
        return image
    
    def _apply_color_correction(self, edited_image: Image.Image, original_image: Image.Image) -> Image.Image:
        """Apply color correction based on original image"""
        try:
            # Simple color balance correction
            # This is a basic implementation - could be enhanced with more sophisticated methods
            
            # Calculate average colors
            edited_avg = np.array(edited_image).mean(axis=(0, 1))
            original_avg = np.array(original_image.resize(edited_image.size)).mean(axis=(0, 1))
            
            # Apply subtle correction
            correction_factor = 0.1  # Subtle correction
            corrected_avg = edited_avg * (1 - correction_factor) + original_avg * correction_factor
            
            # Apply correction (simplified)
            edited_array = np.array(edited_image).astype(np.float32)
            correction = corrected_avg - edited_avg
            edited_array += correction * 0.1  # Very subtle
            
            edited_array = np.clip(edited_array, 0, 255).astype(np.uint8)
            return Image.fromarray(edited_array)
            
        except Exception as e:
            logger.warning(f"Color correction failed: {e}")
            return edited_image
    
    def _final_cleanup(self, image: Image.Image) -> Image.Image:
        """Final image cleanup"""
        # Remove any potential artifacts
        # Apply very light noise reduction
        image = image.filter(ImageFilter.MedianFilter(size=3))
        
        return image


class TiledProcessor:
    """Enhanced utilities for tiled processing of large images with automatic detection and progress tracking"""
    
    def __init__(self, tile_size: int = 512, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_tile_size = 256
        self.max_tile_size = 1024
        self.progress_callback = None
        
        # Memory thresholds for automatic tiling detection
        self.memory_thresholds = {
            'low': 2.0,    # GB - Always use tiling above this
            'medium': 4.0, # GB - Use tiling for complex operations
            'high': 8.0    # GB - Use tiling only for very large images
        }
    
    def should_use_tiled_processing(
        self,
        image: Image.Image,
        operation_type: str = "edit",
        available_memory_gb: Optional[float] = None
    ) -> bool:
        """
        Enhanced automatic tiling detection based on image size and available memory
        
        Args:
            image: Input image
            operation_type: Type of operation (edit, inpaint, outpaint, style_transfer)
            available_memory_gb: Available GPU memory in GB (auto-detected if None)
            
        Returns:
            True if tiled processing recommended
        """
        width, height = image.size
        
        # Get available memory
        if available_memory_gb is None:
            available_memory_gb = self._get_available_memory()
        
        # Calculate estimated memory usage for the operation
        estimated_memory_gb = self._estimate_memory_usage(image, operation_type)
        
        # Operation complexity factors
        complexity_factors = {
            'edit': 1.0,
            'inpaint': 1.2,
            'outpaint': 1.5,
            'style_transfer': 2.0
        }
        
        complexity_factor = complexity_factors.get(operation_type, 1.0)
        adjusted_memory = estimated_memory_gb * complexity_factor
        
        # Decision logic
        # 1. Always use tiling for very large images
        if width > 1536 or height > 1536:
            logger.info(f"🔧 Tiling recommended: Large image {width}x{height}")
            return True
        
        # 2. Use tiling if estimated memory exceeds 70% of available memory
        memory_ratio = adjusted_memory / available_memory_gb
        if memory_ratio > 0.7:
            logger.info(f"🔧 Tiling recommended: Memory usage {adjusted_memory:.1f}GB > 70% of {available_memory_gb:.1f}GB")
            return True
        
        # 3. Use tiling for complex operations on medium-large images
        if operation_type in ['outpaint', 'style_transfer'] and (width > 1024 or height > 1024):
            logger.info(f"🔧 Tiling recommended: Complex operation {operation_type} on {width}x{height}")
            return True
        
        # 4. Conservative approach for low memory systems
        if available_memory_gb < self.memory_thresholds['low'] and (width > 768 or height > 768):
            logger.info(f"🔧 Tiling recommended: Low memory system ({available_memory_gb:.1f}GB) with {width}x{height}")
            return True
        
        logger.debug(f"Single-pass processing suitable for {width}x{height} image")
        return False
    
    def _get_available_memory(self) -> float:
        """Get available GPU memory in GB"""
        try:
            if torch.cuda.is_available():
                # Get current GPU memory usage
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                # Conservative estimate: use 80% of free memory
                available = (total - reserved) * 0.8
                
                logger.debug(f"GPU Memory - Total: {total:.1f}GB, Reserved: {reserved:.1f}GB, Available: {available:.1f}GB")
                return max(available, 1.0)  # Minimum 1GB
            else:
                # CPU processing - use system memory
                import psutil
                available_gb = psutil.virtual_memory().available / 1e9
                return min(available_gb * 0.5, 8.0)  # Use up to 50% of system memory, max 8GB
                
        except Exception as e:
            logger.warning(f"Failed to get available memory: {e}")
            return 4.0  # Default fallback
    
    def _estimate_memory_usage(self, image: Image.Image, operation_type: str) -> float:
        """Estimate memory usage for image operation in GB"""
        width, height = image.size
        pixels = width * height
        
        # Base memory usage (input + output + intermediate tensors)
        base_memory = pixels * 3 * 4 * 3 / 1e9  # RGB float32 * 3 copies
        
        # Operation-specific multipliers
        operation_multipliers = {
            'edit': 1.0,
            'inpaint': 1.3,  # Additional mask processing
            'outpaint': 1.8,  # Larger canvas
            'style_transfer': 2.5  # Style image + additional processing
        }
        
        multiplier = operation_multipliers.get(operation_type, 1.0)
        estimated_memory = base_memory * multiplier
        
        # Add model overhead (approximate)
        model_overhead = 2.0  # GB
        
        return estimated_memory + model_overhead
    
    def calculate_optimal_tiles(self, image: Image.Image, available_memory_gb: Optional[float] = None) -> List[Tuple[int, int, int, int]]:
        """
        Calculate optimal tile coordinates based on image size and available memory
        
        Args:
            image: Input image
            available_memory_gb: Available memory for processing
            
        Returns:
            List of optimized tile coordinates (x1, y1, x2, y2)
        """
        width, height = image.size
        
        if available_memory_gb is None:
            available_memory_gb = self._get_available_memory()
        
        # Calculate optimal tile size based on available memory
        optimal_tile_size = self._calculate_optimal_tile_size(available_memory_gb)
        
        # Use optimal tile size for this calculation
        original_tile_size = self.tile_size
        self.tile_size = optimal_tile_size
        
        try:
            tiles = self._calculate_tiles_with_optimization(image)
            logger.info(f"🔧 Calculated {len(tiles)} optimal tiles (size: {optimal_tile_size}x{optimal_tile_size}) for {width}x{height} image")
            return tiles
        finally:
            # Restore original tile size
            self.tile_size = original_tile_size
    
    def _calculate_optimal_tile_size(self, available_memory_gb: float) -> int:
        """Calculate optimal tile size based on available memory"""
        # Conservative memory usage per tile (including overhead)
        memory_per_tile_gb = 0.5  # 500MB per tile
        
        # Calculate how many tiles we can process simultaneously
        max_concurrent_tiles = max(1, int(available_memory_gb / memory_per_tile_gb))
        
        # Adjust tile size based on memory
        if available_memory_gb >= 8.0:
            tile_size = 768
        elif available_memory_gb >= 4.0:
            tile_size = 512
        elif available_memory_gb >= 2.0:
            tile_size = 384
        else:
            tile_size = 256
        
        # Ensure tile size is within bounds
        tile_size = max(self.min_tile_size, min(tile_size, self.max_tile_size))
        
        logger.debug(f"Optimal tile size: {tile_size}x{tile_size} for {available_memory_gb:.1f}GB memory")
        return tile_size
    
    def _calculate_tiles_with_optimization(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Calculate tiles with optimization for edge cases"""
        width, height = image.size
        tiles = []
        
        # Calculate step size (tile size minus overlap)
        step_x = self.tile_size - self.overlap
        step_y = self.tile_size - self.overlap
        
        # Calculate number of tiles needed
        tiles_x = max(1, (width + step_x - 1) // step_x)
        tiles_y = max(1, (height + step_y - 1) // step_y)
        
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                # Calculate tile boundaries
                x1 = tx * step_x
                y1 = ty * step_y
                x2 = min(x1 + self.tile_size, width)
                y2 = min(y1 + self.tile_size, height)
                
                # Adjust for edge tiles to ensure minimum size
                if x2 - x1 < self.min_tile_size and tx > 0:
                    x1 = max(0, x2 - self.min_tile_size)
                if y2 - y1 < self.min_tile_size and ty > 0:
                    y1 = max(0, y2 - self.min_tile_size)
                
                # Only add tiles that meet minimum size requirements
                if (x2 - x1) >= self.min_tile_size and (y2 - y1) >= self.min_tile_size:
                    tiles.append((x1, y1, x2, y2))
        
        return tiles
    
    def calculate_tiles(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Calculate tile coordinates for image (backward compatibility)
        
        Args:
            image: Input image
            
        Returns:
            List of tile coordinates (x1, y1, x2, y2)
        """
        return self.calculate_optimal_tiles(image)
    
    def set_progress_callback(self, callback):
        """Set progress callback function for tracking tiled operations"""
        self.progress_callback = callback
    
    def process_tiles_with_progress(
        self,
        image: Image.Image,
        process_function,
        *args,
        **kwargs
    ) -> Image.Image:
        """
        Process image tiles with progress tracking
        
        Args:
            image: Input image
            process_function: Function to process each tile
            *args, **kwargs: Arguments for process_function
            
        Returns:
            Processed image
        """
        # Calculate tiles
        tile_coords = self.calculate_optimal_tiles(image)
        total_tiles = len(tile_coords)
        
        if total_tiles == 1:
            # Single tile - no need for tiling
            logger.debug("Single tile processing")
            if self.progress_callback:
                self.progress_callback(0, 1, "Processing image...")
            
            result = process_function(image, *args, **kwargs)
            
            if self.progress_callback:
                self.progress_callback(1, 1, "Complete")
            
            return result
        
        logger.info(f"🔧 Processing {total_tiles} tiles with progress tracking")
        processed_tiles = []
        
        for i, (x1, y1, x2, y2) in enumerate(tile_coords):
            # Update progress
            if self.progress_callback:
                progress_msg = f"Processing tile {i+1}/{total_tiles} ({x1},{y1} to {x2},{y2})"
                self.progress_callback(i, total_tiles, progress_msg)
            
            # Extract and process tile
            tile = image.crop((x1, y1, x2, y2))
            
            try:
                processed_tile = process_function(tile, *args, **kwargs)
                if processed_tile is None:
                    logger.warning(f"Tile {i+1} processing failed, using original")
                    processed_tile = tile
                processed_tiles.append(processed_tile)
                
            except Exception as e:
                logger.error(f"Error processing tile {i+1}: {e}")
                processed_tiles.append(tile)  # Use original tile as fallback
        
        # Final progress update
        if self.progress_callback:
            self.progress_callback(total_tiles, total_tiles, "Merging tiles...")
        
        # Merge tiles
        merged_image = self.merge_tiles_with_blending(processed_tiles, tile_coords, image.size)
        
        if self.progress_callback:
            self.progress_callback(total_tiles, total_tiles, "Complete")
        
        logger.info(f"✅ Tiled processing completed: {total_tiles} tiles merged")
        return merged_image
    
    def merge_tiles_with_blending(
        self,
        tiles: List[Image.Image],
        tile_coords: List[Tuple[int, int, int, int]],
        output_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Enhanced tile merging with overlap blending
        
        Args:
            tiles: List of processed tile images
            tile_coords: List of tile coordinates
            output_size: Final output image size
            
        Returns:
            Merged image with blended overlaps
        """
        try:
            # Create output image and weight map
            merged_image = Image.new('RGB', output_size, (0, 0, 0))
            weight_map = Image.new('L', output_size, 0)
            
            for tile, (x1, y1, x2, y2) in zip(tiles, tile_coords):
                # Resize tile if necessary
                expected_size = (x2 - x1, y2 - y1)
                if tile.size != expected_size:
                    tile = tile.resize(expected_size, Image.Resampling.LANCZOS)
                
                # Create weight mask for this tile (higher weight in center, lower at edges)
                tile_weight = self._create_tile_weight_mask(expected_size)
                
                # Convert images to arrays for blending
                tile_array = np.array(tile).astype(np.float32)
                merged_array = np.array(merged_image.crop((x1, y1, x2, y2))).astype(np.float32)
                
                # Get current weights
                current_weight = np.array(weight_map.crop((x1, y1, x2, y2))).astype(np.float32) / 255.0
                new_weight = np.array(tile_weight).astype(np.float32) / 255.0
                
                # Blend images
                total_weight = current_weight + new_weight
                total_weight = np.maximum(total_weight, 1e-8)  # Avoid division by zero
                
                blended = (merged_array * current_weight[..., np.newaxis] + 
                          tile_array * new_weight[..., np.newaxis]) / total_weight[..., np.newaxis]
                
                blended = np.clip(blended, 0, 255).astype(np.uint8)
                
                # Update merged image
                blended_tile = Image.fromarray(blended)
                merged_image.paste(blended_tile, (x1, y1))
                
                # Update weight map
                updated_weight = np.clip(total_weight * 255, 0, 255).astype(np.uint8)
                weight_tile = Image.fromarray(updated_weight, mode='L')
                weight_map.paste(weight_tile, (x1, y1))
            
            logger.debug(f"✅ Merged {len(tiles)} tiles with blending into {output_size} image")
            return merged_image
            
        except Exception as e:
            logger.error(f"Failed to merge tiles with blending: {e}")
            # Fallback to simple merging
            return self.merge_tiles(tiles, tile_coords, output_size)
    
    def _create_tile_weight_mask(self, size: Tuple[int, int]) -> Image.Image:
        """Create weight mask for tile blending (higher weight in center)"""
        width, height = size
        
        # Create distance-based weight mask
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Distance from edges
        dist_from_left = x_coords
        dist_from_right = width - 1 - x_coords
        dist_from_top = y_coords
        dist_from_bottom = height - 1 - y_coords
        
        # Minimum distance to any edge
        min_dist = np.minimum(
            np.minimum(dist_from_left, dist_from_right),
            np.minimum(dist_from_top, dist_from_bottom)
        )
        
        # Normalize to 0-1 range
        max_dist = min(width, height) // 2
        weight = np.minimum(min_dist / max_dist, 1.0)
        
        # Convert to 0-255 range
        weight_mask = (weight * 255).astype(np.uint8)
        
        return Image.fromarray(weight_mask, mode='L')
    
    def merge_tiles(
        self,
        tiles: List[Image.Image],
        tile_coords: List[Tuple[int, int, int, int]],
        output_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Simple tile merging (backward compatibility)
        
        Args:
            tiles: List of processed tile images
            tile_coords: List of tile coordinates
            output_size: Final output image size
            
        Returns:
            Merged image
        """
        try:
            # Create output image
            merged_image = Image.new('RGB', output_size, (0, 0, 0))
            
            # Simple paste without blending
            for tile, (x1, y1, x2, y2) in zip(tiles, tile_coords):
                # Resize tile if necessary
                expected_size = (x2 - x1, y2 - y1)
                if tile.size != expected_size:
                    tile = tile.resize(expected_size, Image.Resampling.LANCZOS)
                
                # Simple paste
                merged_image.paste(tile, (x1, y1))
            
            logger.debug(f"✅ Merged {len(tiles)} tiles into {output_size} image")
            return merged_image
            
        except Exception as e:
            logger.error(f"Failed to merge tiles: {e}")
            # Return first tile as fallback
            return tiles[0] if tiles else Image.new('RGB', output_size, (0, 0, 0))


# Utility functions
def estimate_processing_time(image: Image.Image, operation: str = "edit") -> float:
    """Estimate processing time for image operation"""
    width, height = image.size
    pixels = width * height
    
    # Base time estimates (in seconds)
    base_times = {
        "edit": 0.000002,  # 2 microseconds per pixel
        "inpaint": 0.000003,  # 3 microseconds per pixel
        "outpaint": 0.000004,  # 4 microseconds per pixel
        "style_transfer": 0.000005  # 5 microseconds per pixel
    }
    
    base_time = base_times.get(operation, 0.000003)
    estimated_time = pixels * base_time
    
    # Add base overhead
    estimated_time += 2.0  # 2 seconds base overhead
    
    return estimated_time


def get_optimal_dimensions(width: int, height: int, max_dimension: int = 1024) -> Tuple[int, int]:
    """Get optimal dimensions for processing"""
    if width <= max_dimension and height <= max_dimension:
        return width, height
    
    # Scale down maintaining aspect ratio
    scale = min(max_dimension / width, max_dimension / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Ensure multiples of 8
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    return new_width, new_height