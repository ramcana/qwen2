"""
Preset Management System for DiffSynth Enhanced UI
Provides comprehensive preset management for different use cases and workflows
"""

import json
import os
import shutil
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PresetCategory(Enum):
    """Preset categories for different use cases"""
    PHOTO_EDITING = "photo_editing"
    ARTISTIC_CREATION = "artistic_creation"
    TECHNICAL_ILLUSTRATION = "technical_illustration"
    CONTROLNET = "controlnet"
    DIFFSYNTH = "diffsynth"
    GENERAL = "general"
    CUSTOM = "custom"


class PresetType(Enum):
    """Types of presets"""
    GENERATION = "generation"
    EDITING = "editing"
    CONTROLNET = "controlnet"
    SYSTEM = "system"
    WORKFLOW = "workflow"


@dataclass
class PresetMetadata:
    """Metadata for presets"""
    name: str
    description: str
    category: PresetCategory
    preset_type: PresetType
    version: str = "1.0"
    author: str = "user"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    compatibility: Dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0
    rating: float = 0.0


@dataclass
class GenerationPreset:
    """Preset for text-to-image generation"""
    width: int = 768
    height: int = 768
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    negative_prompt: str = ""
    seed: Optional[int] = None
    
    # Advanced settings
    scheduler: str = "DPMSolverMultistepScheduler"
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_cpu_offload: bool = False
    
    # Quality settings
    quality_preset: str = "balanced"
    upscale_factor: float = 1.0
    enhance_details: bool = False


@dataclass
class EditingPreset:
    """Preset for image editing operations"""
    strength: float = 0.8
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    negative_prompt: str = ""
    
    # Editing specific
    inpaint_full_res: bool = True
    inpaint_full_res_padding: int = 32
    mask_blur: int = 4
    
    # Style transfer
    style_strength: float = 0.7
    preserve_color: bool = False
    
    # Outpainting
    outpaint_direction: str = "all"
    outpaint_pixels: int = 256
    
    # Advanced
    use_tiled_processing: bool = False
    tile_overlap: int = 64
    enable_eligen: bool = True
    eligen_mode: str = "enhanced"


@dataclass
class ControlNetPreset:
    """Preset for ControlNet operations"""
    control_type: str = "canny"
    conditioning_scale: float = 1.0
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0
    
    # Detection settings
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200
    depth_near_plane: float = 0.1
    depth_far_plane: float = 100.0
    
    # Processing
    detect_resolution: int = 512
    image_resolution: int = 768
    
    # Advanced
    multi_controlnet: bool = False
    controlnet_weights: List[float] = field(default_factory=lambda: [1.0])


@dataclass
class SystemPreset:
    """Preset for system configuration"""
    device: str = "cuda"
    torch_dtype: str = "float16"
    enable_xformers: bool = True
    enable_memory_efficient_attention: bool = True
    
    # Memory management
    max_memory_usage_gb: float = 8.0
    enable_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    
    # Performance
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True
    
    # Safety
    safety_checker: bool = True
    requires_safety_checker: bool = False


@dataclass
class WorkflowPreset:
    """Preset for complete workflows"""
    workflow_type: str = "text_to_image"
    steps: List[Dict[str, Any]] = field(default_factory=list)
    
    # Workflow settings
    auto_save: bool = True
    save_intermediate: bool = False
    comparison_mode: bool = True
    
    # Integration
    use_qwen: bool = True
    use_diffsynth: bool = False
    use_controlnet: bool = False


@dataclass
class Preset:
    """Complete preset configuration"""
    metadata: PresetMetadata
    generation: Optional[GenerationPreset] = None
    editing: Optional[EditingPreset] = None
    controlnet: Optional[ControlNetPreset] = None
    system: Optional[SystemPreset] = None
    workflow: Optional[WorkflowPreset] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert preset to dictionary"""
        return {
            "metadata": asdict(self.metadata),
            "generation": asdict(self.generation) if self.generation else None,
            "editing": asdict(self.editing) if self.editing else None,
            "controlnet": asdict(self.controlnet) if self.controlnet else None,
            "system": asdict(self.system) if self.system else None,
            "workflow": asdict(self.workflow) if self.workflow else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Preset':
        """Create preset from dictionary"""
        metadata = PresetMetadata(**data["metadata"])
        
        generation = GenerationPreset(**data["generation"]) if data.get("generation") else None
        editing = EditingPreset(**data["editing"]) if data.get("editing") else None
        controlnet = ControlNetPreset(**data["controlnet"]) if data.get("controlnet") else None
        system = SystemPreset(**data["system"]) if data.get("system") else None
        workflow = WorkflowPreset(**data["workflow"]) if data.get("workflow") else None
        
        return cls(
            metadata=metadata,
            generation=generation,
            editing=editing,
            controlnet=controlnet,
            system=system,
            workflow=workflow
        )


class PresetManager:
    """
    Comprehensive preset management system for DiffSynth Enhanced UI
    Handles saving, loading, categorization, and management of user configurations
    """
    
    def __init__(self, presets_dir: str = "config/presets"):
        """Initialize preset manager"""
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        
        # Create category directories
        for category in PresetCategory:
            category_dir = self.presets_dir / category.value
            category_dir.mkdir(exist_ok=True)
        
        # Internal storage
        self._presets: Dict[str, Preset] = {}
        self._preset_index: Dict[str, str] = {}  # name -> file_path
        
        # Load existing presets
        self._load_all_presets()
        
        # Create default presets if none exist
        if not self._presets:
            self._create_default_presets()
        
        logger.info(f"PresetManager initialized with {len(self._presets)} presets")
    
    def _load_all_presets(self) -> None:
        """Load all presets from disk"""
        try:
            for preset_file in self.presets_dir.rglob("*.json"):
                try:
                    preset = self._load_preset_file(preset_file)
                    if preset:
                        self._presets[preset.metadata.name] = preset
                        self._preset_index[preset.metadata.name] = str(preset_file)
                except Exception as e:
                    logger.error(f"Failed to load preset {preset_file}: {e}")
            
            logger.info(f"Loaded {len(self._presets)} presets from disk")
            
        except Exception as e:
            logger.error(f"Failed to load presets: {e}")
    
    def _load_preset_file(self, file_path: Path) -> Optional[Preset]:
        """Load a single preset file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert enum strings back to enums
            if "metadata" in data:
                metadata = data["metadata"]
                if "category" in metadata:
                    metadata["category"] = PresetCategory(metadata["category"])
                if "preset_type" in metadata:
                    metadata["preset_type"] = PresetType(metadata["preset_type"])
            
            return Preset.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to load preset file {file_path}: {e}")
            return None
    
    def _create_default_presets(self) -> None:
        """Create default presets for different use cases"""
        logger.info("Creating default presets...")
        
        # Photo Editing Presets
        self._create_photo_editing_presets()
        
        # Artistic Creation Presets
        self._create_artistic_presets()
        
        # Technical Illustration Presets
        self._create_technical_presets()
        
        # ControlNet Presets
        self._create_controlnet_presets()
        
        # System Presets
        self._create_system_presets()
        
        logger.info(f"Created {len(self._presets)} default presets")
    
    def _create_photo_editing_presets(self) -> None:
        """Create presets for photo editing use cases"""
        
        # Portrait Enhancement
        portrait_preset = Preset(
            metadata=PresetMetadata(
                name="Portrait Enhancement",
                description="Optimized settings for portrait photo enhancement and retouching",
                category=PresetCategory.PHOTO_EDITING,
                preset_type=PresetType.EDITING,
                tags=["portrait", "enhancement", "photo", "retouch"]
            ),
            editing=EditingPreset(
                strength=0.6,
                num_inference_steps=25,
                guidance_scale=6.0,
                negative_prompt="blurry, low quality, distorted, deformed",
                inpaint_full_res=True,
                mask_blur=2,
                enable_eligen=True,
                eligen_mode="enhanced"
            ),
            generation=GenerationPreset(
                width=768,
                height=768,
                num_inference_steps=25,
                guidance_scale=6.0,
                quality_preset="quality",
                enhance_details=True
            )
        )
        self.save_preset(portrait_preset)
        
        # Landscape Enhancement
        landscape_preset = Preset(
            metadata=PresetMetadata(
                name="Landscape Enhancement",
                description="Settings for enhancing landscape and nature photography",
                category=PresetCategory.PHOTO_EDITING,
                preset_type=PresetType.EDITING,
                tags=["landscape", "nature", "enhancement", "photo"]
            ),
            editing=EditingPreset(
                strength=0.7,
                num_inference_steps=20,
                guidance_scale=7.0,
                negative_prompt="blurry, low quality, oversaturated",
                use_tiled_processing=True,
                enable_eligen=True,
                eligen_mode="enhanced"
            ),
            generation=GenerationPreset(
                width=1024,
                height=768,
                num_inference_steps=20,
                guidance_scale=7.0,
                quality_preset="quality"
            )
        )
        self.save_preset(landscape_preset)
    
    def _create_artistic_presets(self) -> None:
        """Create presets for artistic creation"""
        
        # Digital Art
        digital_art_preset = Preset(
            metadata=PresetMetadata(
                name="Digital Art Creation",
                description="Settings optimized for creating digital artwork and illustrations",
                category=PresetCategory.ARTISTIC_CREATION,
                preset_type=PresetType.GENERATION,
                tags=["digital art", "illustration", "creative", "artistic"]
            ),
            generation=GenerationPreset(
                width=768,
                height=768,
                num_inference_steps=30,
                guidance_scale=8.0,
                negative_prompt="photorealistic, photograph, low quality",
                quality_preset="quality",
                enhance_details=True
            ),
            editing=EditingPreset(
                strength=0.8,
                num_inference_steps=25,
                guidance_scale=8.0,
                style_strength=0.8,
                enable_eligen=True,
                eligen_mode="ultra"
            )
        )
        self.save_preset(digital_art_preset)
        
        # Style Transfer
        style_transfer_preset = Preset(
            metadata=PresetMetadata(
                name="Artistic Style Transfer",
                description="Specialized settings for transferring artistic styles between images",
                category=PresetCategory.ARTISTIC_CREATION,
                preset_type=PresetType.EDITING,
                tags=["style transfer", "artistic", "transformation"]
            ),
            editing=EditingPreset(
                strength=0.9,
                num_inference_steps=20,
                guidance_scale=7.5,
                style_strength=0.9,
                preserve_color=False,
                enable_eligen=True,
                eligen_mode="enhanced"
            )
        )
        self.save_preset(style_transfer_preset)
    
    def _create_technical_presets(self) -> None:
        """Create presets for technical illustration"""
        
        # Technical Diagram
        technical_preset = Preset(
            metadata=PresetMetadata(
                name="Technical Illustration",
                description="Settings for creating technical diagrams and illustrations",
                category=PresetCategory.TECHNICAL_ILLUSTRATION,
                preset_type=PresetType.GENERATION,
                tags=["technical", "diagram", "illustration", "precise"]
            ),
            generation=GenerationPreset(
                width=1024,
                height=768,
                num_inference_steps=25,
                guidance_scale=9.0,
                negative_prompt="artistic, painterly, blurry, low quality",
                quality_preset="quality",
                enhance_details=True
            ),
            controlnet=ControlNetPreset(
                control_type="canny",
                conditioning_scale=1.2,
                detect_resolution=768,
                image_resolution=1024,
                canny_low_threshold=50,
                canny_high_threshold=150
            )
        )
        self.save_preset(technical_preset)
    
    def _create_controlnet_presets(self) -> None:
        """Create ControlNet-specific presets"""
        
        # Canny Edge Control
        canny_preset = Preset(
            metadata=PresetMetadata(
                name="Canny Edge Control",
                description="Precise edge-based image generation using Canny edge detection",
                category=PresetCategory.CONTROLNET,
                preset_type=PresetType.CONTROLNET,
                tags=["canny", "edges", "precise", "structure"]
            ),
            controlnet=ControlNetPreset(
                control_type="canny",
                conditioning_scale=1.0,
                canny_low_threshold=100,
                canny_high_threshold=200,
                detect_resolution=512,
                image_resolution=768
            ),
            generation=GenerationPreset(
                width=768,
                height=768,
                num_inference_steps=20,
                guidance_scale=7.5
            )
        )
        self.save_preset(canny_preset)
        
        # Depth Control
        depth_preset = Preset(
            metadata=PresetMetadata(
                name="Depth-Guided Generation",
                description="3D-aware image generation using depth maps",
                category=PresetCategory.CONTROLNET,
                preset_type=PresetType.CONTROLNET,
                tags=["depth", "3d", "spatial", "structure"]
            ),
            controlnet=ControlNetPreset(
                control_type="depth",
                conditioning_scale=0.8,
                depth_near_plane=0.1,
                depth_far_plane=100.0,
                detect_resolution=384,
                image_resolution=768
            ),
            generation=GenerationPreset(
                width=768,
                height=768,
                num_inference_steps=25,
                guidance_scale=7.0
            )
        )
        self.save_preset(depth_preset)
    
    def _create_system_presets(self) -> None:
        """Create system configuration presets"""
        
        # Performance Optimized
        performance_preset = Preset(
            metadata=PresetMetadata(
                name="Performance Optimized",
                description="System settings optimized for speed and efficiency",
                category=PresetCategory.GENERAL,
                preset_type=PresetType.SYSTEM,
                tags=["performance", "speed", "efficiency"]
            ),
            system=SystemPreset(
                device="cuda",
                torch_dtype="float16",
                enable_xformers=True,
                enable_memory_efficient_attention=True,
                max_memory_usage_gb=6.0,
                enable_attention_slicing=True,
                enable_vae_slicing=True,
                enable_vae_tiling=True
            ),
            generation=GenerationPreset(
                width=512,
                height=512,
                num_inference_steps=15,
                guidance_scale=6.0,
                quality_preset="fast"
            )
        )
        self.save_preset(performance_preset)
        
        # Quality Optimized
        quality_preset = Preset(
            metadata=PresetMetadata(
                name="Quality Optimized",
                description="System settings optimized for maximum quality output",
                category=PresetCategory.GENERAL,
                preset_type=PresetType.SYSTEM,
                tags=["quality", "high-res", "detailed"]
            ),
            system=SystemPreset(
                device="cuda",
                torch_dtype="float16",
                enable_xformers=True,
                enable_memory_efficient_attention=True,
                max_memory_usage_gb=12.0,
                enable_cpu_offload=False
            ),
            generation=GenerationPreset(
                width=1024,
                height=1024,
                num_inference_steps=30,
                guidance_scale=8.0,
                quality_preset="ultra",
                enhance_details=True,
                upscale_factor=1.2
            )
        )
        self.save_preset(quality_preset)
    
    def save_preset(self, preset: Preset) -> bool:
        """
        Save a preset to disk and memory
        
        Args:
            preset: Preset to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update metadata
            preset.metadata.updated_at = datetime.now().isoformat()
            
            # Determine file path
            category_dir = self.presets_dir / preset.metadata.category.value
            category_dir.mkdir(exist_ok=True)
            
            # Create safe filename
            safe_name = self._create_safe_filename(preset.metadata.name)
            file_path = category_dir / f"{safe_name}.json"
            
            # Convert to dictionary with enum handling
            preset_dict = preset.to_dict()
            preset_dict["metadata"]["category"] = preset.metadata.category.value
            preset_dict["metadata"]["preset_type"] = preset.metadata.preset_type.value
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(preset_dict, f, indent=2, ensure_ascii=False)
            
            # Update internal storage
            self._presets[preset.metadata.name] = preset
            self._preset_index[preset.metadata.name] = str(file_path)
            
            logger.info(f"Saved preset '{preset.metadata.name}' to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save preset '{preset.metadata.name}': {e}")
            return False
    
    def load_preset(self, name: str) -> Optional[Preset]:
        """
        Load a preset by name
        
        Args:
            name: Name of the preset to load
            
        Returns:
            Preset if found, None otherwise
        """
        try:
            if name in self._presets:
                # Update usage count
                self._presets[name].metadata.usage_count += 1
                return self._presets[name]
            
            logger.warning(f"Preset '{name}' not found")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load preset '{name}': {e}")
            return None
    
    def delete_preset(self, name: str) -> bool:
        """
        Delete a preset
        
        Args:
            name: Name of the preset to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if name not in self._presets:
                logger.warning(f"Preset '{name}' not found for deletion")
                return False
            
            # Remove from disk
            if name in self._preset_index:
                file_path = Path(self._preset_index[name])
                if file_path.exists():
                    file_path.unlink()
            
            # Remove from memory
            del self._presets[name]
            if name in self._preset_index:
                del self._preset_index[name]
            
            logger.info(f"Deleted preset '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete preset '{name}': {e}")
            return False
    
    def list_presets(
        self, 
        category: Optional[PresetCategory] = None,
        preset_type: Optional[PresetType] = None,
        tags: Optional[List[str]] = None
    ) -> List[Preset]:
        """
        List presets with optional filtering
        
        Args:
            category: Filter by category
            preset_type: Filter by type
            tags: Filter by tags (any match)
            
        Returns:
            List of matching presets
        """
        try:
            presets = list(self._presets.values())
            
            # Filter by category
            if category:
                presets = [p for p in presets if p.metadata.category == category]
            
            # Filter by type
            if preset_type:
                presets = [p for p in presets if p.metadata.preset_type == preset_type]
            
            # Filter by tags
            if tags:
                presets = [
                    p for p in presets 
                    if any(tag in p.metadata.tags for tag in tags)
                ]
            
            # Sort by usage count and rating
            presets.sort(
                key=lambda p: (p.metadata.usage_count, p.metadata.rating),
                reverse=True
            )
            
            return presets
            
        except Exception as e:
            logger.error(f"Failed to list presets: {e}")
            return []
    
    def get_preset_categories(self) -> Dict[PresetCategory, int]:
        """
        Get preset categories with counts
        
        Returns:
            Dictionary mapping categories to preset counts
        """
        try:
            categories = {}
            for preset in self._presets.values():
                category = preset.metadata.category
                categories[category] = categories.get(category, 0) + 1
            
            return categories
            
        except Exception as e:
            logger.error(f"Failed to get preset categories: {e}")
            return {}
    
    def export_preset(self, name: str, export_path: str) -> bool:
        """
        Export a preset to a file
        
        Args:
            name: Name of the preset to export
            export_path: Path to export the preset to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            preset = self.load_preset(name)
            if not preset:
                return False
            
            # Convert to dictionary with enum handling
            preset_dict = preset.to_dict()
            preset_dict["metadata"]["category"] = preset.metadata.category.value
            preset_dict["metadata"]["preset_type"] = preset.metadata.preset_type.value
            
            # Add export metadata
            preset_dict["export_info"] = {
                "exported_at": datetime.now().isoformat(),
                "exported_by": "PresetManager",
                "version": "1.0"
            }
            
            # Save to export path
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(preset_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported preset '{name}' to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export preset '{name}': {e}")
            return False
    
    def import_preset(self, import_path: str, overwrite: bool = False) -> bool:
        """
        Import a preset from a file
        
        Args:
            import_path: Path to the preset file to import
            overwrite: Whether to overwrite existing presets with the same name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import_file = Path(import_path)
            if not import_file.exists():
                logger.error(f"Import file not found: {import_path}")
                return False
            
            # Load preset data
            with open(import_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Remove export info if present
            if "export_info" in data:
                del data["export_info"]
            
            # Convert enum strings back to enums
            if "metadata" in data:
                metadata = data["metadata"]
                if "category" in metadata:
                    metadata["category"] = PresetCategory(metadata["category"])
                if "preset_type" in metadata:
                    metadata["preset_type"] = PresetType(metadata["preset_type"])
            
            # Create preset
            preset = Preset.from_dict(data)
            
            # Check for existing preset
            if preset.metadata.name in self._presets and not overwrite:
                logger.warning(f"Preset '{preset.metadata.name}' already exists (use overwrite=True)")
                return False
            
            # Save imported preset
            return self.save_preset(preset)
            
        except Exception as e:
            logger.error(f"Failed to import preset from '{import_path}': {e}")
            return False
    
    def duplicate_preset(self, name: str, new_name: str) -> bool:
        """
        Duplicate an existing preset with a new name
        
        Args:
            name: Name of the preset to duplicate
            new_name: Name for the new preset
            
        Returns:
            True if successful, False otherwise
        """
        try:
            original = self.load_preset(name)
            if not original:
                return False
            
            # Create copy with new metadata
            duplicate_dict = original.to_dict()
            duplicate_dict["metadata"]["name"] = new_name
            duplicate_dict["metadata"]["created_at"] = datetime.now().isoformat()
            duplicate_dict["metadata"]["updated_at"] = datetime.now().isoformat()
            duplicate_dict["metadata"]["usage_count"] = 0
            duplicate_dict["metadata"]["rating"] = 0.0
            
            # Convert back to preset
            duplicate = Preset.from_dict(duplicate_dict)
            
            return self.save_preset(duplicate)
            
        except Exception as e:
            logger.error(f"Failed to duplicate preset '{name}': {e}")
            return False
    
    def update_preset_rating(self, name: str, rating: float) -> bool:
        """
        Update the rating of a preset
        
        Args:
            name: Name of the preset
            rating: New rating (0.0 to 5.0)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if name not in self._presets:
                return False
            
            # Clamp rating to valid range
            rating = max(0.0, min(5.0, rating))
            
            # Update rating
            self._presets[name].metadata.rating = rating
            self._presets[name].metadata.updated_at = datetime.now().isoformat()
            
            # Save updated preset
            return self.save_preset(self._presets[name])
            
        except Exception as e:
            logger.error(f"Failed to update rating for preset '{name}': {e}")
            return False
    
    def search_presets(self, query: str) -> List[Preset]:
        """
        Search presets by name, description, or tags
        
        Args:
            query: Search query
            
        Returns:
            List of matching presets
        """
        try:
            query_lower = query.lower()
            matching_presets = []
            
            for preset in self._presets.values():
                # Search in name
                if query_lower in preset.metadata.name.lower():
                    matching_presets.append(preset)
                    continue
                
                # Search in description
                if query_lower in preset.metadata.description.lower():
                    matching_presets.append(preset)
                    continue
                
                # Search in tags
                if any(query_lower in tag.lower() for tag in preset.metadata.tags):
                    matching_presets.append(preset)
                    continue
            
            # Sort by relevance (usage count and rating)
            matching_presets.sort(
                key=lambda p: (p.metadata.usage_count, p.metadata.rating),
                reverse=True
            )
            
            return matching_presets
            
        except Exception as e:
            logger.error(f"Failed to search presets: {e}")
            return []
    
    def get_preset_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the preset collection
        
        Returns:
            Dictionary with statistics
        """
        try:
            stats = {
                "total_presets": len(self._presets),
                "categories": {},
                "types": {},
                "most_used": None,
                "highest_rated": None,
                "recent_presets": []
            }
            
            # Category and type counts
            for preset in self._presets.values():
                category = preset.metadata.category.value
                preset_type = preset.metadata.preset_type.value
                
                stats["categories"][category] = stats["categories"].get(category, 0) + 1
                stats["types"][preset_type] = stats["types"].get(preset_type, 0) + 1
            
            # Most used preset
            if self._presets:
                most_used = max(self._presets.values(), key=lambda p: p.metadata.usage_count)
                stats["most_used"] = {
                    "name": most_used.metadata.name,
                    "usage_count": most_used.metadata.usage_count
                }
                
                # Highest rated preset
                highest_rated = max(self._presets.values(), key=lambda p: p.metadata.rating)
                stats["highest_rated"] = {
                    "name": highest_rated.metadata.name,
                    "rating": highest_rated.metadata.rating
                }
                
                # Recent presets (last 5 created)
                recent = sorted(
                    self._presets.values(),
                    key=lambda p: p.metadata.created_at,
                    reverse=True
                )[:5]
                stats["recent_presets"] = [
                    {
                        "name": p.metadata.name,
                        "created_at": p.metadata.created_at,
                        "category": p.metadata.category.value
                    }
                    for p in recent
                ]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get preset statistics: {e}")
            return {}
    
    def _create_safe_filename(self, name: str) -> str:
        """Create a safe filename from preset name"""
        # Replace invalid characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        safe_name = "".join(c if c in safe_chars else "_" for c in name)
        
        # Limit length
        if len(safe_name) > 50:
            safe_name = safe_name[:50]
        
        return safe_name or "preset"
    
    def backup_presets(self, backup_path: str) -> bool:
        """
        Create a backup of all presets
        
        Args:
            backup_path: Path to save the backup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup data
            backup_data = {
                "backup_info": {
                    "created_at": datetime.now().isoformat(),
                    "preset_count": len(self._presets),
                    "version": "1.0"
                },
                "presets": {}
            }
            
            # Add all presets
            for name, preset in self._presets.items():
                preset_dict = preset.to_dict()
                preset_dict["metadata"]["category"] = preset.metadata.category.value
                preset_dict["metadata"]["preset_type"] = preset.metadata.preset_type.value
                backup_data["presets"][name] = preset_dict
            
            # Save backup
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created preset backup with {len(self._presets)} presets at {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create preset backup: {e}")
            return False
    
    def restore_presets(self, backup_path: str, overwrite: bool = False) -> bool:
        """
        Restore presets from a backup
        
        Args:
            backup_path: Path to the backup file
            overwrite: Whether to overwrite existing presets
            
        Returns:
            True if successful, False otherwise
        """
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Load backup data
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            if "presets" not in backup_data:
                logger.error("Invalid backup file format")
                return False
            
            restored_count = 0
            
            # Restore each preset
            for name, preset_data in backup_data["presets"].items():
                try:
                    # Convert enum strings back to enums
                    if "metadata" in preset_data:
                        metadata = preset_data["metadata"]
                        if "category" in metadata:
                            metadata["category"] = PresetCategory(metadata["category"])
                        if "preset_type" in metadata:
                            metadata["preset_type"] = PresetType(metadata["preset_type"])
                    
                    # Create preset
                    preset = Preset.from_dict(preset_data)
                    
                    # Check for existing preset
                    if preset.metadata.name in self._presets and not overwrite:
                        logger.debug(f"Skipping existing preset '{preset.metadata.name}'")
                        continue
                    
                    # Save restored preset
                    if self.save_preset(preset):
                        restored_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to restore preset '{name}': {e}")
            
            logger.info(f"Restored {restored_count} presets from backup")
            return restored_count > 0
            
        except Exception as e:
            logger.error(f"Failed to restore presets from backup: {e}")
            return False