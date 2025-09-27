"""
Tests for PresetManager
Comprehensive testing of preset management functionality
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pytest

from src.preset_manager import (
    PresetManager, Preset, PresetMetadata, GenerationPreset, EditingPreset,
    ControlNetPreset, SystemPreset, WorkflowPreset, PresetCategory, PresetType
)


class TestPresetManager:
    """Test suite for PresetManager"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create temporary directory for test presets
        self.temp_dir = tempfile.mkdtemp()
        self.preset_manager = PresetManager(presets_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_preset(self, name: str = "Test Preset") -> Preset:
        """Create a test preset for testing"""
        return Preset(
            metadata=PresetMetadata(
                name=name,
                description="A test preset for unit testing",
                category=PresetCategory.GENERAL,
                preset_type=PresetType.GENERATION,
                tags=["test", "unit_test"]
            ),
            generation=GenerationPreset(
                width=512,
                height=512,
                num_inference_steps=20,
                guidance_scale=7.5
            )
        )
    
    def test_initialization(self):
        """Test PresetManager initialization"""
        # Check that directories are created
        assert os.path.exists(self.temp_dir)
        
        # Check that category directories are created
        for category in PresetCategory:
            category_dir = Path(self.temp_dir) / category.value
            assert category_dir.exists()
        
        # Check that default presets are created
        assert len(self.preset_manager._presets) > 0
        
        # Check that presets are properly indexed
        assert len(self.preset_manager._preset_index) == len(self.preset_manager._presets)
    
    def test_save_preset(self):
        """Test saving presets"""
        preset = self.create_test_preset("Save Test Preset")
        
        # Save preset
        result = self.preset_manager.save_preset(preset)
        assert result is True
        
        # Check that preset is in memory
        assert "Save Test Preset" in self.preset_manager._presets
        
        # Check that preset file exists
        assert "Save Test Preset" in self.preset_manager._preset_index
        file_path = self.preset_manager._preset_index["Save Test Preset"]
        assert os.path.exists(file_path)
        
        # Verify file content
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        assert data["metadata"]["name"] == "Save Test Preset"
        assert data["metadata"]["category"] == "general"
        assert data["generation"]["width"] == 512
    
    def test_load_preset(self):
        """Test loading presets"""
        preset = self.create_test_preset("Load Test Preset")
        self.preset_manager.save_preset(preset)
        
        # Load preset
        loaded_preset = self.preset_manager.load_preset("Load Test Preset")
        
        assert loaded_preset is not None
        assert loaded_preset.metadata.name == "Load Test Preset"
        assert loaded_preset.generation.width == 512
        
        # Check usage count increment
        original_count = loaded_preset.metadata.usage_count
        self.preset_manager.load_preset("Load Test Preset")
        reloaded_preset = self.preset_manager.load_preset("Load Test Preset")
        assert reloaded_preset.metadata.usage_count > original_count
    
    def test_load_nonexistent_preset(self):
        """Test loading non-existent preset"""
        result = self.preset_manager.load_preset("Nonexistent Preset")
        assert result is None
    
    def test_delete_preset(self):
        """Test deleting presets"""
        preset = self.create_test_preset("Delete Test Preset")
        self.preset_manager.save_preset(preset)
        
        # Verify preset exists
        assert "Delete Test Preset" in self.preset_manager._presets
        
        # Delete preset
        result = self.preset_manager.delete_preset("Delete Test Preset")
        assert result is True
        
        # Verify preset is removed
        assert "Delete Test Preset" not in self.preset_manager._presets
        assert "Delete Test Preset" not in self.preset_manager._preset_index
        
        # Verify file is removed
        category_dir = Path(self.temp_dir) / "general"
        preset_files = list(category_dir.glob("Delete_Test_Preset.json"))
        assert len(preset_files) == 0
    
    def test_delete_nonexistent_preset(self):
        """Test deleting non-existent preset"""
        result = self.preset_manager.delete_preset("Nonexistent Preset")
        assert result is False
    
    def test_list_presets(self):
        """Test listing presets with filtering"""
        # Create test presets with different categories and types
        presets = [
            Preset(
                metadata=PresetMetadata(
                    name="Photo Preset",
                    description="Photo editing preset",
                    category=PresetCategory.PHOTO_EDITING,
                    preset_type=PresetType.EDITING,
                    tags=["photo", "editing"]
                ),
                editing=EditingPreset()
            ),
            Preset(
                metadata=PresetMetadata(
                    name="Art Preset",
                    description="Artistic creation preset",
                    category=PresetCategory.ARTISTIC_CREATION,
                    preset_type=PresetType.GENERATION,
                    tags=["art", "creative"]
                ),
                generation=GenerationPreset()
            ),
            Preset(
                metadata=PresetMetadata(
                    name="ControlNet Preset",
                    description="ControlNet preset",
                    category=PresetCategory.CONTROLNET,
                    preset_type=PresetType.CONTROLNET,
                    tags=["controlnet", "structure"]
                ),
                controlnet=ControlNetPreset()
            )
        ]
        
        for preset in presets:
            self.preset_manager.save_preset(preset)
        
        # Test listing all presets
        all_presets = self.preset_manager.list_presets()
        assert len(all_presets) >= 3  # At least our test presets
        
        # Test filtering by category
        photo_presets = self.preset_manager.list_presets(category=PresetCategory.PHOTO_EDITING)
        assert len(photo_presets) >= 1
        assert all(p.metadata.category == PresetCategory.PHOTO_EDITING for p in photo_presets)
        
        # Test filtering by type
        editing_presets = self.preset_manager.list_presets(preset_type=PresetType.EDITING)
        assert len(editing_presets) >= 1
        assert all(p.metadata.preset_type == PresetType.EDITING for p in editing_presets)
        
        # Test filtering by tags
        photo_tagged = self.preset_manager.list_presets(tags=["photo"])
        assert len(photo_tagged) >= 1
        assert all("photo" in p.metadata.tags for p in photo_tagged)
    
    def test_get_preset_categories(self):
        """Test getting preset categories with counts"""
        # Create presets in different categories
        presets = [
            self.create_test_preset("General 1"),
            self.create_test_preset("General 2"),
        ]
        
        # Modify categories
        presets[0].metadata.category = PresetCategory.PHOTO_EDITING
        presets[1].metadata.category = PresetCategory.ARTISTIC_CREATION
        
        for preset in presets:
            self.preset_manager.save_preset(preset)
        
        categories = self.preset_manager.get_preset_categories()
        
        # Should have at least our test categories plus defaults
        assert PresetCategory.PHOTO_EDITING in categories
        assert PresetCategory.ARTISTIC_CREATION in categories
        assert categories[PresetCategory.PHOTO_EDITING] >= 1
        assert categories[PresetCategory.ARTISTIC_CREATION] >= 1
    
    def test_export_preset(self):
        """Test exporting presets"""
        preset = self.create_test_preset("Export Test Preset")
        self.preset_manager.save_preset(preset)
        
        # Export preset
        export_path = os.path.join(self.temp_dir, "exported_preset.json")
        result = self.preset_manager.export_preset("Export Test Preset", export_path)
        
        assert result is True
        assert os.path.exists(export_path)
        
        # Verify export content
        with open(export_path, 'r') as f:
            data = json.load(f)
        
        assert data["metadata"]["name"] == "Export Test Preset"
        assert "export_info" in data
        assert "exported_at" in data["export_info"]
    
    def test_export_nonexistent_preset(self):
        """Test exporting non-existent preset"""
        export_path = os.path.join(self.temp_dir, "nonexistent.json")
        result = self.preset_manager.export_preset("Nonexistent", export_path)
        assert result is False
    
    def test_import_preset(self):
        """Test importing presets"""
        # Create and export a preset first
        preset = self.create_test_preset("Import Test Preset")
        self.preset_manager.save_preset(preset)
        
        export_path = os.path.join(self.temp_dir, "import_test.json")
        self.preset_manager.export_preset("Import Test Preset", export_path)
        
        # Delete the preset from manager
        self.preset_manager.delete_preset("Import Test Preset")
        assert "Import Test Preset" not in self.preset_manager._presets
        
        # Import the preset back
        result = self.preset_manager.import_preset(export_path)
        assert result is True
        
        # Verify preset is imported
        assert "Import Test Preset" in self.preset_manager._presets
        imported_preset = self.preset_manager.load_preset("Import Test Preset")
        assert imported_preset.metadata.name == "Import Test Preset"
    
    def test_import_nonexistent_file(self):
        """Test importing from non-existent file"""
        result = self.preset_manager.import_preset("nonexistent.json")
        assert result is False
    
    def test_import_existing_preset_no_overwrite(self):
        """Test importing existing preset without overwrite"""
        preset = self.create_test_preset("Existing Preset")
        self.preset_manager.save_preset(preset)
        
        # Export preset
        export_path = os.path.join(self.temp_dir, "existing.json")
        self.preset_manager.export_preset("Existing Preset", export_path)
        
        # Try to import without overwrite
        result = self.preset_manager.import_preset(export_path, overwrite=False)
        assert result is False
        
        # Try to import with overwrite
        result = self.preset_manager.import_preset(export_path, overwrite=True)
        assert result is True
    
    def test_duplicate_preset(self):
        """Test duplicating presets"""
        preset = self.create_test_preset("Original Preset")
        self.preset_manager.save_preset(preset)
        
        # Duplicate preset
        result = self.preset_manager.duplicate_preset("Original Preset", "Duplicated Preset")
        assert result is True
        
        # Verify both presets exist
        assert "Original Preset" in self.preset_manager._presets
        assert "Duplicated Preset" in self.preset_manager._presets
        
        # Verify duplicate has different metadata
        original = self.preset_manager._presets["Original Preset"]
        duplicate = self.preset_manager._presets["Duplicated Preset"]
        
        assert original.metadata.name != duplicate.metadata.name
        assert duplicate.metadata.usage_count == 0
        assert duplicate.metadata.rating == 0.0
        
        # Verify content is the same
        assert original.generation.width == duplicate.generation.width
        assert original.generation.height == duplicate.generation.height
    
    def test_duplicate_nonexistent_preset(self):
        """Test duplicating non-existent preset"""
        result = self.preset_manager.duplicate_preset("Nonexistent", "New Name")
        assert result is False
    
    def test_update_preset_rating(self):
        """Test updating preset ratings"""
        preset = self.create_test_preset("Rating Test Preset")
        self.preset_manager.save_preset(preset)
        
        # Update rating
        result = self.preset_manager.update_preset_rating("Rating Test Preset", 4.5)
        assert result is True
        
        # Verify rating is updated
        updated_preset = self.preset_manager.load_preset("Rating Test Preset")
        assert updated_preset.metadata.rating == 4.5
        
        # Test rating clamping
        self.preset_manager.update_preset_rating("Rating Test Preset", 10.0)
        clamped_preset = self.preset_manager.load_preset("Rating Test Preset")
        assert clamped_preset.metadata.rating == 5.0
        
        self.preset_manager.update_preset_rating("Rating Test Preset", -1.0)
        clamped_preset = self.preset_manager.load_preset("Rating Test Preset")
        assert clamped_preset.metadata.rating == 0.0
    
    def test_update_rating_nonexistent_preset(self):
        """Test updating rating for non-existent preset"""
        result = self.preset_manager.update_preset_rating("Nonexistent", 4.0)
        assert result is False
    
    def test_search_presets(self):
        """Test searching presets"""
        # Create presets with different names, descriptions, and tags
        presets = [
            Preset(
                metadata=PresetMetadata(
                    name="Portrait Enhancement",
                    description="Enhance portrait photos",
                    category=PresetCategory.PHOTO_EDITING,
                    preset_type=PresetType.EDITING,
                    tags=["portrait", "photo", "enhancement"]
                ),
                editing=EditingPreset()
            ),
            Preset(
                metadata=PresetMetadata(
                    name="Landscape Art",
                    description="Create artistic landscapes",
                    category=PresetCategory.ARTISTIC_CREATION,
                    preset_type=PresetType.GENERATION,
                    tags=["landscape", "art", "nature"]
                ),
                generation=GenerationPreset()
            ),
            Preset(
                metadata=PresetMetadata(
                    name="Technical Drawing",
                    description="Generate technical illustrations",
                    category=PresetCategory.TECHNICAL_ILLUSTRATION,
                    preset_type=PresetType.GENERATION,
                    tags=["technical", "drawing", "illustration"]
                ),
                generation=GenerationPreset()
            )
        ]
        
        for preset in presets:
            self.preset_manager.save_preset(preset)
        
        # Search by name
        name_results = self.preset_manager.search_presets("Portrait")
        assert len(name_results) >= 1
        assert any("Portrait" in p.metadata.name for p in name_results)
        
        # Search by description
        desc_results = self.preset_manager.search_presets("artistic")
        assert len(desc_results) >= 1
        assert any("artistic" in p.metadata.description.lower() for p in desc_results)
        
        # Search by tag
        tag_results = self.preset_manager.search_presets("landscape")
        assert len(tag_results) >= 1
        assert any("landscape" in p.metadata.tags for p in tag_results)
        
        # Search with no results
        no_results = self.preset_manager.search_presets("nonexistent_term")
        assert len(no_results) == 0
    
    def test_get_preset_statistics(self):
        """Test getting preset statistics"""
        # Create presets with different properties
        presets = [
            self.create_test_preset("Stats Test 1"),
            self.create_test_preset("Stats Test 2"),
        ]
        
        # Modify properties for testing
        presets[0].metadata.category = PresetCategory.PHOTO_EDITING
        presets[0].metadata.usage_count = 10
        presets[0].metadata.rating = 4.5
        
        presets[1].metadata.category = PresetCategory.ARTISTIC_CREATION
        presets[1].metadata.usage_count = 5
        presets[1].metadata.rating = 3.0
        
        for preset in presets:
            self.preset_manager.save_preset(preset)
        
        stats = self.preset_manager.get_preset_statistics()
        
        # Check basic statistics
        assert "total_presets" in stats
        assert stats["total_presets"] >= 2
        
        assert "categories" in stats
        assert "types" in stats
        
        # Check most used and highest rated
        assert "most_used" in stats
        assert "highest_rated" in stats
        
        if stats["most_used"]:
            assert "name" in stats["most_used"]
            assert "usage_count" in stats["most_used"]
        
        if stats["highest_rated"]:
            assert "name" in stats["highest_rated"]
            assert "rating" in stats["highest_rated"]
        
        # Check recent presets
        assert "recent_presets" in stats
        assert isinstance(stats["recent_presets"], list)
    
    def test_backup_presets(self):
        """Test creating preset backups"""
        # Create test presets
        presets = [
            self.create_test_preset("Backup Test 1"),
            self.create_test_preset("Backup Test 2"),
        ]
        
        for preset in presets:
            self.preset_manager.save_preset(preset)
        
        # Create backup
        backup_path = os.path.join(self.temp_dir, "backup.json")
        result = self.preset_manager.backup_presets(backup_path)
        
        assert result is True
        assert os.path.exists(backup_path)
        
        # Verify backup content
        with open(backup_path, 'r') as f:
            backup_data = json.load(f)
        
        assert "backup_info" in backup_data
        assert "presets" in backup_data
        assert backup_data["backup_info"]["preset_count"] >= 2
        assert "Backup Test 1" in backup_data["presets"]
        assert "Backup Test 2" in backup_data["presets"]
    
    def test_restore_presets(self):
        """Test restoring presets from backup"""
        # Create and backup presets
        presets = [
            self.create_test_preset("Restore Test 1"),
            self.create_test_preset("Restore Test 2"),
        ]
        
        for preset in presets:
            self.preset_manager.save_preset(preset)
        
        backup_path = os.path.join(self.temp_dir, "restore_backup.json")
        self.preset_manager.backup_presets(backup_path)
        
        # Delete presets
        self.preset_manager.delete_preset("Restore Test 1")
        self.preset_manager.delete_preset("Restore Test 2")
        
        assert "Restore Test 1" not in self.preset_manager._presets
        assert "Restore Test 2" not in self.preset_manager._presets
        
        # Restore from backup
        result = self.preset_manager.restore_presets(backup_path)
        assert result is True
        
        # Verify presets are restored
        assert "Restore Test 1" in self.preset_manager._presets
        assert "Restore Test 2" in self.preset_manager._presets
    
    def test_restore_nonexistent_backup(self):
        """Test restoring from non-existent backup"""
        result = self.preset_manager.restore_presets("nonexistent_backup.json")
        assert result is False
    
    def test_preset_data_models(self):
        """Test preset data model functionality"""
        # Test complete preset with all components
        preset = Preset(
            metadata=PresetMetadata(
                name="Complete Test Preset",
                description="A complete preset for testing all components",
                category=PresetCategory.GENERAL,
                preset_type=PresetType.WORKFLOW,
                tags=["complete", "test", "all_components"]
            ),
            generation=GenerationPreset(
                width=1024,
                height=768,
                num_inference_steps=25,
                guidance_scale=8.0,
                quality_preset="quality"
            ),
            editing=EditingPreset(
                strength=0.8,
                num_inference_steps=20,
                guidance_scale=7.5,
                enable_eligen=True
            ),
            controlnet=ControlNetPreset(
                control_type="canny",
                conditioning_scale=1.0,
                canny_low_threshold=100,
                canny_high_threshold=200
            ),
            system=SystemPreset(
                device="cuda",
                torch_dtype="float16",
                max_memory_usage_gb=8.0
            ),
            workflow=WorkflowPreset(
                workflow_type="complete_pipeline",
                auto_save=True,
                use_qwen=True,
                use_diffsynth=True,
                use_controlnet=True
            )
        )
        
        # Test serialization
        preset_dict = preset.to_dict()
        assert "metadata" in preset_dict
        assert "generation" in preset_dict
        assert "editing" in preset_dict
        assert "controlnet" in preset_dict
        assert "system" in preset_dict
        assert "workflow" in preset_dict
        
        # Test deserialization
        restored_preset = Preset.from_dict(preset_dict)
        assert restored_preset.metadata.name == preset.metadata.name
        assert restored_preset.generation.width == preset.generation.width
        assert restored_preset.editing.strength == preset.editing.strength
        assert restored_preset.controlnet.control_type == preset.controlnet.control_type
        assert restored_preset.system.device == preset.system.device
        assert restored_preset.workflow.workflow_type == preset.workflow.workflow_type
    
    def test_safe_filename_creation(self):
        """Test safe filename creation"""
        # Test normal name
        safe_name = self.preset_manager._create_safe_filename("Normal Preset Name")
        assert safe_name == "Normal_Preset_Name"
        
        # Test name with special characters
        special_name = self.preset_manager._create_safe_filename("Special!@#$%^&*()Preset")
        assert all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in special_name)
        
        # Test very long name
        long_name = "A" * 100
        safe_long = self.preset_manager._create_safe_filename(long_name)
        assert len(safe_long) <= 50
        
        # Test empty name
        empty_safe = self.preset_manager._create_safe_filename("")
        assert empty_safe == "preset"


class TestPresetIntegration:
    """Integration tests for preset system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.preset_manager = PresetManager(presets_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_full_preset_lifecycle(self):
        """Test complete preset lifecycle"""
        # Create preset
        preset = Preset(
            metadata=PresetMetadata(
                name="Lifecycle Test Preset",
                description="Testing complete preset lifecycle",
                category=PresetCategory.PHOTO_EDITING,
                preset_type=PresetType.EDITING,
                tags=["lifecycle", "test", "integration"]
            ),
            editing=EditingPreset(
                strength=0.7,
                num_inference_steps=25,
                guidance_scale=7.0
            ),
            generation=GenerationPreset(
                width=768,
                height=768,
                num_inference_steps=20
            )
        )
        
        # Save preset
        assert self.preset_manager.save_preset(preset) is True
        
        # Load and use preset
        loaded = self.preset_manager.load_preset("Lifecycle Test Preset")
        assert loaded is not None
        assert loaded.metadata.usage_count > 0
        
        # Update rating
        assert self.preset_manager.update_preset_rating("Lifecycle Test Preset", 4.0) is True
        
        # Search for preset
        search_results = self.preset_manager.search_presets("lifecycle")
        assert len(search_results) >= 1
        
        # Export preset
        export_path = os.path.join(self.temp_dir, "lifecycle_export.json")
        assert self.preset_manager.export_preset("Lifecycle Test Preset", export_path) is True
        
        # Duplicate preset
        assert self.preset_manager.duplicate_preset("Lifecycle Test Preset", "Lifecycle Copy") is True
        
        # Create backup
        backup_path = os.path.join(self.temp_dir, "lifecycle_backup.json")
        assert self.preset_manager.backup_presets(backup_path) is True
        
        # Delete original
        assert self.preset_manager.delete_preset("Lifecycle Test Preset") is True
        
        # Restore from backup
        assert self.preset_manager.restore_presets(backup_path) is True
        
        # Verify restoration
        restored = self.preset_manager.load_preset("Lifecycle Test Preset")
        assert restored is not None
        assert restored.metadata.name == "Lifecycle Test Preset"
    
    def test_concurrent_preset_operations(self):
        """Test handling of concurrent preset operations"""
        # Create multiple presets
        presets = []
        for i in range(10):
            preset = Preset(
                metadata=PresetMetadata(
                    name=f"Concurrent Test {i}",
                    description=f"Concurrent test preset {i}",
                    category=PresetCategory.GENERAL,
                    preset_type=PresetType.GENERATION,
                    tags=[f"concurrent_{i}", "test"]
                ),
                generation=GenerationPreset(width=512 + i * 64, height=512 + i * 64)
            )
            presets.append(preset)
        
        # Save all presets
        for preset in presets:
            assert self.preset_manager.save_preset(preset) is True
        
        # Perform concurrent operations
        for i in range(10):
            # Load preset
            loaded = self.preset_manager.load_preset(f"Concurrent Test {i}")
            assert loaded is not None
            
            # Update rating
            self.preset_manager.update_preset_rating(f"Concurrent Test {i}", float(i % 5))
            
            # Search
            results = self.preset_manager.search_presets(f"concurrent_{i}")
            assert len(results) >= 1
        
        # Verify all presets still exist and are valid
        all_presets = self.preset_manager.list_presets()
        concurrent_presets = [p for p in all_presets if "Concurrent Test" in p.metadata.name]
        assert len(concurrent_presets) == 10
        
        # Verify statistics
        stats = self.preset_manager.get_preset_statistics()
        assert stats["total_presets"] >= 10


if __name__ == "__main__":
    pytest.main([__file__])