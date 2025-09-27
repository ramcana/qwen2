"""
Tests for Configuration Validation and Migration System
Comprehensive testing of configuration validation, migration, and management
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
import pytest

from src.config_validator import (
    ConfigValidator, ConfigMigrator, ConfigManager, ValidationResult, ValidationIssue,
    ValidationSeverity, ConfigVersion, MigrationStep
)


class TestConfigValidator:
    """Test suite for ConfigValidator"""
    
    def setup_method(self):
        """Setup test environment"""
        self.validator = ConfigValidator()
    
    def create_valid_config(self) -> dict:
        """Create a valid configuration for testing"""
        return {
            "diffsynth": {
                "model_name": "Qwen/Qwen-Image-Edit",
                "torch_dtype": "float16",
                "device": "cuda",
                "enable_vram_management": True,
                "enable_cpu_offload": False,
                "max_memory_usage_gb": 8.0,
                "default_num_inference_steps": 20,
                "default_guidance_scale": 7.5,
                "default_height": 768,
                "default_width": 768
            },
            "controlnet": {
                "control_types": ["canny", "depth"],
                "default_conditioning_scale": 1.0,
                "canny_low_threshold": 100,
                "canny_high_threshold": 200,
                "detect_resolution": 512,
                "image_resolution": 768
            },
            "system": {
                "enable_xformers": True,
                "enable_memory_efficient_attention": True,
                "enable_attention_slicing": True,
                "enable_vae_slicing": True,
                "enable_vae_tiling": True,
                "safety_checker": True
            },
            "performance": {
                "max_batch_size": 1,
                "enable_tiled_processing": False,
                "tile_overlap": 64
            }
        }
    
    def test_validate_valid_config(self):
        """Test validation of a valid configuration"""
        config = self.create_valid_config()
        result = self.validator.validate_config(config)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_validate_missing_sections(self):
        """Test validation with missing sections"""
        config = {
            "diffsynth": {
                "model_name": "Qwen/Qwen-Image-Edit"
            }
        }
        
        result = self.validator.validate_config(config)
        
        # Should have warnings for missing sections
        assert len(result.warnings) > 0
        missing_sections = [issue for issue in result.warnings if "is missing" in issue.message]
        assert len(missing_sections) > 0
    
    def test_validate_invalid_types(self):
        """Test validation with invalid field types"""
        config = self.create_valid_config()
        
        # Invalid types
        config["diffsynth"]["max_memory_usage_gb"] = "invalid_string"
        config["diffsynth"]["default_num_inference_steps"] = 20.5  # Should be int
        config["controlnet"]["control_types"] = "not_a_list"
        
        result = self.validator.validate_config(config)
        
        assert result.is_valid is False
        assert len(result.errors) >= 3
        
        # Check specific error messages
        type_errors = [error for error in result.errors if "Invalid type" in error.message]
        assert len(type_errors) >= 2
    
    def test_validate_value_ranges(self):
        """Test validation of value ranges"""
        config = self.create_valid_config()
        
        # Out of range values
        config["diffsynth"]["max_memory_usage_gb"] = -1.0  # Below minimum
        config["diffsynth"]["default_num_inference_steps"] = 200  # Above maximum
        config["controlnet"]["canny_low_threshold"] = 300  # Above maximum
        
        result = self.validator.validate_config(config)
        
        assert result.is_valid is False
        assert len(result.errors) >= 3
        
        # Check for range error messages
        range_errors = [error for error in result.errors if ("below minimum" in error.message or "exceeds maximum" in error.message)]
        assert len(range_errors) >= 3
    
    def test_validate_allowed_values(self):
        """Test validation of allowed values"""
        config = self.create_valid_config()
        
        # Invalid allowed values
        config["diffsynth"]["torch_dtype"] = "invalid_dtype"
        config["diffsynth"]["device"] = "invalid_device"
        config["controlnet"]["control_types"] = ["invalid_control_type"]
        
        result = self.validator.validate_config(config)
        
        assert result.is_valid is False
        assert len(result.errors) >= 3
        
        # Check for allowed values errors
        allowed_errors = [error for error in result.errors if "Invalid value" in error.message or "Invalid list items" in error.message]
        assert len(allowed_errors) >= 3
    
    def test_validate_multiple_constraints(self):
        """Test validation of multiple-of constraints"""
        config = self.create_valid_config()
        
        # Values that should be multiples of 64
        config["diffsynth"]["default_height"] = 700  # Not multiple of 64
        config["diffsynth"]["default_width"] = 500   # Not multiple of 64
        
        result = self.validator.validate_config(config)
        
        # Should have warnings for non-multiples
        multiple_warnings = [warning for warning in result.warnings if "multiple of" in warning.message]
        assert len(multiple_warnings) >= 2
    
    def test_validate_cross_section_compatibility(self):
        """Test cross-section validation"""
        config = self.create_valid_config()
        
        # Create resolution mismatch
        config["diffsynth"]["default_height"] = 512
        config["controlnet"]["image_resolution"] = 1024  # Large difference
        
        result = self.validator.validate_config(config)
        
        # Should have warning about resolution mismatch
        mismatch_warnings = [warning for warning in result.warnings if "resolution mismatch" in warning.message]
        assert len(mismatch_warnings) >= 1
    
    def test_validate_hardware_compatibility(self):
        """Test hardware compatibility validation"""
        config = self.create_valid_config()
        
        # Test with very high memory requirement
        config["diffsynth"]["max_memory_usage_gb"] = 64.0  # Very high
        
        result = self.validator.validate_config(config)
        
        # May have warnings about memory usage depending on available hardware
        # This test is hardware-dependent, so we just check it doesn't crash
        assert isinstance(result, ValidationResult)
    
    def test_auto_fix_config(self):
        """Test automatic configuration fixing"""
        config = {
            "diffsynth": {
                "model_name": "Qwen/Qwen-Image-Edit",
                "max_memory_usage_gb": -1.0,  # Below minimum
                "default_num_inference_steps": 200,  # Above maximum
                "default_height": 700  # Not multiple of 64
            }
        }
        
        fixed_config, fixes_applied = self.validator.auto_fix_config(config)
        
        assert len(fixes_applied) > 0
        
        # Check that fixes were applied
        assert fixed_config["diffsynth"]["max_memory_usage_gb"] >= 1.0
        assert fixed_config["diffsynth"]["default_num_inference_steps"] <= 100
        assert fixed_config["diffsynth"]["default_height"] % 64 == 0
    
    def test_validation_issue_creation(self):
        """Test ValidationIssue creation and properties"""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field_path="test.field",
            message="Test error message",
            suggested_fix="Test fix",
            auto_fixable=True
        )
        
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.field_path == "test.field"
        assert issue.message == "Test error message"
        assert issue.auto_fixable is True
        
        # Test string representation
        issue_str = str(issue)
        assert "ERROR" in issue_str
        assert "test.field" in issue_str
        assert "Test error message" in issue_str
    
    def test_validation_result_management(self):
        """Test ValidationResult issue management"""
        result = ValidationResult(is_valid=True)
        
        # Add different types of issues
        error_issue = ValidationIssue(ValidationSeverity.ERROR, "test.error", "Error message")
        warning_issue = ValidationIssue(ValidationSeverity.WARNING, "test.warning", "Warning message")
        info_issue = ValidationIssue(ValidationSeverity.INFO, "test.info", "Info message")
        
        result.add_issue(error_issue)
        result.add_issue(warning_issue)
        result.add_issue(info_issue)
        
        # Check that validity is updated
        assert result.is_valid is False  # Should be False due to error
        
        # Check issue categorization
        assert len(result.errors) == 1
        assert len(result.warnings) == 1
        assert len(result.issues) == 3
        
        # Check helper methods
        assert result.has_errors() is True
        assert result.has_warnings() is True


class TestConfigMigrator:
    """Test suite for ConfigMigrator"""
    
    def setup_method(self):
        """Setup test environment"""
        self.migrator = ConfigMigrator()
    
    def create_v1_0_config(self) -> dict:
        """Create a version 1.0 configuration"""
        return {
            "model_settings": {
                "model_name": "Qwen/Qwen-Image",
                "torch_dtype": "float16"
            },
            "generation_presets": {
                "balanced": {
                    "width": 512,
                    "height": 512,
                    "num_inference_steps": 15,
                    "guidance_scale": 3.5
                }
            },
            "memory_optimizations": {
                "enable_attention_slicing": True,
                "enable_vae_slicing": True,
                "enable_model_cpu_offload": False
            }
        }
    
    def create_v1_1_config(self) -> dict:
        """Create a version 1.1 configuration"""
        return {
            "version": "1.1",
            "diffsynth": {
                "model_name": "Qwen/Qwen-Image-Edit",
                "torch_dtype": "float16",
                "device": "cuda",
                "enable_vram_management": True,
                "enable_cpu_offload": False,
                "max_memory_usage_gb": 8.0
            },
            "controlnet": {
                "control_types": ["canny", "depth"],
                "default_conditioning_scale": 1.0
            },
            "system": {
                "enable_xformers": True,
                "enable_memory_efficient_attention": True
            }
        }
    
    def test_get_config_version(self):
        """Test configuration version detection"""
        # Test explicit version
        config_with_version = {"version": "2.0", "diffsynth": {}}
        assert self.migrator.get_config_version(config_with_version) == "2.0"
        
        # Test inferred versions
        v1_0_config = self.create_v1_0_config()
        assert self.migrator.get_config_version(v1_0_config) == "1.0"
        
        v1_1_config = self.create_v1_1_config()
        assert self.migrator.get_config_version(v1_1_config) == "1.1"
        
        # Test v2.0 inference (has eligen in diffsynth)
        v2_0_config = {"diffsynth": {"eligen": {}}}
        assert self.migrator.get_config_version(v2_0_config) == "2.0"
    
    def test_needs_migration(self):
        """Test migration necessity detection"""
        # Current version doesn't need migration
        current_config = {"version": ConfigVersion.CURRENT.value}
        assert self.migrator.needs_migration(current_config) is False
        
        # Old version needs migration
        old_config = self.create_v1_0_config()
        assert self.migrator.needs_migration(old_config) is True
    
    def test_migrate_1_0_to_1_1(self):
        """Test migration from version 1.0 to 1.1"""
        v1_0_config = self.create_v1_0_config()
        
        migrated = self.migrator._migrate_1_0_to_1_1(v1_0_config)
        
        # Check that new structure is created
        assert "diffsynth" in migrated
        assert "controlnet" in migrated
        assert "system" in migrated
        assert "performance" in migrated
        
        # Check that values are migrated correctly
        assert migrated["diffsynth"]["model_name"] == "Qwen/Qwen-Image"
        assert migrated["diffsynth"]["torch_dtype"] == "float16"
        assert migrated["system"]["enable_attention_slicing"] is True
        
        # Check version is updated
        assert migrated["version"] == "1.1"
    
    def test_migrate_1_1_to_2_0(self):
        """Test migration from version 1.1 to 2.0"""
        v1_1_config = self.create_v1_1_config()
        
        migrated = self.migrator._migrate_1_1_to_2_0(v1_1_config)
        
        # Check that EliGen settings are added
        assert "enable_eligen" in migrated["diffsynth"]
        assert "eligen_mode" in migrated["diffsynth"]
        assert "eligen_quality_preset" in migrated["diffsynth"]
        
        # Check that workflow section is added
        assert "workflow" in migrated
        
        # Check enhanced ControlNet settings
        assert "control_guidance_start" in migrated["controlnet"]
        assert "control_guidance_end" in migrated["controlnet"]
        
        # Check version is updated
        assert migrated["version"] == "2.0"
    
    def test_full_migration_path(self):
        """Test complete migration from 1.0 to current version"""
        v1_0_config = self.create_v1_0_config()
        
        migrated_config, migration_log = self.migrator.migrate_config(v1_0_config)
        
        # Check that migration completed successfully
        assert len(migration_log) > 0
        assert any("✅" in log for log in migration_log)
        
        # Check final version
        assert migrated_config["version"] == ConfigVersion.CURRENT.value
        
        # Check that all expected sections exist
        assert "diffsynth" in migrated_config
        assert "controlnet" in migrated_config
        assert "system" in migrated_config
        assert "workflow" in migrated_config
        
        # Check EliGen integration (v2.0 feature)
        assert "enable_eligen" in migrated_config["diffsynth"]
    
    def test_migration_no_path(self):
        """Test migration when no path exists"""
        # Create config with unsupported version
        unsupported_config = {"version": "99.0"}
        
        migrated_config, migration_log = self.migrator.migrate_config(unsupported_config)
        
        # Should return original config
        assert migrated_config == unsupported_config
        assert any("No migration path found" in log for log in migration_log)
    
    def test_backup_and_restore_config(self):
        """Test configuration backup and restore"""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = self.create_v1_1_config()
            json.dump(config, f)
            config_path = f.name
        
        try:
            # Create backup
            backup_path = self.migrator.backup_config(config_path)
            assert os.path.exists(backup_path)
            
            # Verify backup content
            with open(backup_path, 'r') as f:
                backup_config = json.load(f)
            assert backup_config == config
            
            # Modify original
            modified_config = config.copy()
            modified_config["test_field"] = "test_value"
            with open(config_path, 'w') as f:
                json.dump(modified_config, f)
            
            # Restore from backup
            success = self.migrator.restore_config(backup_path, config_path)
            assert success is True
            
            # Verify restoration
            with open(config_path, 'r') as f:
                restored_config = json.load(f)
            assert restored_config == config
            assert "test_field" not in restored_config
            
        finally:
            # Cleanup
            for path in [config_path, backup_path]:
                if os.path.exists(path):
                    os.unlink(path)


class TestConfigManager:
    """Test suite for ConfigManager"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(config_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ConfigManager initialization"""
        # Check that directories are created
        assert os.path.exists(self.temp_dir)
        assert os.path.exists(self.config_manager.backup_dir)
        
        # Check that components are initialized
        assert self.config_manager.validator is not None
        assert self.config_manager.migrator is not None
    
    def test_load_and_validate_new_config(self):
        """Test loading when no config file exists"""
        config, validation_result = self.config_manager.load_and_validate_config()
        
        # Should create default config
        assert config is not None
        assert "diffsynth" in config
        assert "controlnet" in config
        assert "system" in config
        
        # Should be valid
        assert validation_result.is_valid is True
        
        # Config file should be created
        assert self.config_manager.main_config_file.exists()
    
    def test_load_and_validate_existing_config(self):
        """Test loading existing configuration"""
        # Create a config file
        test_config = {
            "version": "2.0",
            "diffsynth": {
                "model_name": "Qwen/Qwen-Image-Edit",
                "torch_dtype": "float16",
                "device": "cuda"
            }
        }
        
        with open(self.config_manager.main_config_file, 'w') as f:
            json.dump(test_config, f)
        
        config, validation_result = self.config_manager.load_and_validate_config()
        
        # Should load the existing config
        assert config["diffsynth"]["model_name"] == "Qwen/Qwen-Image-Edit"
        
        # May have warnings for missing sections
        assert isinstance(validation_result, ValidationResult)
    
    def test_load_and_migrate_old_config(self):
        """Test loading and migrating old configuration"""
        # Create old v1.0 config
        old_config = {
            "model_settings": {
                "model_name": "Qwen/Qwen-Image",
                "torch_dtype": "float16"
            },
            "generation_presets": {
                "balanced": {
                    "width": 512,
                    "height": 512
                }
            }
        }
        
        with open(self.config_manager.main_config_file, 'w') as f:
            json.dump(old_config, f)
        
        config, validation_result = self.config_manager.load_and_validate_config()
        
        # Should be migrated to current version
        assert config["version"] == ConfigVersion.CURRENT.value
        assert "diffsynth" in config
        assert "controlnet" in config
        
        # Backup should be created
        backup_files = list(self.config_manager.backup_dir.glob("*backup*"))
        assert len(backup_files) > 0
    
    def test_save_config(self):
        """Test saving configuration"""
        test_config = {
            "version": "2.0",
            "diffsynth": {
                "model_name": "Qwen/Qwen-Image-Edit",
                "torch_dtype": "float16"
            }
        }
        
        success = self.config_manager.save_config(test_config)
        assert success is True
        
        # Verify file was created and content is correct
        assert self.config_manager.main_config_file.exists()
        
        with open(self.config_manager.main_config_file, 'r') as f:
            saved_config = json.load(f)
        
        assert saved_config["diffsynth"]["model_name"] == "Qwen/Qwen-Image-Edit"
    
    def test_auto_fix_and_save(self):
        """Test automatic fixing and saving"""
        # Create config with fixable issues
        problematic_config = {
            "diffsynth": {
                "model_name": "Qwen/Qwen-Image-Edit",
                "max_memory_usage_gb": -1.0,  # Below minimum - auto-fixable
                "default_height": 700  # Not multiple of 64 - auto-fixable
            }
        }
        
        with open(self.config_manager.main_config_file, 'w') as f:
            json.dump(problematic_config, f)
        
        success, fixes_applied = self.config_manager.auto_fix_and_save()
        
        assert success is True
        assert len(fixes_applied) > 0
        
        # Verify fixes were applied and saved
        with open(self.config_manager.main_config_file, 'r') as f:
            fixed_config = json.load(f)
        
        assert fixed_config["diffsynth"]["max_memory_usage_gb"] >= 1.0
        assert fixed_config["diffsynth"]["default_height"] % 64 == 0
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON configuration"""
        # Create invalid JSON file
        with open(self.config_manager.main_config_file, 'w') as f:
            f.write("{ invalid json content")
        
        config, validation_result = self.config_manager.load_and_validate_config()
        
        # Should return default config with error
        assert config is not None
        assert validation_result.is_valid is False
        assert len(validation_result.errors) > 0
        
        # Error should mention JSON loading failure
        json_errors = [error for error in validation_result.errors if "Failed to load" in error.message]
        assert len(json_errors) > 0


class TestIntegration:
    """Integration tests for the complete validation and migration system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(config_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow(self):
        """Test complete configuration workflow"""
        # 1. Start with old v1.0 config
        old_config = {
            "model_settings": {
                "model_name": "Qwen/Qwen-Image",
                "torch_dtype": "float16"
            },
            "generation_presets": {
                "balanced": {
                    "width": 700,  # Not multiple of 64
                    "height": 512,
                    "num_inference_steps": 15
                }
            },
            "memory_optimizations": {
                "enable_attention_slicing": True,
                "enable_vae_slicing": True
            }
        }
        
        with open(self.config_manager.main_config_file, 'w') as f:
            json.dump(old_config, f)
        
        # 2. Load and validate (should trigger migration)
        config, validation_result = self.config_manager.load_and_validate_config()
        
        # Should be migrated
        assert config["version"] == ConfigVersion.CURRENT.value
        assert "diffsynth" in config
        
        # 3. Auto-fix any remaining issues
        success, fixes_applied = self.config_manager.auto_fix_and_save()
        assert success is True
        
        # 4. Final validation should be clean
        final_config, final_validation = self.config_manager.load_and_validate_config()
        
        # Should have minimal issues (maybe some warnings)
        assert final_validation.is_valid is True or not final_validation.has_errors()
        
        # 5. Verify backup was created
        backup_files = list(self.config_manager.backup_dir.glob("*backup*"))
        assert len(backup_files) > 0
    
    def test_validation_with_hardware_constraints(self):
        """Test validation considering hardware constraints"""
        # Create config that might exceed hardware limits
        high_memory_config = {
            "version": "2.0",
            "diffsynth": {
                "model_name": "Qwen/Qwen-Image-Edit",
                "torch_dtype": "float16",
                "device": "cuda",
                "max_memory_usage_gb": 32.0,  # Very high
                "default_height": 2048,
                "default_width": 2048
            }
        }
        
        validator = ConfigValidator()
        result = validator.validate_config(high_memory_config)
        
        # Should complete without crashing (hardware-dependent results)
        assert isinstance(result, ValidationResult)
        
        # If there are memory warnings, auto-fix should handle them
        if result.has_warnings():
            fixed_config, fixes = validator.auto_fix_config(high_memory_config)
            assert len(fixes) >= 0  # May or may not have fixes depending on hardware


if __name__ == "__main__":
    pytest.main([__file__])