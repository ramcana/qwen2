"""
Configuration Validation and Migration System
Provides validation for DiffSynth and ControlNet settings with automatic migration
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConfigVersion(Enum):
    """Configuration version enumeration"""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"
    CURRENT = "2.0"


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a configuration validation issue"""
    severity: ValidationSeverity
    field_path: str
    message: str
    suggested_fix: Optional[str] = None
    auto_fixable: bool = False
    
    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.field_path}: {self.message}"


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    errors: List[ValidationIssue] = field(default_factory=list)
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue"""
        self.issues.append(issue)
        
        if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.errors.append(issue)
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings.append(issue)
    
    def has_errors(self) -> bool:
        """Check if there are any errors"""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings"""
        return len(self.warnings) > 0


@dataclass
class MigrationStep:
    """Represents a single migration step"""
    from_version: str
    to_version: str
    description: str
    migration_function: callable
    backup_required: bool = True


class ConfigValidator:
    """
    Configuration validator for DiffSynth and ControlNet settings
    Provides comprehensive validation with detailed error reporting
    """
    
    def __init__(self):
        """Initialize configuration validator"""
        self.validation_rules = self._setup_validation_rules()
        self.hardware_limits = self._get_hardware_limits()
    
    def _setup_validation_rules(self) -> Dict[str, Any]:
        """Setup validation rules for different configuration sections"""
        return {
            "diffsynth": {
                "model_name": {
                    "type": str,
                    "required": True,
                    "allowed_values": [
                        "Qwen/Qwen-Image-Edit",
                        "Qwen/Qwen-Image",
                        "custom"
                    ]
                },
                "torch_dtype": {
                    "type": str,
                    "required": True,
                    "allowed_values": ["float16", "bfloat16", "float32"]
                },
                "device": {
                    "type": str,
                    "required": True,
                    "allowed_values": ["cuda", "cpu", "auto"]
                },
                "enable_vram_management": {
                    "type": bool,
                    "default": True
                },
                "enable_cpu_offload": {
                    "type": bool,
                    "default": False
                },
                "max_memory_usage_gb": {
                    "type": [int, float],
                    "min_value": 1.0,
                    "max_value": 64.0,
                    "default": 8.0
                },
                "default_num_inference_steps": {
                    "type": int,
                    "min_value": 1,
                    "max_value": 100,
                    "default": 20
                },
                "default_guidance_scale": {
                    "type": [int, float],
                    "min_value": 1.0,
                    "max_value": 20.0,
                    "default": 7.5
                },
                "default_height": {
                    "type": int,
                    "min_value": 256,
                    "max_value": 2048,
                    "multiple_of": 64,
                    "default": 768
                },
                "default_width": {
                    "type": int,
                    "min_value": 256,
                    "max_value": 2048,
                    "multiple_of": 64,
                    "default": 768
                }
            },
            "controlnet": {
                "control_types": {
                    "type": list,
                    "allowed_items": [
                        "canny", "depth", "pose", "normal", "segmentation",
                        "lineart", "mlsd", "scribble", "softedge"
                    ],
                    "default": ["canny", "depth"]
                },
                "default_conditioning_scale": {
                    "type": [int, float],
                    "min_value": 0.0,
                    "max_value": 2.0,
                    "default": 1.0
                },
                "canny_low_threshold": {
                    "type": int,
                    "min_value": 1,
                    "max_value": 255,
                    "default": 100
                },
                "canny_high_threshold": {
                    "type": int,
                    "min_value": 1,
                    "max_value": 255,
                    "default": 200
                },
                "detect_resolution": {
                    "type": int,
                    "min_value": 256,
                    "max_value": 1024,
                    "multiple_of": 64,
                    "default": 512
                },
                "image_resolution": {
                    "type": int,
                    "min_value": 256,
                    "max_value": 2048,
                    "multiple_of": 64,
                    "default": 768
                }
            },
            "system": {
                "enable_xformers": {
                    "type": bool,
                    "default": True
                },
                "enable_memory_efficient_attention": {
                    "type": bool,
                    "default": True
                },
                "enable_attention_slicing": {
                    "type": bool,
                    "default": True
                },
                "enable_vae_slicing": {
                    "type": bool,
                    "default": True
                },
                "enable_vae_tiling": {
                    "type": bool,
                    "default": True
                },
                "safety_checker": {
                    "type": bool,
                    "default": True
                }
            },
            "performance": {
                "max_batch_size": {
                    "type": int,
                    "min_value": 1,
                    "max_value": 8,
                    "default": 1
                },
                "enable_tiled_processing": {
                    "type": bool,
                    "default": False
                },
                "tile_overlap": {
                    "type": int,
                    "min_value": 32,
                    "max_value": 256,
                    "default": 64
                }
            }
        }
    
    def _get_hardware_limits(self) -> Dict[str, Any]:
        """Get hardware-specific limits for validation"""
        try:
            import torch
            import psutil
            
            limits = {
                "has_cuda": torch.cuda.is_available(),
                "gpu_memory_gb": 0.0,
                "cpu_memory_gb": psutil.virtual_memory().total / 1e9,
                "cpu_count": psutil.cpu_count()
            }
            
            if limits["has_cuda"]:
                limits["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
                limits["gpu_count"] = torch.cuda.device_count()
            
            return limits
            
        except Exception as e:
            logger.warning(f"Failed to get hardware limits: {e}")
            return {
                "has_cuda": False,
                "gpu_memory_gb": 0.0,
                "cpu_memory_gb": 8.0,
                "cpu_count": 4
            }
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate a complete configuration
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            ValidationResult with issues and suggestions
        """
        result = ValidationResult(is_valid=True)
        
        # Validate each section
        for section_name, section_rules in self.validation_rules.items():
            if section_name in config:
                section_result = self._validate_section(
                    config[section_name], 
                    section_rules, 
                    section_name
                )
                
                for issue in section_result.issues:
                    result.add_issue(issue)
            else:
                # Missing section - add warning
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field_path=section_name,
                    message=f"Configuration section '{section_name}' is missing",
                    suggested_fix=f"Add default {section_name} configuration",
                    auto_fixable=True
                ))
        
        # Perform cross-section validation
        self._validate_cross_section(config, result)
        
        # Perform hardware compatibility validation
        self._validate_hardware_compatibility(config, result)
        
        return result
    
    def _validate_section(
        self, 
        section_config: Dict[str, Any], 
        section_rules: Dict[str, Any], 
        section_name: str
    ) -> ValidationResult:
        """Validate a configuration section"""
        result = ValidationResult(is_valid=True)
        
        for field_name, field_rules in section_rules.items():
            field_path = f"{section_name}.{field_name}"
            
            if field_name in section_config:
                # Validate existing field
                field_value = section_config[field_name]
                field_issues = self._validate_field(field_value, field_rules, field_path)
                
                for issue in field_issues:
                    result.add_issue(issue)
            else:
                # Missing field
                if field_rules.get("required", False):
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        field_path=field_path,
                        message=f"Required field '{field_name}' is missing",
                        suggested_fix=f"Add {field_name} with default value: {field_rules.get('default', 'N/A')}",
                        auto_fixable="default" in field_rules
                    ))
                elif "default" in field_rules:
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        field_path=field_path,
                        message=f"Optional field '{field_name}' is missing, will use default",
                        suggested_fix=f"Add {field_name}: {field_rules['default']}",
                        auto_fixable=True
                    ))
        
        return result
    
    def _validate_field(
        self, 
        value: Any, 
        rules: Dict[str, Any], 
        field_path: str
    ) -> List[ValidationIssue]:
        """Validate a single field"""
        issues = []
        
        # Type validation
        expected_types = rules.get("type")
        if expected_types:
            if not isinstance(expected_types, list):
                expected_types = [expected_types]
            
            if not any(isinstance(value, t) for t in expected_types):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path=field_path,
                    message=f"Invalid type. Expected {expected_types}, got {type(value).__name__}",
                    suggested_fix=f"Convert value to one of: {[t.__name__ for t in expected_types]}"
                ))
                return issues  # Skip further validation if type is wrong
        
        # Value range validation
        if "min_value" in rules and value < rules["min_value"]:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field_path=field_path,
                message=f"Value {value} is below minimum {rules['min_value']}",
                suggested_fix=f"Set value to at least {rules['min_value']}",
                auto_fixable=True
            ))
        
        if "max_value" in rules and value > rules["max_value"]:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field_path=field_path,
                message=f"Value {value} exceeds maximum {rules['max_value']}",
                suggested_fix=f"Set value to at most {rules['max_value']}",
                auto_fixable=True
            ))
        
        # Multiple validation
        if "multiple_of" in rules and isinstance(value, int):
            if value % rules["multiple_of"] != 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field_path=field_path,
                    message=f"Value {value} should be multiple of {rules['multiple_of']}",
                    suggested_fix=f"Use nearest multiple: {(value // rules['multiple_of']) * rules['multiple_of']}",
                    auto_fixable=True
                ))
        
        # Allowed values validation
        if "allowed_values" in rules and value not in rules["allowed_values"]:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field_path=field_path,
                message=f"Invalid value '{value}'. Allowed values: {rules['allowed_values']}",
                suggested_fix=f"Use one of: {rules['allowed_values']}"
            ))
        
        # List item validation
        if "allowed_items" in rules and isinstance(value, list):
            invalid_items = [item for item in value if item not in rules["allowed_items"]]
            if invalid_items:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path=field_path,
                    message=f"Invalid list items: {invalid_items}. Allowed: {rules['allowed_items']}",
                    suggested_fix=f"Remove invalid items or use allowed values: {rules['allowed_items']}"
                ))
        
        return issues
    
    def _validate_cross_section(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate relationships between configuration sections"""
        
        # Validate DiffSynth and ControlNet compatibility
        if "diffsynth" in config and "controlnet" in config:
            diffsynth_config = config["diffsynth"]
            controlnet_config = config["controlnet"]
            
            # Check resolution compatibility
            if "default_height" in diffsynth_config and "image_resolution" in controlnet_config:
                diff_height = diffsynth_config["default_height"]
                cn_resolution = controlnet_config["image_resolution"]
                
                if abs(diff_height - cn_resolution) > 128:
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        field_path="diffsynth.default_height vs controlnet.image_resolution",
                        message=f"Large resolution mismatch: DiffSynth {diff_height} vs ControlNet {cn_resolution}",
                        suggested_fix="Use similar resolutions for better compatibility"
                    ))
        
        # Validate memory settings consistency
        if "diffsynth" in config and "system" in config:
            diffsynth_config = config["diffsynth"]
            system_config = config["system"]
            
            max_memory = diffsynth_config.get("max_memory_usage_gb", 8.0)
            cpu_offload = diffsynth_config.get("enable_cpu_offload", False)
            
            # Ensure max_memory is numeric before comparison
            if isinstance(max_memory, (int, float)) and max_memory > 12.0 and not cpu_offload:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field_path="diffsynth.max_memory_usage_gb",
                    message=f"High memory usage ({max_memory}GB) without CPU offload may cause issues",
                    suggested_fix="Enable CPU offload for high memory configurations"
                ))
    
    def _validate_hardware_compatibility(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate configuration against available hardware"""
        
        # GPU availability check
        if "diffsynth" in config:
            device = config["diffsynth"].get("device", "cuda")
            
            if device == "cuda" and not self.hardware_limits["has_cuda"]:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path="diffsynth.device",
                    message="CUDA device specified but CUDA is not available",
                    suggested_fix="Change device to 'cpu' or install CUDA",
                    auto_fixable=True
                ))
        
        # Memory limits check
        if "diffsynth" in config and self.hardware_limits["has_cuda"]:
            max_memory = config["diffsynth"].get("max_memory_usage_gb", 8.0)
            available_memory = self.hardware_limits["gpu_memory_gb"]
            
            # Ensure max_memory is numeric before comparison
            if isinstance(max_memory, (int, float)) and max_memory > available_memory * 0.9:  # Leave 10% headroom
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field_path="diffsynth.max_memory_usage_gb",
                    message=f"Configured memory ({max_memory}GB) exceeds available GPU memory ({available_memory:.1f}GB)",
                    suggested_fix=f"Reduce to {available_memory * 0.8:.1f}GB or enable CPU offload",
                    auto_fixable=True
                ))
        
        # Resolution vs memory check
        if "diffsynth" in config:
            height = config["diffsynth"].get("default_height", 768)
            width = config["diffsynth"].get("default_width", 768)
            max_memory = config["diffsynth"].get("max_memory_usage_gb", 8.0)
            
            # Ensure all values are numeric before calculation
            if all(isinstance(val, (int, float)) for val in [height, width, max_memory]):
                # Rough memory estimation (very approximate)
                estimated_memory = (height * width * 4) / (1024**3) * 10  # Rough estimate
                
                if estimated_memory > max_memory:
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        field_path="diffsynth resolution vs memory",
                        message=f"High resolution ({width}x{height}) may exceed memory limit ({max_memory}GB)",
                        suggested_fix="Reduce resolution or increase memory limit"
                    ))
    
    def auto_fix_config(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Automatically fix configuration issues where possible
        
        Args:
            config: Configuration to fix
            
        Returns:
            Tuple of (fixed_config, list_of_fixes_applied)
        """
        fixed_config = config.copy()
        fixes_applied = []
        
        # Validate to get fixable issues
        validation_result = self.validate_config(config)
        
        for issue in validation_result.issues:
            if issue.auto_fixable:
                fix_applied = self._apply_auto_fix(fixed_config, issue)
                if fix_applied:
                    fixes_applied.append(f"{issue.field_path}: {fix_applied}")
        
        return fixed_config, fixes_applied
    
    def _apply_auto_fix(self, config: Dict[str, Any], issue: ValidationIssue) -> Optional[str]:
        """Apply an automatic fix to the configuration"""
        try:
            field_path_parts = issue.field_path.split(".")
            
            if len(field_path_parts) == 2:
                section, field = field_path_parts
                
                # Get the validation rules for this field
                section_rules = self.validation_rules.get(section, {})
                field_rules = section_rules.get(field, {})
                
                # Ensure section exists
                if section not in config:
                    config[section] = {}
                
                # Apply specific fixes based on issue type
                if "is missing" in issue.message and "default" in field_rules:
                    config[section][field] = field_rules["default"]
                    return f"Set to default value: {field_rules['default']}"
                
                elif "below minimum" in issue.message and "min_value" in field_rules:
                    config[section][field] = field_rules["min_value"]
                    return f"Set to minimum value: {field_rules['min_value']}"
                
                elif "exceeds maximum" in issue.message and "max_value" in field_rules:
                    config[section][field] = field_rules["max_value"]
                    return f"Set to maximum value: {field_rules['max_value']}"
                
                elif "multiple of" in issue.message and "multiple_of" in field_rules:
                    current_value = config[section][field]
                    multiple = field_rules["multiple_of"]
                    fixed_value = (current_value // multiple) * multiple
                    config[section][field] = fixed_value
                    return f"Rounded to multiple of {multiple}: {fixed_value}"
                
                elif "CUDA device specified but CUDA is not available" in issue.message:
                    config[section][field] = "cpu"
                    return "Changed device to CPU"
                
                elif "exceeds available GPU memory" in issue.message:
                    available_memory = self.hardware_limits["gpu_memory_gb"]
                    safe_memory = available_memory * 0.8
                    config[section][field] = safe_memory
                    return f"Reduced to safe memory limit: {safe_memory:.1f}GB"
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to apply auto-fix for {issue.field_path}: {e}")
            return None


class ConfigMigrator:
    """
    Configuration migration system for handling version upgrades
    Provides automatic migration between configuration versions
    """
    
    def __init__(self):
        """Initialize configuration migrator"""
        self.migration_steps = self._setup_migration_steps()
        self.validator = ConfigValidator()
    
    def _setup_migration_steps(self) -> List[MigrationStep]:
        """Setup migration steps between versions"""
        return [
            MigrationStep(
                from_version="1.0",
                to_version="1.1",
                description="Add EliGen integration settings",
                migration_function=self._migrate_1_0_to_1_1
            ),
            MigrationStep(
                from_version="1.1",
                to_version="2.0",
                description="Restructure configuration for DiffSynth Enhanced UI",
                migration_function=self._migrate_1_1_to_2_0
            )
        ]
    
    def get_config_version(self, config: Dict[str, Any]) -> str:
        """
        Determine the version of a configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Version string
        """
        # Check for explicit version
        if "version" in config:
            return config["version"]
        
        # Infer version from structure
        if "diffsynth" in config and "eligen" in config.get("diffsynth", {}):
            return "2.0"
        elif "diffsynth" in config:
            return "1.1"
        elif "model_settings" in config and "generation_presets" in config:
            return "1.0"
        else:
            return "1.0"  # Default to oldest version
    
    def needs_migration(self, config: Dict[str, Any]) -> bool:
        """
        Check if configuration needs migration
        
        Args:
            config: Configuration to check
            
        Returns:
            True if migration is needed
        """
        current_version = self.get_config_version(config)
        return current_version != ConfigVersion.CURRENT.value
    
    def migrate_config(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Migrate configuration to current version
        
        Args:
            config: Configuration to migrate
            
        Returns:
            Tuple of (migrated_config, migration_log)
        """
        current_version = self.get_config_version(config)
        target_version = ConfigVersion.CURRENT.value
        
        if current_version == target_version:
            return config, ["No migration needed"]
        
        migrated_config = config.copy()
        migration_log = []
        
        # Find migration path
        migration_path = self._find_migration_path(current_version, target_version)
        
        if not migration_path:
            migration_log.append(f"No migration path found from {current_version} to {target_version}")
            return config, migration_log
        
        # Apply migrations in sequence
        for step in migration_path:
            try:
                logger.info(f"Applying migration: {step.description}")
                migrated_config = step.migration_function(migrated_config)
                migration_log.append(f"✅ {step.from_version} → {step.to_version}: {step.description}")
                
            except Exception as e:
                error_msg = f"❌ Migration failed {step.from_version} → {step.to_version}: {e}"
                migration_log.append(error_msg)
                logger.error(error_msg)
                break
        
        # Set final version
        migrated_config["version"] = target_version
        migration_log.append(f"Set configuration version to {target_version}")
        
        return migrated_config, migration_log
    
    def _find_migration_path(self, from_version: str, to_version: str) -> List[MigrationStep]:
        """Find the migration path between versions"""
        # Simple linear migration path for now
        path = []
        current_version = from_version
        
        while current_version != to_version:
            # Find next step
            next_step = None
            for step in self.migration_steps:
                if step.from_version == current_version:
                    next_step = step
                    break
            
            if not next_step:
                return []  # No path found
            
            path.append(next_step)
            current_version = next_step.to_version
        
        return path
    
    def _migrate_1_0_to_1_1(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from version 1.0 to 1.1"""
        migrated = {}
        
        # Convert old model_settings to diffsynth section
        if "model_settings" in config:
            model_settings = config["model_settings"]
            migrated["diffsynth"] = {
                "model_name": model_settings.get("model_name", "Qwen/Qwen-Image"),
                "torch_dtype": model_settings.get("torch_dtype", "float16"),
                "device": "cuda",
                "enable_vram_management": True,
                "enable_cpu_offload": False,
                "max_memory_usage_gb": 8.0,
                "default_num_inference_steps": 20,
                "default_guidance_scale": 7.5,
                "default_height": 768,
                "default_width": 768
            }
        
        # Convert generation_presets to system defaults
        if "generation_presets" in config:
            presets = config["generation_presets"]
            if "balanced" in presets:
                balanced = presets["balanced"]
                migrated["diffsynth"].update({
                    "default_height": balanced.get("height", 768),
                    "default_width": balanced.get("width", 768),
                    "default_num_inference_steps": balanced.get("num_inference_steps", 20),
                    "default_guidance_scale": balanced.get("guidance_scale", 7.5)
                })
        
        # Convert memory_optimizations to system section
        if "memory_optimizations" in config:
            mem_opts = config["memory_optimizations"]
            migrated["system"] = {
                "enable_attention_slicing": mem_opts.get("enable_attention_slicing", True),
                "enable_vae_slicing": mem_opts.get("enable_vae_slicing", True),
                "enable_vae_tiling": mem_opts.get("enable_vae_tiling", True),
                "enable_memory_efficient_attention": True,
                "enable_xformers": True,
                "safety_checker": True
            }
            
            # Update diffsynth section with CPU offload setting
            migrated["diffsynth"]["enable_cpu_offload"] = mem_opts.get("enable_model_cpu_offload", False)
        
        # Add new ControlNet section with defaults
        migrated["controlnet"] = {
            "control_types": ["canny", "depth"],
            "default_conditioning_scale": 1.0,
            "canny_low_threshold": 100,
            "canny_high_threshold": 200,
            "detect_resolution": 512,
            "image_resolution": 768
        }
        
        # Add performance section
        migrated["performance"] = {
            "max_batch_size": 1,
            "enable_tiled_processing": False,
            "tile_overlap": 64
        }
        
        # Preserve other sections
        for key, value in config.items():
            if key not in ["model_settings", "generation_presets", "memory_optimizations"]:
                migrated[key] = value
        
        migrated["version"] = "1.1"
        return migrated
    
    def _migrate_1_1_to_2_0(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from version 1.1 to 2.0"""
        migrated = config.copy()
        
        # Add EliGen integration to DiffSynth section
        if "diffsynth" in migrated:
            migrated["diffsynth"].update({
                "enable_eligen": True,
                "eligen_mode": "enhanced",
                "eligen_quality_preset": "balanced"
            })
        
        # Enhance ControlNet section with new features
        if "controlnet" in migrated:
            migrated["controlnet"].update({
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "depth_near_plane": 0.1,
                "depth_far_plane": 100.0,
                "multi_controlnet": False,
                "controlnet_weights": [1.0]
            })
        
        # Add workflow section for enhanced UI
        migrated["workflow"] = {
            "default_workflow": "text_to_image",
            "enable_comparison_mode": True,
            "auto_save_results": True,
            "save_intermediate_steps": False,
            "enable_version_history": True
        }
        
        # Update performance section with new options
        if "performance" in migrated:
            migrated["performance"].update({
                "enable_async_processing": True,
                "queue_size": 5,
                "enable_progress_tracking": True
            })
        
        migrated["version"] = "2.0"
        return migrated
    
    def backup_config(self, config_path: str) -> str:
        """
        Create a backup of the configuration file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Path to backup file
        """
        try:
            config_file = Path(config_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = config_file.parent / f"{config_file.stem}_backup_{timestamp}{config_file.suffix}"
            
            shutil.copy2(config_path, backup_path)
            logger.info(f"Configuration backed up to {backup_path}")
            
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to backup configuration: {e}")
            raise
    
    def restore_config(self, backup_path: str, target_path: str) -> bool:
        """
        Restore configuration from backup
        
        Args:
            backup_path: Path to backup file
            target_path: Path to restore to
            
        Returns:
            True if successful
        """
        try:
            shutil.copy2(backup_path, target_path)
            logger.info(f"Configuration restored from {backup_path} to {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore configuration: {e}")
            return False


class ConfigManager:
    """
    High-level configuration management with validation and migration
    Provides a unified interface for configuration operations
    """
    
    def __init__(self, config_dir: str = "config"):
        """Initialize configuration manager"""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.validator = ConfigValidator()
        self.migrator = ConfigMigrator()
        
        # Default configuration files
        self.main_config_file = self.config_dir / "diffsynth_config.json"
        self.backup_dir = self.config_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
    
    def load_and_validate_config(self, config_path: Optional[str] = None) -> Tuple[Dict[str, Any], ValidationResult]:
        """
        Load and validate configuration
        
        Args:
            config_path: Optional path to config file
            
        Returns:
            Tuple of (config, validation_result)
        """
        config_file = Path(config_path) if config_path else self.main_config_file
        
        try:
            # Load configuration
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = self._create_default_config()
                self.save_config(config, str(config_file))
            
            # Migrate if needed
            if self.migrator.needs_migration(config):
                logger.info("Configuration migration required")
                
                # Backup before migration
                backup_path = self.migrator.backup_config(str(config_file))
                
                # Move backup to backup directory
                backup_file = Path(backup_path)
                backup_dest = self.backup_dir / backup_file.name
                if backup_file.exists():
                    import shutil
                    shutil.move(str(backup_file), str(backup_dest))
                    backup_path = str(backup_dest)
                
                # Migrate
                migrated_config, migration_log = self.migrator.migrate_config(config)
                
                # Save migrated config
                self.save_config(migrated_config, str(config_file))
                
                logger.info(f"Configuration migrated successfully. Backup: {backup_path}")
                for log_entry in migration_log:
                    logger.info(log_entry)
                
                config = migrated_config
            
            # Validate
            validation_result = self.validator.validate_config(config)
            
            return config, validation_result
            
        except Exception as e:
            logger.error(f"Failed to load and validate config: {e}")
            # Return default config with error
            default_config = self._create_default_config()
            error_result = ValidationResult(is_valid=False)
            error_result.add_issue(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field_path="config_file",
                message=f"Failed to load configuration: {e}",
                suggested_fix="Check file permissions and JSON syntax"
            ))
            return default_config, error_result
    
    def save_config(self, config: Dict[str, Any], config_path: Optional[str] = None) -> bool:
        """
        Save configuration to file
        
        Args:
            config: Configuration to save
            config_path: Optional path to save to
            
        Returns:
            True if successful
        """
        try:
            config_file = Path(config_path) if config_path else self.main_config_file
            
            # Validate before saving
            validation_result = self.validator.validate_config(config)
            
            if validation_result.has_errors():
                logger.warning("Saving configuration with validation errors")
                for error in validation_result.errors:
                    logger.warning(f"Config error: {error}")
            
            # Save configuration
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def auto_fix_and_save(self, config_path: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Auto-fix configuration issues and save
        
        Args:
            config_path: Optional path to config file
            
        Returns:
            Tuple of (success, fixes_applied)
        """
        try:
            config, validation_result = self.load_and_validate_config(config_path)
            
            if not validation_result.has_errors():
                return True, ["No fixes needed"]
            
            # Apply auto-fixes
            fixed_config, fixes_applied = self.validator.auto_fix_config(config)
            
            # Save fixed configuration
            if fixes_applied:
                success = self.save_config(fixed_config, config_path)
                return success, fixes_applied
            else:
                return True, ["No auto-fixable issues found"]
                
        except Exception as e:
            logger.error(f"Failed to auto-fix configuration: {e}")
            return False, [f"Auto-fix failed: {e}"]
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create a default configuration"""
        return {
            "version": ConfigVersion.CURRENT.value,
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
                "default_width": 768,
                "enable_eligen": True,
                "eligen_mode": "enhanced",
                "eligen_quality_preset": "balanced"
            },
            "controlnet": {
                "control_types": ["canny", "depth"],
                "default_conditioning_scale": 1.0,
                "canny_low_threshold": 100,
                "canny_high_threshold": 200,
                "detect_resolution": 512,
                "image_resolution": 768,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "depth_near_plane": 0.1,
                "depth_far_plane": 100.0,
                "multi_controlnet": False,
                "controlnet_weights": [1.0]
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
                "tile_overlap": 64,
                "enable_async_processing": True,
                "queue_size": 5,
                "enable_progress_tracking": True
            },
            "workflow": {
                "default_workflow": "text_to_image",
                "enable_comparison_mode": True,
                "auto_save_results": True,
                "save_intermediate_steps": False,
                "enable_version_history": True
            }
        }