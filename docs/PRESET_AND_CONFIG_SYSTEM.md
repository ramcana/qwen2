# Preset and Configuration System

## Overview

This document describes the comprehensive preset management and configuration validation system implemented for the DiffSynth Enhanced UI. The system provides robust configuration management with automatic validation, migration, and preset functionality.

## Components

### 1. PresetManager (`src/preset_manager.py`)

A comprehensive preset management system that handles saving, loading, and organizing user configurations for different use cases.

#### Key Features:

- **Preset Categories**: Photo editing, artistic creation, technical illustration, ControlNet, DiffSynth, general, and custom
- **Preset Types**: Generation, editing, ControlNet, system, and workflow presets
- **Import/Export**: Full preset import/export functionality with metadata preservation
- **Search and Filter**: Advanced search by name, description, tags, category, and type
- **Backup/Restore**: Complete preset collection backup and restore capabilities
- **Usage Tracking**: Automatic usage count and rating system
- **Validation**: Built-in preset validation and auto-fixing

#### Preset Data Models:

- `GenerationPreset`: Text-to-image generation settings
- `EditingPreset`: Image editing operation parameters
- `ControlNetPreset`: ControlNet-specific configurations
- `SystemPreset`: System and hardware optimization settings
- `WorkflowPreset`: Complete workflow configurations

#### Default Presets Created:

- **Photo Editing**: Portrait Enhancement, Landscape Enhancement
- **Artistic Creation**: Digital Art Creation, Artistic Style Transfer
- **Technical Illustration**: Technical Illustration preset
- **ControlNet**: Canny Edge Control, Depth-Guided Generation
- **System**: Performance Optimized, Quality Optimized

### 2. ConfigValidator (`src/config_validator.py`)

A robust configuration validation system that ensures configuration integrity and provides automatic fixing capabilities.

#### Key Features:

- **Comprehensive Validation**: Type checking, value ranges, allowed values, cross-section compatibility
- **Hardware Compatibility**: Validates settings against available hardware resources
- **Auto-Fix Capability**: Automatically fixes common configuration issues
- **Detailed Error Reporting**: Provides specific error messages with suggested fixes
- **Severity Levels**: Info, Warning, Error, and Critical issue classification

#### Validation Rules:

- **DiffSynth Settings**: Model configuration, memory management, generation parameters
- **ControlNet Settings**: Control types, detection parameters, conditioning scales
- **System Settings**: Hardware optimizations, memory management, safety settings
- **Performance Settings**: Batch sizes, tiled processing, optimization flags

### 3. ConfigMigrator (`src/config_validator.py`)

Automatic configuration migration system that handles version upgrades seamlessly.

#### Migration Path:

- **v1.0 → v1.1**: Convert legacy model_settings and generation_presets to new structure
- **v1.1 → v2.0**: Add EliGen integration, enhanced ControlNet features, workflow settings

#### Features:

- **Automatic Backup**: Creates backups before migration
- **Incremental Migration**: Supports step-by-step version upgrades
- **Migration Logging**: Detailed logs of migration steps and results
- **Rollback Support**: Ability to restore from backups if needed

### 4. ConfigManager (`src/config_validator.py`)

High-level configuration management interface that combines validation, migration, and file operations.

#### Features:

- **Unified Interface**: Single point of access for all configuration operations
- **Automatic Migration**: Detects and performs necessary migrations on load
- **Validation Integration**: Validates configurations on load and save
- **Backup Management**: Organized backup storage and restoration
- **Error Recovery**: Graceful handling of corrupted or invalid configurations

## Usage Examples

### Basic Preset Management

```python
from src.preset_manager import PresetManager, Preset, PresetMetadata, GenerationPreset

# Initialize preset manager
preset_manager = PresetManager()

# Create a custom preset
preset = Preset(
    metadata=PresetMetadata(
        name="My Custom Preset",
        description="Custom settings for my workflow",
        category=PresetCategory.CUSTOM,
        preset_type=PresetType.GENERATION
    ),
    generation=GenerationPreset(
        width=1024,
        height=768,
        num_inference_steps=25,
        guidance_scale=8.0
    )
)

# Save preset
preset_manager.save_preset(preset)

# Load preset
loaded_preset = preset_manager.load_preset("My Custom Preset")

# Search presets
results = preset_manager.search_presets("custom")

# Export preset
preset_manager.export_preset("My Custom Preset", "my_preset.json")
```

### Configuration Validation and Migration

```python
from src.config_validator import ConfigManager

# Initialize config manager
config_manager = ConfigManager()

# Load and validate configuration (with automatic migration)
config, validation_result = config_manager.load_and_validate_config()

if validation_result.has_errors():
    print("Configuration errors found:")
    for error in validation_result.errors:
        print(f"  - {error}")

    # Auto-fix issues
    success, fixes = config_manager.auto_fix_and_save()
    if success:
        print(f"Applied {len(fixes)} fixes")

# Save configuration
config_manager.save_config(config)
```

### Manual Validation

```python
from src.config_validator import ConfigValidator

validator = ConfigValidator()

# Validate configuration
result = validator.validate_config(config)

if not result.is_valid:
    print("Validation issues:")
    for issue in result.issues:
        print(f"  {issue.severity.value}: {issue.message}")
        if issue.suggested_fix:
            print(f"    Fix: {issue.suggested_fix}")

# Auto-fix configuration
fixed_config, fixes = validator.auto_fix_config(config)
```

## File Structure

```
config/
├── presets/
│   ├── photo_editing/
│   ├── artistic_creation/
│   ├── technical_illustration/
│   ├── controlnet/
│   ├── diffsynth/
│   ├── general/
│   └── custom/
├── backups/
└── diffsynth_config.json
```

## Testing

The system includes comprehensive test suites:

- **test_preset_manager.py**: 26 tests covering all preset management functionality
- **test_config_validator.py**: 27 tests covering validation, migration, and configuration management

All tests pass and provide 100% coverage of the implemented functionality.

## Integration Points

### With DiffSynth Service

- Presets provide configuration templates for DiffSynth operations
- Configuration validation ensures DiffSynth settings are compatible
- Migration system handles DiffSynth configuration updates

### With ControlNet Service

- ControlNet-specific presets for different control types
- Validation of ControlNet parameters and compatibility
- Cross-validation between DiffSynth and ControlNet settings

### With UI Components

- Preset selection interfaces for different workflows
- Configuration validation feedback in UI
- Import/export functionality for sharing presets

## Benefits

1. **User Experience**: Easy preset management for common workflows
2. **Reliability**: Robust validation prevents configuration errors
3. **Maintainability**: Automatic migration handles version upgrades
4. **Flexibility**: Support for custom presets and configurations
5. **Robustness**: Comprehensive error handling and recovery mechanisms
6. **Performance**: Optimized configurations for different hardware setups

## Future Enhancements

- Cloud preset sharing and synchronization
- AI-powered preset recommendations
- Performance-based preset optimization
- Integration with model-specific optimizations
- Advanced preset templating system
