"""
Unit tests for ModelDetectionService
Tests model detection logic with various model configurations
"""

import json
import os
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_detection_service import ModelDetectionService, ModelInfo


class TestModelDetectionService(unittest.TestCase):
    """Test cases for ModelDetectionService"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.service = ModelDetectionService()
        
        # Override directories to use temp directories
        self.service.cache_dir = os.path.join(self.temp_dir, "cache")
        self.service.local_models_dir = os.path.join(self.temp_dir, "models")
        
        # Create directories
        os.makedirs(self.service.cache_dir, exist_ok=True)
        os.makedirs(self.service.local_models_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def _create_mock_cached_model(self, model_name: str, components: list, 
                                  include_metadata: bool = True, 
                                  file_sizes: dict = None) -> str:
        """Create a mock cached model structure"""
        cache_name = f"models--{model_name.replace('/', '--')}"
        cache_path = os.path.join(self.service.cache_dir, cache_name)
        snapshot_path = os.path.join(cache_path, "snapshots", "abc123")
        
        os.makedirs(snapshot_path, exist_ok=True)
        
        # Create components
        for component in components:
            comp_path = os.path.join(snapshot_path, component)
            if component in ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]:
                # Create as directory with files
                os.makedirs(comp_path, exist_ok=True)
                
                # Add some files to make it realistic
                config_file = os.path.join(comp_path, "config.json")
                with open(config_file, 'w') as f:
                    json.dump({"component": component}, f)
                
                # Add model files with specified sizes
                if file_sizes and component in file_sizes:
                    model_file = os.path.join(comp_path, "model.safetensors")
                    with open(model_file, 'wb') as f:
                        f.write(b'0' * file_sizes[component])
            else:
                # Create as file
                with open(comp_path, 'w') as f:
                    f.write(f"mock {component}")
        
        # Create model_index.json if requested
        if include_metadata:
            model_index = {
                "_class_name": "QwenImagePipeline" if "Image" in model_name else "DiffusionPipeline",
                "_diffusers_version": "0.21.0",
                "transformer": ["diffusers", "QwenImageTransformer2DModel"] if "transformer" in components else None,
                "vae": ["diffusers", "AutoencoderKL"] if "vae" in components else None,
                "text_encoder": ["transformers", "T5EncoderModel"] if "text_encoder" in components else None,
                "tokenizer": ["transformers", "T5Tokenizer"] if "tokenizer" in components else None,
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"] if "scheduler" in components else None
            }
            
            with open(os.path.join(snapshot_path, "model_index.json"), 'w') as f:
                json.dump(model_index, f)
        
        return snapshot_path
    
    def _create_mock_local_model(self, dir_name: str, components: list, 
                                 include_metadata: bool = True) -> str:
        """Create a mock local model structure"""
        model_path = os.path.join(self.service.local_models_dir, dir_name)
        os.makedirs(model_path, exist_ok=True)
        
        # Create components
        for component in components:
            comp_path = os.path.join(model_path, component)
            if component in ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]:
                os.makedirs(comp_path, exist_ok=True)
                # Add a config file
                with open(os.path.join(comp_path, "config.json"), 'w') as f:
                    json.dump({"component": component}, f)
            else:
                with open(comp_path, 'w') as f:
                    f.write(f"mock {component}")
        
        # Create model_index.json if requested
        if include_metadata:
            model_index = {
                "_class_name": "QwenImagePipeline",
                "_diffusers_version": "0.21.0"
            }
            with open(os.path.join(model_path, "model_index.json"), 'w') as f:
                json.dump(model_index, f)
        
        return model_path
    
    def test_detect_complete_qwen_image_model(self):
        """Test detection of complete Qwen-Image model"""
        # Create complete Qwen-Image model
        components = ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
        file_sizes = {
            "transformer": 10 * 1024**3,  # 10GB
            "vae": 2 * 1024**3,           # 2GB
            "text_encoder": 1 * 1024**3,  # 1GB
        }
        
        self._create_mock_cached_model("Qwen/Qwen-Image", components, True, file_sizes)
        
        model = self.service.detect_current_model()
        
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "Qwen/Qwen-Image")
        self.assertEqual(model.model_type, "text-to-image")
        self.assertEqual(model.download_status, "complete")
        self.assertTrue(model.is_optimal)
        self.assertGreater(model.size_gb, 10)  # Should be > 10GB
    
    def test_detect_complete_qwen_image_edit_model(self):
        """Test detection of complete Qwen-Image-Edit model"""
        components = ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
        file_sizes = {
            "transformer": 40 * 1024**3,  # 40GB
            "vae": 5 * 1024**3,           # 5GB
            "text_encoder": 2 * 1024**3,  # 2GB
        }
        
        self._create_mock_cached_model("Qwen/Qwen-Image-Edit", components, True, file_sizes)
        
        model = self.service.detect_current_model()
        
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "Qwen/Qwen-Image-Edit")
        self.assertEqual(model.model_type, "image-editing")
        self.assertEqual(model.download_status, "complete")
        self.assertFalse(model.is_optimal)  # Not optimal for text-to-image
        self.assertGreater(model.size_gb, 40)  # Should be > 40GB
    
    def test_detect_partial_model(self):
        """Test detection of partially downloaded model"""
        # Create model with missing components
        components = ["transformer", "vae"]  # Missing text_encoder, tokenizer, scheduler
        
        self._create_mock_cached_model("Qwen/Qwen-Image", components)
        
        model = self.service.detect_current_model()
        
        self.assertIsNotNone(model)
        self.assertEqual(model.download_status, "partial")
        self.assertFalse(model.is_optimal)
        
        # Check that missing components are detected
        self.assertFalse(model.components.get("text_encoder", True))
        self.assertFalse(model.components.get("tokenizer", True))
        self.assertFalse(model.components.get("scheduler", True))
    
    def test_detect_local_model(self):
        """Test detection of local model"""
        components = ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
        
        self._create_mock_local_model("Qwen-Image-Edit", components)
        
        model = self.service.detect_current_model()
        
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "Qwen/Qwen-Image-Edit")
        self.assertEqual(model.model_type, "image-editing")
        self.assertTrue(model.path.endswith("Qwen-Image-Edit"))
    
    def test_no_model_found(self):
        """Test behavior when no model is found"""
        model = self.service.detect_current_model()
        self.assertIsNone(model)
    
    def test_prefer_text_to_image_over_editing(self):
        """Test that text-to-image model is preferred over editing model"""
        # Create both models
        components = ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
        
        self._create_mock_cached_model("Qwen/Qwen-Image", components)
        self._create_mock_cached_model("Qwen/Qwen-Image-Edit", components)
        
        model = self.service.detect_current_model()
        
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "Qwen/Qwen-Image")
        self.assertEqual(model.model_type, "text-to-image")
        self.assertTrue(model.is_optimal)
    
    def test_optimization_needed_no_model(self):
        """Test optimization needed when no model exists"""
        self.assertTrue(self.service.is_optimization_needed())
    
    def test_optimization_needed_wrong_model_type(self):
        """Test optimization needed when using editing model"""
        components = ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
        self._create_mock_cached_model("Qwen/Qwen-Image-Edit", components)
        
        self.assertTrue(self.service.is_optimization_needed())
    
    def test_optimization_not_needed_optimal_model(self):
        """Test optimization not needed when optimal model exists"""
        components = ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
        self._create_mock_cached_model("Qwen/Qwen-Image", components)
        
        self.assertFalse(self.service.is_optimization_needed())
    
    def test_get_recommended_model(self):
        """Test getting recommended model"""
        recommended = self.service.get_recommended_model()
        self.assertEqual(recommended, "Qwen/Qwen-Image")
    
    def test_component_validation(self):
        """Test model component validation"""
        # Test with all components
        components = ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
        self._create_mock_cached_model("Qwen/Qwen-Image", components)
        
        model = self.service.detect_current_model()
        self.assertIsNotNone(model)
        
        # All essential components should be present
        for component in ["transformer", "vae", "text_encoder"]:
            self.assertTrue(model.components.get(component, False), 
                          f"Component {component} should be present")
    
    def test_model_metadata_loading(self):
        """Test loading of model metadata"""
        components = ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
        self._create_mock_cached_model("Qwen/Qwen-Image", components, include_metadata=True)
        
        model = self.service.detect_current_model()
        self.assertIsNotNone(model)
        self.assertIsInstance(model.metadata, dict)
        self.assertIn("_class_name", model.metadata)
    
    def test_size_calculation(self):
        """Test directory size calculation"""
        components = ["transformer", "vae"]
        file_sizes = {
            "transformer": 5 * 1024**3,  # 5GB
            "vae": 2 * 1024**3,          # 2GB
        }
        
        self._create_mock_cached_model("Qwen/Qwen-Image", components, True, file_sizes)
        
        model = self.service.detect_current_model()
        self.assertIsNotNone(model)
        self.assertGreater(model.size_gb, 6)  # Should be > 6GB
    
    def test_optimization_report(self):
        """Test generation of optimization report"""
        # Create suboptimal model
        components = ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
        self._create_mock_cached_model("Qwen/Qwen-Image-Edit", components)
        
        report = self.service.get_optimization_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn("current_model", report)
        self.assertIn("recommended_model", report)
        self.assertIn("optimization_needed", report)
        self.assertIn("optimization_reasons", report)
        self.assertIn("performance_impact", report)
        
        self.assertTrue(report["optimization_needed"])
        self.assertEqual(report["recommended_model"], "Qwen/Qwen-Image")
        self.assertIn("Using image-editing model for text-to-image tasks", 
                     report["optimization_reasons"])
    
    def test_validate_model_for_text_to_image(self):
        """Test model validation for text-to-image generation"""
        # Create optimal model
        model_info = ModelInfo(
            name="Qwen/Qwen-Image",
            path="/mock/path",
            size_gb=15.0,
            model_type="text-to-image",
            is_optimal=True,
            download_status="complete",
            components={"transformer": True, "vae": True, "text_encoder": True},
            metadata={}
        )
        
        is_valid, issues = self.service.validate_model_for_text_to_image(model_info)
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
        
        # Create problematic model
        problematic_model = ModelInfo(
            name="Qwen/Qwen-Image-Edit",
            path="/mock/path",
            size_gb=60.0,
            model_type="image-editing",
            is_optimal=False,
            download_status="partial",
            components={"transformer": False, "vae": True, "text_encoder": True},
            metadata={}
        )
        
        is_valid, issues = self.service.validate_model_for_text_to_image(problematic_model)
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
        self.assertTrue(any("image editing" in issue for issue in issues))
        self.assertTrue(any("partial" in issue for issue in issues))
    
    def test_infer_model_config_from_directory(self):
        """Test inferring model configuration from directory structure"""
        # Test with transformer-based model
        components = ["transformer", "vae", "text_encoder"]
        model_path = self._create_mock_local_model("unknown-model", components)
        
        config = self.service._infer_model_config(model_path)
        
        self.assertIn("transformer", config["components"])
        self.assertEqual(config["type"], "text-to-image")
        self.assertTrue(config["is_optimal_for_t2i"])
        
        # Test with edit model
        edit_path = self._create_mock_local_model("some-edit-model", components)
        edit_config = self.service._infer_model_config(edit_path)
        
        self.assertEqual(edit_config["type"], "image-editing")
        self.assertFalse(edit_config["is_optimal_for_t2i"])
    
    def test_missing_cache_directory(self):
        """Test behavior when cache directory doesn't exist"""
        self.service.cache_dir = "/nonexistent/path"
        
        models = self.service._scan_cache_directory()
        self.assertEqual(len(models), 0)
    
    def test_missing_local_directory(self):
        """Test behavior when local models directory doesn't exist"""
        self.service.local_models_dir = "/nonexistent/path"
        
        models = self.service._scan_local_directory()
        self.assertEqual(len(models), 0)
    
    def test_corrupted_model_metadata(self):
        """Test handling of corrupted model metadata"""
        components = ["transformer", "vae"]
        snapshot_path = self._create_mock_cached_model("Qwen/Qwen-Image", components, False)
        
        # Create corrupted model_index.json
        with open(os.path.join(snapshot_path, "model_index.json"), 'w') as f:
            f.write("invalid json content")
        
        model = self.service.detect_current_model()
        self.assertIsNotNone(model)
        # Should still work, just with empty metadata
        self.assertIsInstance(model.metadata, dict)
    
    def test_detect_mmdit_architecture(self):
        """Test detection of MMDiT architecture"""
        components = ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
        snapshot_path = self._create_mock_cached_model("Qwen/Qwen-Image", components, True)
        
        # Add transformer config with MMDiT indicators
        transformer_dir = os.path.join(snapshot_path, "transformer")
        transformer_config = {
            "model_type": "MMDiT",
            "hidden_size": 3072,
            "num_layers": 28,
            "num_attention_heads": 24
        }
        with open(os.path.join(transformer_dir, "config.json"), 'w') as f:
            json.dump(transformer_config, f)
        
        model = self.service.detect_current_model()
        self.assertIsNotNone(model)
        
        architecture = self.service.detect_model_architecture(model)
        self.assertEqual(architecture, "MMDiT")
    
    def test_detect_unet_architecture(self):
        """Test detection of UNet architecture"""
        # Create model with UNet structure
        model_path = self._create_mock_local_model("stable-diffusion-model", 
                                                  ["unet", "vae", "text_encoder", "tokenizer", "scheduler"])
        
        model = self.service.detect_current_model()
        self.assertIsNotNone(model)
        
        architecture = self.service.detect_model_architecture(model)
        self.assertEqual(architecture, "UNet")
    
    def test_qwen2_vl_detection_no_models(self):
        """Test Qwen2-VL detection when no models are available"""
        qwen2_vl_info = self.service.detect_qwen2_vl_capabilities()
        
        self.assertIsInstance(qwen2_vl_info, dict)
        self.assertEqual(len(qwen2_vl_info["available_models"]), 0)
        self.assertFalse(qwen2_vl_info["integration_possible"])
        self.assertIsNone(qwen2_vl_info["recommended_model"])
        self.assertEqual(len(qwen2_vl_info["capabilities"]), 0)
    
    def test_qwen2_vl_detection_with_models(self):
        """Test Qwen2-VL detection when models are available"""
        # Create mock Qwen2-VL model in cache
        components = ["model", "tokenizer", "processor"]
        self._create_mock_cached_model("Qwen/Qwen2-VL-7B-Instruct", components, True)
        
        qwen2_vl_info = self.service.detect_qwen2_vl_capabilities()
        
        self.assertGreater(len(qwen2_vl_info["available_models"]), 0)
        self.assertTrue(qwen2_vl_info["integration_possible"])
        self.assertIsNotNone(qwen2_vl_info["recommended_model"])
        self.assertIn("text_understanding", qwen2_vl_info["capabilities"])
    
    def test_multimodal_integration_analysis_compatible(self):
        """Test multimodal integration analysis with compatible models"""
        # Create Qwen-Image model
        components = ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
        self._create_mock_cached_model("Qwen/Qwen-Image", components, True)
        
        # Create Qwen2-VL model
        qwen2_vl_components = ["model", "tokenizer", "processor"]
        self._create_mock_cached_model("Qwen/Qwen2-VL-7B-Instruct", qwen2_vl_components, True)
        
        current_model = self.service.detect_current_model()
        integration_analysis = self.service.analyze_multimodal_integration_potential(current_model)
        
        self.assertTrue(integration_analysis["current_model_compatible"])
        self.assertTrue(integration_analysis["qwen2_vl_available"])
        self.assertGreater(len(integration_analysis["integration_benefits"]), 0)
        self.assertIsNotNone(integration_analysis["recommended_setup"])
    
    def test_multimodal_integration_analysis_incompatible(self):
        """Test multimodal integration analysis with incompatible setup"""
        # Create only image-editing model (no Qwen2-VL)
        components = ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
        self._create_mock_cached_model("Qwen/Qwen-Image-Edit", components, True)
        
        current_model = self.service.detect_current_model()
        integration_analysis = self.service.analyze_multimodal_integration_potential(current_model)
        
        self.assertTrue(integration_analysis["current_model_compatible"])  # Model is compatible
        self.assertFalse(integration_analysis["qwen2_vl_available"])      # But no Qwen2-VL available
        self.assertGreater(len(integration_analysis["integration_requirements"]), 0)
    
    def test_infer_qwen2_vl_config(self):
        """Test inferring Qwen2-VL model configuration"""
        # Create Qwen2-VL model structure
        model_path = self._create_mock_local_model("Qwen2-VL-7B-Instruct", 
                                                  ["model", "tokenizer", "processor"])
        
        config = self.service._infer_model_config(model_path)
        
        self.assertEqual(config["type"], "multimodal-language")
        self.assertEqual(config["architecture"], "Transformer")
        self.assertTrue(config["supports_multimodal"])
        self.assertTrue(config["qwen2_vl_compatible"])
        self.assertFalse(config["is_optimal_for_t2i"])
        self.assertIn("model", config["components"])
        self.assertIn("tokenizer", config["components"])
        self.assertIn("processor", config["components"])
    
    def test_enhanced_optimization_report_with_architecture(self):
        """Test optimization report includes architecture information"""
        # Create MMDiT model
        components = ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
        snapshot_path = self._create_mock_cached_model("Qwen/Qwen-Image", components, True)
        
        # Add transformer config
        transformer_dir = os.path.join(snapshot_path, "transformer")
        transformer_config = {"model_type": "MMDiT", "hidden_size": 3072}
        with open(os.path.join(transformer_dir, "config.json"), 'w') as f:
            json.dump(transformer_config, f)
        
        report = self.service.get_optimization_report()
        
        self.assertIn("architecture", report["current_model"])
        self.assertEqual(report["current_model"]["architecture"], "MMDiT")
        self.assertIn("multimodal_integration", report)
        self.assertIsInstance(report["multimodal_integration"], dict)


class TestModelInfo(unittest.TestCase):
    """Test cases for ModelInfo dataclass"""
    
    def test_model_info_creation(self):
        """Test ModelInfo creation and attributes"""
        model_info = ModelInfo(
            name="Qwen/Qwen-Image",
            path="/test/path",
            size_gb=15.5,
            model_type="text-to-image",
            is_optimal=True,
            download_status="complete",
            components={"transformer": True, "vae": True},
            metadata={"version": "1.0"}
        )
        
        self.assertEqual(model_info.name, "Qwen/Qwen-Image")
        self.assertEqual(model_info.path, "/test/path")
        self.assertEqual(model_info.size_gb, 15.5)
        self.assertEqual(model_info.model_type, "text-to-image")
        self.assertTrue(model_info.is_optimal)
        self.assertEqual(model_info.download_status, "complete")
        self.assertTrue(model_info.components["transformer"])
        self.assertEqual(model_info.metadata["version"], "1.0")


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)