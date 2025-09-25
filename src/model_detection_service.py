"""
Model Detection and Validation Service
Detects existing models and determines optimization needs for Qwen image generation
"""

import json
import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a detected model"""
    name: str
    path: str
    size_gb: float
    model_type: str  # "text-to-image" or "image-editing"
    is_optimal: bool
    download_status: str  # "complete", "partial", "missing"
    components: Dict[str, bool]  # Component availability
    metadata: Dict[str, any]  # Additional metadata


class ModelDetectionService:
    """Service for detecting and validating Qwen models"""
    
    def __init__(self):
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        self.local_models_dir = "./models"
        
        # Known model configurations based on official Qwen-Image documentation
        self.known_models = {
            "Qwen/Qwen-Image": {
                "type": "text-to-image",
                "expected_size_gb": 60,  # Corrected: ~60GB (text_encoder ~16GB + transformer ~45GB + other components)
                "is_optimal_for_t2i": True,
                "components": ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"],
                "pipeline_class": "AutoPipelineForText2Image",
                "architecture": "MMDiT",  # Multimodal Diffusion Transformer
                "supports_multimodal": False,
                "qwen2_vl_compatible": True
            },
            "Qwen/Qwen-Image-Edit": {
                "type": "image-editing", 
                "expected_size_gb": 60,  # Corrected: Similar size to Qwen-Image, both are large MMDiT models
                "is_optimal_for_t2i": False,
                "components": ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"],
                "pipeline_class": "DiffusionPipeline",
                "architecture": "MMDiT",
                "supports_multimodal": False,
                "qwen2_vl_compatible": True
            },
            "Qwen/Qwen2-VL-7B-Instruct": {
                "type": "multimodal-language",
                "expected_size_gb": 15,
                "is_optimal_for_t2i": False,
                "components": ["model", "tokenizer", "processor"],
                "pipeline_class": "Qwen2VLForConditionalGeneration",
                "architecture": "Transformer",
                "supports_multimodal": True,
                "qwen2_vl_compatible": True,
                "capabilities": ["text_understanding", "image_analysis", "prompt_enhancement"]
            },
            "Qwen/Qwen2-VL-2B-Instruct": {
                "type": "multimodal-language",
                "expected_size_gb": 4,
                "is_optimal_for_t2i": False,
                "components": ["model", "tokenizer", "processor"],
                "pipeline_class": "Qwen2VLForConditionalGeneration",
                "architecture": "Transformer",
                "supports_multimodal": True,
                "qwen2_vl_compatible": True,
                "capabilities": ["text_understanding", "image_analysis", "prompt_enhancement"]
            }
        }
    
    def detect_current_model(self) -> Optional[ModelInfo]:
        """Detect the currently available model"""
        logger.info("🔍 Detecting current model configuration...")
        
        # Check both cache and local directories
        models_found = []
        
        # Check HuggingFace cache
        cache_models = self._scan_cache_directory()
        models_found.extend(cache_models)
        
        # Check local models directory
        local_models = self._scan_local_directory()
        models_found.extend(local_models)
        
        if not models_found:
            logger.warning("❌ No Qwen models found")
            return None
        
        # Return the most complete model
        complete_models = [m for m in models_found if m.download_status == "complete"]
        if complete_models:
            # Prefer text-to-image model if available
            t2i_models = [m for m in complete_models if m.model_type == "text-to-image"]
            if t2i_models:
                logger.info(f"✅ Found optimal text-to-image model: {t2i_models[0].name}")
                return t2i_models[0]
            else:
                logger.info(f"⚠️ Found image-editing model: {complete_models[0].name}")
                return complete_models[0]
        
        # Return partial model if no complete ones
        logger.warning(f"⚠️ Only partial models found: {models_found[0].name}")
        return models_found[0]
    
    def _scan_cache_directory(self) -> List[ModelInfo]:
        """Scan HuggingFace cache directory for models"""
        models = []
        
        if not os.path.exists(self.cache_dir):
            logger.warning(f"Cache directory not found: {self.cache_dir}")
            return models
        
        for model_name, config in self.known_models.items():
            cache_path = os.path.join(
                self.cache_dir, 
                f"models--{model_name.replace('/', '--')}"
            )
            
            if os.path.exists(cache_path):
                model_info = self._analyze_cached_model(model_name, cache_path, config)
                if model_info:
                    models.append(model_info)
        
        return models
    
    def _scan_local_directory(self) -> List[ModelInfo]:
        """Scan local models directory"""
        models = []
        
        if not os.path.exists(self.local_models_dir):
            logger.info(f"Local models directory not found: {self.local_models_dir}")
            return models
        
        # Look for model directories
        for item in os.listdir(self.local_models_dir):
            item_path = os.path.join(self.local_models_dir, item)
            if os.path.isdir(item_path):
                model_info = self._analyze_local_model(item, item_path)
                if model_info:
                    models.append(model_info)
        
        return models
    
    def _analyze_cached_model(self, model_name: str, cache_path: str, config: Dict) -> Optional[ModelInfo]:
        """Analyze a cached model"""
        try:
            # Find snapshot directory
            snapshots_dir = os.path.join(cache_path, "snapshots")
            if not os.path.exists(snapshots_dir):
                return None
            
            snapshots = os.listdir(snapshots_dir)
            if not snapshots:
                return None
            
            snapshot_path = os.path.join(snapshots_dir, snapshots[0])
            
            # Analyze components
            components = self._check_model_components(snapshot_path, config["components"])
            
            # Calculate size
            size_gb = self._calculate_directory_size(cache_path)
            
            # Determine download status
            missing_components = [comp for comp, exists in components.items() if not exists]
            if not missing_components:
                download_status = "complete"
            elif len(missing_components) < len(components) / 2:
                download_status = "partial"
            else:
                download_status = "missing"
            
            # Load metadata
            metadata = self._load_model_metadata(snapshot_path)
            
            return ModelInfo(
                name=model_name,
                path=snapshot_path,
                size_gb=size_gb,
                model_type=config["type"],
                is_optimal=(config["is_optimal_for_t2i"] and download_status == "complete"),
                download_status=download_status,
                components=components,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error analyzing cached model {model_name}: {e}")
            return None
    
    def _analyze_local_model(self, dir_name: str, model_path: str) -> Optional[ModelInfo]:
        """Analyze a local model directory"""
        try:
            # Try to match with known models
            model_name = None
            config = None
            
            # Check for exact matches or variations
            for known_name, known_config in self.known_models.items():
                if (dir_name.lower() in known_name.lower() or 
                    known_name.split('/')[-1].lower() in dir_name.lower()):
                    model_name = known_name
                    config = known_config
                    break
            
            if not config:
                # Try to infer from directory structure
                config = self._infer_model_config(model_path)
                model_name = f"local/{dir_name}"
            
            # Analyze components
            components = self._check_model_components(model_path, config.get("components", []))
            
            # Calculate size
            size_gb = self._calculate_directory_size(model_path)
            
            # Determine status
            missing_components = [comp for comp, exists in components.items() if not exists]
            if not missing_components:
                download_status = "complete"
            elif len(missing_components) < len(components) / 2:
                download_status = "partial"
            else:
                download_status = "missing"
            
            # Load metadata
            metadata = self._load_model_metadata(model_path)
            
            return ModelInfo(
                name=model_name,
                path=model_path,
                size_gb=size_gb,
                model_type=config.get("type", "unknown"),
                is_optimal=(config.get("is_optimal_for_t2i", False) and download_status == "complete"),
                download_status=download_status,
                components=components,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error analyzing local model {dir_name}: {e}")
            return None
    
    def _check_model_components(self, model_path: str, expected_components: List[str]) -> Dict[str, bool]:
        """Check which model components are present"""
        components = {}
        
        for component in expected_components:
            component_path = os.path.join(model_path, component)
            
            # Check if component exists as file or directory
            if os.path.exists(component_path):
                if os.path.isdir(component_path):
                    # Directory should have files
                    components[component] = len(os.listdir(component_path)) > 0
                else:
                    # File should exist and have size > 0
                    components[component] = os.path.getsize(component_path) > 0
            else:
                components[component] = False
        
        # Also check for essential files
        essential_files = ["model_index.json"]
        for file_name in essential_files:
            file_path = os.path.join(model_path, file_name)
            components[file_name] = os.path.exists(file_path)
        
        return components
    
    def _calculate_directory_size(self, directory: str) -> float:
        """Calculate directory size in GB"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        continue
            return total_size / (1024**3)  # Convert to GB
        except Exception as e:
            logger.error(f"Error calculating directory size: {e}")
            return 0.0
    
    def _load_model_metadata(self, model_path: str) -> Dict[str, any]:
        """Load model metadata from model_index.json and other config files"""
        metadata = {}
        
        try:
            # Load main model index
            model_index_path = os.path.join(model_path, "model_index.json")
            if os.path.exists(model_index_path):
                with open(model_index_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            # Load transformer config for additional details
            transformer_config_path = os.path.join(model_path, "transformer", "config.json")
            if os.path.exists(transformer_config_path):
                try:
                    with open(transformer_config_path, 'r', encoding='utf-8') as f:
                        transformer_config = json.load(f)
                        metadata["transformer_config"] = transformer_config
                        
                        # Extract key architecture details
                        if "model_type" in transformer_config:
                            metadata["architecture_type"] = transformer_config["model_type"]
                        if "hidden_size" in transformer_config:
                            metadata["hidden_size"] = transformer_config["hidden_size"]
                        if "num_layers" in transformer_config:
                            metadata["num_layers"] = transformer_config["num_layers"]
                            
                except Exception as e:
                    logger.warning(f"Could not load transformer config: {e}")
            
            # Check for README to get model description
            readme_path = os.path.join(model_path, "README.md")
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        readme_content = f.read()
                        # Extract key information from README
                        if "text-to-image" in readme_content.lower():
                            metadata["supports_text_to_image"] = True
                        if "image editing" in readme_content.lower() or "image-to-image" in readme_content.lower():
                            metadata["supports_image_editing"] = True
                        if "MMDiT" in readme_content or "multimodal" in readme_content.lower():
                            metadata["uses_mmdit"] = True
                except Exception as e:
                    logger.warning(f"Could not load README: {e}")
                    
        except Exception as e:
            logger.warning(f"Could not load model metadata: {e}")
        
        return metadata
    
    def _infer_model_config(self, model_path: str) -> Dict[str, any]:
        """Infer model configuration from directory structure"""
        config = {
            "type": "unknown",
            "expected_size_gb": 0,
            "is_optimal_for_t2i": False,
            "components": [],
            "architecture": "Unknown",
            "supports_multimodal": False,
            "qwen2_vl_compatible": False
        }
        
        # Check directory contents to infer type
        if os.path.exists(model_path):
            contents = os.listdir(model_path)
            
            # Check for Qwen2-VL model structure
            if any(item in contents for item in ["model", "tokenizer", "processor"]) and "qwen2" in model_path.lower() and "vl" in model_path.lower():
                config["type"] = "multimodal-language"
                config["components"].extend(["model", "tokenizer", "processor"])
                config["architecture"] = "Transformer"
                config["supports_multimodal"] = True
                config["qwen2_vl_compatible"] = True
                config["expected_size_gb"] = 15 if "7b" in model_path.lower() else 4
                config["is_optimal_for_t2i"] = False
            
            # Look for transformer (indicates MMDiT architecture)
            elif "transformer" in contents:
                config["components"].extend(["transformer", "vae", "text_encoder", "tokenizer", "scheduler"])
                config["architecture"] = "MMDiT"
                config["qwen2_vl_compatible"] = True
                
                # Try to determine if it's editing or text-to-image
                if "edit" in model_path.lower():
                    config["type"] = "image-editing"
                    config["expected_size_gb"] = 50
                    config["is_optimal_for_t2i"] = False
                else:
                    config["type"] = "text-to-image"
                    config["expected_size_gb"] = 15
                    config["is_optimal_for_t2i"] = True
            
            # Look for unet (indicates UNet architecture)
            elif "unet" in contents:
                config["components"].extend(["unet", "vae", "text_encoder", "tokenizer", "scheduler"])
                config["architecture"] = "UNet"
                config["type"] = "text-to-image"
                config["expected_size_gb"] = 10
                config["is_optimal_for_t2i"] = True
                config["qwen2_vl_compatible"] = False  # UNet models typically not compatible
        
        return config
    
    def is_optimization_needed(self) -> bool:
        """Determine if model optimization is needed"""
        current_model = self.detect_current_model()
        
        if not current_model:
            logger.info("🔄 Optimization needed: No model found")
            return True
        
        if current_model.model_type == "image-editing":
            logger.info("🔄 Optimization needed: Using image-editing model for text-to-image")
            return True
        
        if current_model.download_status != "complete":
            logger.info("🔄 Optimization needed: Model incomplete")
            return True
        
        if not current_model.is_optimal:
            logger.info("🔄 Optimization needed: Suboptimal model configuration")
            return True
        
        logger.info("✅ No optimization needed: Optimal model already available")
        return False
    
    def get_recommended_model(self) -> str:
        """Get the recommended model for text-to-image generation"""
        current_model = self.detect_current_model()
        
        # Always recommend Qwen-Image for text-to-image tasks
        recommended = "Qwen/Qwen-Image"
        
        if current_model and current_model.name == recommended and current_model.is_optimal:
            logger.info(f"✅ Current model is already optimal: {recommended}")
        else:
            logger.info(f"💡 Recommended model for optimization: {recommended}")
        
        return recommended
    
    def get_recommended_pipeline_class(self, model_name: str) -> str:
        """Get the recommended pipeline class for a model"""
        if model_name in self.known_models:
            return self.known_models[model_name]["pipeline_class"]
        
        # Default recommendations based on model name
        if "edit" in model_name.lower():
            return "DiffusionPipeline"  # Generic pipeline for editing models
        else:
            return "AutoPipelineForText2Image"  # Optimized for text-to-image
    
    def analyze_performance_characteristics(self, model_info: ModelInfo) -> Dict[str, any]:
        """Analyze expected performance characteristics of a model"""
        characteristics = {
            "expected_generation_time": "unknown",
            "memory_usage": "unknown",
            "optimization_level": "unknown",
            "bottlenecks": []
        }
        
        if model_info.model_type == "text-to-image":
            if model_info.size_gb <= 10:
                characteristics["expected_generation_time"] = "2-5 seconds per step"
                characteristics["memory_usage"] = "8-12GB VRAM"
                characteristics["optimization_level"] = "optimal"
            else:
                characteristics["expected_generation_time"] = "5-15 seconds per step"
                characteristics["memory_usage"] = "12-16GB VRAM"
                characteristics["optimization_level"] = "suboptimal"
                characteristics["bottlenecks"].append("Large model size")
        
        elif model_info.model_type == "image-editing":
            characteristics["expected_generation_time"] = "30-180+ seconds per step"
            characteristics["memory_usage"] = "16+ GB VRAM"
            characteristics["optimization_level"] = "poor for text-to-image"
            characteristics["bottlenecks"].extend([
                "Wrong model type for text-to-image",
                "Large model size",
                "Editing-specific overhead"
            ])
        
        # Check for specific issues
        if model_info.download_status != "complete":
            characteristics["bottlenecks"].append("Incomplete model download")
        
        if model_info.size_gb > 40:
            characteristics["bottlenecks"].append("Extremely large model size")
        
        return characteristics

    def detect_model_architecture(self, model_info: ModelInfo) -> str:
        """Detect model architecture type (MMDiT vs UNet)"""
        try:
            # Check transformer config for architecture details
            if "transformer_config" in model_info.metadata:
                transformer_config = model_info.metadata["transformer_config"]
                
                # Look for MMDiT-specific indicators
                if "model_type" in transformer_config:
                    model_type = transformer_config["model_type"].lower()
                    if "mmdit" in model_type or "multimodal" in model_type:
                        return "MMDiT"
                
                # Check for transformer-based architecture indicators
                if any(key in transformer_config for key in ["num_layers", "hidden_size", "num_attention_heads"]):
                    # Modern transformer architecture
                    return "MMDiT"
            
            # Check components to infer architecture
            if model_info.components.get("transformer", False):
                return "MMDiT"  # Has transformer component = MMDiT architecture
            elif model_info.components.get("unet", False):
                return "UNet"   # Has unet component = UNet architecture
            
            # Check model name patterns
            if "qwen" in model_info.name.lower() and "image" in model_info.name.lower():
                return "MMDiT"  # Qwen-Image models use MMDiT
            
            # Default fallback based on model metadata
            if model_info.metadata.get("uses_mmdit", False):
                return "MMDiT"
            
            return "Unknown"
            
        except Exception as e:
            logger.warning(f"Could not detect architecture for {model_info.name}: {e}")
            return "Unknown"
    
    def detect_qwen2_vl_capabilities(self) -> Dict[str, any]:
        """Detect available Qwen2-VL models and their capabilities"""
        qwen2_vl_info = {
            "available_models": [],
            "recommended_model": None,
            "integration_possible": False,
            "capabilities": []
        }
        
        try:
            # Check for Qwen2-VL models in cache and local directories
            for model_name in ["Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-2B-Instruct"]:
                # Check cache
                cache_path = os.path.join(
                    self.cache_dir, 
                    f"models--{model_name.replace('/', '--')}"
                )
                
                if os.path.exists(cache_path):
                    model_info = self._analyze_cached_model(model_name, cache_path, self.known_models[model_name])
                    if model_info and model_info.download_status == "complete":
                        qwen2_vl_info["available_models"].append({
                            "name": model_name,
                            "path": model_info.path,
                            "size_gb": model_info.size_gb,
                            "capabilities": self.known_models[model_name]["capabilities"]
                        })
                
                # Check local directory
                local_variations = [
                    model_name.split('/')[-1],  # Qwen2-VL-7B-Instruct
                    model_name.split('/')[-1].lower(),  # qwen2-vl-7b-instruct
                    model_name.split('/')[-1].replace('-', '_'),  # Qwen2_VL_7B_Instruct
                ]
                
                for variation in local_variations:
                    local_path = os.path.join(self.local_models_dir, variation)
                    if os.path.exists(local_path):
                        model_info = self._analyze_local_model(variation, local_path)
                        if model_info and model_info.download_status == "complete":
                            qwen2_vl_info["available_models"].append({
                                "name": model_name,
                                "path": model_info.path,
                                "size_gb": model_info.size_gb,
                                "capabilities": self.known_models[model_name]["capabilities"]
                            })
            
            # Determine integration possibilities
            if qwen2_vl_info["available_models"]:
                qwen2_vl_info["integration_possible"] = True
                
                # Recommend the best available model (prefer 7B over 2B for better performance)
                for model in qwen2_vl_info["available_models"]:
                    if "7B" in model["name"]:
                        qwen2_vl_info["recommended_model"] = model
                        break
                
                if not qwen2_vl_info["recommended_model"]:
                    qwen2_vl_info["recommended_model"] = qwen2_vl_info["available_models"][0]
                
                # Aggregate capabilities
                all_capabilities = set()
                for model in qwen2_vl_info["available_models"]:
                    all_capabilities.update(model["capabilities"])
                qwen2_vl_info["capabilities"] = list(all_capabilities)
            
            logger.info(f"Qwen2-VL detection: {len(qwen2_vl_info['available_models'])} models found")
            
        except Exception as e:
            logger.error(f"Error detecting Qwen2-VL capabilities: {e}")
        
        return qwen2_vl_info
    
    def analyze_multimodal_integration_potential(self, current_model: Optional[ModelInfo] = None) -> Dict[str, any]:
        """Analyze potential for multimodal integration with current setup"""
        if not current_model:
            current_model = self.detect_current_model()
        
        qwen2_vl_info = self.detect_qwen2_vl_capabilities()
        
        integration_analysis = {
            "current_model_compatible": False,
            "qwen2_vl_available": qwen2_vl_info["integration_possible"],
            "integration_benefits": [],
            "integration_requirements": [],
            "recommended_setup": None
        }
        
        if current_model:
            # Check if current model supports Qwen2-VL integration
            model_config = self.known_models.get(current_model.name, {})
            integration_analysis["current_model_compatible"] = model_config.get("qwen2_vl_compatible", False)
            
            if integration_analysis["current_model_compatible"] and qwen2_vl_info["integration_possible"]:
                integration_analysis["integration_benefits"] = [
                    "Enhanced text understanding for better prompt interpretation",
                    "Image analysis capabilities for context-aware generation",
                    "Improved prompt enhancement and refinement",
                    "Better handling of complex multimodal instructions"
                ]
                
                integration_analysis["recommended_setup"] = {
                    "primary_model": current_model.name,
                    "multimodal_model": qwen2_vl_info["recommended_model"]["name"] if qwen2_vl_info["recommended_model"] else None,
                    "integration_mode": "prompt_enhancement",
                    "expected_benefits": "10-30% better prompt understanding and image quality"
                }
            else:
                integration_analysis["integration_requirements"] = [
                    "Compatible image generation model (Qwen-Image or Qwen-Image-Edit)",
                    "Qwen2-VL model (7B-Instruct recommended for best performance)",
                    "Sufficient VRAM for running both models (24GB+ recommended)"
                ]
        
        return integration_analysis

    def get_optimization_report(self) -> Dict[str, any]:
        """Generate a comprehensive optimization report"""
        current_model = self.detect_current_model()
        recommended_model = self.get_recommended_model()
        optimization_needed = self.is_optimization_needed()
        
        # Analyze performance characteristics
        performance_analysis = None
        architecture_type = "Unknown"
        if current_model:
            performance_analysis = self.analyze_performance_characteristics(current_model)
            architecture_type = self.detect_model_architecture(current_model)
        
        # Analyze multimodal integration potential
        multimodal_analysis = self.analyze_multimodal_integration_potential(current_model)
        
        report = {
            "current_model": {
                "name": current_model.name if current_model else None,
                "type": current_model.model_type if current_model else None,
                "size_gb": current_model.size_gb if current_model else 0,
                "status": current_model.download_status if current_model else "missing",
                "is_optimal": current_model.is_optimal if current_model else False,
                "path": current_model.path if current_model else None,
                "pipeline_class": self.get_recommended_pipeline_class(current_model.name) if current_model else None,
                "architecture": architecture_type,
                "performance_analysis": performance_analysis
            },
            "recommended_model": {
                "name": recommended_model,
                "pipeline_class": self.get_recommended_pipeline_class(recommended_model),
                "expected_size_gb": self.known_models[recommended_model]["expected_size_gb"],
                "architecture": self.known_models[recommended_model]["architecture"]
            },
            "optimization_needed": optimization_needed,
            "optimization_reasons": [],
            "performance_impact": {
                "current_estimated_time": performance_analysis["expected_generation_time"] if performance_analysis else "188+ seconds per step",
                "optimized_estimated_time": "2-5 seconds per step",
                "expected_speedup": "50-100x faster",
                "bottlenecks": performance_analysis["bottlenecks"] if performance_analysis else []
            },
            "multimodal_integration": multimodal_analysis
        }
        
        # Add optimization reasons
        if not current_model:
            report["optimization_reasons"].append("No model found")
        elif current_model.model_type == "image-editing":
            report["optimization_reasons"].append("Using image-editing model for text-to-image tasks")
        elif current_model.download_status != "complete":
            report["optimization_reasons"].append("Model download incomplete")
        elif not current_model.is_optimal:
            report["optimization_reasons"].append("Suboptimal model configuration")
        
        return report
    
    def validate_model_for_text_to_image(self, model_info: ModelInfo) -> Tuple[bool, List[str]]:
        """Validate if a model is suitable for text-to-image generation"""
        issues = []
        
        if model_info.model_type == "image-editing":
            issues.append("Model is designed for image editing, not text-to-image generation")
        
        if model_info.download_status != "complete":
            issues.append(f"Model download is {model_info.download_status}")
        
        if model_info.size_gb > 40:
            issues.append("Model is unusually large, may cause performance issues")
        
        # Check essential components
        essential_components = ["transformer", "vae", "text_encoder"]
        missing_components = [comp for comp in essential_components 
                            if comp in model_info.components and not model_info.components[comp]]
        
        if missing_components:
            issues.append(f"Missing essential components: {', '.join(missing_components)}")
        
        is_valid = len(issues) == 0
        return is_valid, issues


def main():
    """Test the model detection service"""
    print("🔍 Testing Model Detection Service")
    print("=" * 50)
    
    service = ModelDetectionService()
    
    # Test current model detection
    current_model = service.detect_current_model()
    if current_model:
        print(f"📋 Current Model: {current_model.name}")
        print(f"   Type: {current_model.model_type}")
        print(f"   Size: {current_model.size_gb:.1f}GB")
        print(f"   Status: {current_model.download_status}")
        print(f"   Optimal: {current_model.is_optimal}")
        print(f"   Path: {current_model.path}")
    else:
        print("❌ No model detected")
    
    # Test optimization check
    print(f"\n🔄 Optimization needed: {service.is_optimization_needed()}")
    print(f"💡 Recommended model: {service.get_recommended_model()}")
    
    # Generate report
    report = service.get_optimization_report()
    print(f"\n📊 Optimization Report:")
    print(f"   Current: {report['current_model']['name']} ({report['current_model']['type']})")
    print(f"   Recommended: {report['recommended_model']}")
    print(f"   Reasons: {', '.join(report['optimization_reasons'])}")


if __name__ == "__main__":
    main()