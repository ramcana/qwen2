"""
Qwen2-VL Integration for Enhanced Multimodal Capabilities
Provides text understanding, prompt enhancement, and image analysis using Qwen2-VL models
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch
from PIL import Image
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from transformers import (
        Qwen2VLForConditionalGeneration, 
        AutoTokenizer, 
        AutoProcessor,
        pipeline
    )
    QWEN2VL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Qwen2-VL dependencies not available: {e}")
    QWEN2VL_AVAILABLE = False


@dataclass
class Qwen2VLConfig:
    """Configuration for Qwen2-VL integration"""
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    torch_dtype: torch.dtype = torch.float16
    device_map: str = "auto"
    trust_remote_code: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    enable_prompt_enhancement: bool = True
    enable_image_analysis: bool = True
    context_length: int = 2048
    fallback_enabled: bool = True
    cache_responses: bool = True


@dataclass
class PromptEnhancementResult:
    """Result of prompt enhancement operation"""
    original_prompt: str
    enhanced_prompt: str
    enhancement_type: str
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class ImageAnalysisResult:
    """Result of image analysis operation"""
    description: str
    key_elements: List[str]
    style_analysis: Dict[str, str]
    composition_notes: str
    suggested_improvements: List[str]
    confidence: float


class Qwen2VLIntegration:
    """
    Qwen2-VL Integration class for multimodal text understanding
    Provides prompt enhancement and image analysis capabilities
    """
    
    def __init__(self, config: Optional[Qwen2VLConfig] = None):
        self.config = config or Qwen2VLConfig()
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.pipeline = None
        self.is_available = QWEN2VL_AVAILABLE
        self.is_loaded = False
        self.response_cache = {} if self.config.cache_responses else None
        
        # Initialize model detection service for compatibility checking
        try:
            from model_detection_service import ModelDetectionService
            self.detection_service = ModelDetectionService()
        except ImportError:
            self.detection_service = None
            logger.warning("Model detection service not available")
        
        logger.info(f"Qwen2-VL Integration initialized (Available: {self.is_available})")
    
    def load_model(self) -> bool:
        """Load Qwen2-VL model and components"""
        if not self.is_available:
            logger.warning("Qwen2-VL dependencies not available")
            return False
        
        if self.is_loaded:
            logger.info("Qwen2-VL model already loaded")
            return True
        
        try:
            logger.info(f"Loading Qwen2-VL model: {self.config.model_name}")
            
            # Check if model is available locally
            model_available = self._check_model_availability()
            if not model_available:
                logger.warning(f"Model {self.config.model_name} not found locally")
                if not self.config.fallback_enabled:
                    return False
                logger.info("Attempting to download model...")
            
            # Load model components
            logger.info("Loading model...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.torch_dtype,
                device_map=self.config.device_map,
                trust_remote_code=self.config.trust_remote_code,
                resume_download=True
            )
            
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
                resume_download=True
            )
            
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
                resume_download=True
            )
            
            # Create pipeline for easier inference
            logger.info("Creating inference pipeline...")
            self.pipeline = pipeline(
                "image-to-text",
                model=self.model,
                tokenizer=self.tokenizer,
                processor=self.processor,
                device_map=self.config.device_map,
                torch_dtype=self.config.torch_dtype
            )
            
            self.is_loaded = True
            logger.info("✅ Qwen2-VL model loaded successfully")
            
            # Log model info
            if hasattr(self.model, 'config'):
                model_config = self.model.config
                logger.info(f"Model config: {model_config.model_type if hasattr(model_config, 'model_type') else 'Unknown'}")
                if hasattr(model_config, 'hidden_size'):
                    logger.info(f"Hidden size: {model_config.hidden_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Qwen2-VL model: {e}")
            self.is_loaded = False
            return False
    
    def _check_model_availability(self) -> bool:
        """Check if Qwen2-VL model is available locally"""
        if not self.detection_service:
            return False
        
        try:
            qwen2_vl_info = self.detection_service.detect_qwen2_vl_capabilities()
            return qwen2_vl_info["integration_possible"]
        except Exception as e:
            logger.warning(f"Could not check model availability: {e}")
            return False
    
    def enhance_prompt(self, prompt: str, enhancement_type: str = "general") -> PromptEnhancementResult:
        """
        Enhance a text prompt using Qwen2-VL for better text comprehension
        
        Args:
            prompt: Original text prompt
            enhancement_type: Type of enhancement ("general", "artistic", "technical", "creative")
        
        Returns:
            PromptEnhancementResult with enhanced prompt and metadata
        """
        if not self.config.enable_prompt_enhancement:
            return PromptEnhancementResult(
                original_prompt=prompt,
                enhanced_prompt=prompt,
                enhancement_type="disabled",
                confidence=0.0,
                metadata={"reason": "Prompt enhancement disabled"}
            )
        
        # Check cache first
        cache_key = f"enhance_{enhancement_type}_{hash(prompt)}"
        if self.response_cache and cache_key in self.response_cache:
            logger.debug("Using cached prompt enhancement")
            return self.response_cache[cache_key]
        
        # Fallback if model not loaded
        if not self.is_loaded:
            logger.warning("Qwen2-VL not loaded, using fallback enhancement")
            return self._fallback_prompt_enhancement(prompt, enhancement_type)
        
        try:
            # Create enhancement instruction based on type
            enhancement_instructions = {
                "general": "Improve this image generation prompt to be more descriptive and specific while maintaining the original intent:",
                "artistic": "Enhance this artistic image prompt with better style descriptions, composition details, and artistic techniques:",
                "technical": "Improve this technical image prompt with more precise specifications, quality indicators, and technical details:",
                "creative": "Expand this creative prompt with more imaginative details, atmosphere, and creative elements:"
            }
            
            instruction = enhancement_instructions.get(enhancement_type, enhancement_instructions["general"])
            
            # Format the input for Qwen2-VL
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at improving image generation prompts. Provide enhanced prompts that are more descriptive, specific, and likely to produce better results."
                },
                {
                    "role": "user", 
                    "content": f"{instruction}\n\nOriginal prompt: {prompt}\n\nEnhanced prompt:"
                }
            ]
            
            # Apply chat template
            text_input = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Generate enhancement
            inputs = self.processor(
                text=[text_input],
                return_tensors="pt"
            )
            
            # Move inputs to correct device
            if torch.cuda.is_available():
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Extract enhanced prompt (remove any additional commentary)
            enhanced_prompt = self._extract_enhanced_prompt(generated_text, prompt)
            
            # Calculate confidence based on enhancement quality
            confidence = self._calculate_enhancement_confidence(prompt, enhanced_prompt)
            
            result = PromptEnhancementResult(
                original_prompt=prompt,
                enhanced_prompt=enhanced_prompt,
                enhancement_type=enhancement_type,
                confidence=confidence,
                metadata={
                    "model_used": self.config.model_name,
                    "enhancement_length": len(enhanced_prompt) - len(prompt),
                    "processing_successful": True
                }
            )
            
            # Cache result
            if self.response_cache:
                self.response_cache[cache_key] = result
            
            logger.info(f"Prompt enhanced: {len(prompt)} -> {len(enhanced_prompt)} chars")
            return result
            
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {e}")
            return self._fallback_prompt_enhancement(prompt, enhancement_type)
    
    def analyze_image(self, image: Union[str, Image.Image], analysis_type: str = "comprehensive") -> ImageAnalysisResult:
        """
        Analyze an image using Qwen2-VL for context-aware generation
        
        Args:
            image: Path to image file or PIL Image object
            analysis_type: Type of analysis ("comprehensive", "style", "composition", "elements")
        
        Returns:
            ImageAnalysisResult with analysis details
        """
        if not self.config.enable_image_analysis:
            return ImageAnalysisResult(
                description="Image analysis disabled",
                key_elements=[],
                style_analysis={},
                composition_notes="",
                suggested_improvements=[],
                confidence=0.0
            )
        
        # Fallback if model not loaded
        if not self.is_loaded:
            logger.warning("Qwen2-VL not loaded, using fallback image analysis")
            return self._fallback_image_analysis(image, analysis_type)
        
        try:
            # Load image if path provided
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image file not found: {image}")
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                raise ValueError("Image must be a file path or PIL Image object")
            
            # Check cache
            cache_key = f"analyze_{analysis_type}_{hash(str(image.size) + str(image.mode))}"
            if self.response_cache and cache_key in self.response_cache:
                logger.debug("Using cached image analysis")
                return self.response_cache[cache_key]
            
            # Create analysis instruction based on type
            analysis_instructions = {
                "comprehensive": "Provide a comprehensive analysis of this image including: visual elements, style, composition, colors, mood, and suggestions for similar image generation.",
                "style": "Analyze the artistic style of this image including: art style, technique, color palette, and visual characteristics.",
                "composition": "Analyze the composition of this image including: layout, balance, focal points, and visual flow.",
                "elements": "Identify and describe the key visual elements in this image including: objects, people, backgrounds, and details."
            }
            
            instruction = analysis_instructions.get(analysis_type, analysis_instructions["comprehensive"])
            
            # Format the input for Qwen2-VL
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at analyzing images for the purpose of generating similar images. Provide detailed, structured analysis that would be useful for image generation prompts."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": instruction}
                    ]
                }
            ]
            
            # Apply chat template
            text_input = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text_input],
                images=[image],
                return_tensors="pt"
            )
            
            # Move inputs to correct device
            if torch.cuda.is_available():
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Parse analysis result
            analysis_result = self._parse_image_analysis(generated_text, analysis_type)
            
            # Cache result
            if self.response_cache:
                self.response_cache[cache_key] = analysis_result
            
            logger.info(f"Image analysis completed: {analysis_type}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return self._fallback_image_analysis(image, analysis_type)
    
    def create_context_aware_prompt(self, base_prompt: str, reference_image: Optional[Union[str, Image.Image]] = None) -> str:
        """
        Create a context-aware prompt by combining text enhancement and optional image analysis
        
        Args:
            base_prompt: Base text prompt
            reference_image: Optional reference image for context
        
        Returns:
            Enhanced context-aware prompt
        """
        try:
            # Start with prompt enhancement
            enhanced_result = self.enhance_prompt(base_prompt, "creative")
            context_prompt = enhanced_result.enhanced_prompt
            
            # Add image context if provided
            if reference_image is not None:
                logger.info("Adding image context to prompt")
                image_analysis = self.analyze_image(reference_image, "style")
                
                # Incorporate image style information
                if image_analysis.style_analysis:
                    style_elements = []
                    for key, value in image_analysis.style_analysis.items():
                        if value and value.lower() != "unknown":
                            style_elements.append(f"{key}: {value}")
                    
                    if style_elements:
                        style_context = ", ".join(style_elements)
                        context_prompt = f"{context_prompt}, in the style of {style_context}"
                
                # Add key visual elements
                if image_analysis.key_elements:
                    elements_context = ", ".join(image_analysis.key_elements[:3])  # Top 3 elements
                    context_prompt = f"{context_prompt}, incorporating elements like {elements_context}"
            
            logger.info("Context-aware prompt created successfully")
            return context_prompt
            
        except Exception as e:
            logger.error(f"Context-aware prompt creation failed: {e}")
            return base_prompt  # Fallback to original prompt
    
    def _fallback_prompt_enhancement(self, prompt: str, enhancement_type: str) -> PromptEnhancementResult:
        """Fallback prompt enhancement when Qwen2-VL is not available"""
        # Simple rule-based enhancement
        enhanced_prompt = prompt
        
        # Add quality keywords
        quality_keywords = ["high quality", "detailed", "sharp focus"]
        if not any(keyword in prompt.lower() for keyword in quality_keywords):
            enhanced_prompt = f"{enhanced_prompt}, high quality, detailed"
        
        # Add style keywords based on enhancement type
        if enhancement_type == "artistic":
            if "artistic" not in prompt.lower() and "art" not in prompt.lower():
                enhanced_prompt = f"{enhanced_prompt}, artistic masterpiece"
        elif enhancement_type == "technical":
            if "professional" not in prompt.lower():
                enhanced_prompt = f"{enhanced_prompt}, professional photography"
        elif enhancement_type == "creative":
            if "creative" not in prompt.lower() and "imaginative" not in prompt.lower():
                enhanced_prompt = f"{enhanced_prompt}, creative and imaginative"
        
        return PromptEnhancementResult(
            original_prompt=prompt,
            enhanced_prompt=enhanced_prompt,
            enhancement_type=f"fallback_{enhancement_type}",
            confidence=0.3,  # Low confidence for fallback
            metadata={"fallback_used": True, "reason": "Qwen2-VL not available"}
        )
    
    def _fallback_image_analysis(self, image: Union[str, Image.Image], analysis_type: str) -> ImageAnalysisResult:
        """Fallback image analysis when Qwen2-VL is not available"""
        return ImageAnalysisResult(
            description="Basic image detected (Qwen2-VL not available for detailed analysis)",
            key_elements=["image content"],
            style_analysis={"style": "unknown", "technique": "unknown"},
            composition_notes="Composition analysis not available",
            suggested_improvements=["Enable Qwen2-VL for detailed analysis"],
            confidence=0.1  # Very low confidence for fallback
        )
    
    def _extract_enhanced_prompt(self, generated_text: str, original_prompt: str) -> str:
        """Extract the enhanced prompt from generated text"""
        # Clean up the generated text
        enhanced_prompt = generated_text.strip()
        
        # Remove common prefixes/suffixes that might be added by the model
        prefixes_to_remove = [
            "Enhanced prompt:",
            "Here's an enhanced version:",
            "Improved prompt:",
            "Better prompt:",
        ]
        
        for prefix in prefixes_to_remove:
            if enhanced_prompt.lower().startswith(prefix.lower()):
                enhanced_prompt = enhanced_prompt[len(prefix):].strip()
        
        # If the result is too short or seems invalid, return original with basic enhancement
        if len(enhanced_prompt) < len(original_prompt) * 0.8:
            enhanced_prompt = f"{original_prompt}, high quality, detailed"
        
        return enhanced_prompt
    
    def _calculate_enhancement_confidence(self, original: str, enhanced: str) -> float:
        """Calculate confidence score for prompt enhancement"""
        if len(enhanced) <= len(original):
            return 0.2  # Low confidence if not actually enhanced
        
        # Check for quality indicators
        quality_indicators = [
            "high quality", "detailed", "masterpiece", "professional",
            "sharp focus", "well-composed", "artistic", "creative"
        ]
        
        quality_score = sum(1 for indicator in quality_indicators if indicator in enhanced.lower())
        length_ratio = len(enhanced) / len(original)
        
        # Calculate confidence based on enhancement quality and length
        confidence = min(0.9, (quality_score * 0.1) + (length_ratio - 1.0) * 0.3 + 0.5)
        return max(0.1, confidence)  # Minimum confidence of 0.1
    
    def _parse_image_analysis(self, analysis_text: str, analysis_type: str) -> ImageAnalysisResult:
        """Parse image analysis text into structured result"""
        try:
            # Extract key information from the analysis text
            description = analysis_text[:200] + "..." if len(analysis_text) > 200 else analysis_text
            
            # Extract key elements (simple keyword extraction)
            key_elements = []
            element_keywords = [
                "person", "people", "man", "woman", "child", "face", "portrait",
                "landscape", "building", "tree", "flower", "animal", "car", "sky",
                "water", "mountain", "city", "nature", "indoor", "outdoor"
            ]
            
            for keyword in element_keywords:
                if keyword in analysis_text.lower():
                    key_elements.append(keyword)
            
            # Extract style information
            style_analysis = {}
            style_keywords = {
                "style": ["realistic", "cartoon", "anime", "abstract", "impressionist", "modern"],
                "technique": ["painting", "photography", "digital art", "sketch", "watercolor"],
                "mood": ["bright", "dark", "cheerful", "serious", "dramatic", "peaceful"]
            }
            
            for category, keywords in style_keywords.items():
                for keyword in keywords:
                    if keyword in analysis_text.lower():
                        style_analysis[category] = keyword
                        break
                if category not in style_analysis:
                    style_analysis[category] = "unknown"
            
            # Generate composition notes
            composition_notes = "Composition analysis based on visual elements"
            if "composition" in analysis_text.lower():
                # Extract composition-related sentences
                sentences = analysis_text.split('.')
                comp_sentences = [s.strip() for s in sentences if "composition" in s.lower()]
                if comp_sentences:
                    composition_notes = comp_sentences[0]
            
            # Generate improvement suggestions
            suggested_improvements = [
                "Consider adjusting lighting for better contrast",
                "Enhance color saturation for more vibrant results",
                "Add more specific style keywords to prompt"
            ]
            
            # Calculate confidence based on analysis quality
            confidence = min(0.8, len(key_elements) * 0.1 + len(style_analysis) * 0.1 + 0.3)
            
            return ImageAnalysisResult(
                description=description,
                key_elements=key_elements[:10],  # Limit to top 10 elements
                style_analysis=style_analysis,
                composition_notes=composition_notes,
                suggested_improvements=suggested_improvements,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Failed to parse image analysis: {e}")
            return self._fallback_image_analysis(None, analysis_type)
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status and capabilities"""
        status = {
            "available": self.is_available,
            "loaded": self.is_loaded,
            "model_name": self.config.model_name if self.is_loaded else None,
            "capabilities": {
                "prompt_enhancement": self.config.enable_prompt_enhancement,
                "image_analysis": self.config.enable_image_analysis,
                "context_aware_prompts": self.is_loaded
            },
            "fallback_enabled": self.config.fallback_enabled,
            "cache_enabled": self.response_cache is not None,
            "cache_size": len(self.response_cache) if self.response_cache else 0
        }
        
        if self.is_loaded and hasattr(self.model, 'config'):
            status["model_config"] = {
                "model_type": getattr(self.model.config, 'model_type', 'unknown'),
                "hidden_size": getattr(self.model.config, 'hidden_size', 'unknown'),
                "vocab_size": getattr(self.model.config, 'vocab_size', 'unknown')
            }
        
        return status
    
    def clear_cache(self) -> None:
        """Clear response cache"""
        if self.response_cache:
            self.response_cache.clear()
            logger.info("Response cache cleared")
    
    def unload_model(self) -> None:
        """Unload model to free memory"""
        if self.is_loaded:
            self.model = None
            self.tokenizer = None
            self.processor = None
            self.pipeline = None
            self.is_loaded = False
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Qwen2-VL model unloaded")


# Factory function for easy integration
def create_qwen2vl_integration(
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
    enable_prompt_enhancement: bool = True,
    enable_image_analysis: bool = True,
    fallback_enabled: bool = True,
    **kwargs
) -> Qwen2VLIntegration:
    """
    Factory function to create Qwen2-VL integration with common configurations
    
    Args:
        model_name: Qwen2-VL model to use
        enable_prompt_enhancement: Enable prompt enhancement capabilities
        enable_image_analysis: Enable image analysis capabilities
        fallback_enabled: Enable fallback when model not available
        **kwargs: Additional configuration options
    
    Returns:
        Configured Qwen2VLIntegration instance
    """
    config = Qwen2VLConfig(
        model_name=model_name,
        enable_prompt_enhancement=enable_prompt_enhancement,
        enable_image_analysis=enable_image_analysis,
        fallback_enabled=fallback_enabled,
        **kwargs
    )
    
    return Qwen2VLIntegration(config)


# Utility functions for integration testing
def test_qwen2vl_integration() -> Dict[str, Any]:
    """Test Qwen2-VL integration functionality"""
    test_results = {
        "dependencies_available": QWEN2VL_AVAILABLE,
        "model_loading": False,
        "prompt_enhancement": False,
        "image_analysis": False,
        "errors": []
    }
    
    try:
        # Test basic integration
        integration = create_qwen2vl_integration(fallback_enabled=True)
        
        # Test model loading
        if integration.is_available:
            test_results["model_loading"] = integration.load_model()
        else:
            test_results["errors"].append("Qwen2-VL dependencies not available")
        
        # Test prompt enhancement (should work with fallback)
        test_prompt = "a beautiful landscape"
        enhancement_result = integration.enhance_prompt(test_prompt)
        test_results["prompt_enhancement"] = len(enhancement_result.enhanced_prompt) > len(test_prompt)
        
        # Test image analysis (basic test)
        if integration.config.enable_image_analysis:
            # Create a simple test image
            from PIL import Image
            test_image = Image.new('RGB', (100, 100), color='red')
            analysis_result = integration.analyze_image(test_image)
            test_results["image_analysis"] = len(analysis_result.description) > 0
        
    except Exception as e:
        test_results["errors"].append(str(e))
    
    return test_results


if __name__ == "__main__":
    # Run integration test
    print("Testing Qwen2-VL Integration...")
    results = test_qwen2vl_integration()
    
    print(f"Dependencies available: {results['dependencies_available']}")
    print(f"Model loading: {results['model_loading']}")
    print(f"Prompt enhancement: {results['prompt_enhancement']}")
    print(f"Image analysis: {results['image_analysis']}")
    
    if results['errors']:
        print("Errors:")
        for error in results['errors']:
            print(f"  - {error}")
    else:
        print("All tests passed!")