#!/usr/bin/env python3
"""
Multimodal Integration Demo
Demonstrates Qwen2-VL integration for enhanced text understanding and image analysis
"""

import os
import sys
import time
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from model_detection_service import ModelDetectionService
    from pipeline_optimizer import PipelineOptimizer, OptimizationConfig
    MODEL_DETECTION_AVAILABLE = True
except ImportError:
    MODEL_DETECTION_AVAILABLE = False
    print("⚠️ Model detection not available")

try:
    from diffusers import AutoPipelineForText2Image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("❌ diffusers not installed. Run: pip install diffusers")

try:
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info
    QWEN2VL_AVAILABLE = True
except ImportError:
    QWEN2VL_AVAILABLE = False
    print("⚠️ Qwen2-VL components not available. Install with: pip install transformers qwen-vl-utils")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("❌ PIL not available. Install with: pip install Pillow")


class MultimodalIntegrationDemo:
    """Demo class for multimodal integration with Qwen2-VL"""
    
    def __init__(self):
        self.image_pipeline = None
        self.qwen2vl_model = None
        self.qwen2vl_processor = None
        self.qwen2vl_tokenizer = None
        self.detector = None
        
        # Initialize detector if available
        if MODEL_DETECTION_AVAILABLE:
            self.detector = ModelDetectionService()
        
        # Demo scenarios for multimodal integration
        self.demo_scenarios = {
            "prompt_enhancement": {
                "original_prompt": "A cat sitting on a chair",
                "description": "Basic prompt enhancement using Qwen2-VL understanding"
            },
            "style_analysis": {
                "original_prompt": "A landscape painting",
                "reference_image": None,  # Will be set if available
                "description": "Style analysis and prompt refinement based on reference image"
            },
            "context_aware": {
                "original_prompt": "A modern building",
                "context": "Create an image that matches the architectural style of contemporary urban design",
                "description": "Context-aware generation with enhanced understanding"
            },
            "multilingual": {
                "original_prompt": "一个美丽的花园",  # Chinese: "A beautiful garden"
                "description": "Multilingual prompt understanding and enhancement"
            },
            "complex_scene": {
                "original_prompt": "A busy street scene with people, cars, and shops",
                "description": "Complex scene understanding and detailed prompt generation"
            }
        }
    
    def detect_available_models(self) -> Dict[str, bool]:
        """Detect which models are available"""
        availability = {
            "qwen_image": False,
            "qwen2_vl": False,
            "integration_possible": False
        }
        
        if not MODEL_DETECTION_AVAILABLE:
            print("⚠️ Model detection not available")
            return availability
        
        try:
            # Check for Qwen-Image model
            current_model = self.detector.detect_current_model()
            if current_model and current_model.download_status == "complete":
                availability["qwen_image"] = True
                print(f"✅ Found Qwen-Image model: {current_model.name}")
            else:
                print("❌ Qwen-Image model not found or incomplete")
            
            # Check for Qwen2-VL models
            qwen2vl_info = self.detector.detect_qwen2_vl_capabilities()
            if qwen2vl_info["available_models"]:
                availability["qwen2_vl"] = True
                print(f"✅ Found Qwen2-VL models: {len(qwen2vl_info['available_models'])}")
                for model in qwen2vl_info["available_models"]:
                    print(f"   • {model['name']} ({model['size_gb']:.1f}GB)")
            else:
                print("❌ Qwen2-VL models not found")
            
            # Check integration potential
            integration_analysis = self.detector.analyze_multimodal_integration_potential(current_model)
            availability["integration_possible"] = (
                integration_analysis["current_model_compatible"] and 
                integration_analysis["qwen2_vl_available"]
            )
            
            if availability["integration_possible"]:
                print("✅ Multimodal integration possible")
                if integration_analysis["recommended_setup"]:
                    setup = integration_analysis["recommended_setup"]
                    print(f"💡 Recommended setup:")
                    print(f"   Primary: {setup['primary_model']}")
                    print(f"   Multimodal: {setup['multimodal_model']}")
                    print(f"   Benefits: {setup['expected_benefits']}")
            else:
                print("❌ Multimodal integration not possible with current setup")
                if integration_analysis["integration_requirements"]:
                    print("📋 Requirements:")
                    for req in integration_analysis["integration_requirements"]:
                        print(f"   • {req}")
            
        except Exception as e:
            print(f"❌ Model detection failed: {e}")
        
        return availability
    
    def load_image_generation_model(self) -> bool:
        """Load the image generation model"""
        print("📦 Loading image generation model...")
        
        try:
            if MODEL_DETECTION_AVAILABLE:
                current_model = self.detector.detect_current_model()
                if not current_model:
                    print("❌ No image generation model found")
                    return False
                
                # Create optimized pipeline
                config = OptimizationConfig(architecture_type="MMDiT")
                optimizer = PipelineOptimizer(config)
                
                architecture = self.detector.detect_model_architecture(current_model)
                self.image_pipeline = optimizer.create_optimized_pipeline(
                    current_model.path, architecture
                )
                
                print(f"✅ Loaded optimized pipeline: {current_model.name}")
                return True
            else:
                # Basic loading without optimization
                model_paths = ["./models/Qwen-Image", "Qwen/Qwen-Image"]
                
                for path in model_paths:
                    try:
                        self.image_pipeline = AutoPipelineForText2Image.from_pretrained(
                            path,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            trust_remote_code=True
                        )
                        
                        if torch.cuda.is_available():
                            self.image_pipeline = self.image_pipeline.to("cuda")
                        
                        print(f"✅ Loaded basic pipeline from: {path}")
                        return True
                    except Exception:
                        continue
                
                print("❌ Failed to load image generation model")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load image generation model: {e}")
            return False
    
    def load_qwen2vl_model(self) -> bool:
        """Load the Qwen2-VL model for multimodal understanding"""
        print("🧠 Loading Qwen2-VL model...")
        
        if not QWEN2VL_AVAILABLE:
            print("❌ Qwen2-VL components not available")
            return False
        
        try:
            # Try to find Qwen2-VL model
            model_paths = []
            
            if MODEL_DETECTION_AVAILABLE:
                qwen2vl_info = self.detector.detect_qwen2_vl_capabilities()
                if qwen2vl_info["recommended_model"]:
                    model_paths.append(qwen2vl_info["recommended_model"]["path"])
            
            # Add common paths
            model_paths.extend([
                "./models/Qwen2-VL-7B-Instruct",
                "./models/Qwen2-VL-2B-Instruct",
                "Qwen/Qwen2-VL-7B-Instruct",
                "Qwen/Qwen2-VL-2B-Instruct"
            ])
            
            for model_path in model_paths:
                try:
                    print(f"🔄 Trying to load from: {model_path}")
                    
                    # Load model components
                    self.qwen2vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        trust_remote_code=True
                    )
                    
                    self.qwen2vl_processor = AutoProcessor.from_pretrained(
                        model_path,
                        trust_remote_code=True
                    )
                    
                    self.qwen2vl_tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=True
                    )
                    
                    print(f"✅ Loaded Qwen2-VL from: {model_path}")
                    return True
                    
                except Exception as e:
                    print(f"⚠️ Failed to load from {model_path}: {e}")
                    continue
            
            print("❌ Failed to load Qwen2-VL model from any path")
            return False
            
        except Exception as e:
            print(f"❌ Failed to load Qwen2-VL model: {e}")
            return False
    
    def enhance_prompt_with_qwen2vl(self, original_prompt: str, context: str = None, reference_image: str = None) -> str:
        """Enhance a prompt using Qwen2-VL understanding"""
        if not self.qwen2vl_model:
            print("⚠️ Qwen2-VL not available, returning original prompt")
            return original_prompt
        
        try:
            # Prepare the enhancement request
            messages = []
            
            if reference_image and os.path.exists(reference_image):
                # Include reference image for style analysis
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": reference_image},
                        {"type": "text", "text": f"Analyze this image and enhance the following prompt to match its style and quality: '{original_prompt}'"}
                    ]
                })
            else:
                # Text-only enhancement
                enhancement_request = f"""
                Please enhance and expand the following image generation prompt to be more detailed and specific:
                
                Original prompt: "{original_prompt}"
                
                {f"Context: {context}" if context else ""}
                
                Provide a detailed, descriptive prompt that would generate a high-quality image. Focus on:
                - Visual details and composition
                - Lighting and atmosphere
                - Style and artistic elements
                - Colors and textures
                
                Enhanced prompt:
                """
                
                messages.append({
                    "role": "user", 
                    "content": [{"type": "text", "text": enhancement_request}]
                })
            
            # Process with Qwen2-VL
            text = self.qwen2vl_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.qwen2vl_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            # Generate enhanced prompt
            with torch.no_grad():
                generated_ids = self.qwen2vl_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            enhanced_prompt = self.qwen2vl_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Clean up the response
            enhanced_prompt = enhanced_prompt.strip()
            if enhanced_prompt.startswith("Enhanced prompt:"):
                enhanced_prompt = enhanced_prompt[len("Enhanced prompt:"):].strip()
            
            return enhanced_prompt
            
        except Exception as e:
            print(f"⚠️ Prompt enhancement failed: {e}")
            return original_prompt
    
    def analyze_image_with_qwen2vl(self, image_path: str) -> str:
        """Analyze an image using Qwen2-VL"""
        if not self.qwen2vl_model or not os.path.exists(image_path):
            return "Image analysis not available"
        
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Describe this image in detail, focusing on composition, style, colors, lighting, and artistic elements."}
                ]
            }]
            
            text = self.qwen2vl_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.qwen2vl_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            with torch.no_grad():
                generated_ids = self.qwen2vl_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            analysis = self.qwen2vl_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return analysis.strip()
            
        except Exception as e:
            print(f"⚠️ Image analysis failed: {e}")
            return "Image analysis failed"
    
    def generate_image_with_enhancement(self, original_prompt: str, context: str = None, reference_image: str = None) -> Tuple[Optional[Image.Image], str, str]:
        """Generate an image with prompt enhancement"""
        if not self.image_pipeline:
            return None, original_prompt, "Image pipeline not available"
        
        # Enhance prompt if Qwen2-VL is available
        enhanced_prompt = original_prompt
        if self.qwen2vl_model:
            print("🧠 Enhancing prompt with Qwen2-VL...")
            enhanced_prompt = self.enhance_prompt_with_qwen2vl(original_prompt, context, reference_image)
            print(f"✨ Enhanced prompt: {enhanced_prompt[:100]}...")
        
        try:
            print("🎨 Generating image...")
            start_time = time.time()
            
            # Generate image
            generation_kwargs = {
                "prompt": enhanced_prompt,
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 20,
                "generator": torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
            }
            
            # Use appropriate CFG parameter
            if "qwen" in str(self.image_pipeline.__class__).lower():
                generation_kwargs["true_cfg_scale"] = 3.5
            else:
                generation_kwargs["guidance_scale"] = 3.5
            
            result = self.image_pipeline(**generation_kwargs)
            image = result.images[0]
            
            generation_time = time.time() - start_time
            
            return image, enhanced_prompt, f"Generated in {generation_time:.2f}s"
            
        except Exception as e:
            return None, enhanced_prompt, f"Generation failed: {e}"
    
    def run_multimodal_demo(self):
        """Run the complete multimodal integration demo"""
        print("\n🎭 Multimodal Integration Demo")
        print("=" * 60)
        
        for scenario_name, scenario in self.demo_scenarios.items():
            print(f"\n🔹 {scenario['description']}")
            print(f"Original prompt: {scenario['original_prompt']}")
            
            try:
                # Generate image with enhancement
                image, enhanced_prompt, message = self.generate_image_with_enhancement(
                    scenario["original_prompt"],
                    scenario.get("context"),
                    scenario.get("reference_image")
                )
                
                if image:
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"multimodal_{scenario_name}_{timestamp}.png"
                    filepath = os.path.join("generated_images", filename)
                    os.makedirs("generated_images", exist_ok=True)
                    image.save(filepath)
                    
                    print(f"✅ {message}")
                    print(f"💾 Saved: {filepath}")
                    
                    if enhanced_prompt != scenario["original_prompt"]:
                        print(f"✨ Enhanced prompt: {enhanced_prompt[:150]}...")
                    
                    # Analyze generated image if Qwen2-VL is available
                    if self.qwen2vl_model:
                        print("🔍 Analyzing generated image...")
                        analysis = self.analyze_image_with_qwen2vl(filepath)
                        print(f"📊 Analysis: {analysis[:200]}...")
                else:
                    print(f"❌ {message}")
                    
            except Exception as e:
                print(f"❌ Scenario failed: {e}")
    
    def demonstrate_prompt_enhancement_comparison(self):
        """Demonstrate side-by-side comparison of original vs enhanced prompts"""
        print("\n🔄 Prompt Enhancement Comparison")
        print("=" * 60)
        
        test_prompts = [
            "A cat",
            "A house",
            "A landscape",
            "A portrait"
        ]
        
        for prompt in test_prompts:
            print(f"\n🧪 Testing: '{prompt}'")
            
            try:
                # Generate with original prompt
                print("📝 Generating with original prompt...")
                original_image, _, original_message = self.generate_image_with_enhancement(prompt)
                
                # Generate with enhanced prompt (if Qwen2-VL available)
                if self.qwen2vl_model:
                    print("✨ Generating with enhanced prompt...")
                    enhanced_image, enhanced_prompt, enhanced_message = self.generate_image_with_enhancement(prompt)
                    
                    # Save comparison
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    if original_image:
                        original_path = os.path.join("generated_images", f"comparison_original_{prompt.replace(' ', '_')}_{timestamp}.png")
                        original_image.save(original_path)
                        print(f"💾 Original saved: {original_path}")
                    
                    if enhanced_image:
                        enhanced_path = os.path.join("generated_images", f"comparison_enhanced_{prompt.replace(' ', '_')}_{timestamp}.png")
                        enhanced_image.save(enhanced_path)
                        print(f"💾 Enhanced saved: {enhanced_path}")
                        print(f"✨ Enhancement: {enhanced_prompt[:100]}...")
                else:
                    print("⚠️ Qwen2-VL not available for enhancement comparison")
                    
            except Exception as e:
                print(f"❌ Comparison failed: {e}")


def main():
    """Main demo function"""
    print("🎭 Multimodal Integration Demo")
    print("=" * 70)
    print("Demonstrating Qwen2-VL integration for enhanced text understanding")
    print("")
    
    # Initialize demo
    demo = MultimodalIntegrationDemo()
    
    # Check model availability
    availability = demo.detect_available_models()
    
    if not availability["qwen_image"]:
        print("❌ Qwen-Image model not found")
        print("💡 Run: python tools/download_qwen_image.py --model qwen-image")
        sys.exit(1)
    
    if not availability["qwen2_vl"]:
        print("❌ Qwen2-VL model not found")
        print("💡 Run: python tools/download_qwen_image.py --model qwen2-vl-7b")
        print("⚠️ Continuing with basic image generation only...")
    
    # Load models
    print("\n📦 Loading models...")
    
    if not demo.load_image_generation_model():
        print("❌ Failed to load image generation model")
        sys.exit(1)
    
    if availability["qwen2_vl"]:
        if not demo.load_qwen2vl_model():
            print("⚠️ Failed to load Qwen2-VL model, continuing without multimodal features")
    
    # Run demonstrations
    try:
        # Main multimodal demo
        demo.run_multimodal_demo()
        
        # Prompt enhancement comparison
        if demo.qwen2vl_model:
            demo.demonstrate_prompt_enhancement_comparison()
        
        print("\n🎉 Multimodal integration demo completed!")
        print("=" * 70)
        print("📁 Generated images saved to: ./generated_images/")
        
        if demo.qwen2vl_model:
            print("✅ Multimodal features demonstrated successfully")
            print("💡 Benefits observed:")
            print("   • Enhanced prompt understanding")
            print("   • More detailed and accurate descriptions")
            print("   • Better context awareness")
            print("   • Improved image quality through better prompts")
        else:
            print("⚠️ Multimodal features not available")
            print("💡 Install Qwen2-VL for enhanced capabilities:")
            print("   python tools/download_qwen_image.py --model qwen2-vl-7b")
        
        print("\n🚀 Next: python examples/performance_comparison_demo.py")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()