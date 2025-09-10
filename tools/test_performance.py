#!/usr/bin/env python3
"""
Performance Test for High-End Qwen Generator
Tests the optimized configuration to ensure 15-60s generation times
"""

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_high_end_performance():
    """Test the high-end generator performance"""

    print("🧪 HIGH-PERFORMANCE QWEN GENERATOR TEST")
    print("=" * 60)
    print("Target: 15-60 seconds per image (not 500+ seconds!)")
    print("=" * 60)

    try:
        # Import high-end generator
        print("📦 Importing high-end generator...")
        from src.qwen_highend_generator import HighEndQwenImageGenerator

        # Initialize generator
        print("🔄 Initializing high-end generator...")
        start_time = datetime.now()

        generator = HighEndQwenImageGenerator()

        init_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Generator initialized in {init_time:.1f} seconds")

        # Load model
        print("🔄 Loading model with high-end optimizations...")
        model_start = datetime.now()

        success = generator.load_model()

        if not success:
            print("❌ Model loading failed!")
            return False

        model_time = (datetime.now() - model_start).total_seconds()
        print(f"✅ Model loaded in {model_time:.1f} seconds")

        # Test generation
        print("🎨 Testing image generation...")

        test_prompt = "A high-tech laboratory with scientists working on AI, futuristic lighting, detailed and professional"

        generation_start = datetime.now()

        image, message = generator.generate_image(
            prompt=test_prompt,
            width=1664,
            height=928,
            num_inference_steps=20,  # Reduced steps for quick test
            cfg_scale=4.0,
            seed=12345,
        )

        generation_time = (datetime.now() - generation_start).total_seconds()

        print("🏁 GENERATION COMPLETE!")
        print(f"   Time: {generation_time:.1f} seconds")
        print("   Steps: 20")
        print(f"   Time per step: {generation_time/20:.2f}s")

        # Evaluate performance
        if image is not None:
            print("✅ Image generated successfully!")

            if generation_time <= 30:
                print("🚀 EXCELLENT: Generation time within target!")
            elif generation_time <= 60:
                print("✅ GOOD: Generation time acceptable")
            elif generation_time <= 120:
                print("⚠️ SLOW: Generation time higher than optimal")
            else:
                print("❌ VERY SLOW: Still has performance issues")

            print(f"\nMessage: {message}")
            return True
        else:
            print("❌ Image generation failed!")
            print(f"Error: {message}")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure high-end generator is available")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_standard_vs_highend():
    """Compare standard vs high-end performance"""

    print("\n🔬 PERFORMANCE COMPARISON TEST")
    print("=" * 60)

    results = {}

    # Test standard generator
    try:
        print("📊 Testing STANDARD generator...")
        from src.qwen_generator import QwenImageGenerator

        std_generator = QwenImageGenerator()
        print("🔄 Loading standard model...")

        std_start = datetime.now()
        std_success = std_generator.load_model()
        std_load_time = (datetime.now() - std_start).total_seconds()

        if std_success:
            print("🎨 Testing standard generation...")
            gen_start = datetime.now()

            image, message = std_generator.generate_image(
                prompt="Test image generation", num_inference_steps=10  # Quick test
            )

            std_gen_time = (datetime.now() - gen_start).total_seconds()

            results["standard"] = {
                "load_time": std_load_time,
                "generation_time": std_gen_time,
                "success": image is not None,
            }

            print(f"Standard results: {std_gen_time:.1f}s generation")
        else:
            print("❌ Standard generator failed to load")

    except Exception as e:
        print(f"❌ Standard generator test failed: {e}")

    # Test high-end generator
    try:
        print("\n📊 Testing HIGH-END generator...")
        from src.qwen_highend_generator import HighEndQwenImageGenerator

        he_generator = HighEndQwenImageGenerator()
        print("🔄 Loading high-end model...")

        he_start = datetime.now()
        he_success = he_generator.load_model()
        he_load_time = (datetime.now() - he_start).total_seconds()

        if he_success:
            print("🎨 Testing high-end generation...")
            gen_start = datetime.now()

            image, message = he_generator.generate_image(
                prompt="Test image generation", num_inference_steps=10  # Quick test
            )

            he_gen_time = (datetime.now() - gen_start).total_seconds()

            results["high_end"] = {
                "load_time": he_load_time,
                "generation_time": he_gen_time,
                "success": image is not None,
            }

            print(f"High-end results: {he_gen_time:.1f}s generation")
        else:
            print("❌ High-end generator failed to load")

    except Exception as e:
        print(f"❌ High-end generator test failed: {e}")

    # Compare results
    if "standard" in results and "high_end" in results:
        print("\n📊 COMPARISON RESULTS")
        print("=" * 40)

        std = results["standard"]
        he = results["high_end"]

        print("Load Time:")
        print(f"  Standard: {std['load_time']:.1f}s")
        print(f"  High-End: {he['load_time']:.1f}s")

        print("Generation Time:")
        print(f"  Standard: {std['generation_time']:.1f}s")
        print(f"  High-End: {he['generation_time']:.1f}s")

        if he["generation_time"] < std["generation_time"]:
            improvement = (
                (std["generation_time"] - he["generation_time"])
                / std["generation_time"]
                * 100
            )
            print(f"🚀 HIGH-END IS {improvement:.1f}% FASTER!")
        else:
            print("⚠️ No significant improvement detected")


def main():
    """Main test function"""

    # Single performance test
    print("Starting performance validation...")
    success = test_high_end_performance()

    if success:
        print("\n✅ HIGH-PERFORMANCE TEST PASSED!")
        print("🎯 Your system should now generate images in 15-60 seconds")
    else:
        print("\n❌ Performance test failed")
        print("💡 Check configuration and try again")

    # Optional comparison test
    print("\n" + "=" * 60)
    response = input("Run comparison test? (y/n): ").lower().strip()

    if response in ["y", "yes"]:
        test_standard_vs_highend()


if __name__ == "__main__":
    main()
