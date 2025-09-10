#!/usr/bin/env python3
"""
Qwen-Image-Edit Model Downloader
Pre-downloads the Qwen-Image-Edit model for enhanced features
"""

import os

import torch
from diffusers import QwenImageEditPipeline


def download_qwen_edit_model():
    """Download and cache the Qwen-Image-Edit model"""
    print("🚀 Qwen-Image-Edit Model Downloader")
    print("=" * 50)

    # Check if QwenImageEditPipeline is available
    try:
        from diffusers import QwenImageEditPipeline

        print("✅ QwenImageEditPipeline is available")
    except ImportError:
        print("❌ QwenImageEditPipeline not available.")
        print("Please install the latest diffusers:")
        print("pip install git+https://github.com/huggingface/diffusers.git")
        return False

    # Create models directory
    models_dir = "./models/qwen-image-edit"
    os.makedirs(models_dir, exist_ok=True)

    print("\n📥 Downloading Qwen-Image-Edit model...")
    print(f"📁 Cache directory: {models_dir}")
    print("📊 Expected size: ~20GB")
    print("⏳ This may take 10-30 minutes depending on your internet speed...")

    try:
        # Download the model
        pipeline = QwenImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit",
            torch_dtype=torch.bfloat16,
            cache_dir=models_dir,
            low_cpu_mem_usage=True,
        )

        print("✅ Model downloaded successfully!")

        # Test loading to GPU if available
        if torch.cuda.is_available():
            print("🔄 Testing GPU loading...")
            try:
                pipeline = pipeline.to("cuda")
                print("✅ GPU loading successful!")
            except Exception as e:
                print(f"⚠️ GPU loading failed: {e}")
                print("Model will work on CPU")

        print("\n🎉 Qwen-Image-Edit model is ready!")
        print("You can now use the enhanced features:")
        print("• Image-to-Image generation")
        print("• Inpainting")
        print("• Advanced image editing")

        return True

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have sufficient disk space (~25GB)")
        print("3. Try running the script again")
        print("4. Check if Hugging Face is accessible")
        return False


def check_model_status():
    """Check if the model is already downloaded"""
    print("🔍 Checking Qwen-Image-Edit model status...")

    try:
        # Try to load from cache
        pipeline = QwenImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit",
            cache_dir="./models/qwen-image-edit",
            local_files_only=True,  # Only check local cache
        )
        print("✅ Model is already downloaded and ready!")
        return True
    except Exception:
        print("❌ Model not found in cache")
        return False


def main():
    print("Select an option:")
    print("1. Check model status")
    print("2. Download model (standard method)")
    print("3. Download model (enhanced HF Hub API) - RECOMMENDED")
    print("4. Exit")

    choice = input("\nEnter your choice (1/2/3/4): ").strip()

    if choice == "1":
        check_model_status()
    elif choice == "2":
        if not check_model_status():
            download_qwen_edit_model()
        else:
            print("Model is already available!")
    elif choice == "3":
        print("🚀 Launching enhanced downloader...")
        print("💡 This method provides better progress tracking and resume capability")
        try:
            import subprocess
            import sys

            subprocess.run([sys.executable, "tools/download_qwen_edit_hub.py"])
        except Exception as e:
            print(f"❌ Could not launch enhanced downloader: {e}")
            print("💡 Try: python tools/download_qwen_edit_hub.py")
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
