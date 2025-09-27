#!/usr/bin/env python3
"""
Safe memory test that won't crash the system
"""
import torch
import gc
import os
import psutil

def check_system_resources():
    """Check available system resources before loading"""
    print("🔍 System Resource Check")
    print("=" * 30)
    
    # GPU info
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_memory_gb = gpu_props.total_memory / (1024**3)
        print(f"🎮 GPU: {gpu_props.name}")
        print(f"🎮 GPU Memory: {gpu_memory_gb:.1f} GB")
        
        # Current GPU usage
        current_gpu = torch.cuda.memory_allocated() / (1024**3)
        print(f"🎮 GPU Used: {current_gpu:.2f} GB")
        
        if current_gpu > gpu_memory_gb * 0.8:
            print("⚠️ WARNING: GPU memory usage is very high!")
            return False
    else:
        print("❌ No GPU available")
        return False
    
    # System RAM
    memory = psutil.virtual_memory()
    ram_gb = memory.total / (1024**3)
    ram_used_gb = memory.used / (1024**3)
    ram_available_gb = memory.available / (1024**3)
    
    print(f"💾 System RAM: {ram_gb:.1f} GB")
    print(f"💾 RAM Used: {ram_used_gb:.1f} GB")
    print(f"💾 RAM Available: {ram_available_gb:.1f} GB")
    
    if ram_available_gb < 8:
        print("⚠️ WARNING: Low system RAM available!")
        return False
    
    return True

def clear_all_memory():
    """Aggressively clear all memory"""
    print("🧹 Clearing all memory...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    gc.collect()
    
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()

def test_model_loading_only():
    """Test just loading the model without generation"""
    print("\n📁 Testing Model Loading (No Generation)")
    print("=" * 45)
    
    try:
        clear_all_memory()
        
        # Check memory before loading
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated() / (1024**3)
            print(f"📊 GPU Memory before: {memory_before:.2f} GB")
        
        # Try to load just the model info first
        from huggingface_hub import snapshot_download
        model_path = snapshot_download('Qwen/Qwen-Image', local_files_only=True)
        print(f"✅ Model path verified: {model_path}")
        
        # Check model size on disk
        import os
        from pathlib import Path
        model_dir = Path(model_path)
        total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        print(f"📊 Model size on disk: {size_gb:.1f} GB")
        
        if size_gb > 50:
            print("⚠️ WARNING: Model is very large (>50GB)")
            print("💡 This model may be too large for your 16GB GPU")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def suggest_alternatives():
    """Suggest alternative approaches"""
    print("\n💡 Alternative Solutions")
    print("=" * 25)
    
    print("1. 🔄 Use a smaller Qwen model:")
    print("   - Qwen/Qwen-VL-Chat (smaller)")
    print("   - Qwen/Qwen-7B-Chat (text only)")
    
    print("\n2. ⚙️ Use CPU-only mode:")
    print("   - Slower but won't crash")
    print("   - Uses system RAM instead of GPU")
    
    print("\n3. 🌐 Use online inference:")
    print("   - Hugging Face Inference API")
    print("   - No local memory requirements")
    
    print("\n4. 🔧 Model quantization:")
    print("   - Use 8-bit or 4-bit quantization")
    print("   - Significantly reduces memory usage")

def check_for_smaller_models():
    """Check if smaller models are available"""
    print("\n🔍 Checking for Alternative Models")
    print("=" * 35)
    
    # Check what models are in cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    if os.path.exists(cache_dir):
        models = [d for d in os.listdir(cache_dir) if d.startswith("models--")]
        
        print("📂 Models in cache:")
        for model in models:
            model_name = model.replace("models--", "").replace("--", "/")
            print(f"   - {model_name}")
        
        # Check sizes
        for model in models:
            model_path = os.path.join(cache_dir, model)
            if os.path.isdir(model_path):
                try:
                    size = sum(f.stat().st_size for f in Path(model_path).rglob('*') if f.is_file())
                    size_gb = size / (1024**3)
                    print(f"     Size: {size_gb:.1f} GB")
                except:
                    print(f"     Size: Unknown")

def main():
    """Main safety check function"""
    print("🛡️ Safe Memory Test - System Protection")
    print("=" * 50)
    
    # Step 1: Check system resources
    resources_ok = check_system_resources()
    
    if not resources_ok:
        print("\n❌ System resources insufficient or unsafe")
        suggest_alternatives()
        return False
    
    # Step 2: Test model loading only
    loading_ok = test_model_loading_only()
    
    if not loading_ok:
        print("\n❌ Model loading test failed")
        suggest_alternatives()
        check_for_smaller_models()
        return False
    
    # Step 3: Memory recommendations
    print("\n✅ Basic checks passed")
    print("\n🎯 Recommendations for your 16GB GPU:")
    print("1. The current Qwen-Image model (53GB) is too large")
    print("2. Consider using a quantized version")
    print("3. Or use CPU-only mode with sufficient RAM")
    print("4. Or switch to a smaller model variant")
    
    return True

if __name__ == "__main__":
    main()