#!/usr/bin/env python3
"""
Check model repository information
"""

try:
    from huggingface_hub import HfApi, list_repo_files
    
    models_to_check = [
        "Qwen/Qwen-Image",
        "Qwen/Qwen-Image-Edit"
    ]
    
    print("🔍 Checking model repository structure...")
    print("=" * 60)
    
    for model_id in models_to_check:
        try:
            print(f"\n📦 {model_id}")
            
            # List files in the repository
            files = list_repo_files(model_id)
            
            print(f"   📁 Files found: {len(files)}")
            
            # Show key files
            key_files = [f for f in files if any(keyword in f.lower() 
                for keyword in ['model', 'config', 'safetensors', 'bin', 'index'])]
            
            for file in sorted(key_files)[:10]:  # Show first 10 key files
                print(f"   📄 {file}")
            
            if len(key_files) > 10:
                print(f"   ... and {len(key_files) - 10} more files")
            
            # Check for model_index.json to determine pipeline type
            if 'model_index.json' in files:
                print(f"   ✅ Diffusion pipeline detected (has model_index.json)")
            
            # Check for transformer files (MMDiT)
            transformer_files = [f for f in files if 'transformer' in f.lower()]
            if transformer_files:
                print(f"   🏗️ MMDiT architecture (has transformer files)")
                for tf in transformer_files[:3]:
                    print(f"      • {tf}")
            
            # Check for UNet files (traditional)
            unet_files = [f for f in files if 'unet' in f.lower()]
            if unet_files:
                print(f"   🏗️ UNet architecture (has unet files)")
                
        except Exception as e:
            print(f"   ❌ Error checking {model_id}: {e}")
    
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
except Exception as e:
    print(f"❌ Error: {e}")