#!/usr/bin/env python3
"""
Check actual model sizes from HuggingFace without downloading
"""

try:
    from huggingface_hub import HfApi
    api = HfApi()
    
    models_to_check = [
        "Qwen/Qwen-Image",
        "Qwen/Qwen-Image-Edit", 
        "Qwen/Qwen2-VL-7B-Instruct"
    ]
    
    print("🔍 Checking actual model sizes from HuggingFace...")
    print("=" * 60)
    
    for model_id in models_to_check:
        try:
            print(f"\n📦 {model_id}")
            
            # Get model info
            model_info = api.model_info(model_id)
            
            # Get file sizes
            total_size = 0
            file_count = 0
            
            if hasattr(model_info, 'siblings') and model_info.siblings:
                for file_info in model_info.siblings:
                    if hasattr(file_info, 'size') and file_info.size:
                        total_size += file_info.size
                        file_count += 1
                        
                        # Show largest files
                        if file_info.size > 1e9:  # > 1GB
                            size_gb = file_info.size / 1e9
                            print(f"   📄 {file_info.rfilename}: {size_gb:.1f}GB")
            
            total_gb = total_size / 1e9
            print(f"   📊 Total: {file_count} files, {total_gb:.1f}GB")
            
            # Check if it's actually a text-to-image model
            if hasattr(model_info, 'tags') and model_info.tags:
                relevant_tags = [tag for tag in model_info.tags if any(keyword in tag.lower() 
                    for keyword in ['text-to-image', 'image-generation', 'diffusion', 'edit'])]
                if relevant_tags:
                    print(f"   🏷️ Tags: {', '.join(relevant_tags)}")
            
        except Exception as e:
            print(f"   ❌ Error checking {model_id}: {e}")
    
    print(f"\n💡 This helps us verify the correct model sizes before downloading")
    
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
except Exception as e:
    print(f"❌ Error: {e}")