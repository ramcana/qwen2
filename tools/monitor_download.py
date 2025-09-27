#!/usr/bin/env python3
"""
Monitor and verify Qwen-Image download progress
"""
import os
import time
from pathlib import Path
from huggingface_hub import snapshot_download, repo_info

def check_download_status():
    """Check the current download status"""
    try:
        # Try to get the model path (will work if download is complete)
        model_path = snapshot_download('Qwen/Qwen-Image', local_files_only=True)
        
        # Calculate size
        model_dir = Path(model_path)
        total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        
        print(f"✅ Model found: {model_path}")
        print(f"📊 Current size: {total_size / (1024**3):.2f} GB")
        
        # Check if all files are present
        try:
            repo_data = repo_info('Qwen/Qwen-Image')
            remote_files = [f.rfilename for f in repo_data.siblings]
            
            missing_files = []
            for remote_file in remote_files:
                local_file = model_dir / remote_file
                if not local_file.exists():
                    missing_files.append(remote_file)
            
            if missing_files:
                print(f"⚠️ Missing {len(missing_files)} files:")
                for f in missing_files[:5]:  # Show first 5
                    print(f"   - {f}")
                if len(missing_files) > 5:
                    print(f"   ... and {len(missing_files) - 5} more")
                return False
            else:
                print("✅ All files present - download complete!")
                return True
                
        except Exception as e:
            print(f"⚠️ Could not verify completeness: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Model not ready: {e}")
        return False

def verify_model_integrity():
    """Verify the model can be loaded"""
    try:
        print("\n🔍 Verifying model integrity...")
        
        # Try to load the model
        from diffusers import DiffusionPipeline
        import torch
        
        model_path = snapshot_download('Qwen/Qwen-Image', local_files_only=True)
        
        # Quick load test (don't move to GPU to save time)
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=None  # Keep on CPU for verification
        )
        
        print("✅ Model loads successfully!")
        print(f"📋 Model type: {type(pipe).__name__}")
        
        # Check key components
        components = ['transformer', 'text_encoder', 'tokenizer', 'scheduler', 'vae']
        for component in components:
            if hasattr(pipe, component):
                print(f"   ✓ {component}")
            else:
                print(f"   ❌ Missing {component}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

def main():
    """Main monitoring function"""
    print("🔍 Qwen-Image Download Monitor")
    print("=" * 40)
    
    # Check current status
    is_complete = check_download_status()
    
    if is_complete:
        # Verify integrity
        verify_model_integrity()
        
        # Create local symlink if needed
        local_dir = Path("./models/Qwen-Image")
        if not local_dir.exists():
            try:
                model_path = snapshot_download('Qwen/Qwen-Image', local_files_only=True)
                local_dir.parent.mkdir(exist_ok=True)
                local_dir.symlink_to(model_path)
                print(f"🔗 Created symlink: {local_dir} -> {model_path}")
            except Exception as e:
                print(f"⚠️ Could not create symlink: {e}")
        
        print("\n🎉 Download verification complete!")
        print("💡 You can now run: make smoke")
        
    else:
        print("\n⏳ Download still in progress...")
        print("💡 Run this script again to check status")

if __name__ == "__main__":
    main()