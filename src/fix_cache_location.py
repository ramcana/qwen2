"""
Fix Cache Location - Move misplaced HuggingFace cache to correct location
"""

import os
import shutil
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_qwen_image_edit_cache():
    """Move the misplaced Qwen-Image-Edit cache to correct HuggingFace location"""
    
    # Source: misplaced cache in local models
    source_cache = Path("models/qwen-image-edit/models--Qwen--Qwen-Image-Edit")
    
    # Destination: proper HuggingFace cache location
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    dest_cache = hf_cache_dir / "models--Qwen--Qwen-Image-Edit"
    
    logger.info("🔧 Fixing Qwen-Image-Edit cache location...")
    logger.info(f"Source: {source_cache}")
    logger.info(f"Destination: {dest_cache}")
    
    if not source_cache.exists():
        logger.error("❌ Source cache not found")
        return False
    
    try:
        # Create destination parent directory if needed
        dest_cache.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if destination already exists
        if dest_cache.exists():
            logger.info("🗑️ Removing existing incomplete cache at destination...")
            shutil.rmtree(dest_cache)
        
        # Move the cache structure
        logger.info("📦 Moving cache structure...")
        shutil.move(str(source_cache), str(dest_cache))
        
        logger.info("✅ Cache moved successfully!")
        
        # Clean up empty parent directory
        parent_dir = source_cache.parent
        if parent_dir.exists() and not any(parent_dir.iterdir()):
            logger.info(f"🧹 Removing empty directory: {parent_dir}")
            parent_dir.rmdir()
            
            # Also remove models/qwen-image-edit if it's empty
            if parent_dir.parent.name == "models" and not any(parent_dir.parent.iterdir()):
                logger.info(f"🧹 Removing empty directory: {parent_dir.parent}")
                parent_dir.parent.rmdir()
        
        # Verify the move
        snapshots_dir = dest_cache / "snapshots"
        if snapshots_dir.exists():
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                logger.info(f"✅ Verification: Found {len(snapshots)} snapshot(s) in destination")
                
                # Calculate size
                total_size = sum(
                    f.stat().st_size for f in dest_cache.rglob("*") if f.is_file()
                ) / (1024**3)
                logger.info(f"📊 Cache size: {total_size:.1f}GB")
                
                return True
        
        logger.error("❌ Verification failed - snapshots not found in destination")
        return False
        
    except Exception as e:
        logger.error(f"❌ Failed to move cache: {e}")
        return False


def verify_final_state():
    """Verify the final cache state after fixing"""
    logger.info("🔍 Verifying final cache state...")
    
    # Check HuggingFace cache
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    qwen_image_edit_cache = hf_cache_dir / "models--Qwen--Qwen-Image-Edit"
    
    if qwen_image_edit_cache.exists():
        size = sum(
            f.stat().st_size for f in qwen_image_edit_cache.rglob("*") if f.is_file()
        ) / (1024**3)
        logger.info(f"✅ Qwen-Image-Edit in HF cache: {size:.1f}GB")
    else:
        logger.error("❌ Qwen-Image-Edit not found in HF cache")
    
    # Check local models directory
    models_dir = Path("models")
    if models_dir.exists():
        remaining_items = list(models_dir.iterdir())
        total_local_size = 0
        
        for item in remaining_items:
            if item.is_dir():
                size = sum(
                    f.stat().st_size for f in item.rglob("*") if f.is_file()
                ) / (1024**3)
                total_local_size += size
                logger.info(f"📁 Local: {item.name} ({size:.1f}GB)")
        
        logger.info(f"📊 Total local storage: {total_local_size:.1f}GB")
        
        if total_local_size < 1.0:  # Less than 1GB indicates mostly empty directories
            logger.info("✅ Local storage successfully minimized!")
        
    else:
        logger.info("✅ No local models directory (completely clean)")


def main():
    """Main function"""
    logger.info("🚀 Starting cache location fix...")
    logger.info("=" * 60)
    
    success = fix_qwen_image_edit_cache()
    
    if success:
        logger.info("\n🔍 Verifying results...")
        verify_final_state()
        
        logger.info("\n✅ Cache fix completed successfully!")
        logger.info("💡 Benefits:")
        logger.info("  • Qwen-Image-Edit now in proper HuggingFace cache")
        logger.info("  • Local models directory cleaned up")
        logger.info("  • No more duplicate storage")
        logger.info("  • Better integration with transformers/diffusers")
        
    else:
        logger.error("\n❌ Cache fix failed!")
        logger.info("💡 You may need to manually move the files or re-download the model")


if __name__ == "__main__":
    main()