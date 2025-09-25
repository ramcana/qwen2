"""
Advanced Cache Consolidator for Qwen Models
Handles complex cache situations including multiple local copies
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedCacheConsolidator:
    """
    Advanced consolidator that handles complex cache situations
    """
    
    def __init__(self):
        self.hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        self.local_models_dir = Path("./models")
        
    def analyze_complex_cache(self) -> Dict[str, Dict]:
        """Analyze complex cache situation with multiple copies"""
        logger.info("🔍 Analyzing complex cache situation...")
        
        analysis = {}
        
        # Check for Qwen-Image-Edit specifically
        model_name = "Qwen/Qwen-Image-Edit"
        
        # Check HuggingFace cache
        hf_cache_path = self.hf_cache_dir / "models--Qwen--Qwen-Image-Edit"
        hf_cache_size = self._calculate_directory_size(hf_cache_path) if hf_cache_path.exists() else 0
        
        # Check local copies
        local_copies = []
        
        # Check models/Qwen-Image-Edit/
        local_path1 = self.local_models_dir / "Qwen-Image-Edit"
        if local_path1.exists():
            size1 = self._calculate_directory_size(local_path1)
            local_copies.append({
                "path": str(local_path1),
                "size_gb": size1,
                "type": "direct_download",
                "has_model_files": self._has_model_files(local_path1)
            })
        
        # Check models/qwen-image-edit/
        local_path2 = self.local_models_dir / "qwen-image-edit"
        if local_path2.exists():
            size2 = self._calculate_directory_size(local_path2)
            local_copies.append({
                "path": str(local_path2),
                "size_gb": size2,
                "type": "cache_structure",
                "has_model_files": self._has_model_files(local_path2)
            })
        
        analysis[model_name] = {
            "hf_cache": {
                "path": str(hf_cache_path),
                "size_gb": hf_cache_size,
                "exists": hf_cache_path.exists(),
                "complete": hf_cache_size > 1.0  # >1GB indicates actual model data
            },
            "local_copies": local_copies,
            "total_local_size": sum(copy["size_gb"] for copy in local_copies),
            "duplicate_count": len(local_copies),
            "can_consolidate": len(local_copies) > 0
        }
        
        return analysis
    
    def print_complex_analysis(self):
        """Print detailed analysis of complex cache situation"""
        analysis = self.analyze_complex_cache()
        
        print("\n📊 Complex Cache Analysis")
        print("=" * 80)
        
        for model_name, info in analysis.items():
            print(f"\n🔍 {model_name}:")
            
            # HuggingFace cache status
            hf_info = info["hf_cache"]
            cache_status = "✅ Complete" if hf_info["complete"] else "❌ Empty/Incomplete"
            print(f"  HF Cache: {cache_status} ({hf_info['size_gb']:.1f}GB)")
            print(f"    Path: {hf_info['path']}")
            
            # Local copies
            print(f"  Local Copies: {info['duplicate_count']} found")
            for i, copy in enumerate(info["local_copies"], 1):
                model_status = "✅ Has Models" if copy["has_model_files"] else "❌ No Models"
                print(f"    Copy {i}: {model_status} ({copy['size_gb']:.1f}GB)")
                print(f"      Path: {copy['path']}")
                print(f"      Type: {copy['type']}")
            
            print(f"  Total Duplicate Storage: {info['total_local_size']:.1f}GB")
            
            # Recommendations
            if info["duplicate_count"] > 1:
                print(f"  💡 Recommendation: Consolidate {info['duplicate_count']} copies")
            elif info["duplicate_count"] == 1 and not hf_info["complete"]:
                print(f"  💡 Recommendation: Move local copy to HF cache")
            elif info["duplicate_count"] == 1 and hf_info["complete"]:
                print(f"  💡 Recommendation: Remove local duplicate")
    
    def consolidate_qwen_image_edit(self, dry_run: bool = True) -> bool:
        """
        Specifically consolidate Qwen-Image-Edit model
        """
        analysis = self.analyze_complex_cache()
        model_name = "Qwen/Qwen-Image-Edit"
        
        if model_name not in analysis:
            logger.error("❌ Qwen-Image-Edit not found in analysis")
            return False
        
        info = analysis[model_name]
        
        if dry_run:
            logger.info("🔍 DRY RUN - Qwen-Image-Edit Consolidation Plan:")
        else:
            logger.info("🚀 Consolidating Qwen-Image-Edit...")
        
        # Strategy: Keep the best local copy and remove others
        local_copies = info["local_copies"]
        
        if len(local_copies) == 0:
            logger.info("✅ No local copies to consolidate")
            return True
        
        # Find the best copy (largest with model files)
        best_copy = None
        copies_to_remove = []
        
        for copy in local_copies:
            if copy["has_model_files"] and copy["size_gb"] > 50:  # Reasonable size for this model
                if best_copy is None or copy["size_gb"] > best_copy["size_gb"]:
                    if best_copy is not None:
                        copies_to_remove.append(best_copy)
                    best_copy = copy
                else:
                    copies_to_remove.append(copy)
            else:
                copies_to_remove.append(copy)
        
        if best_copy is None:
            logger.error("❌ No valid model copy found")
            return False
        
        # Show plan
        logger.info(f"📦 Best copy: {best_copy['path']} ({best_copy['size_gb']:.1f}GB)")
        
        total_savings = 0
        for copy in copies_to_remove:
            logger.info(f"🗑️ Will remove: {copy['path']} ({copy['size_gb']:.1f}GB)")
            total_savings += copy["size_gb"]
        
        logger.info(f"💾 Total savings: {total_savings:.1f}GB")
        
        if dry_run:
            logger.info("💡 Run with dry_run=False to apply changes")
            return True
        
        # Actually remove duplicates
        removed_count = 0
        actual_savings = 0
        
        for copy in copies_to_remove:
            try:
                copy_path = Path(copy["path"])
                if copy_path.exists():
                    logger.info(f"🗑️ Removing: {copy_path}")
                    shutil.rmtree(copy_path)
                    removed_count += 1
                    actual_savings += copy["size_gb"]
                    logger.info(f"✅ Removed successfully")
            except Exception as e:
                logger.error(f"❌ Failed to remove {copy['path']}: {e}")
        
        logger.info(f"✅ Consolidation complete:")
        logger.info(f"  Copies removed: {removed_count}")
        logger.info(f"  Space freed: {actual_savings:.1f}GB")
        logger.info(f"  Remaining copy: {best_copy['path']}")
        
        return True
    
    def move_to_hf_cache(self, source_path: str, model_name: str, dry_run: bool = True) -> bool:
        """
        Move a local model copy to HuggingFace cache structure
        """
        if dry_run:
            logger.info(f"🔍 DRY RUN - Would move {source_path} to HF cache for {model_name}")
            return True
        
        logger.info(f"📦 Moving {source_path} to HuggingFace cache...")
        
        try:
            source = Path(source_path)
            if not source.exists():
                logger.error(f"❌ Source path does not exist: {source_path}")
                return False
            
            # Create HF cache structure
            cache_model_dir = self.hf_cache_dir / f"models--{model_name.replace('/', '--')}"
            snapshots_dir = cache_model_dir / "snapshots"
            
            # Generate a snapshot ID (use timestamp)
            import time
            snapshot_id = f"snapshot_{int(time.time())}"
            snapshot_path = snapshots_dir / snapshot_id
            
            # Create directories
            snapshot_path.mkdir(parents=True, exist_ok=True)
            
            # Move files
            logger.info(f"📁 Moving files to: {snapshot_path}")
            
            for item in source.iterdir():
                dest_item = snapshot_path / item.name
                if item.is_dir():
                    shutil.copytree(item, dest_item)
                else:
                    shutil.copy2(item, dest_item)
            
            # Create refs/main file
            refs_dir = cache_model_dir / "refs"
            refs_dir.mkdir(exist_ok=True)
            (refs_dir / "main").write_text(snapshot_id)
            
            logger.info(f"✅ Successfully moved to HF cache")
            logger.info(f"📁 Cache location: {snapshot_path}")
            
            # Remove original
            logger.info(f"🗑️ Removing original: {source_path}")
            shutil.rmtree(source)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to move to HF cache: {e}")
            return False
    
    def create_comprehensive_solution(self, dry_run: bool = True) -> bool:
        """
        Create a comprehensive solution for the cache situation
        """
        analysis = self.analyze_complex_cache()
        
        if dry_run:
            logger.info("🔍 COMPREHENSIVE SOLUTION - DRY RUN")
        else:
            logger.info("🚀 IMPLEMENTING COMPREHENSIVE SOLUTION")
        
        logger.info("=" * 60)
        
        total_savings = 0
        
        for model_name, info in analysis.items():
            logger.info(f"\n📦 Processing {model_name}:")
            
            if info["duplicate_count"] > 1:
                # Multiple local copies - consolidate
                logger.info(f"  Strategy: Consolidate {info['duplicate_count']} local copies")
                
                if model_name == "Qwen/Qwen-Image-Edit":
                    success = self.consolidate_qwen_image_edit(dry_run)
                    if success:
                        # Calculate savings (all but the largest copy)
                        sizes = [copy["size_gb"] for copy in info["local_copies"]]
                        sizes.sort(reverse=True)
                        savings = sum(sizes[1:])  # All except the largest
                        total_savings += savings
                        logger.info(f"  💾 Savings: {savings:.1f}GB")
            
            elif info["duplicate_count"] == 1 and not info["hf_cache"]["complete"]:
                # One local copy, no HF cache - move to cache
                local_copy = info["local_copies"][0]
                logger.info(f"  Strategy: Move to HF cache")
                logger.info(f"  Source: {local_copy['path']} ({local_copy['size_gb']:.1f}GB)")
                
                if not dry_run:
                    success = self.move_to_hf_cache(local_copy["path"], model_name, dry_run)
                    if success:
                        logger.info(f"  ✅ Moved to HF cache")
                else:
                    logger.info(f"  Would move to HF cache")
            
            elif info["duplicate_count"] == 1 and info["hf_cache"]["complete"]:
                # One local copy, HF cache exists - remove local
                local_copy = info["local_copies"][0]
                logger.info(f"  Strategy: Remove local duplicate (HF cache exists)")
                logger.info(f"  Would remove: {local_copy['path']} ({local_copy['size_gb']:.1f}GB)")
                total_savings += local_copy["size_gb"]
                
                if not dry_run:
                    try:
                        shutil.rmtree(local_copy["path"])
                        logger.info(f"  ✅ Removed local duplicate")
                    except Exception as e:
                        logger.error(f"  ❌ Failed to remove: {e}")
        
        logger.info(f"\n📊 SOLUTION SUMMARY:")
        logger.info(f"  Total potential savings: {total_savings:.1f}GB")
        
        if dry_run:
            logger.info(f"💡 Run with dry_run=False to implement solution")
        
        return True
    
    def _calculate_directory_size(self, directory: Path) -> float:
        """Calculate directory size in GB"""
        try:
            if not directory.exists():
                return 0.0
            total_size = sum(
                f.stat().st_size for f in directory.rglob("*") if f.is_file()
            )
            return total_size / (1024**3)  # Convert to GB
        except Exception:
            return 0.0
    
    def _has_model_files(self, directory: Path) -> bool:
        """Check if directory contains actual model files"""
        try:
            if not directory.exists():
                return False
            
            # Look for common model files
            model_indicators = [
                "*.safetensors",
                "*.bin", 
                "model_index.json",
                "config.json",
                "pytorch_model*.bin"
            ]
            
            for pattern in model_indicators:
                if list(directory.rglob(pattern)):
                    return True
            
            # Check for model directories
            model_dirs = ["transformer", "unet", "vae", "text_encoder"]
            for dir_name in model_dirs:
                if (directory / dir_name).exists():
                    return True
            
            return False
            
        except Exception:
            return False


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Cache Consolidator")
    parser.add_argument("--analyze", action="store_true", help="Analyze complex cache situation")
    parser.add_argument("--consolidate", action="store_true", help="Consolidate Qwen-Image-Edit specifically")
    parser.add_argument("--solution", action="store_true", help="Implement comprehensive solution")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    consolidator = AdvancedCacheConsolidator()
    
    if args.analyze:
        consolidator.print_complex_analysis()
    
    elif args.consolidate:
        consolidator.consolidate_qwen_image_edit(dry_run=args.dry_run)
    
    elif args.solution:
        consolidator.create_comprehensive_solution(dry_run=args.dry_run)
    
    else:
        print("Use --help to see available options")


if __name__ == "__main__":
    main()