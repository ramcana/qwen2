"""
Cache Manager for Qwen Models
Fixes duplicate model storage and provides efficient cache management
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheInfo:
    """Information about cached models"""
    model_name: str
    cache_path: str
    local_path: Optional[str]
    cache_size_gb: float
    local_size_gb: float
    total_size_gb: float
    is_duplicate: bool
    can_consolidate: bool


class CacheManager:
    """
    Manages model cache to eliminate duplicates and optimize storage
    """
    
    def __init__(self):
        self.hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        self.local_models_dir = Path("./models")
        
    def analyze_cache_usage(self) -> Dict[str, CacheInfo]:
        """Analyze current cache usage and identify duplicates"""
        logger.info("🔍 Analyzing cache usage...")
        
        cache_info = {}
        
        # Check HuggingFace cache
        if self.hf_cache_dir.exists():
            for model_dir in self.hf_cache_dir.iterdir():
                if model_dir.is_dir() and model_dir.name.startswith("models--"):
                    model_name = model_dir.name.replace("models--", "").replace("--", "/")
                    
                    if "Qwen" in model_name:
                        cache_size = self._calculate_directory_size(model_dir)
                        
                        # Check for corresponding local model
                        local_path = None
                        local_size = 0.0
                        
                        # Check multiple possible local paths
                        possible_local_paths = [
                            self.local_models_dir / model_name.split("/")[-1],
                            self.local_models_dir / model_name.split("/")[-1].lower(),
                            self.local_models_dir / model_name.split("/")[-1].replace("-", "_"),
                        ]
                        
                        for path in possible_local_paths:
                            if path.exists():
                                local_path = str(path)
                                local_size = self._calculate_directory_size(path)
                                break
                        
                        cache_info[model_name] = CacheInfo(
                            model_name=model_name,
                            cache_path=str(model_dir),
                            local_path=local_path,
                            cache_size_gb=cache_size,
                            local_size_gb=local_size,
                            total_size_gb=cache_size + local_size,
                            is_duplicate=local_path is not None and local_size > 1.0,  # >1GB threshold
                            can_consolidate=local_path is not None
                        )
        
        return cache_info
    
    def print_cache_analysis(self):
        """Print detailed cache analysis"""
        cache_info = self.analyze_cache_usage()
        
        print("\n📊 Cache Usage Analysis")
        print("=" * 80)
        
        total_cache_size = 0
        total_local_size = 0
        total_duplicate_size = 0
        
        for model_name, info in cache_info.items():
            total_cache_size += info.cache_size_gb
            total_local_size += info.local_size_gb
            
            if info.is_duplicate:
                total_duplicate_size += info.local_size_gb
            
            status = "🔄 DUPLICATE" if info.is_duplicate else "✅ OK"
            
            print(f"\n{model_name}:")
            print(f"  Status: {status}")
            print(f"  HF Cache: {info.cache_size_gb:.1f}GB ({info.cache_path})")
            
            if info.local_path:
                print(f"  Local:    {info.local_size_gb:.1f}GB ({info.local_path})")
            else:
                print(f"  Local:    None")
            
            print(f"  Total:    {info.total_size_gb:.1f}GB")
        
        print(f"\n📈 Summary:")
        print(f"  HuggingFace Cache: {total_cache_size:.1f}GB")
        print(f"  Local Models:      {total_local_size:.1f}GB")
        print(f"  Duplicate Storage: {total_duplicate_size:.1f}GB")
        print(f"  Total Storage:     {total_cache_size + total_local_size:.1f}GB")
        
        if total_duplicate_size > 0:
            print(f"\n💡 Potential Savings: {total_duplicate_size:.1f}GB can be freed by consolidating duplicates")
    
    def consolidate_cache(self, dry_run: bool = True) -> bool:
        """
        Consolidate cache by removing duplicates and using HuggingFace cache as primary
        
        Args:
            dry_run: If True, only show what would be done without making changes
        """
        cache_info = self.analyze_cache_usage()
        
        if dry_run:
            logger.info("🔍 DRY RUN - Showing what would be done:")
        else:
            logger.info("🚀 Consolidating cache...")
        
        total_savings = 0
        actions_taken = 0
        
        for model_name, info in cache_info.items():
            if info.is_duplicate and info.can_consolidate:
                logger.info(f"\n📦 Processing {model_name}:")
                logger.info(f"  Cache: {info.cache_size_gb:.1f}GB")
                logger.info(f"  Local: {info.local_size_gb:.1f}GB")
                
                if dry_run:
                    logger.info(f"  Would remove: {info.local_path}")
                    logger.info(f"  Would save: {info.local_size_gb:.1f}GB")
                else:
                    try:
                        # Verify cache is complete before removing local copy
                        if self._verify_cache_completeness(info.cache_path, info.local_path):
                            logger.info(f"  ✅ Cache verified complete")
                            logger.info(f"  🗑️ Removing local copy: {info.local_path}")
                            shutil.rmtree(info.local_path)
                            logger.info(f"  💾 Freed: {info.local_size_gb:.1f}GB")
                            actions_taken += 1
                        else:
                            logger.warning(f"  ⚠️ Cache incomplete, keeping local copy")
                            continue
                    except Exception as e:
                        logger.error(f"  ❌ Failed to remove {info.local_path}: {e}")
                        continue
                
                total_savings += info.local_size_gb
        
        if dry_run:
            logger.info(f"\n📊 Dry Run Summary:")
            logger.info(f"  Models to consolidate: {sum(1 for info in cache_info.values() if info.is_duplicate)}")
            logger.info(f"  Potential savings: {total_savings:.1f}GB")
            logger.info(f"\n💡 Run with dry_run=False to apply changes")
        else:
            logger.info(f"\n✅ Consolidation Complete:")
            logger.info(f"  Models processed: {actions_taken}")
            logger.info(f"  Space freed: {total_savings:.1f}GB")
        
        return True
    
    def clean_broken_downloads(self) -> bool:
        """Clean up broken or incomplete downloads"""
        logger.info("🧹 Cleaning broken downloads...")
        
        cleaned_size = 0
        cleaned_count = 0
        
        # Check local models directory
        if self.local_models_dir.exists():
            for model_dir in self.local_models_dir.iterdir():
                if model_dir.is_dir():
                    # Check if model appears incomplete
                    if self._is_incomplete_download(model_dir):
                        size = self._calculate_directory_size(model_dir)
                        logger.info(f"🗑️ Removing incomplete download: {model_dir.name} ({size:.1f}GB)")
                        
                        try:
                            shutil.rmtree(model_dir)
                            cleaned_size += size
                            cleaned_count += 1
                        except Exception as e:
                            logger.error(f"❌ Failed to remove {model_dir}: {e}")
        
        # Check HuggingFace cache for incomplete downloads
        if self.hf_cache_dir.exists():
            for model_dir in self.hf_cache_dir.iterdir():
                if model_dir.is_dir() and model_dir.name.startswith("models--"):
                    if self._is_incomplete_cache_download(model_dir):
                        size = self._calculate_directory_size(model_dir)
                        logger.info(f"🗑️ Removing incomplete cache: {model_dir.name} ({size:.1f}GB)")
                        
                        try:
                            shutil.rmtree(model_dir)
                            cleaned_size += size
                            cleaned_count += 1
                        except Exception as e:
                            logger.error(f"❌ Failed to remove {model_dir}: {e}")
        
        logger.info(f"✅ Cleanup complete: {cleaned_count} items removed, {cleaned_size:.1f}GB freed")
        return True
    
    def optimize_download_strategy(self) -> Dict[str, str]:
        """Provide recommendations for optimal download strategy"""
        recommendations = {
            "strategy": "use_hf_cache_only",
            "reasoning": "Use HuggingFace cache as single source of truth",
            "benefits": [
                "Eliminates duplicate storage",
                "Automatic resume capability", 
                "Better integration with transformers library",
                "Shared across all projects"
            ],
            "implementation": "Modify ModelDownloadManager to use cache_dir instead of local_dir"
        }
        
        return recommendations
    
    def create_symlinks_to_cache(self) -> bool:
        """Create symlinks from local models directory to HuggingFace cache"""
        logger.info("🔗 Creating symlinks to HuggingFace cache...")
        
        cache_info = self.analyze_cache_usage()
        created_links = 0
        
        for model_name, info in cache_info.items():
            if info.cache_size_gb > 0:  # Model exists in cache
                model_short_name = model_name.split("/")[-1]
                local_link_path = self.local_models_dir / model_short_name
                
                # Remove existing local copy if it exists
                if local_link_path.exists() and not local_link_path.is_symlink():
                    logger.info(f"🗑️ Removing existing local copy: {local_link_path}")
                    if local_link_path.is_dir():
                        shutil.rmtree(local_link_path)
                    else:
                        local_link_path.unlink()
                
                # Create symlink if it doesn't exist
                if not local_link_path.exists():
                    try:
                        # Find the actual model directory in cache
                        cache_model_dir = Path(info.cache_path)
                        
                        # Look for snapshots directory
                        snapshots_dir = cache_model_dir / "snapshots"
                        if snapshots_dir.exists():
                            # Use the latest snapshot
                            snapshots = list(snapshots_dir.iterdir())
                            if snapshots:
                                latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
                                
                                # Create parent directory if needed
                                local_link_path.parent.mkdir(parents=True, exist_ok=True)
                                
                                # Create symlink
                                local_link_path.symlink_to(latest_snapshot, target_is_directory=True)
                                logger.info(f"🔗 Created symlink: {local_link_path} -> {latest_snapshot}")
                                created_links += 1
                    
                    except Exception as e:
                        logger.error(f"❌ Failed to create symlink for {model_name}: {e}")
        
        logger.info(f"✅ Created {created_links} symlinks")
        return True
    
    def _calculate_directory_size(self, directory: Path) -> float:
        """Calculate directory size in GB"""
        try:
            total_size = sum(
                f.stat().st_size for f in directory.rglob("*") if f.is_file()
            )
            return total_size / (1024**3)  # Convert to GB
        except Exception:
            return 0.0
    
    def _verify_cache_completeness(self, cache_path: str, local_path: str) -> bool:
        """Verify that cache contains all files from local copy"""
        try:
            cache_dir = Path(cache_path)
            local_dir = Path(local_path)
            
            # Find the actual model files in cache (in snapshots directory)
            snapshots_dir = cache_dir / "snapshots"
            if not snapshots_dir.exists():
                return False
            
            snapshots = list(snapshots_dir.iterdir())
            if not snapshots:
                return False
            
            # Use the latest snapshot
            cache_model_dir = max(snapshots, key=lambda x: x.stat().st_mtime)
            
            # Check if essential files exist in cache
            essential_files = ["model_index.json", "config.json"]
            
            for file_name in essential_files:
                cache_file = cache_model_dir / file_name
                local_file = local_dir / file_name
                
                if local_file.exists() and not cache_file.exists():
                    return False
            
            # Check if cache has at least as many files as local
            cache_files = list(cache_model_dir.rglob("*"))
            local_files = list(local_dir.rglob("*"))
            
            cache_file_count = len([f for f in cache_files if f.is_file()])
            local_file_count = len([f for f in local_files if f.is_file()])
            
            return cache_file_count >= local_file_count * 0.9  # Allow 10% tolerance
            
        except Exception as e:
            logger.warning(f"Cache verification failed: {e}")
            return False
    
    def _is_incomplete_download(self, model_dir: Path) -> bool:
        """Check if a model directory appears to be an incomplete download"""
        try:
            # Check for essential files
            essential_files = ["model_index.json", "config.json"]
            
            has_essential = any(
                (model_dir / file_name).exists() for file_name in essential_files
            )
            
            if not has_essential:
                return True
            
            # Check for .tmp files or other indicators of incomplete download
            tmp_files = list(model_dir.rglob("*.tmp"))
            if tmp_files:
                return True
            
            # Check if directory is suspiciously small
            size_gb = self._calculate_directory_size(model_dir)
            if size_gb < 0.1:  # Less than 100MB is suspicious for these models
                return True
            
            return False
            
        except Exception:
            return True  # If we can't check, assume it's incomplete
    
    def _is_incomplete_cache_download(self, cache_dir: Path) -> bool:
        """Check if a cache directory appears to be an incomplete download"""
        try:
            # Check for snapshots directory
            snapshots_dir = cache_dir / "snapshots"
            if not snapshots_dir.exists():
                return True
            
            snapshots = list(snapshots_dir.iterdir())
            if not snapshots:
                return True
            
            # Check the latest snapshot
            latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
            
            # Check for essential files in snapshot
            essential_files = ["model_index.json", "config.json"]
            has_essential = any(
                (latest_snapshot / file_name).exists() for file_name in essential_files
            )
            
            return not has_essential
            
        except Exception:
            return True


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen Cache Manager")
    parser.add_argument("--analyze", action="store_true", help="Analyze cache usage")
    parser.add_argument("--consolidate", action="store_true", help="Consolidate cache (remove duplicates)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--clean", action="store_true", help="Clean broken downloads")
    parser.add_argument("--symlinks", action="store_true", help="Create symlinks to cache")
    parser.add_argument("--recommendations", action="store_true", help="Show optimization recommendations")
    
    args = parser.parse_args()
    
    cache_manager = CacheManager()
    
    if args.analyze:
        cache_manager.print_cache_analysis()
    
    elif args.consolidate:
        cache_manager.consolidate_cache(dry_run=args.dry_run)
    
    elif args.clean:
        cache_manager.clean_broken_downloads()
    
    elif args.symlinks:
        cache_manager.create_symlinks_to_cache()
    
    elif args.recommendations:
        recommendations = cache_manager.optimize_download_strategy()
        print("\n💡 Optimization Recommendations:")
        print("=" * 50)
        print(f"Strategy: {recommendations['strategy']}")
        print(f"Reasoning: {recommendations['reasoning']}")
        print("\nBenefits:")
        for benefit in recommendations['benefits']:
            print(f"  • {benefit}")
        print(f"\nImplementation: {recommendations['implementation']}")
    
    else:
        print("Use --help to see available options")


if __name__ == "__main__":
    main()