#!/usr/bin/env python3
"""
Demonstration of ModelDownloadManager capabilities
Shows how to use the download manager for Qwen models with Qwen2-VL support
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_download_manager import (
    ModelDownloadManager,
    ModelDownloadConfig,
    download_qwen_image,
    download_qwen2_vl,
    check_model_status
)


def demo_basic_usage():
    """Demonstrate basic ModelDownloadManager usage"""
    print("🚀 ModelDownloadManager Demo")
    print("=" * 50)
    
    # Initialize the manager
    manager = ModelDownloadManager()
    
    print(f"✅ Initialized with {len(manager.supported_models)} supported models:")
    for model_name, config in manager.supported_models.items():
        print(f"  - {model_name}")
        print(f"    Type: {config['type']}")
        print(f"    Size: {config['expected_size_gb']}GB")
        print(f"    Architecture: {config['architecture']}")
        if 'capabilities' in config:
            print(f"    Capabilities: {', '.join(config['capabilities'])}")
        print()


def demo_model_status_checking():
    """Demonstrate model status checking"""
    print("📊 Model Status Checking Demo")
    print("=" * 40)
    
    manager = ModelDownloadManager()
    
    # Check status of all supported models
    models_status = manager.list_available_models()
    
    for model_name, status in models_status.items():
        print(f"📋 {model_name}:")
        print(f"  Remote Available: {'✅' if status.get('remote_available') else '❌'}")
        print(f"  Local Available: {'✅' if status.get('local_available') else '❌'}")
        print(f"  Local Complete: {'✅' if status.get('local_complete') else '❌'}")
        
        if status.get('local_size_bytes', 0) > 0:
            size_str = manager._format_size(status['local_size_bytes'])
            print(f"  Local Size: {size_str}")
        
        if status.get('missing_files'):
            print(f"  Missing Files: {len(status['missing_files'])}")
        
        print()


def demo_download_configuration():
    """Demonstrate download configuration options"""
    print("⚙️ Download Configuration Demo")
    print("=" * 35)
    
    # Basic configuration
    basic_config = ModelDownloadConfig(
        model_name="Qwen/Qwen-Image"
    )
    print("📝 Basic Configuration:")
    print(f"  Resume Download: {basic_config.resume_download}")
    print(f"  Max Workers: {basic_config.max_workers}")
    print(f"  Verify Integrity: {basic_config.verify_integrity}")
    print(f"  Cleanup on Failure: {basic_config.cleanup_on_failure}")
    print()
    
    # Advanced configuration
    advanced_config = ModelDownloadConfig(
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        max_workers=2,
        timeout=600,
        retry_attempts=5,
        ignore_patterns=["*.md", "*.txt", ".gitattributes", "*.json"]
    )
    print("🔧 Advanced Configuration:")
    print(f"  Model: {advanced_config.model_name}")
    print(f"  Max Workers: {advanced_config.max_workers}")
    print(f"  Timeout: {advanced_config.timeout}s")
    print(f"  Retry Attempts: {advanced_config.retry_attempts}")
    print(f"  Ignore Patterns: {advanced_config.ignore_patterns}")
    print()


def demo_utility_functions():
    """Demonstrate utility functions"""
    print("🛠️ Utility Functions Demo")
    print("=" * 30)
    
    manager = ModelDownloadManager()
    
    # Size formatting
    sizes = [1024, 1024**2, 1024**3, 15 * 1024**3]
    print("📏 Size Formatting:")
    for size in sizes:
        formatted = manager._format_size(size)
        print(f"  {size:>12} bytes = {formatted}")
    print()
    
    # Download time estimation
    print("⏱️ Download Time Estimation (50 Mbps):")
    for model_name in manager.supported_models:
        time_est = manager.estimate_download_time(model_name, 50)
        size_gb = manager.supported_models[model_name]['expected_size_gb']
        print(f"  {model_name}: {size_gb}GB → {time_est}")
    print()
    
    # Disk space checking
    print("💾 Disk Space Checking:")
    space_checks = [5, 20, 100]
    for required_gb in space_checks:
        has_space = manager.check_disk_space(required_gb)
        status = "✅ Sufficient" if has_space else "❌ Insufficient"
        print(f"  {required_gb}GB required: {status}")
    print()


def demo_convenience_functions():
    """Demonstrate convenience functions"""
    print("🎯 Convenience Functions Demo")
    print("=" * 35)
    
    print("📦 Available convenience functions:")
    print("  - download_qwen_image(force_redownload=False)")
    print("  - download_qwen2_vl(model_size='7B', force_redownload=False)")
    print("  - check_model_status(model_name)")
    print()
    
    # Example usage (without actually downloading)
    print("💡 Example usage:")
    print("  # Download Qwen-Image model")
    print("  success = download_qwen_image()")
    print()
    print("  # Download Qwen2-VL 7B model")
    print("  success = download_qwen2_vl('7B')")
    print()
    print("  # Check model status")
    print("  status = check_model_status('Qwen/Qwen-Image')")
    print()


def demo_progress_tracking():
    """Demonstrate progress tracking capabilities"""
    print("📈 Progress Tracking Demo")
    print("=" * 30)
    
    manager = ModelDownloadManager()
    
    # Example progress callback
    def progress_callback(progress):
        if progress.total_files > 0:
            percent = (progress.completed_files / progress.total_files) * 100
            print(f"  Progress: {progress.completed_files}/{progress.total_files} files ({percent:.1f}%)")
        
        if progress.total_size_bytes > 0:
            size_percent = (progress.downloaded_bytes / progress.total_size_bytes) * 100
            downloaded_str = manager._format_size(progress.downloaded_bytes)
            total_str = manager._format_size(progress.total_size_bytes)
            print(f"  Downloaded: {downloaded_str}/{total_str} ({size_percent:.1f}%)")
        
        if progress.current_file:
            print(f"  Current file: {progress.current_file}")
        
        if progress.error_message:
            print(f"  Error: {progress.error_message}")
    
    print("📊 Progress callback example:")
    print("  def progress_callback(progress):")
    print("      # Handle progress updates")
    print("      print(f'Progress: {progress.completed_files}/{progress.total_files}')")
    print()
    print("  manager.add_progress_callback(progress_callback)")
    print("  manager.download_model('Qwen/Qwen-Image')")
    print()


def demo_error_handling():
    """Demonstrate error handling capabilities"""
    print("🛡️ Error Handling Demo")
    print("=" * 25)
    
    manager = ModelDownloadManager()
    
    print("🔧 Built-in error handling:")
    print("  ✅ Network interruption recovery")
    print("  ✅ Download resume capability")
    print("  ✅ Disk space validation")
    print("  ✅ Model integrity verification")
    print("  ✅ Graceful cleanup on failure")
    print("  ✅ Repository not found handling")
    print("  ✅ Permission error handling")
    print()
    
    print("💡 Error handling examples:")
    print("  # Network error → automatic retry with exponential backoff")
    print("  # Disk full → clear error message and cleanup")
    print("  # Interrupted download → resume from last checkpoint")
    print("  # Corrupted files → re-download affected files")
    print()


def main():
    """Run all demonstrations"""
    print("🎪 ModelDownloadManager Complete Demo")
    print("=" * 60)
    print()
    
    try:
        demo_basic_usage()
        print()
        
        demo_model_status_checking()
        print()
        
        demo_download_configuration()
        print()
        
        demo_utility_functions()
        print()
        
        demo_convenience_functions()
        print()
        
        demo_progress_tracking()
        print()
        
        demo_error_handling()
        print()
        
        print("✅ Demo completed successfully!")
        print()
        print("🚀 Ready to use ModelDownloadManager!")
        print("   Run with --help to see command-line options")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())