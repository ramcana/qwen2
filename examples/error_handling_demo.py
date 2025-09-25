#!/usr/bin/env python3
"""
Demonstration of comprehensive error handling and diagnostics system
Shows how the error handler works with different types of errors and architectures
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from error_handler import (
    ArchitectureAwareErrorHandler,
    handle_download_error,
    handle_pipeline_error,
    get_system_diagnostics,
    create_diagnostic_report
)


def demo_system_diagnostics():
    """Demonstrate system diagnostics"""
    print("=" * 60)
    print("🔍 SYSTEM DIAGNOSTICS DEMO")
    print("=" * 60)
    
    # Get system diagnostics
    diagnostics = get_system_diagnostics()
    
    print(f"GPU Available: {diagnostics.gpu_available}")
    print(f"GPU Memory: {diagnostics.gpu_memory_gb:.1f}GB")
    print(f"System Memory: {diagnostics.system_memory_gb:.1f}GB")
    print(f"Disk Space: {diagnostics.disk_space_gb:.1f}GB")
    print(f"CUDA Version: {diagnostics.cuda_version}")
    print(f"PyTorch Version: {diagnostics.pytorch_version}")
    
    print("\nArchitecture Support:")
    for arch, supported in diagnostics.architecture_support.items():
        status = "✅" if supported else "❌"
        print(f"  {status} {arch}")
    
    print(f"\nNetwork Connectivity: {'✅' if diagnostics.network_connectivity else '❌'}")
    print(f"Permissions OK: {'✅' if diagnostics.permissions_ok else '❌'}")


def demo_download_error_handling():
    """Demonstrate download error handling"""
    print("\n" + "=" * 60)
    print("📥 DOWNLOAD ERROR HANDLING DEMO")
    print("=" * 60)
    
    # Simulate different download errors
    test_errors = [
        (ConnectionError("Network connection failed"), "Network Error"),
        (OSError("No space left on device"), "Disk Space Error"),
        (PermissionError("Access denied"), "Permission Error"),
        (Exception("Repository not found: 404"), "Repository Error"),
        (Exception("Corrupted file detected"), "Corruption Error")
    ]
    
    for error, error_type in test_errors:
        print(f"\n🚨 Simulating {error_type}:")
        print(f"Error: {error}")
        
        # Handle the error
        error_info = handle_download_error(error, "Qwen/Qwen-Image")
        
        print(f"Category: {error_info.category.value}")
        print(f"Severity: {error_info.severity.value}")
        print(f"Message: {error_info.message}")
        print(f"Suggested Fixes:")
        for i, fix in enumerate(error_info.suggested_fixes[:3], 1):
            print(f"  {i}. {fix}")
        
        if error_info.user_feedback:
            print(f"User Feedback: {error_info.user_feedback}")


def demo_pipeline_error_handling():
    """Demonstrate pipeline error handling"""
    print("\n" + "=" * 60)
    print("⚙️ PIPELINE ERROR HANDLING DEMO")
    print("=" * 60)
    
    # Simulate architecture-specific errors
    test_scenarios = [
        (IndexError("tuple index out of range"), "models/Qwen-Image", "MMDiT", "Tensor Unpacking"),
        (RuntimeError("Flash attention not compatible"), "models/Qwen-Image", "MMDiT", "Attention Issue"),
        (RuntimeError("CUDA out of memory"), "models/stable-diffusion", "UNet", "Memory Issue"),
        (FileNotFoundError("Model file not found"), "models/missing-model", "Unknown", "File Missing"),
        (ValueError("Invalid configuration"), "models/test-model", "Unknown", "Configuration Error")
    ]
    
    for error, model_path, architecture, scenario in test_scenarios:
        print(f"\n🚨 Simulating {scenario} ({architecture}):")
        print(f"Error: {error}")
        print(f"Model: {model_path}")
        
        # Handle the error
        error_info = handle_pipeline_error(error, model_path, architecture)
        
        print(f"Category: {error_info.category.value}")
        print(f"Severity: {error_info.severity.value}")
        print(f"Architecture Context: {error_info.architecture_context}")
        print(f"Suggested Fixes:")
        for i, fix in enumerate(error_info.suggested_fixes[:3], 1):
            print(f"  {i}. {fix}")
        
        if error_info.user_feedback:
            print(f"User Feedback: {error_info.user_feedback}")


def demo_recovery_workflow():
    """Demonstrate complete error recovery workflow"""
    print("\n" + "=" * 60)
    print("🔄 RECOVERY WORKFLOW DEMO")
    print("=" * 60)
    
    # Create error handler with user feedback
    handler = ArchitectureAwareErrorHandler()
    
    feedback_messages = []
    def capture_feedback(message):
        feedback_messages.append(message)
        print(f"📢 User Feedback: {message}")
    
    handler.add_user_feedback_callback(capture_feedback)
    
    # Simulate MMDiT tensor unpacking error
    print("🚨 Simulating MMDiT tensor unpacking error...")
    error = IndexError("tuple index out of range")
    model_path = "models/Qwen-Image"
    architecture_type = "MMDiT"
    
    # Handle error
    error_info = handler.handle_pipeline_error(error, model_path, architecture_type)
    
    # Log error
    handler.log_error(error_info)
    
    print(f"\n📊 Error Analysis:")
    print(f"Category: {error_info.category.value}")
    print(f"Severity: {error_info.severity.value}")
    print(f"Architecture Context: {error_info.architecture_context}")
    print(f"Recovery Actions Available: {len(error_info.recovery_actions) if error_info.recovery_actions else 0}")
    
    # Execute recovery actions (mocked)
    print(f"\n🔧 Executing recovery actions...")
    if error_info.recovery_actions:
        # Mock the recovery actions to return success
        for action in error_info.recovery_actions:
            action.__defaults__ = (True,) if action.__defaults__ else (True,)
        
        recovery_success = handler.execute_recovery_actions(error_info)
        print(f"Recovery Result: {'✅ Success' if recovery_success else '❌ Failed'}")
    
    print(f"\n📝 Feedback Messages Received: {len(feedback_messages)}")
    for msg in feedback_messages:
        print(f"  • {msg}")


def demo_diagnostic_report():
    """Demonstrate diagnostic report generation"""
    print("\n" + "=" * 60)
    print("📋 DIAGNOSTIC REPORT DEMO")
    print("=" * 60)
    
    # Create some test errors first
    handler = ArchitectureAwareErrorHandler()
    
    # Add some test errors to history
    test_errors = [
        handler.handle_download_error(ConnectionError("Network error"), "Qwen/Qwen-Image"),
        handler.handle_pipeline_error(IndexError("tensor unpacking"), "models/Qwen-Image", "MMDiT"),
        handler.handle_pipeline_error(RuntimeError("CUDA out of memory"), "models/test", "UNet")
    ]
    
    for error_info in test_errors:
        handler.log_error(error_info)
    
    # Generate comprehensive report
    report = create_diagnostic_report()
    
    print(f"Report Timestamp: {report['timestamp']}")
    print(f"\n🖥️ System Information:")
    for key, value in report['system_info'].items():
        print(f"  {key}: {value}")
    
    print(f"\n🏗️ Architecture Support:")
    for arch, supported in report['architecture_support'].items():
        status = "✅" if supported else "❌"
        print(f"  {status} {arch}")
    
    print(f"\n🌐 Connectivity:")
    for key, value in report['connectivity'].items():
        status = "✅" if value else "❌"
        print(f"  {status} {key}")
    
    print(f"\n🚨 Recent Errors ({len(report['recent_errors'])}):")
    for i, error in enumerate(report['recent_errors'], 1):
        print(f"  {i}. [{error['severity'].upper()}] {error['category']}: {error['message']}")
        if error['architecture_context']:
            print(f"     Architecture: {error['architecture_context']}")
    
    print(f"\n💡 Recommendations ({len(report['recommendations'])}):")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")


def demo_architecture_specific_handling():
    """Demonstrate architecture-specific error handling"""
    print("\n" + "=" * 60)
    print("🏗️ ARCHITECTURE-SPECIFIC HANDLING DEMO")
    print("=" * 60)
    
    handler = ArchitectureAwareErrorHandler()
    
    # MMDiT specific issues
    print("🔸 MMDiT Architecture Issues:")
    mmdit_errors = [
        ("tensor unpacking", IndexError("tuple index out of range")),
        ("attention issues", RuntimeError("Flash attention not compatible")),
        ("parameter mismatch", TypeError("unexpected keyword argument 'guidance_scale'"))
    ]
    
    for issue_type, error in mmdit_errors:
        print(f"\n  • {issue_type.title()}:")
        error_info = handler.handle_pipeline_error(error, "models/Qwen-Image", "MMDiT")
        print(f"    Context: {error_info.architecture_context}")
        print(f"    Key Fix: {error_info.suggested_fixes[0] if error_info.suggested_fixes else 'N/A'}")
    
    # UNet specific issues
    print(f"\n🔸 UNet Architecture Issues:")
    unet_errors = [
        ("memory issues", RuntimeError("CUDA out of memory")),
        ("attention compatibility", AttributeError("'UNet2DConditionModel' has no attribute 'set_attn_processor'"))
    ]
    
    for issue_type, error in unet_errors:
        print(f"\n  • {issue_type.title()}:")
        error_info = handler.handle_pipeline_error(error, "models/stable-diffusion", "UNet")
        print(f"    Context: {error_info.architecture_context}")
        print(f"    Key Fix: {error_info.suggested_fixes[0] if error_info.suggested_fixes else 'N/A'}")
    
    # Fallback strategies
    print(f"\n🔧 Fallback Strategies:")
    mmdit_fallback = handler._apply_architecture_fallback("test/path", "MMDiT")
    unet_fallback = handler._apply_architecture_fallback("test/path", "UNet")
    
    print(f"  MMDiT Fallback:")
    for key, value in mmdit_fallback.items():
        print(f"    {key}: {value}")
    
    print(f"  UNet Fallback:")
    for key, value in unet_fallback.items():
        print(f"    {key}: {value}")


def main():
    """Run all error handling demos"""
    print("🚀 COMPREHENSIVE ERROR HANDLING AND DIAGNOSTICS DEMO")
    print("This demo showcases the architecture-aware error handling system")
    
    try:
        demo_system_diagnostics()
        demo_download_error_handling()
        demo_pipeline_error_handling()
        demo_recovery_workflow()
        demo_diagnostic_report()
        demo_architecture_specific_handling()
        
        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("The error handling system provides:")
        print("• Comprehensive error detection and classification")
        print("• Architecture-aware error handling (MMDiT vs UNet)")
        print("• Automatic recovery strategies")
        print("• Detailed diagnostics and reporting")
        print("• User feedback and guidance")
        print("• Troubleshooting for common issues")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()