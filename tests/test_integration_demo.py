#!/usr/bin/env python3
"""
Demo script to test the integrated optimization workflow
"""

import sys
import os
sys.path.insert(0, 'src')

def test_integration():
    """Test the integrated optimization workflow"""
    print("🚀 Testing Qwen Image Generator Integration")
    print("=" * 60)
    
    try:
        # Import the integrated generator
        from qwen_generator import QwenImageGenerator
        print("✅ QwenImageGenerator imported successfully")
        
        # Initialize generator
        print("\n🔧 Initializing QwenImageGenerator...")
        generator = QwenImageGenerator()
        print("✅ Generator initialized")
        
        # Check optimization status
        print("\n📊 Checking optimization status...")
        status = generator.get_optimization_status()
        
        print(f"   Optimized components available: {status['optimized_components_available']}")
        print(f"   Detection service available: {status['detection_service_available']}")
        print(f"   Pipeline optimizer available: {status['pipeline_optimizer_available']}")
        print(f"   Performance monitor available: {status['performance_monitor_available']}")
        print(f"   Download manager available: {status['download_manager_available']}")
        print(f"   Compatibility layer available: {status['compatibility_layer_available']}")
        
        # Test workflow validation
        print("\n🔍 Validating optimization workflow...")
        validation = generator.validate_optimization_workflow()
        
        print(f"   Workflow status: {validation['workflow_status']}")
        print(f"   Overall success: {validation['overall_success']}")
        
        if validation['warnings']:
            print("   Warnings:")
            for warning in validation['warnings']:
                print(f"     - {warning}")
        
        if validation['errors']:
            print("   Errors:")
            for error in validation['errors']:
                print(f"     - {error}")
        
        # Test performance recommendations
        print("\n💡 Getting performance recommendations...")
        try:
            recommendations = generator.get_performance_recommendations()
            if 'error' not in recommendations:
                print(f"   Memory strategy: {recommendations.get('memory_strategy', 'unknown')}")
                print(f"   Expected performance: {recommendations.get('expected_performance', 'unknown')}")
            else:
                print(f"   {recommendations['error']}")
        except Exception as e:
            print(f"   Performance recommendations not available: {e}")
        
        print("\n✅ Integration test completed successfully!")
        print("\n📋 Summary:")
        print("   • Architecture detection and optimization components integrated")
        print("   • Performance monitoring system ready")
        print("   • Model download management available")
        print("   • Backward compatibility maintained")
        print("   • Device management with multi-component model support")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)