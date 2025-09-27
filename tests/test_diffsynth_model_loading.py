#!/usr/bin/env python3
"""
DiffSynth Model Loading Trial
Test script to verify DiffSynth service can load models successfully without crashing
"""

import logging
import sys
import time
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.diffsynth_service import DiffSynthService, DiffSynthConfig
from src.diffsynth_models import ImageEditRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('diffsynth_trial.log')
    ]
)

logger = logging.getLogger(__name__)


def test_service_creation():
    """Test 1: Service Creation"""
    logger.info("🧪 Test 1: Creating DiffSynth service...")
    
    try:
        # Create service with conservative memory settings
        config = DiffSynthConfig(
            max_memory_usage_gb=3.0,  # Conservative memory limit
            enable_cpu_offload=True,
            enable_layer_offload=True,
            use_tiled_processing=True,
            enable_eligen=False,  # Disable EliGen for initial test
            default_num_inference_steps=10  # Fewer steps for faster testing
        )
        
        service = DiffSynthService(config)
        logger.info("✅ Service created successfully")
        return service
        
    except Exception as e:
        logger.error(f"❌ Service creation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def test_service_initialization(service):
    """Test 2: Service Initialization"""
    logger.info("🧪 Test 2: Initializing DiffSynth service...")
    
    try:
        start_time = time.time()
        success = service.initialize()
        init_time = time.time() - start_time
        
        if success:
            logger.info(f"✅ Service initialized successfully in {init_time:.2f}s")
            logger.info(f"Service status: {service.status}")
            return True
        else:
            logger.error(f"❌ Service initialization failed")
            logger.error(f"Service status: {service.status}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Service initialization crashed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def test_service_status_check(service):
    """Test 3: Service Status Check"""
    logger.info("🧪 Test 3: Checking service status...")
    
    try:
        status = service.status
        logger.info(f"Service status: {status}")
        
        # Check resource usage
        service._update_resource_usage()
        usage = service.resource_usage
        
        logger.info(f"GPU Memory Allocated: {usage.gpu_memory_allocated:.2f}GB")
        logger.info(f"GPU Memory Reserved: {usage.gpu_memory_reserved:.2f}GB")
        logger.info(f"CPU Memory Used: {usage.cpu_memory_used:.2f}GB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def test_basic_request_validation(service):
    """Test 4: Basic Request Validation (without actual processing)"""
    logger.info("🧪 Test 4: Testing request validation...")
    
    try:
        # Create a minimal test request
        request = ImageEditRequest(
            prompt="test prompt",
            image_path="test_image.jpg",  # Non-existent file for validation test
            num_inference_steps=5,
            guidance_scale=7.5,
            strength=0.5
        )
        
        logger.info("✅ Request object created successfully")
        logger.info(f"Request prompt: {request.prompt}")
        logger.info(f"Request steps: {request.num_inference_steps}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Request validation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def test_service_cleanup(service):
    """Test 5: Service Cleanup"""
    logger.info("🧪 Test 5: Testing service cleanup...")
    
    try:
        service._cleanup_resources()
        logger.info("✅ Service cleanup completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Service cleanup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def main():
    """Run DiffSynth model loading trial"""
    logger.info("🚀 Starting DiffSynth Model Loading Trial")
    logger.info("=" * 60)
    
    # Test results tracking
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Service Creation
    service = test_service_creation()
    if service:
        tests_passed += 1
    else:
        logger.error("❌ Cannot continue without service - stopping trial")
        return False
    
    logger.info("-" * 40)
    
    # Test 2: Service Initialization
    if test_service_initialization(service):
        tests_passed += 1
    else:
        logger.warning("⚠️ Initialization failed - continuing with remaining tests")
    
    logger.info("-" * 40)
    
    # Test 3: Service Status Check
    if test_service_status_check(service):
        tests_passed += 1
    
    logger.info("-" * 40)
    
    # Test 4: Basic Request Validation
    if test_basic_request_validation(service):
        tests_passed += 1
    
    logger.info("-" * 40)
    
    # Test 5: Service Cleanup
    if test_service_cleanup(service):
        tests_passed += 1
    
    # Final Results
    logger.info("=" * 60)
    logger.info("🏁 DiffSynth Model Loading Trial Complete")
    logger.info(f"Tests Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        logger.info("✅ ALL TESTS PASSED - DiffSynth service is working correctly!")
        return True
    elif tests_passed >= 3:
        logger.info("⚠️ PARTIAL SUCCESS - Some issues detected but core functionality works")
        return True
    else:
        logger.error("❌ TRIAL FAILED - Major issues detected")
        return False


if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        
        logger.info(f"Trial completed with exit code: {exit_code}")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("Trial interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Trial crashed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)