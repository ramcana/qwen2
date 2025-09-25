"""
Performance Monitor for MMDiT Architecture
Provides comprehensive timing and metrics collection for Qwen-Image generation
"""

import time
import logging
import psutil
import torch
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import deque
import json
import threading
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Data class for storing performance metrics"""
    # Timing metrics
    model_load_time: float = 0.0
    generation_time_per_step: float = 0.0
    total_generation_time: float = 0.0
    pipeline_setup_time: float = 0.0
    
    # Memory metrics
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_memory_utilization_percent: float = 0.0
    system_memory_used_gb: float = 0.0
    system_memory_utilization_percent: float = 0.0
    
    # Generation metrics
    num_inference_steps: int = 0
    image_resolution: Tuple[int, int] = (0, 0)
    batch_size: int = 1
    
    # Performance validation
    target_met: bool = False
    target_generation_time: float = 5.0  # seconds
    performance_score: float = 0.0  # 0-100 scale
    
    # Architecture-specific metrics
    architecture_type: str = "MMDiT"
    model_name: str = ""
    torch_dtype: str = "float16"
    device: str = "cuda"
    
    # Additional metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    gpu_name: str = ""
    driver_version: str = ""


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for MMDiT architecture
    Tracks timing, memory usage, and validates performance targets
    """
    
    def __init__(self, target_generation_time: float = 5.0):
        """
        Initialize performance monitor
        
        Args:
            target_generation_time: Target generation time in seconds (default: 5.0)
        """
        self.target_generation_time = target_generation_time
        self.current_metrics = PerformanceMetrics(target_generation_time=target_generation_time)
        
        # Timing state
        self._start_time: Optional[float] = None
        self._step_start_time: Optional[float] = None
        self._step_times: List[float] = []
        
        # Historical data (thread-safe collections)
        self._lock = threading.Lock()
        self._generation_history: deque = deque(maxlen=100)  # Last 100 generations
        self._step_time_history: deque = deque(maxlen=1000)  # Last 1000 steps
        
        # GPU information
        self._gpu_available = torch.cuda.is_available()
        if self._gpu_available:
            self.current_metrics.gpu_name = torch.cuda.get_device_name()
            self.current_metrics.gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.current_metrics.driver_version = pynvml.nvmlSystemGetDriverVersion().decode()
            except ImportError:
                logger.warning("pynvml not available - GPU driver version not tracked")
        
        logger.info(f"PerformanceMonitor initialized with target: {target_generation_time}s")
        if self._gpu_available:
            logger.info(f"GPU: {self.current_metrics.gpu_name} ({self.current_metrics.gpu_memory_total_gb:.1f}GB)")
    
    @contextmanager
    def monitor_generation(self, model_name: str = "", architecture_type: str = "MMDiT", 
                          num_steps: int = 20, resolution: Tuple[int, int] = (1024, 1024)):
        """
        Context manager for monitoring complete generation process
        
        Args:
            model_name: Name of the model being used
            architecture_type: Architecture type (MMDiT or UNet)
            num_steps: Number of inference steps
            resolution: Image resolution (width, height)
        """
        # Initialize metrics for this generation
        self.current_metrics = PerformanceMetrics(
            target_generation_time=self.target_generation_time,
            model_name=model_name,
            architecture_type=architecture_type,
            num_inference_steps=num_steps,
            image_resolution=resolution,
            gpu_name=self.current_metrics.gpu_name,
            gpu_memory_total_gb=self.current_metrics.gpu_memory_total_gb,
            driver_version=self.current_metrics.driver_version
        )
        
        # Start timing
        self.start_timing()
        
        try:
            # Capture initial memory state
            self._capture_memory_metrics()
            
            yield self
            
        finally:
            # End timing and finalize metrics
            self.end_timing()
            self._finalize_generation_metrics()
            
            # Store in history
            with self._lock:
                self._generation_history.append(self.current_metrics)
            
            # Log results
            self._log_generation_results()
    
    def start_timing(self) -> None:
        """Start timing measurement"""
        self._start_time = time.perf_counter()
        self._step_times.clear()
        logger.debug("Performance timing started")
    
    def end_timing(self) -> float:
        """
        End timing measurement and return total time
        
        Returns:
            Total generation time in seconds
        """
        if self._start_time is None:
            logger.warning("end_timing() called without start_timing()")
            return 0.0
        
        total_time = time.perf_counter() - self._start_time
        self.current_metrics.total_generation_time = total_time
        
        # Calculate per-step time if we have step data
        if self._step_times:
            self.current_metrics.generation_time_per_step = sum(self._step_times) / len(self._step_times)
        elif self.current_metrics.num_inference_steps > 0:
            self.current_metrics.generation_time_per_step = total_time / self.current_metrics.num_inference_steps
        
        logger.debug(f"Performance timing ended: {total_time:.3f}s")
        return total_time
    
    def start_step_timing(self) -> None:
        """Start timing for a single generation step"""
        self._step_start_time = time.perf_counter()
    
    def end_step_timing(self) -> float:
        """
        End timing for a single generation step
        
        Returns:
            Step time in seconds
        """
        if self._step_start_time is None:
            logger.warning("end_step_timing() called without start_step_timing()")
            return 0.0
        
        step_time = time.perf_counter() - self._step_start_time
        self._step_times.append(step_time)
        
        # Store in historical data
        with self._lock:
            self._step_time_history.append(step_time)
        
        return step_time
    
    def measure_model_load_time(self, load_function, *args, **kwargs):
        """
        Measure model loading time
        
        Args:
            load_function: Function to call for loading
            *args, **kwargs: Arguments to pass to load_function
            
        Returns:
            Result of load_function and load time
        """
        start_time = time.perf_counter()
        
        try:
            result = load_function(*args, **kwargs)
            load_time = time.perf_counter() - start_time
            self.current_metrics.model_load_time = load_time
            
            logger.info(f"Model loaded in {load_time:.3f}s")
            return result, load_time
            
        except Exception as e:
            load_time = time.perf_counter() - start_time
            logger.error(f"Model loading failed after {load_time:.3f}s: {e}")
            raise
    
    def _capture_memory_metrics(self) -> None:
        """Capture current memory usage metrics"""
        try:
            # GPU memory metrics
            if self._gpu_available:
                gpu_memory_used = torch.cuda.memory_allocated() / 1e9
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9
                
                self.current_metrics.gpu_memory_used_gb = gpu_memory_used
                self.current_metrics.gpu_memory_utilization_percent = (
                    gpu_memory_used / self.current_metrics.gpu_memory_total_gb * 100
                )
                
                logger.debug(f"GPU Memory: {gpu_memory_used:.2f}GB used, "
                           f"{gpu_memory_reserved:.2f}GB reserved")
            
            # System memory metrics
            memory_info = psutil.virtual_memory()
            self.current_metrics.system_memory_used_gb = memory_info.used / 1e9
            self.current_metrics.system_memory_utilization_percent = memory_info.percent
            
        except Exception as e:
            logger.warning(f"Failed to capture memory metrics: {e}")
    
    def _finalize_generation_metrics(self) -> None:
        """Finalize metrics after generation completion"""
        # Capture final memory state
        self._capture_memory_metrics()
        
        # Validate performance against target
        self.current_metrics.target_met = (
            self.current_metrics.total_generation_time <= self.target_generation_time
        )
        
        # Calculate performance score (0-100)
        if self.current_metrics.total_generation_time > 0:
            # Score based on how close we are to target (100 = meets target, 0 = very slow)
            time_ratio = self.target_generation_time / self.current_metrics.total_generation_time
            self.current_metrics.performance_score = min(100.0, time_ratio * 100.0)
        
        # Set device and dtype information
        if self._gpu_available:
            self.current_metrics.device = f"cuda:{torch.cuda.current_device()}"
        else:
            self.current_metrics.device = "cpu"
    
    def _log_generation_results(self) -> None:
        """Log generation performance results"""
        metrics = self.current_metrics
        
        # Performance summary
        status = "✅ TARGET MET" if metrics.target_met else "❌ TARGET MISSED"
        logger.info(f"🚀 Generation Performance Summary - {status}")
        logger.info(f"   Model: {metrics.model_name} ({metrics.architecture_type})")
        logger.info(f"   Total Time: {metrics.total_generation_time:.3f}s (target: {metrics.target_generation_time:.1f}s)")
        logger.info(f"   Per Step: {metrics.generation_time_per_step:.3f}s ({metrics.num_inference_steps} steps)")
        logger.info(f"   Performance Score: {metrics.performance_score:.1f}/100")
        
        # Memory summary
        if self._gpu_available:
            logger.info(f"   GPU Memory: {metrics.gpu_memory_used_gb:.2f}GB / "
                       f"{metrics.gpu_memory_total_gb:.1f}GB ({metrics.gpu_memory_utilization_percent:.1f}%)")
        
        logger.info(f"   System Memory: {metrics.system_memory_used_gb:.2f}GB "
                   f"({metrics.system_memory_utilization_percent:.1f}%)")
        
        # Performance warnings
        if not metrics.target_met:
            self._log_performance_warnings()
    
    def _log_performance_warnings(self) -> None:
        """Log performance warnings and suggestions"""
        metrics = self.current_metrics
        
        logger.warning("⚠️ Performance target not met - Diagnostic information:")
        
        # Time-based diagnostics
        if metrics.total_generation_time > metrics.target_generation_time * 2:
            logger.warning("   - Generation time is significantly slower than target")
            logger.warning("   - Consider checking model architecture and optimizations")
        
        # Memory-based diagnostics
        if self._gpu_available and metrics.gpu_memory_utilization_percent < 50:
            logger.warning("   - Low GPU memory utilization - possible CPU bottleneck")
        
        if metrics.gpu_memory_utilization_percent > 90:
            logger.warning("   - High GPU memory usage - possible memory fragmentation")
        
        # Architecture-specific diagnostics
        if metrics.architecture_type == "MMDiT":
            if metrics.generation_time_per_step > 5.0:
                logger.warning("   - MMDiT step time too high - check attention optimizations")
                logger.warning("   - Ensure AttnProcessor2_0 is NOT used (causes tensor unpacking issues)")
        
        # Suggestions
        logger.warning("💡 Performance improvement suggestions:")
        logger.warning("   - Verify correct model is loaded (Qwen-Image vs Qwen-Image-Edit)")
        logger.warning("   - Check GPU optimizations (TF32, cuDNN benchmark)")
        logger.warning("   - Disable memory-saving features on high-VRAM GPUs")
        logger.warning("   - Ensure pipeline is using AutoPipelineForText2Image")
    
    def validate_performance_target(self, generation_time: float) -> bool:
        """
        Validate if generation time meets performance target
        
        Args:
            generation_time: Generation time to validate
            
        Returns:
            True if target is met, False otherwise
        """
        target_met = generation_time <= self.target_generation_time
        
        if target_met:
            logger.info(f"✅ Performance target met: {generation_time:.3f}s <= {self.target_generation_time:.1f}s")
        else:
            logger.warning(f"❌ Performance target missed: {generation_time:.3f}s > {self.target_generation_time:.1f}s")
        
        return target_met
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return self.current_metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary
        
        Returns:
            Dictionary containing performance summary
        """
        metrics = self.current_metrics
        
        summary = {
            "current_generation": {
                "total_time": metrics.total_generation_time,
                "per_step_time": metrics.generation_time_per_step,
                "target_met": metrics.target_met,
                "performance_score": metrics.performance_score,
                "model_name": metrics.model_name,
                "architecture": metrics.architecture_type,
                "steps": metrics.num_inference_steps,
                "resolution": f"{metrics.image_resolution[0]}x{metrics.image_resolution[1]}"
            },
            "memory_usage": {
                "gpu_memory_gb": metrics.gpu_memory_used_gb,
                "gpu_utilization_percent": metrics.gpu_memory_utilization_percent,
                "system_memory_gb": metrics.system_memory_used_gb,
                "system_utilization_percent": metrics.system_memory_utilization_percent
            },
            "hardware_info": {
                "gpu_name": metrics.gpu_name,
                "gpu_total_memory_gb": metrics.gpu_memory_total_gb,
                "device": metrics.device,
                "driver_version": metrics.driver_version
            },
            "target_configuration": {
                "target_time": self.target_generation_time,
                "torch_dtype": metrics.torch_dtype
            }
        }
        
        # Add historical statistics if available
        with self._lock:
            if self._generation_history:
                recent_times = [m.total_generation_time for m in self._generation_history]
                recent_scores = [m.performance_score for m in self._generation_history]
                
                summary["historical_performance"] = {
                    "total_generations": len(self._generation_history),
                    "avg_generation_time": sum(recent_times) / len(recent_times),
                    "min_generation_time": min(recent_times),
                    "max_generation_time": max(recent_times),
                    "avg_performance_score": sum(recent_scores) / len(recent_scores),
                    "success_rate": sum(1 for m in self._generation_history if m.target_met) / len(self._generation_history) * 100
                }
            
            if self._step_time_history:
                step_times = list(self._step_time_history)
                summary["step_performance"] = {
                    "total_steps": len(step_times),
                    "avg_step_time": sum(step_times) / len(step_times),
                    "min_step_time": min(step_times),
                    "max_step_time": max(step_times)
                }
        
        return summary
    
    def get_before_after_comparison(self, before_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """
        Compare current metrics with previous metrics
        
        Args:
            before_metrics: Previous performance metrics
            
        Returns:
            Comparison dictionary
        """
        current = self.current_metrics
        
        # Calculate improvements
        time_improvement = (before_metrics.total_generation_time - current.total_generation_time) / before_metrics.total_generation_time * 100
        step_improvement = (before_metrics.generation_time_per_step - current.generation_time_per_step) / before_metrics.generation_time_per_step * 100
        
        comparison = {
            "before": {
                "total_time": before_metrics.total_generation_time,
                "per_step_time": before_metrics.generation_time_per_step,
                "target_met": before_metrics.target_met,
                "performance_score": before_metrics.performance_score,
                "model": before_metrics.model_name
            },
            "after": {
                "total_time": current.total_generation_time,
                "per_step_time": current.generation_time_per_step,
                "target_met": current.target_met,
                "performance_score": current.performance_score,
                "model": current.model_name
            },
            "improvement": {
                "total_time_percent": time_improvement,
                "per_step_time_percent": step_improvement,
                "performance_score_change": current.performance_score - before_metrics.performance_score,
                "target_status_change": current.target_met and not before_metrics.target_met
            },
            "summary": {
                "faster": time_improvement > 0,
                "improvement_factor": before_metrics.total_generation_time / current.total_generation_time if current.total_generation_time > 0 else 0,
                "meets_target_now": current.target_met,
                "significant_improvement": time_improvement > 50  # 50%+ improvement
            }
        }
        
        return comparison
    
    def log_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log custom performance metrics
        
        Args:
            metrics: Dictionary of metric names and values
        """
        logger.info("📊 Custom Performance Metrics:")
        for name, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"   {name}: {value:.3f}")
            else:
                logger.info(f"   {name}: {value}")
    
    def export_metrics_to_json(self, filepath: str) -> None:
        """
        Export current metrics to JSON file
        
        Args:
            filepath: Path to save JSON file
        """
        try:
            # Convert dataclass to dictionary
            metrics_dict = {
                "current_metrics": self.current_metrics.__dict__,
                "performance_summary": self.get_performance_summary(),
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics_dict, f, indent=2, default=str)
            
            logger.info(f"Performance metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def reset_metrics(self) -> None:
        """Reset current metrics and timing state"""
        self.current_metrics = PerformanceMetrics(target_generation_time=self.target_generation_time)
        self._start_time = None
        self._step_start_time = None
        self._step_times.clear()
        
        logger.debug("Performance metrics reset")
    
    def clear_history(self) -> None:
        """Clear historical performance data"""
        with self._lock:
            self._generation_history.clear()
            self._step_time_history.clear()
        
        logger.info("Performance history cleared")


# Utility functions for easy integration

def create_mmdit_performance_monitor(target_time: float = 5.0) -> PerformanceMonitor:
    """
    Create a performance monitor optimized for MMDiT architecture
    
    Args:
        target_time: Target generation time in seconds
        
    Returns:
        Configured PerformanceMonitor instance
    """
    monitor = PerformanceMonitor(target_generation_time=target_time)
    logger.info(f"MMDiT Performance Monitor created with {target_time}s target")
    return monitor


@contextmanager
def monitor_generation_performance(model_name: str = "", target_time: float = 5.0, 
                                 architecture_type: str = "MMDiT", num_steps: int = 20,
                                 resolution: Tuple[int, int] = (1024, 1024)):
    """
    Convenience context manager for monitoring generation performance
    
    Args:
        model_name: Name of the model
        target_time: Target generation time
        architecture_type: Architecture type
        num_steps: Number of inference steps
        resolution: Image resolution
        
    Yields:
        PerformanceMonitor instance
    """
    monitor = PerformanceMonitor(target_generation_time=target_time)
    
    with monitor.monitor_generation(
        model_name=model_name,
        architecture_type=architecture_type,
        num_steps=num_steps,
        resolution=resolution
    ) as perf_monitor:
        yield perf_monitor