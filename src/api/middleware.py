#!/usr/bin/env python3
"""
Memory Optimization Middleware for Qwen-Image FastAPI
Advanced GPU memory management and request optimization
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Optional

import torch
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class MemoryOptimizationMiddleware(BaseHTTPMiddleware):
    """
    Advanced memory optimization middleware for GPU-intensive operations
    Features:
    - Automatic memory cleanup after requests
    - Memory threshold monitoring
    - Request prioritization based on memory usage
    - Background memory optimization
    """

    def __init__(
        self,
        app: ASGIApp,
        memory_threshold: float = 0.85,  # 85% VRAM usage threshold
        cleanup_interval: int = 30,  # Cleanup every 30 seconds
        enable_monitoring: bool = True,
    ):
        super().__init__(app)
        self.memory_threshold = memory_threshold
        self.cleanup_interval = cleanup_interval
        self.enable_monitoring = enable_monitoring
        self.request_count = 0
        self.total_memory_freed = 0

        # Start background memory monitor
        if self.enable_monitoring and torch.cuda.is_available():
            asyncio.create_task(self._background_memory_monitor())

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with memory optimization"""

        # Pre-request memory check
        pre_memory = await self._get_memory_info()

        # Check if we're approaching memory limits
        if (
            pre_memory
            and pre_memory.get("usage_percent", 0) > self.memory_threshold * 100
        ):
            logger.warning(
                f"High memory usage detected: {pre_memory['usage_percent']}%"
            )
            await self._aggressive_cleanup()

        # Log request start
        start_time = time.time()
        self.request_count += 1

        try:
            # Process the request
            response = await call_next(request)

            # Post-request memory optimization
            post_memory = await self._get_memory_info()
            processing_time = time.time() - start_time

            # Add memory info to response headers
            if post_memory:
                response.headers[
                    "X-GPU-Memory-Used"
                ] = f"{post_memory['allocated_gb']}GB"
                response.headers[
                    "X-GPU-Memory-Percent"
                ] = f"{post_memory['usage_percent']}%"

            response.headers["X-Processing-Time"] = f"{processing_time:.2f}s"

            # Schedule background cleanup for high-memory requests
            if processing_time > 5.0:  # Long-running requests
                asyncio.create_task(self._schedule_cleanup(delay=2.0))

            return response

        except Exception as e:
            # Emergency cleanup on errors
            await self._emergency_cleanup()
            logger.error(f"Request failed, performed emergency cleanup: {e}")
            raise

    async def _get_memory_info(self) -> Optional[Dict[str, Any]]:
        """Get current GPU memory information"""
        if not torch.cuda.is_available():
            return None

        try:
            allocated = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory

            return {
                "allocated_gb": round(allocated / 1e9, 2),
                "total_gb": round(total / 1e9, 2),
                "usage_percent": round(100 * allocated / total, 1),
                "available_gb": round((total - allocated) / 1e9, 2),
            }
        except Exception:
            return None

    async def _basic_cleanup(self):
        """Basic GPU memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    async def _aggressive_cleanup(self):
        """Aggressive memory cleanup with garbage collection"""
        if torch.cuda.is_available():
            # Multiple cleanup passes
            import gc

            # Pass 1: Basic cleanup
            torch.cuda.empty_cache()
            gc.collect()

            # Pass 2: Force synchronization
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Pass 3: Final cleanup
            gc.collect()
            torch.cuda.empty_cache()

            freed_memory = await self._get_memory_info()
            if freed_memory:
                self.total_memory_freed += freed_memory.get("available_gb", 0)
                logger.info(
                    f"Aggressive cleanup completed. Available: {freed_memory['available_gb']}GB"
                )

    async def _emergency_cleanup(self):
        """Emergency cleanup for error conditions"""
        logger.warning("Performing emergency memory cleanup")
        await self._aggressive_cleanup()

        # Additional emergency measures
        if torch.cuda.is_available():
            try:
                # Clear any remaining allocations
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

                # Force garbage collection multiple times
                import gc

                for _ in range(3):
                    gc.collect()

            except Exception as e:
                logger.error(f"Emergency cleanup failed: {e}")

    async def _schedule_cleanup(self, delay: float = 1.0):
        """Schedule cleanup after delay"""
        await asyncio.sleep(delay)
        await self._basic_cleanup()

    async def _background_memory_monitor(self):
        """Background task for continuous memory monitoring"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)

                memory_info = await self._get_memory_info()
                if memory_info:
                    usage_percent = memory_info["usage_percent"]

                    # Log memory status
                    logger.info(
                        f"Memory monitor: {usage_percent}% used ({memory_info['allocated_gb']}GB)"
                    )

                    # Proactive cleanup if usage is high
                    if usage_percent > self.memory_threshold * 100:
                        logger.info("Proactive memory cleanup triggered")
                        await self._aggressive_cleanup()
                    elif usage_percent > 60:  # Moderate usage
                        await self._basic_cleanup()

            except Exception as e:
                logger.error(f"Background memory monitor error: {e}")


class RequestQueueMiddleware(BaseHTTPMiddleware):
    """
    Request queuing middleware for managing concurrent generation requests
    """

    def __init__(self, app: ASGIApp, max_concurrent: int = 1, queue_size: int = 10):
        super().__init__(app)
        self.max_concurrent = max_concurrent
        self.queue_size = queue_size
        self.active_requests = 0
        self.request_queue = asyncio.Queue(maxsize=queue_size)
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Queue and process requests based on endpoints"""

        # Only queue generation endpoints
        if "/generate/" in str(request.url):
            async with self.semaphore:
                self.active_requests += 1
                try:
                    response = await call_next(request)
                    return response
                finally:
                    self.active_requests -= 1
        else:
            # Non-generation endpoints pass through
            return await call_next(request)


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Performance monitoring middleware for tracking API metrics
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.request_times = []
        self.generation_times = []
        self.error_count = 0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor request performance"""
        start_time = time.time()

        try:
            response = await call_next(request)

            # Record successful request
            processing_time = time.time() - start_time
            self.request_times.append(processing_time)

            # Keep only recent data (last 100 requests)
            if len(self.request_times) > 100:
                self.request_times = self.request_times[-100:]

            # Track generation-specific times
            if "/generate/" in str(request.url):
                self.generation_times.append(processing_time)
                if len(self.generation_times) > 50:
                    self.generation_times = self.generation_times[-50:]

            # Add performance headers
            response.headers["X-Request-Time"] = f"{processing_time:.3f}s"

            if self.request_times:
                avg_time = sum(self.request_times) / len(self.request_times)
                response.headers["X-Avg-Request-Time"] = f"{avg_time:.3f}s"

            return response

        except Exception as e:
            self.error_count += 1
            logger.error(f"Request failed: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.request_times:
            return {"message": "No requests processed yet"}

        avg_request_time = sum(self.request_times) / len(self.request_times)
        max_request_time = max(self.request_times)
        min_request_time = min(self.request_times)

        stats = {
            "total_requests": len(self.request_times),
            "error_count": self.error_count,
            "avg_request_time": round(avg_request_time, 3),
            "max_request_time": round(max_request_time, 3),
            "min_request_time": round(min_request_time, 3),
            "success_rate": round(
                (len(self.request_times) / (len(self.request_times) + self.error_count))
                * 100,
                2,
            ),
        }

        if self.generation_times:
            avg_gen_time = sum(self.generation_times) / len(self.generation_times)
            stats.update(
                {
                    "total_generations": len(self.generation_times),
                    "avg_generation_time": round(avg_gen_time, 3),
                    "max_generation_time": round(max(self.generation_times), 3),
                    "min_generation_time": round(min(self.generation_times), 3),
                }
            )

        return stats


@asynccontextmanager
async def memory_managed_operation():
    """
    Context manager for memory-managed operations
    Usage:
        async with memory_managed_operation():
            # GPU-intensive code here
            result = model.generate(...)
    """
    # Pre-operation cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        yield
    finally:
        # Post-operation cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Additional cleanup for critical operations
            import gc

            gc.collect()
            torch.cuda.empty_cache()


# Utility functions for manual memory management
async def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    if torch.cuda.is_available():
        import gc

        # Multiple cleanup passes
        for _ in range(3):
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()

        # Final cleanup
        torch.cuda.empty_cache()


async def get_memory_pressure() -> str:
    """Get current memory pressure level"""
    if not torch.cuda.is_available():
        return "none"

    allocated = torch.cuda.memory_allocated(0)
    total = torch.cuda.get_device_properties(0).total_memory
    usage_percent = 100 * allocated / total

    if usage_percent > 90:
        return "critical"
    elif usage_percent > 75:
        return "high"
    elif usage_percent > 50:
        return "moderate"
    else:
        return "low"


# Export all middleware classes and utilities
__all__ = [
    "MemoryOptimizationMiddleware",
    "RequestQueueMiddleware",
    "PerformanceMonitoringMiddleware",
    "memory_managed_operation",
    "force_memory_cleanup",
    "get_memory_pressure",
]
