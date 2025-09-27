"""
Simple Service Manager for testing
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration"""
    NOT_INITIALIZED = "not_initialized"
    READY = "ready"
    OFFLINE = "offline"


class SimpleServiceManager:
    """Simple service manager for testing"""
    
    def __init__(self):
        """Initialize service manager"""
        self.services: Dict[str, Any] = {}
        self.is_running = False
        logger.info("SimpleServiceManager initialized")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "total_services": len(self.services),
            "management_active": self.is_running
        }
    
    def start_management(self) -> None:
        """Start management"""
        self.is_running = True
        logger.info("Management started")
    
    def stop_management(self) -> None:
        """Stop management"""
        self.is_running = False
        logger.info("Management stopped")


def get_simple_service_manager() -> SimpleServiceManager:
    """Get simple service manager"""
    return SimpleServiceManager()