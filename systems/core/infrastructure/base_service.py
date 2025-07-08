"""
Base Infrastructure Service

This module defines the foundational infrastructure components.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseService(ABC):
    """Base class for all infrastructure services."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the service with configuration."""
        self.config = config
        self.is_running = False
    
    @abstractmethod
    def start(self) -> None:
        """Start the service."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the service."""
        pass
    
    @abstractmethod
    def status(self) -> Dict[str, Any]:
        """Get current service status."""
        pass

class StorageService(BaseService):
    """Abstract storage service interface."""
    
    @abstractmethod
    def save(self, key: str, data: Any) -> None:
        """Save data with the given key."""
        pass
    
    @abstractmethod
    def load(self, key: str) -> Any:
        """Load data by key."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete data by key."""
        pass

class MessagingService(BaseService):
    """Abstract messaging service interface."""
    
    @abstractmethod
    def publish(self, topic: str, message: Any) -> None:
        """Publish a message to a topic."""
        pass
    
    @abstractmethod
    def subscribe(self, topic: str, callback: callable) -> None:
        """Subscribe to a topic with a callback."""
        pass
