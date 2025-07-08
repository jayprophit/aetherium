"""
Unit tests for core infrastructure services.
"""

import unittest
from unittest.mock import MagicMock
from systems.core.infrastructure.base_service import BaseService, StorageService, MessagingService

class TestBaseService(unittest.TestCase):
    """Test cases for BaseService."""
    
    def test_abstract_methods(self):
        """Test that abstract methods are implemented in concrete subclasses."""
        # Create a concrete subclass for BaseService
        class ConcreteService(BaseService):
            def start(self):
                pass
            def stop(self):
                pass
            def status(self):
                return {}
        
        # Instantiate and test
        service = ConcreteService(config={})
        service.start()
        service.stop()
        self.assertIsInstance(service.status(), dict)

class TestStorageService(unittest.TestCase):
    """Test cases for StorageService."""
    
    def test_abstract_methods(self):
        """Test that abstract methods are implemented in concrete subclasses."""
        class ConcreteStorage(StorageService):
            def start(self):
                pass
            def stop(self):
                pass
            def status(self):
                return {}
            def save(self, key, data):
                pass
            def load(self, key):
                return None
            def delete(self, key):
                pass
        
        storage = ConcreteStorage(config={})
        storage.save("test", "data")
        self.assertIsNone(storage.load("test"))
        storage.delete("test")

class TestMessagingService(unittest.TestCase):
    """Test cases for MessagingService."""
    
    def test_abstract_methods(self):
        """Test that abstract methods are implemented in concrete subclasses."""
        class ConcreteMessaging(MessagingService):
            def start(self):
                pass
            def stop(self):
                pass
            def status(self):
                return {}
            def publish(self, topic, message):
                pass
            def subscribe(self, topic, callback):
                pass
        
        messaging = ConcreteMessaging(config={})
        messaging.publish("topic", "message")
        messaging.subscribe("topic", lambda x: x)
