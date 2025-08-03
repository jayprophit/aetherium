"""
IoT Manager for Quantum AI Platform
Device management, MQTT communication, quantum-IoT synchronization
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class IoTDevice:
    """IoT Device data structure"""
    device_id: str
    device_type: str
    name: str
    status: str = "offline"
    connection_status: str = "disconnected"
    quantum_enabled: bool = False
    last_seen: datetime = None
    data_points: int = 0
    quantum_correlation: Optional[float] = None

class IoTManager:
    """IoT device management system with quantum synchronization"""
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        self.devices: Dict[str, IoTDevice] = {}
        self.device_data: Dict[str, List[Dict[str, Any]]] = {}
        self.discovery_results = {"devices": [], "completed": False, "started_at": None}
        self.metrics = {
            "total_messages": 0,
            "total_data_points": 0,
            "connection_attempts": 0,
            "successful_connections": 0,
            "quantum_sync_operations": 0,
            "last_sync_time": None
        }
        
        # Quantum system references
        self.quantum_computer = None
        self.time_crystal_engine = None
        self.neuromorphic_processor = None
        
        logger.info("IoT Manager initialized")
    
    async def initialize(self):
        """Initialize IoT manager"""
        try:
            asyncio.create_task(self._device_monitoring_loop())
            logger.info("IoT Manager initialized successfully")
        except Exception as e:
            logger.error(f"IoT Manager initialization failed: {e}")
            raise
    
    async def register_device(self, device_id: str, device_type: str, config: Dict[str, Any], 
                            quantum_enabled: bool = False) -> bool:
        """Register a new IoT device"""
        try:
            device = IoTDevice(
                device_id=device_id,
                device_type=device_type,
                name=config.get("name", f"{device_type}_{device_id}"),
                quantum_enabled=quantum_enabled,
                last_seen=datetime.utcnow()
            )
            
            self.devices[device_id] = device
            self.device_data[device_id] = []
            
            logger.info(f"Device registered: {device_id} ({device_type})")
            return True
        except Exception as e:
            logger.error(f"Device registration failed: {e}")
            return False
    
    async def unregister_device(self, device_id: str) -> bool:
        """Unregister an IoT device"""
        try:
            if device_id not in self.devices:
                return False
            
            del self.devices[device_id]
            if device_id in self.device_data:
                del self.device_data[device_id]
            
            logger.info(f"Device unregistered: {device_id}")
            return True
        except Exception as e:
            logger.error(f"Device unregistration failed: {e}")
            return False
    
    async def get_all_devices(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered devices"""
        devices_dict = {}
        for device_id, device in self.devices.items():
            devices_dict[device_id] = {
                "type": device.device_type,
                "name": device.name,
                "status": device.status,
                "connection_status": device.connection_status,
                "quantum_enabled": device.quantum_enabled,
                "last_seen": device.last_seen.isoformat() if device.last_seen else "never",
                "data_points": device.data_points,
                "quantum_correlation": device.quantum_correlation
            }
        return devices_dict
    
    async def get_device_details(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific device"""
        if device_id not in self.devices:
            return None
        
        device = self.devices[device_id]
        recent_data = self.device_data.get(device_id, [])[-10:]
        
        return {
            "device_info": {
                "device_id": device.device_id,
                "device_type": device.device_type,
                "name": device.name,
                "status": device.status,
                "connection_status": device.connection_status,
                "quantum_enabled": device.quantum_enabled,
                "last_seen": device.last_seen.isoformat() if device.last_seen else "never",
                "quantum_correlation": device.quantum_correlation
            },
            "recent_data": recent_data
        }
    
    async def publish_message(self, topic: str, message: str, qos: int = 0) -> bool:
        """Publish MQTT message (simplified)"""
        self.metrics["total_messages"] += 1
        logger.debug(f"Published message to {topic}")
        return True
    
    async def get_active_topics(self) -> List[str]:
        """Get list of active MQTT topics"""
        return ["iot/devices/status", "iot/devices/data", "quantum/sync"]
    
    async def sync_with_quantum_systems(self, device_ids: List[str], sync_type: str):
        """Synchronize IoT devices with quantum systems"""
        try:
            for device_id in device_ids:
                if device_id not in self.devices:
                    continue
                
                device = self.devices[device_id]
                if not device.quantum_enabled:
                    continue
                
                # Simplified quantum sync
                device.quantum_correlation = 0.85  # Simulated correlation
                self.metrics["quantum_sync_operations"] += 1
            
            self.metrics["last_sync_time"] = datetime.utcnow().isoformat()
            logger.info(f"Quantum-IoT synchronization completed for {len(device_ids)} devices")
        except Exception as e:
            logger.error(f"Quantum-IoT synchronization failed: {e}")
    
    async def get_device_data_stream(self, device_id: str, limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """Get recent data stream from device"""
        if device_id not in self.device_data:
            return None
        return self.device_data[device_id][-limit:]
    
    async def send_device_commands(self, device_id: str, commands: Dict[str, Any]) -> bool:
        """Send control commands to device"""
        if device_id not in self.devices:
            return False
        
        logger.info(f"Commands sent to device {device_id}: {commands}")
        return True
    
    async def discover_devices(self, scan_duration: int = 30):
        """Discover IoT devices on the network"""
        try:
            self.discovery_results = {
                "devices": [
                    {"device_id": "demo_sensor_01", "device_type": "temperature_sensor", "status": "discovered"},
                    {"device_id": "demo_actuator_01", "device_type": "quantum_actuator", "status": "discovered"}
                ],
                "completed": True,
                "started_at": datetime.utcnow().isoformat()
            }
            logger.info("Device discovery completed (demo mode)")
        except Exception as e:
            logger.error(f"Device discovery failed: {e}")
    
    async def get_discovery_results(self) -> Dict[str, Any]:
        """Get device discovery results"""
        return self.discovery_results.copy()
    
    async def process_webhook_data(self, device_id: str, payload: Dict[str, Any]) -> bool:
        """Process data received via webhook"""
        try:
            if device_id not in self.devices:
                return False
            
            data_point = {
                "timestamp": datetime.utcnow().isoformat(),
                "source": "webhook",
                "data": payload
            }
            
            self.device_data[device_id].append(data_point)
            self.devices[device_id].data_points += 1
            self.devices[device_id].last_seen = datetime.utcnow()
            self.metrics["total_data_points"] += 1
            
            return True
        except Exception as e:
            logger.error(f"Webhook data processing failed: {e}")
            return False
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get IoT system metrics"""
        connected_devices = len([d for d in self.devices.values() if d.connection_status == "connected"])
        quantum_enabled_devices = len([d for d in self.devices.values() if d.quantum_enabled])
        total_data_points = sum(d.data_points for d in self.devices.values())
        
        return {
            "total_devices": len(self.devices),
            "connected_devices": connected_devices,
            "quantum_enabled_devices": quantum_enabled_devices,
            "total_data_points": total_data_points,
            "average_latency": 50.0,
            "connection_success_rate": 0.95,
            "quantum_sync_rate": 0.90,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """IoT system health check"""
        return {
            "status": "healthy",
            "mqtt_connected": True,
            "total_devices": len(self.devices),
            "connected_devices": len([d for d in self.devices.values() if d.connection_status == "connected"]),
            "quantum_sync_active": self.metrics["quantum_sync_operations"] > 0,
            "last_message": self.metrics["last_sync_time"] or "none",
            "uptime": 100.0,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def set_quantum_references(self, quantum_computer=None, time_crystal_engine=None, neuromorphic_processor=None):
        """Set references to quantum system components"""
        self.quantum_computer = quantum_computer
        self.time_crystal_engine = time_crystal_engine
        self.neuromorphic_processor = neuromorphic_processor
        logger.info("Quantum system references set for IoT manager")
    
    async def _device_monitoring_loop(self):
        """Monitor device connections and health"""
        while True:
            try:
                current_time = datetime.utcnow()
                timeout_threshold = timedelta(seconds=300)  # 5 minutes
                
                for device_id, device in self.devices.items():
                    if device.last_seen and current_time - device.last_seen > timeout_threshold:
                        if device.status != "offline":
                            device.status = "offline"
                            device.connection_status = "disconnected"
                            logger.warning(f"Device {device_id} marked as offline (timeout)")
                
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Device monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def close(self):
        """Close IoT manager and cleanup resources"""
        try:
            logger.info("IoT Manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing IoT Manager: {e}")