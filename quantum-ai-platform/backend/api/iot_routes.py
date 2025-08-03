"""
IoT Integration API Routes
REST endpoints for IoT device management and quantum system integration
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response validation
class IoTDeviceRequest(BaseModel):
    device_id: str = Field(..., description="Unique device identifier")
    device_type: str = Field(..., description="Type of IoT device")
    connection_config: Dict[str, Any] = Field(..., description="Connection configuration")
    quantum_enabled: Optional[bool] = Field(False, description="Enable quantum integration")

class MQTTMessageRequest(BaseModel):
    topic: str = Field(..., description="MQTT topic")
    message: str = Field(..., description="Message content")
    qos: Optional[int] = Field(0, description="Quality of Service level")

class QuantumIoTSyncRequest(BaseModel):
    device_ids: List[str] = Field(..., description="Device IDs to sync with quantum state")
    sync_type: str = Field(..., description="Type of synchronization")

class IoTDeviceResponse(BaseModel):
    device_id: str
    device_type: str
    status: str
    connection_status: str
    quantum_enabled: bool
    last_seen: str
    data_points: int
    quantum_correlation: Optional[float]

class IoTMetricsResponse(BaseModel):
    total_devices: int
    connected_devices: int
    quantum_enabled_devices: int
    total_data_points: int
    average_latency: float
    connection_success_rate: float
    quantum_sync_rate: float
    timestamp: str

# Global reference to IoT manager
iot_manager = None

def get_iot_manager():
    """Dependency to get IoT manager instance"""
    global iot_manager
    if iot_manager is None:
        raise HTTPException(status_code=503, detail="IoT manager not initialized")
    return iot_manager

@router.post("/devices/register")
async def register_iot_device(
    request: IoTDeviceRequest,
    iot = Depends(get_iot_manager)
):
    """Register a new IoT device"""
    
    try:
        success = await iot.register_device(
            device_id=request.device_id,
            device_type=request.device_type,
            config=request.connection_config,
            quantum_enabled=request.quantum_enabled
        )
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to register device '{request.device_id}'"
            )
        
        return {
            "message": f"Device '{request.device_id}' registered successfully",
            "device_id": request.device_id,
            "device_type": request.device_type,
            "quantum_enabled": request.quantum_enabled,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Device registration failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/devices")
async def list_iot_devices(iot = Depends(get_iot_manager)):
    """List all registered IoT devices"""
    
    try:
        devices = await iot.get_all_devices()
        
        device_list = []
        for device_id, device_info in devices.items():
            device_list.append(IoTDeviceResponse(
                device_id=device_id,
                device_type=device_info.get("type", "unknown"),
                status=device_info.get("status", "offline"),
                connection_status=device_info.get("connection_status", "disconnected"),
                quantum_enabled=device_info.get("quantum_enabled", False),
                last_seen=device_info.get("last_seen", "never"),
                data_points=device_info.get("data_points", 0),
                quantum_correlation=device_info.get("quantum_correlation")
            ))
        
        return {
            "devices": device_list,
            "total_devices": len(device_list),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to list IoT devices: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/devices/{device_id}")
async def get_iot_device_details(device_id: str, iot = Depends(get_iot_manager)):
    """Get detailed information about a specific IoT device"""
    
    try:
        device_info = await iot.get_device_details(device_id)
        
        if not device_info:
            raise HTTPException(status_code=404, detail=f"Device '{device_id}' not found")
        
        return {
            "device_id": device_id,
            "details": device_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get device details: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/devices/{device_id}")
async def unregister_iot_device(device_id: str, iot = Depends(get_iot_manager)):
    """Unregister an IoT device"""
    
    try:
        success = await iot.unregister_device(device_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Device '{device_id}' not found")
        
        return {
            "message": f"Device '{device_id}' unregistered successfully",
            "device_id": device_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Device unregistration failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/mqtt/publish")
async def publish_mqtt_message(
    request: MQTTMessageRequest,
    iot = Depends(get_iot_manager)
):
    """Publish message to MQTT topic"""
    
    try:
        success = await iot.publish_message(
            topic=request.topic,
            message=request.message,
            qos=request.qos
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to publish MQTT message")
        
        return {
            "message": "MQTT message published successfully",
            "topic": request.topic,
            "qos": request.qos,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MQTT publish failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/mqtt/topics")
async def get_mqtt_topics(iot = Depends(get_iot_manager)):
    """Get list of active MQTT topics"""
    
    try:
        topics = await iot.get_active_topics()
        
        return {
            "topics": topics,
            "total_topics": len(topics),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get MQTT topics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/quantum-sync")
async def synchronize_quantum_iot(
    request: QuantumIoTSyncRequest,
    background_tasks: BackgroundTasks,
    iot = Depends(get_iot_manager)
):
    """Synchronize IoT devices with quantum systems"""
    
    try:
        # Start quantum-IoT synchronization in background
        background_tasks.add_task(
            iot.sync_with_quantum_systems,
            request.device_ids,
            request.sync_type
        )
        
        return {
            "message": "Quantum-IoT synchronization started",
            "device_ids": request.device_ids,
            "sync_type": request.sync_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Quantum-IoT sync failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/data/stream/{device_id}")
async def get_device_data_stream(
    device_id: str,
    limit: int = 100,
    iot = Depends(get_iot_manager)
):
    """Get recent data stream from an IoT device"""
    
    try:
        data_stream = await iot.get_device_data_stream(device_id, limit)
        
        if data_stream is None:
            raise HTTPException(status_code=404, detail=f"Device '{device_id}' not found")
        
        return {
            "device_id": device_id,
            "data_stream": data_stream,
            "total_points": len(data_stream),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get device data stream: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/control/{device_id}")
async def control_iot_device(
    device_id: str,
    commands: Dict[str, Any],
    iot = Depends(get_iot_manager)
):
    """Send control commands to an IoT device"""
    
    try:
        result = await iot.send_device_commands(device_id, commands)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Device '{device_id}' not found or not responsive"
            )
        
        return {
            "message": f"Commands sent to device '{device_id}'",
            "device_id": device_id,
            "commands": commands,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Device control failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/metrics", response_model=IoTMetricsResponse)
async def get_iot_metrics(iot = Depends(get_iot_manager)):
    """Get IoT system performance metrics"""
    
    try:
        metrics = await iot.get_system_metrics()
        return IoTMetricsResponse(**metrics)
        
    except Exception as e:
        logger.error(f"Failed to get IoT metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health")
async def iot_health_check(iot = Depends(get_iot_manager)):
    """IoT system health check"""
    
    try:
        health = await iot.health_check()
        
        return {
            "status": health.get("status", "unknown"),
            "mqtt_connected": health.get("mqtt_connected", False),
            "total_devices": health.get("total_devices", 0),
            "connected_devices": health.get("connected_devices", 0),
            "quantum_sync_active": health.get("quantum_sync_active", False),
            "last_message": health.get("last_message", "none"),
            "uptime": health.get("uptime", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"IoT health check failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/discovery/scan")
async def scan_for_devices(
    scan_duration: int = 30,
    background_tasks: BackgroundTasks,
    iot = Depends(get_iot_manager)
):
    """Scan for new IoT devices on the network"""
    
    try:
        # Start device discovery in background
        background_tasks.add_task(iot.discover_devices, scan_duration)
        
        return {
            "message": "Device discovery scan started",
            "scan_duration": scan_duration,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Device discovery failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/discovery/results")
async def get_discovery_results(iot = Depends(get_iot_manager)):
    """Get results from the latest device discovery scan"""
    
    try:
        results = await iot.get_discovery_results()
        
        return {
            "discovered_devices": results.get("devices", []),
            "scan_completed": results.get("completed", False),
            "scan_started": results.get("started_at", "unknown"),
            "total_discovered": len(results.get("devices", [])),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get discovery results: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/webhooks/{device_id}")
async def iot_device_webhook(
    device_id: str,
    payload: Dict[str, Any],
    iot = Depends(get_iot_manager)
):
    """Webhook endpoint for IoT device data submission"""
    
    try:
        success = await iot.process_webhook_data(device_id, payload)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process webhook data from device '{device_id}'"
            )
        
        return {
            "message": f"Webhook data processed for device '{device_id}'",
            "device_id": device_id,
            "data_points": len(payload) if isinstance(payload, dict) else 1,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Initialize IoT manager reference (called from main app)
def set_iot_manager_instance(iot_instance):
    """Set the IoT manager instance for API routes"""
    global iot_manager
    iot_manager = iot_instance