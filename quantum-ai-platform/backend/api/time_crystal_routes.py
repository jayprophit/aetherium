"""
Time Crystal API Routes
REST endpoints for Time Crystal Engine operations
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response validation
class CrystalParametersRequest(BaseModel):
    crystal_id: str = Field(..., description="ID of the crystal to modify")
    parameters: Dict[str, Any] = Field(..., description="Parameters to set")

class AddCrystalRequest(BaseModel):
    crystal_id: str = Field(..., description="ID for the new crystal")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Initial parameters")

class CrystalNetworkResponse(BaseModel):
    crystals: Dict[str, Dict[str, Any]]
    global_coherence: float
    collective_phase: float
    sync_operations: int
    average_coherence: float
    coupling_strength: float
    quantum_sync_enabled: bool
    network_topology: Dict[str, List[str]]
    timestamp: str

class CrystalHealthResponse(BaseModel):
    status: str
    num_crystals: int
    global_coherence: float
    sync_operations: int
    quantum_sync_enabled: bool
    average_energy: float
    phase_synchronization: float
    coupling_strength: float
    temperature: float
    coherence_trend: str
    timestamp: str

class CoherenceEnhancementRequest(BaseModel):
    target_coherence: float = Field(0.95, ge=0.0, le=1.0, description="Target coherence level")

# Global reference to time crystal engine
time_crystal_engine = None

def get_time_crystal_engine():
    """Dependency to get time crystal engine instance"""
    global time_crystal_engine
    if time_crystal_engine is None:
        raise HTTPException(status_code=503, detail="Time crystal engine not initialized")
    return time_crystal_engine

@router.get("/network-state", response_model=CrystalNetworkResponse)
async def get_crystal_network_state(tce = Depends(get_time_crystal_engine)):
    """Get comprehensive state of the time crystal network"""
    
    try:
        state = await tce.get_crystal_network_state()
        return CrystalNetworkResponse(**state)
        
    except Exception as e:
        logger.error(f"Failed to get crystal network state: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/synchronize")
async def trigger_synchronization(
    background_tasks: BackgroundTasks,
    tce = Depends(get_time_crystal_engine)
):
    """Trigger manual crystal synchronization"""
    
    try:
        # Run synchronization in background
        background_tasks.add_task(tce.synchronize_crystals)
        
        return {
            "message": "Crystal synchronization triggered",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger synchronization: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/enhance-coherence")
async def enhance_coherence(
    request: CoherenceEnhancementRequest,
    tce = Depends(get_time_crystal_engine)
):
    """Enhance quantum coherence of the crystal network"""
    
    try:
        success = await tce.enhance_quantum_coherence(request.target_coherence)
        
        # Get updated state
        state = await tce.get_crystal_network_state()
        
        return {
            "success": success,
            "target_coherence": request.target_coherence,
            "achieved_coherence": state["global_coherence"],
            "improvement": state["global_coherence"] - (1.0 - request.target_coherence),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Coherence enhancement failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/crystal/parameters")
async def set_crystal_parameters(
    request: CrystalParametersRequest,
    tce = Depends(get_time_crystal_engine)
):
    """Set parameters for a specific time crystal"""
    
    try:
        success = await tce.set_crystal_parameters(request.crystal_id, request.parameters)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Crystal {request.crystal_id} not found")
        
        return {
            "message": f"Parameters updated for crystal {request.crystal_id}",
            "crystal_id": request.crystal_id,
            "parameters": request.parameters,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set crystal parameters: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/crystal/add")
async def add_crystal(
    request: AddCrystalRequest,
    tce = Depends(get_time_crystal_engine)
):
    """Add a new crystal to the network"""
    
    try:
        success = await tce.add_crystal(request.crystal_id, request.parameters)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Crystal {request.crystal_id} already exists")
        
        return {
            "message": f"Crystal {request.crystal_id} added to network",
            "crystal_id": request.crystal_id,
            "parameters": request.parameters,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add crystal: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/crystal/{crystal_id}")
async def remove_crystal(crystal_id: str, tce = Depends(get_time_crystal_engine)):
    """Remove a crystal from the network"""
    
    try:
        success = await tce.remove_crystal(crystal_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Crystal {crystal_id} not found")
        
        return {
            "message": f"Crystal {crystal_id} removed from network",
            "crystal_id": crystal_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove crystal: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/crystals")
async def list_crystals(tce = Depends(get_time_crystal_engine)):
    """List all crystals in the network"""
    
    try:
        state = await tce.get_crystal_network_state()
        crystals = state["crystals"]
        
        crystal_summary = {}
        for crystal_id, crystal_data in crystals.items():
            crystal_summary[crystal_id] = {
                "energy": crystal_data["energy"],
                "phase": crystal_data["phase"],
                "frequency": crystal_data["frequency"],
                "coherence": crystal_data["coherence"],
                "quantum_sync": crystal_data["quantum_sync"],
                "entanglement_partners": crystal_data["entanglement_partners"]
            }
        
        return {
            "crystals": crystal_summary,
            "total_crystals": len(crystals),
            "global_coherence": state["global_coherence"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to list crystals: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/crystal/{crystal_id}")
async def get_crystal_details(crystal_id: str, tce = Depends(get_time_crystal_engine)):
    """Get detailed information about a specific crystal"""
    
    try:
        state = await tce.get_crystal_network_state()
        crystals = state["crystals"]
        
        if crystal_id not in crystals:
            raise HTTPException(status_code=404, detail=f"Crystal {crystal_id} not found")
        
        crystal_data = crystals[crystal_id]
        
        return {
            "crystal_id": crystal_id,
            "details": crystal_data,
            "network_context": {
                "global_coherence": state["global_coherence"],
                "collective_phase": state["collective_phase"],
                "entanglement_network": state["network_topology"].get(crystal_id, [])
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get crystal details: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/metrics")
async def get_crystal_metrics(tce = Depends(get_time_crystal_engine)):
    """Get time crystal performance metrics"""
    
    try:
        health = await tce.health_check()
        state = await tce.get_crystal_network_state()
        
        # Calculate additional metrics
        phase_variance = 0.0
        energy_variance = 0.0
        coherence_values = []
        
        for crystal_data in state["crystals"].values():
            coherence_values.append(crystal_data["coherence"])
        
        if coherence_values:
            coherence_std = sum((c - sum(coherence_values)/len(coherence_values))**2 for c in coherence_values) / len(coherence_values)
            coherence_std = coherence_std ** 0.5
        else:
            coherence_std = 0.0
        
        metrics = {
            "synchronization": {
                "global_coherence": health["global_coherence"],
                "phase_synchronization": health["phase_synchronization"],
                "sync_operations": health["sync_operations"],
                "coherence_trend": health["coherence_trend"]
            },
            "network": {
                "num_crystals": health["num_crystals"],
                "coupling_strength": health["coupling_strength"],
                "quantum_sync_enabled": health["quantum_sync_enabled"],
                "coherence_std_dev": coherence_std
            },
            "physics": {
                "average_energy": health["average_energy"],
                "temperature": health["temperature"],
                "collective_phase": state["collective_phase"]
            },
            "timestamp": health["timestamp"]
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get crystal metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health", response_model=CrystalHealthResponse)
async def crystal_health_check(tce = Depends(get_time_crystal_engine)):
    """Comprehensive time crystal engine health check"""
    
    try:
        health = await tce.health_check()
        return CrystalHealthResponse(**health)
        
    except Exception as e:
        logger.error(f"Crystal health check failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/config/coupling-strength")
async def set_coupling_strength(
    coupling_strength: float = Field(..., ge=1e-22, le=1e-18),
    tce = Depends(get_time_crystal_engine)
):
    """Set the coupling strength between crystals"""
    
    try:
        old_strength = tce.config.coupling_strength
        tce.config.coupling_strength = coupling_strength
        
        return {
            "message": "Coupling strength updated",
            "old_strength": old_strength,
            "new_strength": coupling_strength,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to set coupling strength: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/config/temperature")
async def set_system_temperature(
    temperature: float = Field(..., ge=0.001, le=1.0),
    tce = Depends(get_time_crystal_engine)
):
    """Set the system temperature affecting all crystals"""
    
    try:
        old_temperature = tce.config.temperature
        tce.config.temperature = temperature
        
        # Update temperature for all crystals
        for crystal in tce.crystals.values():
            crystal.temperature = temperature
        
        return {
            "message": "System temperature updated",
            "old_temperature": old_temperature,
            "new_temperature": temperature,
            "affected_crystals": len(tce.crystals),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to set temperature: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/entanglement-network")
async def get_entanglement_network(tce = Depends(get_time_crystal_engine)):
    """Get the quantum entanglement network topology"""
    
    try:
        state = await tce.get_crystal_network_state()
        
        # Calculate network statistics
        total_connections = sum(len(partners) for partners in state["network_topology"].values())
        avg_connections = total_connections / len(state["network_topology"]) if state["network_topology"] else 0
        
        # Find most connected crystal
        most_connected = ""
        max_connections = 0
        for crystal_id, partners in state["network_topology"].items():
            if len(partners) > max_connections:
                max_connections = len(partners)
                most_connected = crystal_id
        
        return {
            "network_topology": state["network_topology"],
            "statistics": {
                "total_crystals": len(state["network_topology"]),
                "total_connections": total_connections,
                "average_connections": avg_connections,
                "most_connected_crystal": most_connected,
                "max_connections": max_connections
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get entanglement network: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Initialize time crystal engine reference (called from main app)
def set_time_crystal_engine_instance(tce_instance):
    """Set the time crystal engine instance for API routes"""
    global time_crystal_engine
    time_crystal_engine = tce_instance