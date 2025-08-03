"""
Neuromorphic Computing API Routes
REST endpoints for Spiking Neural Network operations
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response validation
class SpikePatternRequest(BaseModel):
    neuron_ids: List[str] = Field(..., description="List of neuron IDs to inject spikes")
    pattern: List[float] = Field(..., description="Spike timing pattern (time offsets)")
    amplitude: Optional[float] = Field(1.0, description="Spike amplitude")

class QuantumFieldRequest(BaseModel):
    field_strength: float = Field(..., ge=-1.0, le=1.0, description="Quantum field strength")

class CrystalSyncRequest(BaseModel):
    crystal_phases: List[float] = Field(..., description="Time crystal phases for synchronization")

class NetworkStateResponse(BaseModel):
    num_neurons: int
    num_synapses: int
    total_spikes_processed: int
    network_firing_rate: float
    synchronization_index: float
    plasticity_updates: int
    quantum_field_strength: float
    time_crystal_sync: bool
    quantum_entangled_neurons: int
    sample_neurons: Dict[str, Dict[str, Any]]
    recent_spikes: int
    timestamp: str

class NetworkAnalysisResponse(BaseModel):
    total_spikes: int
    active_neurons: int
    mean_firing_rate: float
    firing_rate_std: float
    burst_events: int
    synchronization_index: float
    quantum_coherence: float
    network_efficiency: float
    timestamp: str

class NeuroHealthResponse(BaseModel):
    status: str
    num_neurons: int
    num_synapses: int
    total_spikes_processed: int
    network_firing_rate: float
    synchronization_index: float
    quantum_integration: bool
    learning_enabled: bool
    real_time_processing: bool
    plasticity_updates: int
    quantum_entangled_fraction: float
    network_efficiency: float
    timestamp: str

# Global reference to neuromorphic processor
neuromorphic_processor = None

def get_neuromorphic_processor():
    """Dependency to get neuromorphic processor instance"""
    global neuromorphic_processor
    if neuromorphic_processor is None:
        raise HTTPException(status_code=503, detail="Neuromorphic processor not initialized")
    return neuromorphic_processor

@router.get("/network-state", response_model=NetworkStateResponse)
async def get_network_state(neuro = Depends(get_neuromorphic_processor)):
    """Get comprehensive state of the neuromorphic network"""
    
    try:
        state = await neuro.get_network_state()
        return NetworkStateResponse(**state)
        
    except Exception as e:
        logger.error(f"Failed to get network state: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/analysis", response_model=NetworkAnalysisResponse)
async def analyze_network_patterns(neuro = Depends(get_neuromorphic_processor)):
    """Analyze network activity patterns and compute metrics"""
    
    try:
        analysis = await neuro.analyze_network_patterns()
        
        if "status" in analysis and analysis["status"] == "no_recent_activity":
            return {
                "message": "No recent network activity to analyze",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return NetworkAnalysisResponse(**analysis)
        
    except Exception as e:
        logger.error(f"Network analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/inject-spikes")
async def inject_spike_pattern(
    request: SpikePatternRequest,
    neuro = Depends(get_neuromorphic_processor)
):
    """Inject specific spike pattern into selected neurons"""
    
    try:
        success = await neuro.inject_spike_pattern(
            neuron_ids=request.neuron_ids,
            pattern=request.pattern,
            amplitude=request.amplitude
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to inject spike pattern")
        
        return {
            "message": "Spike pattern injected successfully",
            "neuron_ids": request.neuron_ids,
            "pattern_length": len(request.pattern),
            "amplitude": request.amplitude,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Spike injection failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/quantum-field")
async def set_quantum_field(
    request: QuantumFieldRequest,
    neuro = Depends(get_neuromorphic_processor)
):
    """Set quantum field strength affecting all neurons"""
    
    try:
        await neuro.set_quantum_field(request.field_strength)
        
        return {
            "message": "Quantum field strength updated",
            "field_strength": request.field_strength,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to set quantum field: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/sync-time-crystals")
async def synchronize_with_time_crystals(
    request: CrystalSyncRequest,
    neuro = Depends(get_neuromorphic_processor)
):
    """Synchronize neuromorphic processing with time crystals"""
    
    try:
        await neuro.synchronize_with_time_crystals(request.crystal_phases)
        
        return {
            "message": "Synchronized with time crystals",
            "crystal_phases": request.crystal_phases,
            "affected_neurons": len([n for n in neuro.neurons.values() if n.quantum_entangled]),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Time crystal synchronization failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/process-events")
async def trigger_event_processing(
    background_tasks: BackgroundTasks,
    neuro = Depends(get_neuromorphic_processor)
):
    """Trigger manual event processing"""
    
    try:
        # Process events in background
        background_tasks.add_task(neuro.process_pending_events)
        
        return {
            "message": "Event processing triggered",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger event processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/neurons")
async def get_neuron_summary(
    limit: int = 100,
    neuro = Depends(get_neuromorphic_processor)
):
    """Get summary information about neurons in the network"""
    
    try:
        neuron_summary = {}
        neuron_items = list(neuro.neurons.items())[:limit]
        
        for neuron_id, neuron in neuron_items:
            neuron_summary[neuron_id] = {
                "membrane_potential": neuron.membrane_potential,
                "threshold": neuron.threshold,
                "firing_rate": neuron.firing_rate,
                "quantum_entangled": neuron.quantum_entangled,
                "num_synapses": len(neuron.synaptic_weights),
                "recent_spikes": len([t for t in neuron.spike_times 
                                   if datetime.utcnow().timestamp() - t < 10.0])
            }
        
        return {
            "neurons": neuron_summary,
            "total_neurons": len(neuro.neurons),
            "shown_neurons": len(neuron_summary),
            "quantum_entangled_count": len([n for n in neuro.neurons.values() if n.quantum_entangled]),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get neuron summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/neuron/{neuron_id}")
async def get_neuron_details(neuron_id: str, neuro = Depends(get_neuromorphic_processor)):
    """Get detailed information about a specific neuron"""
    
    try:
        if neuron_id not in neuro.neurons:
            raise HTTPException(status_code=404, detail=f"Neuron {neuron_id} not found")
        
        neuron = neuro.neurons[neuron_id]
        
        # Get connected synapses
        incoming_synapses = []
        outgoing_synapses = []
        
        for synapse_id, synapse in neuro.synapses.items():
            if synapse.post_neuron_id == neuron_id:
                incoming_synapses.append({
                    "from_neuron": synapse.pre_neuron_id,
                    "weight": synapse.weight,
                    "delay": synapse.delay,
                    "plasticity_type": synapse.plasticity_type
                })
            elif synapse.pre_neuron_id == neuron_id:
                outgoing_synapses.append({
                    "to_neuron": synapse.post_neuron_id,
                    "weight": synapse.weight,
                    "delay": synapse.delay,
                    "plasticity_type": synapse.plasticity_type
                })
        
        # Recent spike history (last 10 seconds)
        current_time = datetime.utcnow().timestamp()
        recent_spikes = [t for t in neuron.spike_times if current_time - t < 10.0]
        
        return {
            "neuron_id": neuron_id,
            "properties": {
                "membrane_potential": neuron.membrane_potential,
                "threshold": neuron.threshold,
                "refractory_period": neuron.refractory_period,
                "refractory_timer": neuron.refractory_timer,
                "adaptation": neuron.adaptation,
                "noise_level": neuron.noise_level,
                "quantum_phase": neuron.quantum_phase,
                "firing_rate": neuron.firing_rate,
                "quantum_entangled": neuron.quantum_entangled,
                "entanglement_partners": neuron.entanglement_partners
            },
            "connectivity": {
                "incoming_synapses": incoming_synapses,
                "outgoing_synapses": outgoing_synapses,
                "total_incoming": len(incoming_synapses),
                "total_outgoing": len(outgoing_synapses)
            },
            "activity": {
                "total_spikes": len(neuron.spike_times),
                "recent_spikes": len(recent_spikes),
                "last_spike": neuron.last_spike,
                "recent_spike_times": recent_spikes[-10:]  # Last 10 recent spikes
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get neuron details: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/synapses")
async def get_synapse_summary(
    limit: int = 100,
    neuro = Depends(get_neuromorphic_processor)
):
    """Get summary information about synapses in the network"""
    
    try:
        synapse_summary = {}
        synapse_items = list(neuro.synapses.items())[:limit]
        
        for synapse_id, synapse in synapse_items:
            synapse_summary[synapse_id] = {
                "pre_neuron": synapse.pre_neuron_id,
                "post_neuron": synapse.post_neuron_id,
                "weight": synapse.weight,
                "delay": synapse.delay,
                "plasticity_type": synapse.plasticity_type,
                "trace": synapse.trace,
                "quantum_modulated": synapse.quantum_modulated
            }
        
        # Calculate statistics
        weights = [s.weight for s in neuro.synapses.values()]
        excitatory_count = len([w for w in weights if w > 0])
        inhibitory_count = len([w for w in weights if w < 0])
        
        return {
            "synapses": synapse_summary,
            "total_synapses": len(neuro.synapses),
            "shown_synapses": len(synapse_summary),
            "statistics": {
                "excitatory_synapses": excitatory_count,
                "inhibitory_synapses": inhibitory_count,
                "average_weight": sum(weights) / len(weights) if weights else 0.0,
                "quantum_modulated_count": len([s for s in neuro.synapses.values() if s.quantum_modulated])
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get synapse summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/metrics")
async def get_neuromorphic_metrics(neuro = Depends(get_neuromorphic_processor)):
    """Get neuromorphic computing performance metrics"""
    
    try:
        state = await neuro.get_network_state()
        analysis = await neuro.analyze_network_patterns()
        
        # Calculate derived metrics
        spike_efficiency = state["total_spikes_processed"] / max(1, state["num_neurons"])
        quantum_integration_level = state["quantum_entangled_neurons"] / state["num_neurons"]
        
        metrics = {
            "performance": {
                "total_spikes_processed": state["total_spikes_processed"],
                "network_firing_rate": state["network_firing_rate"],
                "synchronization_index": state["synchronization_index"],
                "spike_efficiency": spike_efficiency,
                "network_efficiency": analysis.get("network_efficiency", 0.0) if "status" not in analysis else 0.0
            },
            "learning": {
                "plasticity_updates": state["plasticity_updates"],
                "learning_enabled": True,  # From processor initialization
                "homeostatic_scaling": True  # From processor initialization
            },
            "quantum_integration": {
                "quantum_field_strength": state["quantum_field_strength"],
                "quantum_entangled_neurons": state["quantum_entangled_neurons"],
                "quantum_integration_level": quantum_integration_level,
                "time_crystal_sync": state["time_crystal_sync"],
                "quantum_coherence": analysis.get("quantum_coherence", 0.0) if "status" not in analysis else 0.0
            },
            "network_structure": {
                "num_neurons": state["num_neurons"],
                "num_synapses": state["num_synapses"],
                "recent_spikes": state["recent_spikes"]
            },
            "timestamp": state["timestamp"]
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get neuromorphic metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health", response_model=NeuroHealthResponse)
async def neuromorphic_health_check(neuro = Depends(get_neuromorphic_processor)):
    """Comprehensive neuromorphic processor health check"""
    
    try:
        health = await neuro.health_check()
        return NeuroHealthResponse(**health)
        
    except Exception as e:
        logger.error(f"Neuromorphic health check failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/spike-events")
async def get_recent_spike_events(
    limit: int = 100,
    neuro = Depends(get_neuromorphic_processor)
):
    """Get recent spike events from the network"""
    
    try:
        # Get recent spike events (limited to avoid overwhelming response)
        recent_events = list(neuro.spike_events)[-limit:]
        
        events_data = []
        for event in recent_events:
            events_data.append({
                "neuron_id": event.neuron_id,
                "timestamp": event.timestamp,
                "amplitude": event.amplitude,
                "quantum_phase": event.quantum_phase,
                "metadata": event.metadata
            })
        
        return {
            "spike_events": events_data,
            "total_events_shown": len(events_data),
            "total_events_stored": len(neuro.spike_events),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get spike events: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/entanglement-network")
async def get_quantum_entanglement_network(neuro = Depends(get_neuromorphic_processor)):
    """Get the quantum entanglement network topology for neurons"""
    
    try:
        entanglement_map = neuro.quantum_entanglement_network
        
        # Calculate network statistics
        total_entangled = len(entanglement_map)
        total_connections = sum(len(partners) for partners in entanglement_map.values())
        avg_connections = total_connections / total_entangled if total_entangled > 0 else 0
        
        return {
            "entanglement_network": entanglement_map,
            "statistics": {
                "total_entangled_neurons": total_entangled,
                "total_entanglement_connections": total_connections,
                "average_connections": avg_connections,
                "entanglement_density": total_entangled / len(neuro.neurons) if neuro.neurons else 0
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get entanglement network: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Initialize neuromorphic processor reference (called from main app)
def set_neuromorphic_processor_instance(neuro_instance):
    """Set the neuromorphic processor instance for API routes"""
    global neuromorphic_processor
    neuromorphic_processor = neuro_instance