"""
Quantum Computing API Routes
REST endpoints for Virtual Quantum Computer operations
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response validation
class QuantumCircuitRequest(BaseModel):
    template_name: str = Field(..., description="Name of quantum circuit template")
    parameters: Optional[List[float]] = Field(None, description="Circuit parameters")
    custom_gates: Optional[List[Dict[str, Any]]] = Field(None, description="Custom gates to apply")
    shots: Optional[int] = Field(1024, description="Number of shots for execution")
    backend: Optional[str] = Field(None, description="Backend to use for execution")

class QuantumCircuitResponse(BaseModel):
    state_id: str
    counts: Dict[str, int]
    probabilities: Dict[str, float]
    statevector: Optional[List[complex]]
    fidelity: float
    coherence_enhancement: float
    execution_time: str
    shots: int
    backend: str

class QuantumStateResponse(BaseModel):
    state_id: str
    amplitudes: List[float]
    phases: List[float]
    entanglement_measure: float
    coherence_time: float
    timestamp: str
    crystal_sync: bool

class QuantumHealthResponse(BaseModel):
    status: str
    num_qubits: int
    total_operations: int
    successful_optimizations: int
    average_fidelity: float
    coherence_enhancement: float
    time_crystal_sync: bool
    quantum_states_stored: int
    circuit_templates: List[str]
    current_backend: str
    timestamp: str

# Global reference to quantum computer (injected from main app)
quantum_computer = None

def get_quantum_computer():
    """Dependency to get quantum computer instance"""
    global quantum_computer
    if quantum_computer is None:
        raise HTTPException(status_code=503, detail="Quantum computer not initialized")
    return quantum_computer

@router.post("/create-circuit", response_model=QuantumCircuitResponse)
async def create_and_execute_circuit(
    request: QuantumCircuitRequest,
    qc = Depends(get_quantum_computer)
):
    """Create and execute a quantum circuit from template"""
    
    try:
        # Create quantum circuit
        circuit = await qc.create_quantum_circuit(
            template_name=request.template_name,
            parameters=request.parameters,
            custom_gates=request.custom_gates
        )
        
        # Execute circuit
        result = await qc.execute_circuit(
            circuit=circuit,
            shots=request.shots,
            backend_name=request.backend
        )
        
        return QuantumCircuitResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Circuit creation/execution failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/templates")
async def get_circuit_templates(qc = Depends(get_quantum_computer)):
    """Get available quantum circuit templates"""
    
    try:
        templates = {}
        for name, template in qc.circuit_templates.items():
            templates[name] = {
                "name": template.name,
                "num_qubits": template.num_qubits,
                "depth": template.depth,
                "gates": template.gates,
                "num_parameters": len(template.parameters),
                "optimization_target": template.optimization_target
            }
        
        return {
            "templates": templates,
            "total_templates": len(templates),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get templates: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/state/{state_id}", response_model=QuantumStateResponse)
async def get_quantum_state(state_id: str, qc = Depends(get_quantum_computer)):
    """Get detailed information about a quantum state"""
    
    try:
        state_info = await qc.get_quantum_state(state_id)
        
        if state_info is None:
            raise HTTPException(status_code=404, detail=f"State {state_id} not found")
        
        return QuantumStateResponse(**state_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quantum state: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/states")
async def list_quantum_states(qc = Depends(get_quantum_computer)):
    """List all stored quantum states"""
    
    try:
        state_ids = await qc.list_quantum_states()
        
        return {
            "state_ids": state_ids,
            "total_states": len(state_ids),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to list quantum states: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/states/cleanup")
async def cleanup_old_states(
    max_age_hours: int = 24,
    qc = Depends(get_quantum_computer)
):
    """Clean up old quantum states"""
    
    try:
        removed_count = await qc.clear_old_states(max_age_hours)
        
        return {
            "removed_states": removed_count,
            "max_age_hours": max_age_hours,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup states: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/optimize")
async def trigger_optimization(
    background_tasks: BackgroundTasks,
    qc = Depends(get_quantum_computer)
):
    """Trigger quantum circuit optimization"""
    
    try:
        # Run optimization in background
        background_tasks.add_task(qc.optimize_circuits)
        
        return {
            "message": "Quantum circuit optimization started",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/metrics")
async def get_quantum_metrics(qc = Depends(get_quantum_computer)):
    """Get quantum computing performance metrics"""
    
    try:
        health = await qc.health_check()
        
        # Additional computed metrics
        success_rate = health["successful_optimizations"] / max(1, health["total_operations"])
        coherence_improvement = health["coherence_enhancement"] - 1.0
        
        metrics = {
            "performance": {
                "total_operations": health["total_operations"],
                "successful_optimizations": health["successful_optimizations"],
                "success_rate": success_rate,
                "average_fidelity": health["average_fidelity"]
            },
            "coherence": {
                "enhancement_factor": health["coherence_enhancement"],
                "improvement_percentage": coherence_improvement * 100,
                "time_crystal_sync": health["time_crystal_sync"]
            },
            "resources": {
                "num_qubits": health["num_qubits"],
                "quantum_states_stored": health["quantum_states_stored"],
                "circuit_templates": len(health["circuit_templates"])
            },
            "timestamp": health["timestamp"]
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get quantum metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health", response_model=QuantumHealthResponse)
async def quantum_health_check(qc = Depends(get_quantum_computer)):
    """Comprehensive quantum computer health check"""
    
    try:
        health = await qc.health_check()
        return QuantumHealthResponse(**health)
        
    except Exception as e:
        logger.error(f"Quantum health check failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/execute-custom")
async def execute_custom_circuit(
    circuit_definition: Dict[str, Any],
    shots: int = 1024,
    backend: Optional[str] = None,
    qc = Depends(get_quantum_computer)
):
    """Execute a custom quantum circuit from raw definition"""
    
    try:
        # This would require additional circuit parsing logic
        # For now, return a placeholder response
        
        return {
            "message": "Custom circuit execution not yet implemented",
            "circuit_definition": circuit_definition,
            "shots": shots,
            "backend": backend,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Custom circuit execution failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/backends")
async def get_available_backends(qc = Depends(get_quantum_computer)):
    """Get list of available quantum computing backends"""
    
    try:
        backends = {}
        for name, backend in qc.backends.items():
            backends[name] = {
                "name": name,
                "type": str(type(backend).__name__),
                "description": getattr(backend, 'description', 'Quantum simulator backend'),
                "max_qubits": getattr(backend, 'configuration', lambda: type('obj', (object,), {'n_qubits': qc.num_qubits}))().n_qubits if hasattr(getattr(backend, 'configuration', lambda: None)(), 'n_qubits') else qc.num_qubits
            }
        
        return {
            "backends": backends,
            "current_backend": str(qc.current_backend),
            "total_backends": len(backends),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get backends: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/switch-backend")
async def switch_backend(
    backend_name: str,
    qc = Depends(get_quantum_computer)
):
    """Switch to a different quantum computing backend"""
    
    try:
        if backend_name not in qc.backends:
            raise HTTPException(status_code=400, detail=f"Backend {backend_name} not available")
        
        old_backend = str(qc.current_backend)
        qc.current_backend = qc.backends[backend_name]
        
        return {
            "message": f"Switched from {old_backend} to {backend_name}",
            "old_backend": old_backend,
            "new_backend": backend_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backend switch failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Initialize quantum computer reference (called from main app)
def set_quantum_computer_instance(qc_instance):
    """Set the quantum computer instance for API routes"""
    global quantum_computer
    quantum_computer = qc_instance