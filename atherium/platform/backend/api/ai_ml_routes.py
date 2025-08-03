"""
AI/ML Hybrid Optimization API Routes
REST endpoints for quantum-classical-neuromorphic AI optimization
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from ai_ml.hybrid_optimizer import OptimizationTarget

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response validation
class OptimizationTaskRequest(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the optimization task")
    target: str = Field(..., description="Optimization target type")
    parameters: Dict[str, Any] = Field(..., description="Initial parameters to optimize")
    constraints: Dict[str, List[float]] = Field(..., description="Parameter constraints [min, max]")
    max_iterations: Optional[int] = Field(100, description="Maximum optimization iterations")

class OptimizationStatusResponse(BaseModel):
    active_tasks: Dict[str, Dict[str, Any]]
    total_optimizations: int
    successful_optimizations: int
    success_rate: float
    completed_tasks: int
    timestamp: str

class OptimizationHealthResponse(BaseModel):
    status: str
    active_tasks: int
    completed_tasks: int
    total_optimizations: int
    success_rate: float
    components_connected: Dict[str, bool]
    timestamp: str

# Global reference to hybrid optimizer
hybrid_optimizer = None

def get_hybrid_optimizer():
    """Dependency to get hybrid optimizer instance"""
    global hybrid_optimizer
    if hybrid_optimizer is None:
        raise HTTPException(status_code=503, detail="Hybrid optimizer not initialized")
    return hybrid_optimizer

@router.post("/optimize/start")
async def start_optimization_task(
    request: OptimizationTaskRequest,
    optimizer = Depends(get_hybrid_optimizer)
):
    """Start a new optimization task"""
    
    try:
        # Validate optimization target
        try:
            target_enum = OptimizationTarget(request.target)
        except ValueError:
            valid_targets = [t.value for t in OptimizationTarget]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid target '{request.target}'. Valid targets: {valid_targets}"
            )
        
        # Convert constraints to tuples
        constraints = {}
        for param_name, bounds in request.constraints.items():
            if len(bounds) != 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"Constraint for {param_name} must have exactly 2 values [min, max]"
                )
            constraints[param_name] = (bounds[0], bounds[1])
        
        # Start optimization task
        success = await optimizer.start_optimization_task(
            task_id=request.task_id,
            target=target_enum,
            parameters=request.parameters,
            constraints=constraints,
            max_iterations=request.max_iterations
        )
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to start optimization task '{request.task_id}' (may already exist)"
            )
        
        return {
            "message": f"Optimization task '{request.task_id}' started successfully",
            "task_id": request.task_id,
            "target": request.target,
            "max_iterations": request.max_iterations,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start optimization task: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/optimize/step/{task_id}")
async def run_optimization_step(
    task_id: str,
    optimizer = Depends(get_hybrid_optimizer)
):
    """Run a single optimization step for the specified task"""
    
    try:
        success = await optimizer.run_optimization_step(task_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Optimization task '{task_id}' not found or not running"
            )
        
        # Get updated task status
        status = await optimizer.get_optimization_status()
        task_status = status["active_tasks"].get(task_id, {})
        
        return {
            "message": f"Optimization step completed for task '{task_id}'",
            "task_id": task_id,
            "current_status": task_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to run optimization step: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/optimize/run/{task_id}")
async def run_optimization_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    max_steps: Optional[int] = None,
    optimizer = Depends(get_hybrid_optimizer)
):
    """Run optimization task to completion (or max_steps) in background"""
    
    async def run_task():
        """Background task to run optimization"""
        try:
            steps_run = 0
            max_task_steps = max_steps or 1000
            
            while steps_run < max_task_steps:
                success = await optimizer.run_optimization_step(task_id)
                if not success:
                    break  # Task completed or failed
                steps_run += 1
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Background optimization task failed: {e}")
    
    try:
        # Check if task exists
        status = await optimizer.get_optimization_status()
        if task_id not in status["active_tasks"]:
            raise HTTPException(
                status_code=404,
                detail=f"Optimization task '{task_id}' not found"
            )
        
        # Start background optimization
        background_tasks.add_task(run_task)
        
        return {
            "message": f"Optimization task '{task_id}' started in background",
            "task_id": task_id,
            "max_steps": max_steps,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start background optimization: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/optimize/status", response_model=OptimizationStatusResponse)
async def get_optimization_status(optimizer = Depends(get_hybrid_optimizer)):
    """Get status of all optimization tasks"""
    
    try:
        status = await optimizer.get_optimization_status()
        return OptimizationStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Failed to get optimization status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/optimize/task/{task_id}")
async def get_task_details(task_id: str, optimizer = Depends(get_hybrid_optimizer)):
    """Get detailed information about a specific optimization task"""
    
    try:
        status = await optimizer.get_optimization_status()
        
        # Check active tasks
        if task_id in status["active_tasks"]:
            task_info = status["active_tasks"][task_id]
            task_info["status"] = "active"
            return {
                "task_id": task_id,
                "details": task_info,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Check completed tasks
        completed_task = None
        for task in optimizer.completed_tasks:
            if task.id == task_id:
                completed_task = task
                break
        
        if completed_task:
            return {
                "task_id": task_id,
                "details": {
                    "target": completed_task.target.value,
                    "iterations": completed_task.iterations_completed,
                    "max_iterations": completed_task.max_iterations,
                    "best_value": completed_task.current_best_value,
                    "best_parameters": completed_task.current_best_params,
                    "status": completed_task.status,
                    "started_at": completed_task.started_at.isoformat(),
                    "completed_at": completed_task.completed_at.isoformat() if completed_task.completed_at else None
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task details: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/optimize/task/{task_id}")
async def stop_optimization_task(task_id: str, optimizer = Depends(get_hybrid_optimizer)):
    """Stop an active optimization task"""
    
    try:
        if task_id not in optimizer.active_tasks:
            raise HTTPException(status_code=404, detail=f"Active task '{task_id}' not found")
        
        # Mark task as stopped
        task = optimizer.active_tasks[task_id]
        task.status = "stopped"
        
        # Move to completed tasks
        optimizer.completed_tasks.append(task)
        del optimizer.active_tasks[task_id]
        
        return {
            "message": f"Optimization task '{task_id}' stopped",
            "task_id": task_id,
            "final_value": task.current_best_value,
            "iterations_completed": task.iterations_completed,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop optimization task: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/targets")
async def get_optimization_targets():
    """Get available optimization targets"""
    
    try:
        targets = {}
        for target in OptimizationTarget:
            targets[target.value] = {
                "name": target.value,
                "description": f"Optimize {target.value.replace('_', ' ')}",
                "requires_quantum": "quantum" in target.value,
                "requires_crystals": "crystal" in target.value,
                "requires_neuromorphic": "neuromorphic" in target.value
            }
        
        return {
            "targets": targets,
            "total_targets": len(targets),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get optimization targets: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/metrics")
async def get_optimization_metrics(optimizer = Depends(get_hybrid_optimizer)):
    """Get optimization performance metrics"""
    
    try:
        status = await optimizer.get_optimization_status()
        
        # Calculate additional metrics
        active_task_types = {}
        for task_info in status["active_tasks"].values():
            target = task_info["target"]
            active_task_types[target] = active_task_types.get(target, 0) + 1
        
        # Analyze recent performance
        recent_tasks = optimizer.completed_tasks[-10:]  # Last 10 completed tasks
        avg_iterations = sum(t.iterations_completed for t in recent_tasks) / len(recent_tasks) if recent_tasks else 0
        avg_improvement = sum(1.0 - t.current_best_value for t in recent_tasks if t.current_best_value < 1.0) / len(recent_tasks) if recent_tasks else 0
        
        metrics = {
            "performance": {
                "total_optimizations": status["total_optimizations"],
                "successful_optimizations": status["successful_optimizations"],
                "success_rate": status["success_rate"],
                "average_iterations": avg_iterations,
                "average_improvement": avg_improvement
            },
            "active_tasks": {
                "total_active": len(status["active_tasks"]),
                "by_target": active_task_types
            },
            "system_integration": {
                "quantum_available": optimizer.quantum_computer is not None,
                "time_crystals_available": optimizer.time_crystal_engine is not None,
                "neuromorphic_available": optimizer.neuromorphic_processor is not None
            },
            "timestamp": status["timestamp"]
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get optimization metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health", response_model=OptimizationHealthResponse)
async def optimization_health_check(optimizer = Depends(get_hybrid_optimizer)):
    """Comprehensive optimization system health check"""
    
    try:
        health = await optimizer.health_check()
        return OptimizationHealthResponse(**health)
        
    except Exception as e:
        logger.error(f"Optimization health check failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/evaluate")
async def evaluate_parameters(
    target: str,
    parameters: Dict[str, Any],
    optimizer = Depends(get_hybrid_optimizer)
):
    """Evaluate parameters for a specific optimization target"""
    
    try:
        # Validate optimization target
        try:
            target_enum = OptimizationTarget(target)
        except ValueError:
            valid_targets = [t.value for t in OptimizationTarget]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid target '{target}'. Valid targets: {valid_targets}"
            )
        
        # Evaluate parameters
        value = await optimizer._evaluate_objective(target_enum, parameters)
        
        return {
            "target": target,
            "parameters": parameters,
            "objective_value": value,
            "evaluation_time": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Parameter evaluation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/history")
async def get_optimization_history(
    limit: int = 50,
    target_filter: Optional[str] = None,
    optimizer = Depends(get_hybrid_optimizer)
):
    """Get optimization history with optional filtering"""
    
    try:
        # Get completed tasks
        tasks = optimizer.completed_tasks
        
        # Apply target filter if specified
        if target_filter:
            try:
                target_enum = OptimizationTarget(target_filter)
                tasks = [t for t in tasks if t.target == target_enum]
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid target filter '{target_filter}'"
                )
        
        # Limit results
        tasks = tasks[-limit:]
        
        # Format task history
        history = []
        for task in tasks:
            history.append({
                "task_id": task.id,
                "target": task.target.value,
                "status": task.status,
                "iterations": task.iterations_completed,
                "best_value": task.current_best_value,
                "improvement": 1.0 - task.current_best_value if task.current_best_value < 1.0 else 0.0,
                "started_at": task.started_at.isoformat(),
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            })
        
        return {
            "history": history,
            "total_shown": len(history),
            "total_completed": len(optimizer.completed_tasks),
            "filter_applied": target_filter,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Initialize hybrid optimizer reference (called from main app)
def set_hybrid_optimizer_instance(optimizer_instance):
    """Set the hybrid optimizer instance for API routes"""
    global hybrid_optimizer
    hybrid_optimizer = optimizer_instance