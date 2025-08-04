"""
Quantum AI Platform - Main FastAPI Application
Integrated platform with quantum computing, time crystals, neuromorphic processing, and IoT
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import uvicorn

# Import all the components we've created
from database.db_manager import DatabaseManager
from security.auth_manager import AuthenticationManager
from config.config_manager import ConfigManager, load_config
from iot.iot_manager import IoTManager

# Import quantum system components
from quantum.vqc_engine import VirtualQuantumComputer
from time_crystals.time_crystal_engine import TimeCrystalEngine
from neuromorphic.snn_processor import SpikingNeuralProcessor
from ai_ml.hybrid_optimizer import HybridQuantumClassicalNeuromorphicOptimizer

# Import API routes
from api.quantum_routes import router as quantum_router, set_quantum_computer_instance
from api.time_crystal_routes import router as time_crystal_router, set_time_crystal_engine_instance
from api.neuromorphic_routes import router as neuromorphic_router, set_neuromorphic_processor_instance
from api.ai_ml_routes import router as ai_ml_router, set_hybrid_optimizer_instance
from api.iot_routes import router as iot_router, set_iot_manager_instance
from api.productivity_suite_routes import router as productivity_router

# Global component instances
db_manager = None
auth_manager = None
config_manager = None
iot_manager = None
quantum_computer = None
time_crystal_engine = None
neuromorphic_processor = None
hybrid_optimizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management with proper component initialization"""
    
    global db_manager, auth_manager, config_manager, iot_manager
    global quantum_computer, time_crystal_engine, neuromorphic_processor, hybrid_optimizer
    
    logger.info("Starting Quantum AI Platform...")
    
    try:
        # Initialize configuration
        config_manager = load_config(
            config_file="config/quantum_ai_config.yaml", 
            environment="development"
        )
        logger.info("Configuration loaded successfully")
        
        # Initialize database manager
        db_manager = DatabaseManager()
        await db_manager.initialize()
        logger.info("Database connections established")
        
        # Initialize authentication manager
        auth_manager = AuthenticationManager(secret_key=config_manager.security.secret_key)
        auth_manager.set_database_manager(db_manager)
        logger.info("Authentication system initialized")
        
        # Initialize core quantum components
        quantum_computer = VirtualQuantumComputer(
            num_qubits=config_manager.quantum.num_qubits,
            backend_name=config_manager.quantum.backend_type,
            optimization_level=1
        )
        await quantum_computer.initialize()
        logger.info("Virtual Quantum Computer initialized")
        
        time_crystal_engine = TimeCrystalEngine(
            num_time_crystals=config_manager.time_crystal.num_crystals,
            dimensions=config_manager.time_crystal.dimensions,
            driving_frequency=config_manager.time_crystal.driving_frequency_ghz
        )
        await time_crystal_engine.initialize()
        logger.info("Time Crystal Engine initialized")
        
        neuromorphic_processor = SpikingNeuralProcessor(
            num_neurons=config_manager.neuromorphic.num_neurons,
            num_synapses=config_manager.neuromorphic.num_synapses
        )
        await neuromorphic_processor.initialize()
        logger.info("Neuromorphic Processor initialized")
        
        hybrid_optimizer = HybridQuantumClassicalNeuromorphicOptimizer(
            learning_rate=config_manager.ai_ml.learning_rate,
            architecture=config_manager.ai_ml.model_architecture
        )
        await hybrid_optimizer.initialize(
            quantum_computer, time_crystal_engine, neuromorphic_processor
        )
        logger.info("Hybrid AI Optimizer initialized")
        
        # Initialize IoT manager
        iot_manager = IoTManager(config_manager)
        await iot_manager.initialize()
        iot_manager.set_quantum_references(
            quantum_computer, time_crystal_engine, neuromorphic_processor
        )
        logger.info("IoT Manager initialized")
        
        # Set component references for API routes
        set_quantum_computer_instance(quantum_computer)
        set_time_crystal_engine_instance(time_crystal_engine)
        set_neuromorphic_processor_instance(neuromorphic_processor)
        set_hybrid_optimizer_instance(hybrid_optimizer)
        set_iot_manager_instance(iot_manager)
        
        # Start background optimization tasks
        asyncio.create_task(continuous_optimization_loop())
        asyncio.create_task(quantum_time_crystal_synchronization_loop())
        
        logger.info("Quantum AI Platform startup completed successfully")
        
        yield  # Application runs here
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down Quantum AI Platform...")
        
        if iot_manager:
            await iot_manager.close()
        
        if hybrid_optimizer:
            await hybrid_optimizer.close()
        
        if neuromorphic_processor:
            await neuromorphic_processor.close()
        
        if time_crystal_engine:
            await time_crystal_engine.close()
        
        if quantum_computer:
            await quantum_computer.close()
        
        if db_manager:
            await db_manager.close()
        
        logger.info("Quantum AI Platform shutdown completed")

app = FastAPI(
    title="Quantum AI Platform",
    description="Advanced quantum computing platform with time crystals, neuromorphic processing, and IoT integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

def get_current_user(token: str = Depends(security)):
    """Get current authenticated user"""
    if not auth_manager:
        raise HTTPException(status_code=503, detail="Authentication not available")
    
    # This will be implemented with actual token verification
    return {"user_id": "demo_user", "permissions": ["quantum:read", "quantum:write"]}

# Include API routers
app.include_router(quantum_router, prefix="/api/quantum", tags=["Quantum Computing"])
app.include_router(time_crystal_router, prefix="/api/time-crystals", tags=["Time Crystals"])
app.include_router(neuromorphic_router, prefix="/api/neuromorphic", tags=["Neuromorphic Computing"])
app.include_router(ai_ml_router, prefix="/api/ai-ml", tags=["AI/ML Optimization"])
app.include_router(iot_router, prefix="/api/iot", tags=["IoT Integration"])
app.include_router(productivity_router, prefix="/api", tags=["AI Productivity Suite"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Quantum AI Platform API",
        "version": "1.0.0",
        "status": "running",
        "components": {
            "quantum_computer": quantum_computer is not None,
            "time_crystal_engine": time_crystal_engine is not None,
            "neuromorphic_processor": neuromorphic_processor is not None,
            "hybrid_optimizer": hybrid_optimizer is not None,
            "iot_manager": iot_manager is not None,
            "database": db_manager is not None,
            "authentication": auth_manager is not None
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive system health check"""
    
    health_status = {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "components": {}
    }
    
    try:
        # Check quantum computer
        if quantum_computer:
            qc_health = await quantum_computer.health_check()
            health_status["components"]["quantum_computer"] = qc_health
        
        # Check time crystal engine
        if time_crystal_engine:
            tc_health = await time_crystal_engine.health_check()
            health_status["components"]["time_crystal_engine"] = tc_health
        
        # Check neuromorphic processor
        if neuromorphic_processor:
            np_health = await neuromorphic_processor.health_check()
            health_status["components"]["neuromorphic_processor"] = np_health
        
        # Check hybrid optimizer
        if hybrid_optimizer:
            ho_health = await hybrid_optimizer.health_check()
            health_status["components"]["hybrid_optimizer"] = ho_health
        
        # Check IoT manager
        if iot_manager:
            iot_health = await iot_manager.health_check()
            health_status["components"]["iot_manager"] = iot_health
        
        # Check database
        if db_manager:
            db_health = await db_manager.health_check()
            health_status["components"]["database"] = db_health
        
        # Check authentication
        if auth_manager:
            auth_health = await auth_manager.health_check()
            health_status["components"]["authentication"] = auth_health
        
        # Determine overall status
        component_statuses = [comp.get("status", "unhealthy") for comp in health_status["components"].values()]
        if all(status == "healthy" for status in component_statuses):
            health_status["status"] = "healthy"
        elif any(status == "healthy" for status in component_statuses):
            health_status["status"] = "degraded"
        else:
            health_status["status"] = "unhealthy"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"
        }

@app.get("/metrics")
async def system_metrics():
    """Get comprehensive system metrics"""
    
    try:
        metrics = {
            "timestamp": "2024-01-01T00:00:00Z",
            "quantum_metrics": {},
            "time_crystal_metrics": {},
            "neuromorphic_metrics": {},
            "ai_ml_metrics": {},
            "iot_metrics": {},
            "system_metrics": {}
        }
        
        if quantum_computer:
            metrics["quantum_metrics"] = await quantum_computer.get_metrics()
        
        if time_crystal_engine:
            metrics["time_crystal_metrics"] = await time_crystal_engine.get_metrics()
        
        if neuromorphic_processor:
            metrics["neuromorphic_metrics"] = await neuromorphic_processor.get_metrics()
        
        if hybrid_optimizer:
            metrics["ai_ml_metrics"] = await hybrid_optimizer.get_metrics()
        
        if iot_manager:
            metrics["iot_metrics"] = await iot_manager.get_system_metrics()
        
        # System-wide metrics
        metrics["system_metrics"] = {
            "total_components": len([c for c in [quantum_computer, time_crystal_engine, neuromorphic_processor, hybrid_optimizer, iot_manager] if c]),
            "memory_usage": "N/A",  # Would implement actual memory monitoring
            "cpu_usage": "N/A",     # Would implement actual CPU monitoring
            "uptime": "N/A"         # Would implement actual uptime tracking
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics collection failed")

async def continuous_optimization_loop():
    """Background task for continuous system optimization"""
    
    while True:
        try:
            if hybrid_optimizer and config_manager:
                # Run optimization step
                await hybrid_optimizer.run_optimization_step("system_performance")
                
                # Wait for next optimization cycle
                await asyncio.sleep(config_manager.system.background_task_interval)
            else:
                await asyncio.sleep(60)
                
        except Exception as e:
            logger.error(f"Optimization loop error: {e}")
            await asyncio.sleep(60)

async def quantum_time_crystal_synchronization_loop():
    """Background task for quantum-time crystal synchronization"""
    
    while True:
        try:
            if quantum_computer and time_crystal_engine:
                # Synchronize quantum states with time crystal network
                await time_crystal_engine.synchronize_crystals()
                
                # Enhanced quantum coherence using time crystals
                await time_crystal_engine.enhance_quantum_coherence()
                
                # Wait for next synchronization cycle
                await asyncio.sleep(30)  # Sync every 30 seconds
            else:
                await asyncio.sleep(60)
                
        except Exception as e:
            logger.error(f"Quantum-time crystal sync error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('quantum_ai_platform.log')
        ]
    )
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )