"""
Database Manager for Quantum AI Platform
Multi-database support: MongoDB, PostgreSQL, Vector DBs (Qdrant, ChromaDB)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json

# Database imports
import motor.motor_asyncio
import asyncpg
from qdrant_client import QdrantClient
from qdrant_client.http import models
import chromadb
from chromadb.config import Settings
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Unified database manager supporting multiple database types:
    - MongoDB: Document storage for quantum states, crystal configurations, etc.
    - PostgreSQL: Relational data for users, sessions, optimization history
    - Vector DBs: Qdrant/ChromaDB for quantum state embeddings and similarity search
    - Redis: Caching and real-time data
    """
    
    def __init__(self):
        # Database connections
        self.mongo_client = None
        self.mongo_db = None
        self.postgres_pool = None
        self.qdrant_client = None
        self.chroma_client = None
        self.redis_client = None
        
        # Connection status
        self.connections = {
            "mongodb": False,
            "postgresql": False,
            "qdrant": False,
            "chromadb": False,
            "redis": False
        }
        
        # Collection/table names
        self.collections = {
            "quantum_states": "quantum_states",
            "time_crystals": "time_crystals",
            "neuromorphic_events": "neuromorphic_events",
            "optimization_tasks": "optimization_tasks",
            "iot_devices": "iot_devices",
            "user_sessions": "user_sessions",
            "system_metrics": "system_metrics"
        }
        
        logger.info("Database Manager initialized")
    
    async def initialize(self):
        """Initialize all database connections"""
        
        try:
            # Initialize MongoDB
            await self._init_mongodb()
            
            # Initialize PostgreSQL
            await self._init_postgresql()
            
            # Initialize Vector databases
            await self._init_vector_dbs()
            
            # Initialize Redis
            await self._init_redis()
            
            # Create necessary collections/tables
            await self._create_schemas()
            
            logger.info("All database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def _init_mongodb(self):
        """Initialize MongoDB connection"""
        
        try:
            # MongoDB connection string (using default local instance)
            mongo_url = "mongodb://localhost:27017"
            
            self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(mongo_url)
            self.mongo_db = self.mongo_client.quantum_ai_platform
            
            # Test connection
            await self.mongo_client.admin.command('ping')
            self.connections["mongodb"] = True
            
            logger.info("MongoDB connection established")
            
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}. Using fallback storage.")
            # Fallback to in-memory storage for demo
            self._mongodb_fallback = {}
            self.connections["mongodb"] = False
    
    async def _init_postgresql(self):
        """Initialize PostgreSQL connection pool"""
        
        try:
            # PostgreSQL connection (using default local instance)
            dsn = "postgresql://quantum_user:quantum_pass@localhost:5432/quantum_ai_db"
            
            self.postgres_pool = await asyncpg.create_pool(
                dsn,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            self.connections["postgresql"] = True
            logger.info("PostgreSQL connection pool established")
            
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}. Using fallback storage.")
            # Fallback to in-memory storage for demo
            self._postgres_fallback = {}
            self.connections["postgresql"] = False
    
    async def _init_vector_dbs(self):
        """Initialize vector databases (Qdrant and ChromaDB)"""
        
        # Initialize Qdrant
        try:
            self.qdrant_client = QdrantClient(
                host="localhost",
                port=6333,
                timeout=60
            )
            
            # Test connection
            collections = self.qdrant_client.get_collections()
            self.connections["qdrant"] = True
            logger.info("Qdrant connection established")
            
        except Exception as e:
            logger.warning(f"Qdrant connection failed: {e}. Using fallback storage.")
            self.connections["qdrant"] = False
        
        # Initialize ChromaDB
        try:
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))
            
            self.connections["chromadb"] = True
            logger.info("ChromaDB connection established")
            
        except Exception as e:
            logger.warning(f"ChromaDB connection failed: {e}. Using fallback storage.")
            self.connections["chromadb"] = False
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            self.connections["redis"] = True
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using fallback storage.")
            # Fallback to in-memory storage for demo
            self._redis_fallback = {}
            self.connections["redis"] = False
    
    async def _create_schemas(self):
        """Create necessary database schemas and collections"""
        
        # Create MongoDB collections with indexes
        if self.connections["mongodb"]:
            await self._create_mongodb_indexes()
        
        # Create PostgreSQL tables
        if self.connections["postgresql"]:
            await self._create_postgresql_tables()
        
        # Create Qdrant collections
        if self.connections["qdrant"]:
            await self._create_qdrant_collections()
        
        # Create ChromaDB collections
        if self.connections["chromadb"]:
            await self._create_chromadb_collections()
    
    async def _create_mongodb_indexes(self):
        """Create MongoDB indexes for performance"""
        
        try:
            # Quantum states collection
            await self.mongo_db.quantum_states.create_index("state_id")
            await self.mongo_db.quantum_states.create_index("timestamp")
            await self.mongo_db.quantum_states.create_index("fidelity")
            
            # Time crystals collection
            await self.mongo_db.time_crystals.create_index("crystal_id")
            await self.mongo_db.time_crystals.create_index("coherence")
            await self.mongo_db.time_crystals.create_index("timestamp")
            
            # Neuromorphic events collection
            await self.mongo_db.neuromorphic_events.create_index("neuron_id")
            await self.mongo_db.neuromorphic_events.create_index("timestamp")
            await self.mongo_db.neuromorphic_events.create_index("event_type")
            
            # IoT devices collection
            await self.mongo_db.iot_devices.create_index("device_id")
            await self.mongo_db.iot_devices.create_index("device_type")
            await self.mongo_db.iot_devices.create_index("last_seen")
            
            logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create MongoDB indexes: {e}")
    
    async def _create_postgresql_tables(self):
        """Create PostgreSQL tables"""
        
        try:
            async with self.postgres_pool.acquire() as connection:
                # Users table
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(255) UNIQUE NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        is_active BOOLEAN DEFAULT true,
                        is_admin BOOLEAN DEFAULT false,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP
                    )
                """)
                
                # User sessions table
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        session_token VARCHAR(255) UNIQUE NOT NULL,
                        expires_at TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ip_address INET,
                        user_agent TEXT
                    )
                """)
                
                # Optimization history table
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS optimization_history (
                        id SERIAL PRIMARY KEY,
                        task_id VARCHAR(255) NOT NULL,
                        target_type VARCHAR(100) NOT NULL,
                        initial_parameters JSONB,
                        final_parameters JSONB,
                        initial_value REAL,
                        final_value REAL,
                        iterations INTEGER,
                        status VARCHAR(50),
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        user_id INTEGER REFERENCES users(id)
                    )
                """)
                
                # System metrics table
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id SERIAL PRIMARY KEY,
                        metric_name VARCHAR(255) NOT NULL,
                        metric_value REAL NOT NULL,
                        component VARCHAR(100) NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB
                    )
                """)
                
                # API keys table
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        key_hash VARCHAR(255) UNIQUE NOT NULL,
                        name VARCHAR(255),
                        permissions JSONB,
                        is_active BOOLEAN DEFAULT true,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_used TIMESTAMP,
                        expires_at TIMESTAMP
                    )
                """)
                
            logger.info("PostgreSQL tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL tables: {e}")
    
    async def _create_qdrant_collections(self):
        """Create Qdrant collections for vector storage"""
        
        try:
            # Quantum state embeddings collection
            try:
                self.qdrant_client.create_collection(
                    collection_name="quantum_state_embeddings",
                    vectors_config=models.VectorParams(
                        size=512,  # Embedding dimension
                        distance=models.Distance.COSINE
                    )
                )
            except Exception:
                pass  # Collection might already exist
            
            # Time crystal embeddings collection
            try:
                self.qdrant_client.create_collection(
                    collection_name="time_crystal_embeddings",
                    vectors_config=models.VectorParams(
                        size=256,
                        distance=models.Distance.COSINE
                    )
                )
            except Exception:
                pass
            
            # Neuromorphic pattern embeddings
            try:
                self.qdrant_client.create_collection(
                    collection_name="neuromorphic_patterns",
                    vectors_config=models.VectorParams(
                        size=1024,
                        distance=models.Distance.COSINE
                    )
                )
            except Exception:
                pass
            
            logger.info("Qdrant collections created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create Qdrant collections: {e}")
    
    async def _create_chromadb_collections(self):
        """Create ChromaDB collections"""
        
        try:
            # Quantum knowledge base
            try:
                self.chroma_client.create_collection(
                    name="quantum_knowledge",
                    metadata={"description": "Quantum computing knowledge and documentation"}
                )
            except Exception:
                pass  # Collection might already exist
            
            # AI/ML model embeddings
            try:
                self.chroma_client.create_collection(
                    name="ai_model_embeddings",
                    metadata={"description": "AI/ML model state embeddings"}
                )
            except Exception:
                pass
            
            logger.info("ChromaDB collections created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create ChromaDB collections: {e}")
    
    # Data access methods
    async def store_quantum_state(self, state_data: Dict[str, Any]) -> bool:
        """Store quantum state data"""
        
        try:
            if self.connections["mongodb"]:
                result = await self.mongo_db.quantum_states.insert_one(state_data)
                return result.inserted_id is not None
            else:
                # Fallback storage
                state_id = state_data.get("state_id", f"state_{datetime.utcnow().timestamp()}")
                if not hasattr(self, '_mongodb_fallback'):
                    self._mongodb_fallback = {}
                self._mongodb_fallback[state_id] = state_data
                return True
                
        except Exception as e:
            logger.error(f"Failed to store quantum state: {e}")
            return False
    
    async def get_quantum_state(self, state_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve quantum state data"""
        
        try:
            if self.connections["mongodb"]:
                result = await self.mongo_db.quantum_states.find_one({"state_id": state_id})
                if result:
                    result.pop("_id", None)  # Remove MongoDB ObjectId
                return result
            else:
                # Fallback storage
                if hasattr(self, '_mongodb_fallback'):
                    return self._mongodb_fallback.get(state_id)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get quantum state: {e}")
            return None
    
    async def store_time_crystal_state(self, crystal_data: Dict[str, Any]) -> bool:
        """Store time crystal state data"""
        
        try:
            if self.connections["mongodb"]:
                result = await self.mongo_db.time_crystals.insert_one(crystal_data)
                return result.inserted_id is not None
            else:
                # Fallback storage
                crystal_id = crystal_data.get("crystal_id", f"crystal_{datetime.utcnow().timestamp()}")
                if not hasattr(self, '_mongodb_fallback'):
                    self._mongodb_fallback = {}
                self._mongodb_fallback[f"crystal_{crystal_id}"] = crystal_data
                return True
                
        except Exception as e:
            logger.error(f"Failed to store time crystal state: {e}")
            return False
    
    async def store_optimization_history(self, optimization_data: Dict[str, Any]) -> bool:
        """Store optimization task history"""
        
        try:
            if self.connections["postgresql"]:
                async with self.postgres_pool.acquire() as connection:
                    await connection.execute("""
                        INSERT INTO optimization_history 
                        (task_id, target_type, initial_parameters, final_parameters, 
                         initial_value, final_value, iterations, status, started_at, completed_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """, 
                    optimization_data["task_id"],
                    optimization_data["target_type"],
                    json.dumps(optimization_data.get("initial_parameters", {})),
                    json.dumps(optimization_data.get("final_parameters", {})),
                    optimization_data.get("initial_value", 0.0),
                    optimization_data.get("final_value", 0.0),
                    optimization_data.get("iterations", 0),
                    optimization_data.get("status", "unknown"),
                    optimization_data.get("started_at"),
                    optimization_data.get("completed_at")
                    )
                return True
            else:
                # Fallback storage
                if not hasattr(self, '_postgres_fallback'):
                    self._postgres_fallback = {}
                task_id = optimization_data.get("task_id", f"task_{datetime.utcnow().timestamp()}")
                self._postgres_fallback[task_id] = optimization_data
                return True
                
        except Exception as e:
            logger.error(f"Failed to store optimization history: {e}")
            return False
    
    async def cache_set(self, key: str, value: Any, expiry_seconds: int = 3600) -> bool:
        """Set value in cache (Redis)"""
        
        try:
            if self.connections["redis"]:
                await self.redis_client.setex(key, expiry_seconds, json.dumps(value))
                return True
            else:
                # Fallback storage
                if not hasattr(self, '_redis_fallback'):
                    self._redis_fallback = {}
                self._redis_fallback[key] = {
                    "value": value,
                    "expires_at": datetime.utcnow() + timedelta(seconds=expiry_seconds)
                }
                return True
                
        except Exception as e:
            logger.error(f"Failed to set cache: {e}")
            return False
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache (Redis)"""
        
        try:
            if self.connections["redis"]:
                value = await self.redis_client.get(key)
                return json.loads(value) if value else None
            else:
                # Fallback storage
                if hasattr(self, '_redis_fallback') and key in self._redis_fallback:
                    cached = self._redis_fallback[key]
                    if datetime.utcnow() < cached["expires_at"]:
                        return cached["value"]
                    else:
                        del self._redis_fallback[key]
                return None
                
        except Exception as e:
            logger.error(f"Failed to get cache: {e}")
            return None
    
    async def store_vector_embedding(self, collection: str, embedding: List[float], 
                                   metadata: Dict[str, Any], vector_id: str = None) -> bool:
        """Store vector embedding in Qdrant"""
        
        try:
            if self.connections["qdrant"]:
                point_id = vector_id or f"point_{datetime.utcnow().timestamp()}"
                
                self.qdrant_client.upsert(
                    collection_name=collection,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload=metadata
                        )
                    ]
                )
                return True
            else:
                # Fallback - store in MongoDB or memory
                return await self.store_quantum_state({
                    "type": "vector_embedding",
                    "collection": collection,
                    "embedding": embedding,
                    "metadata": metadata,
                    "vector_id": vector_id or f"point_{datetime.utcnow().timestamp()}"
                })
                
        except Exception as e:
            logger.error(f"Failed to store vector embedding: {e}")
            return False
    
    async def search_similar_vectors(self, collection: str, query_vector: List[float], 
                                   limit: int = 10, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search for similar vectors in Qdrant"""
        
        try:
            if self.connections["qdrant"]:
                search_result = self.qdrant_client.search(
                    collection_name=collection,
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold
                )
                
                results = []
                for hit in search_result:
                    results.append({
                        "id": hit.id,
                        "score": hit.score,
                        "metadata": hit.payload
                    })
                
                return results
            else:
                # Fallback - return empty results
                return []
                
        except Exception as e:
            logger.error(f"Failed to search similar vectors: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all database connections"""
        
        health_status = {
            "status": "healthy",
            "connections": self.connections.copy(),
            "total_connections": len([c for c in self.connections.values() if c]),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Test each connection
        for db_type, connected in self.connections.items():
            if connected:
                try:
                    if db_type == "mongodb" and self.mongo_client:
                        await self.mongo_client.admin.command('ping')
                    elif db_type == "postgresql" and self.postgres_pool:
                        async with self.postgres_pool.acquire() as conn:
                            await conn.execute("SELECT 1")
                    elif db_type == "redis" and self.redis_client:
                        await self.redis_client.ping()
                    # Qdrant and ChromaDB don't need constant health checks
                        
                except Exception as e:
                    logger.warning(f"{db_type} health check failed: {e}")
                    health_status["connections"][db_type] = False
        
        # Overall status
        if not any(health_status["connections"].values()):
            health_status["status"] = "degraded"
        
        return health_status
    
    async def close(self):
        """Close all database connections"""
        
        try:
            if self.mongo_client:
                self.mongo_client.close()
            
            if self.postgres_pool:
                await self.postgres_pool.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            # Qdrant and ChromaDB clients don't need explicit closing
            
            logger.info("All database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")