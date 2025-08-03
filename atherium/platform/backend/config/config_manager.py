"""
Configuration Manager for Quantum AI Platform
Environment-based configuration, settings management, and system parameters
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from pydantic import BaseSettings, Field
import yaml

logger = logging.getLogger(__name__)

@dataclass
class QuantumConfig:
    """Quantum computing configuration"""
    num_qubits: int = 32
    max_circuits: int = 1000
    optimization_iterations: int = 100
    coherence_time_ms: float = 100.0
    gate_fidelity: float = 0.99
    measurement_error: float = 0.01
    backend_type: str = "qasm_simulator"
    error_correction_enabled: bool = True
    quantum_volume: int = 32

@dataclass
class TimeCrystalConfig:
    """Time crystal configuration"""
    num_crystals: int = 8
    dimensions: int = 3
    driving_frequency_ghz: float = 1.0
    coupling_strength: float = 0.1
    coherence_threshold: float = 0.95
    synchronization_timeout_ms: int = 1000
    thermal_noise_amplitude: float = 0.001
    max_entanglement_distance: float = 10.0

@dataclass
class NeuromorphicConfig:
    """Neuromorphic computing configuration"""
    num_neurons: int = 10000
    num_synapses: int = 100000
    membrane_time_constant_ms: float = 20.0
    threshold_voltage: float = -55.0
    reset_voltage: float = -70.0
    refractory_period_ms: float = 2.0
    synaptic_delay_ms: float = 1.0
    plasticity_learning_rate: float = 0.001
    quantum_entanglement_strength: float = 0.1

@dataclass
class AIMLConfig:
    """AI/ML optimization configuration"""
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 1000
    early_stopping_patience: int = 50
    optimization_targets: List[str] = field(default_factory=lambda: [
        "quantum_fidelity", "time_crystal_coherence", "neuromorphic_efficiency"
    ])
    model_architecture: List[int] = field(default_factory=lambda: [128, 64, 32])
    activation_function: str = "relu"
    optimizer: str = "adam"
    regularization_l2: float = 0.0001

@dataclass
class IoTConfig:
    """IoT integration configuration"""
    mqtt_broker_host: str = "localhost"
    mqtt_broker_port: int = 1883
    mqtt_username: str = ""
    mqtt_password: str = ""
    mqtt_keepalive: int = 60
    websocket_port: int = 8765
    max_devices: int = 1000
    device_timeout_seconds: int = 300
    data_retention_days: int = 30
    quantum_sync_enabled: bool = True

@dataclass
class DatabaseConfig:
    """Database configuration"""
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "quantum_ai_platform"
    postgresql_url: str = "postgresql://quantum_user:quantum_pass@localhost:5432/quantum_ai_db"
    redis_url: str = "redis://localhost:6379"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    chromadb_path: str = "./chroma_db"
    connection_pool_size: int = 20
    query_timeout: int = 30

@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 8
    max_login_attempts: int = 5
    account_lockout_minutes: int = 30
    api_key_expire_days: int = 365
    encryption_enabled: bool = True
    https_only: bool = True

@dataclass
class SystemConfig:
    """System-wide configuration"""
    debug_mode: bool = False
    log_level: str = "INFO"
    max_concurrent_tasks: int = 100
    background_task_interval: int = 60
    health_check_interval: int = 30
    metrics_collection_enabled: bool = True
    monitoring_enabled: bool = True
    auto_scaling_enabled: bool = False
    maintenance_mode: bool = False

class ConfigManager:
    """
    Centralized configuration manager that:
    - Loads configuration from environment variables
    - Supports configuration files (JSON, YAML)
    - Provides typed configuration objects
    - Handles configuration validation
    - Supports hot reloading of configuration
    - Environment-specific settings (dev, staging, prod)
    """
    
    def __init__(self, config_file: Optional[str] = None, environment: str = "development"):
        self.environment = environment
        self.config_file = config_file
        self.config_data = {}
        
        # Configuration objects
        self.quantum = QuantumConfig()
        self.time_crystal = TimeCrystalConfig()
        self.neuromorphic = NeuromorphicConfig()
        self.ai_ml = AIMLConfig()
        self.iot = IoTConfig()
        self.database = DatabaseConfig()
        self.security = SecurityConfig()
        self.system = SystemConfig()
        
        # Environment variable mappings
        self.env_mappings = {
            # Quantum settings
            "QUANTUM_NUM_QUBITS": ("quantum", "num_qubits", int),
            "QUANTUM_MAX_CIRCUITS": ("quantum", "max_circuits", int),
            "QUANTUM_BACKEND_TYPE": ("quantum", "backend_type", str),
            "QUANTUM_ERROR_CORRECTION": ("quantum", "error_correction_enabled", bool),
            
            # Time crystal settings
            "TIME_CRYSTAL_NUM_CRYSTALS": ("time_crystal", "num_crystals", int),
            "TIME_CRYSTAL_DIMENSIONS": ("time_crystal", "dimensions", int),
            "TIME_CRYSTAL_FREQUENCY": ("time_crystal", "driving_frequency_ghz", float),
            "TIME_CRYSTAL_COHERENCE_THRESHOLD": ("time_crystal", "coherence_threshold", float),
            
            # Neuromorphic settings
            "NEUROMORPHIC_NUM_NEURONS": ("neuromorphic", "num_neurons", int),
            "NEUROMORPHIC_NUM_SYNAPSES": ("neuromorphic", "num_synapses", int),
            "NEUROMORPHIC_LEARNING_RATE": ("neuromorphic", "plasticity_learning_rate", float),
            
            # AI/ML settings
            "AI_LEARNING_RATE": ("ai_ml", "learning_rate", float),
            "AI_BATCH_SIZE": ("ai_ml", "batch_size", int),
            "AI_MAX_EPOCHS": ("ai_ml", "max_epochs", int),
            "AI_OPTIMIZER": ("ai_ml", "optimizer", str),
            
            # IoT settings
            "IOT_MQTT_HOST": ("iot", "mqtt_broker_host", str),
            "IOT_MQTT_PORT": ("iot", "mqtt_broker_port", int),
            "IOT_MQTT_USERNAME": ("iot", "mqtt_username", str),
            "IOT_MQTT_PASSWORD": ("iot", "mqtt_password", str),
            "IOT_MAX_DEVICES": ("iot", "max_devices", int),
            
            # Database settings
            "DATABASE_MONGODB_URL": ("database", "mongodb_url", str),
            "DATABASE_POSTGRESQL_URL": ("database", "postgresql_url", str),
            "DATABASE_REDIS_URL": ("database", "redis_url", str),
            "DATABASE_QDRANT_HOST": ("database", "qdrant_host", str),
            "DATABASE_QDRANT_PORT": ("database", "qdrant_port", int),
            
            # Security settings
            "SECURITY_SECRET_KEY": ("security", "secret_key", str),
            "SECURITY_JWT_ALGORITHM": ("security", "jwt_algorithm", str),
            "SECURITY_TOKEN_EXPIRE_MINUTES": ("security", "access_token_expire_minutes", int),
            "SECURITY_HTTPS_ONLY": ("security", "https_only", bool),
            
            # System settings
            "SYSTEM_DEBUG_MODE": ("system", "debug_mode", bool),
            "SYSTEM_LOG_LEVEL": ("system", "log_level", str),
            "SYSTEM_MAX_CONCURRENT_TASKS": ("system", "max_concurrent_tasks", int),
            "SYSTEM_ENVIRONMENT": ("system", "environment", str),
        }
        
        logger.info(f"Configuration Manager initialized for environment: {environment}")
    
    def load_configuration(self):
        """Load configuration from all sources"""
        
        try:
            # Load from configuration file first
            if self.config_file:
                self._load_from_file()
            
            # Load from environment variables (override file settings)
            self._load_from_environment()
            
            # Apply environment-specific overrides
            self._apply_environment_overrides()
            
            # Validate configuration
            self._validate_configuration()
            
            logger.info(f"Configuration loaded successfully for {self.environment} environment")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_from_file(self):
        """Load configuration from file (JSON or YAML)"""
        
        try:
            config_path = Path(self.config_file)
            
            if not config_path.exists():
                logger.warning(f"Configuration file not found: {self.config_file}")
                return
            
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    self.config_data = yaml.safe_load(f)
                else:
                    self.config_data = json.load(f)
            
            # Apply configuration data to objects
            self._apply_config_data()
            
            logger.info(f"Configuration loaded from file: {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
            raise
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        
        for env_var, (section, attr, type_func) in self.env_mappings.items():
            value = os.getenv(env_var)
            
            if value is not None:
                try:
                    # Convert value to appropriate type
                    if type_func == bool:
                        typed_value = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        typed_value = type_func(value)
                    
                    # Apply to configuration object
                    config_obj = getattr(self, section)
                    setattr(config_obj, attr, typed_value)
                    
                    logger.debug(f"Environment variable applied: {env_var} = {typed_value}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment variable value: {env_var} = {value}, error: {e}")
    
    def _apply_config_data(self):
        """Apply configuration data from file to configuration objects"""
        
        for section_name, section_data in self.config_data.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                config_obj = getattr(self, section_name)
                
                for key, value in section_data.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides"""
        
        if self.environment == "development":
            self.system.debug_mode = True
            self.system.log_level = "DEBUG"
            self.security.https_only = False
            
        elif self.environment == "staging":
            self.system.debug_mode = False
            self.system.log_level = "INFO"
            self.security.https_only = True
            
        elif self.environment == "production":
            self.system.debug_mode = False
            self.system.log_level = "WARNING"
            self.security.https_only = True
            self.system.monitoring_enabled = True
            self.system.auto_scaling_enabled = True
    
    def _validate_configuration(self):
        """Validate configuration values"""
        
        # Quantum validation
        if self.quantum.num_qubits < 1 or self.quantum.num_qubits > 1000:
            raise ValueError("Invalid number of qubits: must be between 1 and 1000")
        
        if not 0 < self.quantum.gate_fidelity <= 1:
            raise ValueError("Gate fidelity must be between 0 and 1")
        
        # Time crystal validation
        if self.time_crystal.num_crystals < 1:
            raise ValueError("Number of time crystals must be positive")
        
        if not 0 < self.time_crystal.coherence_threshold <= 1:
            raise ValueError("Coherence threshold must be between 0 and 1")
        
        # Neuromorphic validation
        if self.neuromorphic.num_neurons < 1:
            raise ValueError("Number of neurons must be positive")
        
        if self.neuromorphic.plasticity_learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        # AI/ML validation
        if self.ai_ml.learning_rate <= 0:
            raise ValueError("AI learning rate must be positive")
        
        if self.ai_ml.batch_size < 1:
            raise ValueError("Batch size must be positive")
        
        # Security validation
        if len(self.security.secret_key) < 32:
            logger.warning("Security secret key should be at least 32 characters long")
        
        if self.security.password_min_length < 8:
            raise ValueError("Minimum password length should be at least 8 characters")
        
        logger.info("Configuration validation completed successfully")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        
        return {
            "quantum": self.quantum.__dict__,
            "time_crystal": self.time_crystal.__dict__,
            "neuromorphic": self.neuromorphic.__dict__,
            "ai_ml": self.ai_ml.__dict__,
            "iot": self.iot.__dict__,
            "database": self.database.__dict__,
            "security": {k: v for k, v in self.security.__dict__.items() if k != "secret_key"},
            "system": self.system.__dict__,
            "environment": self.environment
        }
    
    def save_config_to_file(self, file_path: str):
        """Save current configuration to file"""
        
        try:
            config_dict = self.get_config_dict()
            
            with open(file_path, 'w') as f:
                if file_path.endswith(('.yaml', '.yml')):
                    yaml.dump(config_dict, f, default_flow_style=False)
                else:
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def update_config(self, section: str, updates: Dict[str, Any]):
        """Update configuration section"""
        
        if not hasattr(self, section):
            raise ValueError(f"Unknown configuration section: {section}")
        
        config_obj = getattr(self, section)
        
        for key, value in updates.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
                logger.info(f"Configuration updated: {section}.{key} = {value}")
            else:
                logger.warning(f"Unknown configuration key: {section}.{key}")
    
    def reload_configuration(self):
        """Reload configuration from all sources"""
        
        logger.info("Reloading configuration...")
        self.load_configuration()
        logger.info("Configuration reloaded successfully")
    
    def get_database_url(self, db_type: str) -> str:
        """Get database URL for specific database type"""
        
        url_map = {
            "mongodb": self.database.mongodb_url,
            "postgresql": self.database.postgresql_url,
            "redis": self.database.redis_url
        }
        
        return url_map.get(db_type, "")
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled"""
        
        feature_map = {
            "quantum_error_correction": self.quantum.error_correction_enabled,
            "quantum_sync": self.iot.quantum_sync_enabled,
            "encryption": self.security.encryption_enabled,
            "monitoring": self.system.monitoring_enabled,
            "auto_scaling": self.system.auto_scaling_enabled,
            "metrics_collection": self.system.metrics_collection_enabled,
            "debug_mode": self.system.debug_mode,
            "maintenance_mode": self.system.maintenance_mode
        }
        
        return feature_map.get(feature, False)
    
    def get_resource_limits(self) -> Dict[str, int]:
        """Get system resource limits"""
        
        return {
            "max_qubits": self.quantum.num_qubits,
            "max_circuits": self.quantum.max_circuits,
            "max_neurons": self.neuromorphic.num_neurons,
            "max_synapses": self.neuromorphic.num_synapses,
            "max_crystals": self.time_crystal.num_crystals,
            "max_iot_devices": self.iot.max_devices,
            "max_concurrent_tasks": self.system.max_concurrent_tasks,
            "db_pool_size": self.database.connection_pool_size
        }
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance-related settings"""
        
        return {
            "quantum_optimization_iterations": self.quantum.optimization_iterations,
            "ai_batch_size": self.ai_ml.batch_size,
            "ai_max_epochs": self.ai_ml.max_epochs,
            "background_task_interval": self.system.background_task_interval,
            "health_check_interval": self.system.health_check_interval,
            "db_query_timeout": self.database.query_timeout,
            "iot_device_timeout": self.iot.device_timeout_seconds
        }


# Global configuration instance
config = ConfigManager()

def get_config() -> ConfigManager:
    """Get global configuration instance"""
    return config

def load_config(config_file: str = None, environment: str = None):
    """Load global configuration"""
    global config
    
    if environment:
        config.environment = environment
    
    if config_file:
        config.config_file = config_file
    
    config.load_configuration()
    return config