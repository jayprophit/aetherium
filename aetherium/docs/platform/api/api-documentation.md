# Quantum AI Platform - API Documentation

## Overview

The Quantum AI Platform provides a comprehensive REST API and WebSocket interface for interacting with quantum computing, time crystals, neuromorphic processing, and IoT systems. All APIs follow RESTful conventions with JSON payloads and standard HTTP status codes.

## Base URLs

- **Development:** `http://localhost:8000`
- **Production:** `https://your-domain.com`
- **API Base Path:** `/api/v1`

## Authentication

The platform uses JWT-based authentication with role-based access control (RBAC).

### Authentication Endpoints

#### POST /api/auth/login
Authenticate user and receive JWT token.

**Request:**
```json
{
    "username": "string",
    "password": "string"
}
```

**Response:**
```json
{
    "access_token": "jwt_token_here",
    "refresh_token": "refresh_token_here",
    "token_type": "bearer",
    "expires_in": 3600,
    "user": {
        "id": "uuid",
        "username": "string",
        "email": "string",
        "roles": ["user", "quantum_researcher"],
        "permissions": ["quantum:read", "quantum:execute"]
    }
}
```

#### POST /api/auth/refresh
Refresh JWT token using refresh token.

#### POST /api/auth/logout
Invalidate current session.

### Authorization Headers
Include JWT token in all protected endpoints:
```
Authorization: Bearer <jwt_token>
```

## Core System APIs

### Health Check

#### GET /health
System health and status check.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2025-01-03T16:05:21Z",
    "version": "1.0.0",
    "services": {
        "quantum_engine": "running",
        "time_crystals": "running",
        "neuromorphic": "running",
        "database": "connected",
        "redis": "connected",
        "iot": "running"
    },
    "metrics": {
        "uptime": 86400,
        "memory_usage": "45%",
        "cpu_usage": "12%"
    }
}
```

## Quantum Computing APIs

### Quantum Status

#### GET /api/quantum/status
Get quantum system status and metrics.

**Response:**
```json
{
    "status": "running",
    "qubits_available": 32,
    "qubits_in_use": 8,
    "active_circuits": 3,
    "queue_size": 2,
    "coherence_time_ms": 100.5,
    "error_rate": 0.001,
    "last_calibration": "2025-01-03T14:30:00Z",
    "temperature_mk": 15,
    "performance_metrics": {
        "avg_execution_time_ms": 45.2,
        "success_rate": 0.995,
        "circuits_executed_24h": 1247
    }
}
```

### Circuit Management

#### POST /api/quantum/circuits
Create a new quantum circuit.

**Request:**
```json
{
    "name": "my_circuit",
    "template": "grover_search",
    "parameters": [0.5, 1.0, 0.785],
    "qubits": 8,
    "description": "Grover search algorithm implementation",
    "optimization_level": 2
}
```

**Response:**
```json
{
    "circuit_id": "uuid",
    "name": "my_circuit",
    "qubits": 8,
    "depth": 15,
    "gates": 42,
    "created_at": "2025-01-03T16:05:21Z",
    "estimated_runtime_ms": 50,
    "status": "ready"
}
```

#### GET /api/quantum/circuits
List user's quantum circuits.

#### GET /api/quantum/circuits/{circuit_id}
Get circuit details and visualization data.

#### DELETE /api/quantum/circuits/{circuit_id}
Delete a quantum circuit.

### Circuit Execution

#### POST /api/quantum/execute
Execute a quantum circuit.

**Request:**
```json
{
    "circuit_id": "uuid",
    "shots": 1024,
    "optimization": true,
    "error_correction": true,
    "priority": "normal",
    "webhook_url": "https://your-app.com/webhooks/quantum-results"
}
```

**Response:**
```json
{
    "execution_id": "uuid",
    "status": "queued",
    "queue_position": 1,
    "estimated_wait_time_ms": 30000,
    "circuit_id": "uuid",
    "shots": 1024,
    "submitted_at": "2025-01-03T16:05:21Z"
}
```

#### GET /api/quantum/executions/{execution_id}
Get execution status and results.

**Response:**
```json
{
    "execution_id": "uuid",
    "status": "completed",
    "circuit_id": "uuid",
    "shots": 1024,
    "results": {
        "counts": {
            "00000": 312,
            "11111": 298,
            "01010": 205,
            "10101": 209
        },
        "probabilities": {
            "00000": 0.3047,
            "11111": 0.2910,
            "01010": 0.2002,
            "10101": 0.2041
        },
        "measurement_data": "base64_encoded_raw_data",
        "fidelity": 0.987,
        "execution_time_ms": 47.3
    },
    "metadata": {
        "quantum_volume": 64,
        "error_rates": {
            "single_qubit": 0.0001,
            "two_qubit": 0.0015
        },
        "calibration_data": {
            "timestamp": "2025-01-03T14:30:00Z",
            "coherence_times": [102.1, 98.7, 101.5]
        }
    },
    "completed_at": "2025-01-03T16:05:45Z"
}
```

#### GET /api/quantum/executions
List user's circuit executions with filtering.

**Query Parameters:**
- `status`: Filter by execution status
- `start_date`: Start date for results
- `end_date`: End date for results
- `limit`: Number of results (default: 50, max: 500)
- `offset`: Pagination offset

### Quantum Templates

#### GET /api/quantum/templates
List available circuit templates.

**Response:**
```json
{
    "templates": [
        {
            "name": "grover_search",
            "display_name": "Grover's Search Algorithm",
            "description": "Quantum search algorithm with quadratic speedup",
            "parameters": [
                {
                    "name": "search_items",
                    "type": "integer",
                    "min": 1,
                    "max": 1024,
                    "description": "Number of items to search"
                }
            ],
            "min_qubits": 2,
            "max_qubits": 20,
            "category": "algorithms",
            "complexity": "intermediate"
        }
    ]
}
```

## Time Crystals APIs

### Time Crystal Status

#### GET /api/time-crystals/status
Get time crystal system status.

**Response:**
```json
{
    "status": "synchronized",
    "crystal_count": 8,
    "active_crystals": 8,
    "synchronization_level": 0.98,
    "phase_coherence": 0.95,
    "coupling_strength": 0.5,
    "temperature": 0.1,
    "external_field": 0.3,
    "floquet_frequency": 2.4,
    "last_sync": "2025-01-03T16:05:00Z",
    "performance_metrics": {
        "avg_sync_time_ms": 12.5,
        "coherence_improvement": 0.23,
        "stability_index": 0.97
    }
}
```

### Crystal Management

#### POST /api/time-crystals/crystals
Create or configure a time crystal.

**Request:**
```json
{
    "crystal_id": "crystal_001",
    "dimensions": [100, 100, 100],
    "coupling_neighbors": ["crystal_002", "crystal_003"],
    "initial_phase": 0.0,
    "frequency": 2.4,
    "amplitude": 1.0,
    "configuration": {
        "symmetry_group": "Z2",
        "periodicity": 2,
        "disorder_strength": 0.1
    }
}
```

#### GET /api/time-crystals/crystals
List time crystals and their configurations.

#### GET /api/time-crystals/crystals/{crystal_id}
Get detailed crystal information and state.

### Synchronization Control

#### POST /api/time-crystals/synchronize
Trigger time crystal synchronization.

**Request:**
```json
{
    "target_crystals": ["crystal_001", "crystal_002"],
    "sync_mode": "global",
    "target_coherence": 0.95,
    "max_iterations": 1000
}
```

**Response:**
```json
{
    "sync_id": "uuid",
    "status": "initiated",
    "target_crystals": 2,
    "estimated_time_ms": 500,
    "initiated_at": "2025-01-03T16:05:21Z"
}
```

#### GET /api/time-crystals/synchronize/{sync_id}
Get synchronization status and results.

### Quantum Enhancement

#### POST /api/time-crystals/enhance-coherence
Use time crystals to enhance quantum coherence.

**Request:**
```json
{
    "quantum_circuit_id": "uuid",
    "target_coherence": 0.95,
    "enhancement_duration_ms": 1000
}
```

## Neuromorphic Computing APIs

### Neuromorphic Status

#### GET /api/neuromorphic/status
Get spiking neural network status.

**Response:**
```json
{
    "status": "running",
    "total_neurons": 10000,
    "active_neurons": 8745,
    "total_synapses": 100000,
    "active_synapses": 87234,
    "spike_rate_hz": 42.5,
    "membrane_potential_avg": -65.2,
    "plasticity_enabled": true,
    "quantum_coupling": 0.5,
    "performance_metrics": {
        "processing_speed_spikes_per_sec": 125000,
        "accuracy": 0.943,
        "energy_efficiency": 0.89
    }
}
```

### Network Management

#### POST /api/neuromorphic/networks
Create a spiking neural network.

**Request:**
```json
{
    "name": "vision_network",
    "neurons": {
        "input_layer": 784,
        "hidden_layers": [512, 256],
        "output_layer": 10
    },
    "connectivity": 0.1,
    "neuron_model": "LIF",
    "learning_rule": "STDP",
    "parameters": {
        "threshold": -55.0,
        "resting_potential": -70.0,
        "refractory_period_ms": 2.0,
        "membrane_time_constant_ms": 20.0
    }
}
```

#### GET /api/neuromorphic/networks
List neural networks.

#### GET /api/neuromorphic/networks/{network_id}
Get network details and current state.

### Spike Processing

#### POST /api/neuromorphic/inject-spikes
Inject spike patterns into the network.

**Request:**
```json
{
    "network_id": "uuid",
    "spike_pattern": {
        "neuron_001": [1.0, 0.0, 0.5, 1.0],
        "neuron_002": [0.0, 1.0, 0.0, 0.5]
    },
    "amplitude": 1.0,
    "duration_ms": 100
}
```

#### GET /api/neuromorphic/spikes/{network_id}
Get recent spike events and patterns.

#### POST /api/neuromorphic/train
Train the spiking neural network.

**Request:**
```json
{
    "network_id": "uuid",
    "training_data": "base64_encoded_spike_patterns",
    "epochs": 100,
    "learning_rate": 0.01,
    "validation_split": 0.2
}
```

## IoT Device APIs

### Device Management

#### GET /api/iot/devices
List registered IoT devices.

**Response:**
```json
{
    "devices": [
        {
            "device_id": "sensor_001",
            "name": "Temperature Sensor #1",
            "type": "sensor",
            "category": "environmental",
            "status": "online",
            "last_seen": "2025-01-03T16:05:15Z",
            "location": "Lab Room A",
            "firmware_version": "1.2.3",
            "battery_level": 85,
            "quantum_sync_enabled": true,
            "metrics": {
                "uptime_hours": 720,
                "data_points_24h": 8640,
                "error_count_24h": 2
            }
        }
    ],
    "total": 1,
    "online": 1,
    "offline": 0
}
```

#### POST /api/iot/devices
Register a new IoT device.

**Request:**
```json
{
    "device_id": "sensor_002",
    "name": "Pressure Sensor #1",
    "type": "sensor",
    "category": "industrial",
    "location": "Production Floor",
    "capabilities": ["temperature", "pressure", "humidity"],
    "quantum_sync_enabled": true,
    "communication_protocol": "mqtt"
}
```

#### GET /api/iot/devices/{device_id}
Get detailed device information and recent data.

#### PUT /api/iot/devices/{device_id}
Update device configuration.

#### DELETE /api/iot/devices/{device_id}
Unregister a device.

### Device Control

#### POST /api/iot/devices/{device_id}/command
Send command to device.

**Request:**
```json
{
    "command": "set_sampling_rate",
    "parameters": {
        "rate_hz": 10,
        "duration_seconds": 3600
    }
}
```

#### POST /api/iot/devices/{device_id}/restart
Restart a device remotely.

### Device Data

#### GET /api/iot/devices/{device_id}/data
Get device sensor data with filtering.

**Query Parameters:**
- `start_time`: ISO timestamp
- `end_time`: ISO timestamp  
- `metric`: Specific metric name
- `aggregation`: none, avg, min, max, sum
- `interval`: Aggregation interval (1m, 5m, 1h, 1d)

**Response:**
```json
{
    "device_id": "sensor_001",
    "metric": "temperature",
    "unit": "celsius",
    "data_points": [
        {
            "timestamp": "2025-01-03T16:05:00Z",
            "value": 23.5,
            "quality": "good"
        }
    ],
    "statistics": {
        "count": 100,
        "min": 22.1,
        "max": 24.8,
        "avg": 23.3,
        "std_dev": 0.7
    }
}
```

## AI/ML APIs

### Model Management

#### GET /api/ai/models
List available AI/ML models.

#### POST /api/ai/models
Upload or register a new model.

#### GET /api/ai/models/{model_id}
Get model details and metadata.

#### DELETE /api/ai/models/{model_id}
Delete a model.

### Inference

#### POST /api/ai/predict
Run inference on a model.

**Request:**
```json
{
    "model_id": "uuid",
    "input_data": {
        "features": [1.2, 3.4, 5.6, 7.8],
        "metadata": {
            "source": "sensor_001",
            "timestamp": "2025-01-03T16:05:21Z"
        }
    },
    "options": {
        "confidence_threshold": 0.8,
        "return_probabilities": true
    }
}
```

**Response:**
```json
{
    "prediction_id": "uuid",
    "model_id": "uuid",
    "predictions": {
        "class": "anomaly",
        "confidence": 0.92,
        "probabilities": {
            "normal": 0.08,
            "anomaly": 0.92
        }
    },
    "processing_time_ms": 12.5,
    "timestamp": "2025-01-03T16:05:21Z"
}
```

### Training

#### POST /api/ai/train
Start model training.

#### GET /api/ai/training/{job_id}
Get training job status and metrics.

## WebSocket APIs

### Connection
Connect to WebSocket endpoint: `ws://localhost:8000/ws`

### Authentication
Send authentication message immediately after connection:
```json
{
    "type": "auth",
    "token": "jwt_token_here"
}
```

### Subscriptions
Subscribe to real-time events:
```json
{
    "type": "subscribe",
    "channels": [
        "quantum.executions",
        "time_crystals.synchronization", 
        "neuromorphic.spikes",
        "iot.device_data",
        "system.alerts"
    ]
}
```

### Event Types

#### Quantum Execution Updates
```json
{
    "type": "quantum.execution.status",
    "execution_id": "uuid",
    "status": "running",
    "progress": 0.65,
    "timestamp": "2025-01-03T16:05:21Z"
}
```

#### Time Crystal Synchronization
```json
{
    "type": "time_crystals.sync.update",
    "crystal_ids": ["crystal_001", "crystal_002"],
    "synchronization_level": 0.95,
    "phase_coherence": 0.98,
    "timestamp": "2025-01-03T16:05:21Z"
}
```

#### Neuromorphic Spike Events
```json
{
    "type": "neuromorphic.spike",
    "network_id": "uuid",
    "neuron_id": "neuron_001",
    "spike_time": "2025-01-03T16:05:21.123Z",
    "amplitude": 1.2
}
```

#### IoT Device Data
```json
{
    "type": "iot.data",
    "device_id": "sensor_001",
    "metric": "temperature",
    "value": 23.5,
    "unit": "celsius",
    "timestamp": "2025-01-03T16:05:21Z"
}
```

#### System Alerts
```json
{
    "type": "system.alert",
    "level": "warning",
    "component": "quantum_engine",
    "message": "High queue size detected",
    "details": {
        "queue_size": 25,
        "threshold": 20
    },
    "timestamp": "2025-01-03T16:05:21Z"
}
```

## Error Handling

### HTTP Status Codes
- `200 OK` - Success
- `201 Created` - Resource created
- `400 Bad Request` - Invalid request data
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `409 Conflict` - Resource conflict
- `422 Unprocessable Entity` - Validation errors
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service temporarily unavailable

### Error Response Format
```json
{
    "error": {
        "code": "INVALID_CIRCUIT_PARAMETERS",
        "message": "Circuit parameters are invalid",
        "details": {
            "field": "parameters",
            "issue": "Parameter value out of range",
            "valid_range": [0, 2]
        },
        "request_id": "uuid",
        "timestamp": "2025-01-03T16:05:21Z"
    }
}
```

## Rate Limits

- **Default:** 100 requests per minute per user
- **Authenticated:** 1000 requests per minute per user  
- **Premium:** 10000 requests per minute per user
- **Circuit Execution:** 50 executions per hour per user
- **WebSocket:** 1000 messages per minute per connection

Rate limit headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1641234567
X-RateLimit-Retry-After: 60
```

## SDK and Client Libraries

- **Python:** `pip install quantum-ai-python-sdk`
- **JavaScript/Node.js:** `npm install @quantum-ai/sdk`
- **Curl Examples:** See `/docs/examples/` directory

## API Versioning

- Current version: `v1`
- Version specified in URL: `/api/v1/...`
- Backward compatibility maintained for 2 major versions
- Deprecation notices sent via `X-API-Deprecated` header

## Support and Resources

- **API Documentation:** `/docs/api/`
- **Interactive API Explorer:** `/docs/swagger/`
- **SDK Documentation:** `/docs/sdks/`
- **Examples and Tutorials:** `/docs/examples/`
- **Postman Collection:** Available for download