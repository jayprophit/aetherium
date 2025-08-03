// MongoDB initialization script for Quantum AI Platform
// This script sets up the initial collections and indexes

print('🚀 Initializing MongoDB for Quantum AI Platform...');

// Switch to the main database
db = db.getSiblingDB('quantumai_mongo');

// Create admin user for the database
db.createUser({
    user: "quantumai",
    pwd: "password", // Change in production
    roles: [
        {
            role: "readWrite",
            db: "quantumai_mongo"
        },
        {
            role: "dbAdmin",
            db: "quantumai_mongo"
        }
    ]
});

// Create collections with validation schemas
// Quantum circuit results collection
db.createCollection("quantum_results", {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["circuit_id", "execution_id", "results", "timestamp"],
            properties: {
                circuit_id: {
                    bsonType: "string",
                    description: "UUID of the quantum circuit"
                },
                execution_id: {
                    bsonType: "string",
                    description: "UUID of the execution"
                },
                results: {
                    bsonType: "object",
                    description: "Quantum execution results"
                },
                shots: {
                    bsonType: "int",
                    minimum: 1,
                    maximum: 100000
                },
                fidelity: {
                    bsonType: "double",
                    minimum: 0,
                    maximum: 1
                },
                timestamp: {
                    bsonType: "date",
                    description: "Execution timestamp"
                },
                metadata: {
                    bsonType: "object",
                    description: "Additional metadata"
                }
            }
        }
    }
});

// Time crystal state history
db.createCollection("time_crystal_states", {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["crystal_id", "state", "timestamp"],
            properties: {
                crystal_id: {
                    bsonType: "string",
                    description: "UUID of the time crystal"
                },
                state: {
                    bsonType: "object",
                    description: "Complete crystal state"
                },
                coherence: {
                    bsonType: "double",
                    minimum: 0,
                    maximum: 1
                },
                frequency: {
                    bsonType: "double",
                    minimum: 0
                },
                phase: {
                    bsonType: "double"
                },
                entanglement_matrix: {
                    bsonType: "array",
                    description: "Entanglement strength matrix"
                },
                timestamp: {
                    bsonType: "date"
                }
            }
        }
    }
});

// Neuromorphic spike trains
db.createCollection("spike_trains", {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["neuron_id", "spikes", "start_time", "end_time"],
            properties: {
                neuron_id: {
                    bsonType: "string",
                    description: "UUID of the neuron"
                },
                network_id: {
                    bsonType: "string",
                    description: "UUID of the network"
                },
                spikes: {
                    bsonType: "array",
                    items: {
                        bsonType: "object",
                        required: ["timestamp", "amplitude"],
                        properties: {
                            timestamp: {
                                bsonType: "double",
                                description: "Spike timestamp in milliseconds"
                            },
                            amplitude: {
                                bsonType: "double",
                                description: "Spike amplitude"
                            }
                        }
                    }
                },
                start_time: {
                    bsonType: "date"
                },
                end_time: {
                    bsonType: "date"
                }
            }
        }
    }
});

// IoT device sensor data
db.createCollection("sensor_data", {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["device_id", "sensor_type", "value", "timestamp"],
            properties: {
                device_id: {
                    bsonType: "string",
                    description: "UUID of the IoT device"
                },
                sensor_type: {
                    bsonType: "string",
                    enum: ["temperature", "humidity", "pressure", "motion", "light", "quantum_field"],
                    description: "Type of sensor"
                },
                value: {
                    bsonType: "double",
                    description: "Sensor reading value"
                },
                unit: {
                    bsonType: "string",
                    description: "Unit of measurement"
                },
                location: {
                    bsonType: "object",
                    properties: {
                        latitude: { bsonType: "double" },
                        longitude: { bsonType: "double" },
                        altitude: { bsonType: "double" }
                    }
                },
                timestamp: {
                    bsonType: "date"
                },
                quality_score: {
                    bsonType: "double",
                    minimum: 0,
                    maximum: 1,
                    description: "Data quality score"
                }
            }
        }
    }
});

// AI model training metrics
db.createCollection("training_metrics", {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["model_id", "training_session_id", "metrics", "epoch", "timestamp"],
            properties: {
                model_id: {
                    bsonType: "string",
                    description: "UUID of the AI model"
                },
                training_session_id: {
                    bsonType: "string",
                    description: "UUID of training session"
                },
                metrics: {
                    bsonType: "object",
                    properties: {
                        loss: { bsonType: "double" },
                        accuracy: { bsonType: "double" },
                        precision: { bsonType: "double" },
                        recall: { bsonType: "double" },
                        f1_score: { bsonType: "double" },
                        learning_rate: { bsonType: "double" }
                    }
                },
                epoch: {
                    bsonType: "int",
                    minimum: 0
                },
                batch_size: {
                    bsonType: "int",
                    minimum: 1
                },
                timestamp: {
                    bsonType: "date"
                }
            }
        }
    }
});

// System performance metrics
db.createCollection("system_metrics", {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["metric_type", "value", "timestamp"],
            properties: {
                metric_type: {
                    bsonType: "string",
                    enum: ["cpu_usage", "memory_usage", "disk_usage", "network_io", "quantum_operations", "api_requests"],
                    description: "Type of system metric"
                },
                value: {
                    bsonType: "double",
                    description: "Metric value"
                },
                component: {
                    bsonType: "string",
                    description: "System component name"
                },
                instance_id: {
                    bsonType: "string",
                    description: "Instance identifier"
                },
                timestamp: {
                    bsonType: "date"
                },
                tags: {
                    bsonType: "object",
                    description: "Additional metric tags"
                }
            }
        }
    }
});

// User activity logs
db.createCollection("activity_logs", {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["user_id", "action", "timestamp"],
            properties: {
                user_id: {
                    bsonType: "string",
                    description: "UUID of the user"
                },
                action: {
                    bsonType: "string",
                    description: "Action performed"
                },
                resource_type: {
                    bsonType: "string",
                    description: "Type of resource accessed"
                },
                resource_id: {
                    bsonType: "string",
                    description: "ID of resource accessed"
                },
                ip_address: {
                    bsonType: "string",
                    description: "Client IP address"
                },
                user_agent: {
                    bsonType: "string",
                    description: "Client user agent"
                },
                timestamp: {
                    bsonType: "date"
                },
                duration_ms: {
                    bsonType: "int",
                    minimum: 0,
                    description: "Action duration in milliseconds"
                },
                status: {
                    bsonType: "string",
                    enum: ["success", "error", "warning"],
                    description: "Action status"
                }
            }
        }
    }
});

// Create indexes for performance
print('📊 Creating indexes for optimal performance...');

// Quantum results indexes
db.quantum_results.createIndex({ "circuit_id": 1, "timestamp": -1 });
db.quantum_results.createIndex({ "execution_id": 1 });
db.quantum_results.createIndex({ "timestamp": -1 });
db.quantum_results.createIndex({ "fidelity": 1 });

// Time crystal indexes
db.time_crystal_states.createIndex({ "crystal_id": 1, "timestamp": -1 });
db.time_crystal_states.createIndex({ "timestamp": -1 });
db.time_crystal_states.createIndex({ "coherence": 1 });

// Neuromorphic indexes
db.spike_trains.createIndex({ "neuron_id": 1, "start_time": -1 });
db.spike_trains.createIndex({ "network_id": 1, "start_time": -1 });
db.spike_trains.createIndex({ "start_time": -1, "end_time": -1 });

// IoT sensor data indexes
db.sensor_data.createIndex({ "device_id": 1, "timestamp": -1 });
db.sensor_data.createIndex({ "sensor_type": 1, "timestamp": -1 });
db.sensor_data.createIndex({ "timestamp": -1 });
db.sensor_data.createIndex({ "location.latitude": 1, "location.longitude": 1 });

// Training metrics indexes
db.training_metrics.createIndex({ "model_id": 1, "epoch": 1 });
db.training_metrics.createIndex({ "training_session_id": 1, "timestamp": -1 });
db.training_metrics.createIndex({ "timestamp": -1 });

// System metrics indexes
db.system_metrics.createIndex({ "metric_type": 1, "timestamp": -1 });
db.system_metrics.createIndex({ "component": 1, "timestamp": -1 });
db.system_metrics.createIndex({ "timestamp": -1 });

// Activity logs indexes
db.activity_logs.createIndex({ "user_id": 1, "timestamp": -1 });
db.activity_logs.createIndex({ "action": 1, "timestamp": -1 });
db.activity_logs.createIndex({ "timestamp": -1 });
db.activity_logs.createIndex({ "resource_type": 1, "resource_id": 1 });

// Create TTL indexes for data retention
print('⏰ Setting up TTL indexes for data retention...');

// Keep quantum results for 1 year
db.quantum_results.createIndex({ "timestamp": 1 }, { expireAfterSeconds: 31536000 });

// Keep time crystal states for 6 months
db.time_crystal_states.createIndex({ "timestamp": 1 }, { expireAfterSeconds: 15552000 });

// Keep spike trains for 3 months
db.spike_trains.createIndex({ "start_time": 1 }, { expireAfterSeconds: 7776000 });

// Keep sensor data for 1 year
db.sensor_data.createIndex({ "timestamp": 1 }, { expireAfterSeconds: 31536000 });

// Keep training metrics for 2 years
db.training_metrics.createIndex({ "timestamp": 1 }, { expireAfterSeconds: 63072000 });

// Keep system metrics for 3 months
db.system_metrics.createIndex({ "timestamp": 1 }, { expireAfterSeconds: 7776000 });

// Keep activity logs for 1 year
db.activity_logs.createIndex({ "timestamp": 1 }, { expireAfterSeconds: 31536000 });

// Insert initial sample data
print('💾 Inserting sample data...');

// Sample system metrics
db.system_metrics.insertMany([
    {
        metric_type: "cpu_usage",
        value: 45.2,
        component: "quantum_engine",
        instance_id: "quantum-01",
        timestamp: new Date(),
        tags: { environment: "production", region: "us-east-1" }
    },
    {
        metric_type: "memory_usage", 
        value: 62.8,
        component: "neuromorphic_processor",
        instance_id: "neuro-01",
        timestamp: new Date(),
        tags: { environment: "production", region: "us-east-1" }
    },
    {
        metric_type: "api_requests",
        value: 1247,
        component: "api_gateway",
        instance_id: "api-01", 
        timestamp: new Date(),
        tags: { environment: "production", endpoint: "/quantum/circuits" }
    }
]);

// Create database info collection
db.createCollection("db_info");
db.db_info.insertOne({
    initialized_at: new Date(),
    version: "1.0.0",
    description: "Quantum AI Platform MongoDB Database",
    collections: [
        "quantum_results",
        "time_crystal_states", 
        "spike_trains",
        "sensor_data",
        "training_metrics",
        "system_metrics",
        "activity_logs"
    ],
    features: [
        "Document validation schemas",
        "Performance-optimized indexes", 
        "TTL-based data retention",
        "Geospatial indexing for IoT devices",
        "Time-series optimizations"
    ]
});

print('✅ MongoDB initialization completed successfully!');
print('📊 Collections created: ' + db.getCollectionNames().length);
print('🔍 Indexes created for optimal query performance');
print('⏰ TTL policies configured for automatic data cleanup');
print('🚀 Quantum AI Platform MongoDB ready for production!');

// Show database statistics
print('📈 Database Statistics:');
printjson(db.stats());