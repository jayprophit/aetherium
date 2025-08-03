-- Quantum AI Platform Database Initialization Script
-- This script sets up the initial database schema and data

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "hstore";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS quantum_ai;
CREATE SCHEMA IF NOT EXISTS quantum_circuits;
CREATE SCHEMA IF NOT EXISTS time_crystals;
CREATE SCHEMA IF NOT EXISTS neuromorphic;
CREATE SCHEMA IF NOT EXISTS iot_devices;
CREATE SCHEMA IF NOT EXISTS ai_models;
CREATE SCHEMA IF NOT EXISTS security;

-- Set default schema search path
ALTER DATABASE quantumai_db SET search_path TO quantum_ai, quantum_circuits, time_crystals, neuromorphic, iot_devices, ai_models, security, public;

-- Users and authentication
CREATE TABLE IF NOT EXISTS security.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    roles TEXT[] DEFAULT ARRAY['user'],
    permissions TEXT[] DEFAULT ARRAY[],
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    two_factor_enabled BOOLEAN DEFAULT false,
    two_factor_secret VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS security.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES security.users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    permissions TEXT[] DEFAULT ARRAY[],
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS security.sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES security.users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT
);

-- Quantum circuits and simulations
CREATE TABLE IF NOT EXISTS quantum_circuits.circuits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    num_qubits INTEGER NOT NULL,
    circuit_data JSONB NOT NULL,
    created_by UUID REFERENCES security.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_template BOOLEAN DEFAULT false,
    category VARCHAR(100),
    tags TEXT[]
);

CREATE TABLE IF NOT EXISTS quantum_circuits.executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    circuit_id UUID REFERENCES quantum_circuits.circuits(id) ON DELETE CASCADE,
    parameters JSONB,
    shots INTEGER DEFAULT 1024,
    results JSONB,
    execution_time_ms INTEGER,
    status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Time crystal systems
CREATE TABLE IF NOT EXISTS time_crystals.crystals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    state VARCHAR(50) DEFAULT 'inactive',
    coherence DECIMAL(5,4) DEFAULT 0.0,
    frequency DECIMAL(10,6),
    phase DECIMAL(8,6),
    entanglement_strength DECIMAL(5,4),
    configuration JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_sync TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS time_crystals.synchronization_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    crystal_ids UUID[],
    event_type VARCHAR(100),
    parameters JSONB,
    success BOOLEAN,
    duration_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Neuromorphic computing
CREATE TABLE IF NOT EXISTS neuromorphic.networks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    num_neurons INTEGER NOT NULL,
    connectivity DECIMAL(5,4),
    configuration JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT false
);

CREATE TABLE IF NOT EXISTS neuromorphic.neurons (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    network_id UUID REFERENCES neuromorphic.networks(id) ON DELETE CASCADE,
    neuron_type VARCHAR(50) NOT NULL,
    state VARCHAR(50) DEFAULT 'inactive',
    membrane_potential DECIMAL(8,4),
    spike_count INTEGER DEFAULT 0,
    quantum_coherence DECIMAL(5,4),
    last_spike_time TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS neuromorphic.spike_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    neuron_id UUID REFERENCES neuromorphic.neurons(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    amplitude DECIMAL(6,3),
    post_synaptic_targets UUID[]
);

-- IoT device management
CREATE TABLE IF NOT EXISTS iot_devices.devices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    device_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'offline',
    ip_address INET,
    mqtt_topic VARCHAR(255),
    quantum_sync_enabled BOOLEAN DEFAULT false,
    configuration JSONB,
    firmware_version VARCHAR(50),
    last_seen TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS iot_devices.device_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id UUID REFERENCES iot_devices.devices(id) ON DELETE CASCADE,
    metric_type VARCHAR(100) NOT NULL,
    value DECIMAL(10,4),
    unit VARCHAR(20),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS iot_devices.device_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id UUID REFERENCES iot_devices.devices(id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB,
    severity VARCHAR(20) DEFAULT 'info',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- AI/ML models and training
CREATE TABLE IF NOT EXISTS ai_models.models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    architecture JSONB,
    parameters JSONB,
    training_status VARCHAR(50) DEFAULT 'not_trained',
    accuracy DECIMAL(5,4),
    loss DECIMAL(10,6),
    created_by UUID REFERENCES security.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    trained_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS ai_models.training_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES ai_models.models(id) ON DELETE CASCADE,
    dataset_info JSONB,
    hyperparameters JSONB,
    metrics JSONB,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'running'
);

-- System configurations
CREATE TABLE IF NOT EXISTS quantum_ai.system_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit logs
CREATE TABLE IF NOT EXISTS quantum_ai.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES security.users(id),
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_username ON security.users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON security.users(email);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON security.api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON security.sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON security.sessions(expires_at);

CREATE INDEX IF NOT EXISTS idx_circuits_created_by ON quantum_circuits.circuits(created_by);
CREATE INDEX IF NOT EXISTS idx_circuits_category ON quantum_circuits.circuits(category);
CREATE INDEX IF NOT EXISTS idx_executions_circuit_id ON quantum_circuits.executions(circuit_id);
CREATE INDEX IF NOT EXISTS idx_executions_status ON quantum_circuits.executions(status);

CREATE INDEX IF NOT EXISTS idx_crystals_state ON time_crystals.crystals(state);
CREATE INDEX IF NOT EXISTS idx_sync_events_timestamp ON time_crystals.synchronization_events(created_at);

CREATE INDEX IF NOT EXISTS idx_neurons_network_id ON neuromorphic.neurons(network_id);
CREATE INDEX IF NOT EXISTS idx_neurons_state ON neuromorphic.neurons(state);
CREATE INDEX IF NOT EXISTS idx_spike_events_neuron_id ON neuromorphic.spike_events(neuron_id);
CREATE INDEX IF NOT EXISTS idx_spike_events_timestamp ON neuromorphic.spike_events(timestamp);

CREATE INDEX IF NOT EXISTS idx_devices_status ON iot_devices.devices(status);
CREATE INDEX IF NOT EXISTS idx_devices_type ON iot_devices.devices(device_type);
CREATE INDEX IF NOT EXISTS idx_device_metrics_device_id ON iot_devices.device_metrics(device_id);
CREATE INDEX IF NOT EXISTS idx_device_metrics_timestamp ON iot_devices.device_metrics(timestamp);

CREATE INDEX IF NOT EXISTS idx_models_created_by ON ai_models.models(created_by);
CREATE INDEX IF NOT EXISTS idx_models_type ON ai_models.models(model_type);
CREATE INDEX IF NOT EXISTS idx_training_sessions_model_id ON ai_models.training_sessions(model_id);

CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON quantum_ai.audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON quantum_ai.audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON quantum_ai.audit_logs(action);

-- Insert default system configurations
INSERT INTO quantum_ai.system_configs (config_key, config_value, description) VALUES
    ('system_name', '"Quantum AI Platform"', 'Name of the system'),
    ('version', '"1.0.0"', 'Current system version'),
    ('quantum_default_qubits', '32', 'Default number of qubits for quantum simulations'),
    ('time_crystal_default_count', '8', 'Default number of time crystals in network'),
    ('neuromorphic_default_neurons', '10000', 'Default number of neurons in SNN'),
    ('session_timeout_seconds', '3600', 'User session timeout in seconds'),
    ('max_api_requests_per_hour', '1000', 'Maximum API requests per hour per user')
ON CONFLICT (config_key) DO NOTHING;

-- Create default admin user (password: admin123 - CHANGE IN PRODUCTION!)
INSERT INTO security.users (username, email, password_hash, roles, permissions, is_active) VALUES
    ('admin', 'admin@quantumai.local', crypt('admin123', gen_salt('bf', 12)), 
     ARRAY['admin', 'user'], 
     ARRAY['admin.*', 'quantum.*', 'neuromorphic.*', 'iot.*', 'ai.*'], 
     true)
ON CONFLICT (username) DO NOTHING;

-- Create system user for internal operations
INSERT INTO security.users (username, email, password_hash, roles, permissions, is_active) VALUES
    ('system', 'system@quantumai.local', crypt(gen_random_uuid()::text, gen_salt('bf', 12)), 
     ARRAY['system'], 
     ARRAY['system.*'], 
     true)
ON CONFLICT (username) DO NOTHING;

-- Create triggers for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON security.users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_circuits_updated_at BEFORE UPDATE ON quantum_circuits.circuits
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_configs_updated_at BEFORE UPDATE ON quantum_ai.system_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT USAGE ON SCHEMA quantum_ai TO quantumai;
GRANT USAGE ON SCHEMA quantum_circuits TO quantumai;
GRANT USAGE ON SCHEMA time_crystals TO quantumai;
GRANT USAGE ON SCHEMA neuromorphic TO quantumai;
GRANT USAGE ON SCHEMA iot_devices TO quantumai;
GRANT USAGE ON SCHEMA ai_models TO quantumai;
GRANT USAGE ON SCHEMA security TO quantumai;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA quantum_ai TO quantumai;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA quantum_circuits TO quantumai;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA time_crystals TO quantumai;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA neuromorphic TO quantumai;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA iot_devices TO quantumai;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ai_models TO quantumai;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA security TO quantumai;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA quantum_ai TO quantumai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA quantum_circuits TO quantumai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA time_crystals TO quantumai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA neuromorphic TO quantumai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA iot_devices TO quantumai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ai_models TO quantumai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA security TO quantumai;

-- Create database initialized marker
CREATE TABLE IF NOT EXISTS quantum_ai.db_initialization (
    initialized_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    version VARCHAR(50) DEFAULT '1.0.0',
    notes TEXT DEFAULT 'Initial database setup for Quantum AI Platform'
);

INSERT INTO quantum_ai.db_initialization DEFAULT VALUES;

COMMIT;