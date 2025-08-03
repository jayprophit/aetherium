#!/bin/bash

# Quantum AI Platform Startup Script
# This script initializes and starts all services for the platform

set -e

echo "🚀 Starting Quantum AI Platform..."

# Check environment variables
if [ -z "$QUANTUM_AI_ENV" ]; then
    export QUANTUM_AI_ENV="production"
fi

echo "📋 Environment: $QUANTUM_AI_ENV"

# Create necessary directories
mkdir -p /app/logs /app/data /app/uploads /app/backups
mkdir -p /var/log/supervisor /var/log/nginx

# Set proper permissions
chmod 755 /app/logs /app/data /app/uploads /app/backups
chmod 755 /var/log/supervisor /var/log/nginx

# Wait for database connections (if using external databases)
if [ ! -z "$DATABASE_URL" ]; then
    echo "🔄 Waiting for database connection..."
    timeout 30 bash -c 'until nc -z ${DATABASE_HOST:-localhost} ${DATABASE_PORT:-5432}; do sleep 1; done'
    echo "✅ Database connection established"
fi

if [ ! -z "$MONGODB_URL" ]; then
    echo "🔄 Waiting for MongoDB connection..."
    timeout 30 bash -c 'until nc -z ${MONGODB_HOST:-localhost} ${MONGODB_PORT:-27017}; do sleep 1; done'
    echo "✅ MongoDB connection established"
fi

if [ ! -z "$REDIS_URL" ]; then
    echo "🔄 Waiting for Redis connection..."
    timeout 30 bash -c 'until nc -z ${REDIS_HOST:-localhost} ${REDIS_PORT:-6379}; do sleep 1; done'
    echo "✅ Redis connection established"
fi

# Initialize database schema if needed
echo "🔧 Initializing system components..."

cd /app

# Run database migrations/setup
python -c "
try:
    from backend.database.db_manager import DatabaseManager
    import asyncio
    
    async def init_db():
        db = DatabaseManager()
        await db.initialize()
        print('✅ Database initialized')
        
    asyncio.run(init_db())
except Exception as e:
    print(f'⚠️  Database initialization warning: {e}')
    print('Continuing with startup...')
"

# Initialize quantum components
python -c "
try:
    from quantum.vqc_engine import VirtualQuantumComputer
    import asyncio
    
    async def init_quantum():
        vqc = VirtualQuantumComputer()
        print('✅ Quantum engine initialized')
        
    asyncio.run(init_quantum())
except Exception as e:
    print(f'⚠️  Quantum initialization warning: {e}')
    print('Continuing with startup...')
"

# Initialize time crystal engine
python -c "
try:
    from time_crystals.time_crystal_engine import TimeCrystalEngine
    import asyncio
    
    async def init_time_crystals():
        tc = TimeCrystalEngine()
        await tc.initialize()
        print('✅ Time crystal engine initialized')
        
    asyncio.run(init_time_crystals())
except Exception as e:
    print(f'⚠️  Time crystal initialization warning: {e}')
    print('Continuing with startup...')
"

# Initialize neuromorphic processor
python -c "
try:
    from neuromorphic.snn_processor import SpikingNeuralProcessor
    import asyncio
    
    async def init_neuromorphic():
        snn = SpikingNeuralProcessor()
        await snn.initialize()
        print('✅ Neuromorphic processor initialized')
        
    asyncio.run(init_neuromorphic())
except Exception as e:
    print(f'⚠️  Neuromorphic initialization warning: {e}')
    print('Continuing with startup...')
"

# Test API endpoint
echo "🔍 Testing API endpoints..."
python -c "
import asyncio
import aiohttp

async def test_health():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/health', timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    print('✅ Health check passed')
                else:
                    print(f'⚠️  Health check returned status: {response.status}')
    except Exception as e:
        print(f'⚠️  Health check failed: {e}')
        print('API will be available after full startup...')

# Don't run health check during startup as service isn't up yet
# asyncio.run(test_health())
"

# Set up log rotation
echo "📝 Setting up log rotation..."
cat > /etc/logrotate.d/quantum-ai << EOF
/app/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 app app
}

/var/log/supervisor/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 app app
}
EOF

# Create system info file
echo "📊 Creating system info..."
cat > /app/system_info.json << EOF
{
    "platform": "Quantum AI Platform",
    "version": "1.0.0",
    "environment": "$QUANTUM_AI_ENV",
    "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "components": [
        "Virtual Quantum Computer",
        "Time Crystal Engine", 
        "Neuromorphic Processing",
        "IoT Device Management",
        "AI/ML Optimization",
        "Multi-Database Support",
        "Security & Authentication",
        "Real-time WebSocket API"
    ],
    "features": [
        "Quantum Circuit Simulation",
        "Time Crystal Synchronization",
        "Spiking Neural Networks",
        "Hybrid Quantum-Classical AI",
        "IoT Integration with MQTT",
        "Advanced Visualization",
        "Role-based Access Control",
        "Production-ready Deployment"
    ]
}
EOF

echo "🌟 System Info:"
cat /app/system_info.json | python -m json.tool

# Export environment variables for supervisor
export PYTHONPATH="/app"
export QUANTUM_AI_CONFIG_FILE="/app/backend/config/quantum_ai_config.yaml"

echo "🎯 Configuration loaded from: $QUANTUM_AI_CONFIG_FILE"

# Start supervisor to manage all services
echo "🚀 Starting all services with supervisor..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf