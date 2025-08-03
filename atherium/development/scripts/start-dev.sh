#!/bin/bash

# Quantum AI Platform Development Startup Script
# This script sets up and starts the development environment

set -e

echo "🚀 Starting Quantum AI Platform - Development Environment"
echo "============================================================"

# Check if we're in the correct directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: Please run this script from the quantum-ai-platform directory"
    exit 1
fi

# Check for required tools
echo "🔍 Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "⚠️  Node.js not found. Frontend development will require Node.js 18+"
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "⚠️  Python 3 not found. Backend development will require Python 3.11+"
fi

echo "✅ Prerequisites check completed"

# Setup environment file
if [ ! -f ".env" ]; then
    echo "📝 Creating environment file from template..."
    cp .env.example .env
    echo "✅ Environment file created. Please review and modify .env as needed."
else
    echo "✅ Environment file already exists"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs data uploads backups temp
mkdir -p frontend/build backend/logs
chmod 755 logs data uploads backups temp

# Start development services
echo "🚀 Starting development services..."

# Start databases and infrastructure first
echo "🗄️  Starting databases..."
docker-compose up -d postgres mongo redis qdrant chromadb mosquitto

# Wait for databases to be ready
echo "⏳ Waiting for databases to initialize..."
sleep 15

# Check database health
echo "🏥 Checking database health..."
timeout=60
while [ $timeout -gt 0 ]; do
    if docker-compose exec -T postgres pg_isready -U quantumai > /dev/null 2>&1; then
        echo "✅ PostgreSQL is ready"
        break
    fi
    echo "⏳ Waiting for PostgreSQL... ($timeout seconds remaining)"
    sleep 2
    timeout=$((timeout-2))
done

timeout=60
while [ $timeout -gt 0 ]; do
    if docker-compose exec -T mongo mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
        echo "✅ MongoDB is ready"
        break
    fi
    echo "⏳ Waiting for MongoDB... ($timeout seconds remaining)"
    sleep 2
    timeout=$((timeout-2))
done

# Install backend dependencies if running locally
if [ "$1" == "--local" ]; then
    echo "🐍 Setting up local Python environment..."
    if [ ! -d "backend/venv" ]; then
        python3 -m venv backend/venv
    fi
    
    source backend/venv/bin/activate
    pip install --upgrade pip
    pip install -r backend/requirements.txt
    echo "✅ Backend dependencies installed"
    
    # Install frontend dependencies
    if command -v npm &> /dev/null; then
        echo "📦 Installing frontend dependencies..."
        cd frontend
        npm install
        cd ..
        echo "✅ Frontend dependencies installed"
    fi
    
    echo "🚀 Starting services locally..."
    echo "Backend: python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"
    echo "Frontend: cd frontend && npm start"
    echo ""
    echo "💡 Run these commands in separate terminals:"
    echo "Terminal 1: cd backend && source venv/bin/activate && python -m uvicorn main:app --reload"
    echo "Terminal 2: cd frontend && npm start"
else
    # Start the main application
    echo "🚀 Starting main application..."
    docker-compose up -d quantum-ai-platform
    
    # Start monitoring services
    echo "📊 Starting monitoring services..."
    docker-compose up -d prometheus grafana
fi

# Wait for main application to be ready
echo "⏳ Waiting for application to start..."
timeout=120
while [ $timeout -gt 0 ]; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Application is ready!"
        break
    fi
    echo "⏳ Starting application... ($timeout seconds remaining)"
    sleep 3
    timeout=$((timeout-3))
done

# Show service status
echo ""
echo "🎯 Service Status:"
echo "=================="
docker-compose ps

echo ""
echo "🌐 Access URLs:"
echo "==============="
echo "🔗 Frontend:          http://localhost"
echo "🔗 Backend API:       http://localhost:8000"
echo "🔗 API Documentation: http://localhost:8000/docs"
echo "🔗 Grafana Dashboard: http://localhost:3000 (admin/quantum_admin_2025)"
echo "🔗 Prometheus:        http://localhost:9090"

echo ""
echo "📊 System Information:"
echo "======================"
echo "🔄 Database Status:"
docker-compose exec postgres pg_isready -U quantumai 2>/dev/null && echo "  ✅ PostgreSQL: Connected" || echo "  ❌ PostgreSQL: Not connected"
docker-compose exec mongo mongosh --quiet --eval "print('  ✅ MongoDB: Connected')" 2>/dev/null || echo "  ❌ MongoDB: Not connected"
docker-compose exec redis redis-cli ping | grep -q PONG && echo "  ✅ Redis: Connected" || echo "  ❌ Redis: Not connected"

echo ""
echo "🔧 Development Commands:"
echo "========================"
echo "📋 View logs:           docker-compose logs -f [service_name]"
echo "🔄 Restart service:     docker-compose restart [service_name]"
echo "🛑 Stop all services:   docker-compose down"
echo "🗑️  Clean volumes:       docker-compose down -v"
echo "🏗️  Rebuild images:      docker-compose build --no-cache"
echo "📊 Service status:      docker-compose ps"
echo "🐚 Shell access:        docker-compose exec [service_name] bash"

echo ""
echo "✅ Quantum AI Platform development environment is ready!"
echo "🚀 Happy coding! 🧬⚛️🤖"