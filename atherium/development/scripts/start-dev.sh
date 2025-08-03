#!/bin/bash

# Atherium - Advanced Quantum AI Platform
# Development Environment Startup Script

set -e

echo "🌌 Starting Atherium Development Environment..."
echo "================================================="

# Configuration
ATHERIUM_ENV="development"
PLATFORM_DIR="$(pwd)/platform"
AI_SYSTEMS_DIR="$(pwd)/ai-systems"
DOCS_DIR="$(pwd)/docs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo -e "\n${PURPLE}=== $1 ===${NC}"
}

# Check prerequisites
print_section "Prerequisites Check"

check_command() {
    if command -v $1 &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

# Check required commands
MISSING_DEPS=0
check_command "docker" || MISSING_DEPS=1
check_command "docker-compose" || MISSING_DEPS=1
check_command "python3" || MISSING_DEPS=1
check_command "node" || MISSING_DEPS=1
check_command "npm" || MISSING_DEPS=1

if [ $MISSING_DEPS -eq 1 ]; then
    print_error "Missing required dependencies. Please install them before continuing."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_status "Python version: $PYTHON_VERSION"

# Check Node version
NODE_VERSION=$(node --version)
print_status "Node.js version: $NODE_VERSION"

# Environment Setup
print_section "Environment Setup"

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_success "Created .env from .env.example"
    else
        print_warning ".env.example not found, using platform/.env.example"
        if [ -f "platform/.env.example" ]; then
            cp platform/.env.example .env
            print_success "Created .env from platform/.env.example"
        else
            print_error "No .env.example found"
            exit 1
        fi
    fi
else
    print_success ".env file exists"
fi

# Source environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    print_success "Environment variables loaded"
fi

# Database Services Setup
print_section "Database Services"

print_status "Starting database services with Docker Compose..."

# Start databases
if [ -f "platform/docker-compose.yml" ]; then
    cd platform
    docker-compose up -d mongodb postgresql redis qdrant
    cd ..
    print_success "Database services started"
else
    print_warning "platform/docker-compose.yml not found, skipping database startup"
fi

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 10

# Backend Setup
print_section "Backend Setup"

if [ -d "$PLATFORM_DIR" ]; then
    print_status "Setting up Atherium Platform Backend..."
    
    # Install Python dependencies
    if [ -f "$PLATFORM_DIR/requirements.txt" ]; then
        print_status "Installing Python dependencies..."
        pip3 install -r "$PLATFORM_DIR/requirements.txt"
        print_success "Python dependencies installed"
    else
        print_warning "platform/requirements.txt not found"
    fi
    
    # Start backend server
    print_status "Starting Atherium Platform Backend..."
    cd "$PLATFORM_DIR"
    python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
    cd ..
    print_success "Backend started (PID: $BACKEND_PID)"
else
    print_error "Platform directory not found"
    exit 1
fi

# Frontend Setup
print_section "Frontend Setup"

if [ -d "$PLATFORM_DIR/frontend" ]; then
    print_status "Setting up Atherium Platform Frontend..."
    
    cd "$PLATFORM_DIR/frontend"
    
    # Install Node dependencies
    if [ -f "package.json" ]; then
        print_status "Installing Node.js dependencies..."
        npm install
        print_success "Node.js dependencies installed"
        
        # Start frontend development server
        print_status "Starting Atherium Platform Frontend..."
        npm start &
        FRONTEND_PID=$!
        print_success "Frontend started (PID: $FRONTEND_PID)"
    else
        print_warning "platform/frontend/package.json not found"
    fi
    
    cd ../..
else
    print_warning "Platform frontend directory not found"
fi

# Health Checks
print_section "Health Checks"

sleep 5

# Check backend health
print_status "Checking backend health..."
if curl -f http://localhost:8000/health &> /dev/null; then
    print_success "Backend is healthy"
else
    print_warning "Backend health check failed"
fi

# Check frontend
print_status "Checking frontend..."
if curl -f http://localhost:3000 &> /dev/null; then
    print_success "Frontend is accessible"
else
    print_warning "Frontend not accessible yet (may still be starting)"
fi

# Development Information
print_section "Development Environment Ready"

echo ""
echo "🌌 Atherium Development Environment is Ready!"
echo "=============================================="
echo ""
echo "📚 Platform Access:"
echo "   Frontend:      http://localhost:3000"
echo "   Backend API:   http://localhost:8000"
echo "   API Docs:      http://localhost:8000/docs"
echo ""
echo "📊 Database Services:"
echo "   MongoDB:       localhost:27017"
echo "   PostgreSQL:    localhost:5432"
echo "   Redis:         localhost:6379"
echo "   Qdrant:        localhost:6333"
echo ""
echo "🔧 Development Commands:"
echo "   Stop services: docker-compose -f platform/docker-compose.yml down"
echo "   View logs:     docker-compose -f platform/docker-compose.yml logs -f"
echo "   Run tests:     ./development/scripts/run-tests.sh"
echo ""
echo "📝 Configuration:"
echo "   Main config:   atherium-config.yaml"
echo "   Environment:   .env"
echo ""

# Save PIDs for cleanup
echo $BACKEND_PID > /tmp/atherium_backend.pid
echo $FRONTEND_PID > /tmp/atherium_frontend.pid

print_success "Atherium development environment is running!"
print_status "Press Ctrl+C to stop all services"

# Wait for interrupt
trap 'echo -e "\n${YELLOW}Shutting down Atherium development environment...${NC}"; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; docker-compose -f platform/docker-compose.yml down; exit 0' INT

wait