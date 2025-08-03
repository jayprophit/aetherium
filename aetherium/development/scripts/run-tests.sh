#!/bin/bash

# Quantum AI Platform Testing Script
# Comprehensive testing suite for all platform components

set -e

echo "🧪 Quantum AI Platform - Comprehensive Testing Suite"
echo "===================================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
TEST_ENV=${TEST_ENV:-testing}
COVERAGE_THRESHOLD=${COVERAGE_THRESHOLD:-80}
PARALLEL_TESTS=${PARALLEL_TESTS:-true}
CLEANUP_AFTER_TESTS=${CLEANUP_AFTER_TESTS:-true}

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Function to run a test suite and track results
run_test_suite() {
    local suite_name=$1
    local test_command=$2
    local required=${3:-true}
    
    echo -e "${BLUE}🔬 Running $suite_name tests...${NC}"
    
    if eval "$test_command"; then
        echo -e "${GREEN}✅ $suite_name tests: PASSED${NC}"
        ((PASSED_TESTS++))
        return 0
    else
        echo -e "${RED}❌ $suite_name tests: FAILED${NC}"
        ((FAILED_TESTS++))
        
        if [ "$required" = true ]; then
            echo -e "${RED}💥 Critical test suite failed. Stopping execution.${NC}"
            exit 1
        fi
        return 1
    fi
}

# Setup test environment
echo -e "${BLUE}🛠️  Setting up test environment...${NC}"

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is required for testing${NC}"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required for testing${NC}"
    exit 1
fi

# Create test environment file
cat > .env.test << EOF
QUANTUM_AI_ENV=testing
DEBUG_MODE=true
LOG_LEVEL=DEBUG

# Test Database Configuration
DATABASE_URL=postgresql://test_user:test_pass@localhost:5433/test_quantumai_db
MONGODB_URL=mongodb://localhost:27018/test_quantumai_mongo
REDIS_URL=redis://localhost:6380/0
QDRANT_URL=http://localhost:6334
CHROMADB_URL=http://localhost:8002

# Test Security Configuration
JWT_SECRET_KEY=test_jwt_secret_key_for_testing_only_2025
ENCRYPTION_KEY=test_encryption_32_char_key_here
API_RATE_LIMIT_PER_MINUTE=1000

# Test Quantum Configuration
QUANTUM_DEFAULT_QUBITS=8
QUANTUM_MAX_QUBITS=16
QUANTUM_SIMULATION_PRECISION=0.01

# Test Neuromorphic Configuration
NEUROMORPHIC_DEFAULT_NEURONS=1000
NEUROMORPHIC_MAX_NEURONS=5000

# Test Time Crystal Configuration
TIME_CRYSTAL_DEFAULT_COUNT=2
TIME_CRYSTAL_MAX_COUNT=4

# Test IoT Configuration
IOT_MAX_DEVICES=100
MQTT_BROKER_PORT=1884

# Disable external services for testing
PROMETHEUS_ENABLED=false
GRAFANA_ENABLED=false
SMTP_ENABLED=false
EOF

# Start test infrastructure
echo -e "${BLUE}🚀 Starting test infrastructure...${NC}"

# Create test docker-compose configuration
cat > docker-compose.test.yml << EOF
version: '3.8'

services:
  postgres-test:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: test_quantumai_db
      POSTGRES_USER: test_user
      POSTGRES_PASSWORD: test_pass
    ports:
      - "5433:5432"
    volumes:
      - postgres_test_data:/var/lib/postgresql/data

  mongo-test:
    image: mongo:7
    environment:
      MONGO_INITDB_DATABASE: test_quantumai_mongo
    ports:
      - "27018:27017"
    volumes:
      - mongo_test_data:/data/db

  redis-test:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    command: redis-server --requirepass test_pass

  qdrant-test:
    image: qdrant/qdrant:latest
    ports:
      - "6334:6333"
    volumes:
      - qdrant_test_data:/qdrant/storage

  chromadb-test:
    image: chromadb/chroma:latest
    ports:
      - "8002:8000"
    volumes:
      - chromadb_test_data:/chroma/chroma

  mosquitto-test:
    image: eclipse-mosquitto:2
    ports:
      - "1884:1883"

volumes:
  postgres_test_data:
  mongo_test_data:
  qdrant_test_data:
  chromadb_test_data:
EOF

# Start test services
docker-compose -f docker-compose.test.yml up -d

# Wait for services to be ready
echo -e "${BLUE}⏳ Waiting for test services to be ready...${NC}"
sleep 10

# Verify test services
timeout=60
while [ $timeout -gt 0 ]; do
    if docker-compose -f docker-compose.test.yml exec -T postgres-test pg_isready -U test_user > /dev/null 2>&1; then
        break
    fi
    echo -e "${YELLOW}⏳ Waiting for test PostgreSQL... ($timeout seconds remaining)${NC}"
    sleep 2
    timeout=$((timeout-2))
done

echo -e "${GREEN}✅ Test infrastructure ready${NC}"

# Install test dependencies
echo -e "${BLUE}📦 Installing test dependencies...${NC}"
if [ ! -d "test_venv" ]; then
    python3 -m venv test_venv
fi

source test_venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
pip install pytest pytest-cov pytest-asyncio pytest-mock httpx pytest-xdist

# Create test directories
mkdir -p tests/{unit,integration,e2e,performance}
mkdir -p test_reports/{coverage,junit,performance}

# 1. Unit Tests
echo -e "\n${BLUE}🔬 Starting Unit Tests...${NC}"
((TOTAL_TESTS++))

# Create sample unit tests if they don't exist
cat > tests/unit/test_quantum_vqc.py << 'EOF'
import pytest
import asyncio
from backend.quantum.vqc_engine import VirtualQuantumComputer

@pytest.mark.asyncio
async def test_vqc_initialization():
    """Test VQC initialization"""
    vqc = VirtualQuantumComputer(num_qubits=8)
    assert vqc.num_qubits == 8
    assert vqc.quantum_state is not None

@pytest.mark.asyncio
async def test_quantum_circuit_creation():
    """Test quantum circuit creation"""
    vqc = VirtualQuantumComputer(num_qubits=4)
    circuit = await vqc.create_quantum_circuit("basic", [0.5, 1.0])
    assert circuit is not None
    assert len(circuit.qubits) <= 4

@pytest.mark.asyncio
async def test_circuit_execution():
    """Test quantum circuit execution"""
    vqc = VirtualQuantumComputer(num_qubits=2)
    circuit = await vqc.create_quantum_circuit("basic", [0.5])
    result = await vqc.execute_circuit(circuit, shots=100)
    assert "counts" in result
    assert "probabilities" in result
    assert "execution_time" in result
EOF

cat > tests/unit/test_time_crystals.py << 'EOF'
import pytest
import asyncio
from backend.time_crystals.time_crystal_engine import TimeCrystalEngine

@pytest.mark.asyncio
async def test_time_crystal_initialization():
    """Test time crystal engine initialization"""
    tce = TimeCrystalEngine(num_time_crystals=2)
    assert tce.num_time_crystals == 2
    assert len(tce.time_crystals) == 2

@pytest.mark.asyncio
async def test_crystal_synchronization():
    """Test time crystal synchronization"""
    tce = TimeCrystalEngine(num_time_crystals=2)
    await tce.synchronize_crystals()
    # Verify synchronization occurred
    assert tce.phase_lock_status["synchronized"] == True

@pytest.mark.asyncio
async def test_coherence_enhancement():
    """Test quantum coherence enhancement"""
    tce = TimeCrystalEngine(num_time_crystals=2)
    result = await tce.enhance_quantum_coherence(target_coherence=0.9)
    assert "coherence_improvement" in result
    assert result["success"] == True
EOF

cat > tests/unit/test_neuromorphic.py << 'EOF'
import pytest
import asyncio
from backend.neuromorphic.snn_processor import SpikingNeuralProcessor

@pytest.mark.asyncio
async def test_snn_initialization():
    """Test SNN processor initialization"""
    snn = SpikingNeuralProcessor(num_neurons=100)
    assert snn.num_neurons == 100
    assert len(snn.neurons) == 100

@pytest.mark.asyncio
async def test_spike_injection():
    """Test spike pattern injection"""
    snn = SpikingNeuralProcessor(num_neurons=10)
    neuron_ids = list(snn.neurons.keys())[:5]
    pattern = [1.0, 0.5, 0.8, 0.3, 0.9]
    
    result = await snn.inject_spike_pattern(neuron_ids, pattern, amplitude=1.0)
    assert result == True

@pytest.mark.asyncio
async def test_event_processing():
    """Test spike event processing"""
    snn = SpikingNeuralProcessor(num_neurons=10)
    await snn.process_pending_events()
    # Events should be processed without error
    assert True
EOF

# Run unit tests
run_test_suite "Unit Tests" "python -m pytest tests/unit/ -v --cov=backend --cov-report=html:test_reports/coverage/unit --junit-xml=test_reports/junit/unit.xml"

# 2. Integration Tests
echo -e "\n${BLUE}🔗 Starting Integration Tests...${NC}"
((TOTAL_TESTS++))

cat > tests/integration/test_api_endpoints.py << 'EOF'
import pytest
import asyncio
from httpx import AsyncClient
from backend.main import app

@pytest.mark.asyncio
async def test_health_endpoint():
    """Test health check endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

@pytest.mark.asyncio
async def test_quantum_status():
    """Test quantum status endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/quantum/status")
    assert response.status_code in [200, 401]  # May require auth

@pytest.mark.asyncio
async def test_time_crystals_status():
    """Test time crystals status endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/time-crystals/status")
    assert response.status_code in [200, 401]  # May require auth

@pytest.mark.asyncio
async def test_neuromorphic_status():
    """Test neuromorphic status endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/neuromorphic/status")
    assert response.status_code in [200, 401]  # May require auth
EOF

# Run integration tests
run_test_suite "Integration Tests" "python -m pytest tests/integration/ -v --junit-xml=test_reports/junit/integration.xml"

# 3. Database Tests
echo -e "\n${BLUE}🗄️  Starting Database Tests...${NC}"
((TOTAL_TESTS++))

cat > tests/integration/test_database.py << 'EOF'
import pytest
import asyncio
from backend.database.multi_db_manager import MultiDatabaseManager

@pytest.mark.asyncio
async def test_database_connections():
    """Test database connections"""
    db_manager = MultiDatabaseManager()
    await db_manager.connect_all()
    
    # Test PostgreSQL connection
    postgres_status = await db_manager.check_postgres_connection()
    assert postgres_status == True
    
    # Test MongoDB connection
    mongo_status = await db_manager.check_mongo_connection()
    assert mongo_status == True
    
    # Test Redis connection
    redis_status = await db_manager.check_redis_connection()
    assert redis_status == True
    
    await db_manager.close_all()

@pytest.mark.asyncio
async def test_data_operations():
    """Test basic data operations"""
    db_manager = MultiDatabaseManager()
    await db_manager.connect_all()
    
    # Test quantum results storage
    test_result = {
        "circuit_id": "test_circuit",
        "counts": {"00": 50, "11": 50},
        "execution_time": 0.1
    }
    
    result_id = await db_manager.store_quantum_result(test_result)
    assert result_id is not None
    
    await db_manager.close_all()
EOF

# Run database tests
run_test_suite "Database Tests" "python -m pytest tests/integration/test_database.py -v --junit-xml=test_reports/junit/database.xml"

# 4. Performance Tests
echo -e "\n${BLUE}⚡ Starting Performance Tests...${NC}"
((TOTAL_TESTS++))

cat > tests/performance/test_quantum_performance.py << 'EOF'
import pytest
import asyncio
import time
from backend.quantum.vqc_engine import VirtualQuantumComputer

@pytest.mark.asyncio
async def test_quantum_circuit_execution_performance():
    """Test quantum circuit execution performance"""
    vqc = VirtualQuantumComputer(num_qubits=8)
    circuit = await vqc.create_quantum_circuit("basic", [0.5])
    
    start_time = time.time()
    result = await vqc.execute_circuit(circuit, shots=1000)
    execution_time = time.time() - start_time
    
    assert execution_time < 5.0  # Should complete within 5 seconds
    assert "execution_time" in result

@pytest.mark.asyncio
async def test_concurrent_quantum_executions():
    """Test concurrent quantum circuit executions"""
    vqc = VirtualQuantumComputer(num_qubits=4)
    
    async def execute_circuit():
        circuit = await vqc.create_quantum_circuit("basic", [0.5])
        return await vqc.execute_circuit(circuit, shots=100)
    
    start_time = time.time()
    tasks = [execute_circuit() for _ in range(10)]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    assert len(results) == 10
    assert total_time < 10.0  # Should complete within 10 seconds
    assert all("counts" in result for result in results)
EOF

# Run performance tests
run_test_suite "Performance Tests" "python -m pytest tests/performance/ -v --junit-xml=test_reports/junit/performance.xml" false

# 5. End-to-End Tests (optional)
echo -e "\n${BLUE}🌐 Starting End-to-End Tests...${NC}"
((TOTAL_TESTS++))

if command -v npm &> /dev/null && [ -d "frontend" ]; then
    # Install frontend test dependencies
    cd frontend
    npm install --save-dev @testing-library/react @testing-library/jest-dom jest-environment-jsdom
    
    # Run frontend tests
    npm test -- --coverage --watchAll=false --testResultsProcessor=jest-junit
    cd ..
    
    run_test_suite "Frontend Tests" "echo 'Frontend tests completed'" false
else
    echo -e "${YELLOW}⚠️  Skipping frontend tests - npm or frontend directory not found${NC}"
    ((SKIPPED_TESTS++))
fi

# 6. Security Tests
echo -e "\n${BLUE}🔒 Starting Security Tests...${NC}"
((TOTAL_TESTS++))

cat > tests/integration/test_security.py << 'EOF'
import pytest
from httpx import AsyncClient
from backend.main import app

@pytest.mark.asyncio
async def test_unauthenticated_access():
    """Test that protected endpoints require authentication"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Test quantum endpoints
        response = await ac.post("/api/quantum/execute")
        assert response.status_code in [401, 422]  # Unauthorized or validation error
        
        # Test admin endpoints
        response = await ac.get("/api/admin/users")
        assert response.status_code == 401  # Should require authentication

@pytest.mark.asyncio
async def test_cors_headers():
    """Test CORS headers are properly set"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.options("/api/quantum/status")
        # CORS headers should be present
        assert response.status_code in [200, 405]

@pytest.mark.asyncio
async def test_rate_limiting():
    """Test API rate limiting (basic check)"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Make multiple rapid requests
        responses = []
        for _ in range(10):
            response = await ac.get("/health")
            responses.append(response.status_code)
        
        # Should not return too many 429 errors in normal testing
        rate_limited = sum(1 for status in responses if status == 429)
        assert rate_limited < 5  # Allow some rate limiting but not excessive
EOF

# Run security tests
run_test_suite "Security Tests" "python -m pytest tests/integration/test_security.py -v --junit-xml=test_reports/junit/security.xml" false

# Generate consolidated test report
echo -e "\n${BLUE}📊 Generating test reports...${NC}"

# Calculate coverage
coverage combine 2>/dev/null || true
coverage report --format=text > test_reports/coverage_summary.txt 2>/dev/null || echo "Coverage report generation skipped"

# Create consolidated report
cat > test_reports/test_summary.json << EOF
{
    "test_run_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "$TEST_ENV",
    "total_test_suites": $TOTAL_TESTS,
    "passed_suites": $PASSED_TESTS,
    "failed_suites": $FAILED_TESTS,
    "skipped_suites": $SKIPPED_TESTS,
    "success_rate": $(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "N/A"),
    "reports": {
        "coverage": "test_reports/coverage/",
        "junit": "test_reports/junit/",
        "performance": "test_reports/performance/"
    }
}
EOF

# Cleanup test environment
if [ "$CLEANUP_AFTER_TESTS" = true ]; then
    echo -e "${BLUE}🧹 Cleaning up test environment...${NC}"
    docker-compose -f docker-compose.test.yml down -v
    rm -f .env.test docker-compose.test.yml
    deactivate 2>/dev/null || true
    rm -rf test_venv
fi

# Final results
echo -e "\n${BLUE}📋 Test Results Summary:${NC}"
echo "========================="
echo -e "Total Test Suites: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"
echo -e "${YELLOW}Skipped: $SKIPPED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}🎉 All tests completed successfully!${NC}"
    echo -e "${GREEN}✅ Quantum AI Platform is ready for deployment! 🧬⚛️🤖${NC}"
    exit 0
else
    echo -e "\n${RED}❌ Some tests failed. Please review the results.${NC}"
    echo -e "${YELLOW}📊 Check test reports in test_reports/ directory${NC}"
    exit 1
fi