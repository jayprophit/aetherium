#!/bin/bash

# Quantum AI Platform Production Deployment Script
# This script deploys the platform to a production environment

set -e

echo "🚀 Quantum AI Platform - Production Deployment"
echo "==============================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_ENV=${DEPLOYMENT_ENV:-production}
BACKUP_ENABLED=${BACKUP_ENABLED:-true}
HEALTH_CHECK_TIMEOUT=${HEALTH_CHECK_TIMEOUT:-300}
ROLLBACK_ON_FAILURE=${ROLLBACK_ON_FAILURE:-true}

# Pre-deployment checks
echo -e "${BLUE}🔍 Running pre-deployment checks...${NC}"

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Warning: Running as root. Consider using a dedicated user for production.${NC}"
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    exit 1
fi

# Check environment file
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ .env file not found. Please create from .env.example${NC}"
    exit 1
fi

# Validate critical environment variables
echo -e "${BLUE}🔧 Validating environment configuration...${NC}"

source .env

# Critical variables that must be set for production
CRITICAL_VARS=(
    "JWT_SECRET_KEY"
    "ENCRYPTION_KEY"
    "DATABASE_PASSWORD"
    "MONGODB_PASSWORD"
    "REDIS_PASSWORD"
)

for var in "${CRITICAL_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo -e "${RED}❌ Critical environment variable $var is not set${NC}"
        exit 1
    fi
done

# Check if using default passwords (security risk)
DEFAULT_PASSWORDS=("password" "admin" "secret" "123456")
for default_pwd in "${DEFAULT_PASSWORDS[@]}"; do
    if [ "$DATABASE_PASSWORD" == "$default_pwd" ] || [ "$MONGODB_PASSWORD" == "$default_pwd" ]; then
        echo -e "${RED}❌ Default password detected. Please use strong passwords in production.${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✅ Environment validation passed${NC}"

# Create backup if enabled
if [ "$BACKUP_ENABLED" = true ]; then
    echo -e "${BLUE}💾 Creating pre-deployment backup...${NC}"
    
    # Create backup directory with timestamp
    BACKUP_DIR="./backups/pre-deployment-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup databases if they exist
    if docker-compose ps postgres | grep -q "Up"; then
        echo "Backing up PostgreSQL..."
        docker-compose exec -T postgres pg_dump -U quantumai quantumai_db > "$BACKUP_DIR/postgres_backup.sql"
    fi
    
    if docker-compose ps mongo | grep -q "Up"; then
        echo "Backing up MongoDB..."
        docker-compose exec -T mongo mongodump --db quantumai_mongo --archive > "$BACKUP_DIR/mongo_backup.archive"
    fi
    
    # Backup persistent volumes
    echo "Backing up persistent data..."
    cp -r ./data "$BACKUP_DIR/data_backup" 2>/dev/null || true
    cp -r ./logs "$BACKUP_DIR/logs_backup" 2>/dev/null || true
    
    echo -e "${GREEN}✅ Backup created at $BACKUP_DIR${NC}"
fi

# Build production images
echo -e "${BLUE}🏗️  Building production images...${NC}"
docker-compose build --no-cache

# Pull latest images for external services
echo -e "${BLUE}📦 Pulling latest service images...${NC}"
docker-compose pull postgres mongo redis qdrant chromadb prometheus grafana mosquitto

# Stop existing services
echo -e "${BLUE}🛑 Stopping existing services...${NC}"
docker-compose down

# Clean up old containers and images (optional)
if [ "${CLEANUP_OLD_IMAGES:-false}" = true ]; then
    echo -e "${BLUE}🧹 Cleaning up old images...${NC}"
    docker system prune -f
    docker image prune -a -f
fi

# Start infrastructure services first
echo -e "${BLUE}🗄️  Starting infrastructure services...${NC}"
docker-compose up -d postgres mongo redis qdrant chromadb mosquitto

# Wait for databases to be ready
echo -e "${BLUE}⏳ Waiting for databases to initialize...${NC}"
timeout=120
while [ $timeout -gt 0 ]; do
    postgres_ready=false
    mongo_ready=false
    redis_ready=false
    
    # Check PostgreSQL
    if docker-compose exec -T postgres pg_isready -U quantumai >/dev/null 2>&1; then
        postgres_ready=true
    fi
    
    # Check MongoDB
    if docker-compose exec -T mongo mongosh --eval "db.adminCommand('ping')" >/dev/null 2>&1; then
        mongo_ready=true
    fi
    
    # Check Redis
    if docker-compose exec -T redis redis-cli ping | grep -q PONG; then
        redis_ready=true
    fi
    
    if [ "$postgres_ready" = true ] && [ "$mongo_ready" = true ] && [ "$redis_ready" = true ]; then
        echo -e "${GREEN}✅ All databases are ready${NC}"
        break
    fi
    
    echo -e "${YELLOW}⏳ Waiting for databases... ($timeout seconds remaining)${NC}"
    sleep 5
    timeout=$((timeout-5))
done

if [ $timeout -le 0 ]; then
    echo -e "${RED}❌ Databases failed to start within timeout period${NC}"
    if [ "$ROLLBACK_ON_FAILURE" = true ]; then
        echo -e "${YELLOW}🔄 Rolling back...${NC}"
        docker-compose down
        exit 1
    fi
fi

# Run database migrations if needed
echo -e "${BLUE}🔧 Running database migrations...${NC}"
# Add migration commands here if you have them
# docker-compose exec quantum-ai-platform python -c "from backend.database.migrations import run_migrations; run_migrations()"

# Start monitoring services
echo -e "${BLUE}📊 Starting monitoring services...${NC}"
docker-compose up -d prometheus grafana

# Start main application
echo -e "${BLUE}🚀 Starting main application...${NC}"
docker-compose up -d quantum-ai-platform

# Health check
echo -e "${BLUE}🏥 Running health checks...${NC}"
timeout=$HEALTH_CHECK_TIMEOUT
health_check_passed=false

while [ $timeout -gt 0 ]; do
    # Check main application health
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        health_check_passed=true
        break
    fi
    
    echo -e "${YELLOW}⏳ Health check in progress... ($timeout seconds remaining)${NC}"
    sleep 5
    timeout=$((timeout-5))
done

if [ "$health_check_passed" = true ]; then
    echo -e "${GREEN}✅ Health check passed${NC}"
else
    echo -e "${RED}❌ Health check failed${NC}"
    
    if [ "$ROLLBACK_ON_FAILURE" = true ]; then
        echo -e "${YELLOW}🔄 Rolling back deployment...${NC}"
        docker-compose down
        
        # Restore from backup if available
        if [ "$BACKUP_ENABLED" = true ] && [ -d "$BACKUP_DIR" ]; then
            echo -e "${YELLOW}📦 Restoring from backup...${NC}"
            # Restore data
            cp -r "$BACKUP_DIR/data_backup" ./data 2>/dev/null || true
            
            # Start services and restore databases
            docker-compose up -d postgres mongo redis
            sleep 10
            
            # Restore PostgreSQL
            if [ -f "$BACKUP_DIR/postgres_backup.sql" ]; then
                docker-compose exec -T postgres psql -U quantumai -d quantumai_db < "$BACKUP_DIR/postgres_backup.sql"
            fi
            
            # Restore MongoDB
            if [ -f "$BACKUP_DIR/mongo_backup.archive" ]; then
                docker-compose exec -T mongo mongorestore --db quantumai_mongo --archive < "$BACKUP_DIR/mongo_backup.archive"
            fi
        fi
        
        exit 1
    fi
fi

# Post-deployment verification
echo -e "${BLUE}🔍 Running post-deployment verification...${NC}"

# Check all services are running
services_status=$(docker-compose ps --services --filter "status=running")
all_services=$(docker-compose ps --services)

if [ "$(echo "$services_status" | wc -l)" -eq "$(echo "$all_services" | wc -l)" ]; then
    echo -e "${GREEN}✅ All services are running${NC}"
else
    echo -e "${YELLOW}⚠️  Some services may not be running correctly${NC}"
fi

# Test key endpoints
echo -e "${BLUE}🧪 Testing key endpoints...${NC}"

# Test API health
if curl -f http://localhost:8000/health >/dev/null 2>&1; then
    echo -e "${GREEN}✅ API health endpoint: OK${NC}"
else
    echo -e "${RED}❌ API health endpoint: FAILED${NC}"
fi

# Test frontend
if curl -f http://localhost >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Frontend: OK${NC}"
else
    echo -e "${RED}❌ Frontend: FAILED${NC}"
fi

# Test Grafana
if curl -f http://localhost:3000 >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Grafana: OK${NC}"
else
    echo -e "${YELLOW}⚠️  Grafana: May not be ready yet${NC}"
fi

# Show service status
echo -e "${BLUE}📊 Final service status:${NC}"
docker-compose ps

# Display access information
echo ""
echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo ""
echo -e "${BLUE}🌐 Access URLs:${NC}"
echo "Frontend:          http://localhost"
echo "Backend API:       http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Grafana Dashboard: http://localhost:3000"
echo "Prometheus:        http://localhost:9090"
echo ""

# Security reminders
echo -e "${YELLOW}🔒 Security Reminders:${NC}"
echo "1. Ensure firewall rules are properly configured"
echo "2. Set up SSL/TLS certificates for HTTPS"
echo "3. Review and rotate all passwords and keys regularly"
echo "4. Monitor logs and metrics for suspicious activity"
echo "5. Keep all components updated with latest security patches"
echo ""

# Save deployment information
cat > ./deployment_info.json << EOF
{
    "deployment_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "$DEPLOYMENT_ENV",
    "version": "1.0.0",
    "backup_location": "$BACKUP_DIR",
    "services": {
        "frontend": "http://localhost",
        "api": "http://localhost:8000",
        "docs": "http://localhost:8000/docs",
        "grafana": "http://localhost:3000",
        "prometheus": "http://localhost:9090"
    },
    "health_check_passed": $health_check_passed,
    "deployment_duration": "$(date -u +%H:%M:%S)"
}
EOF

echo -e "${GREEN}✅ Deployment information saved to deployment_info.json${NC}"
echo -e "${GREEN}🚀 Quantum AI Platform is now running in production! 🧬⚛️🤖${NC}"