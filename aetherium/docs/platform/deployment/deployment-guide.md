# Quantum AI Platform - Deployment Guide

## Overview

This comprehensive guide covers deploying the Quantum AI Platform in development, staging, and production environments. The platform supports containerized deployment using Docker and Docker Compose, with options for cloud deployment and scaling.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Development Deployment](#development-deployment)
4. [Production Deployment](#production-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Scaling and High Availability](#scaling-and-high-availability)
7. [Security Configuration](#security-configuration)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Backup and Recovery](#backup-and-recovery)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- **CPU:** 4 cores (8 cores recommended for production)
- **RAM:** 8 GB (32 GB recommended for production)
- **Storage:** 100 GB SSD (500 GB recommended for production)
- **Network:** 1 Gbps connection (10 Gbps for high-throughput scenarios)

**Production Requirements:**
- **CPU:** 16+ cores with AVX2 support for quantum simulation
- **RAM:** 64 GB+ for large-scale quantum circuits and neuromorphic networks
- **Storage:** 1 TB+ NVMe SSD for fast database operations
- **GPU:** NVIDIA GPU with CUDA support (optional, for AI acceleration)

### Software Dependencies

**Required Software:**
- **Docker:** Version 20.10+
- **Docker Compose:** Version 2.0+
- **Git:** For source code management
- **OpenSSL:** For certificate generation

**Optional Software:**
- **Kubernetes:** For container orchestration (production)
- **Terraform:** For infrastructure as code
- **Ansible:** For configuration management

### Network Requirements

**Ports Required:**
- `80` - HTTP (redirects to HTTPS in production)
- `443` - HTTPS (production)
- `8000` - Backend API (development)
- `3000` - Grafana Dashboard
- `9090` - Prometheus Metrics
- `5432` - PostgreSQL (internal)
- `27017` - MongoDB (internal)
- `6379` - Redis (internal)
- `6333` - Qdrant Vector Database (internal)
- `8001` - ChromaDB (internal)
- `1883` - MQTT Broker (internal/IoT devices)

**Firewall Configuration:**
```bash
# Allow HTTP and HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow SSH (secure access)
sudo ufw allow 22/tcp

# Internal services (restrict to local network)
sudo ufw allow from 10.0.0.0/8 to any port 5432
sudo ufw allow from 10.0.0.0/8 to any port 27017
sudo ufw allow from 10.0.0.0/8 to any port 6379

# IoT devices (configure specific network range)
sudo ufw allow from 192.168.1.0/24 to any port 1883
```

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/quantum-ai-platform.git
cd quantum-ai-platform
```

### 2. Environment Configuration

Create environment file from template:
```bash
cp .env.example .env
```

**Critical Configuration Items:**

```bash
# Environment
QUANTUM_AI_ENV=production
DEBUG_MODE=false

# Security (MUST CHANGE IN PRODUCTION)
JWT_SECRET_KEY="your-secure-32-character-secret-key-here"
ENCRYPTION_KEY="exactly-32-characters-for-aes-256"

# Database Passwords (MUST CHANGE)
DATABASE_PASSWORD="secure-postgres-password"
MONGODB_PASSWORD="secure-mongo-password"
REDIS_PASSWORD="secure-redis-password"

# Domain Configuration
CORS_ORIGINS=["https://your-domain.com"]
SESSION_COOKIE_SECURE=true

# SSL/TLS Configuration
SSL_CERT_PATH="/etc/ssl/certs/quantum-ai.crt"
SSL_KEY_PATH="/etc/ssl/private/quantum-ai.key"
```

### 3. SSL Certificate Setup

**Option A: Let's Encrypt (Recommended for Production)**
```bash
# Install Certbot
sudo apt update
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates to platform directory
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ./ssl/quantum-ai.crt
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ./ssl/quantum-ai.key
```

**Option B: Self-Signed (Development Only)**
```bash
mkdir -p ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/quantum-ai.key \
  -out ssl/quantum-ai.crt \
  -subj "/C=US/ST=CA/L=SF/O=QuantumAI/CN=localhost"
```

## Development Deployment

### Quick Start

```bash
# Start development environment
chmod +x scripts/start-dev.sh
./scripts/start-dev.sh
```

### Manual Development Setup

```bash
# Start databases and infrastructure
docker-compose up -d postgres mongo redis qdrant chromadb mosquitto

# Wait for databases to initialize
sleep 15

# Start monitoring (optional)
docker-compose up -d prometheus grafana

# Start main application
docker-compose up -d quantum-ai-platform

# Verify deployment
curl http://localhost:8000/health
```

### Local Development (No Docker)

```bash
# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows
pip install -r requirements.txt

# Start backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend setup (separate terminal)
cd frontend
npm install
npm start
```

## Production Deployment

### Automated Production Deployment

```bash
# Run production deployment script
chmod +x scripts/deploy-production.sh
DEPLOYMENT_ENV=production ./scripts/deploy-production.sh
```

### Manual Production Deployment

**Step 1: System Preparation**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create quantum-ai user
sudo useradd -m -s /bin/bash quantum-ai
sudo usermod -aG docker quantum-ai
```

**Step 2: Application Setup**
```bash
# Switch to quantum-ai user
sudo su - quantum-ai

# Clone and configure
git clone https://github.com/your-org/quantum-ai-platform.git
cd quantum-ai-platform

# Configure environment
cp .env.example .env
# Edit .env with production settings

# Set up SSL certificates (see SSL setup above)
```

**Step 3: Database Initialization**
```bash
# Start databases
docker-compose up -d postgres mongo redis qdrant chromadb

# Wait for initialization
sleep 30

# Run database migrations
docker-compose exec postgres psql -U quantumai -d quantumai_db -f /docker-entrypoint-initdb.d/postgres-init.sql
docker-compose exec mongo mongosh quantumai_mongo /docker-entrypoint-initdb.d/mongo-init.js
```

**Step 4: Application Deployment**
```bash
# Build and start all services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Verify deployment
curl https://your-domain.com/health
```

**Step 5: Post-Deployment Configuration**
```bash
# Create admin user
docker-compose exec quantum-ai-platform python -c "
from backend.security.auth_manager import AuthenticationManager
import asyncio

async def create_admin():
    auth = AuthenticationManager()
    user_id = await auth.register_user(
        'admin',
        'admin@your-domain.com', 
        'secure_admin_password',
        ['system_admin']
    )
    print(f'Admin user created with ID: {user_id}')

asyncio.run(create_admin())
"

# Set up log rotation
sudo tee /etc/logrotate.d/quantum-ai << EOF
/home/quantum-ai/quantum-ai-platform/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 quantum-ai quantum-ai
    postrotate
        docker-compose -f /home/quantum-ai/quantum-ai-platform/docker-compose.yml restart quantum-ai-platform
    endscript
}
EOF
```

## Cloud Deployment

### AWS Deployment

**Prerequisites:**
- AWS CLI configured
- ECR repository created
- ECS cluster or EKS cluster set up
- RDS PostgreSQL instance
- DocumentDB (MongoDB-compatible)
- ElastiCache Redis

**Deployment Steps:**

```bash
# Build and push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com

docker build -t quantum-ai-platform .
docker tag quantum-ai-platform:latest <account-id>.dkr.ecr.us-west-2.amazonaws.com/quantum-ai-platform:latest
docker push <account-id>.dkr.ecr.us-west-2.amazonaws.com/quantum-ai-platform:latest

# Deploy using ECS task definition
aws ecs update-service --cluster quantum-ai --service quantum-ai-platform --task-definition quantum-ai-platform:latest
```

**Terraform Configuration:**
```hcl
# terraform/main.tf
resource "aws_ecs_service" "quantum_ai" {
  name            = "quantum-ai-platform"
  cluster         = aws_ecs_cluster.quantum_ai.id
  task_definition = aws_ecs_task_definition.quantum_ai.arn
  desired_count   = 2

  load_balancer {
    target_group_arn = aws_lb_target_group.quantum_ai.arn
    container_name   = "quantum-ai-platform"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.quantum_ai]
}
```

### Google Cloud Platform (GCP)

```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT_ID/quantum-ai-platform

gcloud run deploy quantum-ai-platform \
  --image gcr.io/PROJECT_ID/quantum-ai-platform \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2
```

### Azure Deployment

```bash
# Create container registry
az acr create --resource-group quantum-ai-rg --name quantumaiacr --sku Basic

# Build and push
az acr build --registry quantumaiacr --image quantum-ai-platform .

# Deploy to Container Instances
az container create \
  --resource-group quantum-ai-rg \
  --name quantum-ai-platform \
  --image quantumaiacr.azurecr.io/quantum-ai-platform:latest \
  --cpu 2 \
  --memory 8 \
  --registry-login-server quantumaiacr.azurecr.io \
  --ports 8000
```

## Scaling and High Availability

### Horizontal Scaling

**Docker Swarm:**
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml -c docker-compose.swarm.yml quantum-ai

# Scale services
docker service scale quantum-ai_quantum-ai-platform=3
```

**Kubernetes Deployment:**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum-ai-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quantum-ai-platform
  template:
    metadata:
      labels:
        app: quantum-ai-platform
    spec:
      containers:
      - name: quantum-ai-platform
        image: quantum-ai-platform:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: quantum-ai-service
spec:
  selector:
    app: quantum-ai-platform
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Database High Availability

**PostgreSQL High Availability:**
```yaml
# docker-compose.ha.yml
version: '3.8'
services:
  postgres-primary:
    image: postgres:15-alpine
    environment:
      POSTGRES_REPLICATION_MODE: master
      POSTGRES_REPLICATION_USER: replica_user
      POSTGRES_REPLICATION_PASSWORD: replica_password
    
  postgres-replica:
    image: postgres:15-alpine
    environment:
      POSTGRES_REPLICATION_MODE: slave
      POSTGRES_REPLICATION_USER: replica_user
      POSTGRES_REPLICATION_PASSWORD: replica_password
      POSTGRES_MASTER_SERVICE: postgres-primary
```

**MongoDB Replica Set:**
```yaml
services:
  mongo-primary:
    image: mongo:7
    command: mongod --replSet rs0
    
  mongo-secondary:
    image: mongo:7
    command: mongod --replSet rs0
    
  mongo-arbiter:
    image: mongo:7
    command: mongod --replSet rs0
```

### Load Balancing

**Nginx Configuration:**
```nginx
# nginx/load-balancer.conf
upstream quantum_ai_backend {
    server quantum-ai-platform-1:8000;
    server quantum-ai-platform-2:8000;
    server quantum-ai-platform-3:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://quantum_ai_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Security Configuration

### SSL/TLS Configuration

**Nginx SSL Configuration:**
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/ssl/certs/quantum-ai.crt;
    ssl_certificate_key /etc/ssl/private/quantum-ai.key;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
}
```

### Database Security

**PostgreSQL Security:**
```sql
-- Create restricted user for application
CREATE USER quantum_app WITH ENCRYPTED PASSWORD 'secure_app_password';
GRANT CONNECT ON DATABASE quantumai_db TO quantum_app;
GRANT USAGE ON SCHEMA public TO quantum_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO quantum_app;

-- Enable row-level security
ALTER TABLE quantum_circuits ENABLE ROW LEVEL SECURITY;
CREATE POLICY quantum_circuit_isolation ON quantum_circuits
    FOR ALL TO quantum_app
    USING (user_id = current_user_id());
```

### Network Security

**Docker Network Isolation:**
```yaml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true
  database:
    driver: bridge
    internal: true

services:
  quantum-ai-platform:
    networks:
      - frontend
      - backend
  
  postgres:
    networks:
      - database
```

## Monitoring and Logging

### Prometheus Configuration

**prometheus.yml:**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'quantum-ai-platform'
    static_configs:
      - targets: ['quantum-ai-platform:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

rule_files:
  - "quantum_ai_alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards

**Dashboard Configuration:**
```json
{
  "dashboard": {
    "title": "Quantum AI Platform Monitoring",
    "panels": [
      {
        "title": "Quantum Circuit Executions",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(quantum_circuit_executions_total[5m])"
          }
        ]
      },
      {
        "title": "System Resource Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(container_cpu_usage_seconds_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### Centralized Logging

**ELK Stack Configuration:**
```yaml
# docker-compose.logging.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    
  logstash:
    image: docker.elastic.co/logstash/logstash:8.5.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
    
  kibana:
    image: docker.elastic.co/kibana/kibana:8.5.0
    ports:
      - "5601:5601"
```

## Backup and Recovery

### Database Backups

**Automated Backup Script:**
```bash
#!/bin/bash
# scripts/backup-databases.sh

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# PostgreSQL backup
docker-compose exec postgres pg_dump -U quantumai quantumai_db > "$BACKUP_DIR/postgres_backup.sql"

# MongoDB backup
docker-compose exec mongo mongodump --db quantumai_mongo --archive > "$BACKUP_DIR/mongo_backup.archive"

# Redis backup
docker-compose exec redis redis-cli BGSAVE
docker cp $(docker-compose ps -q redis):/data/dump.rdb "$BACKUP_DIR/redis_backup.rdb"

# Compress backups
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

**Backup Scheduling:**
```bash
# Add to crontab
0 2 * * * /home/quantum-ai/quantum-ai-platform/scripts/backup-databases.sh
```

### Disaster Recovery

**Recovery Procedure:**
```bash
#!/bin/bash
# scripts/restore-databases.sh

BACKUP_FILE="$1"
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

# Extract backup
RESTORE_DIR=$(mktemp -d)
tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR"
BACKUP_DIR=$(find "$RESTORE_DIR" -type d -name "*_*" | head -n 1)

# Stop services
docker-compose down

# Restore PostgreSQL
docker-compose up -d postgres
sleep 10
docker-compose exec postgres psql -U quantumai -d quantumai_db < "$BACKUP_DIR/postgres_backup.sql"

# Restore MongoDB
docker-compose up -d mongo
sleep 10
docker-compose exec mongo mongorestore --db quantumai_mongo --archive < "$BACKUP_DIR/mongo_backup.archive"

# Restore Redis
docker-compose up -d redis
sleep 5
docker cp "$BACKUP_DIR/redis_backup.rdb" $(docker-compose ps -q redis):/data/dump.rdb
docker-compose restart redis

# Start all services
docker-compose up -d

echo "Database restore completed"
rm -rf "$RESTORE_DIR"
```

## Troubleshooting

### Common Issues

**1. Database Connection Issues**
```bash
# Check database status
docker-compose ps postgres mongo redis

# Check logs
docker-compose logs postgres
docker-compose logs mongo
docker-compose logs redis

# Test connections
docker-compose exec postgres pg_isready -U quantumai
docker-compose exec mongo mongosh --eval "db.adminCommand('ping')"
docker-compose exec redis redis-cli ping
```

**2. Memory Issues**
```bash
# Check memory usage
docker stats

# Increase memory limits in docker-compose.yml
services:
  quantum-ai-platform:
    deploy:
      resources:
        limits:
          memory: 8G
```

**3. SSL Certificate Issues**
```bash
# Check certificate validity
openssl x509 -in ssl/quantum-ai.crt -text -noout

# Renew Let's Encrypt certificate
sudo certbot renew --dry-run
```

**4. Performance Issues**
```bash
# Check system resources
htop
iotop
nethogs

# Analyze quantum simulation performance
docker-compose exec quantum-ai-platform python -c "
from backend.quantum.vqc_engine import VirtualQuantumComputer
import asyncio
import time

async def benchmark():
    vqc = VirtualQuantumComputer(num_qubits=8)
    start = time.time()
    circuit = await vqc.create_quantum_circuit('basic', [0.5])
    result = await vqc.execute_circuit(circuit, shots=1000)
    end = time.time()
    print(f'Execution time: {end - start:.2f}s')

asyncio.run(benchmark())
"
```

### Log Analysis

**Key Log Locations:**
- Application logs: `./logs/quantum_ai.log`
- Access logs: `./logs/access.log`
- Error logs: `./logs/error.log`
- Container logs: `docker-compose logs [service]`

**Log Analysis Commands:**
```bash
# Monitor real-time logs
tail -f logs/quantum_ai.log

# Search for errors
grep -i error logs/quantum_ai.log

# Analyze performance
grep "execution_time" logs/quantum_ai.log | awk '{print $NF}' | sort -n
```

### Performance Optimization

**Database Optimization:**
```sql
-- PostgreSQL optimization
ANALYZE;
REINDEX;

-- Add appropriate indexes
CREATE INDEX idx_quantum_circuits_user_id ON quantum_circuits(user_id);
CREATE INDEX idx_quantum_executions_status ON quantum_executions(status);
```

**Container Optimization:**
```yaml
services:
  quantum-ai-platform:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
    ulimits:
      memlock:
        soft: -1
        hard: -1
```

### Health Checks

**Application Health:**
```bash
# Health check endpoint
curl -f http://localhost:8000/health || exit 1

# Component-specific health checks
curl -f http://localhost:8000/api/quantum/status
curl -f http://localhost:8000/api/time-crystals/status
curl -f http://localhost:8000/api/neuromorphic/status
curl -f http://localhost:8000/api/iot/status
```

This deployment guide provides comprehensive instructions for deploying the Quantum AI Platform in various environments, from development to production, with proper security, monitoring, and maintenance procedures.