# Atherium - Advanced Quantum AI Platform
# Multi-stage Dockerfile for production deployment

# Base Python image
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ATHERIUM_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    nodejs \
    npm \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create atherium user
RUN useradd --create-home --shell /bin/bash atherium

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY platform/requirements.txt /app/platform/
RUN pip install --no-cache-dir -r platform/requirements.txt

# Frontend build stage
FROM node:18-alpine as frontend-builder

WORKDIR /app

# Copy frontend package files
COPY platform/frontend/package*.json /app/platform/frontend/
COPY ai-systems/frontend/package*.json /app/ai-systems/frontend/ 2>/dev/null || true

# Install frontend dependencies
RUN cd platform/frontend && npm ci --only=production

# Copy frontend source and build
COPY platform/frontend/ /app/platform/frontend/
RUN cd platform/frontend && npm run build

# Production stage
FROM base as production

# Copy application code
COPY --chown=atherium:atherium . /app/

# Copy built frontend
COPY --from=frontend-builder --chown=atherium:atherium /app/platform/frontend/build /app/platform/frontend/build

# Switch to atherium user
USER atherium

# Create necessary directories
RUN mkdir -p /app/logs /app/data

# Expose ports
EXPOSE 8000 3000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "platform/backend/main.py"]
