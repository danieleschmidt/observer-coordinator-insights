# Multi-Architecture Dockerfile for Global-First Neuromorphic Clustering System
# Supports AMD64, ARM64 for global cloud deployment

ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG TARGETARCH

FROM --platform=$BUILDPLATFORM python:3.11-slim as base

# Build arguments for multi-arch support
ARG TARGETARCH
ARG BUILDPLATFORM
ARG TARGETPLATFORM

# Set labels for image metadata
LABEL org.opencontainers.image.title="Neuromorphic Clustering System"
LABEL org.opencontainers.image.description="Global-first neuromorphic clustering with i18n and compliance"
LABEL org.opencontainers.image.vendor="Your Organization"
LABEL org.opencontainers.image.version="4.0-global"
LABEL org.opencontainers.image.source="https://github.com/your-org/neuromorphic-clustering"
LABEL org.opencontainers.image.documentation="https://docs.your-org.com/neuromorphic-clustering"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Create non-root user for security
RUN groupadd -r neuromorphic && useradd -r -g neuromorphic -s /bin/false neuromorphic

# Install system dependencies optimized for each architecture
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    curl \
    ca-certificates \
    locales \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure locales for i18n support
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && \
    echo "es_ES.UTF-8 UTF-8" >> /etc/locale.gen && \
    echo "fr_FR.UTF-8 UTF-8" >> /etc/locale.gen && \
    echo "de_DE.UTF-8 UTF-8" >> /etc/locale.gen && \
    echo "ja_JP.UTF-8 UTF-8" >> /etc/locale.gen && \
    echo "zh_CN.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Architecture-specific optimizations
FROM base as amd64
# AMD64 specific optimizations
ENV ARCH_SPECIFIC_FLAGS="--platform-tags=avx2,sse4" \
    CPU_OPTIMIZATION="native"

FROM base as arm64  
# ARM64 specific optimizations
ENV ARCH_SPECIFIC_FLAGS="--platform-tags=neon,armv8" \
    CPU_OPTIMIZATION="armv8-a"

# Select architecture-specific stage
FROM ${TARGETARCH} as platform

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies with architecture-specific optimizations
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt && \
    pip install --no-cache-dir gunicorn uvicorn[standard] && \
    pip cache purge

# Copy application source code
COPY src/ ./src/
COPY locales/ ./locales/
COPY *.py ./
COPY alembic.ini ./
COPY migrations/ ./migrations/

# Copy configuration and documentation
COPY docs/global-deployment/ ./docs/global-deployment/

# Create necessary directories
RUN mkdir -p logs cache data && \
    chown -R neuromorphic:neuromorphic /app

# Health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:8000/api/health || exit 1' > /healthcheck.sh && \
    chmod +x /healthcheck.sh

# Security: Switch to non-root user
USER neuromorphic

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD /healthcheck.sh

# Default command
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]