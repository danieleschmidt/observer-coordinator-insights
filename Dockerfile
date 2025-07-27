# Multi-stage build for optimized production image
FROM python:3.11-slim as base

# Metadata labels
LABEL org.opencontainers.image.source="https://github.com/terragon-labs/observer-coordinator-insights"
LABEL org.opencontainers.image.description="Multi-agent orchestration for organizational analytics"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.vendor="Terragon Labs"
LABEL org.opencontainers.image.title="Observer Coordinator Insights"
LABEL org.opencontainers.image.documentation="https://github.com/terragon-labs/observer-coordinator-insights#readme"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    APP_ENV=production

# Install system dependencies and security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user with proper permissions
RUN groupadd -r insights --gid=1000 && \
    useradd -r -g insights --uid=1000 --home-dir=/app --shell=/bin/bash insights

# Build stage
FROM base as build

WORKDIR /app

# Copy dependency files first for better Docker layer caching
COPY requirements.txt pyproject.toml ./

# Install build dependencies and Python packages
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir build

# Copy source code
COPY src/ ./src/
COPY *.py ./
COPY *.yml ./
COPY *.md ./
COPY LICENSE ./

# Install application in development mode for build stage
RUN pip install -e .

# Production stage
FROM base as production

WORKDIR /app

# Copy only necessary files from build stage
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /app/src ./src/
COPY --from=build /app/*.py ./
COPY --from=build /app/pyproject.toml ./
COPY --from=build /app/LICENSE ./

# Create application directories with proper permissions
RUN mkdir -p /app/data /app/output /app/logs /app/config && \
    chown -R insights:insights /app && \
    chmod -R 755 /app

# Copy configuration files
COPY --chown=insights:insights .env.example /app/config/.env.example

# Switch to non-root user
USER insights

# Add health check script
COPY --chown=insights:insights <<EOF /app/healthcheck.py
#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/app/src')

try:
    from src.main import main
    print("Health check passed")
    sys.exit(0)
except Exception as e:
    print(f"Health check failed: {e}")
    sys.exit(1)
EOF

RUN chmod +x /app/healthcheck.py

# Health check with improved validation
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /app/healthcheck.py || exit 1

# Expose port for metrics/monitoring (if applicable)
EXPOSE 8080

# Security: Run as non-root, read-only filesystem options
USER insights

# Default command with proper error handling
CMD ["python", "-m", "src.main"]

# Development stage
FROM build as development

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov ruff mypy bandit pre-commit

# Copy test files and development configurations
COPY tests/ ./tests/
COPY .pre-commit-config.yaml .editorconfig ./

# Set development environment
ENV APP_ENV=development
ENV DEBUG=true

# Switch to non-root user
USER insights

# Default command for development
CMD ["python", "-m", "src.main", "--help"]