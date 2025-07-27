# Multi-stage Docker build for Observer Coordinator Insights
# Stage 1: Build dependencies and compile any extensions
FROM python:3.11-slim as builder

LABEL org.opencontainers.image.source="https://github.com/terragon-labs/observer-coordinator-insights"
LABEL org.opencontainers.image.description="Multi-agent orchestration for organizational analytics"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.version="0.1.0"
LABEL maintainer="Terragon Labs <contact@terragon-labs.com>"

# Set build arguments
ARG BUILD_ENV=production
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-dev.txt pyproject.toml ./
RUN if [ "$BUILD_ENV" = "development" ]; then \
        pip install -r requirements-dev.txt; \
    else \
        pip install -r requirements.txt; \
    fi

# Copy source code and install the package
COPY . .
RUN pip install -e .

# Stage 2: Runtime image
FROM python:3.11-slim as runtime

# Set runtime arguments
ARG BUILD_ENV=production
ARG USER_ID=1000
ARG GROUP_ID=1000

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create application user and group
RUN groupadd -g ${GROUP_ID} insights && \
    useradd -m -u ${USER_ID} -g insights -s /bin/bash insights

# Set up application directory
WORKDIR /app

# Copy application code
COPY --chown=insights:insights . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/output /app/cache && \
    chown -R insights:insights /app

# Create entrypoint script
RUN echo '#!/bin/bash\nset -e\nexec "$@"' > /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_ENV=${BUILD_ENV}
ENV USER=insights

# Security: Run as non-root user
USER insights

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import src; print('OK')" || exit 1

# Expose ports
EXPOSE 8000 8080 9090

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]

# Default command
CMD ["python", "-m", "src.main"]

# Development stage
FROM runtime as development

USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    less \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
COPY --from=builder /opt/venv /opt/venv
RUN pip install ipython jupyter pytest-cov

# Switch back to app user
USER insights

# Override command for development
CMD ["python", "-m", "src.main", "--debug"]