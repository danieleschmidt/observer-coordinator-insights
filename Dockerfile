FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/terragon-labs/observer-coordinator-insights"
LABEL org.opencontainers.image.description="Multi-agent orchestration for organizational analytics"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd -r insights && useradd -r -g insights insights

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY pyproject.toml ./

# Install application
RUN pip install -e .

# Create directories for data and output
RUN mkdir -p /app/data /app/output && \
    chown -R insights:insights /app

# Switch to non-root user
USER insights

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "src.main", "--help"]