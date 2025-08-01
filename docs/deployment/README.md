# Deployment Guide

This guide covers deployment strategies and procedures for the Observer Coordinator Insights application.

## Overview

The application supports multiple deployment methods:

- **Docker Containers**: Recommended for production deployments
- **Docker Compose**: For local development and small-scale deployments
- **Kubernetes**: For large-scale, cloud-native deployments
- **Native Python**: Direct deployment on servers (not recommended for production)

## Docker Deployment

### Single Container

Build and run the application as a single container:

```bash
# Build the image
make docker

# Run the container
docker run -d \
  --name insights-app \
  -p 8000:8000 \
  -p 8080:8080 \
  -p 9090:9090 \
  -e APP_ENV=production \
  -e LOG_LEVEL=INFO \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/output:/app/output \
  observer-coordinator-insights:latest
```

### Multi-stage Builds

The Dockerfile uses multi-stage builds for optimization:

- **Builder Stage**: Installs dependencies and compiles extensions
- **Runtime Stage**: Minimal runtime environment (production)
- **Development Stage**: Includes development tools and debugging capabilities

```bash
# Build production image (default)
docker build -t insights:prod .

# Build development image
docker build --target development -t insights:dev .

# Build with custom build args
docker build \
  --build-arg BUILD_ENV=production \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  -t insights:custom .
```

## Docker Compose Deployment

### Development Environment

Start the full development stack:

```bash
# Start core services
docker-compose up -d

# Start with development profile
docker-compose --profile dev up -d

# View logs
docker-compose logs -f app
```

### Production Environment

```bash
# Start with production profile
docker-compose --profile production up -d

# Scale the application
docker-compose up -d --scale app=3

# Update configuration
docker-compose restart app
```

### Service Profiles

The docker-compose.yml includes several profiles for different use cases:

- **Default**: Core application with Redis and Prometheus
- **dev**: Development tools including Jupyter
- **production**: Production setup with Nginx reverse proxy
- **testing**: Load testing with Locust
- **database**: PostgreSQL database for persistent storage
- **storage**: MinIO object storage

```bash
# Start specific profiles
docker-compose --profile dev --profile database up -d
```

## Environment Configuration

### Required Environment Variables

```bash
# Application Settings
APP_ENV=production                    # Environment: development, staging, production
DEBUG=false                          # Debug mode
LOG_LEVEL=INFO                       # Logging level
SECRET_KEY=your-secret-key-here      # Application security key

# Data Configuration
DATA_RETENTION_DAYS=180              # GDPR compliance
ENABLE_ANONYMIZATION=true            # Data anonymization
VALIDATION_LEVEL=strict              # Data validation

# Clustering Configuration
CLUSTERING_ALGORITHM=kmeans          # Algorithm choice
DEFAULT_CLUSTERS=0                   # Auto-detect clusters
RANDOM_SEED=42                       # Reproducible results

# Security Settings
HTTPS_ONLY=true                      # Enforce HTTPS
CORS_ORIGINS=https://yourdomain.com  # CORS configuration
RATE_LIMIT=100                       # Rate limiting

# Monitoring
ENABLE_METRICS=true                  # Prometheus metrics
METRICS_PORT=9090                    # Metrics endpoint
```

### Optional Environment Variables

```bash
# Database (if using persistent storage)
DATABASE_URL=postgresql://user:pass@host:5432/db

# External Integrations
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=notifications@example.com
SMTP_PASSWORD=password

# Performance Tuning
MAX_WORKERS=4                        # Worker processes
WORKER_TIMEOUT=30                    # Request timeout
MEMORY_LIMIT=512MB                   # Memory limit per worker
```

## Health Checks and Monitoring

### Health Check Endpoints

The application provides several health check endpoints:

```bash
# Basic health check
curl http://localhost:8080/health

# Detailed health status
curl http://localhost:8080/health/detailed

# Readiness probe (for Kubernetes)
curl http://localhost:8080/ready

# Liveness probe (for Kubernetes)
curl http://localhost:8080/alive
```

### Metrics Collection

Prometheus metrics are available at:

```bash
# Application metrics
curl http://localhost:9090/metrics

# Custom business metrics
curl http://localhost:9090/metrics | grep insights_
```

### Log Management

```bash
# View application logs
docker-compose logs -f app

# View logs with timestamps
docker-compose logs -f -t app

# Export logs
docker-compose logs --no-color app > app.log
```

## Security Considerations

### Container Security

- **Non-root User**: Application runs as unprivileged user
- **Read-only Filesystem**: Most directories are read-only
- **Minimal Base Image**: Uses slim Python base image
- **Security Scanning**: Regular vulnerability scans
- **Secret Management**: Environment-based secret injection

### Network Security

```bash
# Custom network configuration
docker network create insights-secure \
  --driver bridge \
  --opt encrypted=true

# Run with custom network
docker run -d \
  --network insights-secure \
  --name insights-app \
  observer-coordinator-insights:latest
```

### SSL/TLS Configuration

For production deployments, configure SSL/TLS:

```bash
# Generate self-signed certificates (development only)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Production: Use certificates from CA
# Copy certificates to nginx/ssl/ directory
cp your-domain.crt nginx/ssl/
cp your-domain.key nginx/ssl/
```

## Scaling and Performance

### Horizontal Scaling

```bash
# Scale with Docker Compose
docker-compose up -d --scale app=5

# Scale with Docker Swarm
docker service scale insights_app=5

# Load balancer configuration in nginx.conf
upstream insights_backend {
    server app1:8000;
    server app2:8000; 
    server app3:8000;
}
```

### Performance Tuning

#### Container Resources

```yaml
# docker-compose.yml resource limits
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

#### Application Configuration

```bash
# Environment variables for performance
WORKERS=4                            # Number of worker processes
WORKER_CONNECTIONS=1000              # Connections per worker
WORKER_TIMEOUT=30                    # Request timeout
MAX_REQUESTS=1000                    # Requests before worker restart
PRELOAD_APP=true                     # Preload application code
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'insights-app'
    static_configs:
      - targets: ['app:9090']
    scrape_interval: 10s
    metrics_path: /metrics
```

### Grafana Dashboard

Import the pre-configured dashboard:

```bash
# Access Grafana
open http://localhost:3000

# Login: admin/admin
# Import dashboard from monitoring/grafana/dashboards/
```

### Alerting Rules

```yaml
# monitoring/alerts/application-alerts.yml
groups:
  - name: insights.rules
    rules:
      - alert: HighMemoryUsage
        expr: insights_memory_usage_bytes > 1073741824
        for: 5m
        annotations:
          summary: "High memory usage detected"
          
      - alert: SlowResponse
        expr: insights_response_time_seconds > 5
        for: 2m
        annotations:
          summary: "Slow response time detected"
```

## Backup and Recovery

### Data Backup

```bash
# Backup persistent volumes
docker run --rm \
  -v insights_postgres-data:/source:ro \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/postgres-$(date +%Y%m%d).tar.gz -C /source .

# Backup configuration
tar czf config-backup-$(date +%Y%m%d).tar.gz \
  .env docker-compose.yml monitoring/ nginx/
```

### Database Backup (if using PostgreSQL)

```bash
# Database dump
docker-compose exec postgres pg_dump -U insights insights > backup.sql

# Restore database
docker-compose exec -T postgres psql -U insights insights < backup.sql
```

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker-compose logs app

# Common fixes
docker-compose down
docker-compose pull
docker-compose up -d
```

#### Permission Issues

```bash
# Fix file permissions
sudo chown -R $(id -u):$(id -g) data/ output/ logs/

# Set proper permissions
chmod 755 data/
chmod 775 output/ logs/
```

#### Memory Issues

```bash
# Check memory usage
docker stats

# Increase memory limits
# Edit docker-compose.yml memory limits
```

#### Network Issues

```bash
# Check container connectivity
docker-compose exec app ping redis
docker-compose exec app ping prometheus

# Restart networking
docker-compose down
docker network prune
docker-compose up -d
```

### Debug Mode

Enable debug mode for troubleshooting:

```bash
# Debug environment variables
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with debug
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up -d
```

### Performance Debugging

```bash
# Profile application
docker-compose exec app python -m cProfile -o profile.stats src/main.py

# Memory profiling
docker-compose exec app python -m memory_profiler src/main.py

# Load testing
docker-compose --profile testing up -d locust
open http://localhost:8089
```

## Disaster Recovery

### Recovery Procedures

1. **Data Recovery**
   ```bash
   # Restore from backup
   docker-compose down
   tar xzf postgres-backup.tar.gz -C /var/lib/docker/volumes/insights_postgres-data/_data/
   docker-compose up -d
   ```

2. **Configuration Recovery**
   ```bash
   # Restore configuration
   tar xzf config-backup.tar.gz
   docker-compose up -d
   ```

3. **Full System Recovery**
   ```bash
   # Clone repository
   git clone https://github.com/your-org/observer-coordinator-insights.git
   cd observer-coordinator-insights
   
   # Restore configuration and data
   tar xzf config-backup.tar.gz
   tar xzf data-backup.tar.gz
   
   # Start services
   docker-compose up -d
   ```

## Production Checklist

Before deploying to production:

- [ ] SSL/TLS certificates configured
- [ ] Environment variables set (no default values)
- [ ] Resource limits configured
- [ ] Monitoring and alerting set up
- [ ] Backup procedures tested
- [ ] Security scan completed
- [ ] Performance testing completed
- [ ] Documentation updated
- [ ] Incident response plan prepared
- [ ] Team trained on deployment procedures

## Support and Maintenance

### Regular Maintenance

```bash
# Weekly tasks
make clean                           # Clean temporary files
docker system prune                  # Clean Docker resources
docker-compose logs --tail=1000     # Review logs

# Monthly tasks  
make security                        # Security scan
make test                           # Full test suite
docker-compose pull                 # Update base images
```

### Updates and Upgrades

```bash
# Application updates
git pull origin main
make build
docker-compose up -d

# Dependency updates
make check-deps
# Review and update requirements.txt
make install-dev
make test
```

For additional support, see:
- [Architecture Documentation](../ARCHITECTURE.md)
- [Monitoring Guide](./MONITORING.md)
- [Security Guide](./SECURITY.md)
- [Development Guide](./DEVELOPMENT.md)