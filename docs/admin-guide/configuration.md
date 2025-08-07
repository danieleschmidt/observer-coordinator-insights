# Configuration Management Guide

This guide covers comprehensive configuration management for Observer Coordinator Insights, including environment-specific settings, security configurations, performance tuning, and best practices for maintaining configurations across different deployment environments.

## Table of Contents

1. [Configuration Architecture](#configuration-architecture)
2. [Environment Configuration](#environment-configuration)
3. [Database Configuration](#database-configuration)
4. [Security Configuration](#security-configuration)
5. [Performance Configuration](#performance-configuration)
6. [Clustering Algorithm Configuration](#clustering-algorithm-configuration)
7. [API Configuration](#api-configuration)
8. [Logging Configuration](#logging-configuration)
9. [Monitoring Configuration](#monitoring-configuration)
10. [Configuration Management Best Practices](#configuration-management-best-practices)
11. [Configuration Validation](#configuration-validation)

## Configuration Architecture

### Configuration Hierarchy

Observer Coordinator Insights uses a hierarchical configuration system:

```
1. Default Configuration (built-in)
2. Environment Configuration Files
3. Environment Variables
4. Runtime Parameters
5. API Configuration Updates
```

### Configuration File Structure

```
config/
├── default.yml              # Default settings
├── development.yml          # Development overrides
├── testing.yml             # Testing environment
├── staging.yml             # Staging environment  
├── production.yml          # Production settings
├── secrets.yml             # Encrypted secrets (not in VCS)
└── local.yml               # Local developer overrides
```

### Configuration Loading

```python
# Configuration loading priority (highest to lowest):
1. Command line arguments: --config-override key=value
2. Environment variables: INSIGHTS_CONFIG_KEY=value
3. Environment config file: config/{environment}.yml
4. Default config file: config/default.yml
5. Built-in defaults
```

## Environment Configuration

### Development Configuration

Create `config/development.yml`:

```yaml
# Development Environment Configuration
environment: development
debug: true

# Database - SQLite for simplicity
database:
  url: "sqlite:///dev_insights.db"
  echo: true  # Log SQL queries
  pool_size: 5
  pool_timeout: 30

# Caching - Redis optional in development
redis:
  enabled: false
  url: "redis://localhost:6379/0"

# Clustering - Faster algorithms for development
clustering:
  default_method: "esn"  # Echo State Network - fastest
  default_clusters: 4
  timeout_seconds: 300
  max_concurrent_jobs: 2
  enable_gpu: false

# Security - Relaxed for development
security:
  require_auth: false
  secure_mode: false
  audit_logging: false
  cors_origins: ["http://localhost:3000", "http://127.0.0.1:3000"]
  rate_limiting: false

# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  reload: true
  access_log: true

# Logging
logging:
  level: "DEBUG"
  format: "detailed"
  console: true
  file: false
  
# File Storage
storage:
  type: "local"
  path: "/tmp/insights_dev"
  max_file_size: "10MB"
```

### Staging Configuration

Create `config/staging.yml`:

```yaml
# Staging Environment Configuration
environment: staging
debug: false

# Database - Production-like PostgreSQL
database:
  url: "postgresql://insights_user:${DB_PASSWORD}@staging-db:5432/insights_staging"
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
  echo: false

# Redis for caching and session management
redis:
  enabled: true
  url: "redis://staging-redis:6379/0"
  max_connections: 50
  socket_timeout: 30

# Clustering - Production algorithms with smaller timeouts
clustering:
  default_method: "hybrid_reservoir"
  default_clusters: 4
  timeout_seconds: 900  # 15 minutes
  max_concurrent_jobs: 5
  enable_optimization: true
  
# Security - Production-like with staging-specific settings
security:
  require_auth: true
  secure_mode: true
  audit_logging: true
  cors_origins: ["https://staging-insights.company.com"]
  rate_limiting: true
  max_requests_per_hour: 500

# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  workers: 2
  worker_class: "gevent"
  reload: false
  access_log: true

# Logging
logging:
  level: "INFO"
  format: "json"
  console: false
  file: true
  file_path: "/var/log/insights/staging.log"
  max_size: "100MB"
  backup_count: 3
  
# Monitoring
monitoring:
  prometheus_enabled: true
  metrics_port: 9090
  health_check_interval: 60

# File Storage
storage:
  type: "s3"
  bucket: "company-insights-staging"
  region: "us-west-2"
  max_file_size: "50MB"
```

### Production Configuration

Create `config/production.yml`:

```yaml
# Production Environment Configuration
environment: production
debug: false

# Database - High-performance PostgreSQL cluster
database:
  # Primary database for writes
  url: "postgresql://insights_user:${DB_PASSWORD}@prod-db-primary:5432/insights_prod"
  # Read replicas for load distribution
  read_urls:
    - "postgresql://insights_user:${DB_PASSWORD}@prod-db-replica-1:5432/insights_prod"
    - "postgresql://insights_user:${DB_PASSWORD}@prod-db-replica-2:5432/insights_prod"
  pool_size: 20
  max_overflow: 40
  pool_timeout: 30
  pool_recycle: 3600
  echo: false

# Redis cluster for high availability
redis:
  enabled: true
  cluster_mode: true
  nodes:
    - "redis-1.company.com:6379"
    - "redis-2.company.com:6379"
    - "redis-3.company.com:6379"
  password: "${REDIS_PASSWORD}"
  max_connections: 100
  socket_timeout: 30
  health_check_interval: 30

# Clustering - Full production algorithms
clustering:
  default_method: "hybrid_reservoir"
  default_clusters: 4
  timeout_seconds: 1800  # 30 minutes
  max_concurrent_jobs: 20
  enable_optimization: true
  enable_gpu: true
  batch_processing: true
  
  # Algorithm-specific tuning
  esn_params:
    reservoir_size: 200
    spectral_radius: 0.95
    sparsity: 0.1
    
  snn_params:
    n_neurons: 100
    threshold: 1.0
    learning_rate: 0.01
    
  lsm_params:
    liquid_size: 128
    connection_prob: 0.3

# Security - Maximum security for production
security:
  require_auth: true
  secure_mode: true
  audit_logging: true
  encryption_at_rest: true
  encryption_in_transit: true
  
  # JWT Configuration
  jwt_secret_file: "/opt/insights/secrets/jwt_secret.key"
  jwt_expiration_hours: 24
  jwt_refresh_enabled: true
  
  # API Security
  cors_origins: ["https://insights.company.com"]
  rate_limiting: true
  max_requests_per_hour: 10000
  
  # Data Protection
  pii_anonymization: true
  data_retention_days: 180
  gdpr_compliance: true
  ccpa_compliance: true
  
# API Configuration - Production grade
api:
  host: "0.0.0.0"
  port: 8000
  workers: 8
  worker_class: "gevent"
  worker_connections: 1000
  max_requests: 1000
  max_requests_jitter: 50
  timeout: 300
  keepalive: 30
  preload: true
  reload: false
  access_log: true

# Logging - Structured logging for production
logging:
  level: "INFO"
  format: "json"
  console: false
  file: true
  file_path: "/var/log/insights/production.log"
  max_size: "500MB"
  backup_count: 10
  
  # Separate audit logging
  audit:
    enabled: true
    file_path: "/var/log/insights/audit.log"
    max_size: "100MB"
    backup_count: 50  # Long retention for compliance
    
# Monitoring - Comprehensive monitoring
monitoring:
  prometheus_enabled: true
  metrics_port: 9090
  health_check_interval: 30
  performance_monitoring: true
  error_tracking: true
  
  # Alerting thresholds
  alerts:
    cpu_threshold: 80
    memory_threshold: 85
    disk_threshold: 90
    error_rate_threshold: 0.05
    response_time_threshold: 5000  # 5 seconds

# File Storage - Enterprise cloud storage
storage:
  type: "s3"
  bucket: "company-insights-production"
  region: "us-east-1"
  max_file_size: "500MB"
  encryption: "AES256"
  versioning: true
  lifecycle_policy: "delete_after_180_days"

# Backup Configuration
backup:
  enabled: true
  schedule: "0 2 * * *"  # Daily at 2 AM
  retention_days: 90
  s3_bucket: "company-insights-backups"
  encryption: true
```

## Database Configuration

### PostgreSQL Optimization

#### Connection Pool Configuration

```yaml
database:
  # Connection pooling
  pool_size: 20                    # Number of persistent connections
  max_overflow: 40                 # Additional connections when needed
  pool_timeout: 30                 # Seconds to wait for connection
  pool_recycle: 3600               # Recycle connections every hour
  pool_pre_ping: true              # Validate connections before use
  
  # Query optimization
  statement_timeout: 300000        # 5 minutes max query time
  idle_in_transaction_timeout: 30000  # 30 seconds max idle transaction
  
  # Logging
  echo: false                      # Don't log SQL in production
  echo_pool: true                  # Log connection pool events
```

#### Read Replica Configuration

```yaml
database:
  # Primary database (writes)
  url: "postgresql://user:pass@primary:5432/db"
  
  # Read replicas (queries)
  read_urls:
    - "postgresql://user:pass@replica1:5432/db"
    - "postgresql://user:pass@replica2:5432/db"
    
  # Read/write splitting
  read_preference: "secondary_preferred"
  read_load_balancing: "round_robin"
```

### Redis Configuration

#### Single Instance

```yaml
redis:
  enabled: true
  url: "redis://localhost:6379/0"
  password: "${REDIS_PASSWORD}"
  max_connections: 100
  socket_timeout: 30
  socket_connect_timeout: 30
  retry_on_timeout: true
  health_check_interval: 30
```

#### Redis Cluster

```yaml
redis:
  enabled: true
  cluster_mode: true
  nodes:
    - "redis-1.example.com:6379"
    - "redis-2.example.com:6379" 
    - "redis-3.example.com:6379"
  password: "${REDIS_PASSWORD}"
  max_connections_per_node: 50
  readonly_mode: false
  skip_full_coverage_check: false
```

## Security Configuration

### Authentication & Authorization

```yaml
security:
  # Authentication
  require_auth: true
  auth_providers:
    - type: "jwt"
      secret_file: "/opt/insights/secrets/jwt_secret.key"
      expiration_hours: 24
      refresh_enabled: true
    - type: "oauth2"
      provider: "google"
      client_id: "${OAUTH_CLIENT_ID}"
      client_secret_file: "/opt/insights/secrets/oauth_secret.key"
    - type: "ldap"
      server: "ldap.company.com"
      base_dn: "ou=users,dc=company,dc=com"
      
  # Authorization
  rbac_enabled: true
  roles:
    admin:
      permissions: ["*"]
    analyst:
      permissions: ["analytics:read", "analytics:create", "teams:read"]
    viewer:
      permissions: ["analytics:read", "teams:read"]
```

### Data Protection

```yaml
security:
  # Encryption
  encryption_at_rest: true
  encryption_key_file: "/opt/insights/secrets/encryption.key"
  encryption_algorithm: "AES-256-GCM"
  
  # PII Protection
  pii_anonymization: true
  anonymization_salt_file: "/opt/insights/secrets/anon_salt.key"
  
  # Compliance
  gdpr_compliance: true
  ccpa_compliance: true
  pdpa_compliance: true
  data_retention_days: 180
  
  # Audit
  audit_logging: true
  audit_log_file: "/var/log/insights/audit.log"
  audit_log_retention_days: 2555  # 7 years
```

### Network Security

```yaml
security:
  # CORS
  cors_origins: ["https://insights.company.com"]
  cors_methods: ["GET", "POST", "PUT", "DELETE"]
  cors_headers: ["Content-Type", "Authorization"]
  
  # Rate Limiting
  rate_limiting: true
  rate_limit_per_hour: 1000
  burst_limit: 100
  
  # IP Filtering
  ip_whitelist:
    - "10.0.0.0/8"     # Internal network
    - "192.168.1.0/24" # Office network
  ip_blacklist:
    - "203.0.113.0/24" # Known bad actors
    
  # SSL/TLS
  ssl_required: true
  ssl_cert_file: "/opt/insights/ssl/insights.crt"
  ssl_key_file: "/opt/insights/ssl/insights.key"
  ssl_protocols: ["TLSv1.2", "TLSv1.3"]
```

## Performance Configuration

### Application Performance

```yaml
performance:
  # Worker Configuration
  workers: 8                    # Number of worker processes
  worker_class: "gevent"       # Async worker type
  worker_connections: 1000     # Connections per worker
  
  # Request Handling
  timeout: 300                 # Request timeout (5 minutes)
  keepalive: 30               # Keep-alive timeout
  max_requests: 1000          # Max requests per worker
  max_requests_jitter: 50     # Random jitter for max_requests
  
  # Memory Management
  preload: true               # Preload application
  max_memory_per_worker: "2GB" # Memory limit per worker
  worker_tmp_dir: "/dev/shm"  # Fast temporary storage
  
  # Caching
  enable_caching: true
  cache_ttl: 3600            # 1 hour default TTL
  cache_max_size: "1GB"      # Max cache size
```

### Clustering Performance

```yaml
clustering:
  # Resource Management
  max_concurrent_jobs: 10      # Maximum parallel clustering jobs
  max_memory_per_job: "4GB"   # Memory limit per job
  timeout_seconds: 1800       # 30 minutes max processing time
  
  # Processing Optimization
  batch_processing: true      # Enable batch processing
  batch_size: 1000           # Records per batch
  enable_gpu: true           # Use GPU acceleration if available
  num_threads: 8             # CPU threads for parallel processing
  
  # Algorithm-specific Tuning
  esn_params:
    reservoir_size: 200       # Larger for better accuracy
    spectral_radius: 0.95     # Close to 1 for better memory
    sparsity: 0.1            # Sparse connections for efficiency
    
  snn_params:
    n_neurons: 100           # More neurons for complex patterns
    threshold: 1.0           # Spike threshold
    learning_rate: 0.01      # Learning rate for STDP
```

## Clustering Algorithm Configuration

### Echo State Network (ESN) Configuration

```yaml
clustering:
  esn_params:
    reservoir_size: 200           # Number of reservoir neurons
    spectral_radius: 0.95         # Spectral radius of weight matrix
    sparsity: 0.1                # Connection sparsity (0.05-0.2)
    leaking_rate: 0.3            # Memory leaking rate (0.1-0.5)
    noise_level: 0.001           # Input noise for robustness
    washout_length: 100          # Transient washout period
    
    # Performance optimization
    use_sparse_matrix: true      # Use sparse matrices for efficiency
    parallel_processing: true    # Enable parallel reservoir computation
    batch_computation: true      # Batch multiple samples together
```

### Spiking Neural Network (SNN) Configuration

```yaml
clustering:
  snn_params:
    n_neurons: 100               # Number of spiking neurons
    threshold: 1.0               # Spike threshold voltage
    tau_membrane: 20.0           # Membrane time constant (ms)
    tau_synapse: 5.0            # Synaptic time constant (ms)
    learning_rate: 0.01          # STDP learning rate
    
    # Plasticity parameters
    stdp_window: 20.0           # STDP time window (ms)
    a_plus: 0.1                 # LTP amplitude
    a_minus: 0.105              # LTD amplitude
    
    # Network topology
    connection_prob: 0.3         # Probability of connection
    inhibitory_fraction: 0.2     # Fraction of inhibitory neurons
```

### Liquid State Machine (LSM) Configuration

```yaml
clustering:
  lsm_params:
    liquid_size: 128            # Number of liquid neurons (cubic root of total)
    connection_prob: 0.3        # Local connection probability
    tau_membrane: 30.0          # Membrane time constant
    tau_synapse: 3.0           # Synaptic time constant
    
    # Liquid structure
    structure: "3d_grid"        # 3D grid topology
    boundary_conditions: "periodic"  # Wrap-around connections
    
    # Input/output configuration
    input_scaling: 1.0          # Input signal scaling
    readout_neurons: 32         # Number of readout neurons
    readout_learning_rate: 0.001 # Readout learning rate
```

### Hybrid Reservoir Configuration

```yaml
clustering:
  hybrid_params:
    # Component weights (must sum to 1.0)
    esn_weight: 0.4             # Echo State Network contribution
    snn_weight: 0.3             # Spiking Neural Network contribution
    lsm_weight: 0.3             # Liquid State Machine contribution
    
    # Fusion method
    fusion_method: "weighted_average"  # How to combine outputs
    feature_selection: true     # Enable feature selection
    dimension_reduction: "pca"  # PCA, ICA, or none
    
    # Ensemble parameters
    voting_method: "soft"       # Soft or hard voting
    confidence_threshold: 0.7   # Minimum confidence for decisions
```

## API Configuration

### Server Configuration

```yaml
api:
  # Basic server settings
  host: "0.0.0.0"
  port: 8000
  
  # Worker configuration
  workers: 8
  worker_class: "gevent"        # gevent, uvicorn, sync
  worker_connections: 1000
  
  # Request handling
  timeout: 300
  keepalive: 30
  max_requests: 1000
  max_requests_jitter: 50
  
  # File upload limits
  max_file_size: "500MB"
  max_form_size: "10MB"
  
  # Response configuration
  gzip_enabled: true
  gzip_minimum_size: 1024
  
  # SSL Configuration (if not using reverse proxy)
  ssl_enabled: false
  ssl_cert_file: "/path/to/cert.pem"
  ssl_key_file: "/path/to/key.pem"
```

### Endpoint Configuration

```yaml
api:
  endpoints:
    # Analytics endpoints
    analytics:
      upload_timeout: 300       # File upload timeout
      processing_timeout: 1800  # Analysis timeout
      max_concurrent_uploads: 10 # Parallel uploads
      
    # Team formation endpoints  
    teams:
      generation_timeout: 60    # Team generation timeout
      max_team_size: 12        # Maximum team size
      min_team_size: 3         # Minimum team size
      
    # Health check configuration
    health:
      detailed_checks: true     # Include component health
      cache_duration: 30       # Cache health status (seconds)
      
  # API versioning
  versioning:
    enabled: true
    default_version: "v1"
    supported_versions: ["v1"]
```

## Logging Configuration

### Structured Logging

```yaml
logging:
  # Basic configuration
  level: "INFO"                 # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "json"               # json, structured, simple
  
  # Output destinations
  console: false               # Log to console
  file: true                   # Log to file
  syslog: false               # Log to syslog
  
  # File configuration
  file_path: "/var/log/insights/app.log"
  max_size: "100MB"           # Max file size before rotation
  backup_count: 5             # Number of backup files
  
  # Log levels by component
  loggers:
    "src.api": "INFO"
    "src.clustering": "INFO"
    "src.database": "WARNING"
    "sqlalchemy.engine": "WARNING"
    "uvicorn": "INFO"
    
  # Performance logging
  performance:
    enabled: true
    slow_query_threshold: 1000  # Log queries slower than 1 second
    log_request_duration: true
    
  # Security logging
  security:
    enabled: true
    log_authentication: true
    log_authorization: true
    log_data_access: true
```

### Log Format Configuration

```yaml
logging:
  formats:
    json:
      include_fields:
        - timestamp
        - level
        - logger
        - message
        - module
        - function
        - line_number
        - user_id
        - request_id
        - duration
        
    structured:
      pattern: "[{timestamp}] {level} {logger}: {message}"
      date_format: "%Y-%m-%d %H:%M:%S"
      
    simple:
      pattern: "{level}: {message}"
```

## Monitoring Configuration

### Prometheus Metrics

```yaml
monitoring:
  prometheus:
    enabled: true
    port: 9090
    path: "/metrics"
    
    # Custom metrics
    custom_metrics:
      clustering_duration:
        type: "histogram"
        description: "Time taken for clustering operations"
        buckets: [0.1, 0.5, 1, 5, 10, 30, 60, 300, 600]
        
      active_jobs:
        type: "gauge"
        description: "Number of active clustering jobs"
        
      api_requests_total:
        type: "counter"
        description: "Total API requests"
        labels: ["method", "endpoint", "status_code"]
        
      memory_usage:
        type: "gauge"
        description: "Memory usage by component"
        labels: ["component"]
```

### Health Checks

```yaml
monitoring:
  health_checks:
    enabled: true
    interval: 30                # Check every 30 seconds
    timeout: 10                # 10 second timeout per check
    
    checks:
      database:
        enabled: true
        query: "SELECT 1"
        timeout: 5
        
      redis:
        enabled: true
        command: "PING"
        timeout: 3
        
      disk_space:
        enabled: true
        threshold: 90          # Alert if > 90% full
        paths: ["/opt/insights", "/var/log"]
        
      memory:
        enabled: true
        threshold: 85          # Alert if > 85% used
        
      clustering_service:
        enabled: true
        test_job: true         # Run test clustering job
        timeout: 60
```

### Alerting Configuration

```yaml
monitoring:
  alerts:
    enabled: true
    
    # Alert thresholds
    thresholds:
      cpu_usage: 80            # CPU > 80%
      memory_usage: 85         # Memory > 85%
      disk_usage: 90           # Disk > 90%
      error_rate: 0.05         # Error rate > 5%
      response_time: 5000      # Response time > 5 seconds
      
    # Notification channels
    notifications:
      email:
        enabled: true
        smtp_server: "smtp.company.com"
        recipients: ["admin@company.com"]
        
      slack:
        enabled: true
        webhook_url: "${SLACK_WEBHOOK_URL}"
        channel: "#alerts"
        
      pagerduty:
        enabled: false
        service_key: "${PAGERDUTY_KEY}"
```

## Configuration Management Best Practices

### Environment Variables

```bash
# Environment-specific variables
export INSIGHTS_ENV=production
export INSIGHTS_CONFIG_FILE=/opt/insights/config/production.yml

# Database credentials
export DB_PASSWORD=secure_db_password
export REDIS_PASSWORD=secure_redis_password

# API Keys and secrets
export JWT_SECRET_KEY=jwt_secret_here
export ENCRYPTION_KEY=encryption_key_here

# Monitoring
export PROMETHEUS_ENABLED=true
export GRAFANA_API_KEY=grafana_key_here
```

### Configuration Validation

Create validation schemas using JSON Schema:

```yaml
# config/schema.yml
type: object
required:
  - environment
  - database
  - clustering
  - security
properties:
  environment:
    type: string
    enum: [development, testing, staging, production]
  database:
    type: object
    required: [url]
    properties:
      url:
        type: string
        pattern: "^(postgresql|sqlite)://.*"
      pool_size:
        type: integer
        minimum: 1
        maximum: 100
  clustering:
    type: object
    required: [default_method, default_clusters]
    properties:
      default_method:
        type: string
        enum: [esn, snn, lsm, hybrid_reservoir]
      default_clusters:
        type: integer
        minimum: 2
        maximum: 50
```

### Configuration Testing

```python
#!/usr/bin/env python3
"""
Configuration validation script
"""

import yaml
import jsonschema
from pathlib import Path

def validate_config(config_file, schema_file):
    """Validate configuration file against schema"""
    
    # Load configuration
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    # Load schema
    with open(schema_file) as f:
        schema = yaml.safe_load(f)
    
    # Validate
    try:
        jsonschema.validate(config, schema)
        print(f"✅ Configuration {config_file} is valid")
        return True
    except jsonschema.ValidationError as e:
        print(f"❌ Configuration {config_file} is invalid:")
        print(f"   {e.message}")
        return False

def test_database_connection(config):
    """Test database connectivity"""
    from sqlalchemy import create_engine
    
    try:
        engine = create_engine(config['database']['url'])
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: validate_config.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    schema_file = "config/schema.yml"
    
    # Validate configuration structure
    if not validate_config(config_file, schema_file):
        sys.exit(1)
    
    # Test configuration functionality
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    if not test_database_connection(config):
        sys.exit(1)
    
    print("✅ All configuration tests passed")
```

## Configuration Validation

### Automated Configuration Testing

```bash
#!/bin/bash
# Configuration validation script

CONFIG_DIR="/opt/insights/config"
SCHEMA_FILE="$CONFIG_DIR/schema.yml"

echo "Validating Observer Coordinator Insights configurations..."

# Validate each environment configuration
for config_file in "$CONFIG_DIR"/*.yml; do
    if [[ $(basename "$config_file") != "schema.yml" ]]; then
        echo "Validating $(basename "$config_file")..."
        python scripts/validate_config.py "$config_file"
        
        if [ $? -ne 0 ]; then
            echo "❌ Configuration validation failed for $(basename "$config_file")"
            exit 1
        fi
    fi
done

echo "✅ All configurations validated successfully"
```

### Runtime Configuration Monitoring

```python
"""
Configuration monitoring service
"""

import time
import hashlib
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigurationMonitor(FileSystemEventHandler):
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.config_hashes = {}
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        if event.src_path.endswith('.yml'):
            print(f"Configuration file modified: {event.src_path}")
            
            # Validate new configuration
            if self.validate_configuration(event.src_path):
                # Reload configuration
                self.config_manager.reload_configuration()
                print("✅ Configuration reloaded successfully")
            else:
                print("❌ Invalid configuration - keeping current settings")
    
    def validate_configuration(self, config_path):
        """Validate configuration file"""
        try:
            # Add validation logic here
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

# Start configuration monitoring
def start_config_monitoring():
    observer = Observer()
    event_handler = ConfigurationMonitor(config_manager)
    observer.schedule(event_handler, '/opt/insights/config', recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
```

This comprehensive configuration management guide ensures that Observer Coordinator Insights can be properly configured for any environment while maintaining security, performance, and reliability standards.