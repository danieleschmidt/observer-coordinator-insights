# Installation Guide - Administrator Documentation

This guide provides comprehensive instructions for installing and configuring Observer Coordinator Insights in various environments, from development setups to enterprise production deployments.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Development Installation](#development-installation)
3. [Production Installation](#production-installation)
4. [Database Setup](#database-setup)
5. [Security Configuration](#security-configuration)
6. [Load Balancer Configuration](#load-balancer-configuration)
7. [Monitoring Setup](#monitoring-setup)
8. [Backup Configuration](#backup-configuration)
9. [Troubleshooting Installation](#troubleshooting-installation)
10. [Post-Installation Validation](#post-installation-validation)

## System Requirements

### Minimum Requirements (Development)
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- **Python**: 3.11 or higher
- **RAM**: 4GB available
- **CPU**: 2 cores
- **Storage**: 10GB available space
- **Network**: Internet access for package installation

### Recommended Requirements (Production)
- **OS**: Linux (Ubuntu 22.04 LTS or CentOS 8+)
- **Python**: 3.11+
- **RAM**: 16GB+ (32GB for large organizations)
- **CPU**: 8+ cores
- **Storage**: 100GB+ SSD
- **Database**: PostgreSQL 13+ or MySQL 8.0+
- **Load Balancer**: Nginx or HAProxy
- **Monitoring**: Prometheus + Grafana

### Enterprise Requirements (High Availability)
- **Multiple Application Servers**: 3+ nodes
- **Database**: PostgreSQL cluster with read replicas
- **Cache**: Redis cluster (3+ nodes)
- **Load Balancer**: Hardware load balancer or cloud ALB
- **Storage**: Network-attached storage (NAS) or cloud storage
- **Monitoring**: Full observability stack with alerting

## Development Installation

### Quick Development Setup

```bash
# 1. Clone repository
git clone https://github.com/terragon-labs/observer-coordinator-insights.git
cd observer-coordinator-insights

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -e .

# 4. Install development dependencies
pip install -r requirements-dev.txt

# 5. Run quality gates
python scripts/run_quality_gates.py

# 6. Initialize development database
python scripts/init_database.py --mode development

# 7. Start development server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 8. Verify installation
curl http://localhost:8000/api/health
```

### Development Configuration

Create `config/development.yml`:

```yaml
# Development configuration
environment: development
debug: true

database:
  url: sqlite:///dev_database.db
  echo: true  # Log SQL queries

clustering:
  default_method: esn  # Faster for development
  default_clusters: 4
  timeout_seconds: 300

security:
  require_auth: false  # Disable for development
  secure_mode: false
  audit_logging: false

logging:
  level: DEBUG
  format: detailed
  
api:
  cors_origins: ["http://localhost:3000", "http://127.0.0.1:3000"]
  rate_limiting: false
```

### IDE Integration

#### VS Code Configuration

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.ruffEnabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestPath": "./venv/bin/pytest",
    "python.testing.pytestArgs": ["tests/"],
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"]
}
```

#### PyCharm Configuration

1. Open project in PyCharm
2. Configure interpreter: Settings → Python Interpreter → Add → Existing Environment
3. Point to `venv/bin/python`
4. Configure run configuration for `src.api.main:app`

## Production Installation

### Prerequisites

```bash
# Update system (Ubuntu/Debian)
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3.11 python3.11-venv python3.11-dev \
    build-essential curl wget git nginx postgresql-client redis-tools \
    supervisor htop iotop

# Create application user
sudo adduser --system --group --home /opt/insights insights
sudo usermod -aG sudo insights
```

### Application Installation

```bash
# Switch to application user
sudo su - insights

# Clone repository to production location
cd /opt/insights
git clone https://github.com/terragon-labs/observer-coordinator-insights.git app
cd app

# Create production virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install production dependencies
pip install --upgrade pip
pip install -e .
pip install gunicorn[gevent] supervisor

# Set permissions
sudo chown -R insights:insights /opt/insights
sudo chmod -R 755 /opt/insights
```

### Production Configuration

Create `/opt/insights/config/production.yml`:

```yaml
environment: production
debug: false

database:
  url: "postgresql://insights_user:secure_password@db-server:5432/insights_prod"
  pool_size: 20
  max_overflow: 40
  pool_timeout: 30

redis:
  url: "redis://redis-server:6379/0"
  max_connections: 100

clustering:
  default_method: hybrid_reservoir
  default_clusters: 4
  timeout_seconds: 1800
  max_concurrent_jobs: 10

security:
  require_auth: true
  secure_mode: true
  audit_logging: true
  encryption_key_file: /opt/insights/keys/encryption.key
  jwt_secret_file: /opt/insights/keys/jwt_secret.key

logging:
  level: INFO
  file: /var/log/insights/app.log
  max_size: 100MB
  backup_count: 5

api:
  cors_origins: ["https://insights.company.com"]
  rate_limiting: true
  max_requests_per_hour: 1000

monitoring:
  prometheus_enabled: true
  metrics_port: 9090
  health_check_interval: 30
```

### Service Configuration

#### Systemd Service

Create `/etc/systemd/system/insights.service`:

```ini
[Unit]
Description=Observer Coordinator Insights API
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=insights
Group=insights
WorkingDirectory=/opt/insights/app
Environment=PATH=/opt/insights/app/venv/bin
ExecStart=/opt/insights/app/venv/bin/gunicorn src.api.main:app \
    --workers 4 \
    --worker-class gevent \
    --worker-connections 1000 \
    --bind 0.0.0.0:8000 \
    --timeout 300 \
    --keepalive 30 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --preload \
    --log-level info \
    --access-logfile /var/log/insights/access.log \
    --error-logfile /var/log/insights/error.log
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### Enable and Start Service

```bash
# Enable service
sudo systemctl enable insights.service

# Start service
sudo systemctl start insights.service

# Check status
sudo systemctl status insights.service

# View logs
sudo journalctl -u insights.service -f
```

#### Supervisor Configuration (Alternative)

Create `/etc/supervisor/conf.d/insights.conf`:

```ini
[program:insights]
command=/opt/insights/app/venv/bin/gunicorn src.api.main:app --workers 4 --worker-class gevent --bind 0.0.0.0:8000
directory=/opt/insights/app
user=insights
group=insights
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/insights/app.log
environment=PATH="/opt/insights/app/venv/bin"
```

## Database Setup

### PostgreSQL Configuration

#### Install PostgreSQL

```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib

# CentOS/RHEL
sudo yum install postgresql-server postgresql-contrib
sudo postgresql-setup initdb
```

#### Configure PostgreSQL

```bash
# Switch to postgres user
sudo su - postgres

# Create database and user
psql << EOF
CREATE DATABASE insights_prod;
CREATE USER insights_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE insights_prod TO insights_user;

-- Performance tuning
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;

-- Security settings
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET password_encryption = 'scram-sha-256';
EOF

# Reload configuration
sudo systemctl reload postgresql
```

#### Configure pg_hba.conf

Edit `/etc/postgresql/13/main/pg_hba.conf`:

```
# Database administrative login by Unix domain socket
local   all             postgres                                peer

# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all             all                                     scram-sha-256
host    insights_prod   insights_user   127.0.0.1/32           scram-sha-256
host    insights_prod   insights_user   10.0.0.0/8             scram-sha-256
```

#### Run Database Migrations

```bash
# From application directory
cd /opt/insights/app
source venv/bin/activate

# Initialize database schema
python scripts/init_database.py --mode production

# Run migrations
alembic upgrade head

# Verify database setup
python scripts/verify_database.py
```

### Redis Configuration

#### Install Redis

```bash
# Ubuntu/Debian
sudo apt install redis-server

# CentOS/RHEL
sudo yum install redis
```

#### Configure Redis

Edit `/etc/redis/redis.conf`:

```
# Memory optimization
maxmemory 1gb
maxmemory-policy allkeys-lru

# Security
bind 127.0.0.1 10.0.0.1  # Bind to specific IPs
requirepass your_secure_redis_password
rename-command FLUSHDB ""
rename-command FLUSHALL ""

# Persistence
save 900 1
save 300 10
save 60 10000

# Performance
tcp-keepalive 300
timeout 0
```

#### Start Redis

```bash
sudo systemctl enable redis-server
sudo systemctl start redis-server
sudo systemctl status redis-server
```

## Security Configuration

### SSL/TLS Certificate Setup

#### Using Let's Encrypt

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d insights.company.com

# Test renewal
sudo certbot renew --dry-run

# Setup auto-renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo tee -a /etc/crontab
```

#### Using Self-Signed Certificates (Development)

```bash
# Create SSL directory
sudo mkdir -p /opt/insights/ssl

# Generate self-signed certificate
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /opt/insights/ssl/insights.key \
    -out /opt/insights/ssl/insights.crt \
    -subj "/C=US/ST=State/L=City/O=Organization/OU=Department/CN=insights.company.com"

# Set permissions
sudo chown insights:insights /opt/insights/ssl/*
sudo chmod 600 /opt/insights/ssl/*
```

### Encryption Keys Setup

```bash
# Create keys directory
sudo mkdir -p /opt/insights/keys
sudo chown insights:insights /opt/insights/keys
sudo chmod 700 /opt/insights/keys

# Generate encryption key
python -c "
import secrets
with open('/opt/insights/keys/encryption.key', 'wb') as f:
    f.write(secrets.token_bytes(32))
"

# Generate JWT secret
python -c "
import secrets
with open('/opt/insights/keys/jwt_secret.key', 'w') as f:
    f.write(secrets.token_urlsafe(64))
"

# Set permissions
sudo chmod 600 /opt/insights/keys/*
```

### Firewall Configuration

```bash
# Configure UFW (Ubuntu)
sudo ufw enable
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw allow 5432/tcp    # PostgreSQL (from app servers only)
sudo ufw allow 6379/tcp    # Redis (from app servers only)

# Restrict database access
sudo ufw allow from 10.0.1.0/24 to any port 5432
sudo ufw allow from 10.0.1.0/24 to any port 6379
```

## Load Balancer Configuration

### Nginx Configuration

Create `/etc/nginx/sites-available/insights`:

```nginx
upstream insights_backend {
    server 10.0.1.10:8000 weight=3;
    server 10.0.1.11:8000 weight=3;
    server 10.0.1.12:8000 weight=2 backup;
    keepalive 32;
}

server {
    listen 80;
    server_name insights.company.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name insights.company.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/insights.company.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/insights.company.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'";

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # File upload settings
    client_max_body_size 50M;
    client_body_timeout 60s;
    client_header_timeout 60s;

    # Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 10240;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    # Main application proxy
    location / {
        proxy_pass http://insights_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_connect_timeout 30s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # API endpoints with longer timeout
    location /api/analytics/upload {
        proxy_pass http://insights_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 30s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        client_max_body_size 100M;
    }

    # Health check endpoint
    location /api/health {
        proxy_pass http://insights_backend;
        access_log off;
    }

    # Static files (if serving directly)
    location /static/ {
        alias /opt/insights/app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

Enable site and restart nginx:

```bash
sudo ln -s /etc/nginx/sites-available/insights /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### HAProxy Configuration (Alternative)

Create `/etc/haproxy/haproxy.cfg`:

```
global
    daemon
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin
    stats timeout 30s
    user haproxy
    group haproxy

    # SSL Configuration
    ssl-default-bind-ciphers ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!SHA1:!AESCCM
    ssl-default-bind-options no-sslv3 no-tlsv10 no-tlsv11 no-tls-tickets
    ssl-default-server-ciphers ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!SHA1:!AESCCM
    ssl-default-server-options no-sslv3 no-tlsv10 no-tlsv11 no-tls-tickets

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog
    option dontlognull

frontend insights_frontend
    bind *:80
    bind *:443 ssl crt /opt/insights/ssl/insights.pem
    redirect scheme https if !{ ssl_fc }
    default_backend insights_backend

backend insights_backend
    balance roundrobin
    option httpchk GET /api/health
    http-check expect status 200
    server insights1 10.0.1.10:8000 check
    server insights2 10.0.1.11:8000 check
    server insights3 10.0.1.12:8000 check backup

listen stats
    bind *:8080
    stats enable
    stats uri /stats
    stats refresh 30s
    stats admin if TRUE
```

## Monitoring Setup

### Prometheus Configuration

Create `/etc/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "/etc/prometheus/rules/*.yml"

scrape_configs:
  - job_name: 'insights-api'
    static_configs:
      - targets: ['10.0.1.10:9090', '10.0.1.11:9090', '10.0.1.12:9090']
    scrape_interval: 15s
    metrics_path: /metrics

  - job_name: 'insights-health'
    static_configs:
      - targets: ['10.0.1.10:8000', '10.0.1.11:8000', '10.0.1.12:8000']
    scrape_interval: 30s
    metrics_path: /api/health

  - job_name: 'postgres'
    static_configs:
      - targets: ['db-server:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-server:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']
```

### Grafana Dashboard

Import the provided dashboard at `/monitoring/grafana-dashboard.json` or create custom dashboards for:
- Application performance metrics
- System resource utilization
- Database performance
- API response times
- Error rates and types

## Backup Configuration

### Database Backup Script

Create `/opt/insights/scripts/backup_database.sh`:

```bash
#!/bin/bash

# Configuration
BACKUP_DIR="/backup/insights"
DB_NAME="insights_prod"
DB_USER="insights_user"
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR

# Generate backup filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/insights_db_$TIMESTAMP.sql.gz"

# Create backup
export PGPASSWORD="$DB_PASSWORD"
pg_dump -h db-server -U $DB_USER -d $DB_NAME | gzip > $BACKUP_FILE

# Verify backup
if [ $? -eq 0 ]; then
    echo "Database backup successful: $BACKUP_FILE"
    
    # Upload to cloud storage (optional)
    aws s3 cp $BACKUP_FILE s3://company-backups/insights/
    
    # Clean old backups
    find $BACKUP_DIR -name "insights_db_*.sql.gz" -mtime +$RETENTION_DAYS -delete
else
    echo "Database backup failed!"
    exit 1
fi
```

### Application Backup Script

Create `/opt/insights/scripts/backup_application.sh`:

```bash
#!/bin/bash

BACKUP_DIR="/backup/insights"
APP_DIR="/opt/insights/app"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup configuration files
tar -czf "$BACKUP_DIR/insights_config_$TIMESTAMP.tar.gz" \
    /opt/insights/config/ \
    /opt/insights/keys/ \
    /opt/insights/ssl/

# Backup application logs
tar -czf "$BACKUP_DIR/insights_logs_$TIMESTAMP.tar.gz" \
    /var/log/insights/

# Backup analysis results (if stored locally)
if [ -d "/opt/insights/data" ]; then
    tar -czf "$BACKUP_DIR/insights_data_$TIMESTAMP.tar.gz" \
        /opt/insights/data/
fi

echo "Application backup completed: $TIMESTAMP"
```

### Automated Backup Schedule

Add to crontab:

```bash
# Edit crontab
sudo crontab -e

# Add backup schedules
# Database backup - daily at 2 AM
0 2 * * * /opt/insights/scripts/backup_database.sh

# Application backup - weekly on Sunday at 3 AM
0 3 * * 0 /opt/insights/scripts/backup_application.sh

# Log rotation - weekly
0 1 * * 0 /usr/sbin/logrotate /etc/logrotate.d/insights
```

## Troubleshooting Installation

### Common Installation Issues

#### Python Version Issues
```bash
# Check Python version
python3.11 --version

# If not available, install from source
wget https://www.python.org/ftp/python/3.11.7/Python-3.11.7.tgz
tar xzf Python-3.11.7.tgz
cd Python-3.11.7
./configure --enable-optimizations
make altinstall
```

#### Database Connection Issues
```bash
# Test database connectivity
psql -h db-server -U insights_user -d insights_prod -c "SELECT version();"

# Check firewall rules
sudo iptables -L | grep 5432

# Check PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-13-main.log
```

#### Permission Issues
```bash
# Fix file permissions
sudo chown -R insights:insights /opt/insights
sudo chmod -R 755 /opt/insights
sudo chmod 600 /opt/insights/keys/*
sudo chmod 600 /opt/insights/ssl/*
```

### Installation Validation Script

Create `/opt/insights/scripts/validate_installation.py`:

```python
#!/usr/bin/env python3
"""
Installation validation script for Observer Coordinator Insights
"""

import os
import sys
import subprocess
import requests
import psycopg2
import redis
from pathlib import Path

def check_python_version():
    """Check Python version requirements"""
    version = sys.version_info
    if version.major != 3 or version.minor < 11:
        print("❌ Python 3.11+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_database_connection():
    """Check database connectivity"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'insights_prod'),
            user=os.getenv('DB_USER', 'insights_user'),
            password=os.getenv('DB_PASSWORD')
        )
        conn.close()
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def check_redis_connection():
    """Check Redis connectivity"""
    try:
        r = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            password=os.getenv('REDIS_PASSWORD')
        )
        r.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def check_api_health():
    """Check API health endpoint"""
    try:
        response = requests.get('http://localhost:8000/api/health', timeout=10)
        if response.status_code == 200:
            print("✅ API health check passed")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False

def check_file_permissions():
    """Check critical file permissions"""
    checks = [
        ('/opt/insights/keys/', 0o700),
        ('/opt/insights/config/', 0o755),
        ('/var/log/insights/', 0o755)
    ]
    
    all_good = True
    for path, expected_mode in checks:
        if os.path.exists(path):
            actual_mode = os.stat(path).st_mode & 0o777
            if actual_mode == expected_mode:
                print(f"✅ Permissions correct for {path}")
            else:
                print(f"❌ Permissions incorrect for {path}: {oct(actual_mode)} (expected {oct(expected_mode)})")
                all_good = False
        else:
            print(f"❌ Path does not exist: {path}")
            all_good = False
    
    return all_good

def main():
    """Run all validation checks"""
    print("Observer Coordinator Insights - Installation Validation")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Database Connection", check_database_connection),
        ("Redis Connection", check_redis_connection),
        ("API Health", check_api_health),
        ("File Permissions", check_file_permissions)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\nChecking {name}...")
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {name} check failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} checks passed! Installation is ready for production.")
        sys.exit(0)
    else:
        print(f"❌ {passed}/{total} checks passed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Post-Installation Validation

### Run Validation Script

```bash
cd /opt/insights/app
source venv/bin/activate
python scripts/validate_installation.py
```

### Manual Validation Steps

```bash
# 1. Test API endpoints
curl http://localhost:8000/api/health
curl http://localhost:8000/docs

# 2. Test file upload
curl -X POST -F "file=@tests/fixtures/sample_data.csv" \
    http://localhost:8000/api/analytics/upload

# 3. Check logs
sudo tail -f /var/log/insights/app.log

# 4. Monitor system resources
htop
sudo iotop -o

# 5. Test load balancer (if configured)
curl https://insights.company.com/api/health

# 6. Verify SSL certificate
openssl s_client -connect insights.company.com:443 -servername insights.company.com
```

### Performance Baseline

```bash
# Run performance baseline tests
python scripts/performance_baseline.py

# Expected results:
# - 100 employees: < 5 seconds
# - 500 employees: < 30 seconds  
# - 1000 employees: < 60 seconds
# - Memory usage: < 2GB for 1000 employees
```

Your Observer Coordinator Insights installation is now complete and ready for production use! For ongoing maintenance and operations, refer to the [Configuration Management](configuration.md) and [Monitoring & Operations](monitoring.md) guides.