# Monitoring & Operations Guide

This comprehensive guide covers monitoring, alerting, and operational procedures for Observer Coordinator Insights in production environments. It includes metrics collection, dashboard setup, alerting strategies, and operational runbooks.

## Table of Contents

1. [Monitoring Overview](#monitoring-overview)
2. [Metrics Collection](#metrics-collection)
3. [Dashboards & Visualization](#dashboards--visualization)
4. [Alerting & Notifications](#alerting--notifications)
5. [Log Management](#log-management)
6. [Performance Monitoring](#performance-monitoring)
7. [Health Checks](#health-checks)
8. [Incident Response](#incident-response)
9. [Capacity Planning](#capacity-planning)
10. [Operational Runbooks](#operational-runbooks)

## Monitoring Overview

### Monitoring Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │────│   Prometheus    │────│    Grafana      │
│   Metrics       │    │   (Collection)  │    │  (Visualization)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Log Files     │────│   Loki/ELK      │────│   Alertmanager  │
│   (Application) │    │ (Log Aggregation)│   │  (Notifications)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   System        │────│   Node Exporter │────│   PagerDuty     │
│   Metrics       │    │   (System Info) │    │   (Escalation)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Monitoring Components

- **Application Metrics**: Custom business and performance metrics
- **System Metrics**: CPU, memory, disk, network utilization
- **Database Metrics**: Query performance, connection pools, locks
- **API Metrics**: Request rates, response times, error rates
- **Clustering Metrics**: Job durations, accuracy scores, resource usage
- **Security Metrics**: Authentication failures, suspicious activities

## Metrics Collection

### Prometheus Configuration

Create `/etc/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'insights-prod'
    environment: 'production'

rule_files:
  - "/etc/prometheus/rules/insights_*.yml"

scrape_configs:
  # Observer Coordinator Insights Application
  - job_name: 'insights-app'
    static_configs:
      - targets: 
          - 'insights-1.company.com:9090'
          - 'insights-2.company.com:9090'
          - 'insights-3.company.com:9090'
    scrape_interval: 15s
    metrics_path: /metrics
    scrape_timeout: 10s

  # System metrics via node_exporter
  - job_name: 'node'
    static_configs:
      - targets:
          - 'insights-1.company.com:9100'
          - 'insights-2.company.com:9100'
          - 'insights-3.company.com:9100'
          - 'db-1.company.com:9100'
          - 'redis-1.company.com:9100'

  # PostgreSQL metrics
  - job_name: 'postgres'
    static_configs:
      - targets: ['db-1.company.com:9187']
    scrape_interval: 30s

  # Redis metrics
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-1.company.com:9121']

  # Load balancer metrics (nginx)
  - job_name: 'nginx'
    static_configs:
      - targets: ['lb-1.company.com:9113']

  # Application health checks
  - job_name: 'insights-health'
    static_configs:
      - targets:
          - 'insights-1.company.com:8000'
          - 'insights-2.company.com:8000'
          - 'insights-3.company.com:8000'
    metrics_path: /api/health/metrics
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

# Remote storage (optional)
remote_write:
  - url: "https://prometheus-remote-storage.company.com/api/v1/write"
    basic_auth:
      username: "insights-prod"
      password_file: "/etc/prometheus/remote-storage-password"
```

### Custom Application Metrics

#### Core Business Metrics

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time
from functools import wraps

# Create custom registry
registry = CollectorRegistry()

# Business metrics
clustering_jobs_total = Counter(
    'clustering_jobs_total', 
    'Total clustering jobs processed',
    ['method', 'status', 'organization'],
    registry=registry
)

clustering_duration = Histogram(
    'clustering_duration_seconds',
    'Time taken for clustering operations',
    ['method', 'cluster_count'],
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 300, 600, 1800],
    registry=registry
)

active_jobs = Gauge(
    'clustering_active_jobs',
    'Number of active clustering jobs',
    registry=registry
)

team_formations_total = Counter(
    'team_formations_total',
    'Total team formations generated',
    ['strategy', 'team_count'],
    registry=registry
)

# API metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status_code'],
    registry=registry
)

api_request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
    registry=registry
)

# Resource metrics
memory_usage = Gauge(
    'app_memory_usage_bytes',
    'Application memory usage',
    ['component'],
    registry=registry
)

database_connections = Gauge(
    'database_connections_active',
    'Active database connections',
    ['pool_name'],
    registry=registry
)

# Quality metrics
clustering_quality = Gauge(
    'clustering_quality_score',
    'Clustering quality metrics',
    ['metric_type', 'method'],
    registry=registry
)

# Decorator for automatic metric collection
def monitor_clustering_job(method):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            active_jobs.inc()
            
            try:
                result = func(*args, **kwargs)
                clustering_jobs_total.labels(
                    method=method, 
                    status='success', 
                    organization=kwargs.get('organization', 'unknown')
                ).inc()
                
                # Record quality metrics
                if 'clustering_metrics' in result:
                    metrics = result['clustering_metrics']
                    clustering_quality.labels(
                        metric_type='silhouette_score',
                        method=method
                    ).set(metrics.get('silhouette_score', 0))
                    
                    clustering_quality.labels(
                        metric_type='stability_score',
                        method=method
                    ).set(metrics.get('stability_score', 0))
                
                return result
                
            except Exception as e:
                clustering_jobs_total.labels(
                    method=method,
                    status='error',
                    organization=kwargs.get('organization', 'unknown')
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                clustering_duration.labels(
                    method=method,
                    cluster_count=kwargs.get('n_clusters', 0)
                ).observe(duration)
                active_jobs.dec()
        
        return wrapper
    return decorator
```

#### System Resource Monitoring

```python
# src/monitoring/system_metrics.py
import psutil
import threading
import time
from prometheus_client import Gauge

# System metrics
cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
memory_usage_percent = Gauge('system_memory_usage_percent', 'Memory usage percentage')
disk_usage_percent = Gauge('system_disk_usage_percent', 'Disk usage percentage', ['mount_point'])
network_bytes_sent = Gauge('system_network_bytes_sent_total', 'Network bytes sent')
network_bytes_recv = Gauge('system_network_bytes_recv_total', 'Network bytes received')

class SystemMetricsCollector:
    def __init__(self, collection_interval=30):
        self.collection_interval = collection_interval
        self.running = False
        self.thread = None
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._collect_metrics)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _collect_metrics(self):
        while self.running:
            try:
                # CPU usage
                cpu_usage.set(psutil.cpu_percent(interval=1))
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_usage_percent.set(memory.percent)
                
                # Disk usage
                for disk in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(disk.mountpoint)
                        disk_usage_percent.labels(mount_point=disk.mountpoint).set(
                            (usage.used / usage.total) * 100
                        )
                    except PermissionError:
                        pass
                
                # Network usage
                net_io = psutil.net_io_counters()
                network_bytes_sent.set(net_io.bytes_sent)
                network_bytes_recv.set(net_io.bytes_recv)
                
            except Exception as e:
                print(f"Error collecting system metrics: {e}")
            
            time.sleep(self.collection_interval)

# Start system metrics collection
system_collector = SystemMetricsCollector()
system_collector.start()
```

### Database Monitoring

#### PostgreSQL Metrics

```yaml
# Install postgres_exporter
version: '3.8'
services:
  postgres_exporter:
    image: prometheuscommunity/postgres-exporter
    environment:
      DATA_SOURCE_NAME: "postgresql://insights_user:${DB_PASSWORD}@db-server:5432/insights_prod?sslmode=disable"
    ports:
      - "9187:9187"
    restart: unless-stopped
```

#### Custom Database Queries

```sql
-- Create monitoring views for custom metrics

-- Active connections by state
CREATE OR REPLACE VIEW monitoring_connections AS
SELECT 
    state,
    COUNT(*) as connection_count,
    NOW() as timestamp
FROM pg_stat_activity 
WHERE datname = 'insights_prod'
GROUP BY state;

-- Long running queries
CREATE OR REPLACE VIEW monitoring_long_queries AS
SELECT 
    pid,
    now() - query_start as duration,
    query,
    state
FROM pg_stat_activity
WHERE datname = 'insights_prod'
  AND now() - query_start > interval '5 minutes'
  AND state != 'idle';

-- Table sizes and statistics
CREATE OR REPLACE VIEW monitoring_table_stats AS
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    last_vacuum,
    last_analyze
FROM pg_stat_user_tables;
```

## Dashboards & Visualization

### Grafana Dashboard Configuration

#### Main Application Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "Observer Coordinator Insights - Production",
    "tags": ["insights", "production"],
    "timezone": "browser",
    "panels": [
      {
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(api_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec"
          }
        ]
      },
      {
        "title": "API Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Active Clustering Jobs",
        "type": "singlestat",
        "targets": [
          {
            "expr": "clustering_active_jobs"
          }
        ]
      },
      {
        "title": "Clustering Success Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(clustering_jobs_total{status=\"success\"}[5m]) / rate(clustering_jobs_total[5m]) * 100",
            "legendFormat": "Success Rate %"
          }
        ]
      },
      {
        "title": "System Resource Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "system_cpu_usage_percent",
            "legendFormat": "CPU %"
          },
          {
            "expr": "system_memory_usage_percent", 
            "legendFormat": "Memory %"
          }
        ]
      }
    ]
  }
}
```

#### Database Performance Dashboard

```json
{
  "dashboard": {
    "title": "Database Performance",
    "panels": [
      {
        "title": "Database Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "pg_stat_database_numbackends{datname=\"insights_prod\"}",
            "legendFormat": "Active Connections"
          }
        ]
      },
      {
        "title": "Query Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(pg_stat_database_tup_fetched[5m])",
            "legendFormat": "Rows Fetched/sec"
          },
          {
            "expr": "rate(pg_stat_database_tup_inserted[5m])",
            "legendFormat": "Rows Inserted/sec"
          }
        ]
      },
      {
        "title": "Lock Waits",
        "type": "graph",
        "targets": [
          {
            "expr": "pg_locks_count{mode=\"AccessExclusiveLock\"}",
            "legendFormat": "Exclusive Locks"
          }
        ]
      }
    ]
  }
}
```

### Custom Dashboard Panels

#### Clustering Performance Panel

```python
# Custom panel for clustering performance metrics
def create_clustering_performance_panel():
    return {
        "title": "Clustering Performance by Method",
        "type": "graph",
        "targets": [
            {
                "expr": "histogram_quantile(0.95, rate(clustering_duration_seconds_bucket[10m]))",
                "legendFormat": "95th percentile - {{method}}"
            },
            {
                "expr": "histogram_quantile(0.50, rate(clustering_duration_seconds_bucket[10m]))",
                "legendFormat": "Median - {{method}}"
            }
        ],
        "yAxes": [
            {
                "label": "Duration (seconds)",
                "logBase": 1
            }
        ],
        "thresholds": [
            {
                "value": 300,  # 5 minutes warning
                "colorMode": "critical",
                "fill": True,
                "op": "gt"
            },
            {
                "value": 600,  # 10 minutes critical
                "colorMode": "critical",
                "fill": True,
                "op": "gt"
            }
        ]
    }
```

## Alerting & Notifications

### Alert Rules Configuration

Create `/etc/prometheus/rules/insights_alerts.yml`:

```yaml
groups:
  - name: insights.alerts
    rules:
      # High error rate alert
      - alert: HighErrorRate
        expr: rate(api_requests_total{status_code=~"5.."}[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
          service: insights-api
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"

      # API response time alert
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m])) > 5
        for: 3m
        labels:
          severity: warning
          service: insights-api
        annotations:
          summary: "High API response time"
          description: "95th percentile response time is {{ $value }}s"

      # Active jobs stuck alert
      - alert: JobsStuck
        expr: clustering_active_jobs > 0 and increase(clustering_jobs_total[10m]) == 0
        for: 10m
        labels:
          severity: critical
          service: insights-clustering
        annotations:
          summary: "Clustering jobs appear stuck"
          description: "{{ $value }} jobs active but no completions in 10 minutes"

      # Database connection alert
      - alert: DatabaseConnectionHigh
        expr: pg_stat_database_numbackends / pg_settings_max_connections > 0.8
        for: 5m
        labels:
          severity: warning
          service: database
        annotations:
          summary: "Database connection usage high"
          description: "Using {{ $value | humanizePercentage }} of max connections"

      # Memory usage alert
      - alert: HighMemoryUsage
        expr: system_memory_usage_percent > 85
        for: 5m
        labels:
          severity: warning
          service: system
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}%"

      # Disk space alert
      - alert: LowDiskSpace
        expr: system_disk_usage_percent > 90
        for: 1m
        labels:
          severity: critical
          service: system
        annotations:
          summary: "Low disk space"
          description: "Disk usage is {{ $value }}% on {{ $labels.mount_point }}"

      # Clustering quality alert
      - alert: LowClusteringQuality
        expr: clustering_quality{metric_type="silhouette_score"} < 0.3
        for: 0m
        labels:
          severity: warning
          service: insights-clustering
        annotations:
          summary: "Low clustering quality detected"
          description: "Silhouette score is {{ $value }} (threshold: 0.3)"

      # Service down alert
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "{{ $labels.instance }} has been down for more than 1 minute"
```

### Alertmanager Configuration

Create `/etc/alertmanager/alertmanager.yml`:

```yaml
global:
  smtp_smarthost: 'smtp.company.com:587'
  smtp_from: 'alerts@company.com'
  slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

# Notification templates
templates:
  - '/etc/alertmanager/templates/*.tmpl'

# Routing tree
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'
  
  routes:
    # Critical alerts go to PagerDuty and Slack
    - match:
        severity: critical
      receiver: 'critical-alerts'
      continue: true
    
    # Database alerts to DBA team
    - match:
        service: database
      receiver: 'dba-team'
    
    # System alerts to infrastructure team
    - match:
        service: system
      receiver: 'infrastructure-team'
    
    # Application alerts to development team
    - match:
        service: insights-api
      receiver: 'dev-team'

# Notification receivers
receivers:
  - name: 'default'
    email_configs:
      - to: 'admin@company.com'
        subject: 'Observer Coordinator Insights Alert'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Labels: {{ range .Labels.SortedPairs }}{{ .Name }}: {{ .Value }} {{ end }}
          {{ end }}

  - name: 'critical-alerts'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        description: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
    slack_configs:
      - channel: '#critical-alerts'
        title: 'Critical Alert: Observer Coordinator Insights'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'dba-team'
    email_configs:
      - to: 'dba-team@company.com'
        subject: 'Database Alert - Observer Coordinator Insights'

  - name: 'infrastructure-team'
    email_configs:
      - to: 'infrastructure@company.com'
        subject: 'Infrastructure Alert - Observer Coordinator Insights'

  - name: 'dev-team'
    slack_configs:
      - channel: '#insights-alerts'
        title: 'Application Alert'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

# Alert inhibition rules
inhibit_rules:
  # If service is down, don't alert on high response times
  - source_match:
      alertname: ServiceDown
    target_match:
      alertname: HighResponseTime
    equal: ['instance']
```

### Alert Testing

```bash
#!/bin/bash
# Test alert configurations

echo "Testing Prometheus alert rules..."

# Validate alert rules syntax
promtool check rules /etc/prometheus/rules/insights_alerts.yml

if [ $? -eq 0 ]; then
    echo "✅ Alert rules syntax is valid"
else
    echo "❌ Alert rules syntax is invalid"
    exit 1
fi

echo "Testing Alertmanager configuration..."

# Validate alertmanager config
amtool check-config /etc/alertmanager/alertmanager.yml

if [ $? -eq 0 ]; then
    echo "✅ Alertmanager configuration is valid"
else
    echo "❌ Alertmanager configuration is invalid"
    exit 1
fi

# Test alert firing
echo "Sending test alert..."
amtool alert add alertname="TestAlert" severity="warning" instance="test" \
    summary="This is a test alert" description="Testing alert system"

echo "✅ Alert tests completed"
```

## Log Management

### Centralized Logging with ELK Stack

#### Logstash Configuration

Create `/etc/logstash/conf.d/insights.conf`:

```ruby
input {
  beats {
    port => 5044
  }
  
  # Direct file input for critical logs
  file {
    path => "/var/log/insights/audit.log"
    start_position => "beginning"
    codec => "json"
    type => "audit"
  }
}

filter {
  if [type] == "insights-api" {
    grok {
      match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{DATA:logger}: %{GREEDYDATA:message}" }
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    # Parse JSON log entries
    if [message] =~ /^\{.*\}$/ {
      json {
        source => "message"
      }
    }
    
    # Extract request IDs for tracing
    if [request_id] {
      mutate {
        add_field => { "trace_id" => "%{request_id}" }
      }
    }
  }
  
  if [type] == "audit" {
    # Audit log processing
    mutate {
      add_field => { "security_event" => "true" }
    }
  }
  
  # Add environment tags
  mutate {
    add_field => { "environment" => "${ENVIRONMENT:production}" }
    add_field => { "service" => "observer-coordinator-insights" }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch-1:9200", "elasticsearch-2:9200"]
    index => "insights-logs-%{+YYYY.MM.dd}"
    
    # Security events to separate index
    if [type] == "audit" {
      index => "insights-audit-%{+YYYY.MM.dd}"
    }
  }
  
  # Send critical errors to dead letter queue
  if [level] == "CRITICAL" or [level] == "ERROR" {
    file {
      path => "/var/log/logstash/insights-errors.log"
      codec => json_lines
    }
  }
}
```

#### Filebeat Configuration

Create `/etc/filebeat/filebeat.yml`:

```yaml
filebeat.inputs:
  # Application logs
  - type: log
    enabled: true
    paths:
      - /var/log/insights/app.log
    fields:
      service: insights-api
      type: insights-api
    fields_under_root: true
    multiline.pattern: '^\d{4}-\d{2}-\d{2}'
    multiline.negate: true
    multiline.match: after
    
  # Audit logs
  - type: log
    enabled: true
    paths:
      - /var/log/insights/audit.log
    fields:
      service: insights-audit
      type: audit
    fields_under_root: true
    
  # Access logs
  - type: log
    enabled: true
    paths:
      - /var/log/nginx/insights-access.log
    fields:
      service: nginx
      type: access-log
    fields_under_root: true

output.logstash:
  hosts: ["logstash-1:5044", "logstash-2:5044"]

processors:
  - add_host_metadata:
      when.not.contains.tags: forwarded
  - add_docker_metadata: ~
  - add_kubernetes_metadata: ~
```

### Log Analysis Queries

#### Elasticsearch Queries for Common Issues

```json
{
  "common_queries": {
    "high_error_rate": {
      "query": {
        "bool": {
          "must": [
            {"term": {"level": "ERROR"}},
            {"range": {"@timestamp": {"gte": "now-5m"}}}
          ]
        }
      },
      "aggs": {
        "error_count": {
          "date_histogram": {
            "field": "@timestamp",
            "interval": "1m"
          }
        }
      }
    },
    
    "slow_requests": {
      "query": {
        "bool": {
          "must": [
            {"exists": {"field": "duration"}},
            {"range": {"duration": {"gte": 5000}}}
          ]
        }
      },
      "sort": [{"duration": "desc"}]
    },
    
    "clustering_failures": {
      "query": {
        "bool": {
          "must": [
            {"term": {"logger": "clustering"}},
            {"terms": {"level": ["ERROR", "CRITICAL"]}}
          ]
        }
      }
    },
    
    "security_events": {
      "query": {
        "bool": {
          "must": [
            {"term": {"type": "audit"}},
            {"terms": {"action": ["login_failed", "unauthorized_access"]}}
          ]
        }
      }
    }
  }
}
```

## Performance Monitoring

### Application Performance Monitoring (APM)

#### OpenTelemetry Integration

```python
# src/monitoring/tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

def setup_tracing(app, service_name="observer-coordinator-insights"):
    # Set up tracer provider
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="jaeger-agent",
        agent_port=6831,
    )
    
    # Add span processor
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Instrument frameworks
    FastAPIInstrumentor.instrument_app(app)
    SQLAlchemyInstrumentor().instrument()
    RedisInstrumentor().instrument()
    
    return tracer

# Custom tracing decorator
def trace_clustering_operation(operation_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(f"clustering.{operation_name}") as span:
                # Add attributes
                span.set_attribute("clustering.method", kwargs.get("method", "unknown"))
                span.set_attribute("clustering.clusters", kwargs.get("n_clusters", 0))
                span.set_attribute("data.employee_count", kwargs.get("employee_count", 0))
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("operation.success", True)
                    
                    # Add result metrics to span
                    if "clustering_metrics" in result:
                        metrics = result["clustering_metrics"]
                        span.set_attribute("clustering.silhouette_score", 
                                         metrics.get("silhouette_score", 0))
                        span.set_attribute("clustering.stability_score",
                                         metrics.get("stability_score", 0))
                    
                    return result
                except Exception as e:
                    span.set_attribute("operation.success", False)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator
```

### Performance Baselines

#### Automated Performance Testing

```python
#!/usr/bin/env python3
"""
Performance baseline monitoring script
"""

import time
import json
import statistics
from datetime import datetime
from pathlib import Path

class PerformanceMonitor:
    def __init__(self):
        self.baselines = {
            "api_response_time": {"p50": 0.1, "p95": 0.5, "p99": 1.0},
            "clustering_duration": {
                "100_employees": 5.0,   # seconds
                "500_employees": 30.0,
                "1000_employees": 60.0
            },
            "memory_usage": {
                "idle": 512,           # MB
                "processing_1000": 2048  # MB
            }
        }
    
    def run_performance_tests(self):
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "tests": []
        }
        
        # API response time test
        api_times = self.test_api_response_times()
        results["tests"].append({
            "name": "api_response_time",
            "results": api_times,
            "baseline_met": self.check_api_baseline(api_times)
        })
        
        # Clustering performance test
        clustering_times = self.test_clustering_performance()
        results["tests"].append({
            "name": "clustering_performance", 
            "results": clustering_times,
            "baseline_met": self.check_clustering_baseline(clustering_times)
        })
        
        # Memory usage test
        memory_usage = self.test_memory_usage()
        results["tests"].append({
            "name": "memory_usage",
            "results": memory_usage,
            "baseline_met": self.check_memory_baseline(memory_usage)
        })
        
        return results
    
    def test_api_response_times(self):
        """Test API endpoint response times"""
        import requests
        
        endpoints = [
            "/api/health",
            "/api/analytics/status",
            "/docs"
        ]
        
        times = []
        for endpoint in endpoints:
            for _ in range(10):
                start = time.time()
                response = requests.get(f"http://localhost:8000{endpoint}")
                duration = time.time() - start
                if response.status_code == 200:
                    times.append(duration)
        
        return {
            "p50": statistics.median(times),
            "p95": statistics.quantiles(times, n=20)[18],  # 95th percentile
            "p99": statistics.quantiles(times, n=100)[98]   # 99th percentile
        }
    
    def test_clustering_performance(self):
        """Test clustering performance with different dataset sizes"""
        # This would use test datasets of different sizes
        # and measure clustering duration
        pass
    
    def check_api_baseline(self, results):
        """Check if API performance meets baseline"""
        baseline = self.baselines["api_response_time"]
        return (
            results["p50"] <= baseline["p50"] and
            results["p95"] <= baseline["p95"] and
            results["p99"] <= baseline["p99"]
        )
    
    def generate_report(self, results):
        """Generate performance report"""
        report = f"""
Performance Monitoring Report
Generated: {results['timestamp']}

Test Results:
"""
        for test in results["tests"]:
            status = "✅ PASS" if test["baseline_met"] else "❌ FAIL"
            report += f"\n{test['name']}: {status}\n"
            
            if isinstance(test["results"], dict):
                for key, value in test["results"].items():
                    report += f"  {key}: {value}\n"
        
        return report

# Run performance monitoring
if __name__ == "__main__":
    monitor = PerformanceMonitor()
    results = monitor.run_performance_tests()
    report = monitor.generate_report(results)
    
    print(report)
    
    # Save results
    with open(f"/var/log/insights/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(results, f, indent=2)
```

## Health Checks

### Comprehensive Health Check System

```python
# src/health/health_checks.py
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass
import asyncio
import time
from datetime import datetime

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheckResult:
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    details: Optional[Dict] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class HealthChecker:
    def __init__(self):
        self.checks = {}
        self.check_timeout = 10  # seconds
    
    def register_check(self, name: str, check_func, timeout: int = None):
        """Register a health check function"""
        self.checks[name] = {
            "func": check_func,
            "timeout": timeout or self.check_timeout
        }
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}
        
        for name, check_config in self.checks.items():
            try:
                start_time = time.time()
                
                # Run check with timeout
                result = await asyncio.wait_for(
                    check_config["func"](),
                    timeout=check_config["timeout"]
                )
                
                duration = (time.time() - start_time) * 1000
                
                if isinstance(result, HealthCheckResult):
                    result.duration_ms = duration
                    results[name] = result
                else:
                    # Legacy boolean result
                    status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                    results[name] = HealthCheckResult(
                        name=name,
                        status=status,
                        message="OK" if result else "Check failed",
                        duration_ms=duration
                    )
                    
            except asyncio.TimeoutError:
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check timed out after {check_config['timeout']}s",
                    duration_ms=check_config["timeout"] * 1000
                )
            except Exception as e:
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}",
                    duration_ms=0
                )
        
        return results
    
    def get_overall_status(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Determine overall system health"""
        if not results:
            return HealthStatus.UNHEALTHY
        
        statuses = [result.status for result in results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

# Health check implementations
async def check_database():
    """Check database connectivity and performance"""
    try:
        from src.database.connection import get_database
        
        db = get_database()
        start_time = time.time()
        
        # Simple query to test connectivity
        result = await db.execute("SELECT 1")
        query_time = (time.time() - start_time) * 1000
        
        # Check connection pool
        pool_info = db.get_pool_info()
        
        if query_time > 1000:  # > 1 second
            return HealthCheckResult(
                name="database",
                status=HealthStatus.DEGRADED,
                message=f"Database slow (query time: {query_time:.2f}ms)",
                details={
                    "query_time_ms": query_time,
                    "pool_size": pool_info.get("pool_size", 0),
                    "checked_out": pool_info.get("checked_out", 0)
                }
            )
        
        return HealthCheckResult(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Database connection healthy",
            details={
                "query_time_ms": query_time,
                "pool_size": pool_info.get("pool_size", 0),
                "checked_out": pool_info.get("checked_out", 0)
            }
        )
        
    except Exception as e:
        return HealthCheckResult(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database connection failed: {str(e)}"
        )

async def check_redis():
    """Check Redis connectivity"""
    try:
        import redis
        from src.config import get_redis_url
        
        r = redis.from_url(get_redis_url())
        start_time = time.time()
        
        # Ping Redis
        response = r.ping()
        ping_time = (time.time() - start_time) * 1000
        
        # Get Redis info
        info = r.info()
        
        return HealthCheckResult(
            name="redis",
            status=HealthStatus.HEALTHY,
            message="Redis connection healthy",
            details={
                "ping_time_ms": ping_time,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown")
            }
        )
        
    except Exception as e:
        return HealthCheckResult(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            message=f"Redis connection failed: {str(e)}"
        )

async def check_clustering_service():
    """Check clustering service health"""
    try:
        # Test basic clustering functionality
        from src.insights_clustering import NeuromorphicClusterer
        import numpy as np
        
        # Create small test dataset
        test_data = np.random.rand(10, 4) * 100
        
        start_time = time.time()
        clusterer = NeuromorphicClusterer(method="esn", n_clusters=2)
        clusterer.fit(test_data)
        clustering_time = (time.time() - start_time) * 1000
        
        if clustering_time > 5000:  # > 5 seconds for tiny dataset
            return HealthCheckResult(
                name="clustering",
                status=HealthStatus.DEGRADED,
                message=f"Clustering service slow ({clustering_time:.2f}ms for test)",
                details={"test_clustering_time_ms": clustering_time}
            )
        
        return HealthCheckResult(
            name="clustering",
            status=HealthStatus.HEALTHY,
            message="Clustering service healthy",
            details={"test_clustering_time_ms": clustering_time}
        )
        
    except Exception as e:
        return HealthCheckResult(
            name="clustering",
            status=HealthStatus.UNHEALTHY,
            message=f"Clustering service failed: {str(e)}"
        )

async def check_disk_space():
    """Check available disk space"""
    import shutil
    
    paths_to_check = ["/opt/insights", "/var/log/insights", "/tmp"]
    
    for path in paths_to_check:
        try:
            total, used, free = shutil.disk_usage(path)
            usage_percent = (used / total) * 100
            
            if usage_percent > 95:
                return HealthCheckResult(
                    name="disk_space",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Disk space critical on {path} ({usage_percent:.1f}% used)",
                    details={"path": path, "usage_percent": usage_percent}
                )
            elif usage_percent > 90:
                return HealthCheckResult(
                    name="disk_space",
                    status=HealthStatus.DEGRADED,
                    message=f"Disk space low on {path} ({usage_percent:.1f}% used)",
                    details={"path": path, "usage_percent": usage_percent}
                )
        
        except Exception:
            continue
    
    return HealthCheckResult(
        name="disk_space",
        status=HealthStatus.HEALTHY,
        message="Disk space healthy"
    )

# Initialize health checker
health_checker = HealthChecker()
health_checker.register_check("database", check_database, timeout=5)
health_checker.register_check("redis", check_redis, timeout=3)
health_checker.register_check("clustering", check_clustering_service, timeout=10)
health_checker.register_check("disk_space", check_disk_space, timeout=5)
```

## Incident Response

### Incident Response Playbook

#### High Error Rate Response

```markdown
# High Error Rate Incident Response

## Immediate Actions (0-5 minutes)
1. **Acknowledge Alert**: Acknowledge the alert to stop notifications
2. **Check Dashboard**: Review Grafana dashboard for error patterns
3. **Check Recent Changes**: Review recent deployments or configuration changes
4. **Scale Resources**: If load-related, scale up application instances

## Investigation (5-15 minutes)
1. **Log Analysis**: 
   ```bash
   # Check recent error logs
   tail -n 100 /var/log/insights/app.log | grep ERROR
   
   # Search for error patterns in ELK
   # Query: level:ERROR AND @timestamp:[now-15m TO now]
   ```

2. **System Health**:
   ```bash
   # Check system resources
   htop
   df -h
   iostat -x 1
   ```

3. **Database Status**:
   ```bash
   # Check database connections
   psql -d insights_prod -c "SELECT count(*) FROM pg_stat_activity;"
   
   # Check for long-running queries
   psql -d insights_prod -c "SELECT pid, query_start, query FROM pg_stat_activity WHERE state = 'active' AND query_start < NOW() - INTERVAL '5 minutes';"
   ```

## Resolution Actions
- **Database Issues**: Restart connections, check for locks
- **Memory Issues**: Restart application, check for memory leaks  
- **High Load**: Scale horizontally, enable caching
- **Code Issues**: Roll back recent deployment, apply hotfix

## Communication
- **Internal**: Update #incidents Slack channel
- **External**: If customer-impacting, update status page
```

#### Service Down Response

```markdown
# Service Down Incident Response

## Immediate Actions (0-2 minutes)
1. **Verify Outage**: Confirm service is actually down
   ```bash
   curl -I http://insights.company.com/api/health
   ```

2. **Check Load Balancer**: Verify upstream servers
   ```bash
   # Check nginx status
   sudo nginx -t
   sudo systemctl status nginx
   
   # Check backend servers
   curl -I http://insights-1:8000/api/health
   curl -I http://insights-2:8000/api/health
   ```

3. **Restart Services**: If simple restart needed
   ```bash
   sudo systemctl restart insights
   ```

## Investigation (2-10 minutes)
1. **Check Logs**:
   ```bash
   # Application logs
   sudo journalctl -u insights -n 50
   
   # System logs
   sudo dmesg | tail -20
   ```

2. **Resource Check**:
   ```bash
   # Check if OOM killed the process
   dmesg | grep -i "killed process"
   
   # Check disk space
   df -h
   ```

3. **Process Status**:
   ```bash
   ps aux | grep insights
   netstat -tlnp | grep :8000
   ```

## Resolution
- **Process Died**: Restart service, investigate root cause
- **Port Conflict**: Kill conflicting process or change port
- **Resource Exhaustion**: Add resources, optimize application
- **Configuration Error**: Fix configuration, restart
```

### Automated Incident Response

```python
#!/usr/bin/env python3
"""
Automated incident response system
"""

import json
import requests
import subprocess
from datetime import datetime
from typing import Dict, List

class IncidentResponder:
    def __init__(self, config):
        self.config = config
        self.slack_webhook = config.get("slack_webhook")
        
    def handle_alert(self, alert_data: Dict):
        """Handle incoming alert and trigger appropriate response"""
        
        alert_name = alert_data.get("alertname")
        severity = alert_data.get("labels", {}).get("severity", "unknown")
        
        print(f"Handling alert: {alert_name} (severity: {severity})")
        
        # Route to appropriate handler
        if alert_name == "HighErrorRate":
            self.handle_high_error_rate(alert_data)
        elif alert_name == "ServiceDown":
            self.handle_service_down(alert_data)
        elif alert_name == "HighMemoryUsage":
            self.handle_high_memory(alert_data)
        elif alert_name == "LowDiskSpace":
            self.handle_low_disk_space(alert_data)
        else:
            self.handle_generic_alert(alert_data)
    
    def handle_high_error_rate(self, alert_data):
        """Handle high error rate alerts"""
        
        # Gather diagnostic information
        diagnostics = {
            "timestamp": datetime.utcnow().isoformat(),
            "alert": alert_data,
            "actions_taken": []
        }
        
        # Check recent deployments
        recent_deployments = self.check_recent_deployments()
        diagnostics["recent_deployments"] = recent_deployments
        
        # Check system resources
        system_status = self.check_system_status()
        diagnostics["system_status"] = system_status
        
        # Automatic remediation based on severity
        severity = alert_data.get("labels", {}).get("severity")
        
        if severity == "critical" and system_status.get("memory_usage", 0) > 90:
            # High memory usage - restart service
            self.restart_service("insights")
            diagnostics["actions_taken"].append("restarted_insights_service")
            
        if len(recent_deployments) > 0:
            # Recent deployment might be the cause
            self.notify_slack(
                f"🚨 High error rate detected after recent deployment. "
                f"Consider rolling back deployment {recent_deployments[0]['id']}"
            )
        
        # Enable detailed logging temporarily
        self.enable_debug_logging(duration_minutes=30)
        diagnostics["actions_taken"].append("enabled_debug_logging")
        
        # Save diagnostics
        self.save_incident_data(alert_data["alertname"], diagnostics)
        
        # Notify team
        self.notify_slack(f"🔥 High error rate incident - automated response initiated")
    
    def handle_service_down(self, alert_data):
        """Handle service down alerts"""
        
        instance = alert_data.get("labels", {}).get("instance")
        
        # Check if service is actually down
        if not self.verify_service_down(instance):
            # False positive - service is actually up
            return
        
        # Attempt automatic restart
        try:
            self.restart_service("insights")
            
            # Wait a moment and verify service is back up
            import time
            time.sleep(10)
            
            if self.check_service_health():
                self.notify_slack(f"✅ Service {instance} automatically restarted and is now healthy")
                return
            
        except Exception as e:
            self.notify_slack(f"❌ Failed to automatically restart service {instance}: {str(e)}")
        
        # If restart didn't work, escalate
        self.escalate_incident(alert_data, "Service restart failed")
    
    def handle_high_memory(self, alert_data):
        """Handle high memory usage"""
        
        # Check for memory leaks in application
        memory_info = self.get_memory_breakdown()
        
        if memory_info.get("insights_app", 0) > 2048:  # > 2GB
            # Application using too much memory
            self.restart_service("insights")
            self.notify_slack("🔄 Restarted Observer Coordinator Insights due to high memory usage")
        
        # Enable memory profiling temporarily
        self.enable_memory_profiling(duration_minutes=60)
    
    def check_system_status(self) -> Dict:
        """Check current system status"""
        try:
            # Get CPU and memory usage
            result = subprocess.run(
                ["top", "-bn1"], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            
            # Parse top output for basic metrics
            # This is simplified - in practice you'd use psutil or similar
            return {"status": "checked", "top_output": result.stdout[:500]}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def restart_service(self, service_name: str):
        """Restart a systemd service"""
        try:
            subprocess.run(
                ["sudo", "systemctl", "restart", service_name],
                check=True,
                timeout=30
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to restart {service_name}: {e}")
            return False
    
    def notify_slack(self, message: str):
        """Send notification to Slack"""
        if not self.slack_webhook:
            return
            
        try:
            requests.post(
                self.slack_webhook,
                json={"text": message},
                timeout=10
            )
        except Exception as e:
            print(f"Failed to send Slack notification: {e}")
    
    def escalate_incident(self, alert_data: Dict, reason: str):
        """Escalate incident to human operators"""
        
        escalation_message = f"""
🚨 INCIDENT ESCALATION REQUIRED 🚨

Alert: {alert_data.get('alertname')}
Reason: {reason}
Time: {datetime.utcnow().isoformat()}

Automated remediation failed. Manual intervention required.
        """
        
        # Send to multiple channels
        self.notify_slack(escalation_message)
        
        # Could also integrate with PagerDuty, email, etc.
        # self.send_pagerduty_alert(alert_data, reason)

# Configuration
incident_config = {
    "slack_webhook": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
}

# Initialize incident responder
responder = IncidentResponder(incident_config)
```

This comprehensive monitoring and operations guide provides the foundation for maintaining a healthy, performant Observer Coordinator Insights deployment in production environments.