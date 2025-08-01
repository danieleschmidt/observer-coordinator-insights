# Incident Response Runbook

## Purpose

This runbook provides step-by-step procedures for responding to incidents affecting the Observer Coordinator Insights application.

## Prerequisites

- Access to monitoring dashboards (Grafana)
- Access to application logs (Docker logs or log aggregation system)
- SSH/kubectl access to production environment
- Incident management system access (PagerDuty/Slack)

## Incident Classification

### P1 - Critical
- **Response Time**: Immediate (< 15 minutes)
- **Examples**: Application down, data loss, security breach
- **Actions**: Page on-call engineer, notify management

### P2 - High
- **Response Time**: 1 hour
- **Examples**: Performance degradation, partial outage
- **Actions**: Notify on-call engineer via Slack

### P3 - Medium
- **Response Time**: 4 hours
- **Examples**: Minor issues, non-critical failures
- **Actions**: Create ticket, schedule for next business day

## Response Procedures

### Step 1: Initial Response (0-5 minutes)

1. **Acknowledge the Alert**
   ```bash
   # If using PagerDuty
   - Click "Acknowledge" in PagerDuty app
   - Post in #incidents Slack channel: "Acknowledging incident [ID]"
   ```

2. **Assess Severity**
   - Check monitoring dashboard: https://grafana.company.com
   - Determine customer impact
   - Classify incident severity (P1-P3)

3. **Create Incident**
   ```bash
   # Create incident ticket
   - Open incident management system
   - Create new incident with severity level
   - Add initial description and impact assessment
   ```

### Step 2: Investigation (5-30 minutes)

1. **Check System Health**
   ```bash
   # Application health check
   curl -f https://insights.company.com/health
   
   # Response should be: {"status": "healthy", "timestamp": "..."}
   ```

2. **Review Monitoring Dashboard**
   - Open Grafana dashboard
   - Check for metric anomalies in last 1-4 hours
   - Look for error rate spikes, memory issues, CPU usage

3. **Examine Recent Changes**
   ```bash
   # Check recent deployments
   kubectl get deployments -n insights --sort-by='.metadata.creationTimestamp'
   
   # Review recent commits
   git log --oneline --since="4 hours ago"
   ```

4. **Analyze Logs**
   ```bash
   # View recent application logs
   docker-compose logs -f --tail=500 app
   
   # Search for errors
   docker-compose logs app | grep -i error | tail -20
   
   # Check for specific error patterns
   docker-compose logs app | grep -E "(exception|traceback|fatal)" | tail -10
   ```

### Step 3: Mitigation (30+ minutes)

#### For Application Downtime

1. **Quick Health Restoration**
   ```bash
   # Restart application
   docker-compose restart app
   
   # Wait 30 seconds and test
   sleep 30
   curl -f https://insights.company.com/health
   ```

2. **If Restart Doesn't Help**
   ```bash
   # Check resource usage
   docker stats
   
   # Check disk space
   df -h
   
   # Check memory usage
   free -m
   ```

3. **Rollback if Recent Deployment**
   ```bash
   # Rollback to previous version
   kubectl rollout undo deployment/insights-app -n insights
   
   # Monitor rollback
   kubectl rollout status deployment/insights-app -n insights
   ```

#### For Performance Issues

1. **Check Resource Constraints**
   ```bash
   # CPU and memory usage
   docker stats insights-app
   
   # If high CPU, check for runaway processes
   docker exec insights-app top
   
   # If high memory, check for memory leaks
   docker exec insights-app python -c "
   import psutil
   process = psutil.Process()
   print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
   "
   ```

2. **Scale Application**
   ```bash
   # Scale up replicas
   docker-compose up -d --scale app=3
   
   # Or with Kubernetes
   kubectl scale deployment insights-app --replicas=3 -n insights
   ```

#### For Database Issues

1. **Check Database Connectivity**
   ```bash
   # Test database connection
   docker-compose exec app python -c "
   import psycopg2
   try:
       conn = psycopg2.connect('postgresql://user:pass@db:5432/insights')
       print('Database connection: OK')
   except Exception as e:
       print(f'Database connection failed: {e}')
   "
   ```

2. **Check Database Health**
   ```bash
   # PostgreSQL specific checks
   docker-compose exec postgres psql -U insights -c "SELECT version();"
   
   # Check active connections
   docker-compose exec postgres psql -U insights -c "
   SELECT count(*) as active_connections 
   FROM pg_stat_activity 
   WHERE state = 'active';
   "
   ```

### Step 4: Communication

1. **Update Stakeholders**
   ```markdown
   # Slack update template
   **Incident Update - [Incident ID]**
   
   **Status**: Investigating/Mitigating/Resolved
   **Impact**: Brief description of customer impact
   **Next Update**: In 30 minutes
   
   **Actions Taken**:
   - List of actions performed
   
   **Next Steps**:
   - Planned actions
   ```

2. **Customer Communication** (for customer-facing issues)
   ```markdown
   # Status page update template
   **Service Degradation - Clustering Analysis**
   
   We are currently experiencing [brief description]. 
   Our team is actively working to resolve this issue.
   
   **Impact**: Users may experience...
   **ETA**: We expect resolution within...
   
   Next update in 30 minutes.
   ```

### Step 5: Resolution and Follow-up

1. **Verify Resolution**
   ```bash
   # Test all critical paths
   curl -f https://insights.company.com/health
   curl -f https://insights.company.com/health/ready
   
   # Test key functionality
   # Run integration test suite
   pytest tests/integration/ -v
   ```

2. **Monitor for Stability**
   - Watch dashboards for 30 minutes
   - Ensure no error rate increases
   - Verify normal resource usage

3. **Update Incident**
   - Mark incident as resolved
   - Add resolution summary
   - Schedule post-incident review

## Post-Incident Review

### Within 24 Hours

1. **Schedule Review Meeting**
   - Include all responders
   - 1-hour meeting within 24 hours
   - Blameless culture - focus on systems

2. **Document Timeline**
   ```markdown
   ## Incident Timeline
   
   - **Time**: Action taken
   - **12:00**: Alert fired for high error rate
   - **12:05**: On-call acknowledged, began investigation
   - **12:15**: Identified root cause as memory leak
   - **12:30**: Restarted application, error rate normalized
   - **13:00**: Confirmed stable, incident resolved
   ```

3. **Root Cause Analysis**
   - What happened?
   - Why did it happen?
   - How did we detect it?
   - How did we resolve it?
   - What can we do to prevent it?

### Follow-up Actions

1. **Preventive Measures**
   - Code fixes
   - Monitoring improvements
   - Process changes
   - Documentation updates

2. **Track Action Items**
   - Assign owners
   - Set due dates
   - Review in next team meeting

## Common Scenarios

### Scenario 1: Application Won't Start

**Symptoms**: Health check fails, application logs show startup errors

**Investigation**:
```bash
# Check startup logs
docker-compose logs app

# Common issues to look for:
# - Missing environment variables
# - Database connection failures
# - Port conflicts
# - File permission issues
```

**Resolution**:
```bash
# Fix environment variables
# Update .env file with correct values

# Fix database connection
docker-compose up -d postgres
# Wait for DB to be ready
sleep 30

# Fix permissions
sudo chown -R app:app /app/data

# Restart application
docker-compose restart app
```

### Scenario 2: High Memory Usage

**Symptoms**: Memory usage > 80%, potential OOM kills

**Investigation**:
```bash
# Check memory usage
docker stats insights-app

# Profile memory usage
docker exec insights-app python -m memory_profiler src/main.py
```

**Resolution**:
```bash
# Immediate: Restart to free memory
docker-compose restart app

# Short-term: Increase memory limits
# Edit docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G

# Long-term: Investigate memory leaks
# Add memory profiling to CI/CD
```

### Scenario 3: Database Connection Pool Exhausted

**Symptoms**: "ConnectionPool exhausted" errors

**Investigation**:
```bash
# Check active connections
docker-compose exec postgres psql -U insights -c "
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';
"

# Check connection pool settings
grep -i pool .env
```

**Resolution**:
```bash
# Increase pool size
# Update environment variable
DB_POOL_SIZE=20

# Restart application
docker-compose restart app

# Monitor connection usage
```

## Escalation Contacts

- **Primary On-call**: +1-xxx-xxx-xxxx
- **Secondary On-call**: +1-xxx-xxx-xxxx
- **Tech Lead**: +1-xxx-xxx-xxxx
- **DevOps Manager**: +1-xxx-xxx-xxxx
- **Security Team**: security@company.com

## Tools and Resources

- **Monitoring**: https://grafana.company.com
- **Logs**: https://logs.company.com
- **Status Page**: https://status.company.com
- **Incident Management**: https://pagerduty.com
- **Documentation**: https://wiki.company.com
- **Code Repository**: https://github.com/company/insights