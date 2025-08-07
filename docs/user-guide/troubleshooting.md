# Troubleshooting Guide

This comprehensive troubleshooting guide helps you diagnose and resolve common issues with Observer Coordinator Insights. Issues are organized by category with step-by-step solutions and preventive measures.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Data Loading Problems](#data-loading-problems)
3. [Performance Issues](#performance-issues)
4. [Clustering Quality Problems](#clustering-quality-problems)
5. [API and Web Interface Issues](#api-and-web-interface-issues)
6. [Security and Compliance Issues](#security-and-compliance-issues)
7. [Integration Problems](#integration-problems)
8. [Resource and Environment Issues](#resource-and-environment-issues)
9. [Advanced Debugging](#advanced-debugging)
10. [Getting Additional Help](#getting-additional-help)

## Installation Issues

### Issue: pip install fails with dependency conflicts

**Symptoms:**
- Package installation fails with version conflicts
- ImportError when trying to use the system
- ModuleNotFoundError for required dependencies

**Solutions:**

```bash
# Solution 1: Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows
pip install --upgrade pip
pip install observer-coordinator-insights

# Solution 2: Force reinstall with no cache
pip install --no-cache-dir --force-reinstall observer-coordinator-insights

# Solution 3: Install from source
git clone https://github.com/terragon-labs/observer-coordinator-insights.git
cd observer-coordinator-insights
pip install -e .
```

**Prevention:**
- Always use virtual environments
- Keep pip updated: `pip install --upgrade pip`
- Regularly update dependencies: `pip install --upgrade observer-coordinator-insights`

### Issue: Python version incompatibility

**Symptoms:**
- "Python version X.X not supported" error
- Syntax errors on import
- Missing features from newer Python versions

**Solution:**
```bash
# Check Python version
python --version

# Required: Python 3.11+
# If using older version, update Python:

# On Ubuntu/Debian:
sudo apt update
sudo apt install python3.11 python3.11-venv

# On macOS with Homebrew:
brew install python@3.11

# On Windows, download from python.org

# Create virtual environment with specific Python version
python3.11 -m venv venv
source venv/bin/activate
pip install observer-coordinator-insights
```

### Issue: Permission denied errors

**Symptoms:**
- "Permission denied" when installing packages
- Cannot create files or directories
- Access denied to system directories

**Solutions:**
```bash
# Solution 1: Use user installation
pip install --user observer-coordinator-insights

# Solution 2: Fix permissions (Linux/Mac)
sudo chown -R $USER:$USER ~/.local/

# Solution 3: Use virtual environment in user space
python -m venv ~/venv/insights
source ~/venv/insights/bin/activate
pip install observer-coordinator-insights
```

## Data Loading Problems

### Issue: CSV file not found or cannot be read

**Symptoms:**
- "FileNotFoundError: [Errno 2] No such file or directory"
- "PermissionError: [Errno 13] Permission denied"
- Empty results despite providing data file

**Solutions:**
```bash
# Check file existence and permissions
ls -la /path/to/your/file.csv
file /path/to/your/file.csv

# Use absolute path
python src/main.py /full/path/to/employee_data.csv --clusters 4

# Check file permissions
chmod 644 /path/to/your/file.csv

# Verify file format
head -5 /path/to/your/file.csv
```

**File Format Requirements:**
```csv
# Required format - header row must match exactly
employee_id,red_energy,blue_energy,green_energy,yellow_energy
EMP001,45,30,15,10
EMP002,20,40,25,15
```

### Issue: Invalid data format or missing columns

**Symptoms:**
- "KeyError: 'column_name'" 
- "ValueError: could not convert string to float"
- "Missing required columns" error

**Diagnostic Commands:**
```bash
# Validate data format
python -c "
import pandas as pd
data = pd.read_csv('your_file.csv')
print('Columns:', list(data.columns))
print('Data types:', data.dtypes)
print('Sample data:')
print(data.head())
print('Missing values:', data.isnull().sum())
"
```

**Solutions:**
```python
# Fix column names (common issues)
# Wrong: employee_ID, Employee_Id, emp_id
# Correct: employee_id

# Fix data types
# Ensure energy values are numeric (0-100)
# Remove any non-numeric characters except decimals

# Example data cleaning
import pandas as pd
data = pd.read_csv('raw_data.csv')

# Standardize column names
data.columns = [col.lower().replace(' ', '_') for col in data.columns]

# Clean energy values
energy_columns = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
for col in energy_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Remove rows with missing energy values
data = data.dropna(subset=energy_columns)

# Validate energy ranges
for col in energy_columns:
    data[col] = data[col].clip(0, 100)

data.to_csv('cleaned_data.csv', index=False)
```

### Issue: Data quality warnings or validation failures

**Symptoms:**
- "Data quality warning: Energy totals inconsistent"
- "Validation failed: Insufficient data points"
- "Warning: Outliers detected in energy values"

**Diagnostic Analysis:**
```python
# Comprehensive data quality check
import pandas as pd
import numpy as np

def diagnose_data_quality(csv_file):
    data = pd.read_csv(csv_file)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Unique employees: {data['employee_id'].nunique()}")
    
    # Check energy totals
    energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
    data['total_energy'] = data[energy_cols].sum(axis=1)
    
    print(f"Energy total statistics:")
    print(data['total_energy'].describe())
    
    # Identify outliers
    outliers = data[(data['total_energy'] < 90) | (data['total_energy'] > 110)]
    print(f"Outliers (energy total not ~100): {len(outliers)} employees")
    
    # Check for missing values
    missing = data.isnull().sum()
    print(f"Missing values:\n{missing}")
    
    return data

# Run diagnosis
diagnose_data_quality('your_data.csv')
```

**Solutions:**
```bash
# Use built-in data validation
python src/main.py data.csv --validate-only

# Auto-correct common issues
python src/main.py data.csv --auto-correct --clusters 4

# Skip validation (use with caution)
python src/main.py data.csv --skip-validation --clusters 4
```

## Performance Issues

### Issue: Very slow processing or analysis hangs

**Symptoms:**
- Analysis takes much longer than expected
- Process appears to hang or freeze
- High CPU usage for extended periods
- Memory usage continuously increasing

**Diagnostic Steps:**
```bash
# Check system resources
top -p $(pgrep -f "python.*main.py")
# or
htop

# Monitor memory usage
python -m memory_profiler src/main.py data.csv --clusters 4

# Check disk I/O
iotop -o -d 1

# Profile CPU usage
python -m cProfile -o profile_output src/main.py data.csv --clusters 4
python -c "import pstats; pstats.Stats('profile_output').sort_stats('cumulative').print_stats(10)"
```

**Solutions by Dataset Size:**

**Small datasets (<100 employees):**
```bash
# Should process in seconds
python src/main.py data.csv --method esn --clusters 4
# If still slow, check data quality
```

**Medium datasets (100-1000 employees):**
```bash
# Optimize for speed
python src/main.py data.csv \
  --method esn \
  --fast-mode \
  --clusters 4 \
  --timeout 300
```

**Large datasets (1000+ employees):**
```bash
# Use batch processing
python src/main.py data.csv \
  --method esn \
  --batch-size 500 \
  --parallel-workers 4 \
  --clusters 4 \
  --timeout 1800
```

### Issue: Out of memory errors

**Symptoms:**
- "MemoryError" or "Out of Memory" 
- Process killed by system
- Swap usage very high

**Solutions:**
```bash
# Check available memory
free -h

# Solution 1: Reduce batch size
python src/main.py data.csv --batch-size 100 --clusters 4

# Solution 2: Use memory-efficient algorithm
python src/main.py data.csv --method esn --clusters 4

# Solution 3: Process in chunks
python scripts/chunk_processor.py data.csv --chunk-size 200

# Solution 4: Increase virtual memory (Linux)
sudo swapon --show
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Memory Requirements by Dataset Size:**
- 100 employees: ~512MB RAM
- 500 employees: ~1GB RAM  
- 1000 employees: ~2GB RAM
- 5000 employees: ~8GB RAM
- 10000+ employees: ~16GB+ RAM

### Issue: Long startup time or slow imports

**Symptoms:**
- Slow imports of Python modules
- Long delay before processing starts
- "Loading..." messages for extended periods

**Solutions:**
```bash
# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Disable unnecessary features
python src/main.py data.csv \
  --disable-visualization \
  --disable-advanced-metrics \
  --clusters 4

# Precompile modules
python -m compileall src/

# Use faster startup mode
python src/main.py data.csv --fast-startup --clusters 4
```

## Clustering Quality Problems

### Issue: Poor cluster separation (low silhouette score)

**Symptoms:**
- Silhouette score < 0.3
- Clusters appear very similar
- Team formation results seem random

**Diagnostic Analysis:**
```python
# Analyze cluster quality
def diagnose_cluster_quality(results):
    metrics = results['clustering_metrics']
    
    print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
    print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.3f}")
    print(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}")
    
    if metrics['silhouette_score'] < 0.3:
        print("Poor cluster separation detected!")
        print("Recommendations:")
        print("1. Check if data has enough personality diversity")
        print("2. Consider reducing number of clusters") 
        print("3. Try different clustering algorithm")
        print("4. Verify data quality and preprocessing")

# Run with diagnostics
python src/main.py data.csv --clusters 4 --detailed-metrics
```

**Solutions:**
```bash
# Solution 1: Optimize cluster count automatically
python src/main.py data.csv --optimize-clusters

# Solution 2: Try different algorithm
python src/main.py data.csv --method lsm --clusters 4

# Solution 3: Improve preprocessing
python src/main.py data.csv --normalize robust --remove-outliers --clusters 4

# Solution 4: Reduce cluster count
python src/main.py data.csv --clusters 3

# Solution 5: Use stability-focused approach
python src/main.py data.csv --stability-mode --multiple-runs 5 --clusters 4
```

### Issue: Unstable clustering results

**Symptoms:**
- Different results on repeated runs
- Low stability scores
- Inconsistent team formations

**Solutions:**
```bash
# Set random seed for reproducibility
python src/main.py data.csv --random-seed 42 --clusters 4

# Use consensus clustering
python src/main.py data.csv --consensus-clustering --runs 10 --clusters 4

# Increase algorithm stability
python src/main.py data.csv \
  --method esn \
  --stability-mode \
  --clusters 4

# Check data consistency
python src/main.py data.csv --validate-stability --clusters 4
```

### Issue: Clusters don't make business sense

**Symptoms:**
- Cluster interpretations seem random
- No clear personality patterns
- Business stakeholders can't understand results

**Solutions:**
```bash
# Enable interpretability features
python src/main.py data.csv \
  --interpretable-clusters \
  --business-context \
  --clusters 4

# Generate detailed explanations
python src/main.py data.csv \
  --explain-clusters \
  --personality-profiles \
  --clusters 4

# Use domain-specific clustering
python src/main.py data.csv \
  --domain-knowledge insights_discovery \
  --clusters 4

# Manual cluster validation
python scripts/validate_clusters.py results/clustering_results.json
```

## API and Web Interface Issues

### Issue: API server won't start

**Symptoms:**
- "Address already in use" error
- Import errors when starting API
- Server starts but endpoints not accessible

**Solutions:**
```bash
# Check if port is already in use
lsof -i :8000
# or
netstat -tlnp | grep 8000

# Kill existing process
pkill -f uvicorn
# or
sudo kill $(lsof -t -i:8000)

# Start on different port
uvicorn src.api.main:app --host 0.0.0.0 --port 8080

# Check firewall settings
sudo ufw status
sudo ufw allow 8000

# Verify API is responding
curl http://localhost:8000/api/health
```

### Issue: File upload fails through web interface

**Symptoms:**
- Upload button doesn't work
- "File too large" errors
- Upload succeeds but analysis fails

**Solutions:**
```bash
# Check file size limits
# Default max file size: 10MB
# Increase if needed:
export MAX_FILE_SIZE=50MB
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Check file permissions
chmod 644 your_data.csv

# Test upload via curl
curl -X POST \
  -F "file=@your_data.csv" \
  http://localhost:8000/api/analytics/upload

# Check browser console for JavaScript errors
# Open browser DevTools (F12) and check Console tab
```

### Issue: API responses are slow or timeout

**Symptoms:**
- API requests take very long time
- Timeout errors (504 Gateway Timeout)
- Requests hang without response

**Solutions:**
```bash
# Increase timeout settings
uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --timeout-keep-alive 300

# Use async processing for large datasets
curl -X POST \
  -F "file=@large_dataset.csv" \
  http://localhost:8000/api/analytics/upload?async=true

# Check API performance
curl -w "Time: %{time_total}s\n" \
  http://localhost:8000/api/health

# Monitor API logs
tail -f logs/api.log
```

## Security and Compliance Issues

### Issue: Secure mode fails or compliance warnings

**Symptoms:**
- "Security validation failed" errors
- Compliance warnings in logs
- Audit log creation fails

**Solutions:**
```bash
# Check secure mode requirements
python src/main.py data.csv --check-security-requirements

# Fix common security issues
chmod 600 /path/to/encryption/keys
export INSIGHTS_SECURITY_SALT="your-secure-salt-here"

# Enable comprehensive auditing
python src/main.py data.csv \
  --secure-mode \
  --audit-log /var/log/insights/audit.log \
  --encryption-key /etc/insights/keys/encryption.key

# Validate compliance configuration
python scripts/validate_compliance.py --mode gdpr
```

### Issue: PII anonymization not working

**Symptoms:**
- Personal information visible in outputs
- Anonymization warnings
- Audit failures

**Solutions:**
```bash
# Force anonymization
python src/main.py data.csv \
  --secure-mode \
  --force-anonymization \
  --anonymization-salt "your-unique-salt"

# Verify anonymization
python scripts/verify_anonymization.py output/results.json

# Custom anonymization configuration
cat > anonymization_config.json << EOF
{
  "employee_id": {"method": "hash", "salt": "your-salt"},
  "remove_columns": ["name", "email", "manager"],
  "anonymize_patterns": ["phone", "address"]
}
EOF

python src/main.py data.csv \
  --secure-mode \
  --anonymization-config anonymization_config.json
```

## Integration Problems

### Issue: HR system integration failures

**Symptoms:**
- Authentication failures with HR systems
- Data sync errors
- API connection timeouts

**Solutions for Common HR Systems:**

**Workday Integration:**
```python
# Test Workday connection
from insights_clustering.integrations import WorkdayConnector

connector = WorkdayConnector(
    base_url="https://your-company.workday.com",
    username="your-username",
    password="your-password",
    tenant="your-tenant"
)

# Test connection
try:
    connector.test_connection()
    print("Workday connection successful")
except Exception as e:
    print(f"Connection failed: {e}")
```

**SAP SuccessFactors:**
```python
# Test SAP connection
from insights_clustering.integrations import SuccessFactorsConnector

connector = SuccessFactorsConnector(
    api_url="https://api4.successfactors.com",
    company_id="your-company-id",
    username="your-username", 
    password="your-password"
)

# Test API access
try:
    data = connector.get_employee_data(limit=10)
    print(f"Retrieved {len(data)} test records")
except Exception as e:
    print(f"API access failed: {e}")
```

### Issue: Database connection problems

**Symptoms:**
- "Connection refused" errors
- Database timeout errors
- Authentication failures

**Solutions:**
```bash
# Test database connectivity
python -c "
from src.database.connection import test_connection
try:
    test_connection()
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
"

# Check database configuration
cat config/database.yml

# Reset database connection
python scripts/reset_database.py

# Initialize database tables
python scripts/init_database.py
```

## Resource and Environment Issues

### Issue: Docker container problems

**Symptoms:**
- Container fails to start
- "No space left on device" errors
- Permission denied in container

**Solutions:**
```bash
# Check Docker resources
docker system df
docker system prune -f

# Rebuild container with more resources
docker build --memory=4g --cpus=2.0 -t observer-coordinator-insights .

# Run with increased limits
docker run --memory=4g --cpus=2.0 \
  -v $(pwd)/data:/app/data \
  -p 8000:8000 \
  observer-coordinator-insights

# Check container logs
docker logs observer-coordinator-insights

# Interactive debugging
docker run -it --entrypoint /bin/bash observer-coordinator-insights
```

### Issue: Kubernetes deployment problems

**Symptoms:**
- Pods failing to start
- Resource quota exceeded
- Service not accessible

**Solutions:**
```bash
# Check pod status
kubectl get pods -l app=observer-coordinator-insights

# Check pod logs
kubectl logs -l app=observer-coordinator-insights

# Describe failing pods
kubectl describe pod <pod-name>

# Check resource quotas
kubectl describe resourcequota

# Scale deployment
kubectl scale deployment observer-coordinator-insights --replicas=2

# Check service connectivity
kubectl get svc observer-coordinator-insights
kubectl port-forward svc/observer-coordinator-insights 8000:8000
```

## Advanced Debugging

### Enable Debug Logging

```bash
# Maximum logging detail
python src/main.py data.csv \
  --log-level DEBUG \
  --log-file debug.log \
  --clusters 4

# Component-specific logging
export INSIGHTS_LOG_CLUSTERING=DEBUG
export INSIGHTS_LOG_TEAMS=INFO
export INSIGHTS_LOG_API=DEBUG

python src/main.py data.csv --clusters 4
```

### Performance Profiling

```bash
# CPU profiling
python -m cProfile -o cpu_profile.prof src/main.py data.csv --clusters 4
python -c "
import pstats
stats = pstats.Stats('cpu_profile.prof')
stats.sort_stats('cumulative')
stats.print_stats(20)
"

# Memory profiling
pip install memory_profiler
python -m memory_profiler src/main.py data.csv --clusters 4

# Line-by-line profiling
pip install line_profiler
kernprof -l -v src/main.py data.csv --clusters 4
```

### Network Debugging

```bash
# Check API connectivity
curl -v http://localhost:8000/api/health

# Monitor network traffic
sudo tcpdump -i any port 8000

# Test API endpoints
curl -X GET http://localhost:8000/api/analytics/status
curl -X POST -F "file=@test.csv" http://localhost:8000/api/analytics/upload
```

### System Resource Monitoring

```bash
# Real-time resource monitoring
watch -n 1 "ps aux | grep python | head -10"

# Memory usage over time
while true; do
  ps -o pid,vsz,rss,pmem,comm -p $(pgrep -f "python.*main.py")
  sleep 5
done

# Disk I/O monitoring
iostat -x 1

# Network monitoring
iftop
```

## Getting Additional Help

### Information to Gather Before Seeking Help

```bash
# System information
echo "System Info:"
uname -a
python --version
pip list | grep observer-coordinator-insights

echo -e "\nData Info:"
wc -l your_data.csv
head -3 your_data.csv

echo -e "\nError Info:"
python src/main.py your_data.csv --clusters 4 2>&1 | tail -20

echo -e "\nResource Info:"
free -h
df -h
```

### Diagnostic Data Collection

```bash
# Generate comprehensive diagnostic report
python scripts/generate_diagnostic_report.py \
  --data-file your_data.csv \
  --output diagnostic_report.zip

# The report includes:
# - System configuration
# - Data quality analysis  
# - Error logs
# - Performance metrics
# - Configuration files
```

### Support Channels

1. **GitHub Issues**: [Report bugs and feature requests](https://github.com/terragon-labs/observer-coordinator-insights/issues)
2. **Stack Overflow**: Tag `observer-coordinator-insights`
3. **Community Forum**: [discussions.terragon-labs.com](https://discussions.terragon-labs.com)
4. **Enterprise Support**: [support@terragon-labs.com](mailto:support@terragon-labs.com)

### Before Contacting Support

Please ensure you have:
- [ ] Checked this troubleshooting guide
- [ ] Searched existing GitHub issues
- [ ] Generated diagnostic report
- [ ] Tested with sample data
- [ ] Documented steps to reproduce the issue

### Emergency Support

For critical production issues:
- **Enterprise customers**: Use priority support channel
- **Open source users**: Create GitHub issue with "urgent" tag
- **Security issues**: Email security@terragon-labs.com

Remember to never include sensitive data in public support requests. Use the diagnostic report feature which automatically anonymizes sensitive information.