# 🚀 Production Deployment Guide - Observer Coordinator Insights

## 📋 Quick Start Checklist

- [ ] Python 3.9+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Database configured (optional)
- [ ] Environment variables set
- [ ] Quality gates passed
- [ ] Production deployment verified

## 🔧 Environment Setup

### 1. System Requirements
```bash
# Minimum Requirements
Python 3.9+
RAM: 4GB minimum, 8GB recommended
CPU: 2 cores minimum, 4 cores recommended  
Disk: 2GB available space

# Recommended Production Environment
Python 3.11+
RAM: 16GB+
CPU: 8 cores+
Disk: 20GB+ SSD
```

### 2. Virtual Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies
```bash
# Core dependencies
pip install -e .

# Additional dependencies for full functionality
pip install fastapi uvicorn[standard]
pip install sqlalchemy alembic aiosqlite
pip install pytest pytest-cov pytest-asyncio

# Optional: Development tools
pip install ruff mypy bandit safety
```

## 🏃‍♀️ Running the Application

### Command Line Interface (Primary)
```bash
# Basic clustering analysis
python src/main.py sample_data.csv --clusters 4 --teams 3

# Secure mode with enhanced security features
python src/main.py sample_data.csv --secure-mode --clusters 4

# Find optimal cluster count
python src/main.py sample_data.csv --optimize-clusters

# Advanced options
python src/main.py sample_data.csv \
  --clusters 4 \
  --teams 3 \
  --secure-mode \
  --log-level DEBUG \
  --timeout 600 \
  --output results/
```

### REST API Server (Optional)
```bash
# Start API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Production deployment with workers
uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --access-log

# API available at: http://localhost:8000
# Documentation: http://localhost:8000/docs
```

## 📊 Sample Data Format

### Input CSV Structure
```csv
employee_id,red_energy,blue_energy,green_energy,yellow_energy
EMP001,30,20,25,25
EMP002,25,30,25,20
EMP003,35,15,25,25
```

### Required Columns
- `employee_id`: Unique identifier for each employee
- `red_energy`: Assertive/Direct energy (0-100)
- `blue_energy`: Analytical/Thinking energy (0-100)  
- `green_energy`: Caring/Supporting energy (0-100)
- `yellow_energy`: Enthusiastic/Inspiring energy (0-100)

**Note**: Energy values should sum to approximately 100 per employee

## 🔒 Security Configuration

### Secure Mode Features
```bash
# Enable secure mode for production
python src/main.py data.csv --secure-mode
```

**Secure Mode Includes**:
- ✅ PII data anonymization
- ✅ Input validation and sanitization
- ✅ Audit logging for all operations
- ✅ Enhanced error handling
- ✅ Security compliance checks

### Environment Variables
```bash
# Optional security configuration
export INSIGHTS_SECURITY_SALT="your-secret-salt"
export INSIGHTS_LOG_LEVEL="INFO"
export INSIGHTS_AUDIT_ENABLED="true"
export INSIGHTS_DATA_RETENTION_DAYS="180"
```

## 🧪 Quality Validation

### Pre-deployment Testing
```bash
# Run comprehensive quality gates
python scripts/run_quality_gates.py

# Individual test suites
python -m pytest tests/unit/ -v              # Unit tests
python -m pytest tests/integration/ -v       # Integration tests
python -m pytest tests/security/ -v          # Security tests

# Code quality checks
ruff check src/                               # Linting
mypy src/ --ignore-missing-imports           # Type checking
bandit -r src/                               # Security scan
```

### Expected Results
- ✅ All unit tests pass
- ✅ Core functionality validated
- ✅ Security tests pass
- ✅ Performance benchmarks meet SLAs
- ✅ No critical security vulnerabilities

## 📈 Performance Optimization

### Small Datasets (< 1,000 employees)
```bash
# Standard processing - fastest startup
python src/main.py data.csv --clusters 4
```

### Large Datasets (1,000+ employees)
```bash
# Enable performance optimizations
python src/main.py data.csv \
  --clusters 4 \
  --timeout 1800 \
  --log-level WARNING
```

### Memory-Constrained Environments
```bash
# Process in smaller batches
python src/main.py data.csv \
  --clusters 4 \
  --secure-mode \
  --log-level ERROR
```

## 🐳 Docker Deployment (Optional)

### Build Container
```bash
# Build image
docker build -t observer-coordinator-insights .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  observer-coordinator-insights
```

### Docker Compose
```yaml
version: '3.8'
services:
  insights:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./output:/app/output
    environment:
      - INSIGHTS_LOG_LEVEL=INFO
      - INSIGHTS_SECURITY_SALT=production-salt
```

## 📁 Output Files

### Generated Reports
```
output/
├── clustering_results.json      # Clustering analysis results
├── team_compositions.json       # Team recommendations  
├── validation_report.json       # Data quality report
├── cluster_optimization.json    # Optimal cluster analysis
└── audit_log.json              # Security audit trail
```

### File Descriptions
- **clustering_results.json**: Cluster centroids, quality metrics, assignments
- **team_compositions.json**: Recommended team structures with balance scores
- **validation_report.json**: Data quality assessment and warnings
- **cluster_optimization.json**: Analysis of optimal cluster counts
- **audit_log.json**: Security and access audit trail (secure mode)

## 🔍 Monitoring & Troubleshooting

### Log Levels
```bash
# Debug mode for troubleshooting
python src/main.py data.csv --log-level DEBUG

# Production logging
python src/main.py data.csv --log-level INFO --log-file app.log
```

### Common Issues & Solutions

#### Issue: Import Errors
```bash
# Solution: Ensure virtual environment is activated
source venv/bin/activate
pip install -e .
```

#### Issue: File Not Found
```bash
# Solution: Check file path and permissions
ls -la data.csv
python src/main.py ./path/to/data.csv
```

#### Issue: Memory Errors
```bash
# Solution: Use smaller datasets or increase system memory
# For large datasets, consider chunking
head -1000 large_data.csv > sample_data.csv
python src/main.py sample_data.csv
```

#### Issue: Clustering Fails
```bash
# Solution: Validate data format and quality
python src/main.py data.csv --validate-only
```

## 🚀 Production Best Practices

### Security
- ✅ Always use `--secure-mode` in production
- ✅ Regularly rotate security salts
- ✅ Monitor audit logs for anomalies
- ✅ Validate data sources and formats
- ✅ Implement access controls

### Performance  
- ✅ Monitor system resources during processing
- ✅ Use appropriate timeout values
- ✅ Enable caching for repeated analyses
- ✅ Consider parallel processing for large datasets
- ✅ Regular performance benchmarking

### Reliability
- ✅ Run quality gates before deployment
- ✅ Implement health checks and monitoring
- ✅ Maintain backup and recovery procedures
- ✅ Document operational procedures
- ✅ Set up alerting for failures

## 📞 Support & Maintenance

### Health Checks
```bash
# Basic functionality test
python -c "from src.main import main; print('✅ System healthy')"

# Performance benchmark
python scripts/run_quality_gates.py

# API health (if running)
curl http://localhost:8000/health
```

### Maintenance Tasks
```bash
# Clear cache (if performance issues)
rm -rf ~/.observer_coordinator_cache/

# Update dependencies
pip install --upgrade -r requirements.txt

# Database migrations (if using database features)
alembic upgrade head
```

### Getting Help
1. **Check logs**: Review application logs for error details
2. **Run diagnostics**: Use quality gates script for system validation
3. **Validate data**: Ensure input data meets format requirements
4. **Check resources**: Verify sufficient memory and CPU availability
5. **Documentation**: Refer to API documentation at `/docs` endpoint

---

## 🎯 Success Criteria

Your deployment is successful when:
- ✅ Quality gates pass without critical errors
- ✅ Application processes sample data without errors
- ✅ Output files are generated with valid content
- ✅ Performance meets expected benchmarks
- ✅ Security features function as expected

**The Observer Coordinator Insights platform is now ready for production use!** 🚀