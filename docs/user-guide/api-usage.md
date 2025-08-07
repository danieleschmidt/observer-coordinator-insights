# API Usage Guide

This guide provides comprehensive documentation for using the Observer Coordinator Insights REST API. The API enables programmatic access to neuromorphic clustering capabilities, team formation algorithms, and organizational analytics.

## Table of Contents

1. [API Overview](#api-overview)
2. [Authentication & Security](#authentication--security)
3. [Analytics Endpoints](#analytics-endpoints)
4. [Team Formation Endpoints](#team-formation-endpoints)
5. [Health & Admin Endpoints](#health--admin-endpoints)
6. [Data Models](#data-models)
7. [Error Handling](#error-handling)
8. [Code Examples](#code-examples)
9. [Rate Limiting & Performance](#rate-limiting--performance)
10. [Integration Patterns](#integration-patterns)

## API Overview

### Base URL
```
Production: https://api.observer-coordinator-insights.com
Development: http://localhost:8000
```

### API Version
Current version: `v1`
All endpoints are prefixed with `/api/`

### Content Types
- Request: `application/json`, `multipart/form-data` (file uploads)
- Response: `application/json`

### Quick Start

```bash
# Start local API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Test API health
curl http://localhost:8000/api/health

# View interactive documentation
open http://localhost:8000/docs
```

## Authentication & Security

### API Key Authentication

```python
import requests

headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
}

response = requests.get(
    'http://localhost:8000/api/analytics/status',
    headers=headers
)
```

### Secure Mode Configuration

```python
# Enable secure mode for enterprise deployments
secure_headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'X-Security-Mode': 'enabled',
    'X-Compliance-Mode': 'gdpr',  # gdpr, ccpa, pdpa
    'Content-Type': 'application/json'
}
```

### Data Encryption

```python
# Request with encryption enabled
response = requests.post(
    'http://localhost:8000/api/analytics/upload',
    headers={
        'Authorization': 'Bearer YOUR_API_KEY',
        'X-Encrypt-Data': 'true',
        'X-Encryption-Algorithm': 'AES-256'
    },
    files={'file': open('employee_data.csv', 'rb')}
)
```

## Analytics Endpoints

### Upload and Analyze Data

**POST** `/api/analytics/upload`

Upload CSV data and perform neuromorphic clustering analysis.

#### Request
```python
import requests

# File upload with parameters
files = {'file': open('employee_data.csv', 'rb')}
data = {
    'n_clusters': 4,
    'method': 'hybrid_reservoir',  # esn, snn, lsm, hybrid_reservoir
    'language': 'en',              # en, de, es, fr, ja, zh
    'secure_mode': True,
    'optimize_clusters': False
}

response = requests.post(
    'http://localhost:8000/api/analytics/upload',
    files=files,
    data=data,
    headers={'Authorization': 'Bearer YOUR_API_KEY'}
)
```

#### Response
```json
{
    "job_id": "job_123456789",
    "status": "processing",
    "message": "Analysis started successfully",
    "estimated_completion": "2025-01-15T14:30:00Z",
    "employee_count": 150,
    "clusters_requested": 4,
    "method": "hybrid_reservoir"
}
```

### Get Analysis Results

**GET** `/api/analytics/results/{job_id}`

Retrieve completed analysis results.

#### Request
```python
response = requests.get(
    'http://localhost:8000/api/analytics/results/job_123456789',
    headers={'Authorization': 'Bearer YOUR_API_KEY'}
)
```

#### Response
```json
{
    "job_id": "job_123456789",
    "status": "completed",
    "completion_time": "2025-01-15T14:28:45Z",
    "processing_time_seconds": 127,
    "results": {
        "cluster_count": 4,
        "employee_count": 150,
        "clustering_metrics": {
            "silhouette_score": 0.742,
            "calinski_harabasz_score": 89.45,
            "davies_bouldin_score": 0.681,
            "stability_score": 0.892
        },
        "clusters": [
            {
                "cluster_id": 0,
                "name": "Analytical Leaders",
                "employee_count": 38,
                "dominant_energy": "blue",
                "characteristics": {
                    "red_energy": 0.23,
                    "blue_energy": 0.45,
                    "green_energy": 0.21,
                    "yellow_energy": 0.11
                },
                "personality_profile": "Detail-oriented systematic thinkers",
                "ideal_roles": ["Data Analyst", "Project Manager"],
                "employees": ["emp_001", "emp_045", "emp_078"]
            }
        ],
        "visualization_url": "/api/analytics/visualization/job_123456789"
    }
}
```

### Get Analysis Status

**GET** `/api/analytics/status/{job_id}`

Check the status of a running analysis.

#### Request
```python
response = requests.get(
    'http://localhost:8000/api/analytics/status/job_123456789',
    headers={'Authorization': 'Bearer YOUR_API_KEY'}
)
```

#### Response
```json
{
    "job_id": "job_123456789",
    "status": "processing",  # queued, processing, completed, failed
    "progress": 0.65,
    "current_stage": "neuromorphic_clustering",
    "estimated_remaining_seconds": 45,
    "message": "Running hybrid reservoir clustering algorithm"
}
```

### List Analysis Jobs

**GET** `/api/analytics/jobs`

List all analysis jobs for the authenticated user.

#### Request
```python
params = {
    'status': 'completed',  # Optional: filter by status
    'limit': 10,            # Optional: limit results
    'offset': 0             # Optional: pagination offset
}

response = requests.get(
    'http://localhost:8000/api/analytics/jobs',
    params=params,
    headers={'Authorization': 'Bearer YOUR_API_KEY'}
)
```

#### Response
```json
{
    "jobs": [
        {
            "job_id": "job_123456789",
            "status": "completed",
            "created_at": "2025-01-15T14:00:00Z",
            "completed_at": "2025-01-15T14:28:45Z",
            "employee_count": 150,
            "clusters": 4,
            "method": "hybrid_reservoir"
        }
    ],
    "total": 25,
    "limit": 10,
    "offset": 0
}
```

### Delete Analysis Job

**DELETE** `/api/analytics/jobs/{job_id}`

Delete an analysis job and its results.

#### Request
```python
response = requests.delete(
    'http://localhost:8000/api/analytics/jobs/job_123456789',
    headers={'Authorization': 'Bearer YOUR_API_KEY'}
)
```

#### Response
```json
{
    "message": "Analysis job deleted successfully",
    "job_id": "job_123456789"
}
```

## Team Formation Endpoints

### Generate Team Compositions

**POST** `/api/teams/generate`

Generate optimal team compositions from clustering results.

#### Request
```python
data = {
    "job_id": "job_123456789",  # Reference to clustering results
    "num_teams": 3,
    "team_size_min": 4,
    "team_size_max": 8,
    "balance_strategy": "energy_balanced",  # energy_balanced, skill_complementary, mixed
    "constraints": {
        "departments": {
            "engineering": {"min": 1, "max": 3},
            "design": {"min": 1, "max": 2}
        },
        "experience_levels": {
            "senior": {"min": 1},
            "junior": {"max": 2}
        }
    }
}

response = requests.post(
    'http://localhost:8000/api/teams/generate',
    json=data,
    headers={'Authorization': 'Bearer YOUR_API_KEY'}
)
```

#### Response
```json
{
    "team_generation_id": "teams_123456789",
    "status": "completed",
    "generation_time_seconds": 2.3,
    "teams": [
        {
            "team_id": "team_1",
            "members": [
                {
                    "employee_id": "emp_001",
                    "cluster_id": 0,
                    "role_in_team": "analytical_lead",
                    "energy_profile": {
                        "red": 0.25,
                        "blue": 0.45,
                        "green": 0.20,
                        "yellow": 0.10
                    }
                }
            ],
            "team_size": 5,
            "balance_score": 0.87,
            "energy_distribution": {
                "red": 0.24,
                "blue": 0.28,
                "green": 0.26,
                "yellow": 0.22
            },
            "predicted_performance": "high",
            "strengths": ["Analytical depth", "Collaborative harmony"],
            "potential_challenges": ["May need more assertiveness"],
            "recommended_projects": ["Complex analysis", "Strategic planning"]
        }
    ],
    "overall_metrics": {
        "average_balance_score": 0.84,
        "energy_coverage": 0.92,
        "conflict_risk": "low"
    }
}
```

### Validate Team Composition

**POST** `/api/teams/validate`

Validate and score a specific team composition.

#### Request
```python
data = {
    "team_members": ["emp_001", "emp_045", "emp_078", "emp_112"],
    "clustering_job_id": "job_123456789",
    "validation_metrics": [
        "balance_score",
        "conflict_risk",
        "performance_prediction",
        "communication_compatibility"
    ]
}

response = requests.post(
    'http://localhost:8000/api/teams/validate',
    json=data,
    headers={'Authorization': 'Bearer YOUR_API_KEY'}
)
```

#### Response
```json
{
    "validation_id": "validation_123456789",
    "team_composition": {
        "members": ["emp_001", "emp_045", "emp_078", "emp_112"],
        "size": 4
    },
    "scores": {
        "balance_score": 0.78,
        "conflict_risk": 0.15,
        "performance_prediction": 0.82,
        "communication_compatibility": 0.74
    },
    "recommendations": [
        "Consider adding a member with higher yellow energy for creativity",
        "Strong analytical capabilities present",
        "Good collaborative potential"
    ],
    "risk_factors": [
        "Potential for over-analysis without sufficient action orientation"
    ]
}
```

### Get Team Formation History

**GET** `/api/teams/history`

Retrieve historical team formation results.

#### Request
```python
params = {
    'limit': 20,
    'include_metrics': True
}

response = requests.get(
    'http://localhost:8000/api/teams/history',
    params=params,
    headers={'Authorization': 'Bearer YOUR_API_KEY'}
)
```

## Health & Admin Endpoints

### System Health Check

**GET** `/api/health`

Check system health and availability.

#### Request
```python
response = requests.get('http://localhost:8000/api/health')
```

#### Response
```json
{
    "status": "healthy",
    "timestamp": "2025-01-15T14:30:00Z",
    "version": "4.0.0",
    "components": {
        "database": "healthy",
        "clustering_engine": "healthy",
        "team_formation": "healthy",
        "file_storage": "healthy"
    },
    "metrics": {
        "active_jobs": 3,
        "completed_jobs_today": 47,
        "average_response_time_ms": 145,
        "uptime_seconds": 86400
    }
}
```

### System Metrics

**GET** `/api/admin/metrics`

Get detailed system metrics (admin access required).

#### Request
```python
response = requests.get(
    'http://localhost:8000/api/admin/metrics',
    headers={
        'Authorization': 'Bearer YOUR_ADMIN_API_KEY',
        'X-Admin-Access': 'true'
    }
)
```

#### Response
```json
{
    "performance": {
        "cpu_usage": 0.45,
        "memory_usage": 0.62,
        "disk_usage": 0.28,
        "active_connections": 12
    },
    "processing": {
        "jobs_queued": 2,
        "jobs_processing": 3,
        "average_processing_time_seconds": 127,
        "throughput_jobs_per_hour": 85
    },
    "errors": {
        "error_rate": 0.02,
        "last_error": "2025-01-15T12:15:00Z",
        "common_errors": [
            {"type": "validation_error", "count": 5},
            {"type": "timeout_error", "count": 2}
        ]
    }
}
```

### Configuration Management

**GET** `/api/admin/config`

Get current system configuration (admin access required).

**PUT** `/api/admin/config`

Update system configuration (admin access required).

#### Request (GET)
```python
response = requests.get(
    'http://localhost:8000/api/admin/config',
    headers={'Authorization': 'Bearer YOUR_ADMIN_API_KEY'}
)
```

#### Request (PUT)
```python
config_update = {
    "clustering": {
        "default_method": "hybrid_reservoir",
        "max_clusters": 20,
        "timeout_seconds": 3600
    },
    "security": {
        "require_encryption": True,
        "audit_logging": True,
        "data_retention_days": 180
    }
}

response = requests.put(
    'http://localhost:8000/api/admin/config',
    json=config_update,
    headers={'Authorization': 'Bearer YOUR_ADMIN_API_KEY'}
)
```

## Data Models

### Employee Data Model
```python
from pydantic import BaseModel
from typing import Optional

class EmployeeData(BaseModel):
    employee_id: str
    red_energy: float      # 0-100
    blue_energy: float     # 0-100  
    green_energy: float    # 0-100
    yellow_energy: float   # 0-100
    department: Optional[str] = None
    role_level: Optional[str] = None
    tenure_years: Optional[float] = None
```

### Cluster Model
```python
class Cluster(BaseModel):
    cluster_id: int
    name: str
    employee_count: int
    dominant_energy: str
    characteristics: dict
    personality_profile: str
    ideal_roles: list[str]
    employees: list[str]
```

### Team Model
```python
class Team(BaseModel):
    team_id: str
    members: list[dict]
    team_size: int
    balance_score: float
    energy_distribution: dict
    predicted_performance: str
    strengths: list[str]
    potential_challenges: list[str]
    recommended_projects: list[str]
```

## Error Handling

### Standard Error Response
```json
{
    "error": "validation_error",
    "message": "Invalid energy values: must be between 0 and 100",
    "details": {
        "field": "red_energy",
        "value": 150,
        "constraint": "max_value_100"
    },
    "timestamp": "2025-01-15T14:30:00Z",
    "request_id": "req_123456789"
}
```

### HTTP Status Codes
- `200`: Success
- `201`: Created successfully  
- `400`: Bad request (validation error)
- `401`: Unauthorized (invalid API key)
- `403`: Forbidden (insufficient permissions)
- `404`: Not found
- `409`: Conflict (duplicate resource)
- `429`: Too many requests (rate limit exceeded)
- `500`: Internal server error
- `503`: Service unavailable

### Error Handling in Code
```python
import requests
from requests.exceptions import RequestException

def safe_api_call(url, **kwargs):
    try:
        response = requests.get(url, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            # Rate limit exceeded
            print("Rate limit exceeded. Waiting...")
            time.sleep(60)
            return safe_api_call(url, **kwargs)
        else:
            print(f"HTTP error: {e.response.status_code}")
            print(f"Error details: {e.response.json()}")
            raise
    except RequestException as e:
        print(f"Request failed: {e}")
        raise
```

## Rate Limiting & Performance

### Rate Limits
- **Standard users**: 100 requests/hour
- **Premium users**: 1000 requests/hour  
- **Enterprise users**: 10000 requests/hour

### Rate Limit Headers
```python
response = requests.get(url, headers=headers)

print(f"Remaining requests: {response.headers.get('X-RateLimit-Remaining')}")
print(f"Reset time: {response.headers.get('X-RateLimit-Reset')}")
```

### Performance Optimization
```python
# Use connection pooling
session = requests.Session()
session.headers.update({'Authorization': 'Bearer YOUR_API_KEY'})

# Reuse session for multiple requests
response1 = session.get('http://localhost:8000/api/analytics/jobs')
response2 = session.get('http://localhost:8000/api/teams/history')

# Enable gzip compression
session.headers.update({'Accept-Encoding': 'gzip, deflate'})
```

## Integration Patterns

### Asynchronous Processing Pattern
```python
import asyncio
import aiohttp

async def async_analysis_workflow():
    async with aiohttp.ClientSession() as session:
        # Upload data for analysis
        with open('employee_data.csv', 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename='employee_data.csv')
            data.add_field('n_clusters', '4')
            
            async with session.post(
                'http://localhost:8000/api/analytics/upload',
                data=data,
                headers={'Authorization': 'Bearer YOUR_API_KEY'}
            ) as response:
                result = await response.json()
                job_id = result['job_id']
        
        # Poll for completion
        while True:
            async with session.get(
                f'http://localhost:8000/api/analytics/status/{job_id}',
                headers={'Authorization': 'Bearer YOUR_API_KEY'}
            ) as response:
                status = await response.json()
                
                if status['status'] == 'completed':
                    break
                elif status['status'] == 'failed':
                    raise Exception('Analysis failed')
                
                await asyncio.sleep(5)  # Wait 5 seconds before checking again
        
        # Get results
        async with session.get(
            f'http://localhost:8000/api/analytics/results/{job_id}',
            headers={'Authorization': 'Bearer YOUR_API_KEY'}
        ) as response:
            return await response.json()

# Run async workflow
results = asyncio.run(async_analysis_workflow())
```

### Batch Processing Pattern
```python
import concurrent.futures
import requests

def process_organization(org_file):
    """Process a single organization's data"""
    with open(org_file, 'rb') as f:
        files = {'file': f}
        data = {'n_clusters': 4, 'method': 'esn'}
        
        response = requests.post(
            'http://localhost:8000/api/analytics/upload',
            files=files,
            data=data,
            headers={'Authorization': 'Bearer YOUR_API_KEY'}
        )
        
        job_id = response.json()['job_id']
        
        # Wait for completion
        while True:
            status_response = requests.get(
                f'http://localhost:8000/api/analytics/status/{job_id}',
                headers={'Authorization': 'Bearer YOUR_API_KEY'}
            )
            status = status_response.json()
            
            if status['status'] == 'completed':
                break
            elif status['status'] == 'failed':
                return {'error': 'Analysis failed', 'file': org_file}
                
            time.sleep(10)
        
        # Get results
        results_response = requests.get(
            f'http://localhost:8000/api/analytics/results/{job_id}',
            headers={'Authorization': 'Bearer YOUR_API_KEY'}
        )
        
        return {
            'file': org_file,
            'job_id': job_id,
            'results': results_response.json()
        }

# Process multiple organizations in parallel
org_files = ['org1.csv', 'org2.csv', 'org3.csv']

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_organization, org_files))

for result in results:
    if 'error' in result:
        print(f"Failed to process {result['file']}: {result['error']}")
    else:
        print(f"Successfully processed {result['file']}")
```

### Webhook Integration Pattern
```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/webhook/analysis-complete', methods=['POST'])
def handle_analysis_complete():
    """Handle webhook notification when analysis completes"""
    data = request.json
    
    job_id = data['job_id']
    status = data['status']
    
    if status == 'completed':
        # Automatically generate teams
        team_request = {
            'job_id': job_id,
            'num_teams': 3,
            'balance_strategy': 'energy_balanced'
        }
        
        response = requests.post(
            'http://localhost:8000/api/teams/generate',
            json=team_request,
            headers={'Authorization': 'Bearer YOUR_API_KEY'}
        )
        
        # Send notification to stakeholders
        send_notification(job_id, response.json())
    
    return jsonify({'status': 'processed'})

def send_notification(job_id, team_results):
    """Send notification about completed analysis and team formation"""
    # Implementation depends on your notification system
    # (email, Slack, Teams, etc.)
    pass

if __name__ == '__main__':
    app.run(port=5000)
```

### Real-time Dashboard Integration
```python
import websocket
import json

def on_message(ws, message):
    """Handle real-time updates from the API"""
    data = json.loads(message)
    
    if data['type'] == 'job_status_update':
        print(f"Job {data['job_id']} status: {data['status']}")
        update_dashboard(data)
    elif data['type'] == 'system_metrics':
        print(f"System load: {data['cpu_usage']}")
        update_metrics_dashboard(data)

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed")

def on_open(ws):
    print("WebSocket connection opened")
    # Subscribe to job updates
    ws.send(json.dumps({
        'action': 'subscribe',
        'channel': 'job_updates',
        'auth_token': 'YOUR_API_KEY'
    }))

# Connect to WebSocket for real-time updates
ws = websocket.WebSocketApp(
    "ws://localhost:8000/ws",
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close
)

ws.run_forever()
```

This comprehensive API documentation provides everything needed to integrate Observer Coordinator Insights into your applications and workflows. For additional examples and advanced use cases, refer to the [Integration Patterns](#integration-patterns) section above.