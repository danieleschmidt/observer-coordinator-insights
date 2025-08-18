"""FastAPI Application - Observer Coordinator Insights API
Multi-agent orchestration for organizational analytics
"""

import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routes.admin import router as admin_router
from api.routes.analytics import router as analytics_router
from api.routes.health import router as health_router
from api.routes.teams import router as teams_router


# Create FastAPI application
app = FastAPI(
    title="Observer Coordinator Insights API",
    description="Multi-agent orchestration for organizational analytics from Insights Discovery data",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="/api/health", tags=["health"])
app.include_router(analytics_router, prefix="/api/analytics", tags=["analytics"])
app.include_router(teams_router, prefix="/api/teams", tags=["teams"])
app.include_router(admin_router, prefix="/api/admin", tags=["admin"])

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard endpoint"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Observer Coordinator Insights</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
            <div class="container">
                <a class="navbar-brand" href="/">Observer Coordinator Insights</a>
                <div class="navbar-nav ms-auto">
                    <a class="nav-link" href="/api/docs">API Docs</a>
                </div>
            </div>
        </nav>
        
        <div class="container mt-4">
            <div class="row">
                <div class="col-12">
                    <h1>Organizational Analytics Dashboard</h1>
                    <p class="lead">Multi-agent orchestration for Insights Discovery data analysis</p>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Upload Data</h5>
                            <p class="card-text">Upload Insights Discovery CSV files for analysis</p>
                            <input type="file" class="form-control" id="csvFile" accept=".csv">
                            <button class="btn btn-primary mt-2" onclick="uploadFile()">Upload & Analyze</button>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Analytics</h5>
                            <p class="card-text">View clustering results and employee insights</p>
                            <button class="btn btn-success" onclick="loadAnalytics()">View Analytics</button>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Team Simulator</h5>
                            <p class="card-text">Generate optimal team compositions</p>
                            <button class="btn btn-info" onclick="openTeamSim()">Team Simulator</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-12">
                    <div id="results" class="card" style="display: none;">
                        <div class="card-body">
                            <h5 class="card-title">Analysis Results</h5>
                            <div id="resultsContent"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            async function uploadFile() {
                const fileInput = document.getElementById('csvFile');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select a CSV file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/api/analytics/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    displayResults(result);
                } catch (error) {
                    alert('Error uploading file: ' + error.message);
                }
            }
            
            function displayResults(data) {
                const resultsDiv = document.getElementById('results');
                const contentDiv = document.getElementById('resultsContent');
                
                let html = `
                    <h6>Clustering Results</h6>
                    <p><strong>Employees Analyzed:</strong> ${data.employee_count}</p>
                    <p><strong>Clusters:</strong> ${data.cluster_count}</p>
                    <p><strong>Quality Score:</strong> ${data.quality_score}</p>
                `;
                
                contentDiv.innerHTML = html;
                resultsDiv.style.display = 'block';
            }
            
            function loadAnalytics() {
                window.open('/api/docs#/analytics', '_blank');
            }
            
            function openTeamSim() {
                window.open('/api/docs#/teams', '_blank');
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
