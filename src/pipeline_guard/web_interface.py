"""Web Interface and Dashboard for Pipeline Guard
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional


# FastAPI for web interface
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Async support
import threading

from .distributed import DistributedPipelineGuard
from .integration import PipelineGuardIntegrator


class WebDashboard:
    """Web dashboard for Pipeline Guard monitoring and management
    """

    def __init__(self,
                 integrator: PipelineGuardIntegrator,
                 distributed_guard: Optional[DistributedPipelineGuard] = None):
        """Initialize web dashboard"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.integrator = integrator
        self.distributed_guard = distributed_guard

        # Web application
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(
                title="Pipeline Guard Dashboard",
                description="Self-Healing Pipeline Guard Monitoring Dashboard",
                version="1.0.0"
            )
            self._setup_routes()
        else:
            self.app = None
            self.logger.warning("FastAPI not available - web dashboard disabled")

        # WebSocket connections
        self.websocket_connections: List[WebSocket] = []

        # Real-time data
        self.realtime_thread: Optional[threading.Thread] = None
        self.is_broadcasting = False

        self.logger.info("Web dashboard initialized")

    def _setup_routes(self) -> None:
        """Setup FastAPI routes"""
        if not self.app:
            return

        # Static files and templates
        # self.app.mount("/static", StaticFiles(directory="static"), name="static")
        # templates = Jinja2Templates(directory="templates")

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Main dashboard page"""
            return self._get_dashboard_html()

        @self.app.get("/api/status")
        async def get_status():
            """Get current system status"""
            try:
                status = self.integrator.get_integration_status()

                if self.distributed_guard:
                    distributed_status = self.distributed_guard.get_distributed_status()
                    status['distributed'] = distributed_status

                return JSONResponse(content=status)

            except Exception as e:
                self.logger.error(f"Error getting status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/components")
        async def get_components():
            """Get component information"""
            try:
                status = self.integrator.get_integration_status()
                components = status['pipeline_guard']['components']

                # Enhance with additional details
                enhanced_components = {}
                for name, component_info in components.items():
                    enhanced_components[name] = {
                        **component_info,
                        'health_percentage': 100.0 if component_info['state'] == 'healthy' else 0.0,
                        'uptime_hours': 24,  # Placeholder
                        'last_recovery': component_info.get('last_failure')
                    }

                return JSONResponse(content=enhanced_components)

            except Exception as e:
                self.logger.error(f"Error getting components: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get system metrics"""
            try:
                status = self.integrator.get_integration_status()
                stats = status['pipeline_guard']['statistics']

                metrics = {
                    'system_health': {
                        'healthy_components': status['pipeline_guard']['system']['healthy_components'],
                        'total_components': status['pipeline_guard']['system']['total_components'],
                        'health_percentage': (
                            status['pipeline_guard']['system']['healthy_components'] /
                            max(status['pipeline_guard']['system']['total_components'], 1) * 100
                        )
                    },
                    'recovery_stats': {
                        'total_failures': stats['total_failures'],
                        'successful_recoveries': stats['successful_recoveries'],
                        'failed_recoveries': stats['failed_recoveries'],
                        'success_rate': (
                            stats['successful_recoveries'] /
                            max(stats['successful_recoveries'] + stats['failed_recoveries'], 1) * 100
                        )
                    },
                    'uptime': {
                        'uptime_seconds': status['pipeline_guard']['system']['uptime_seconds'],
                        'uptime_hours': status['pipeline_guard']['system']['uptime_seconds'] / 3600
                    },
                    'timestamp': time.time()
                }

                return JSONResponse(content=metrics)

            except Exception as e:
                self.logger.error(f"Error getting metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/components/{component_name}/recover")
        async def trigger_recovery(component_name: str):
            """Trigger manual recovery for a component"""
            try:
                success = self.integrator.trigger_recovery(component_name)

                return JSONResponse(content={
                    'success': success,
                    'message': f"Recovery {'triggered' if success else 'failed'} for {component_name}",
                    'timestamp': time.time()
                })

            except Exception as e:
                self.logger.error(f"Error triggering recovery for {component_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/cluster")
        async def get_cluster_status():
            """Get distributed cluster status"""
            if not self.distributed_guard:
                raise HTTPException(status_code=404, detail="Distributed mode not enabled")

            try:
                cluster_status = self.distributed_guard.get_distributed_status()
                return JSONResponse(content=cluster_status)

            except Exception as e:
                self.logger.error(f"Error getting cluster status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.websocket_connections.append(websocket)

            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)

        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return JSONResponse(content={
                'status': 'healthy',
                'timestamp': time.time(),
                'service': 'pipeline-guard-dashboard'
            })

    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Pipeline Guard Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f7fa;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 2rem;
                    border-radius: 10px;
                    margin-bottom: 2rem;
                    text-align: center;
                }
                .dashboard-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 2rem;
                }
                .card {
                    background: white;
                    border-radius: 10px;
                    padding: 1.5rem;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .metric-card {
                    text-align: center;
                    border-left: 4px solid #4CAF50;
                }
                .metric-value {
                    font-size: 2.5rem;
                    font-weight: bold;
                    color: #333;
                    margin: 0.5rem 0;
                }
                .metric-label {
                    color: #666;
                    font-size: 0.9rem;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }
                .status-indicator {
                    display: inline-block;
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    margin-right: 8px;
                }
                .status-healthy { background-color: #4CAF50; }
                .status-degraded { background-color: #FF9800; }
                .status-failing { background-color: #F44336; }
                .status-offline { background-color: #9E9E9E; }
                .component-list {
                    max-height: 400px;
                    overflow-y: auto;
                }
                .component-item {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 10px;
                    border-bottom: 1px solid #eee;
                }
                .component-item:last-child {
                    border-bottom: none;
                }
                .btn {
                    padding: 6px 12px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 0.8rem;
                }
                .btn-primary {
                    background-color: #007bff;
                    color: white;
                }
                .btn-primary:hover {
                    background-color: #0056b3;
                }
                .chart-container {
                    position: relative;
                    height: 300px;
                }
                .loading {
                    text-align: center;
                    color: #666;
                    font-style: italic;
                }
                #connection-status {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    padding: 10px 15px;
                    border-radius: 5px;
                    color: white;
                    font-weight: bold;
                }
                .connected { background-color: #4CAF50; }
                .disconnected { background-color: #F44336; }
            </style>
        </head>
        <body>
            <div id="connection-status" class="disconnected">Connecting...</div>
            
            <div class="header">
                <h1>🛡️ Self-Healing Pipeline Guard</h1>
                <p>Autonomous monitoring, failure detection, and recovery system</p>
            </div>
            
            <div class="dashboard-grid">
                <div class="card metric-card">
                    <div class="metric-label">System Health</div>
                    <div class="metric-value" id="health-percentage">--%</div>
                    <div>Components Healthy</div>
                </div>
                
                <div class="card metric-card">
                    <div class="metric-label">Total Failures</div>
                    <div class="metric-value" id="total-failures">--</div>
                    <div>Since Start</div>
                </div>
                
                <div class="card metric-card">
                    <div class="metric-label">Recovery Rate</div>
                    <div class="metric-value" id="recovery-rate">--%</div>
                    <div>Success Rate</div>
                </div>
                
                <div class="card metric-card">
                    <div class="metric-label">Uptime</div>
                    <div class="metric-value" id="uptime-hours">--h</div>
                    <div>System Uptime</div>
                </div>
            </div>
            
            <div class="dashboard-grid">
                <div class="card">
                    <h3>Components Status</h3>
                    <div id="components-list" class="component-list">
                        <div class="loading">Loading components...</div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>System Health Over Time</h3>
                    <div class="chart-container">
                        <canvas id="health-chart"></canvas>
                    </div>
                </div>
            </div>
            
            <script>
                // WebSocket connection for real-time updates
                let ws;
                let healthChart;
                let healthData = [];
                
                function connectWebSocket() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                    
                    ws.onopen = function() {
                        document.getElementById('connection-status').textContent = 'Connected';
                        document.getElementById('connection-status').className = 'connected';
                    };
                    
                    ws.onclose = function() {
                        document.getElementById('connection-status').textContent = 'Disconnected';
                        document.getElementById('connection-status').className = 'disconnected';
                        // Reconnect after 5 seconds
                        setTimeout(connectWebSocket, 5000);
                    };
                    
                    ws.onerror = function() {
                        document.getElementById('connection-status').textContent = 'Error';
                        document.getElementById('connection-status').className = 'disconnected';
                    };
                }
                
                // Initialize chart
                function initChart() {
                    const ctx = document.getElementById('health-chart').getContext('2d');
                    healthChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'System Health %',
                                data: [],
                                borderColor: '#4CAF50',
                                backgroundColor: 'rgba(76, 175, 80, 0.1)',
                                tension: 0.4,
                                fill: true
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    max: 100
                                }
                            },
                            plugins: {
                                legend: {
                                    display: false
                                }
                            }
                        }
                    });
                }
                
                // Fetch and update metrics
                async function updateMetrics() {
                    try {
                        const response = await fetch('/api/metrics');
                        const metrics = await response.json();
                        
                        // Update metric cards
                        document.getElementById('health-percentage').textContent = 
                            Math.round(metrics.system_health.health_percentage) + '%';
                        document.getElementById('total-failures').textContent = 
                            metrics.recovery_stats.total_failures;
                        document.getElementById('recovery-rate').textContent = 
                            Math.round(metrics.recovery_stats.success_rate) + '%';
                        document.getElementById('uptime-hours').textContent = 
                            Math.round(metrics.uptime.uptime_hours) + 'h';
                        
                        // Update chart
                        const now = new Date().toLocaleTimeString();
                        healthData.push({
                            time: now,
                            value: metrics.system_health.health_percentage
                        });
                        
                        // Keep only last 20 points
                        if (healthData.length > 20) {
                            healthData.shift();
                        }
                        
                        healthChart.data.labels = healthData.map(d => d.time);
                        healthChart.data.datasets[0].data = healthData.map(d => d.value);
                        healthChart.update('none');
                        
                    } catch (error) {
                        console.error('Error updating metrics:', error);
                    }
                }
                
                // Fetch and update components
                async function updateComponents() {
                    try {
                        const response = await fetch('/api/components');
                        const components = await response.json();
                        
                        const container = document.getElementById('components-list');
                        container.innerHTML = '';
                        
                        for (const [name, info] of Object.entries(components)) {
                            const item = document.createElement('div');
                            item.className = 'component-item';
                            
                            const statusClass = `status-${info.state}`;
                            
                            item.innerHTML = `
                                <div>
                                    <span class="status-indicator ${statusClass}"></span>
                                    <strong>${name}</strong>
                                    <small style="color: #666; margin-left: 10px;">
                                        Failures: ${info.failure_count}
                                    </small>
                                </div>
                                <button class="btn btn-primary" onclick="triggerRecovery('${name}')">
                                    Recover
                                </button>
                            `;
                            
                            container.appendChild(item);
                        }
                    } catch (error) {
                        console.error('Error updating components:', error);
                    }
                }
                
                // Trigger recovery for a component
                async function triggerRecovery(componentName) {
                    try {
                        const response = await fetch(`/api/components/${componentName}/recover`, {
                            method: 'POST'
                        });
                        const result = await response.json();
                        
                        if (result.success) {
                            alert(`Recovery triggered for ${componentName}`);
                        } else {
                            alert(`Recovery failed for ${componentName}`);
                        }
                        
                        // Refresh components
                        updateComponents();
                    } catch (error) {
                        console.error('Error triggering recovery:', error);
                        alert('Error triggering recovery');
                    }
                }
                
                // Initialize dashboard
                document.addEventListener('DOMContentLoaded', function() {
                    connectWebSocket();
                    initChart();
                    
                    // Initial load
                    updateMetrics();
                    updateComponents();
                    
                    // Regular updates
                    setInterval(updateMetrics, 5000);  // Every 5 seconds
                    setInterval(updateComponents, 10000);  // Every 10 seconds
                });
            </script>
        </body>
        </html>
        """

    def start_realtime_broadcasting(self) -> None:
        """Start real-time data broadcasting to WebSocket clients"""
        if not self.websocket_connections:
            return

        self.is_broadcasting = True

        self.realtime_thread = threading.Thread(
            target=self._broadcasting_loop,
            daemon=True,
            name="RealtimeBroadcast"
        )
        self.realtime_thread.start()

        self.logger.info("Real-time broadcasting started")

    def stop_realtime_broadcasting(self) -> None:
        """Stop real-time broadcasting"""
        self.is_broadcasting = False

        if self.realtime_thread:
            self.realtime_thread.join(timeout=5)

    def _broadcasting_loop(self) -> None:
        """Broadcasting loop for real-time updates"""
        while self.is_broadcasting:
            try:
                if self.websocket_connections:
                    # Get current metrics
                    status = self.integrator.get_integration_status()

                    # Prepare real-time data
                    realtime_data = {
                        'type': 'metrics_update',
                        'data': {
                            'timestamp': time.time(),
                            'system_health': status['pipeline_guard']['system']['healthy_components'] /
                                           max(status['pipeline_guard']['system']['total_components'], 1) * 100,
                            'total_failures': status['pipeline_guard']['statistics']['total_failures'],
                            'components': status['pipeline_guard']['components']
                        }
                    }

                    # Broadcast to all connected clients
                    asyncio.run(self._broadcast_data(realtime_data))

            except Exception as e:
                self.logger.error(f"Error in broadcasting loop: {e}")

            time.sleep(5)  # Broadcast every 5 seconds

    async def _broadcast_data(self, data: Dict[str, Any]) -> None:
        """Broadcast data to all WebSocket connections"""
        if not self.websocket_connections:
            return

        message = json.dumps(data)
        disconnected = []

        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.append(websocket)

        # Remove disconnected clients
        for websocket in disconnected:
            if websocket in self.websocket_connections:
                self.websocket_connections.remove(websocket)

    def run(self, host: str = "0.0.0.0", port: int = 8080, **kwargs) -> None:
        """Run the web dashboard server"""
        if not FASTAPI_AVAILABLE:
            self.logger.error("FastAPI not available - cannot start web server")
            return

        if not self.app:
            self.logger.error("FastAPI app not initialized")
            return

        # Start real-time broadcasting
        self.start_realtime_broadcasting()

        try:
            self.logger.info(f"Starting web dashboard on http://{host}:{port}")
            uvicorn.run(self.app, host=host, port=port, **kwargs)
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}")
        finally:
            self.stop_realtime_broadcasting()


class APIServer:
    """REST API server for Pipeline Guard integration
    """

    def __init__(self, integrator: PipelineGuardIntegrator):
        """Initialize API server"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.integrator = integrator

        if FASTAPI_AVAILABLE:
            self.app = FastAPI(
                title="Pipeline Guard API",
                description="REST API for Pipeline Guard management",
                version="1.0.0"
            )
            self._setup_api_routes()
        else:
            self.app = None
            self.logger.warning("FastAPI not available - API server disabled")

    def _setup_api_routes(self) -> None:
        """Setup API routes"""
        if not self.app:
            return

        @self.app.get("/api/v1/status")
        async def get_status():
            """Get comprehensive system status"""
            return self.integrator.get_integration_status()

        @self.app.get("/api/v1/components")
        async def list_components():
            """List all registered components"""
            status = self.integrator.get_integration_status()
            return status['pipeline_guard']['components']

        @self.app.get("/api/v1/components/{component_name}")
        async def get_component(component_name: str):
            """Get specific component details"""
            status = self.integrator.get_integration_status()
            components = status['pipeline_guard']['components']

            if component_name not in components:
                raise HTTPException(status_code=404, detail="Component not found")

            return components[component_name]

        @self.app.post("/api/v1/components/{component_name}/recover")
        async def recover_component(component_name: str):
            """Trigger recovery for specific component"""
            success = self.integrator.trigger_recovery(component_name)
            return {
                'success': success,
                'component': component_name,
                'timestamp': time.time()
            }

        @self.app.get("/api/v1/metrics")
        async def get_metrics():
            """Get system metrics"""
            status = self.integrator.get_integration_status()
            return {
                'system': status['pipeline_guard']['system'],
                'statistics': status['pipeline_guard']['statistics'],
                'timestamp': time.time()
            }

        @self.app.get("/api/v1/health")
        async def health_check():
            """API health check"""
            return {
                'status': 'healthy',
                'service': 'pipeline-guard-api',
                'timestamp': time.time()
            }

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs) -> None:
        """Run the API server"""
        if not FASTAPI_AVAILABLE or not self.app:
            self.logger.error("Cannot start API server - FastAPI not available")
            return

        self.logger.info(f"Starting API server on http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port, **kwargs)
