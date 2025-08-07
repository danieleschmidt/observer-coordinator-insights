#!/usr/bin/env python3
"""
API Server Launcher for Observer Coordinator Insights
Multi-agent orchestration REST API with web dashboard
"""

import sys
import uvicorn
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from api.main import app

def main():
    """Main function to start the API server"""
    parser = argparse.ArgumentParser(
        description="Observer Coordinator Insights API Server"
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind to (default: 8000)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes (default: 1)'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    parser.add_argument(
        '--log-level',
        default='info',
        choices=['critical', 'error', 'warning', 'info', 'debug'],
        help='Log level (default: info)'
    )
    
    args = parser.parse_args()
    
    print(f"""
🚀 Starting Observer Coordinator Insights API Server
   
   Host: {args.host}
   Port: {args.port}
   Workers: {args.workers}
   Reload: {args.reload}
   Log Level: {args.log_level}
   
   API Documentation: https://{args.host}:{args.port}/api/docs
   Dashboard: https://{args.host}:{args.port}/
   Health Check: https://{args.host}:{args.port}/api/health
   
   Note: Use HTTPS in production. For local development, you may access via http://localhost:{args.port}
   
🎯 Multi-agent orchestration for organizational analytics
""")
    
    # Start the server
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level=args.log_level,
        access_log=True
    )

if __name__ == '__main__':
    main()