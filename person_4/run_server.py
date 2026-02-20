#!/usr/bin/env python3
"""
Quick server startup script
Run with: python run_server.py
"""

import uvicorn
from src.config import CONFIG

if __name__ == "__main__":
    print("=" * 70)
    print("Content Integrity Platform - Starting Server")
    print("=" * 70)
    print(f"\nServer will start at: http://{CONFIG.api_host}:{CONFIG.api_port}")
    print(f"API Documentation: http://{CONFIG.api_host}:{CONFIG.api_port}/docs")
    print(f"Web Interface: http://{CONFIG.api_host}:{CONFIG.api_port}")
    print("\nPress CTRL+C to stop the server\n")
    print("=" * 70)
    
    uvicorn.run(
        "api.app:app",
        host=CONFIG.api_host,
        port=CONFIG.api_port,
        reload=True,
        log_level=CONFIG.log_level.lower()
    )
