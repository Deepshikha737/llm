#!/usr/bin/env python3
"""
Run the LLM-Powered Intelligent Query-Retrieval System
"""

import sys
import os
import uvicorn
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

import config

def main():
    """Run the FastAPI server"""
    print("🚀 Starting LLM-Powered Intelligent Query-Retrieval System")
    print(f"📝 Server will run on http://{config.API_HOST}:{config.API_PORT}")
    print(f"📚 API Documentation: http://{config.API_HOST}:{config.API_PORT}/docs")
    print(f"🔍 Main endpoint: http://{config.API_HOST}:{config.API_PORT}/api/v1/hackrx/run")
    print("-" * 60)
    
    try:
        uvicorn.run(
            "api.main:app",
            host=config.API_HOST,
            port=config.API_PORT,
            reload=config.DEBUG_MODE,
            log_level=config.LOG_LEVEL.lower()
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()