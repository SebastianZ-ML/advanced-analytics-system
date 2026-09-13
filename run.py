"""
Adaptive Multi-Agent Analytics Platform - Launcher Script.
Initializes the database, starts the FastAPI server on port 8000,
and opens the dark-mode dashboard in your default browser.
"""
import os
import sys
from pathlib import Path
import webbrowser

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import uvicorn
from app.main import app

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Adaptive Multi-Agent Analytics Platform...")
    print("Web Dashboard: http://127.0.0.1:8000")
    print("API Swagger:   http://127.0.0.1:8000/docs")
    print("=" * 60)

    try:
        webbrowser.open("http://127.0.0.1:8000")
    except Exception:
        pass

    uvicorn.run(app, host="127.0.0.1", port=8000)