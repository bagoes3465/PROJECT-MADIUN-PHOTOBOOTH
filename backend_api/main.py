"""
AI Photobooth Kota Madiun - FastAPI Backend
"""
import sys
import os
from pathlib import Path

# Ensure backend_api is on sys.path regardless of working directory
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
os.chdir(_THIS_DIR)

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from routers import health, sessions, photos, assets, mood

app = FastAPI(
    title="AI Photobooth Kota Madiun",
    description="Backend API for AI Photobooth Service",
    version="2.0.0",
)

# CORS - allow Electron frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, prefix="/api")
app.include_router(sessions.router, prefix="/api")
app.include_router(photos.router, prefix="/api")
app.include_router(assets.router, prefix="/api")
app.include_router(mood.router, prefix="/api")


@app.get("/")
async def root():
    return {
        "app": "AI Photobooth Kota Madiun",
        "version": "2.0.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )