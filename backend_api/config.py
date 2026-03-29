"""
AI Photobooth Kota Madiun - Configuration
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


class Settings(BaseSettings):
    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_key: str = ""

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    # AI Enhancement - APIFree.ai (Nano Banana Pro Edit)
    apifree_api_key: str = ""
    apifree_model: str = "google/nano-banana-pro/edit"
    apifree_base_url: str = "https://api.apifree.ai"
    apifree_timeout_seconds: int = 300

    # ML Model Paths
    face_expression_model_path: str = str(
        PROJECT_ROOT / "machine_learning" / "Face_Expression" / "runs" / "detect" / "model-train" / "weights" / "best.pt"
    )
    yolo_fallback_model_path: str = str(
        PROJECT_ROOT / "machine_learning" / "Face_Expression" / "yolo11n.pt"
    )

    # Processing
    max_upload_size_mb: int = 10
    download_link_expiry_hours: int = 24
    session_expiry_minutes: int = 30

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
