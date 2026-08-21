"""
app/config.py
All configuration is read from environment variables with safe defaults.
Override any of these on Render / Docker / locally via a .env file or shell exports.
"""

import os
from pathlib import Path
from typing import List


class Settings:
    # ------------------------------------------------------------------ paths
    # Relative to the working directory (i.e. deployment/)
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/civic.pt")

    # --------------------------------------------------------- inference knobs
    # Confidence threshold — detections below this score are discarded.
    # 0.15 is intentionally low to surface small/partial civic issues.
    # Raise to 0.30–0.45 if you want fewer but higher-confidence detections.
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.15"))

    # IoU threshold for Non-Maximum Suppression
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))

    # Image size fed to YOLO (pixels, square). 640 = model's native training size.
    INFERENCE_IMAGE_SIZE: int = int(os.getenv("INFERENCE_IMAGE_SIZE", "640"))

    # ---------------------------------------------------- upload / validation
    # Maximum accepted upload size in megabytes
    MAX_IMAGE_SIZE_MB: int = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))

    ALLOWED_EXTENSIONS: set = {"jpg", "jpeg", "png", "webp", "bmp"}

    # ------------------------------------------------------------------ CORS
    # Comma-separated list of allowed origins.
    # Default "*" is convenient for first-time deployment; tighten in production
    # by setting ALLOWED_ORIGINS=https://janwani.your-domain.com
    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        raw = os.getenv("ALLOWED_ORIGINS", "*")
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    # --------------------------------------------------------------- logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ----------------------------------------------------------------- server
    PORT: int = int(os.getenv("PORT", "8000"))

    # ------------------------------------------------------- service metadata
    SERVICE_NAME: str = "Janwani AI Detection Service"
    SERVICE_VERSION: str = "1.0.0"
    MODEL_NAME: str = "civic"

    # ---- civic model class-ID → display label mapping
    # Classes trained: pothole (IDs 1, 3, 9) and garbage (ID 4)
    CIVIC_CLASS_MAP: dict = {
        1: "pothole",
        3: "pothole",
        4: "garbage",
        9: "pothole",
    }
    CIVIC_ACTIVE_CLASSES: list = [1, 3, 4, 9]


# Singleton — import `settings` everywhere
settings = Settings()
