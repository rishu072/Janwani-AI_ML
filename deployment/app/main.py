"""
app/main.py
FastAPI application factory.

Responsibilities:
 - Define lifespan: load models on startup, clean up on shutdown
 - Configure structured logging
 - Add CORS middleware with env-configurable origins
 - Add request body size limit middleware
 - Mount the API router
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.services.inference import registry


# ---------------------------------------------------------------------------
# Logging — structured, single-line, Render-friendly
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    # Silence noisy third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


_configure_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Upload size enforcement middleware
# ---------------------------------------------------------------------------

class MaxUploadSizeMiddleware(BaseHTTPMiddleware):
    """Reject multipart bodies larger than MAX_IMAGE_SIZE_MB before they hit routes."""

    def __init__(self, app, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": (
                        f"Request body too large. "
                        f"Maximum allowed: {settings.MAX_IMAGE_SIZE_MB} MB."
                    )
                },
            )
        return await call_next(request)


# ---------------------------------------------------------------------------
# Lifespan — load models once, release on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Starting %s v%s", settings.SERVICE_NAME, settings.SERVICE_VERSION)
    logger.info("=" * 60)

    # Load models (blocking — intentional: we don't serve traffic until ready)
    try:
        registry.load_all()
    except Exception as exc:
        logger.critical("FATAL: Could not load required model: %s", exc)
        # Re-raise so uvicorn exits — Render/Docker will restart the container
        raise

    logger.info("Service ready. Loaded models: %s", registry.loaded_models)
    yield

    # Shutdown
    logger.info("Shutting down %s.", settings.SERVICE_NAME)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    description=(
        "AI-powered civic issue detection microservice for the Janwani platform. "
        "Detects potholes and garbage in uploaded images using a YOLO11n model."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — configurable via ALLOWED_ORIGINS env var
origins = settings.ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=(origins != ["*"]),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
logger.info("CORS configured for origins: %s", origins)

# Upload size guard
app.add_middleware(
    MaxUploadSizeMiddleware,
    max_bytes=settings.MAX_IMAGE_SIZE_MB * 1024 * 1024,
)

# Routes
from app.api.routes import router          # noqa: E402  (after app creation)
app.include_router(router)
