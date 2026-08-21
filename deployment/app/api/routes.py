"""
app/api/routes.py
All API endpoints for the Janwani AI Detection Service.

Endpoints:
  GET  /                 — service information
  GET  /health           — liveness + model status
  POST /detect           — upload image → annotated JPEG
  POST /detect/json      — upload image → structured JSON detections
"""

import logging
import time
from dataclasses import asdict

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from app.config import settings
from app.services.inference import registry
from app.utils.image import (
    draw_detections,
    encode_jpeg,
    image_dimensions,
    validate_and_decode,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Track service start-time for uptime reporting
_START_TIME = time.time()


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

@router.get("/", summary="Service information")
def root():
    """Returns basic information about this AI microservice."""
    return {
        "service":       settings.SERVICE_NAME,
        "version":       settings.SERVICE_VERSION,
        "status":        "running",
        "models_loaded": registry.loaded_models,
        "endpoints": {
            "health":      "GET  /health",
            "detect_img":  "POST /detect       → annotated JPEG",
            "detect_json": "POST /detect/json  → structured JSON",
        },
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health", summary="Health check")
def health():
    """
    Liveness + readiness check.
    Returns 200 if the model is loaded and ready.
    Returns 503 if the model failed to load (Render / Docker will restart).
    """
    uptime_seconds = round(time.time() - _START_TIME, 1)

    if not registry.is_ready:
        return JSONResponse(
            status_code=503,
            content={
                "status":        "unhealthy",
                "reason":        "No models are loaded.",
                "models_loaded": [],
                "uptime_seconds": uptime_seconds,
            },
        )

    return {
        "status":         "healthy",
        "models_loaded":  registry.loaded_models,
        "uptime_seconds": uptime_seconds,
        "service":        settings.SERVICE_NAME,
        "version":        settings.SERVICE_VERSION,
    }


# ---------------------------------------------------------------------------
# POST /detect  →  annotated JPEG image
# ---------------------------------------------------------------------------

@router.post(
    "/detect",
    summary="Detect civic issues — returns annotated image",
    response_class=Response,
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "Annotated image"},
        400: {"description": "Bad request"},
        503: {"description": "Model not loaded"},
    },
)
async def detect_image(file: UploadFile = File(...)):
    """
    Upload an image (JPEG / PNG / WebP / BMP).
    Returns the same image with bounding boxes and labels drawn on it.
    """
    logger.info("POST /detect | file=%s content_type=%s", file.filename, file.content_type)

    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="AI model is not loaded. Try again shortly.")

    # Validate + decode
    img = await validate_and_decode(file)

    # Run inference
    result = registry.detect(img, model_name=settings.MODEL_NAME)
    if not result.success:
        logger.error("Inference failed: %s", result.error)
        raise HTTPException(status_code=500, detail=f"Inference error: {result.error}")

    # Annotate
    annotated = draw_detections(img, [asdict(d) for d in result.detections])
    jpeg_bytes = encode_jpeg(annotated)

    logger.info(
        "POST /detect done | detections=%d time=%.0f ms",
        len(result.detections), result.processing_time_ms,
    )
    return Response(content=jpeg_bytes, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# POST /detect/json  →  structured JSON
# ---------------------------------------------------------------------------

@router.post(
    "/detect/json",
    summary="Detect civic issues — returns structured JSON",
)
async def detect_json(file: UploadFile = File(...)):
    """
    Upload an image (JPEG / PNG / WebP / BMP).
    Returns structured JSON with all detections, confidence scores,
    bounding boxes, processing time, and model metadata.
    """
    logger.info("POST /detect/json | file=%s content_type=%s", file.filename, file.content_type)

    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="AI model is not loaded. Try again shortly.")

    # Validate + decode
    img = await validate_and_decode(file)
    w, h = image_dimensions(img)

    # Run inference
    result = registry.detect(img, model_name=settings.MODEL_NAME)
    if not result.success:
        logger.error("Inference failed: %s", result.error)
        raise HTTPException(status_code=500, detail=f"Inference error: {result.error}")

    # Build response
    detections_payload = []
    for det in result.detections:
        detections_payload.append({
            "class_id":   det.class_id,
            "class_name": det.class_name,
            "confidence": det.confidence,
            "bbox": det.bbox,       # {x1, y1, x2, y2}
        })

    response_body = {
        "success":            True,
        "model":              result.model_name,
        "image": {
            "width":  w,
            "height": h,
            "filename": file.filename,
        },
        "count":              len(detections_payload),
        "detections":         detections_payload,
        "processing_time_ms": result.processing_time_ms,
    }

    logger.info(
        "POST /detect/json done | detections=%d time=%.0f ms",
        len(detections_payload), result.processing_time_ms,
    )
    return response_body
