"""
app/utils/image.py
Image validation, decoding, and annotation helpers.
All uploaded bytes are handled in-memory; nothing is written to disk.
"""

import io
import logging
from typing import Tuple

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from app.config import settings

logger = logging.getLogger(__name__)

# BGR colour palette for annotation boxes
_COLOURS = {
    "pothole": (0, 0, 255),     # Red  (BGR)
    "garbage": (0, 255, 255),   # Yellow (BGR)
    "water":   (255, 100, 0),   # Blue-ish (BGR)
}
_DEFAULT_COLOUR = (0, 200, 0)   # Green for unknown classes


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _extension_ok(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in settings.ALLOWED_EXTENSIONS


async def validate_and_decode(upload: UploadFile) -> np.ndarray:
    """
    Read an UploadFile, validate it, and return a BGR numpy array.
    Raises HTTPException on any problem so the caller gets a clean HTTP error.
    """
    # 1. Extension check
    if not _extension_ok(upload.filename or ""):
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{upload.filename}'. "
                f"Allowed: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}"
            ),
        )

    # 2. Read bytes
    contents = await upload.read()

    # 3. Size check
    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({len(contents) / 1024 / 1024:.1f} MB). "
                f"Maximum allowed: {settings.MAX_IMAGE_SIZE_MB} MB."
            ),
        )

    # 4. Decode
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=422,
            detail="Could not decode the uploaded file as an image. It may be corrupted.",
        )

    logger.debug(
        "Image decoded | file=%s size=%dx%d bytes=%d",
        upload.filename, img.shape[1], img.shape[0], len(contents),
    )
    return img


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw bounding boxes and labels onto a copy of the image.
    `detections` is the list of dicts returned by inference.py.
    Returns a new annotated image (does not mutate the original).
    """
    annotated = image.copy()
    for det in detections:
        label = det["class_name"]
        conf  = det["confidence"]
        bbox  = det["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

        colour = _COLOURS.get(label.lower(), _DEFAULT_COLOUR)
        text   = f"{label.capitalize()} {conf:.0%}"

        # Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 3)

        # Label background for readability
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 4, y1), colour, -1)

        # Label text (white)
        cv2.putText(
            annotated, text, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    return annotated


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode a BGR numpy array to JPEG bytes."""
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buffer = cv2.imencode(".jpg", image, params)
    if not success:
        raise RuntimeError("Failed to encode image to JPEG.")
    return buffer.tobytes()


def image_dimensions(image: np.ndarray) -> Tuple[int, int]:
    """Return (width, height) of a decoded image."""
    h, w = image.shape[:2]
    return w, h
