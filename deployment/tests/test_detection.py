"""
tests/test_detection.py
Tests for /detect and /detect/json endpoints.
Run from the deployment/ directory:
    pytest tests/test_detection.py -v

Note: These tests use a synthetic 640×640 black image (no real detections expected)
to verify the API contract without needing a production image.
"""

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


def _make_jpeg_bytes(width: int = 640, height: int = 480) -> bytes:
    """Create a minimal valid JPEG in memory."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Add a little colour variation so the image isn't completely blank
    img[100:200, 100:300] = (50, 80, 120)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /detect  (image endpoint)
# ---------------------------------------------------------------------------

def test_detect_returns_jpeg(client):
    jpeg = _make_jpeg_bytes()
    response = client.post(
        "/detect",
        files={"file": ("test.jpg", io.BytesIO(jpeg), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


def test_detect_response_is_valid_image(client):
    jpeg = _make_jpeg_bytes()
    response = client.post(
        "/detect",
        files={"file": ("test.jpg", io.BytesIO(jpeg), "image/jpeg")},
    )
    arr = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert img is not None, "Response is not a valid image"
    assert img.shape[0] > 0 and img.shape[1] > 0


def test_detect_png_accepted(client):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    response = client.post(
        "/detect",
        files={"file": ("test.png", io.BytesIO(buf.tobytes()), "image/png")},
    )
    assert response.status_code == 200


def test_detect_rejects_unsupported_extension(client):
    response = client.post(
        "/detect",
        files={"file": ("malware.exe", io.BytesIO(b"fake"), "application/octet-stream")},
    )
    assert response.status_code == 415


def test_detect_rejects_empty_file(client):
    response = client.post(
        "/detect",
        files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
    )
    assert response.status_code == 400


def test_detect_rejects_corrupted_image(client):
    response = client.post(
        "/detect",
        files={"file": ("bad.jpg", io.BytesIO(b"THIS IS NOT AN IMAGE"), "image/jpeg")},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /detect/json  (structured JSON endpoint)
# ---------------------------------------------------------------------------

def test_detect_json_returns_200(client):
    jpeg = _make_jpeg_bytes()
    response = client.post(
        "/detect/json",
        files={"file": ("test.jpg", io.BytesIO(jpeg), "image/jpeg")},
    )
    assert response.status_code == 200


def test_detect_json_structure(client):
    jpeg = _make_jpeg_bytes()
    response = client.post(
        "/detect/json",
        files={"file": ("test.jpg", io.BytesIO(jpeg), "image/jpeg")},
    )
    data = response.json()
    assert data["success"] is True
    assert "detections" in data
    assert "count" in data
    assert "processing_time_ms" in data
    assert "model" in data
    assert "image" in data
    assert data["image"]["width"] == 640
    assert data["image"]["height"] == 480


def test_detect_json_count_matches_detections(client):
    jpeg = _make_jpeg_bytes()
    data = client.post(
        "/detect/json",
        files={"file": ("test.jpg", io.BytesIO(jpeg), "image/jpeg")},
    ).json()
    assert data["count"] == len(data["detections"])


def test_detect_json_detection_fields(client):
    """Each detection must have the required fields."""
    jpeg = _make_jpeg_bytes()
    data = client.post(
        "/detect/json",
        files={"file": ("test.jpg", io.BytesIO(jpeg), "image/jpeg")},
    ).json()
    for det in data["detections"]:
        assert "class_id" in det
        assert "class_name" in det
        assert "confidence" in det
        assert "bbox" in det
        bbox = det["bbox"]
        assert all(k in bbox for k in ("x1", "y1", "x2", "y2"))
        assert 0.0 <= det["confidence"] <= 1.0


def test_detect_json_processing_time_positive(client):
    jpeg = _make_jpeg_bytes()
    data = client.post(
        "/detect/json",
        files={"file": ("test.jpg", io.BytesIO(jpeg), "image/jpeg")},
    ).json()
    assert data["processing_time_ms"] > 0


def test_detect_json_rejects_corrupted_image(client):
    response = client.post(
        "/detect/json",
        files={"file": ("bad.jpg", io.BytesIO(b"GARBAGE DATA"), "image/jpeg")},
    )
    assert response.status_code == 422
