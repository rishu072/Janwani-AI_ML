# Janwani AI Detection Service

> **Production-ready FastAPI microservice** for civic issue detection — potholes and garbage — powered by a YOLO11n model trained on Indian street imagery.

---

## Table of Contents
1. [Overview](#1-overview)
2. [AI Capabilities](#2-ai-capabilities)
3. [Architecture](#3-architecture)
4. [Model Information](#4-model-information)
5. [API Endpoints](#5-api-endpoints)
6. [Local Setup](#6-local-setup)
7. [Environment Variables](#7-environment-variables)
8. [Docker Setup](#8-docker-setup)
9. [API Examples](#9-api-examples)
10. [Render Deployment](#10-render-deployment)
11. [Frontend Integration](#11-frontend-integration)
12. [Testing](#12-testing)
13. [Troubleshooting](#13-troubleshooting)
14. [Future Model Expansion](#14-future-model-expansion)

---

## 1. Overview

The **Janwani AI Detection Service** is a standalone microservice that exposes a REST API for detecting civic issues in uploaded images. It is designed to be consumed by the Janwani frontend/backend over HTTP and is independently deployable via Docker and Render.

**Key design decisions:**
- Model loads **once at startup** (not per request) — fast inference.
- Warm-up inference runs at startup so the first real request has no cold-start penalty.
- All configuration is via **environment variables** — no hardcoded paths or secrets.
- Upload validation rejects invalid files before inference (no crashes on bad input).
- Fully self-contained Docker image — no external model downloads at runtime.

---

## 2. AI Capabilities

| Issue | Classes Detected | Model |
|-------|-----------------|-------|
| **Pothole** | Class IDs: 1, 3, 9 | `civic.pt` |
| **Garbage / Waste** | Class ID: 4 | `civic.pt` |
| Water / Sewage | *Planned — model not yet available* | `water.pt` (future) |

Dataset: [Civic Issues Dataset on Roboflow](https://universe.roboflow.com/snehit-agarwal-otd2m/civic-issues-2ou5i)

---

## 3. Architecture

```
Janwani Frontend / Backend
        │
        │  HTTP POST multipart/form-data
        ▼
┌───────────────────────────────────────────┐
│          Janwani AI Detection Service      │
│  FastAPI + Uvicorn  (port 8000)           │
│                                           │
│  app/config.py   ← env vars               │
│  app/api/routes.py                        │
│  app/services/inference.py (ModelRegistry)│
│  app/utils/image.py                       │
└───────────────────────┬───────────────────┘
                        │
                        ▼
               models/civic.pt
              (YOLO11n, 5.47 MB)
                        │
            ┌───────────┴──────────┐
            │                      │
        Pothole               Garbage
        Detection             Detection
            │                      │
            └───────────┬──────────┘
                        │
                        ▼
            Structured JSON + Annotated JPEG
```

**Service layout:**
```
deployment/
├── app/
│   ├── main.py          # App factory, lifespan, CORS, middleware
│   ├── config.py        # All settings from environment variables
│   ├── api/
│   │   └── routes.py    # GET /, GET /health, POST /detect, POST /detect/json
│   ├── services/
│   │   └── inference.py # ModelRegistry — load-once, warm-up, detect()
│   └── utils/
│       └── image.py     # Validation, decoding, annotation, encoding
├── models/
│   └── civic.pt         # Trained YOLO11n weights (5.47 MB)
├── tests/
│   ├── test_health.py
│   └── test_detection.py
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

---

## 4. Model Information

| Property | Value |
|----------|-------|
| File | `models/civic.pt` |
| Architecture | YOLO11n (Nano) |
| Size | ~5.47 MB |
| Training framework | Ultralytics |
| Training epochs | 30 |
| Training image size | 640 × 640 px |
| Device | CPU (GPU supported if available) |
| Default confidence threshold | 0.15 |
| Default IoU threshold | 0.45 |

> ⚠️ **Do not retrain** `civic.pt` without the original dataset and a proper training pipeline. The existing weights are the production model.

---

## 5. API Endpoints

### `GET /`
Returns service information.

**Response:**
```json
{
  "service": "Janwani AI Detection Service",
  "version": "1.0.0",
  "status": "running",
  "models_loaded": ["civic"],
  "endpoints": { ... }
}
```

---

### `GET /health`
Liveness + readiness check. Returns `200` when model is ready, `503` if not.

**Response (healthy):**
```json
{
  "status": "healthy",
  "models_loaded": ["civic"],
  "uptime_seconds": 45.2,
  "service": "Janwani AI Detection Service",
  "version": "1.0.0"
}
```

---

### `POST /detect`
Upload an image → returns an **annotated JPEG** with bounding boxes drawn.

- **Content-Type:** `multipart/form-data`
- **Field:** `file` (JPEG / PNG / WebP / BMP)
- **Max size:** 10 MB (configurable)
- **Response:** `image/jpeg`

---

### `POST /detect/json`
Upload an image → returns **structured JSON** detections.

- **Content-Type:** `multipart/form-data`
- **Field:** `file` (JPEG / PNG / WebP / BMP)
- **Response:**
```json
{
  "success": true,
  "model": "civic",
  "image": {
    "width": 1920,
    "height": 1080,
    "filename": "road.jpg"
  },
  "count": 2,
  "detections": [
    {
      "class_id": 1,
      "class_name": "pothole",
      "confidence": 0.91,
      "bbox": {
        "x1": 120,
        "y1": 80,
        "x2": 340,
        "y2": 290
      }
    },
    {
      "class_id": 4,
      "class_name": "garbage",
      "confidence": 0.76,
      "bbox": {
        "x1": 600,
        "y1": 200,
        "x2": 900,
        "y2": 450
      }
    }
  ],
  "processing_time_ms": 143.0
}
```

**Interactive docs:** `http://localhost:8000/docs` (Swagger UI)

---

## 6. Local Setup

### Prerequisites
- Python 3.11+
- `pip`

### Steps

```bash
# Navigate to the deployment directory
cd deployment

# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Visit: http://localhost:8000/docs
```

---

## 7. Environment Variables

All configuration is read from environment variables. Set them in a `.env` file, shell, or on Render.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/civic.pt` | Path to YOLO weights file |
| `CONFIDENCE_THRESHOLD` | `0.15` | Minimum detection confidence (0.0–1.0) |
| `IOU_THRESHOLD` | `0.45` | IoU threshold for NMS |
| `INFERENCE_IMAGE_SIZE` | `640` | Image size fed to YOLO (pixels) |
| `MAX_IMAGE_SIZE_MB` | `10` | Maximum upload size in MB |
| `ALLOWED_ORIGINS` | `*` | CORS origins (comma-separated) — **tighten in production** |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `PORT` | `8000` | Server port |

**Example `.env` (local development):**
```env
MODEL_PATH=models/civic.pt
CONFIDENCE_THRESHOLD=0.15
IOU_THRESHOLD=0.45
MAX_IMAGE_SIZE_MB=10
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
LOG_LEVEL=DEBUG
PORT=8000
```

---

## 8. Docker Setup

### Build

```bash
cd deployment
docker build -t janwani-ai-service .
```

### Run

```bash
docker run -p 8000:8000 janwani-ai-service
```

### Run with custom env vars

```bash
docker run -p 8000:8000 \
  -e CONFIDENCE_THRESHOLD=0.35 \
  -e ALLOWED_ORIGINS="https://janwani.vercel.app" \
  janwani-ai-service
```

### Test the container

```bash
# Health check
curl http://localhost:8000/health

# JSON detection
curl -X POST http://localhost:8000/detect/json \
  -F "file=@/path/to/road.jpg" | python -m json.tool

# Image detection (saves annotated image)
curl -X POST http://localhost:8000/detect \
  -F "file=@/path/to/road.jpg" \
  --output annotated.jpg
```

---

## 9. API Examples

### Python

```python
import requests

# JSON detections
with open("road.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/detect/json",
        files={"file": ("road.jpg", f, "image/jpeg")},
    )

data = response.json()
print(f"Found {data['count']} issues in {data['processing_time_ms']:.0f} ms")
for det in data["detections"]:
    print(f"  {det['class_name']}: {det['confidence']:.0%} @ {det['bbox']}")
```

### JavaScript / Fetch (Frontend)

```javascript
async function detectCivicIssues(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await fetch("https://your-render-url.onrender.com/detect/json", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Detection failed");
  }

  return await response.json();
}

// Usage
const result = await detectCivicIssues(imageInputElement.files[0]);
console.log(result.detections);
```

### cURL

```bash
# Health
curl https://your-render-url.onrender.com/health

# JSON detection
curl -X POST https://your-render-url.onrender.com/detect/json \
  -F "file=@road.jpg"

# Annotated image
curl -X POST https://your-render-url.onrender.com/detect \
  -F "file=@road.jpg" \
  --output result.jpg
```

---

## 10. Render Deployment

### Steps

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "feat: production AI service"
   git push origin main
   ```

2. **Connect on Render:**
   - Go to [render.com](https://render.com) → New → Web Service
   - Connect your GitHub repository
   - Render will detect `render.yaml` automatically (Blueprint deploy)
   - Or configure manually:
     - **Runtime:** Docker
     - **Dockerfile Path:** `deployment/Dockerfile`
     - **Docker Context:** `deployment`

3. **Set environment variables** in the Render dashboard:
   - `ALLOWED_ORIGINS` → `https://your-janwani-frontend.com`
   - Other vars are pre-configured in `render.yaml`

4. **Deploy** — Render builds the Docker image and starts the container.

5. **Verify:**
   - Health check: `GET https://your-service.onrender.com/health`
   - Should return `{"status": "healthy", ...}`

> **Note on cold starts:** Render free tier spins down after 15 minutes of inactivity. The first request after a cold start takes ~30–60 s (model loading + warm-up). Upgrade to Starter plan to avoid this.

---

## 11. Frontend Integration

### How to display detections

```javascript
async function processImage(imageFile) {
  // 1. Get structured detections
  const formData = new FormData();
  formData.append("file", imageFile);

  const res = await fetch(`${AI_SERVICE_URL}/detect/json`, {
    method: "POST",
    body: formData,
  });
  const data = await res.json();

  // 2. Draw boxes on a canvas (alternative to fetching the annotated image)
  const canvas = document.getElementById("resultCanvas");
  const ctx = canvas.getContext("2d");
  const img = new Image();
  img.onload = () => {
    canvas.width  = data.image.width;
    canvas.height = data.image.height;
    ctx.drawImage(img, 0, 0);

    for (const det of data.detections) {
      const { x1, y1, x2, y2 } = det.bbox;
      ctx.strokeStyle = det.class_name === "pothole" ? "#ff0000" : "#ffff00";
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.font = "16px Arial";
      ctx.fillText(
        `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`,
        x1, y1 - 5
      );
    }
  };
  img.src = URL.createObjectURL(imageFile);

  // 3. OR simply display the pre-annotated image from /detect
  const annotatedForm = new FormData();
  annotatedForm.append("file", imageFile);
  const imgRes = await fetch(`${AI_SERVICE_URL}/detect`, {
    method: "POST",
    body: annotatedForm,
  });
  const blob = await imgRes.blob();
  document.getElementById("annotatedImg").src = URL.createObjectURL(blob);
}
```

### Complaint form integration

When a user submits a complaint with a photo:
1. POST the image to `/detect/json`
2. Extract `detections[].class_name` → auto-fill the complaint category
3. Extract `detections[].confidence` → attach confidence to the report
4. Display the annotated image from `/detect` as a preview

---

## 12. Testing

```bash
cd deployment

# Install test deps (included in requirements.txt)
pip install pytest httpx

# Run all tests
pytest tests/ -v

# Run only health tests
pytest tests/test_health.py -v

# Run only detection tests
pytest tests/test_detection.py -v
```

Tests use synthetic in-memory images — no external files needed.

---

## 13. Troubleshooting

### `FileNotFoundError: models/civic.pt`
- Ensure you are running uvicorn from the `deployment/` directory.
- Check that `deployment/models/civic.pt` exists (5.47 MB).

### `libGL.so.1: cannot open shared object file`
- You are using `opencv-python` instead of `opencv-python-headless`.
- Run: `pip install opencv-python-headless` and uninstall `opencv-python`.

### `OSError: [Errno 28] No space left on device` (Docker)
- YOLO downloads metadata on first run. Ensure Docker has ≥2 GB disk space.

### Render cold start is slow
- Expected on free tier. First request after inactivity takes 30–60 s.
- Upgrade to Starter plan or use an external ping service to keep the service warm.

### CORS errors in browser
- Set `ALLOWED_ORIGINS` to your exact frontend URL (no trailing slash).
- Example: `ALLOWED_ORIGINS=https://janwani.vercel.app`

### 503 on `/health`
- Model failed to load. Check logs for the error message.
- Verify `MODEL_PATH` is correct and `civic.pt` is present in the container.

---

## 14. Future Model Expansion

Adding a new model (e.g., `water.pt`) requires:

1. Drop `water.pt` into `deployment/models/`

2. Add an entry to `_MODEL_CONFIGS` in `app/services/inference.py`:
   ```python
   "water": {
       "path":      "models/water.pt",
       "conf":      0.25,
       "iou":       0.45,
       "imgsz":     640,
       "classes":   None,          # None = all classes
       "class_map": {0: "flood", 1: "sewage_overflow"},
       "required":  False,         # Don't fail if file is missing
   },
   ```

3. Optionally add a new endpoint in `app/api/routes.py` that calls
   `registry.detect(img, model_name="water")`.

No other code changes needed — the registry handles loading, warm-up, and inference automatically.

### Recommended improvements (not requiring retraining)
- **Raise confidence threshold** to 0.30–0.45 for fewer false positives in production.
- **GPU inference**: Set `device="cuda"` in the YOLO call if a GPU is available.
- **Response caching**: Cache results for identical image hashes (Redis) to reduce load.
- **Async inference**: Move inference to a background task for very large images.

### Model quality improvements (requires dataset + retraining)
- Collect more diverse Indian road images (rain, night, different cities).
- Balance class distribution (if garbage detections are noisy, add more garbage samples).
- Fine-tune with a higher-resolution YOLO variant (YOLO11s or YOLO11m).
- Run post-training evaluation: precision, recall, mAP50, mAP50-95 on a held-out test set.
