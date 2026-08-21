# ============================================================
# Janwani AI Detection Service — Production Dockerfile
# ============================================================
# Base: python:3.11-slim (Debian Bookworm)
# Port: 8000 (configurable via $PORT env var on Render/cloud)
# ============================================================

FROM python:3.11-slim

# ---- metadata -------------------------------------------------------
LABEL maintainer="Janwani Team"
LABEL description="Janwani AI civic-issue detection microservice"

# ---- system dependencies (OpenCV headless needs these) --------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- non-root user --------------------------------------------------
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup --no-create-home appuser

# ---- working directory ----------------------------------------------
WORKDIR /app

# ---- Python dependencies (install before copying code so layer is cached) ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- application code + models ------------------------------------
COPY app/      ./app/
COPY models/   ./models/

# ---- set ownership --------------------------------------------------
RUN chown -R appuser:appgroup /app

# ---- switch to non-root ---------------------------------------------
USER appuser

# ---- expose port (informational; actual port set via $PORT) ---------
EXPOSE 8000

# ---- health check (Docker-native; also used by Render) --------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health')"

# ---- start uvicorn --------------------------------------------------
# Reads $PORT so Render's dynamic port injection works seamlessly.
# Workers=1: YOLO model is large — multiple workers would multiply RAM usage.
# Use gunicorn with workers only if you add a shared model cache (Redis/mmap).
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --log-level warning"