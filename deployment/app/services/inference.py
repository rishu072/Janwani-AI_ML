"""
app/services/inference.py
Model registry: loads YOLO models once at startup and exposes a clean
detect() interface that returns structured detection data.

Design principles:
 - Models are loaded exactly once (at app startup via lifespan).
 - A warm-up forward pass is run so the first real request is not slow.
 - No global mutable state outside the registry instance.
 - Adding a new model requires only a new entry in MODEL_REGISTRY (no code duplication).
 - water.pt is NOT registered because the file does not exist.
   To add it: drop water.pt into models/ and add its entry below.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Dict[str, int]            # {x1, y1, x2, y2}


@dataclass
class InferenceResult:
    success: bool
    model_name: str
    detections: List[Detection]
    image_width: int
    image_height: int
    processing_time_ms: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Model configuration registry
# Each entry describes a model file + its inference parameters.
# Add future models here — the loading & inference code is shared.
# ---------------------------------------------------------------------------

_MODEL_CONFIGS: Dict[str, dict] = {
    "civic": {
        "path":         settings.MODEL_PATH,   # env-configurable
        "conf":         settings.CONFIDENCE_THRESHOLD,
        "iou":          settings.IOU_THRESHOLD,
        "imgsz":        settings.INFERENCE_IMAGE_SIZE,
        "classes":      settings.CIVIC_ACTIVE_CLASSES,
        "class_map":    settings.CIVIC_CLASS_MAP,
        "required":     True,   # If True, failure to load is fatal
    },
    # Example future model entry (uncomment when water.pt is available):
    # "water": {
    #     "path":      "models/water.pt",
    #     "conf":      0.25,
    #     "iou":       0.45,
    #     "imgsz":     640,
    #     "classes":   None,          # None = all classes
    #     "class_map": {},            # use model's built-in names
    #     "required":  False,
    # },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Holds all loaded YOLO models. Instantiated once at app startup."""

    def __init__(self):
        self._models: Dict[str, YOLO] = {}
        self._configs: Dict[str, dict] = {}
        self._loaded: bool = False

    # ---------------------------------------------------------------- setup

    def load_all(self) -> None:
        """Load every model in _MODEL_CONFIGS. Called from lifespan."""
        for name, cfg in _MODEL_CONFIGS.items():
            path = Path(cfg["path"])
            if not path.exists():
                if cfg.get("required", False):
                    raise FileNotFoundError(
                        f"Required model '{name}' not found at '{path.resolve()}'. "
                        "Ensure the models/ directory is present and civic.pt is intact."
                    )
                logger.warning("Optional model '%s' not found at '%s' — skipping.", name, path)
                continue

            logger.info("Loading model '%s' from '%s' …", name, path)
            t0 = time.perf_counter()
            try:
                model = YOLO(str(path))
                self._models[name] = model
                self._configs[name] = cfg
                elapsed = (time.perf_counter() - t0) * 1000
                logger.info("Model '%s' loaded in %.0f ms.", name, elapsed)
                self._warmup(name, model, cfg)
            except Exception as exc:
                if cfg.get("required", False):
                    raise RuntimeError(f"Failed to load required model '{name}': {exc}") from exc
                logger.error("Failed to load optional model '%s': %s", name, exc)

        self._loaded = True
        logger.info(
            "Model registry ready. Loaded: %s",
            list(self._models.keys()) or ["(none)"],
        )

    def _warmup(self, name: str, model: YOLO, cfg: dict) -> None:
        """Run one dummy inference so JIT / cache is primed before real traffic."""
        logger.info("Warming up model '%s' …", name)
        dummy = np.zeros((cfg["imgsz"], cfg["imgsz"], 3), dtype=np.uint8)
        try:
            model(
                dummy,
                conf=cfg["conf"],
                iou=cfg["iou"],
                imgsz=cfg["imgsz"],
                classes=cfg["classes"],
                verbose=False,
            )
            logger.info("Warm-up complete for '%s'.", name)
        except Exception as exc:
            logger.warning("Warm-up failed for '%s': %s", name, exc)

    # -------------------------------------------------------------- status

    @property
    def is_ready(self) -> bool:
        return self._loaded and bool(self._models)

    @property
    def loaded_models(self) -> List[str]:
        return list(self._models.keys())

    # -------------------------------------------------------------- inference

    def detect(self, image: np.ndarray, model_name: str = "civic") -> InferenceResult:
        """
        Run detection on a BGR numpy array.
        Returns an InferenceResult with typed Detection objects.
        """
        h, w = image.shape[:2]

        if model_name not in self._models:
            return InferenceResult(
                success=False,
                model_name=model_name,
                detections=[],
                image_width=w,
                image_height=h,
                processing_time_ms=0,
                error=f"Model '{model_name}' is not loaded.",
            )

        model = self._models[model_name]
        cfg   = self._configs[model_name]

        t0 = time.perf_counter()
        try:
            results = model(
                image,
                conf=cfg["conf"],
                iou=cfg["iou"],
                imgsz=cfg["imgsz"],
                classes=cfg["classes"],
                verbose=False,
            )
        except Exception as exc:
            logger.exception("Inference error on model '%s'.", model_name)
            return InferenceResult(
                success=False,
                model_name=model_name,
                detections=[],
                image_width=w,
                image_height=h,
                processing_time_ms=(time.perf_counter() - t0) * 1000,
                error=str(exc),
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        class_map  = cfg.get("class_map", {})
        detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                # Prefer the mapping; fall back to the model's own label
                label  = class_map.get(cls_id) or result.names.get(cls_id, str(cls_id))
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=label,
                    confidence=round(conf, 4),
                    bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                ))

        logger.info(
            "Inference done | model=%s detections=%d time=%.0f ms",
            model_name, len(detections), elapsed_ms,
        )
        return InferenceResult(
            success=True,
            model_name=model_name,
            detections=detections,
            image_width=w,
            image_height=h,
            processing_time_ms=round(elapsed_ms, 1),
        )


# Singleton — imported by routes and lifespan
registry = ModelRegistry()
