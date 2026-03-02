"""YOLOv11-L object detection model wrapper.

This module provides a clean interface around the Ultralytics YOLO11 Large
model, suitable for integration into an AI agent pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import ModelConfig, load_config
from src.utils.logging import setup_logger

logger = setup_logger("yolo_agent.detection")


@dataclass
class Detection:
    """A single detected object."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    bbox_norm: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 2) for v in self.bbox],
        }


@dataclass
class DetectionResult:
    """Aggregated result for one image."""

    detections: list[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    image_shape: tuple[int, int] = (0, 0)  # height, width

    @property
    def count(self) -> int:
        return len(self.detections)

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts

    def filter_by_class(self, class_names: list[str]) -> list[Detection]:
        return [d for d in self.detections if d.class_name in class_names]

    def filter_by_confidence(self, min_conf: float) -> list[Detection]:
        return [d for d in self.detections if d.confidence >= min_conf]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "image_shape": list(self.image_shape),
            "class_counts": self.class_counts(),
            "detections": [d.to_dict() for d in self.detections],
        }


class YOLODetector:
    """High-level wrapper around the Ultralytics YOLOv11-L model.

    Parameters
    ----------
    config : ModelConfig | None
        Model configuration.  When *None* the default config is loaded.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or load_config().model
        self._model = None
        self._class_names: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Lazy loading – the heavy model is only loaded when first needed.
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        from ultralytics import YOLO  # deferred import to keep startup fast

        weights = self.config.weights
        logger.info("Loading YOLO model: %s (device=%s)", weights, self.config.device)

        device = self.config.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = YOLO(weights)
        self._model.to(device)
        self._class_names = self._model.names  # {0: 'person', 1: 'bicycle', …}
        logger.info(
            "Model loaded – %d classes, device=%s", len(self._class_names), device
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray, **overrides: Any) -> DetectionResult:
        """Run detection on a single BGR image.

        Extra *overrides* are forwarded to ``model.predict()``.
        """
        self._ensure_model()

        conf = overrides.pop("conf", self.config.confidence_threshold)
        iou = overrides.pop("iou", self.config.iou_threshold)
        imgsz = overrides.pop("imgsz", self.config.image_size)
        max_det = overrides.pop("max_det", self.config.max_detections)

        t0 = time.perf_counter()
        results = self._model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            verbose=False,
            **overrides,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return self._parse_results(results[0], elapsed_ms, image.shape[:2])

    def detect_batch(
        self, images: list[np.ndarray], **overrides: Any
    ) -> list[DetectionResult]:
        """Run detection on a batch of BGR images."""
        self._ensure_model()

        conf = overrides.pop("conf", self.config.confidence_threshold)
        iou = overrides.pop("iou", self.config.iou_threshold)
        imgsz = overrides.pop("imgsz", self.config.image_size)

        t0 = time.perf_counter()
        all_results = self._model.predict(
            source=images,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            **overrides,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_image_ms = elapsed_ms / max(len(images), 1)

        return [
            self._parse_results(r, per_image_ms, images[i].shape[:2])
            for i, r in enumerate(all_results)
        ]

    @property
    def class_names(self) -> dict[int, str]:
        self._ensure_model()
        return dict(self._class_names)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_results(
        self, result: Any, elapsed_ms: float, img_shape: tuple[int, int]
    ) -> DetectionResult:
        detections: list[Detection] = []
        boxes = result.boxes

        if boxes is not None and len(boxes):
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                h, w = img_shape
                bbox_norm = (x1 / w, y1 / h, x2 / w, y2 / h) if w and h else None

                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=self._class_names.get(cls_id, str(cls_id)),
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        bbox_norm=bbox_norm,
                    )
                )

        return DetectionResult(
            detections=detections,
            inference_time_ms=elapsed_ms,
            image_shape=img_shape,
        )
