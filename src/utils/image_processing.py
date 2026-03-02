"""Image pre- and post-processing utilities."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image


def load_image(source: str | Path | BinaryIO | bytes) -> np.ndarray:
    """Load an image from various sources and return a BGR numpy array."""
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {source}")
        return img

    if isinstance(source, bytes):
        source = io.BytesIO(source)

    # File-like object
    pil = Image.open(source).convert("RGB")
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def save_image(image: np.ndarray, path: str | Path) -> Path:
    """Write a BGR image to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return path


def draw_detections(
    image: np.ndarray,
    detections: list[dict],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw bounding boxes and labels on a BGR image (returns a copy)."""
    canvas = image.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = f"{det['class_name']} {det['confidence']:.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return canvas


def image_to_bytes(image: np.ndarray, fmt: str = ".jpg") -> bytes:
    """Encode a BGR image to bytes (JPEG by default)."""
    ok, buf = cv2.imencode(fmt, image)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return buf.tobytes()
