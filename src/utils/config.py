"""Configuration management for YOLO Vision Agent."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = _ROOT / "configs" / "default.yaml"


class ModelConfig(BaseModel):
    name: str = "yolo11l"
    weights: str = "yolo11l.pt"
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    max_detections: int = Field(300, ge=1)
    device: str = "auto"
    half_precision: bool = False
    image_size: int = Field(640, ge=32)


class AgentConfig(BaseModel):
    name: str = "VisionAgent"
    decision_interval: float = Field(1.0, ge=0.1)
    alert_classes: list[str] = ["person", "car", "truck", "fire hydrant"]
    counting_classes: list[str] = ["person", "car", "bicycle"]
    zone_monitoring: bool = True
    max_history: int = Field(1000, ge=1)


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535)
    cors_origins: list[str] = ["*"]
    max_upload_size_mb: int = Field(50, ge=1)


class DashboardConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(8501, ge=1, le=65535)


class PathsConfig(BaseModel):
    uploads: str = "data/uploads"
    outputs: str = "data/outputs"
    logs: str = "logs"


class AppConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    agent: AgentConfig = AgentConfig()
    api: APIConfig = APIConfig()
    dashboard: DashboardConfig = DashboardConfig()
    paths: PathsConfig = PathsConfig()


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load application config from a YAML file.

    Falls back to ``configs/default.yaml`` when *config_path* is ``None``.
    Individual values can be overridden via environment variables prefixed with
    ``YOLO_AGENT_`` (e.g. ``YOLO_AGENT_MODEL_DEVICE=cpu``).
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG

    if path.exists():
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
    else:
        raw = {}

    # Allow env-var overrides for the most common knobs
    env_map = {
        "YOLO_AGENT_MODEL_DEVICE": ("model", "device"),
        "YOLO_AGENT_MODEL_WEIGHTS": ("model", "weights"),
        "YOLO_AGENT_MODEL_CONFIDENCE": ("model", "confidence_threshold"),
        "YOLO_AGENT_API_HOST": ("api", "host"),
        "YOLO_AGENT_API_PORT": ("api", "port"),
    }
    for env_key, (section, key) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            raw.setdefault(section, {})[key] = val

    return AppConfig(**raw)
