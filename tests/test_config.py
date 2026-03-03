"""Tests for configuration management."""

from src.utils.config import AppConfig, ModelConfig, load_config


def test_default_model_config():
    cfg = ModelConfig()
    assert cfg.name == "yolo11l"
    assert cfg.confidence_threshold == 0.25
    assert cfg.device == "auto"
    assert cfg.image_size == 640


def test_load_config_returns_app_config():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.model.name == "yolo11l"
    assert cfg.agent.name == "VisionAgent"
    assert cfg.api.port == 8000


def test_load_config_with_missing_file():
    """When the path doesn't exist, defaults are used."""
    cfg = load_config("/tmp/nonexistent.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.model.weights == "yolo11l.pt"


def test_env_override(monkeypatch):
    monkeypatch.setenv("YOLO_AGENT_MODEL_DEVICE", "cpu")
    cfg = load_config()
    assert cfg.model.device == "cpu"
