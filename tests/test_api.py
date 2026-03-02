"""Tests for the FastAPI endpoints (no model required)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.agent.vision_agent import VisionAgent
from src.detection.detector import DetectionResult, YOLODetector


@pytest.fixture()
def client():
    """Create a test client with mocked detector and agent."""
    import src.api.main as api_mod

    mock_detector = MagicMock(spec=YOLODetector)
    mock_detector.detect.return_value = DetectionResult(
        detections=[], inference_time_ms=5.0, image_shape=(480, 640)
    )

    mock_agent = MagicMock(spec=VisionAgent)
    mock_agent.process.return_value = []
    mock_agent.summary.return_value = {
        "frames_processed": 0,
        "total_detections": 0,
        "avg_inference_ms": 0,
        "cumulative_class_counts": {},
        "zones": [],
    }

    # Inject mocks via the lifespan so the TestClient doesn't create real instances
    @asynccontextmanager
    async def _test_lifespan(app):
        api_mod._detector = mock_detector
        api_mod._agent = mock_agent
        yield
        api_mod._detector = None
        api_mod._agent = None

    original_lifespan = api_mod.app.router.lifespan_context
    api_mod.app.router.lifespan_context = _test_lifespan

    with TestClient(api_mod.app, raise_server_exceptions=False) as tc:
        yield tc

    api_mod.app.router.lifespan_context = original_lifespan


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestDetectEndpoint:
    def test_detect_no_file(self, client):
        resp = client.post("/detect")
        assert resp.status_code == 422  # Validation error

    def test_detect_with_valid_image(self, client):
        # Create a minimal valid JPEG (smallest valid JPEG)
        import io

        import numpy as np
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(buf, "JPEG")
        buf.seek(0)

        resp = client.post("/detect", files={"file": ("test.jpg", buf, "image/jpeg")})
        assert resp.status_code == 200
        data = resp.json()
        assert "result" in data
        assert "alerts" in data


class TestAgentEndpoints:
    def test_summary(self, client):
        resp = client.get("/agent/summary")
        assert resp.status_code == 200

    def test_add_zone(self, client):
        resp = client.post(
            "/agent/zone",
            json={"name": "test_zone", "x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.5},
        )
        assert resp.status_code == 200

    def test_reset(self, client):
        resp = client.delete("/agent/reset")
        assert resp.status_code == 200
