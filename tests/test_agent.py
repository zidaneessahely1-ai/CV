"""Tests for the VisionAgent."""

from src.agent.vision_agent import AgentAlert, VisionAgent, ZoneOfInterest
from src.detection.detector import Detection, DetectionResult
from src.utils.config import AgentConfig


def _make_result(detections: list[Detection] | None = None) -> DetectionResult:
    return DetectionResult(
        detections=detections or [],
        inference_time_ms=10.0,
        image_shape=(480, 640),
    )


def _make_detection(cls_name: str = "person", conf: float = 0.9) -> Detection:
    return Detection(
        class_id=0,
        class_name=cls_name,
        confidence=conf,
        bbox=(100.0, 100.0, 200.0, 200.0),
        bbox_norm=(0.15, 0.2, 0.3, 0.4),
    )


class TestVisionAgent:
    def test_process_empty_result(self):
        agent = VisionAgent(config=AgentConfig())
        alerts = agent.process(_make_result())
        assert alerts == []

    def test_alert_class_detected(self):
        agent = VisionAgent(config=AgentConfig(alert_classes=["person"]))
        result = _make_result([_make_detection("person")])
        alerts = agent.process(result)
        alert_types = [a.alert_type for a in alerts]
        assert "class_detected" in alert_types

    def test_count_change_alert(self):
        agent = VisionAgent(config=AgentConfig(counting_classes=["car"]))
        # First frame: 0 cars (baseline)
        agent.process(_make_result())
        # Second frame: 1 car → count_change
        alerts = agent.process(_make_result([_make_detection("car")]))
        assert any(a.alert_type == "count_change" for a in alerts)

    def test_zone_intrusion(self):
        agent = VisionAgent(config=AgentConfig(alert_classes=[]))
        zone = ZoneOfInterest(name="entrance", x1=0.0, y1=0.0, x2=0.5, y2=0.5)
        agent.add_zone(zone)

        det = _make_detection("person")  # bbox_norm centre = (0.225, 0.3) → inside zone
        result = _make_result([det])
        alerts = agent.process(result)
        assert any(a.alert_type == "zone_intrusion" for a in alerts)

    def test_summary(self):
        agent = VisionAgent(config=AgentConfig())
        agent.process(_make_result([_make_detection("person")]))
        summary = agent.summary()
        assert summary["frames_processed"] == 1
        assert summary["total_detections"] == 1

    def test_reset(self):
        agent = VisionAgent(config=AgentConfig())
        agent.process(_make_result([_make_detection()]))
        agent.reset()
        summary = agent.summary()
        assert summary["frames_processed"] == 0


class TestZoneOfInterest:
    def test_contains_center_inside(self):
        zone = ZoneOfInterest(name="z", x1=0.0, y1=0.0, x2=1.0, y2=1.0)
        assert zone.contains_center((0.2, 0.2, 0.4, 0.4)) is True

    def test_contains_center_outside(self):
        zone = ZoneOfInterest(name="z", x1=0.0, y1=0.0, x2=0.1, y2=0.1)
        assert zone.contains_center((0.5, 0.5, 0.7, 0.7)) is False


class TestDetectionResult:
    def test_class_counts(self):
        dets = [_make_detection("person"), _make_detection("car"), _make_detection("person")]
        result = _make_result(dets)
        counts = result.class_counts()
        assert counts == {"person": 2, "car": 1}

    def test_filter_by_class(self):
        dets = [_make_detection("person"), _make_detection("car")]
        result = _make_result(dets)
        filtered = result.filter_by_class(["car"])
        assert len(filtered) == 1
        assert filtered[0].class_name == "car"

    def test_filter_by_confidence(self):
        dets = [
            _make_detection("person"),  # conf=0.9
            Detection(
                class_id=1,
                class_name="car",
                confidence=0.3,
                bbox=(10, 10, 50, 50),
            ),
        ]
        result = _make_result(dets)
        high_conf = result.filter_by_confidence(0.5)
        assert len(high_conf) == 1

    def test_to_dict(self):
        result = _make_result([_make_detection()])
        d = result.to_dict()
        assert "count" in d
        assert "detections" in d
        assert d["count"] == 1
