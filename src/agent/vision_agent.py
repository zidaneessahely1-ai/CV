"""AI Agent that reacts to YOLO detection results.

The ``VisionAgent`` consumes ``DetectionResult`` objects and derives
higher-level insights such as:

* Object counting per class
* Alerting when specific classes are detected
* Zone-based monitoring (region of interest)
* Trend tracking over a sliding window
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from src.detection.detector import DetectionResult
from src.utils.config import AgentConfig, load_config
from src.utils.logging import setup_logger

logger = setup_logger("yolo_agent.agent")


@dataclass
class AgentAlert:
    """An alert emitted by the agent."""

    timestamp: float
    alert_type: str          # "class_detected", "zone_intrusion", "count_change"
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "alert_type": self.alert_type,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class ZoneOfInterest:
    """Rectangular region of interest (normalised 0-1 coordinates)."""

    name: str
    x1: float
    y1: float
    x2: float
    y2: float

    def contains_center(self, bbox: tuple[float, float, float, float]) -> bool:
        """Check whether the *centre* of a detection bbox falls in the zone."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2


class VisionAgent:
    """Stateful agent that analyses detection results over time.

    Parameters
    ----------
    config : AgentConfig | None
        Agent-specific configuration.  When *None* the default is loaded.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or load_config().agent
        self._history: deque[DetectionResult] = deque(
            maxlen=self.config.max_history,
        )
        self._zones: list[ZoneOfInterest] = []
        self._last_counts: dict[str, int] = {}
        logger.info("VisionAgent initialised: %s", self.config.name)

    # ------------------------------------------------------------------
    # Zone management
    # ------------------------------------------------------------------

    def add_zone(self, zone: ZoneOfInterest) -> None:
        self._zones.append(zone)
        logger.info("Zone added: %s", zone.name)

    def clear_zones(self) -> None:
        self._zones.clear()

    @property
    def zones(self) -> list[ZoneOfInterest]:
        return list(self._zones)

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def process(self, result: DetectionResult) -> list[AgentAlert]:
        """Analyse a detection result and return any alerts."""
        self._history.append(result)
        now = time.time()
        alerts: list[AgentAlert] = []

        # 1. Alert-class detection
        for det in result.filter_by_class(self.config.alert_classes):
            alerts.append(
                AgentAlert(
                    timestamp=now,
                    alert_type="class_detected",
                    message=f"Alert class detected: {det.class_name} "
                    f"(conf={det.confidence:.2f})",
                    details=det.to_dict(),
                )
            )

        # 2. Count changes
        current_counts = result.class_counts()
        for cls in self.config.counting_classes:
            cur = current_counts.get(cls, 0)
            prev = self._last_counts.get(cls, 0)
            if cur != prev:
                alerts.append(
                    AgentAlert(
                        timestamp=now,
                        alert_type="count_change",
                        message=f"{cls} count changed: {prev} → {cur}",
                        details={"class": cls, "previous": prev, "current": cur},
                    )
                )
        self._last_counts = current_counts

        # 3. Zone intrusion
        for zone in self._zones:
            for det in result.detections:
                if det.bbox_norm and zone.contains_center(det.bbox_norm):
                    alerts.append(
                        AgentAlert(
                            timestamp=now,
                            alert_type="zone_intrusion",
                            message=f"{det.class_name} entered zone '{zone.name}'",
                            details={
                                "zone": zone.name,
                                "detection": det.to_dict(),
                            },
                        )
                    )

        if alerts:
            logger.info("Agent emitted %d alert(s)", len(alerts))

        return alerts

    # ------------------------------------------------------------------
    # Summary / reporting helpers
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return an aggregate summary of all recorded history."""
        total_dets = sum(r.count for r in self._history)
        avg_time = (
            sum(r.inference_time_ms for r in self._history) / len(self._history)
            if self._history
            else 0
        )
        all_counts: dict[str, int] = {}
        for r in self._history:
            for cls, cnt in r.class_counts().items():
                all_counts[cls] = all_counts.get(cls, 0) + cnt

        return {
            "frames_processed": len(self._history),
            "total_detections": total_dets,
            "avg_inference_ms": round(avg_time, 2),
            "cumulative_class_counts": all_counts,
            "zones": [z.name for z in self._zones],
        }

    def reset(self) -> None:
        """Clear all history and counts."""
        self._history.clear()
        self._last_counts.clear()
        logger.info("Agent state reset")
