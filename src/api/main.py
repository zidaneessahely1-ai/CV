"""FastAPI application for the YOLO Vision Agent.

Endpoints
---------
POST /detect           – Run detection on an uploaded image.
POST /detect/batch     – Run detection on multiple images.
GET  /agent/summary    – Agent summary statistics.
POST /agent/zone       – Add a zone-of-interest.
DELETE /agent/reset     – Reset agent state.
GET  /health           – Health-check.
"""

from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.agent.vision_agent import VisionAgent, ZoneOfInterest
from src.detection.detector import YOLODetector
from src.utils.config import load_config
from src.utils.image_processing import draw_detections, image_to_bytes, load_image

logger = logging.getLogger("yolo_agent.api")

# ── Global singletons set during lifespan ────────────────────────────
_detector: YOLODetector | None = None
_agent: VisionAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up / shutdown hook: load model & agent."""
    global _detector, _agent  # noqa: PLW0603
    cfg = load_config()
    _detector = YOLODetector(config=cfg.model)
    _agent = VisionAgent(config=cfg.agent)
    logger.info("API ready – model=%s", cfg.model.name)
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="YOLO Vision Agent API",
    description="AI Agent powered by YOLOv11-L for real-time object detection",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
_cfg = load_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cfg.api.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────

class ZoneRequest(BaseModel):
    name: str
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)


class DetectQuery(BaseModel):
    confidence: float | None = None
    annotate: bool = False


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": _cfg.model.name, "timestamp": time.time()}


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    confidence: float | None = None,
    annotate: bool = False,
):
    """Run YOLOv11-L detection on an uploaded image."""
    if _detector is None:
        raise HTTPException(503, "Model not loaded")

    contents = await file.read()
    try:
        image = load_image(contents)
    except Exception as exc:
        raise HTTPException(400, f"Invalid image: {exc}") from exc

    overrides = {}
    if confidence is not None:
        overrides["conf"] = confidence

    result = _detector.detect(image, **overrides)

    # Run through agent
    alerts = []
    if _agent is not None:
        alerts = [a.to_dict() for a in _agent.process(result)]

    if annotate:
        annotated = draw_detections(image, [d.to_dict() for d in result.detections])
        img_bytes = image_to_bytes(annotated)
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")

    return JSONResponse(
        {
            "result": result.to_dict(),
            "alerts": alerts,
        }
    )


@app.post("/detect/batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    """Run detection on multiple images."""
    if _detector is None:
        raise HTTPException(503, "Model not loaded")

    images = []
    for f in files:
        data = await f.read()
        try:
            images.append(load_image(data))
        except Exception as exc:
            raise HTTPException(400, f"Invalid image ({f.filename}): {exc}") from exc

    results = _detector.detect_batch(images)
    return JSONResponse(
        {"results": [r.to_dict() for r in results]},
    )


@app.get("/agent/summary")
async def agent_summary():
    if _agent is None:
        raise HTTPException(503, "Agent not initialised")
    return JSONResponse(_agent.summary())


@app.post("/agent/zone")
async def add_zone(zone: ZoneRequest):
    if _agent is None:
        raise HTTPException(503, "Agent not initialised")
    _agent.add_zone(
        ZoneOfInterest(
            name=zone.name, x1=zone.x1, y1=zone.y1, x2=zone.x2, y2=zone.y2
        )
    )
    return {"status": "zone_added", "name": zone.name}


@app.delete("/agent/reset")
async def reset_agent():
    if _agent is None:
        raise HTTPException(503, "Agent not initialised")
    _agent.reset()
    return {"status": "reset"}


# ── Entrypoint ───────────────────────────────────────────────────────

def run() -> None:  # pragma: no cover
    cfg = load_config()
    uvicorn.run(
        "src.api.main:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    run()
