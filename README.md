# 🔍 YOLO Vision Agent

AI Agent powered by **YOLOv11-L** (Large) for real-time object detection, analysis, and alerting — built with FastAPI, Streamlit, and Docker.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![YOLOv11](https://img.shields.io/badge/YOLO-v11--Large-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)

---

## Architecture

```
Image / Video Frame
       │
       ▼
┌──────────────┐      ┌───────────────┐      ┌──────────────┐
│ YOLOv11-L    │─────▶│ Vision Agent  │─────▶│ Alerts /     │
│ Detector     │      │ (Analysis)    │      │ Dashboard    │
└──────────────┘      └───────────────┘      └──────────────┘
       │                                            │
       ▼                                            ▼
  DetectionResult                          REST API (FastAPI)
  • bboxes, classes                        Streamlit Dashboard
  • confidence scores
```

**Components**

| Module | Description |
|--------|-------------|
| `src/detection/detector.py` | YOLOv11-L wrapper — runs inference, returns structured `DetectionResult` objects |
| `src/agent/vision_agent.py` | Stateful agent — alert rules, object counting, zone-of-interest monitoring |
| `src/api/main.py` | FastAPI REST API — upload images, get detections + alerts |
| `src/dashboard/app.py` | Streamlit UI — interactive detection with bounding-box visualisation |
| `src/utils/` | Config management, logging, image processing helpers |

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/zidaneessahely1-ai/CV.git
cd CV
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The first request will automatically download the `yolo11l.pt` weights (~50 MB) from Ultralytics Hub.

### 3. Run the Dashboard

```bash
streamlit run src/dashboard/app.py
```

### 4. Try a detection

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@image.jpg" | python -m json.tool
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/detect` | Detect objects in an uploaded image. Query params: `confidence` (float), `annotate` (bool) |
| `POST` | `/detect/batch` | Detect objects in multiple images |
| `GET` | `/agent/summary` | Agent statistics (frames processed, cumulative counts) |
| `POST` | `/agent/zone` | Add a zone-of-interest for intrusion monitoring |
| `DELETE` | `/agent/reset` | Reset agent history and counters |
| `GET` | `/health` | Health check |

**Example — detection with annotated image response:**

```bash
curl -X POST "http://localhost:8000/detect?annotate=true" \
  -F "file=@image.jpg" --output annotated.jpg
```

---

## Configuration

Edit `configs/default.yaml` or use environment variables:

| Env Variable | Description | Default |
|---|---|---|
| `YOLO_AGENT_MODEL_DEVICE` | Inference device (`cpu`, `cuda`, `auto`) | `auto` |
| `YOLO_AGENT_MODEL_WEIGHTS` | Model weights file | `yolo11l.pt` |
| `YOLO_AGENT_MODEL_CONFIDENCE` | Confidence threshold | `0.25` |
| `YOLO_AGENT_API_HOST` | API bind host | `0.0.0.0` |
| `YOLO_AGENT_API_PORT` | API bind port | `8000` |

---

## Docker Deployment

```bash
# Build and run both API + Dashboard
docker compose up --build

# API  → http://localhost:8000
# Dashboard → http://localhost:8501
```

For GPU support, add the NVIDIA runtime to `docker-compose.yml`:

```yaml
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

---

## Project Structure

```
.
├── configs/
│   └── default.yaml          # Application configuration
├── data/
│   ├── uploads/              # Uploaded images
│   └── outputs/              # Annotated outputs
├── src/
│   ├── agent/
│   │   └── vision_agent.py   # AI Agent (alerts, counting, zones)
│   ├── api/
│   │   └── main.py           # FastAPI REST API
│   ├── dashboard/
│   │   └── app.py            # Streamlit dashboard
│   ├── detection/
│   │   └── detector.py       # YOLOv11-L detector wrapper
│   └── utils/
│       ├── config.py          # YAML + env config loader
│       ├── image_processing.py # OpenCV image helpers
│       └── logging.py         # Structured logging
├── tests/
│   ├── test_agent.py
│   ├── test_api.py
│   ├── test_config.py
│   └── test_image_processing.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Testing

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

All tests run without a GPU or model weights (detection is mocked in API tests).

---

## Model Details

| Property | Value |
|----------|-------|
| Model | YOLOv11-L (Large) |
| Framework | Ultralytics |
| Classes | 80 (COCO dataset) |
| Input size | 640 × 640 |
| Weights | Auto-downloaded `yolo11l.pt` |

---

## License

MIT
