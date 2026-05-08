# Driver AI Guard 

Real-time driver drowsiness detection system using computer vision, served as a REST API with a live web dashboard and MLOps experiment tracking.

![CI](https://github.com/ТУТ_ТВОЙ_GITHUB_НИК/driver-ai-guard/actions/workflows/ci.yml/badge.svg)

## Overview

The system monitors a driver's face through a webcam and detects:
- **Drowsiness** via Eye Aspect Ratio (EAR) — eyes closing over time
- **Yawning** via Mouth Aspect Ratio (MAR) — mouth opening wide
- **Head drop** via Pitch angle estimation (PoseEstimator + OpenCV SolvePnP)

When any threshold is exceeded for a sustained number of frames, a visual alert is triggered.

## Architecture
driver-ai-guard/
├── app/
│   ├── core/
│   │   ├── config.py       # Centralized settings (pydantic-settings)
│   │   ├── engine.py       # Main VisionSystem — frame processing pipeline
│   │   ├── physics.py      # Stateless EAR/MAR calculations
│   │   ├── pose.py         # Head pose estimation (PnP solver)
│   │   ├── visualizer.py   # HUD rendering
│   │   └── logger.py       # MLflow experiment tracker
│   ├── server.py           # FastAPI REST API
│   └── web_ui.py           # Streamlit WebRTC dashboard
├── tests/
│   ├── conftest.py
│   ├── test_physics.py
│   └── test_engine.py
├── models/                 # Place .dat model file here (see Setup)
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml

## Quick Start

### 1. Download the face landmark model

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat models/
```

### 2. Run with Docker (recommended)

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| FastAPI docs | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

### 3. Run locally

```bash
pip install -e ".[dev]"
python main.py --model models/shape_predictor_68_face_landmarks.dat
```

## How It Works

**EAR (Eye Aspect Ratio)**
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

When drowsy, EAR stays consistently low. The system counts consecutive frames below the threshold and triggers an alert after `persistence_frames` frames.

**MAR (Mouth Aspect Ratio)** — same principle applied to inner lip landmarks (points 60–67) to detect yawning.

**Pitch (Head Pose)** — uses OpenCV `solvePnP` to compute the vertical tilt angle of the head. A pitch below `-15°` indicates the head is dropping, which is a strong drowsiness signal independent of eye state.

## Configuration

All parameters can be overridden via environment variables or a `.env` file:

```env
EAR_THRESHOLD=0.25
MAR_THRESHOLD=0.50
PERSISTENCE_FRAMES=15
PITCH_THRESHOLD=-15.0
CAMERA_INDEX=0
MLFLOW_TRACKING_URI=file:./mlruns
```

## Testing

```bash
pytest --cov=app
```

## API

`POST /analyze`

```json
{
  "image_base64": "<base64 encoded JPEG/PNG frame>"
}
```

Response:
```json
{
  "is_drowsy": false,
  "is_yawning": false,
  "face_detected": true,
  "ear": 0.312,
  "mar": 0.143,
  "pitch": -4.2,
  "status": "OK"
}
```

## Tech Stack

- **CV**: OpenCV, dlib (68-point facial landmarks)
- **API**: FastAPI + Uvicorn
- **UI**: Streamlit + streamlit-webrtc
- **MLOps**: MLflow
- **Containerization**: Docker + docker-compose
- **Testing**: pytest + pytest-cov
- **CI**: GitHub Actions