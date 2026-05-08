import cv2
import numpy as np
import base64
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.core.engine import VisionSystem
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.system = VisionSystem()
    yield
    VisionSystem.shutdown()


app = FastAPI(title="Drowsiness Detection API", version="1.0.0", lifespan=lifespan)


class FrameRequest(BaseModel):
    image_base64: str


class AnalysisResponse(BaseModel):
    is_drowsy: bool
    is_yawning: bool
    face_detected: bool
    ear: float
    mar: float
    pitch: float          
    status: str


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_frame(data: FrameRequest):
    try:
        raw = data.image_base64
        if "," in raw:
            raw = raw.split(",", 1)[1]

        img_bytes = base64.b64decode(raw)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Could not decode image. Check base64 encoding.")

        result = app.state.system.process_frame(frame)

        return AnalysisResponse(
            is_drowsy=result.is_drowsy,
            is_yawning=result.is_yawning,
            face_detected=result.face_detected,
            ear=round(result.ear, 3),
            mar=round(result.mar, 3),
            pitch=round(result.pitch, 1),
            status="DANGER" if result.is_drowsy else ("WARNING" if result.is_yawning else "OK"),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)