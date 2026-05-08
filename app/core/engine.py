import cv2
import dlib
import numpy as np
from typing import Optional
from collections import deque
from dataclasses import dataclass

from .physics import PhysicsEngine
from .pose import PoseEstimator
from .config import settings


@dataclass
class DetectionResult:
    is_drowsy: bool
    is_yawning: bool
    ear: float
    mar: float
    pitch: float                   
    frame_overlay: np.ndarray
    face_detected: bool = False
    landmarks: Optional[np.ndarray] = None


class VisionSystem:

    def __init__(
        self,
        predictor_path: Optional[str] = None,
        ear_threshold: Optional[float] = None,
        mar_threshold: Optional[float] = None,
        persistence_frames: Optional[int] = None,
        pitch_threshold: float = -15.0,  
        **kwargs,
    ):
        _path = predictor_path or settings.predictor_path
        self.EAR_THRESH = ear_threshold or settings.ear_threshold
        self.MAR_THRESH = mar_threshold or settings.mar_threshold
        self.PERSISTENCE = persistence_frames or settings.persistence_frames
        self.PITCH_THRESH = pitch_threshold

        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor(_path)
        except RuntimeError:
            raise FileNotFoundError(
                f"Model not found at '{_path}'. "
                "Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )

        self.pose_estimator = PoseEstimator()   

        self.ear_history: deque = deque(maxlen=self.PERSISTENCE * 2)
        self.mar_history: deque = deque(maxlen=self.PERSISTENCE * 2)
        self.drowsy_counter: int = 0

    def _get_landmarks(self, gray: np.ndarray, rect: dlib.rectangle) -> np.ndarray:
        shape = self.predictor(gray, rect)
        return np.array([[p.x, p.y] for p in shape.parts()])

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        if w > settings.resize_width:
            scale = settings.resize_width / w
            frame = cv2.resize(frame, (settings.resize_width, int(h * scale)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return frame, clahe.apply(gray)

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        frame, gray = self._preprocess(frame)
        faces = self.detector(gray, 0)

        if not faces:
            self.drowsy_counter = max(0, self.drowsy_counter - 1)
            return DetectionResult(
                is_drowsy=False,
                is_yawning=False,
                ear=0.0,
                mar=0.0,
                pitch=0.0,
                frame_overlay=frame,
                face_detected=False,
            )

        landmarks = self._get_landmarks(gray, faces[0])

        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        inner_mouth = landmarks[60:68]

        l_ear = PhysicsEngine.compute_ear(left_eye)
        r_ear = PhysicsEngine.compute_ear(right_eye)
        ear = (l_ear + r_ear) / 2.0
        mar = PhysicsEngine.compute_mar(inner_mouth)

        try:
            pitch = float(self.pose_estimator.estimate(landmarks, frame.shape))
        except Exception:
            pitch = 0.0  
        self.ear_history.append(ear)
        self.mar_history.append(mar)

        ear_drowsy = ear < self.EAR_THRESH
        pitch_drowsy = pitch < self.PITCH_THRESH

        if ear_drowsy or pitch_drowsy:
            self.drowsy_counter += 1
        else:
            self.drowsy_counter = max(0, self.drowsy_counter - 1)

        return DetectionResult(
            is_drowsy=self.drowsy_counter >= self.PERSISTENCE,
            is_yawning=mar > self.MAR_THRESH,
            ear=ear,
            mar=mar,
            pitch=round(pitch, 1),
            frame_overlay=frame,
            face_detected=True,
            landmarks=landmarks,
        )

    def reset(self) -> None:
        self.drowsy_counter = 0
        self.ear_history.clear()
        self.mar_history.clear()

    @staticmethod
    def shutdown() -> None:
        cv2.destroyAllWindows()