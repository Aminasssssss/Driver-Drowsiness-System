import cv2
import numpy as np
from typing import Optional
from .engine import DetectionResult


class Visualizer:

    COLOR_SAFE = (0, 255, 0)
    COLOR_WARNING = (0, 255, 255)
    COLOR_DANGER = (0, 0, 255)
    COLOR_TEXT = (255, 255, 255)

    @classmethod
    def render_pipeline(cls, result: DetectionResult) -> np.ndarray:
        frame = result.frame_overlay.copy()

        if result.landmarks is not None:
            cls._draw_landmarks(frame, result.landmarks, result.is_drowsy)

        cls._draw_telemetry(frame, result)

        if result.is_drowsy:
            cls._draw_alert(frame, "!!! DROWSY DETECTED !!!", cls.COLOR_DANGER)
        elif result.is_yawning:
            cls._draw_alert(frame, "YAWN DETECTED", cls.COLOR_WARNING)

        return frame

    @staticmethod
    def _draw_landmarks(frame: np.ndarray, landmarks: np.ndarray, is_critical: bool):
        color = (0, 0, 255) if is_critical else (0, 255, 0)
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 2, color, -1, lineType=cv2.LINE_AA)

    @staticmethod
    def _draw_telemetry(frame: np.ndarray, result: DetectionResult):
        font = cv2.FONT_HERSHEY_SIMPLEX
        data = [
            (f"EAR: {result.ear:.2f}", (10, 30)),
            (f"MAR: {result.mar:.2f}", (10, 60)),
            (f"Pitch: {result.pitch:.1f} deg", (10, 90)),   
        ]
        for text, pos in data:
            cv2.putText(frame, text, (pos[0]+1, pos[1]+1), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, text, pos, font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    @staticmethod
    def _draw_alert(frame: np.ndarray, message: str, color: tuple):
        font = cv2.FONT_HERSHEY_DUPLEX
        text_size = cv2.getTextSize(message, font, 1.2, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (text_x - 10, 100), (text_x + text_size[0] + 10, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, message, (text_x, 145), font, 1.2, color, 3, cv2.LINE_AA)