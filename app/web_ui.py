import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import os

from app.core.engine import VisionSystem
from app.core.visualizer import Visualizer

st.set_page_config(page_title="Driver AI Guard", layout="wide")

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        model_path = "models/shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(model_path):
            st.error(f"Модель не найдена по пути: {model_path}")
        
        self.system = VisionSystem(predictor_path=model_path)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        result = self.system.process_frame(img)
        
        annotated_frame = Visualizer.render_pipeline(result)
        
        return annotated_frame

def main():
    st.title("🚗 AI Driver Drowsiness Detector")
    st.sidebar.title("Настройки системы")
    
    st.markdown("""
    ### Мониторинг безопасности в реальном времени
    Система анализирует:
    1. **EAR** (Eye Aspect Ratio) — сонливость по глазам.
    2. **MAR** (Mouth Aspect Ratio) — детекция зевоты.
    3. **Pitch Angle** — наклон головы (засыпание «носом вниз»).
    """)

    ctx = webrtc_streamer(
        key="driver-safety",
        video_processor_factory=VideoProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.video_processor:
        st.success("Камера подключена. Анализ запущен.")
    
    st.sidebar.info("Параметры MLOps логируются в MLflow автоматически.")

if __name__ == "__main__":
    main()