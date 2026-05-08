from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    predictor_path: str = "models/shape_predictor_68_face_landmarks.dat"

    ear_threshold: float = 0.25
    mar_threshold: float = 0.50
    persistence_frames: int = 15
    pitch_threshold: float = -15.0    

    camera_index: int = 0
    resize_width: int = 800

    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "Driver_Drowsiness_Detection"
    mlflow_log_every_n_frames: int = 10

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()