import cv2
import argparse
from app.core.engine import VisionSystem
from app.core.visualizer import Visualizer
from app.core.logger import MLOpsTracker


def run_local(model_path: str, camera_index: int) -> None:
    params = {
        "ear_threshold": 0.22,
        "mar_threshold": 0.50,
        "persistence_frames": 20,  
    }
    system = VisionSystem(predictor_path=model_path, **params)
    tracker = MLOpsTracker()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    tracker.start_session(params)
    frame_count = 0
    print(" Система запущена. Нажмите 'q' для выхода.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            result = system.process_frame(frame)

            if frame_count % 10 == 0:
                tracker.log_metrics({
                    "ear": result.ear,
                    "mar": result.mar,
                    "pitch": result.pitch,
                    "is_drowsy": 1 if result.is_drowsy else 0,
                    "face_detected": 1 if result.face_detected else 0,
                }, step=frame_count)

            output = Visualizer.render_pipeline(result)
            cv2.imshow("Driver Monitoring HUD", output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.end_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver drowsiness detection — local mode")
    parser.add_argument("--model", default="models/shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()
    run_local(args.model, args.cam)