import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from app.core.engine import VisionSystem, DetectionResult


@pytest.fixture
def mock_vision_system():
    """VisionSystem с замоканным dlib (не нужна реальная модель)."""
    with patch("app.core.engine.dlib.get_frontal_face_detector") as mock_det, \
         patch("app.core.engine.dlib.shape_predictor") as mock_pred:
        system = VisionSystem(predictor_path="fake_path.dat")
        system._mock_detector = mock_det.return_value
        system._mock_predictor = mock_pred.return_value
        yield system


class TestVisionSystemNoFace:
    def test_no_face_returns_safe_result(self, mock_vision_system, blank_frame):
        mock_vision_system.detector = MagicMock(return_value=[])
        result = mock_vision_system.process_frame(blank_frame)

        assert isinstance(result, DetectionResult)
        assert result.is_drowsy is False
        assert result.is_yawning is False
        assert result.face_detected is False
        assert result.ear == 0.0
        assert result.mar == 0.0

    def test_drowsy_counter_decreases_without_face(self, mock_vision_system, blank_frame):
        mock_vision_system.drowsy_counter = 10
        mock_vision_system.detector = MagicMock(return_value=[])
        mock_vision_system.process_frame(blank_frame)
        assert mock_vision_system.drowsy_counter == 9

    def test_reset_clears_state(self, mock_vision_system):
        mock_vision_system.drowsy_counter = 99
        mock_vision_system.reset()
        assert mock_vision_system.drowsy_counter == 0
        assert len(mock_vision_system.ear_history) == 0