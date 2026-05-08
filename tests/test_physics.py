import numpy as np
import pytest
from app.core.physics import PhysicsEngine


class TestEAR:
    def test_open_eye_above_threshold(self, eye_open):
        ear = PhysicsEngine.compute_ear(eye_open)
        assert ear > 0.25, f"Open eye EAR should be > 0.25, got {ear:.3f}"

    def test_closed_eye_below_threshold(self, eye_closed):
        ear = PhysicsEngine.compute_ear(eye_closed)
        assert ear < 0.1, f"Closed eye EAR should be < 0.1, got {ear:.3f}"

    def test_ear_is_float(self, eye_open):
        assert isinstance(PhysicsEngine.compute_ear(eye_open), float)

    def test_ear_is_non_negative(self, eye_open):
        assert PhysicsEngine.compute_ear(eye_open) >= 0


class TestMAR:
    def test_open_mouth_above_threshold(self, mouth_open):
        mar = PhysicsEngine.compute_mar(mouth_open)
        assert mar > 0.5, f"Open mouth MAR should be > 0.5, got {mar:.3f}"

    def test_closed_mouth_below_threshold(self, mouth_closed):
        mar = PhysicsEngine.compute_mar(mouth_closed)
        assert mar < 0.1, f"Closed mouth MAR should be < 0.1, got {mar:.3f}"

    def test_mar_is_float(self, mouth_open):
        assert isinstance(PhysicsEngine.compute_mar(mouth_open), float)