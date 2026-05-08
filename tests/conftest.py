import numpy as np
import pytest


@pytest.fixture
def blank_frame() -> np.ndarray:
    """640x480 чёрный кадр — лицо не будет найдено."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def eye_open() -> np.ndarray:
    """Координаты открытого глаза (синтетические)."""
    return np.array([
        [0, 4], [2, 6], [4, 6],
        [6, 4], [4, 2], [2, 2],
    ], dtype=np.float64)


@pytest.fixture
def eye_closed() -> np.ndarray:
    """Координаты закрытого глаза."""
    return np.array([
        [0, 4], [2, 4.1], [4, 4.1],
        [6, 4], [4, 3.9], [2, 3.9],
    ], dtype=np.float64)


@pytest.fixture
def mouth_open() -> np.ndarray:
    """8 точек внутреннего рта — зевота."""
    return np.array([
        [0, 5], [1, 5], [3, 8],
        [5, 5], [6, 5], [5, 1],
        [3, 0], [1, 1],
    ], dtype=np.float64)


@pytest.fixture
def mouth_closed() -> np.ndarray:
    """8 точек внутреннего рта — закрыт."""
    return np.array([
        [0, 5], [1, 5], [3, 5.2],
        [5, 5], [6, 5], [5, 4.8],
        [3, 5], [1, 5],
    ], dtype=np.float64)