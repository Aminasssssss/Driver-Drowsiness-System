import numpy as np
from scipy.spatial import distance as dist

class PhysicsEngine:
    """
    Stateless computing layer for facial metrics.
    Uses Euclidean distance to determine eye and mouth states.
    """

    @staticmethod
    def compute_ear(eye_landmarks: np.ndarray) -> float:
        """
        Calculates Eye Aspect Ratio (EAR).
        Formula: (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        a = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        b = dist.euclidean(eye_landmarks[2], eye_landmarks[4])

        c = dist.euclidean(eye_landmarks[0], eye_landmarks[3])

        ear = (a + b) / (2.0 * c)
        return ear

    @staticmethod
    def compute_mar(mouth_landmarks: np.ndarray) -> float:
        """
        Calculates Mouth Aspect Ratio (MAR) to detect yawning.
        Focuses on the inner lips points (60-68 in 68-point model).
        """
        a = dist.euclidean(mouth_landmarks[2], mouth_landmarks[6]) 
        b = dist.euclidean(mouth_landmarks[4], mouth_landmarks[0]) 

        c = dist.euclidean(mouth_landmarks[0], mouth_landmarks[4])

        mar = (a + b) / (2.0 * c)
        return mar