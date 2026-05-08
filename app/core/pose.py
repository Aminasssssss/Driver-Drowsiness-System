import cv2
import numpy as np

class PoseEstimator:
    """
    Advanced Head Pose Estimation using OpenCV SolvePnP.
    Calculates Pitch, Yaw, and Roll to detect head dropping/nodding.
    """
    def __init__(self):
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             
            (0.0, -330.0, -65.0),        
            (-225.0, 170.0, -135.0),     
            (225.0, 170.0, -135.0),     
            (-150.0, -150.0, -125.0),   
            (150.0, -150.0, -125.0)      
        ], dtype=np.float64)

    def estimate(self, landmarks: np.ndarray, frame_shape: tuple) -> float:
        """
        Computes the 'Pitch' (vertical tilt) of the head.
        Returns the angle in degrees. Positive = Looking up, Negative = Looking down.
        """
        size = frame_shape
        image_points = np.array([
            landmarks[30],     
            landmarks[8],      
            landmarks[36],     
            landmarks[45],     
            landmarks[48],     
            landmarks[54]     
        ], dtype=np.float64)

        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1)) 

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.decomposeProjectionMatrix(np.hstack((rmat, translation_vector)))
        
        pitch = angles[0] 
        return pitch