"""
Geometric Transformation Models for Planetary Cartography.
Supports Rigid (3-DOF), Similarity (4-DOF), Affine (6-DOF), and Homography (8-DOF).
"""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import cv2

class GeometricModel(ABC):
    def __init__(self, matrix: np.ndarray, name: str, dof: int):
        self.matrix = matrix.astype(np.float32)
        self.name = name
        self.dof = dof

    @abstractmethod
    def transform(self, points: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, points: np.ndarray) -> np.ndarray:
        pass

class SimilarityModel(GeometricModel):
    def __init__(self, matrix: np.ndarray):
        # 2x3 matrix [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]
        super().__init__(matrix, name="SIMILARITY", dof=4)
        self.inv_matrix = cv2.invertAffineTransform(self.matrix)

    def transform(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return cv2.transform(pts.reshape(-1, 1, 2), self.matrix).reshape(-1, 2)

    def inverse_transform(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return cv2.transform(pts.reshape(-1, 1, 2), self.inv_matrix).reshape(-1, 2)

class AffineModel(GeometricModel):
    def __init__(self, matrix: np.ndarray):
        # 2x3 matrix
        super().__init__(matrix, name="AFFINE", dof=6)
        self.inv_matrix = cv2.invertAffineTransform(self.matrix)

    def transform(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return cv2.transform(pts.reshape(-1, 1, 2), self.matrix).reshape(-1, 2)

    def inverse_transform(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return cv2.transform(pts.reshape(-1, 1, 2), self.inv_matrix).reshape(-1, 2)

class HomographyModel(GeometricModel):
    def __init__(self, matrix: np.ndarray):
        # 3x3 matrix
        super().__init__(matrix, name="HOMOGRAPHY", dof=8)
        ret, inv_m = cv2.invert(self.matrix)
        self.inv_matrix = inv_m if ret else np.eye(3, dtype=np.float32)

    def transform(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), self.matrix).reshape(-1, 2)

    def inverse_transform(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), self.inv_matrix).reshape(-1, 2)


