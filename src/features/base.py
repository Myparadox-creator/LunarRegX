"""
Abstract Base Feature Engine.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
import numpy as np
import cv2

class BaseFeatureEngine(ABC):
    @abstractmethod
    def detect_and_compute(
        self,
        image_8u: np.ndarray,
        mask: Optional[np.ndarray] = None,
        reliability_map: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detect keypoints and compute descriptors.
        Returns:
            keypoints: List of cv2.KeyPoint
            descriptors: ndarray of shape (N, D)
        """
        pass
