"""
Classical Feature Detectors: SIFT, AKAZE, ORB.
"""
from typing import Tuple, Optional, List
import numpy as np
import cv2
from src.features.base import BaseFeatureEngine

class ClassicalFeatureEngine(BaseFeatureEngine):
    def __init__(
        self,
        method: str = "SIFT",
        n_features: int = 2000,
        contrast_threshold: float = 0.03,
        edge_threshold: float = 10.0
    ):
        self.method = method.upper()
        self.n_features = n_features

        if self.method == "SIFT":
            self.detector = cv2.SIFT_create(
                nfeatures=n_features,
                contrastThreshold=contrast_threshold,
                edgeThreshold=edge_threshold
            )
        elif self.method == "AKAZE":
            self.detector = cv2.AKAZE_create()
        elif self.method == "ORB":
            self.detector = cv2.ORB_create(nfeatures=n_features)
        else:
            raise ValueError(f"Unknown classical feature method: {method}")

    def detect_and_compute(
        self,
        image_8u: np.ndarray,
        mask: Optional[np.ndarray] = None,
        reliability_map: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        cv_mask = (mask * 255).astype(np.uint8) if mask is not None else None
        kps, desc = self.detector.detectAndCompute(image_8u, cv_mask)
        if desc is None or len(kps) == 0:
            return [], np.empty((0, 128 if self.method == "SIFT" else 64), dtype=np.float32)

        # If reliability map is given, update keypoint response by reliability
        if reliability_map is not None:
            h, w = reliability_map.shape
            filtered_kps = []
            filtered_indices = []
            for idx, kp in enumerate(kps):
                x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
                if 0 <= x < w and 0 <= y < h:
                    rel = reliability_map[y, x]
                    kp.response *= (0.5 + 0.5 * rel)
                    if rel > 0.05:
                        filtered_kps.append(kp)
                        filtered_indices.append(idx)
            if filtered_indices:
                kps = filtered_kps
                desc = desc[filtered_indices]

        # Convert ORB uint8 to float32 for uniform distance metrics if needed
        if desc.dtype == np.uint8 and self.method != "ORB":
            desc = desc.astype(np.float32)

        return kps, desc
