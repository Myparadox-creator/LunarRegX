"""
Multi-Resolution Pyramids for Multi-Scale Lunar Imagery.
Handles scale differences (e.g. OHRC 0.25m vs LROC NAC 0.5m) via coarse-to-fine pyramids.
"""
from typing import List, Tuple
import numpy as np
import cv2

class ImagePyramid:
    def __init__(self, image: np.ndarray, num_levels: int = 3, scale_factor: float = 2.0):
        self.image = image
        self.num_levels = num_levels
        self.scale_factor = scale_factor
        self.levels = self._build_pyramid()

    def _build_pyramid(self) -> List[np.ndarray]:
        pyr = [self.image]
        for l in range(1, self.num_levels):
            # Downsample with anti-aliasing Gaussian blur
            prev = pyr[-1]
            h, w = prev.shape[:2]
            new_h, new_w = int(h / self.scale_factor), int(w / self.scale_factor)
            if new_h < 32 or new_w < 32:
                break
            blurred = cv2.GaussianBlur(prev, (5, 5), 1.0)
            down = cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_AREA)
            pyr.append(down)
        return pyr

    def get_scale_at_level(self, level: int) -> float:
        return self.scale_factor ** level
