"""
Phase-Structural Feature Engine (RIFT-style).
Uses Log-Gabor Phase Congruency moments for keypoint detection
and Maximum Index Map (MIM) patch histograms for contrast-reversal-invariant descriptors.
"""
from typing import Tuple, Optional, List
import numpy as np
import cv2
from src.features.base import BaseFeatureEngine
from src.illumination.phase_congruency import LogGaborPhaseCongruency, PhaseCongruencyResult

class PhaseStructuralEngine(BaseFeatureEngine):
    def __init__(
        self,
        n_features: int = 1500,
        patch_size: int = 32,
        n_scales: int = 3,
        n_orientations: int = 6
    ):
        self.n_features = n_features
        self.patch_size = patch_size
        self.n_orientations = n_orientations
        self.pc_engine = LogGaborPhaseCongruency(n_scales=n_scales, n_orientations=n_orientations)

    def detect_and_compute(
        self,
        image_8u: np.ndarray,
        mask: Optional[np.ndarray] = None,
        reliability_map: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        img_norm = image_8u.astype(np.float32) / 255.0
        pc: PhaseCongruencyResult = self.pc_engine.compute(img_norm)

        # Salience map: M + m
        salience = pc.max_moment + pc.min_moment
        if reliability_map is not None:
            salience *= reliability_map
        if mask is not None:
            salience *= mask.astype(np.float32)

        salience_8u = np.clip(salience * 255.0, 0, 255).astype(np.uint8)

        # Detect corners on structural salience map using GoodFeaturesToTrack / FAST
        corners = cv2.goodFeaturesToTrack(
            salience_8u,
            maxCorners=self.n_features,
            qualityLevel=0.015,
            minDistance=7
        )

        if corners is None or len(corners) == 0:
            return [], np.empty((0, 64), dtype=np.float32)

        h, w = image_8u.shape
        half_p = self.patch_size // 2
        kps = []
        descriptors = []

        # Pad MIM for border extraction
        mim = pc.mim
        mim_padded = np.pad(mim, half_p, mode="reflect")

        # Spatial spatial sub-cells (4x4 cells in patch)
        n_cells = 4
        cell_size = self.patch_size // n_cells

        for pt in corners:
            x, y = pt[0]
            if x < 0 or x >= w or y < 0 or y >= h:
                continue

            ix, iy = int(round(x)), int(round(y))
            # Extract patch from padded MIM
            patch = mim_padded[iy:iy + self.patch_size, ix:ix + self.patch_size]

            # 4x4 spatial cells, histogram of orientations per cell
            desc_parts = []
            for cy in range(n_cells):
                for cx in range(n_cells):
                    cell = patch[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
                    hist, _ = np.histogram(cell, bins=self.n_orientations, range=(0, self.n_orientations))
                    desc_parts.extend(hist.astype(np.float32))

            desc_vec = np.array(desc_parts, dtype=np.float32)
            norm = np.linalg.norm(desc_vec)
            if norm > 1e-6:
                desc_vec /= norm
                desc_vec = np.clip(desc_vec, 0.0, 0.2)  # SIFT-style thresholding
                desc_vec /= (np.linalg.norm(desc_vec) + 1e-6)
            else:
                desc_vec = np.zeros_like(desc_vec)

            kp = cv2.KeyPoint(x=float(x), y=float(y), size=float(self.patch_size), response=float(salience[iy, ix]))
            kps.append(kp)
            descriptors.append(desc_vec)

        if not descriptors:
            return [], np.empty((0, n_cells * n_cells * self.n_orientations), dtype=np.float32)

        return kps, np.array(descriptors, dtype=np.float32)
