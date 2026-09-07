"""
Correspondence Matcher with Cross-Checking, Ratio Test, and Multi-Modal Confidence Scoring.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2

@dataclass
class Correspondence:
    id: int
    src_pt: Tuple[float, float]        # (x, y) in source
    ref_pt: Tuple[float, float]        # (x, y) in reference
    confidence: float                  # [0.0, 1.0] unified quality score
    dist: float                        # Descriptor distance
    ratio: float                       # d1 / d2 ratio
    subpixel_offset: Tuple[float, float] = (0.0, 0.0)  # (dx, dy) refinement
    refined_ref_pt: Optional[Tuple[float, float]] = None
    is_inlier: bool = False
    grid_cell: Optional[Tuple[int, int]] = None
    residual: float = 0.0

class CorrespondenceMatcher:
    def __init__(
        self,
        ratio_thresh: float = 0.80,
        cross_check: bool = True,
        distance_metric: str = "L2"
    ):
        self.ratio_thresh = ratio_thresh
        self.cross_check = cross_check
        self.distance_metric = distance_metric

    def match(
        self,
        kps_src: List[cv2.KeyPoint],
        desc_src: np.ndarray,
        kps_ref: List[cv2.KeyPoint],
        desc_ref: np.ndarray,
        rel_src: Optional[np.ndarray] = None,
        rel_ref: Optional[np.ndarray] = None
    ) -> List[Correspondence]:
        """
        Find mutual nearest neighbor correspondences with ratio test & quality weighting.
        """
        if len(kps_src) == 0 or len(kps_ref) == 0 or desc_src.shape[0] == 0 or desc_ref.shape[0] == 0:
            return []

        # Norm type
        norm_type = cv2.NORM_HAMMING if desc_src.dtype == np.uint8 else cv2.NORM_L2
        bf = cv2.BFMatcher(norm_type, crossCheck=False)

        # 2-NN from src to ref
        matches_s2r = bf.knnMatch(desc_src, desc_ref, k=2)

        # 2-NN from ref to src for cross checking
        matches_r2s = None
        if self.cross_check:
            matches_r2s = bf.knnMatch(desc_ref, desc_src, k=1)
            # Map ref_idx -> best src_idx
            r2s_map = {m[0].queryIdx: m[0].trainIdx for m in matches_r2s if len(m) > 0}

        correspondences: List[Correspondence] = []
        match_id = 0

        for m in matches_s2r:
            if len(m) < 2:
                continue
            m1, m2 = m[0], m[1]
            s_idx = m1.queryIdx
            r_idx = m1.trainIdx
            d1, d2 = m1.distance, m2.distance

            # Ratio test
            if d2 <= 1e-7:
                continue
            ratio = d1 / d2
            if ratio > self.ratio_thresh:
                continue

            # Mutual cross check
            if self.cross_check and matches_r2s is not None:
                if r2s_map.get(r_idx) != s_idx:
                    continue

            pt_s = (float(kps_src[s_idx].pt[0]), float(kps_src[s_idx].pt[1]))
            pt_r = (float(kps_ref[r_idx].pt[0]), float(kps_ref[r_idx].pt[1]))

            # Reliability weights
            r_s = 1.0
            r_r = 1.0
            if rel_src is not None:
                ix, iy = int(round(pt_s[0])), int(round(pt_s[1]))
                if 0 <= iy < rel_src.shape[0] and 0 <= ix < rel_src.shape[1]:
                    r_s = float(rel_src[iy, ix])
            if rel_ref is not None:
                ix, iy = int(round(pt_r[0])), int(round(pt_r[1]))
                if 0 <= iy < rel_ref.shape[0] and 0 <= ix < rel_ref.shape[1]:
                    r_r = float(rel_ref[iy, ix])

            # Confidence score: combines ratio distinctiveness, detector response, and terrain reliability
            distinctiveness = 1.0 - ratio
            conf = distinctiveness * np.sqrt(max(r_s * r_r, 0.01))
            conf = float(np.clip(conf, 0.0, 1.0))

            corr = Correspondence(
                id=match_id,
                src_pt=pt_s,
                ref_pt=pt_r,
                confidence=conf,
                dist=float(d1),
                ratio=float(ratio)
            )
            correspondences.append(corr)
            match_id += 1

        return correspondences
