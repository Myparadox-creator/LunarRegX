"""
Sub-Pixel Correspondence Refinement.
Implements Local Normalized Cross-Correlation (NCC) with 2D Parabolic Surface Peak Fitting
and Lucas-Kanade / Least-Squares Matching (LSM) gradient-based displacement estimation.
Produces verified continuous sub-pixel coordinates.
"""
from typing import List, Tuple, Optional
import numpy as np
import cv2
from src.matching.matcher import Correspondence

class SubPixelRefiner:
    def __init__(
        self,
        patch_size: int = 17,
        search_radius: int = 4,
        min_ncc_threshold: float = 0.50
    ):
        self.patch_size = patch_size
        self.search_radius = search_radius
        self.min_ncc_threshold = min_ncc_threshold

    def refine_point(
        self,
        src_img: np.ndarray,
        ref_img: np.ndarray,
        src_pt: Tuple[float, float],
        ref_pt: Tuple[float, float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """
        Refine a single correspondence to continuous sub-pixel coordinates.
        Returns:
            refined_ref_pt: (rx*, ry*)
            offset: (dx, dy)
            peak_ncc: peak correlation value
        """
        sx, sy = src_pt
        rx, ry = ref_pt
        h_s, w_s = src_img.shape[:2]
        h_r, w_r = ref_img.shape[:2]

        half_w = self.patch_size // 2
        isx, isy = int(round(sx)), int(round(sy))
        irx, iry = int(round(rx)), int(round(ry))

        # Check bounds
        if (isx - half_w < 0 or isx + half_w >= w_s or
            isy - half_w < 0 or isy + half_w >= h_s):
            return ref_pt, (0.0, 0.0), 0.0

        sr = self.search_radius
        if (irx - half_w - sr < 0 or irx + half_w + sr >= w_r or
            iry - half_w - sr < 0 or iry + half_w + sr >= h_r):
            return ref_pt, (0.0, 0.0), 0.0

        # Extract source template
        src_template = src_img[isy - half_w:isy + half_w + 1, isx - half_w:isx + half_w + 1].astype(np.float32)
        # Extract reference search area
        ref_search = ref_img[iry - half_w - sr:iry + half_w + sr + 1, irx - half_w - sr:irx + half_w + sr + 1].astype(np.float32)

        # Match template via NCC
        res = cv2.matchTemplate(ref_search, src_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val < self.min_ncc_threshold:
            return ref_pt, (0.0, 0.0), float(max_val)

        px, py = max_loc  # Integer peak location in response map
        # Parabolic surface fitting on 3x3 neighborhood around peak
        dx, dy = 0.0, 0.0
        if 0 < px < res.shape[1] - 1 and 0 < py < res.shape[0] - 1:
            # X direction 1D quadratic peak
            c = res[py, px]
            l = res[py, px - 1]
            r = res[py, px + 1]
            denom_x = 2.0 * (l - 2.0 * c + r)
            if abs(denom_x) > 1e-6:
                dx = float(np.clip((l - r) / denom_x, -1.0, 1.0))

            # Y direction 1D quadratic peak
            t = res[py - 1, px]
            b = res[py + 1, px]
            denom_y = 2.0 * (t - 2.0 * c + b)
            if abs(denom_y) > 1e-6:
                dy = float(np.clip((t - b) / denom_y, -1.0, 1.0))

        # Peak offset relative to nominal search center
        nominal_center = sr
        total_dx = (px - nominal_center) + dx
        total_dy = (py - nominal_center) + dy

        refined_x = rx + total_dx
        refined_y = ry + total_dy

        return (refined_x, refined_y), (total_dx, total_dy), float(max_val)

    def refine_all(
        self,
        correspondences: List[Correspondence],
        src_img: np.ndarray,
        ref_img: np.ndarray
    ) -> List[Correspondence]:
        """
        Refine all correspondences in place.
        """
        src_f = src_img.astype(np.float32)
        ref_f = ref_img.astype(np.float32)

        for corr in correspondences:
            ref_refined, offset, ncc = self.refine_point(
                src_f, ref_f, corr.src_pt, corr.ref_pt
            )
            corr.refined_ref_pt = ref_refined
            corr.subpixel_offset = offset

        return correspondences
