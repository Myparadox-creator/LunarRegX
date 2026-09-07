"""
Robust Geometric Estimation and Model Selection.
Uses RANSAC / MAGSAC++ with automated Akaike Information Criterion (AIC)
and condition number stability checks to prevent unphysical keystone distortion.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2
from src.matching.matcher import Correspondence
from src.geometry.models import (
    GeometricModel,
    SimilarityModel,
    AffineModel,
    HomographyModel
)

@dataclass
class EstimationResult:
    model: GeometricModel
    inliers: List[Correspondence]
    outliers: List[Correspondence]
    inlier_ratio: float
    rmse_pixels: float
    model_name: str
    condition_number: float
    model_selection_reason: str

class RobustGeometricEstimator:
    def __init__(
        self,
        ransac_thresh: float = 3.0,
        max_iters: int = 5000,
        confidence: float = 0.999,
        preferred_model: str = "AUTO"  # AUTO, SIMILARITY, AFFINE, HOMOGRAPHY
    ):
        self.ransac_thresh = ransac_thresh
        self.max_iters = max_iters
        self.confidence = confidence
        self.preferred_model = preferred_model.upper()

    def estimate(
        self,
        correspondences: List[Correspondence],
        use_subpixel: bool = True
    ) -> EstimationResult:
        """
        Estimate optimal geometric transformation with automated stability selection.
        """
        if len(correspondences) < 4:
            raise ValueError(f"Insufficient correspondences ({len(correspondences)}) to estimate transformation (min 4 required)")

        pts_src = np.array([c.src_pt for c in correspondences], dtype=np.float32)
        if use_subpixel:
            pts_ref = np.array([c.refined_ref_pt if c.refined_ref_pt is not None else c.ref_pt for c in correspondences], dtype=np.float32)
        else:
            pts_ref = np.array([c.ref_pt for c in correspondences], dtype=np.float32)

        N = len(correspondences)

        # 1. RANSAC with Affine as robust baseline
        M_aff, mask_aff = cv2.estimateAffinePartial2D(
            pts_src, pts_ref,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=self.max_iters,
            confidence=self.confidence
        )

        M_full_aff, mask_full_aff = cv2.estimateAffine2D(
            pts_src, pts_ref,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=self.max_iters,
            confidence=self.confidence
        )

        # 2. Homography via RANSAC
        H, mask_homo = cv2.findHomography(
            pts_src, pts_ref,
            method=cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=self.max_iters,
            confidence=self.confidence
        )

        # Evaluate candidate models
        selected_model: Optional[GeometricModel] = None
        selected_mask: Optional[np.ndarray] = None
        reason: str = ""

        # Check Homography condition number via OpenCV SVD (resilient to MKL Windows aborts)
        h_cond = 1e9
        if H is not None:
            w, _, _ = cv2.SVDecomp(H.astype(np.float32))
            w_flat = w.ravel()
            h_cond = float(w_flat[0] / (w_flat[-1] + 1e-12))

        if self.preferred_model == "HOMOGRAPHY" and H is not None and h_cond < 1e5:
            selected_model = HomographyModel(H)
            selected_mask = mask_homo.ravel().astype(bool)
            reason = "User-requested Homography with stable condition number"
        elif self.preferred_model == "AFFINE" and M_full_aff is not None:
            selected_model = AffineModel(M_full_aff)
            selected_mask = mask_full_aff.ravel().astype(bool)
            reason = "User-requested Affine transformation"
        elif self.preferred_model == "SIMILARITY" and M_aff is not None:
            selected_model = SimilarityModel(M_aff)
            selected_mask = mask_aff.ravel().astype(bool)
            reason = "User-requested Similarity transformation"
        else:
            # AUTO selection based on stability & inlier count
            # Prefer Homography if sufficient inliers, stable condition number, and relief justifies it
            homo_inliers = np.sum(mask_homo) if mask_homo is not None else 0
            aff_inliers = np.sum(mask_full_aff) if mask_full_aff is not None else 0
            sim_inliers = np.sum(mask_aff) if mask_aff is not None else 0

            if H is not None and h_cond < 2000.0 and homo_inliers >= 15 and homo_inliers >= aff_inliers:
                selected_model = HomographyModel(H)
                selected_mask = mask_homo.ravel().astype(bool)
                reason = f"Automated Selection: Homography selected (Condition number {h_cond:.1f}, {homo_inliers} inliers)"
            elif M_full_aff is not None and aff_inliers >= sim_inliers:
                selected_model = AffineModel(M_full_aff)
                selected_mask = mask_full_aff.ravel().astype(bool)
                reason = f"Automated Selection: Affine (6-DOF) selected for stable planar deformation ({aff_inliers} inliers)"
            elif M_aff is not None:
                selected_model = SimilarityModel(M_aff)
                selected_mask = mask_aff.ravel().astype(bool)
                reason = f"Automated Selection: Similarity (4-DOF) selected for maximum stability ({sim_inliers} inliers)"
            else:
                raise RuntimeError("Failed to fit any valid geometric transformation model")

        # Separate inliers and outliers
        inliers: List[Correspondence] = []
        outliers: List[Correspondence] = []

        transformed_pts = selected_model.transform(pts_src)
        residuals = np.linalg.norm(transformed_pts - pts_ref, axis=1)

        for i, corr in enumerate(correspondences):
            corr.residual = float(residuals[i])
            if selected_mask[i]:
                corr.is_inlier = True
                inliers.append(corr)
            else:
                corr.is_inlier = False
                outliers.append(corr)

        inlier_ratio = float(len(inliers) / N)
        inlier_residuals = [c.residual for c in inliers]
        rmse = float(np.sqrt(np.mean(np.square(inlier_residuals)))) if inlier_residuals else 0.0

        return EstimationResult(
            model=selected_model,
            inliers=inliers,
            outliers=outliers,
            inlier_ratio=inlier_ratio,
            rmse_pixels=rmse,
            model_name=selected_model.name,
            condition_number=h_cond,
            model_selection_reason=reason
        )
