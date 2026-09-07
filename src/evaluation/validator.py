"""
Independent Held-Out Checkpoint Validation.
Splits correspondences into estimation (training) and validation (check) points
to evaluate true generalization and prevent reporting in-sample accuracy as final ground truth.
"""
from typing import List, Tuple, Optional
import numpy as np
from src.matching.matcher import Correspondence
from src.geometry.models import GeometricModel

def evaluate_checkpoints(
    checkpoints: List[Correspondence],
    model: GeometricModel,
    use_subpixel: bool = True
) -> Tuple[float, float, float]:
    """
    Evaluate independent check points against estimated geometric model.
    Returns:
        rmse_x: px
        rmse_y: px
        rmse_total: px
    """
    if not checkpoints:
        return 0.0, 0.0, 0.0

    pts_src = np.array([c.src_pt for c in checkpoints], dtype=np.float32)
    if use_subpixel:
        pts_ref = np.array([c.refined_ref_pt if c.refined_ref_pt is not None else c.ref_pt for c in checkpoints], dtype=np.float32)
    else:
        pts_ref = np.array([c.ref_pt for c in checkpoints], dtype=np.float32)

    predicted_ref = model.transform(pts_src)
    diffs = predicted_ref - pts_ref  # (N, 2)

    rmse_x = float(np.sqrt(np.mean(diffs[:, 0] ** 2)))
    rmse_y = float(np.sqrt(np.mean(diffs[:, 1] ** 2)))
    rmse_total = float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))

    return rmse_x, rmse_y, rmse_total
