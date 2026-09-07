from typing import Optional
"""
Reliability and Quality Mapping for Lunar Surfaces.
Combines phase congruency strength, local texture variance, and shadow masking
to produce a continuous reliability map for match weighting.
"""
import numpy as np
import cv2
from scipy.ndimage import uniform_filter

def compute_terrain_reliability(
    image_norm: np.ndarray,
    pc_max_moment: np.ndarray,
    shadow_mask: Optional[np.ndarray] = None,
    window_size: int = 15
) -> np.ndarray:
    """
    Compute pixel-wise reliability score in [0.0, 1.0].
    High score: sharp invariant structure, high local variance, illuminated.
    Low score: deep shadow, smooth mare void, or saturated crater glint.
    """
    # 1. Local intensity variance
    mean = uniform_filter(image_norm, size=window_size)
    mean_sq = uniform_filter(image_norm**2, size=window_size)
    local_var = np.maximum(mean_sq - mean**2, 0.0)
    var_p95 = np.percentile(local_var, 95.0)
    texture_score = np.clip(local_var / (var_p95 + 1e-6), 0.0, 1.0)

    # 2. Phase congruency structure score
    structure_score = np.clip(pc_max_moment, 0.0, 1.0)

    # 3. Combine scores
    reliability = 0.6 * structure_score + 0.4 * texture_score

    # 4. Invalidate shadows if mask provided
    if shadow_mask is not None:
        reliability *= shadow_mask.astype(np.float32)

    # Blur slightly for smooth spatial weighting
    reliability = cv2.GaussianBlur(reliability, (5, 5), 1.0)
    return np.clip(reliability, 0.0, 1.0)
