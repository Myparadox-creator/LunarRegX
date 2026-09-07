"""
Preprocessing pipeline for lunar imagery.
Includes robust percentile stretching, CLAHE, bandpass/DoG filtering,
and shadow/reliability detection.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

@dataclass
class PreprocessingConfig:
    use_clahe: bool = True
    clahe_clip_limit: float = 2.5
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    use_bandpass: bool = False
    bandpass_sigma_low: float = 1.0
    bandpass_sigma_high: float = 25.0
    shadow_percentile_thresh: float = 4.0

def normalize_dynamic_range(
    image: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Robust percentile-based dynamic range normalization to [0.0, 1.0].
    Resilient to specular peaks and extreme deep shadows.
    """
    img_f = image.astype(np.float32)
    valid = img_f[mask] if mask is not None and mask.any() else img_f
    if valid.size == 0:
        return np.zeros_like(img_f)

    v_min, v_max = np.percentile(valid, (p_low, p_high))
    if v_max <= v_min:
        return np.zeros_like(img_f)

    norm = np.clip((img_f - v_min) / (v_max - v_min), 0.0, 1.0)
    return norm

def apply_clahe(
    image_8u: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances micro-craters and ejecta blankets while avoiding shadow clipping.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image_8u)

def apply_bandpass(
    image_norm: np.ndarray,
    sigma_low: float = 1.0,
    sigma_high: float = 25.0
) -> np.ndarray:
    """
    Difference of Gaussians (DoG) bandpass filter to eliminate large-scale
    illumination gradients (e.g. from low solar incidence) while preserving crater rims.
    """
    low = gaussian_filter(image_norm, sigma=sigma_low)
    high = gaussian_filter(image_norm, sigma=sigma_high)
    dog = low - high
    d_min, d_max = np.percentile(dog, (1.0, 99.0))
    if d_max > d_min:
        dog_norm = np.clip((dog - d_min) / (d_max - d_min), 0.0, 1.0)
    else:
        dog_norm = np.full_like(dog, 0.5)
    return dog_norm

def compute_shadow_mask(
    image_norm: np.ndarray,
    threshold_percentile: float = 4.0
) -> np.ndarray:
    """
    Detect deeply shadowed, textureless regions where correspondence is ambiguous.
    Returns boolean mask: True = reliable illuminated terrain, False = deep shadow/void.
    """
    thresh_val = np.percentile(image_norm, threshold_percentile)
    gy, gx = np.gradient(image_norm)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_thresh = np.percentile(grad_mag, 5.0)

    is_shadow = (image_norm <= thresh_val) & (grad_mag <= grad_thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    is_shadow = cv2.morphologyEx(is_shadow.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return ~is_shadow

def preprocess_lunar_image(
    image_norm: np.ndarray,
    config: Optional[PreprocessingConfig] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline.
    Returns:
        processed_8u: enhanced 8-bit image for feature detection
        processed_norm: enhanced float32 [0, 1] image
        shadow_mask: reliable terrain mask (True = valid, False = shadow)
    """
    cfg = config or PreprocessingConfig()
    cur = image_norm.copy()

    if cfg.use_bandpass:
        cur = apply_bandpass(cur, cfg.bandpass_sigma_low, cfg.bandpass_sigma_high)

    cur_8u = (cur * 255.0).astype(np.uint8)
    if cfg.use_clahe:
        cur_8u = apply_clahe(cur_8u, cfg.clahe_clip_limit, cfg.clahe_tile_grid_size)

    enhanced_norm = cur_8u.astype(np.float32) / 255.0
    shadow_mask = compute_shadow_mask(enhanced_norm, cfg.shadow_percentile_thresh)

    return cur_8u, enhanced_norm, shadow_mask
