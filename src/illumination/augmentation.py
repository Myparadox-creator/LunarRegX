"""
Physics-Aware Synthetic Lunar Data Augmentation.
Generates physically plausible lunar illumination variations for training and evaluation:
- Solar Azimuth contrast polarity inversion (180 deg crater shadow flip)
- Grazing solar incidence shadow-void masking
- Lommel-Seeliger non-linear photometric gamma curve shifts
- Micro-regolith roughness speckle and noise
"""
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2

@dataclass
class LunarAugmentationConfig:
    invert_contrast_prob: float = 0.5   # Simulates 180 deg shadow reversal
    shadow_mask_prob: float = 0.5       # Simulates grazing angle cast shadows
    photometric_gamma_range: Tuple[float, float] = (0.7, 1.4)
    noise_sigma_range: Tuple[float, float] = (0.005, 0.025)

class LunarPhysicsAugmenter:
    def __init__(self, config: Optional[LunarAugmentationConfig] = None, seed: Optional[int] = 42):
        self.cfg = config or LunarAugmentationConfig()
        self.rng = np.random.RandomState(seed)

    def augment(self, image_norm: np.ndarray) -> Tuple[np.ndarray, dict]:
        img = image_norm.copy().astype(np.float32)
        applied_ops = []

        # 1. Lommel-Seeliger photometric gamma transformation
        gamma = self.rng.uniform(*self.cfg.photometric_gamma_range)
        img = np.power(np.clip(img, 0.0, 1.0), gamma)
        applied_ops.append(f"photometric_gamma_{gamma:.2f}")

        # 2. Solar Azimuth Inversion (crater shadow flip)
        if self.rng.rand() < self.cfg.invert_contrast_prob:
            img = 1.0 - img
            applied_ops.append("solar_azimuth_polarity_invert_180deg")

        # 3. Grazing angle cast shadow mask (void injection)
        if self.rng.rand() < self.cfg.shadow_mask_prob:
            h, w = img.shape[:2]
            shadow_thresh = self.rng.uniform(0.08, 0.22)
            shadow_mask = img <= shadow_thresh
            # Dilate shadows along simulated illumination vector
            angle = self.rng.uniform(0, 2 * np.pi)
            dx = int(round(8.0 * np.cos(angle)))
            dy = int(round(8.0 * np.sin(angle)))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            cast_shadow = cv2.warpAffine(shadow_mask.astype(np.uint8), M, (w, h)).astype(bool)
            img[cast_shadow] = img[cast_shadow] * 0.15
            applied_ops.append("grazing_incidence_cast_shadow")

        # 4. Regolith shot & speckle noise
        noise_sigma = self.rng.uniform(*self.cfg.noise_sigma_range)
        noise = self.rng.randn(*img.shape) * noise_sigma
        img = np.clip(img + noise, 0.0, 1.0)
        applied_ops.append(f"regolith_noise_sigma_{noise_sigma:.3f}")

        return img.astype(np.float32), {"applied_augmentations": applied_ops}