"""
Sensor-Specific Encoders for Chandrayaan-2 and Lunar Reference Missions.
Recognizes distinct optical, stereo, and hyperspectral imaging physics:
- OHRC: Ultra-high resolution (0.25 m/px), panchromatic, micro-crater topography.
- TMC-2: Stereo panchromatic (5.0 m/px), along-track triplet, elevation relief.
- IIRS: Hyperspectral (80 m/px, 256 bands, 0.8-5.0 um), mineral absorption bands.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from src.io.dataset import LunarImage

@dataclass
class CommonTerrainRepresentation:
    structural_map: np.ndarray     # Float32 [0.0, 1.0] normalized edge/salience map
    enhanced_8u: np.ndarray        # Uint8 [0, 255] contrast-normalized display image
    feature_mask: np.ndarray       # Boolean mask of reliable structural terrain
    sensor_type: str               # OHRC, TMC2, IIRS, LROC_NAC, GENERIC
    effective_gsd_m: float         # Ground sampling distance in meters
    spectral_channels: int         # Number of spectral channels processed
    processing_metadata: Dict[str, Any]

class BaseSensorEncoder(ABC):
    @abstractmethod
    def encode(self, image: LunarImage) -> CommonTerrainRepresentation:
        pass

class OHRCEncoder(BaseSensorEncoder):
    """
    Panchromatic high-resolution encoder (Chandrayaan-2 OHRC ~0.25m / LROC NAC ~0.50m).
    Applies multi-scale Laplacian of Gaussian to preserve boulder/rim micro-topography.
    """
    def __init__(self, clahe_clip: float = 2.0, detail_boost: float = 1.3):
        self.clahe_clip = clahe_clip
        self.detail_boost = detail_boost

    def encode(self, image: LunarImage) -> CommonTerrainRepresentation:
        raw_norm = image.normalized.copy()
        h, w = raw_norm.shape[:2]

        # 1. Micro-crater edge sharpening
        blurred = cv2.GaussianBlur(raw_norm, (3, 3), 0.8)
        high_freq = np.clip(raw_norm - blurred, -0.5, 0.5)
        sharpened = np.clip(raw_norm + self.detail_boost * high_freq, 0.0, 1.0)

        # 2. Local contrast equalization
        sharp_8u = (sharpened * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
        enhanced_8u = clahe.apply(sharp_8u)
        enhanced_norm = enhanced_8u.astype(np.float32) / 255.0

        # 3. Structural Sobel magnitude
        gx = cv2.Sobel(enhanced_norm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(enhanced_norm, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.sqrt(gx**2 + gy**2)
        p95 = np.percentile(edge_mag, 95.0)
        structural_map = np.clip(edge_mag / (p95 + 1e-6), 0.0, 1.0)

        # 4. Valid terrain mask
        valid_mask = (enhanced_norm > 0.03) & (enhanced_norm < 0.98)
        if image.mask is not None:
            valid_mask &= image.mask

        return CommonTerrainRepresentation(
            structural_map=structural_map,
            enhanced_8u=enhanced_8u,
            feature_mask=valid_mask,
            sensor_type="OHRC",
            effective_gsd_m=float(image.metadata.gsd),
            spectral_channels=1,
            processing_metadata={"method": "MicroTopographySharpening", "clahe_clip": self.clahe_clip}
        )

class TMC2Encoder(BaseSensorEncoder):
    """
    Stereo optical encoder (Chandrayaan-2 TMC-2 ~5.0m / LROC WAC ~100m).
    Applies bandpass filtering to remove illumination ramps across wide swaths.
    """
    def __init__(self, low_sigma: float = 1.0, high_sigma: float = 20.0):
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma

    def encode(self, image: LunarImage) -> CommonTerrainRepresentation:
        raw_norm = image.normalized.copy()
        
        # Difference-of-Gaussians bandpass to remove regional illumination gradient
        low = gaussian_filter(raw_norm, sigma=self.low_sigma)
        high = gaussian_filter(raw_norm, sigma=self.high_sigma)
        dog = low - high
        
        p2, p98 = np.percentile(dog, (2.0, 98.0))
        if p98 > p2:
            dog_norm = np.clip((dog - p2) / (p98 - p2), 0.0, 1.0)
        else:
            dog_norm = np.full_like(dog, 0.5)

        enhanced_8u = (dog_norm * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_8u = clahe.apply(enhanced_8u)
        structural_map = enhanced_8u.astype(np.float32) / 255.0

        valid_mask = np.ones(raw_norm.shape, dtype=bool)
        if image.mask is not None:
            valid_mask &= image.mask

        return CommonTerrainRepresentation(
            structural_map=structural_map,
            enhanced_8u=enhanced_8u,
            feature_mask=valid_mask,
            sensor_type="TMC2",
            effective_gsd_m=float(image.metadata.gsd),
            spectral_channels=1,
            processing_metadata={"method": "StereoBandpassDoG", "low_sigma": self.low_sigma, "high_sigma": self.high_sigma}
        )

class IIRSEncoder(BaseSensorEncoder):
    """
    Hyperspectral lunar encoder (Chandrayaan-2 IIRS ~80m, 256 spectral channels).
    Extracts structural continuum via PCA / spectral band integration,
    preventing mineral absorption bands from confounding geometric registration.
    """
    def __init__(self, n_components: int = 1):
        self.n_components = n_components

    def encode(self, image: LunarImage) -> CommonTerrainRepresentation:
        raw = image.raw_array
        if raw.ndim == 3 and raw.shape[2] > 1:
            # Multi-band hyperspectral cube: perform PCA dimensionality reduction via OpenCV
            h, w, c = raw.shape
            pixels = raw.reshape(-1, c).astype(np.float32)
            mean_val, eigenvectors = cv2.PCACompute(pixels, mean=np.empty((0, 0), dtype=np.float32), maxComponents=1)
            projected = cv2.PCAProject(pixels, mean_val, eigenvectors)
            reduced = projected.reshape((h, w))
            p1, p99 = np.percentile(reduced, (1.0, 99.0))
            norm = np.clip((reduced - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
            n_chan = c
        else:
            norm = image.normalized.copy()
            n_chan = 1

        # Morphological gradient
        norm_8u = (norm * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_8u = clahe.apply(norm_8u)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_grad = cv2.morphologyEx(enhanced_8u, cv2.MORPH_GRADIENT, kernel)
        structural_map = morph_grad.astype(np.float32) / 255.0

        valid_mask = (norm > 0.05) & (norm < 0.95)
        if image.mask is not None:
            valid_mask &= image.mask

        return CommonTerrainRepresentation(
            structural_map=structural_map,
            enhanced_8u=enhanced_8u,
            feature_mask=valid_mask,
            sensor_type="IIRS",
            effective_gsd_m=float(image.metadata.gsd),
            spectral_channels=n_chan,
            processing_metadata={"method": "HyperspectralPCAMorphology", "spectral_bands_ingested": n_chan}
        )

class GenericLunarEncoder(BaseSensorEncoder):
    def encode(self, image: LunarImage) -> CommonTerrainRepresentation:
        norm_8u = (image.normalized * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enh_8u = clahe.apply(norm_8u)
        return CommonTerrainRepresentation(
            structural_map=image.normalized,
            enhanced_8u=enh_8u,
            feature_mask=image.mask if image.mask is not None else np.ones(image.shape[:2], dtype=bool),
            sensor_type="GENERIC",
            effective_gsd_m=float(image.metadata.gsd),
            spectral_channels=1,
            processing_metadata={"method": "GenericStandardCLAHE"}
        )

def encode_sensor_image(image: LunarImage) -> CommonTerrainRepresentation:
    """
    Factory routing function: Dispatches sensor-specific encoder based on metadata.
    """
    s_name = image.metadata.sensor_name.upper() if image.metadata else ""
    if "OHRC" in s_name or "NAC" in s_name:
        return OHRCEncoder().encode(image)
    elif "TMC" in s_name or "WAC" in s_name:
        return TMC2Encoder().encode(image)
    elif "IIRS" in s_name or "SPECTRAL" in s_name:
        return IIRSEncoder().encode(image)
    else:
        # Infer by GSD
        gsd = image.metadata.gsd if image.metadata else 1.0
        if gsd <= 1.0:
            return OHRCEncoder().encode(image)
        elif gsd <= 20.0:
            return TMC2Encoder().encode(image)
        else:
            return IIRSEncoder().encode(image)