"""
Phase Congruency and Structural Representation for Lunar Imagery.
Based on Kovesi (1999) and Radiation-variation Insensitive Feature Transform (RIFT, Li et al. 2019).
Extracts illumination-invariant and contrast-reversal-invariant feature maps.
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2

@dataclass
class PhaseCongruencyResult:
    max_moment: np.ndarray        # Maximum moment (edges and ridge structures, e.g. crater rims)
    min_moment: np.ndarray        # Minimum moment (corners, intersections, discrete rocks)
    mim: np.ndarray               # Maximum Index Map (dominant orientation index per pixel)
    phase_energy: np.ndarray      # Total phase congruency energy map
    total_amplitude: np.ndarray   # Total amplitude sum across all scales and orientations

class LogGaborPhaseCongruency:
    """
    2D Log-Gabor filter bank for computing Phase Congruency and structural representations.
    """
    def __init__(
        self,
        n_scales: int = 3,
        n_orientations: int = 6,
        min_wavelength: float = 3.0,
        mult: float = 2.1,
        sigma_on_f: float = 0.55,
        d_theta_on_sigma: float = 1.2,
        noise_k: float = 2.0
    ):
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.min_wavelength = min_wavelength
        self.mult = mult
        self.sigma_on_f = sigma_on_f
        self.d_theta_on_sigma = d_theta_on_sigma
        self.noise_k = noise_k

    def _construct_filter_bank(self, rows: int, cols: int):
        # Frequency mesh
        y, x = np.mgrid[-rows//2:int(np.ceil(rows/2)), -cols//2:int(np.ceil(cols/2))]
        y = y / float(rows)
        x = x / float(cols)
        radius = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, -x)
        # Avoid log(0)
        radius[rows//2, cols//2] = 1.0

        theta_sigma = np.pi / (self.n_orientations * self.d_theta_on_sigma)
        filters = []

        for s in range(self.n_scales):
            wavelength = self.min_wavelength * (self.mult ** s)
            fo = 1.0 / wavelength
            # Radial Log-Gabor
            log_gabor = np.exp(-((np.log(radius / fo)) ** 2) / (2.0 * (np.log(self.sigma_on_f) ** 2)))
            log_gabor[rows//2, cols//2] = 0.0  # Zero DC

            scale_filters = []
            for o in range(self.n_orientations):
                angle = o * np.pi / self.n_orientations
                # Angular difference
                d_theta = np.abs(theta - angle)
                d_theta = np.minimum(d_theta, np.pi - d_theta)
                spread = np.exp(-(d_theta ** 2) / (2.0 * (theta_sigma ** 2)))
                filt = np.fft.fftshift(log_gabor * spread)
                scale_filters.append(filt)
            filters.append(scale_filters)

        return filters

    def compute(self, image_norm: np.ndarray) -> PhaseCongruencyResult:
        """
        Compute phase congruency moments (M, m) and Maximum Index Map (MIM).
        """
        rows, cols = image_norm.shape
        img_f = image_norm.astype(np.float32)
        fft_img = np.fft.fft2(img_f)

        filters = self._construct_filter_bank(rows, cols)

        sum_e = np.zeros((self.n_orientations, rows, cols), dtype=np.float32)
        sum_o = np.zeros((self.n_orientations, rows, cols), dtype=np.float32)
        sum_an = np.zeros((self.n_orientations, rows, cols), dtype=np.float32)
        energy = np.zeros((self.n_orientations, rows, cols), dtype=np.float32)
        orient_amplitude = np.zeros((self.n_orientations, rows, cols), dtype=np.float32)

        for o in range(self.n_orientations):
            for s in range(self.n_scales):
                filt = filters[s][o]
                res = np.fft.ifft2(fft_img * filt)
                e = np.real(res)
                o_resp = np.imag(res)
                an = np.sqrt(e**2 + o_resp**2)

                sum_e[o] += e
                sum_o[o] += o_resp
                sum_an[o] += an
                orient_amplitude[o] += an

            # Orientation energy
            e_o = np.sqrt(sum_e[o]**2 + sum_o[o]**2)
            # Estimate noise threshold via smallest scale amplitude median
            tau = np.median(sum_an[o]) / np.sqrt(np.log(4.0))
            thresh = tau * self.noise_k
            energy[o] = np.maximum(e_o - thresh, 0.0)

        # Compute Moments
        # a = sum( (E * cos(theta))^2 )
        # b = 2 * sum( (E * cos(theta)) * (E * sin(theta)) )
        # c = sum( (E * sin(theta))^2 )
        a = np.zeros((rows, cols), dtype=np.float32)
        b = np.zeros((rows, cols), dtype=np.float32)
        c = np.zeros((rows, cols), dtype=np.float32)

        for o in range(self.n_orientations):
            angle = o * np.pi / self.n_orientations
            e_cos = energy[o] * np.cos(angle)
            e_sin = energy[o] * np.sin(angle)
            a += e_cos ** 2
            b += 2.0 * e_cos * e_sin
            c += e_sin ** 2

        # Maximum & Minimum moments
        sqrt_term = np.sqrt(b**2 + (a - c)**2)
        max_moment = 0.5 * (c + a + sqrt_term)
        min_moment = 0.5 * (c + a - sqrt_term)

        # Maximum Index Map (MIM): index of orientation with highest amplitude
        mim = np.argmax(orient_amplitude, axis=0).astype(np.uint8)

        # Normalize moments to [0, 1]
        mm_max = np.percentile(max_moment, 99.0)
        if mm_max > 0:
            max_moment = np.clip(max_moment / mm_max, 0.0, 1.0)
        else:
            max_moment = np.zeros_like(max_moment)

        min_max = np.percentile(min_moment, 99.0)
        if min_max > 0:
            min_moment = np.clip(min_moment / min_max, 0.0, 1.0)
        else:
            min_moment = np.zeros_like(min_moment)

        total_energy = np.sum(energy, axis=0)
        total_amp = np.sum(orient_amplitude, axis=0)

        return PhaseCongruencyResult(
            max_moment=max_moment,
            min_moment=min_moment,
            mim=mim,
            phase_energy=total_energy,
            total_amplitude=total_amp
        )
