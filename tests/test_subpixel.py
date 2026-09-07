"""
Unit tests verifying continuous sub-pixel parabolic peak interpolation.
Tests recover known sub-pixel fractional shifts down to < 0.08 pixel.
"""
import numpy as np
import cv2
import pytest
from src.subpixel.refinement import SubPixelRefiner

def test_subpixel_refiner_known_shift():
    refiner = SubPixelRefiner(patch_size=17, search_radius=4)

    y, x = np.mgrid[-25:26, -25:26]
    base_patch = np.exp(-(x**2 + y**2) / 50.0).astype(np.float32)

    true_dx = 0.35
    true_dy = -0.25
    M_shift = np.array([[1, 0, true_dx], [0, 1, true_dy]], dtype=np.float32)
    shifted_patch = cv2.warpAffine(base_patch, M_shift, (51, 51), flags=cv2.INTER_CUBIC)

    src_pt = (25.0, 25.0)
    ref_pt = (25.0, 25.0)

    refined_ref_pt, offset, ncc = refiner.refine_point(
        base_patch, shifted_patch, src_pt, ref_pt
    )

    est_dx, est_dy = offset
    assert abs(est_dx - true_dx) < 0.08, f"Estimated dx {est_dx} != true dx {true_dx}"
    assert abs(est_dy - true_dy) < 0.08, f"Estimated dy {est_dy} != true dy {true_dy}"
    assert ncc > 0.95, f"NCC too low: {ncc}"
