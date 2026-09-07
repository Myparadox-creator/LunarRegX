"""
Unit tests for geometric transformations and model selection.
"""
import numpy as np
import pytest
from src.geometry.models import SimilarityModel, AffineModel, HomographyModel

def test_similarity_transformation():
    theta = np.radians(30)
    s = 1.5
    tx, ty = 10.0, -5.0
    M = np.array([
        [s * np.cos(theta), -s * np.sin(theta), tx],
        [s * np.sin(theta),  s * np.cos(theta), ty]
    ], dtype=np.float32)
    model = SimilarityModel(M)

    pts = np.array([[0, 0], [10, 20], [50, -30]], dtype=np.float32)
    forward = model.transform(pts)
    recovered = model.inverse_transform(forward)

    np.testing.assert_allclose(recovered, pts, atol=1e-4)

def test_affine_transformation():
    M = np.array([
        [1.2, -0.3, 15.0],
        [0.2,  1.1, -8.0]
    ], dtype=np.float32)
    model = AffineModel(M)

    pts = np.array([[10, 15], [100, 200], [50, 80]], dtype=np.float32)
    forward = model.transform(pts)
    recovered = model.inverse_transform(forward)

    np.testing.assert_allclose(recovered, pts, atol=1e-4)

def test_homography_transformation():
    H = np.array([
        [1.05, 0.02, 12.0],
        [-0.03, 0.98, -10.0],
        [0.0001, -0.0001, 1.0]
    ], dtype=np.float32)
    model = HomographyModel(H)

    pts = np.array([[20, 30], [150, 180], [300, 250]], dtype=np.float32)
    forward = model.transform(pts)
    recovered = model.inverse_transform(forward)

    np.testing.assert_allclose(recovered, pts, atol=1e-3)
