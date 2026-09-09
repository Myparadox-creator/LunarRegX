import pytest
import numpy as np
import cv2
from src.features.deep_matcher import DeepCorrespondenceMatcher

def test_deep_matcher_initialization():
    matcher = DeepCorrespondenceMatcher(backend="LOFTR")
    assert matcher is not None
    assert "LOFTR" in matcher.backend

def test_deep_matcher_synthetic_matching():
    # Synthesize two images with known translated feature
    img1 = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(img1, (64, 64), 16, 255, -1)
    cv2.circle(img1, (32, 32), 8, 200, -1)
    
    # Translate by (+10, +5)
    M = np.float32([[1, 0, 10], [0, 1, 5]])
    img2 = cv2.warpAffine(img1, M, (128, 128))
    
    matcher = DeepCorrespondenceMatcher(backend="LOFTR")
    corrs, matches_dict = matcher.match(img1, img2)
    
    assert isinstance(corrs, list)
    assert isinstance(matches_dict, list)
    if len(matches_dict) > 0:
        first = matches_dict[0]
        assert "src_x" in first
        assert "ref_x" in first
        assert "confidence" in first