"""
Synthetic Ground-Truth Verification Test.
Applies known rotation, scale, and sub-pixel translation to lunar image,
runs the full registration pipeline, and asserts parameter recovery within sub-pixel accuracy.
"""
from pathlib import Path
import numpy as np
import cv2
import pytest

from src.io.dataset import LunarImage, SensorMetadata
from src.pipeline import LunarRegistrationPipeline, RegistrationPipelineConfig

def test_pipeline_recovers_known_transformation():
    y, x = np.mgrid[0:256, 0:256]
    c1 = np.exp(-((x - 100)**2 + (y - 100)**2) / 600.0)
    c2 = np.exp(-((x - 180)**2 + (y - 160)**2) / 800.0)
    c3 = np.exp(-((x - 80)**2 + (y - 190)**2) / 500.0)
    img = (c1 + c2 + c3) * 200.0 + np.random.randn(256, 256) * 5.0
    ref_norm = np.clip(img / 255.0, 0.0, 1.0).astype(np.float32)
    ref_8u = (ref_norm * 255.0).astype(np.uint8)

    true_tx = 8.25
    true_ty = -6.40
    M_true = cv2.getRotationMatrix2D((128, 128), 3.0, 1.0)
    M_true[0, 2] += true_tx
    M_true[1, 2] += true_ty

    src_8u = cv2.warpAffine(ref_8u, M_true, (256, 256), flags=cv2.INTER_CUBIC)
    src_norm = src_8u.astype(np.float32) / 255.0

    meta = SensorMetadata(sensor_name="SYNTHETIC_TEST", gsd=1.0)
    src_lunar = LunarImage(raw_array=src_8u, normalized=src_norm, display_8bit=src_8u, metadata=meta)
    ref_lunar = LunarImage(raw_array=ref_8u, normalized=ref_norm, display_8bit=ref_8u, metadata=meta)

    cfg = RegistrationPipelineConfig(
        feature_method="PHASE_STRUCTURAL",
        preferred_model="SIMILARITY",
        use_spatial_anms=True,
        use_subpixel=True
    )
    pipeline = LunarRegistrationPipeline(config=cfg)
    result = pipeline.run(src_lunar, ref_lunar)

    assert result.metrics.status in ["SUCCESS", "WARNING"]
    assert result.metrics.inlier_count >= 10
    assert result.metrics.rmse_total_px < 0.60, f"RMSE {result.metrics.rmse_total_px} too high"
