"""
End-to-End Integration and Export Tests.
Verifies pipeline execution, CSV control-point export, and JSON metrics generation.
"""
from pathlib import Path
import numpy as np
import pytest

from src.io.dataset import LunarImage, SensorMetadata
from src.pipeline import LunarRegistrationPipeline, RegistrationPipelineConfig

def test_pipeline_export_dataframes():
    from src.io.dataset import load_lunar_image
    sample_ref = Path("data/samples/pair1_baseline_ref.png")
    sample_src = Path("data/samples/pair1_baseline_src.png")
    if sample_ref.exists() and sample_src.exists():
        src = load_lunar_image(sample_src)
        ref = load_lunar_image(sample_ref)
    else:
        y, x = np.mgrid[0:200, 0:200]
        c1 = np.exp(-((x - 80)**2 + (y - 80)**2) / 400.0)
        c2 = np.exp(-((x - 140)**2 + (y - 130)**2) / 500.0)
        img = ((c1 + c2) * 220.0 + np.random.randn(200, 200) * 5.0).astype(np.uint8)
        meta = SensorMetadata(sensor_name="TEST_SENSOR", gsd=0.5)
        src = LunarImage(raw_array=img, normalized=img/255.0, display_8bit=img, metadata=meta)
        ref = LunarImage(raw_array=img, normalized=img/255.0, display_8bit=img, metadata=meta)

    cfg = RegistrationPipelineConfig(feature_method="PHASE_STRUCTURAL", preferred_model="SIMILARITY")
    pipeline = LunarRegistrationPipeline(config=cfg)
    result = pipeline.run(src, ref)

    df = result.export_control_points_dataframe()
    assert not df.empty
    assert "source_x" in df.columns
    assert "refined_ref_x" in df.columns
    assert "is_inlier" in df.columns
    assert "confidence" in df.columns

    json_str = result.metrics.to_json()
    assert "rmse_total_px" in json_str
    assert "inlier_count" in json_str
