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


def test_geotiff_export_bytes():
    import rasterio
    from rasterio.io import MemoryFile
    from src.warping.resampler import export_geotiff_bytes
    from src.gis.lunar_crs import LunarCRS

    # Synthesize test warped image array and Lunar CRS
    test_img = np.arange(120 * 160, dtype=np.uint8).reshape((120, 160))
    crs_obj = LunarCRS(projection_type="EQUIRECTANGULAR", center_lon_deg=0.0, center_lat_deg=0.0)
    crs_wkt = crs_obj.to_wkt()
    bounds_proj = (-500.0, -400.0, 500.0, 400.0)

    geotiff_bytes = export_geotiff_bytes(
        image_array=test_img,
        bounds_proj=bounds_proj,
        crs_wkt=crs_wkt
    )
    assert isinstance(geotiff_bytes, bytes)
    assert len(geotiff_bytes) > 0

    # Open with rasterio and verify dimensions, band count, CRS, and raster content
    with MemoryFile(geotiff_bytes) as mem:
        with mem.open() as ds:
            assert ds.shape == (120, 160)
            assert ds.count == 1
            assert ds.crs is not None
            # Verify Moon / IAU datum or projection is preserved
            wkt = ds.crs.to_wkt().lower()
            assert "moon" in wkt or "1737400" in wkt or ds.crs.is_projected
            read_arr = ds.read(1)
            np.testing.assert_array_equal(read_arr, test_img)


def test_pipeline_requires_xml_when_enforced(tmp_path):
    import cv2
    from src.io.dataset import load_lunar_image

    # Synthetic scene with distinct craters so PHASE_STRUCTURAL has strong features if XML passes
    rng = np.random.RandomState(7)
    arr = (rng.rand(192, 192) * 40 + 60).astype(np.uint8)
    for _ in range(15):
        cx, cy = rng.randint(25, 165, size=2)
        r = rng.randint(8, 22)
        cv2.circle(arr, (int(cx), int(cy)), int(r), 30, -1)
        cv2.circle(arr, (int(cx), int(cy)), int(r), 210, 2)

    src_p = tmp_path / "custom_src.png"
    ref_p = tmp_path / "custom_ref.png"
    cv2.imwrite(str(src_p), arr)
    cv2.imwrite(str(ref_p), arr)

    src_no_xml = load_lunar_image(src_p)
    ref_no_xml = load_lunar_image(ref_p)

    cfg = RegistrationPipelineConfig(feature_method="PHASE_STRUCTURAL", require_xml_metadata=True)
    pipeline = LunarRegistrationPipeline(config=cfg)

    with pytest.raises(ValueError, match="requires corresponding XML metadata"):
        pipeline.run(src_no_xml, ref_no_xml)

    # Now load with valid corresponding XML files and confirm pipeline runs
    xml_template = """<?xml version="1.0" encoding="utf-8"?>
<product>
    <job_id>OHRXXD18CHO2359602NNNN24342131250969_V1_0_03</job_id>
    <Resolution_in_meter>0.25</Resolution_in_meter>
    <Sun_azimuth_in_degree>135.0</Sun_azimuth_in_degree>
    <Sun_elevation_in_degree>30.0</Sun_elevation_in_degree>
</product>"""
    src_xml = tmp_path / "custom_src.xml"
    ref_xml = tmp_path / "custom_ref.xml"
    src_xml.write_text(xml_template, encoding="utf-8")
    ref_xml.write_text(xml_template, encoding="utf-8")

    src_with_xml = load_lunar_image(src_p, xml_path=src_xml, require_xml=True)
    ref_with_xml = load_lunar_image(ref_p, xml_path=ref_xml, require_xml=True)
    out = pipeline.run(src_with_xml, ref_with_xml)
    assert out.metrics.inlier_count >= 4


