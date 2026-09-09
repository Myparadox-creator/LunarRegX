from pathlib import Path
import pytest
import numpy as np
from src.gis.lunar_crs import LunarCRS, LUNAR_MEAN_RADIUS_METERS
from src.gis.footprint import (
    create_footprint_from_dimensions,
    compute_footprint_intersection,
    extract_common_roi
)
from src.gis.pre_registration import GISPreRegistrationAnalyzer
from src.io.dataset import LunarImage, SensorMetadata

def test_lunar_crs_equirectangular_roundtrip():
    crs = LunarCRS(projection_type="EQUIRECTANGULAR", center_lon_deg=0.0, center_lat_deg=0.0)
    lon, lat = 15.5, -20.0
    x, y = crs.forward(lon, lat)
    lon_rec, lat_rec = crs.inverse(x, y)
    assert abs(lon_rec - lon) < 1e-4
    assert abs(lat_rec - lat) < 1e-4

def test_lunar_crs_polar_south():
    crs = LunarCRS(projection_type="POLAR_STEREOGRAPHIC_SOUTH", center_lon_deg=0.0, center_lat_deg=-90.0)
    lon, lat = 45.0, -85.0
    x, y = crs.forward(lon, lat)
    lon_rec, lat_rec = crs.inverse(x, y)
    assert abs(lon_rec - lon) < 1e-4
    assert abs(lat_rec - lat) < 1e-4

def test_footprint_intersection_calculation():
    fp1 = create_footprint_from_dimensions(width_px=500, height_px=500, center_lon=0.0, center_lat=0.0, gsd_m=1.0)
    # 50% horizontal shift: 250 meters
    fp2 = create_footprint_from_dimensions(width_px=500, height_px=500, center_lon=0.00824, center_lat=0.0, gsd_m=1.0)
    
    res = compute_footprint_intersection(fp1, fp2)
    assert res["overlap_detected"] is True
    assert res["intersection_area_km2"] > 0
    assert 0.0 < res["overlap_percentage_src"] < 100.0

def test_common_roi_extraction():
    img = np.arange(10000, dtype=np.uint8).reshape((100, 100))
    fp = create_footprint_from_dimensions(width_px=100, height_px=100, center_lon=0.0, center_lat=0.0, gsd_m=10.0)
    # Crop central half
    min_x, min_y, max_x, max_y = fp.bounds_proj
    mid_x = (min_x + max_x) / 2.0
    mid_y = (min_y + max_y) / 2.0
    target_bounds = (min_x + 250.0, min_y + 250.0, max_x - 250.0, max_y - 250.0)
    
    cropped, window = extract_common_roi(img, fp, target_bounds)
    assert cropped.shape[0] < 100
    assert cropped.shape[1] < 100
    assert window[2] == cropped.shape[1]

def test_gis_pre_registration_analyzer():
    meta1 = SensorMetadata(sensor_name="OHRC", gsd=0.25, solar_azimuth_deg=45.0, solar_elevation_deg=30.0)
    meta2 = SensorMetadata(sensor_name="LROC_NAC", gsd=0.50, solar_azimuth_deg=225.0, solar_elevation_deg=25.0)
    raw = np.zeros((128, 128), dtype=np.uint8)
    src = LunarImage(raw_array=raw, normalized=raw.astype(np.float32), display_8bit=raw, metadata=meta1)
    ref = LunarImage(raw_array=raw, normalized=raw.astype(np.float32), display_8bit=raw, metadata=meta2)
    
    analyzer = GISPreRegistrationAnalyzer()
    res = analyzer.analyze(src, ref)
    assert res.overlap_detected is True
    assert res.sun_geometry["illumination_status"] == "EXTREME_SHADOW_FLIP"
    assert res.scale_ratio == 0.5