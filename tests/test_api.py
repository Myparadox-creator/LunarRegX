"""
Unit tests for the FastAPI REST API Backend.
Verifies health, sensor listing, benchmark listing, and registration endpoints.
"""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "hardware" in data

def test_sensors_endpoint():
    response = client.get("/sensors")
    assert response.status_code == 200
    data = response.json()
    assert "supported_sensors" in data
    assert "ohrc" in data["supported_sensors"]
    assert "tmc2" in data["supported_sensors"]

def test_benchmarks_endpoint():
    response = client.get("/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert "benchmarks" in data
    assert len(data["benchmarks"]) == 5

def test_benchmark_scenario_execution():
    # Test scenario 1
    response = client.post("/register/benchmark/1?method=PHASE_STRUCTURAL&model=AUTO")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["SUCCESS", "WARNING"]
    assert data["inliers"] >= 10
    assert data["rmse_pixels"] < 1.0

def test_register_requires_corresponding_xmls(tmp_path):
    import numpy as np
    import cv2

    rng = np.random.RandomState(11)
    arr = (rng.rand(192, 192) * 50 + 50).astype(np.uint8)
    for _ in range(15):
        cx, cy = rng.randint(25, 165, size=2)
        r = rng.randint(8, 20)
        cv2.circle(arr, (int(cx), int(cy)), int(r), 25, -1)
        cv2.circle(arr, (int(cx), int(cy)), int(r), 220, 2)

    img_bytes = cv2.imencode(".png", arr)[1].tobytes()

    # 1. Missing XMLs must fail validation (422 Unprocessable Entity)
    resp_missing = client.post(
        "/register",
        files={
            "source_file": ("src.png", img_bytes, "image/png"),
            "reference_file": ("ref.png", img_bytes, "image/png"),
        },
        data={"method": "PHASE_STRUCTURAL"}
    )
    assert resp_missing.status_code == 422

    # 2. Invalid/non-lunar XML must be rejected (400 Bad Request)
    resp_bad_xml = client.post(
        "/register",
        files={
            "source_file": ("src.png", img_bytes, "image/png"),
            "source_xml": ("src.xml", b"<invalid_root><hello>world</hello></invalid_root>", "application/xml"),
            "reference_file": ("ref.png", img_bytes, "image/png"),
            "reference_xml": ("ref.xml", b"<invalid_root><hello>world</hello></invalid_root>", "application/xml"),
        },
        data={"method": "PHASE_STRUCTURAL"}
    )
    assert resp_bad_xml.status_code == 400

    # 3. Mismatched sensor XML (IIRS image with OHRC XML) must be rejected (400 Bad Request)
    sac_ohrc_xml = (
        b'<?xml version="1.0" encoding="utf-8"?>'
        b'<product><job_id>OHRXXD18CHO2359602NNNN24342131250969_V1_0</job_id>'
        b'<Resolution_in_meter>0.25</Resolution_in_meter><Sun_azimuth_in_degree>135.0</Sun_azimuth_in_degree></product>'
    )
    resp_mismatch = client.post(
        "/register",
        files={
            "source_file": ("IIRXXD32CHO1519402NNNN23022110758606_V1_1_source.png", img_bytes, "image/png"),
            "source_xml": ("OHRXXD18.xml", sac_ohrc_xml, "application/xml"),
            "reference_file": ("ref.png", img_bytes, "image/png"),
            "reference_xml": ("ref.xml", sac_ohrc_xml, "application/xml"),
        },
        data={"method": "PHASE_STRUCTURAL"}
    )
    assert resp_mismatch.status_code == 400

    # 4. Valid corresponding XMLs for both images succeed (200 OK)
    resp_ok = client.post(
        "/register",
        files={
            "source_file": ("src.png", img_bytes, "image/png"),
            "source_xml": ("src.xml", sac_ohrc_xml, "application/xml"),
            "reference_file": ("ref.png", img_bytes, "image/png"),
            "reference_xml": ("ref.xml", sac_ohrc_xml, "application/xml"),
        },
        data={"method": "PHASE_STRUCTURAL"}
    )
    assert resp_ok.status_code == 200
    body = resp_ok.json()
    assert body["metrics"]["inlier_count"] >= 4
