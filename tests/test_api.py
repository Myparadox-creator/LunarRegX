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
