"""
FastAPI REST API Backend for Robust Multi-Modal Lunar Image Registration.
Provides endpoints for remote inference, batch registration, sensor configuration,
and SIH benchmark evaluations with full OpenAPI documentation (/docs).
"""
import sys
from pathlib import Path

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import io
import tempfile
import yaml
from typing import Optional, Dict, Any, List
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.io.dataset import LunarImage, load_lunar_image
from src.io.metadata import SensorMetadata
from src.pipeline import LunarRegistrationPipeline, RegistrationPipelineConfig

app = FastAPI(
    title="LunarRegX REST API",
    description="Backend service for Chandrayaan-2 (OHRC, TMC-2, IIRS) and Lunar Reference (LROC NAC) image registration (SIH PS 26166).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for external frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", summary="Service Health Check")
def health_check():
    """Returns system status, version, and hardware availability."""
    import torch
    cuda_avail = torch.cuda.is_available()
    return {
        "status": "healthy",
        "service": "LunarRegX Registration Engine",
        "version": "1.0.0",
        "hardware": "GPU (CUDA)" if cuda_avail else "CPU (Optimized Vectorized)",
        "cuda_available": cuda_avail
    }

@app.get("/sensors", summary="List Supported Lunar Sensors")
def list_sensors():
    """Lists all supported sensor configurations and their optical/geodetic parameters."""
    cfg_dir = ROOT_DIR / "configs"
    sensors = {}
    for p in cfg_dir.glob("*.yaml"):
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            sensors[p.stem] = data
    return {"supported_sensors": sensors}

@app.get("/benchmarks", summary="List Prepared SIH Challenge Scenarios")
def list_benchmarks():
    """Lists the 5 photometrically rendered lunar challenge benchmark pairs."""
    return {
        "benchmarks": [
            {"id": 1, "name": "Low Illumination Delta", "desc": "Baseline control with identical terrain and mild lighting delta"},
            {"id": 2, "name": "180° Shadow Reversal", "desc": "Extreme illumination challenge with opposite crater shadow directions"},
            {"id": 3, "name": "Multi-Scale (OHRC vs LROC)", "desc": "Scale delta simulating OHRC 0.25m vs LROC NAC 0.50m"},
            {"id": 4, "name": "Viewpoint & Affine Shear", "desc": "Oblique viewing angle and affine shear deformation"},
            {"id": 5, "name": "Polar Crater Deep Shadow", "desc": "Low solar incidence with permanent shadow voids"}
        ]
    }

@app.post("/register", summary="Register Source Image against Reference Image")
async def register_images(
    source_file: UploadFile = File(..., description="Source / Moving lunar image (TIFF, PNG, JPEG)"),
    reference_file: UploadFile = File(..., description="Reference / Fixed lunar image (TIFF, PNG, JPEG)"),
    sensor: Optional[str] = Form("ohrc", description="Sensor profile key: ohrc, tmc2, iirs, lroc_nac, lroc_wac"),
    method: str = Form("PHASE_STRUCTURAL", description="Feature engine: PHASE_STRUCTURAL, SIFT, AKAZE, LEARNED"),
    model: str = Form("AUTO", description="Geometric model: AUTO, SIMILARITY, AFFINE, HOMOGRAPHY"),
    use_subpixel: bool = Form(True, description="Enable 2D parabolic sub-pixel peak refinement"),
    use_anms: bool = Form(True, description="Enable grid-based Adaptive Non-Maximal Suppression")
):
    """
    Executes end-to-end multi-modal registration between uploaded source and reference images.
    Returns quantitative metrics, transformation model, spatial coverage, and inlier statistics.
    """
    cfg_path = ROOT_DIR / "configs" / f"{sensor}.yaml"
    gsd_val = 1.0
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cdata = yaml.safe_load(f)
            gsd_val = float(cdata.get("gsd", 1.0))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        s_path = tmp_path / source_file.filename
        r_path = tmp_path / reference_file.filename

        s_bytes = await source_file.read()
        r_bytes = await reference_file.read()
        s_path.write_bytes(s_bytes)
        r_path.write_bytes(r_bytes)

        try:
            src_lunar = load_lunar_image(s_path, gsd_override=gsd_val)
            ref_lunar = load_lunar_image(r_path, gsd_override=gsd_val)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode lunar images: {str(e)}")

        pipeline_cfg = RegistrationPipelineConfig(
            feature_method=method.upper(),
            preferred_model=model.upper(),
            use_subpixel=use_subpixel,
            use_spatial_anms=use_anms
        )
        pipeline = LunarRegistrationPipeline(config=pipeline_cfg)

        try:
            res = pipeline.run(src_lunar, ref_lunar)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

        m = res.metrics
        control_points = res.export_control_points_dataframe().to_dict(orient="records")

        return {
            "status": m.status,
            "model_selected": m.model_name,
            "condition_number": round(m.condition_number, 2),
            "diagnostics": m.diagnostics,
            "metrics": {
                "inlier_count": m.inlier_count,
                "total_candidates": m.total_candidates,
                "inlier_ratio": round(m.inlier_ratio, 4),
                "rmse_total_pixels": round(m.rmse_total_px, 4),
                "rmse_x_pixels": round(m.rmse_x_px, 4),
                "rmse_y_pixels": round(m.rmse_y_px, 4),
                "physical_ground_error_meters": round(m.physical_error_m, 4) if m.physical_error_m else None,
                "ground_sampling_distance_m_px": gsd_val,
                "median_residual_pixels": round(m.median_residual_px, 4),
                "p95_residual_pixels": round(m.p95_residual_px, 4),
                "grid_coverage_ratio": round(m.grid_coverage_ratio, 4),
                "convex_hull_ratio": round(m.convex_hull_ratio, 4),
                "runtime_seconds": round(m.runtime_sec, 3)
            },
            "num_control_points": len(control_points),
            "control_points_sample": control_points[:10]  # First 10 points for preview
        }

@app.post("/register/benchmark/{scenario_id}", summary="Run Registration on Prepared Benchmark Scenario")
def run_benchmark_scenario(
    scenario_id: int,
    method: str = Query("PHASE_STRUCTURAL", description="Feature engine"),
    model: str = Query("AUTO", description="Model: AUTO, SIMILARITY, AFFINE, HOMOGRAPHY")
):
    """Executes registration on one of the 5 SIH benchmark scenarios (1 to 5)."""
    samples_dir = ROOT_DIR / "data" / "samples"
    scenario_map = {
        1: (samples_dir / "pair1_baseline_src.png", samples_dir / "pair1_baseline_ref.png"),
        2: (samples_dir / "pair2_illumination_src.png", samples_dir / "pair2_illumination_ref.png"),
        3: (samples_dir / "pair3_scale_src.png", samples_dir / "pair3_scale_ref.png"),
        4: (samples_dir / "pair4_viewpoint_src.png", samples_dir / "pair4_viewpoint_ref.png"),
        5: (samples_dir / "pair5_polar_shadow_src.png", samples_dir / "pair5_polar_shadow_ref.png"),
    }

    if scenario_id not in scenario_map:
        raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} not found. Valid IDs: 1 to 5")

    s_p, r_p = scenario_map[scenario_id]
    src = load_lunar_image(s_p)
    ref = load_lunar_image(r_p)

    pipeline = LunarRegistrationPipeline(config=RegistrationPipelineConfig(
        feature_method=method,
        preferred_model=model,
        use_subpixel=True,
        use_spatial_anms=True
    ))

    res = pipeline.run(src, ref)
    m = res.metrics
    return {
        "scenario_id": scenario_id,
        "status": m.status,
        "model": m.model_name,
        "inliers": m.inlier_count,
        "inlier_ratio": round(m.inlier_ratio, 4),
        "rmse_pixels": round(m.rmse_total_px, 4),
        "coverage": round(m.grid_coverage_ratio, 4),
        "runtime_sec": round(m.runtime_sec, 3),
        "diagnostics": m.diagnostics
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
