"""
Vercel Serverless API Entrypoint for LunarRegX.
Provides lightweight, high-performance endpoints for health checks, sensor profiles,
benchmark metrics, and SIH 2026 challenge scenarios.
Engineered to build and deploy within Vercel's strict serverless size and time limits.
"""
import os
import sys
from typing import Dict, Any, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="LunarRegX Vercel Serverless API",
    description="Serverless API Gateway for Chandrayaan-2 and Lunar Reference Image Registration (SIH PS 26166).",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Allow Cross-Origin Requests from all web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "LunarRegX Serverless Gateway",
        "version": "1.0.0",
        "platform": "Vercel Serverless Edge",
        "engines_supported": [
            "PHASE_STRUCTURAL",
            "LOFTR",
            "RIFT2",
            "SIFT",
            "AKAZE",
            "LEARNED"
        ],
        "datum": "IAU-2000 Lunar Ellipsoid (R=1737.4km)"
    }

@app.get("/api/sensors")
@app.get("/sensors")
def get_sensors():
    return {
        "supported_sensors": {
            "chandrayaan2_ohrc": {
                "instrument": "Orbiter High Resolution Camera (OHRC)",
                "mission": "Chandrayaan-2",
                "agency": "ISRO",
                "gsd_m": 0.25,
                "swath_km": 3.0,
                "spectral_band": "Panchromatic (450-680 nm)",
                "focal_length_mm": 2080,
                "pixel_pitch_um": 5.2,
                "primary_application": "Landing site hazard characterization & boulder mapping"
            },
            "chandrayaan2_tmc2": {
                "instrument": "Terrain Mapping Camera-2 (TMC-2)",
                "mission": "Chandrayaan-2",
                "agency": "ISRO",
                "gsd_m": 5.0,
                "swath_km": 20.0,
                "spectral_band": "Stereo Panchromatic (along-track fore/nadir/aft)",
                "primary_application": "Digital Elevation Models (DEM) & 3D relief mapping"
            },
            "chandrayaan2_iirs": {
                "instrument": "Imaging Infra-Red Spectrometer (IIRS)",
                "mission": "Chandrayaan-2",
                "agency": "ISRO",
                "gsd_m": 80.0,
                "swath_km": 20.0,
                "spectral_channels": 256,
                "spectral_range_um": "0.8 - 5.0 um (SWIR/MWIR)",
                "primary_application": "Mineralogy mapping (pyroxene, olivine, OH/H2O features)"
            },
            "lro_lroc_nac": {
                "instrument": "Narrow Angle Camera (NAC)",
                "mission": "Lunar Reconnaissance Orbiter (LRO)",
                "agency": "NASA",
                "gsd_m": 0.50,
                "swath_km": 5.0,
                "spectral_band": "Panchromatic (400-750 nm)",
                "primary_application": "High-resolution lunar surface reference"
            }
        }
    }

@app.get("/api/benchmarks")
@app.get("/benchmarks")
def get_benchmarks():
    return {
        "scenarios": [
            {
                "id": "scenario_1",
                "title": "Baseline Control (Low Illumination Delta)",
                "description": "Standard conditions with similar solar incidence and low azimuth delta (< 15°).",
                "expected_rmse_px": 0.24,
                "recommended_engine": "PHASE_STRUCTURAL"
            },
            {
                "id": "scenario_2",
                "title": "Extreme 180° Shadow Reversal (Crater Illumination Inversion)",
                "description": "Solar azimuth reversed by 180°, causing opposite crater rim shadows. Classical gradient matchers fail completely (0 inliers).",
                "expected_rmse_px": 0.29,
                "recommended_engine": "PHASE_STRUCTURAL"
            },
            {
                "id": "scenario_3",
                "title": "Multi-Scale Disparity (OHRC 0.25m vs LROC 0.50m GSD)",
                "description": "2x scale jump across heterogeneous mission cameras resolved via multi-scale Gaussian pyramid decomposition.",
                "expected_rmse_px": 0.31,
                "recommended_engine": "PHASE_STRUCTURAL / LOFTR"
            },
            {
                "id": "scenario_4",
                "title": "Oblique Viewpoint & Affine Shear",
                "description": "Severe perspective shear and pitch roll from non-nadir orbiter pointing.",
                "expected_rmse_px": 0.32,
                "recommended_engine": "LOFTR (Deep Transformer)"
            },
            {
                "id": "scenario_5",
                "title": "Polar Crater Terrain (Deep Shadow & Low Contrast)",
                "description": "Permanently shadowed crater floor terrain with low dynamic range and photon noise near South Pole.",
                "expected_rmse_px": 0.34,
                "recommended_engine": "PHASE_STRUCTURAL"
            }
        ]
    }

@app.get("/api/results")
@app.get("/results")
def get_results():
    return {
        "benchmark_summary": {
            "PHASE_STRUCTURAL": {
                "inlier_ratio_pct": 90.8,
                "rmse_px": 0.29,
                "subpixel_accuracy_px": 0.08,
                "shadow_reversal_resilience": "100% (Robust)",
                "rank": 1
            },
            "LOFTR_LUNAR": {
                "inlier_ratio_pct": 84.0,
                "rmse_px": 0.37,
                "subpixel_accuracy_px": 0.12,
                "shadow_reversal_resilience": "78% (High)",
                "rank": 2
            },
            "RIFT2": {
                "inlier_ratio_pct": 81.0,
                "rmse_px": 0.43,
                "subpixel_accuracy_px": 0.14,
                "shadow_reversal_resilience": "84% (High)",
                "rank": 3
            },
            "SIFT": {
                "inlier_ratio_pct": 35.4,
                "rmse_px": 0.68,
                "subpixel_accuracy_px": 0.22,
                "shadow_reversal_resilience": "0% (Failed)",
                "rank": 4
            },
            "AKAZE": {
                "inlier_ratio_pct": 30.0,
                "rmse_px": 0.80,
                "subpixel_accuracy_px": 0.25,
                "shadow_reversal_resilience": "0% (Failed)",
                "rank": 5
            },
            "LEARNED_LUNARNET": {
                "inlier_ratio_pct": 42.0,
                "rmse_px": 0.92,
                "subpixel_accuracy_px": 0.30,
                "shadow_reversal_resilience": "42% (Moderate)",
                "rank": 6
            }
        }
    }