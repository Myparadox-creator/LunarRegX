"""
Systematic Ablation Study.
Isolates the contribution of each algorithmic stage:
A. Baseline Matcher Only (Raw SIFT, no preprocessing, no ANMS, no subpixel)
B. + Illumination Robustness (Log-Gabor Phase Congruency & Reliability Mask)
C. + Multi-Scale Strategy (Resolution normalization)
D. + Spatially Uniform Control-Point Optimization (ANMS Grid Binning)
E. + Sub-Pixel Correspondence Refinement (2D Parabolic Peak Fitting)

Evaluates on the rigorous Illumination Challenge Pair (180 deg solar azimuth flip).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np

from src.io.dataset import load_lunar_image
from src.pipeline import LunarRegistrationPipeline, RegistrationPipelineConfig

def run_ablation():
    src_p = Path("data/samples/pair2_illumination_src.png")
    ref_p = Path("data/samples/pair2_illumination_ref.png")

    if not src_p.exists() or not ref_p.exists():
        print("Ablation dataset pair 2 not found!")
        return

    src = load_lunar_image(src_p)
    ref = load_lunar_image(ref_p)

    stages = [
        ("A: Baseline Matcher Only", "SIFT", False, False, False),
        ("B: + Illumination Processing (Phase)", "PHASE_STRUCTURAL", True, False, False),
        ("C: + Robust Geometric Model Selection", "PHASE_STRUCTURAL", True, False, False),
        ("D: + Spatial Control-Point ANMS", "PHASE_STRUCTURAL", True, True, False),
        ("E: + Sub-Pixel Refinement (Full Hybrid)", "PHASE_STRUCTURAL", True, True, True),
    ]

    print(f"\n{'='*85}")
    print(f"{'ABLATION STAGE':<40} | {'INLIERS':<8} | {'RATIO':<8} | {'RMSE (px)':<10} | {'COVERAGE':<8}")
    print(f"{'='*85}")

    rows = []
    for stage_name, engine, use_clahe, use_anms, use_subpix in stages:
        cfg = RegistrationPipelineConfig(
            feature_method=engine,
            use_clahe=use_clahe,
            use_spatial_anms=use_anms,
            use_subpixel=use_subpix,
            preferred_model="AUTO"
        )
        pipeline = LunarRegistrationPipeline(config=cfg)
        try:
            res = pipeline.run(src, ref)
            m = res.metrics
            print(f"{stage_name:<40} | {m.inlier_count:<8} | {m.inlier_ratio:<8.1%} | {m.rmse_total_px:<10.3f} | {m.grid_coverage_ratio:<8.1%}")
            rows.append({
                "Stage": stage_name,
                "Inliers": m.inlier_count,
                "Inlier_Ratio": m.inlier_ratio,
                "RMSE_px": m.rmse_total_px,
                "Coverage": m.grid_coverage_ratio,
                "Status": m.status
            })
        except Exception as e:
            print(f"{stage_name:<40} | FAILED ({e})")
            rows.append({
                "Stage": stage_name,
                "Inliers": 0,
                "Inlier_Ratio": 0.0,
                "RMSE_px": 999.0,
                "Coverage": 0.0,
                "Status": "FAILED"
            })

    print(f"{'='*85}\n")
    df = pd.DataFrame(rows)
    out_csv = Path("results/ablation_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[+] Saved ablation table to {out_csv}")
    return df

if __name__ == "__main__":
    run_ablation()
