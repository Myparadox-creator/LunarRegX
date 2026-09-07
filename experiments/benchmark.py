"""
Automated Experimental Benchmarking Framework.
Compares Baseline 1 (SIFT + RANSAC), Baseline 2 (AKAZE),
Baseline 3 (Learned CNN Adapter), and Proposed (Phase-Structural Hybrid)
across all 5 Challenge Scenarios:
1. Baseline (Low illumination delta)
2. Extreme Illumination (180 deg solar azimuth flip with shadow reversal)
3. Multi-Scale (1.35x scale delta)
4. Viewpoint & Oblique Shear
5. Polar Permanent Shadow (High dynamic range, low SNR)

Outputs real measured metrics (Zero fabricated data).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time
import pandas as pd
import numpy as np

from src.io.dataset import load_lunar_image
from src.pipeline import LunarRegistrationPipeline, RegistrationPipelineConfig

def run_benchmark():
    data_dir = Path("data/samples")
    pairs = [
        ("Pair 1: Low Illumination Delta", data_dir / "pair1_baseline_src.png", data_dir / "pair1_baseline_ref.png"),
        ("Pair 2: 180° Shadow Reversal", data_dir / "pair2_illumination_src.png", data_dir / "pair2_illumination_ref.png"),
        ("Pair 3: Multi-Scale (OHRC vs LROC)", data_dir / "pair3_scale_src.png", data_dir / "pair3_scale_ref.png"),
        ("Pair 4: Viewpoint / Affine Shear", data_dir / "pair4_viewpoint_src.png", data_dir / "pair4_viewpoint_ref.png"),
        ("Pair 5: Polar Crater Deep Shadow", data_dir / "pair5_polar_shadow_src.png", data_dir / "pair5_polar_shadow_ref.png"),
    ]

    methods = [
        ("Baseline 1 (SIFT + RANSAC)", "SIFT", False, False),
        ("Baseline 2 (AKAZE)", "AKAZE", False, False),
        ("Baseline 3 (Learned CNN)", "LEARNED", False, False),
        ("Proposed (Phase-Structural Hybrid)", "PHASE_STRUCTURAL", True, True),
    ]

    results = []

    print(f"\n{'='*95}")
    print(f"{'EXPERIMENT':<32} | {'METHOD':<30} | {'INLIERS':<8} | {'RATIO':<6} | {'RMSE':<6} | {'COV':<6} | {'STATUS':<7}")
    print(f"{'='*95}")

    for pair_name, src_p, ref_p in pairs:
        if not src_p.exists() or not ref_p.exists():
            print(f"Skipping {pair_name}: files not found")
            continue

        src_lunar = load_lunar_image(src_p)
        ref_lunar = load_lunar_image(ref_p)

        for m_name, feat_engine, use_anms, use_subpix in methods:
            cfg = RegistrationPipelineConfig(
                feature_method=feat_engine,
                preferred_model="AUTO",
                use_spatial_anms=use_anms,
                use_subpixel=use_subpix
            )
            pipeline = LunarRegistrationPipeline(config=cfg)

            try:
                res = pipeline.run(src_lunar, ref_lunar)
                m = res.metrics
                row = {
                    "Scenario": pair_name,
                    "Method": m_name,
                    "Candidates": m.total_candidates,
                    "Inliers": m.inlier_count,
                    "Inlier_Ratio": f"{m.inlier_ratio:.1%}",
                    "RMSE_px": round(m.rmse_total_px, 3),
                    "Coverage": f"{m.grid_coverage_ratio:.1%}",
                    "Runtime_s": round(m.runtime_sec, 2),
                    "Status": m.status
                }
                print(f"{pair_name[:32]:<32} | {m_name[:30]:<30} | {m.inlier_count:<8} | {m.inlier_ratio:<6.1%} | {m.rmse_total_px:<6.3f} | {m.grid_coverage_ratio:<6.1%} | {m.status:<7}")
            except Exception as e:
                row = {
                    "Scenario": pair_name,
                    "Method": m_name,
                    "Candidates": 0,
                    "Inliers": 0,
                    "Inlier_Ratio": "0.0%",
                    "RMSE_px": None,
                    "Coverage": "0.0%",
                    "Runtime_s": 0.0,
                    "Status": "FAILED"
                }
                print(f"{pair_name[:32]:<32} | {m_name[:30]:<30} | {'FAILED':<8} | {'0.0%':<6} | {'N/A':<6} | {'0.0%':<6} | {'FAILED':<7}")

            results.append(row)

    print(f"{'='*95}\n")
    df = pd.DataFrame(results)
    out_csv = Path("results/benchmark_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[+] Saved complete benchmark table to {out_csv}")
    return df

if __name__ == "__main__":
    run_benchmark()
