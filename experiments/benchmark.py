"""
Automated Scientific Experimental Benchmarking Framework.
Compares:
1. Baseline 1: SIFT + RANSAC
2. Baseline 2: AKAZE
3. Baseline 3: Learned CNN Adapter
4. Baseline 4: RIFT2 / Structural MIM
5. Baseline 5: LoFTR (Detector-Free Transformers)
6. Proposed: GIS-Assisted Physics-Aware Hybrid Pipeline
across the 5 Lunar Challenge Scenarios.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time
import argparse
import pandas as pd
import numpy as np

from src.io.dataset import load_lunar_image
from src.pipeline import LunarRegistrationPipeline, RegistrationPipelineConfig

def run_benchmark(selected_method: str = "ALL"):
    data_dir = Path("data/samples")
    pairs = [
        ("Pair 1: Low Illumination Delta", data_dir / "pair1_baseline_src.png", data_dir / "pair1_baseline_ref.png"),
        ("Pair 2: 180° Shadow Reversal", data_dir / "pair2_illumination_src.png", data_dir / "pair2_illumination_ref.png"),
        ("Pair 3: Multi-Scale (OHRC vs LROC)", data_dir / "pair3_scale_src.png", data_dir / "pair3_scale_ref.png"),
        ("Pair 4: Viewpoint / Affine Shear", data_dir / "pair4_viewpoint_src.png", data_dir / "pair4_viewpoint_ref.png"),
        ("Pair 5: Polar Crater Deep Shadow", data_dir / "pair5_polar_shadow_src.png", data_dir / "pair5_polar_shadow_ref.png"),
    ]

    all_methods = [
        ("Baseline 1 (SIFT + RANSAC)", "SIFT", False, False, None),
        ("Baseline 2 (AKAZE)", "AKAZE", False, False, None),
        ("Baseline 3 (Learned CNN)", "LEARNED", False, False, None),
        ("Baseline 4 (RIFT2)", "RIFT2", False, False, None),
        ("Baseline 5a (LoFTR Pretrained)", "LOFTR", False, False, "non_existent_pretrained_fallback"),
        ("Baseline 5b (LoFTR Lunar Fine-Tuned)", "LOFTR", False, False, "models/loftr/lunar_finetuned/best.ckpt"),
        ("Proposed (GIS-Assisted Hybrid)", "PHASE_STRUCTURAL", True, True, None),
    ]

    if selected_method != "ALL":
        methods = [m for m in all_methods if selected_method.upper() in m[1].upper() or selected_method.upper() in m[0].upper()]
        if not methods:
            methods = [("Custom " + selected_method, selected_method.upper(), True, True, None)]
    else:
        methods = all_methods

    results = []

    print(f"\n{'='*115}")
    print(f"{'EXPERIMENT':<32} | {'METHOD':<32} | {'INLIERS':<8} | {'RATIO':<6} | {'RMSE':<8} | {'COV':<6} | {'STATUS':<7}")
    print(f"{'='*115}")

    for pair_name, src_p, ref_p in pairs:
        if not src_p.exists() or not ref_p.exists():
            print(f"Skipping {pair_name}: files not found")
            continue

        src_lunar = load_lunar_image(src_p)
        ref_lunar = load_lunar_image(ref_p)

        for m_name, feat_engine, use_anms, use_subpix, ckpt_p in methods:
            cfg = RegistrationPipelineConfig(
                feature_method=feat_engine,
                preferred_model="AUTO",
                use_spatial_anms=use_anms,
                use_subpixel=use_subpix,
                checkpoint_path=ckpt_p
            )
            pipeline = LunarRegistrationPipeline(config=cfg)

            try:
                out = pipeline.run(src_lunar, ref_lunar)
                m = out.metrics
                provenance = out.matcher_info.get("loftr_mode") if out.matcher_info else feat_engine
                row = {
                    "scenario": pair_name,
                    "method": m_name,
                    "engine": feat_engine,
                    "weights_mode": provenance,
                    "total_candidates": m.total_candidates,
                    "inlier_count": m.inlier_count,
                    "inlier_ratio": round(m.inlier_ratio, 3),
                    "rmse_pixels": round(m.rmse_total_px, 3),
                    "physical_error_m": round(m.physical_error_m, 2) if m.physical_error_m else None,
                    "grid_coverage": round(m.grid_coverage_ratio, 3),
                    "runtime_sec": round(m.runtime_sec, 2),
                    "model_selected": m.model_name,
                    "status": m.status
                }
                print(f"{pair_name:<32} | {m_name:<32} | {m.inlier_count:<8} | {m.inlier_ratio*100:>5.1f}% | {m.rmse_total_px:>6.3f}px | {m.grid_coverage_ratio*100:>5.1f}% | {m.status:<7}")
            except Exception as e:
                row = {
                    "scenario": pair_name,
                    "method": m_name,
                    "engine": feat_engine,
                    "total_candidates": 0,
                    "inlier_count": 0,
                    "inlier_ratio": 0.0,
                    "rmse_pixels": 0.0,
                    "physical_error_m": None,
                    "grid_coverage": 0.0,
                    "runtime_sec": 0.0,
                    "model_selected": "NONE",
                    "status": "FAILURE"
                }
                print(f"{pair_name:<32} | {m_name:<28} | {'FAILED':<8} | {'0.0%':<6} | {'N/A':<8} | {'0.0%':<6} | {'FAILURE':<7} ({e})")

            results.append(row)

    df = pd.DataFrame(results)
    out_csv = Path("results/benchmark_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n{'='*100}")
    print(f"Benchmark results saved to: {out_csv.resolve()}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lunar Image Correspondence Scientific Benchmark")
    parser.add_argument("--method", type=str, default="ALL", help="Method filter: ALL, SIFT, AKAZE, LEARNED, RIFT2, LOFTR, PHASE_STRUCTURAL")
    args = parser.parse_args()
    run_benchmark(selected_method=args.method)