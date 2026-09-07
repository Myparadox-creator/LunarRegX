"""
Robust Multi-Modal Lunar Image Registration CLI.
Command-line entry point for registering Chandrayaan-2 and lunar reference images.
"""
import argparse
from pathlib import Path
import json
import pandas as pd

from src.io.dataset import load_lunar_image, save_lunar_image
from src.io.metadata import SensorMetadata
from src.pipeline import LunarRegistrationPipeline, RegistrationPipelineConfig

def main():
    parser = argparse.ArgumentParser(description="Robust Multi-Modal Lunar Image Registration (SIH Prototype)")
    parser.add_argument("--source", type=str, required=True, help="Path to source (moving) image")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference (fixed) image")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save registered outputs")
    parser.add_argument("--sensor_config", type=str, default=None, help="Optional sensor YAML profile")
    parser.add_argument("--method", type=str, default="PHASE_STRUCTURAL", choices=["SIFT", "AKAZE", "PHASE_STRUCTURAL", "LEARNED"], help="Feature matching method")
    parser.add_argument("--model", type=str, default="AUTO", choices=["AUTO", "SIMILARITY", "AFFINE", "HOMOGRAPHY"], help="Transformation model")
    parser.add_argument("--no_subpixel", action="store_true", help="Disable sub-pixel refinement")
    parser.add_argument("--no_anms", action="store_true", help="Disable spatial ANMS optimization")

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading Source: {args.source}")
    src_lunar = load_lunar_image(args.source)
    print(f"[*] Loading Reference: {args.reference}")
    ref_lunar = load_lunar_image(args.reference)

    if args.sensor_config:
        ref_lunar.metadata = SensorMetadata.from_yaml(Path(args.sensor_config))

    cfg = RegistrationPipelineConfig(
        feature_method=args.method,
        preferred_model=args.model,
        use_subpixel=not args.no_subpixel,
        use_spatial_anms=not args.no_anms
    )

    print(f"[*] Initializing Registration Pipeline (Method: {args.method}, Model: {args.model})...")
    pipeline = LunarRegistrationPipeline(config=cfg)

    print("[*] Running End-to-End Registration...")
    result = pipeline.run(src_lunar, ref_lunar)

    print("\n================ REGISTRATION SUMMARY ================")
    print(f"Status:             {result.metrics.status}")
    print(f"Model:              {result.metrics.model_name}")
    print(f"Inliers:            {result.metrics.inlier_count} / {result.metrics.total_candidates} ({result.metrics.inlier_ratio:.1%})")
    print(f"Total RMSE:         {result.metrics.rmse_total_px:.3f} pixels")
    if result.metrics.physical_error_m is not None:
        print(f"Physical Error:     {result.metrics.physical_error_m:.3f} meters (GSD: {ref_lunar.metadata.gsd} m/px)")
    print(f"Grid Coverage:      {result.metrics.grid_coverage_ratio:.1%}")
    print(f"Convex Hull Ratio:  {result.metrics.convex_hull_ratio:.1%}")
    print(f"Condition Number:   {result.metrics.condition_number:.1f}")
    print(f"Runtime:            {result.metrics.runtime_sec:.2f} seconds")
    print("Diagnostics:")
    for diag in result.metrics.diagnostics:
        print(f"  - {diag}")
    print("====================================================\n")

    # Save outputs
    warped_path = out_dir / "registered_source.tif"
    save_lunar_image(warped_path, result.warped_image)
    print(f"[+] Saved registered image to {warped_path}")

    csv_path = out_dir / "control_points.csv"
    df = result.export_control_points_dataframe()
    df.to_csv(csv_path, index=False)
    print(f"[+] Exported {len(df)} control points to {csv_path}")

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(result.metrics.to_json(), encoding="utf-8")
    print(f"[+] Saved metrics to {metrics_path}")

if __name__ == "__main__":
    main()
