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
    parser.add_argument("--method", type=str, default="PHASE_STRUCTURAL", choices=["SIFT", "AKAZE", "PHASE_STRUCTURAL", "RIFT2", "LOFTR", "LIGHTGLUE", "LEARNED"], help="Feature matching method")
    parser.add_argument("--model", type=str, default="AUTO", choices=["AUTO", "SIMILARITY", "AFFINE", "HOMOGRAPHY"], help="Transformation model")
    parser.add_argument("--no_subpixel", action="store_true", help="Disable sub-pixel refinement")
    parser.add_argument("--no_anms", action="store_true", help="Disable spatial ANMS optimization")
    parser.add_argument("--no_gis", action="store_true", help="Disable GIS pre-registration footprint analysis")

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
        use_gis_pre_registration=not args.no_gis,
        use_subpixel=not args.no_subpixel,
        use_spatial_anms=not args.no_anms
    )

    print(f"[*] Initializing GIS-Assisted Registration Pipeline (Method: {args.method}, Model: {args.model})...")
    pipeline = LunarRegistrationPipeline(config=cfg)

    print("[*] Running End-to-End Registration...")
    result = pipeline.run(src_lunar, ref_lunar)

    if result.spatial_pre_reg:
        gis = result.spatial_pre_reg
        print("\n================ GIS FOOTPRINT SUMMARY ================")
        print(f"Spatial Overlap:    {gis.overlap_percentage:.1f}% ({'Detected' if gis.overlap_detected else 'None'})")
        print(f"Scale Ratio:        {gis.scale_ratio:.2f}x ({gis.source_gsd}m vs {gis.reference_gsd}m)")
        if gis.sun_geometry.get("azimuth_delta_deg") is not None:
            print(f"Sun Azimuth Delta:  {gis.sun_geometry['azimuth_delta_deg']:.1f}° ({gis.sun_geometry['illumination_status']})")
        print("=======================================================")

    print("\n================ REGISTRATION SUMMARY ================")
    if result.matcher_info:
        info = result.matcher_info
        print(f"Matcher Engine:     {info.get('active_backend')} (Mode: {info.get('loftr_mode', 'N/A')})")
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
    result.export_geotiff(str(warped_path))
    print(f"[+] Saved registered GeoTIFF to {warped_path}")

    csv_path = out_dir / "control_points.csv"
    df = result.export_control_points_dataframe()
    df.to_csv(csv_path, index=False)
    print(f"[+] Exported {len(df)} control points (CSV) to {csv_path}")

    geojson_path = out_dir / "control_points.geojson"
    geojson_data = result.export_control_points_geojson()
    geojson_path.write_text(json.dumps(geojson_data, indent=2), encoding="utf-8")
    print(f"[+] Exported {len(geojson_data['features'])} inlier control points (GeoJSON) to {geojson_path}")

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(result.metrics.to_json(), encoding="utf-8")
    print(f"[+] Saved metrics to {metrics_path}")

if __name__ == "__main__":
    main()
