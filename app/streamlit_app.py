"""
Streamlit Web Dashboard for Robust GIS-Assisted Multi-Modal Lunar Image Registration.
Features:
- GIS Selenographic Footprint & Overlap Analysis (IAU 2000 Lunar Datum)
- Sensor-Specific Multimodal Encoders (OHRC, TMC-2, IIRS)
- Multi-Engine Algorithm Selector (Proposed Phase-Structural, LoFTR, RIFT2, SIFT, AKAZE, Learned CNN)
- SIH Judge Demonstration Mode (11-Stage Comprehensive Walkthrough)
- Multi-Artifact Export: GeoTIFF, CSV Control Points, GeoJSON, and Metrics JSON
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import io
import json
import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image

from src.io.dataset import LunarImage, load_lunar_image
from src.io.metadata import SensorMetadata
from src.pipeline import LunarRegistrationPipeline, RegistrationPipelineConfig
from src.gis.pre_registration import GISPreRegistrationAnalyzer

st.set_page_config(
    page_title="LunarRegX | GIS-Assisted Lunar Image Registration",
    page_icon="🌕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Lunar Styling
st.markdown("""
<style>
    .main { background-color: #0b0e14; color: #e6edf3; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; }
    .status-box-success { background-color: #1f6feb22; border: 1px solid #238636; border-radius: 8px; padding: 15px; }
    .status-box-warn { background-color: #bb800922; border: 1px solid #9e6a03; border-radius: 8px; padding: 15px; }
    .status-box-fail { background-color: #da363322; border: 1px solid #f85149; border-radius: 8px; padding: 15px; }
    h1, h2, h3 { color: #f0f6fc; }
</style>
""", unsafe_allow_html=True)

st.title("🌕 Robust GIS-Assisted Multi-Modal Lunar Image Registration")
st.caption("SIH 2026 Prototype (Problem ID 26166) | Chandrayaan-2 (OHRC, TMC-2, IIRS) ↔ Lunar Reference (LROC NAC/WAC)")

# Sidebar Controls
st.sidebar.header("⚙️ System Configuration")

mode = st.sidebar.radio("Operation Mode", ["🏆 SIH Judge Demonstration Mode", "🔬 Custom Registration & Upload"])

# Detect mode switch to prevent stale session state collisions
if "current_mode" not in st.session_state:
    st.session_state["current_mode"] = mode
elif st.session_state["current_mode"] != mode:
    st.session_state["current_mode"] = mode
    st.session_state.pop("reg_result", None)

# Sensor Profiles
sensor_choice = st.sidebar.selectbox(
    "Target Sensor Profile",
    [
        "Chandrayaan-2 OHRC (0.25 m/px)",
        "Chandrayaan-2 TMC-2 (5.0 m/px)",
        "Chandrayaan-2 IIRS (80 m/px)",
        "LRO LROC-NAC (0.50 m/px)",
        "LRO LROC-WAC (100 m/px)"
    ]
)

gsd_map = {
    "Chandrayaan-2 OHRC (0.25 m/px)": 0.25,
    "Chandrayaan-2 TMC-2 (5.0 m/px)": 5.0,
    "Chandrayaan-2 IIRS (80 m/px)": 80.0,
    "LRO LROC-NAC (0.50 m/px)": 0.50,
    "LRO LROC-WAC (100 m/px)": 100.0
}
selected_gsd = gsd_map[sensor_choice]

# Matcher Configuration
st.sidebar.subheader("Algorithmic Engine")
feature_method = st.sidebar.selectbox(
    "Correspondence Engine",
    [
        "PHASE_STRUCTURAL (Proposed Physics-Based)",
        "LOFTR (Deep Transformer)",
        "RIFT2 (Structural MIM)",
        "SIFT (Baseline 1)",
        "AKAZE (Baseline 2)",
        "LEARNED (Lightweight CNN)"
    ],
    index=0
)

if "PHASE_STRUCTURAL" in feature_method:
    engine_key = "PHASE_STRUCTURAL"
elif "LOFTR" in feature_method:
    engine_key = "LOFTR"
elif "RIFT2" in feature_method:
    engine_key = "RIFT2"
elif "SIFT" in feature_method:
    engine_key = "SIFT"
elif "AKAZE" in feature_method:
    engine_key = "AKAZE"
else:
    engine_key = "LEARNED"

if engine_key == "LOFTR":
    loftr_ckpt = ROOT_DIR / "models" / "loftr" / "lunar_finetuned" / "best.ckpt"
    if loftr_ckpt.exists():
        st.sidebar.caption("🟢 **LoFTR Status:** Lunar Fine-Tuned Checkpoint Active (`best.ckpt`)")
    else:
        st.sidebar.caption("🟡 **LoFTR Status:** Pretrained Outdoor Fallback")

model_choice = st.sidebar.selectbox("Transformation Model", ["AUTO (Stability-Guided)", "SIMILARITY (4-DOF)", "AFFINE (6-DOF)", "HOMOGRAPHY (8-DOF)"], index=0)
model_key = "AUTO" if "AUTO" in model_choice else ("SIMILARITY" if "SIMILARITY" in model_choice else ("AFFINE" if "AFFINE" in model_choice else "HOMOGRAPHY"))

use_gis = st.sidebar.checkbox("GIS Pre-Registration & Footprint Check", value=True)
use_multimodal = st.sidebar.checkbox("Sensor-Aware Multimodal Processing", value=True)
use_subpixel = st.sidebar.checkbox("Sub-Pixel Refinement (2D Parabolic Peak)", value=True)
use_anms = st.sidebar.checkbox("Spatially Uniform ANMS Grid", value=True)

# Data Ingestion
samples_dir = ROOT_DIR / "data" / "samples"
src_lunar = None
ref_lunar = None

if mode == "🏆 SIH Judge Demonstration Mode":
    st.sidebar.subheader("Select Prepared Lunar Scenario")
    scenario = st.sidebar.selectbox(
        "Demonstration Scenarios",
        [
            "Scenario 1: Baseline Control (Low illumination delta)",
            "Scenario 2: Extreme 180° Shadow Reversal (Crater Illumination Flip)",
            "Scenario 3: Multi-Scale (OHRC 0.25m vs LROC 0.5m GSD)",
            "Scenario 4: Viewpoint / Oblique Affine Shear",
            "Scenario 5: Polar Crater Terrain (Deep Shadow & Low Contrast)"
        ]
    )

    pair_files = {
        "Scenario 1: Baseline Control (Low illumination delta)": (samples_dir / "pair1_baseline_src.png", samples_dir / "pair1_baseline_ref.png"),
        "Scenario 2: Extreme 180° Shadow Reversal (Crater Illumination Flip)": (samples_dir / "pair2_illumination_src.png", samples_dir / "pair2_illumination_ref.png"),
        "Scenario 3: Multi-Scale (OHRC 0.25m vs LROC 0.5m GSD)": (samples_dir / "pair3_scale_src.png", samples_dir / "pair3_scale_ref.png"),
        "Scenario 4: Viewpoint / Oblique Affine Shear": (samples_dir / "pair4_viewpoint_src.png", samples_dir / "pair4_viewpoint_ref.png"),
        "Scenario 5: Polar Crater Terrain (Deep Shadow & Low Contrast)": (samples_dir / "pair5_polar_shadow_src.png", samples_dir / "pair5_polar_shadow_ref.png"),
    }

    src_path, ref_path = pair_files.get(scenario, (samples_dir / "pair1_baseline_src.png", samples_dir / "pair1_baseline_ref.png"))
    if src_path.exists() and ref_path.exists():
        try:
            # Assign appropriate solar geometry metadata based on scenario
            s_az = 315.0 if "180" in scenario else 135.0
            r_az = 135.0
            src_lunar = load_lunar_image(src_path, gsd_override=selected_gsd)
            ref_lunar = load_lunar_image(ref_path, gsd_override=selected_gsd)
            src_lunar.metadata.solar_azimuth_deg = s_az
            src_lunar.metadata.solar_elevation_deg = 35.0
            ref_lunar.metadata.solar_azimuth_deg = r_az
            ref_lunar.metadata.solar_elevation_deg = 40.0
        except Exception as e:
            st.error(f"❌ Failed to load benchmark image: {e}")
    else:
        st.warning("Benchmark samples not found. Run scripts/generate_lunar_benchmarks.py first.")

else:
    st.sidebar.subheader("Select or Upload Lunar Images")
    custom_source = st.sidebar.radio(
        "Image Selection Method",
        ["📂 Ingested Flight Images (Ready to Use)", "📤 Upload Files from Computer"],
        index=0
    )

    if custom_source == "📂 Ingested Flight Images (Ready to Use)":
        available_flight = {
            "Chandrayaan-2 OHRC (Landing Site Survey, Oct 2021)": samples_dir / "real_ch2_ohrc_landing_site.png",
            "Chandrayaan-2 OHRC (Sept 2019 Vikram Site Pass)": samples_dir / "real_ch2_ohrc_sept2019_vikram.png",
            "Chandrayaan-2 IIRS (Oct 2023 Hyperspectral Infrared)": samples_dir / "real_ch2_iirs_infrared_pass.png",
            "Benchmark: Baseline Control Source": samples_dir / "pair1_baseline_src.png",
            "Benchmark: Extreme 180° Shadow Flip Ref": samples_dir / "pair2_illumination_ref.png"
        }
        src_name = st.sidebar.selectbox("Select Source / Moving Image", list(available_flight.keys()), index=0)
        ref_name = st.sidebar.selectbox("Select Reference / Fixed Image", list(available_flight.keys()), index=1)
        
        s_p = available_flight[src_name]
        r_p = available_flight[ref_name]
        if s_p.exists() and r_p.exists():
            try:
                src_lunar = load_lunar_image(s_p, gsd_override=selected_gsd)
                ref_lunar = load_lunar_image(r_p, gsd_override=selected_gsd)
            except Exception as e:
                st.sidebar.error(f"❌ Error loading selected flight images: {e}")
                src_lunar = None
                ref_lunar = None
    else:
        st.sidebar.caption("Supported: 8/12/16-bit GeoTIFF, TIFF, PNG, JPEG")
        st.sidebar.caption("💡 Real flight PNGs are also saved in `D:\\Downloads2\\Lunar_Flight_Images\\`")
        up_src = st.sidebar.file_uploader("Upload Source / Moving Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
        up_ref = st.sidebar.file_uploader("Upload Reference / Fixed Image", type=["png", "jpg", "jpeg", "tif", "tiff"])

        if up_src and up_ref:
            try:
                tmp_dir = ROOT_DIR / "results" / "temp_uploads"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                s_p = tmp_dir / up_src.name
                r_p = tmp_dir / up_ref.name
                s_p.write_bytes(up_src.read())
                r_p.write_bytes(up_ref.read())
                src_lunar = load_lunar_image(s_p, gsd_override=selected_gsd)
                ref_lunar = load_lunar_image(r_p, gsd_override=selected_gsd)
            except Exception as e:
                st.sidebar.error(f"❌ Error loading uploaded images: {e}")
                src_lunar = None
                ref_lunar = None
        elif up_src or up_ref:
            st.sidebar.info("ℹ️ Uploaded 1 of 2 images. Please upload the matching pair to proceed.")

# Guidance screen when in Custom Upload mode and images are not yet provided
if mode == "🔬 Custom Registration & Upload" and (src_lunar is None or ref_lunar is None):
    if "reg_result" not in st.session_state:
        st.info("👈 **Upload both a Source / Moving image and a Reference / Fixed image in the sidebar to begin custom registration.**")
        
        c_g1, c_g2 = st.columns(2)
        with c_g1:
            st.markdown("""
            ### 🛰️ Supported Lunar Sensors
            * **Chandrayaan-2 OHRC:** Ultra-high resolution ($0.25\text{ m/px}$)
            * **Chandrayaan-2 TMC-2:** Stereo elevation mapping ($5.0\text{ m/px}$)
            * **Chandrayaan-2 IIRS:** Hyperspectral mineralogy ($80\text{ m/px}$)
            * **LRO LROC-NAC:** Narrow Angle Camera ($0.50\text{ m/px}$)
            * **LRO LROC-WAC:** Wide Angle Camera ($100\text{ m/px}$)
            """)
        with c_g2:
            st.markdown("""
            ### 📐 Recommended Input Specifications
            * **File Formats:** GeoTIFF, TIFF, PNG, JPEG, NPY
            * **Radiometric Depths:** 8-bit, 12-bit raw, 16-bit, 32-bit float
            * **GIS Coordinate System:** IAU-2000 Lunar Datum ($R = 1737.4\text{ km}$)
            * **Illumination Resilience:** Fourier Phase Congruency resolves $180^\circ$ shadow inversions
            """)

# Registration Execution Button & Input Previews
if src_lunar is not None and ref_lunar is not None:
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(src_lunar.display_8bit, caption=f"Source / Moving Image ({src_lunar.width}x{src_lunar.height}) | {src_lunar.metadata.bit_depth}-bit", use_container_width=True)
    with col_b:
        st.image(ref_lunar.display_8bit, caption=f"Reference / Fixed Image ({ref_lunar.width}x{ref_lunar.height}) | {ref_lunar.metadata.bit_depth}-bit", use_container_width=True)

    if st.button("🚀 EXECUTE REGISTRATION PIPELINE", type="primary", use_container_width=True):
        with st.spinner("Executing GIS-assisted multi-modal registration pipeline..."):
            try:
                cfg = RegistrationPipelineConfig(
                    feature_method=engine_key,
                    preferred_model=model_key,
                    use_gis_pre_registration=use_gis,
                    use_multimodal_encoder=use_multimodal,
                    use_spatial_anms=use_anms,
                    use_subpixel=use_subpixel
                )
                pipeline = LunarRegistrationPipeline(config=cfg)
                res = pipeline.run(src_lunar, ref_lunar)
                st.session_state["reg_result"] = res
                st.rerun()
            except Exception as e:
                st.error(f"❌ Registration could not be completed: {str(e)}")
                st.markdown("""
                > **💡 Troubleshooting Tips:**
                > 1. Ensure both lunar images cover overlapping geographic terrain.
                > 2. For severe illumination or shadow flips, ensure **PHASE_STRUCTURAL (Proposed)** is selected.
                > 3. If images have large scale differences, ensure the matching sensor profile GSD is selected.
                """)

# Display Results if available
if "reg_result" in st.session_state:
    res = st.session_state["reg_result"]
    m = res.metrics
    s_img = res.source_image
    r_img = res.reference_image
    gis_info = res.spatial_pre_reg

    st.markdown("---")
    c_head1, c_head2 = st.columns([5, 1])
    with c_head1:
        st.subheader("📊 Registration Performance Dashboard")
    with c_head2:
        if st.button("🔄 Clear Results", use_container_width=True):
            st.session_state.pop("reg_result", None)
            st.rerun()

    # Status Banner
    if m.status == "SUCCESS":
        st.markdown(f'<div class="status-box-success"><b>✅ REGISTRATION STATUS: SUCCESS</b><br>Model: {m.model_name} | {m.diagnostics[0]}</div>', unsafe_allow_html=True)
    elif m.status == "WARNING":
        st.markdown(f'<div class="status-box-warn"><b>⚠️ REGISTRATION STATUS: WARNING</b><br>{"<br>".join(m.diagnostics)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-box-fail"><b>❌ REGISTRATION STATUS: FAILURE</b><br>{"<br>".join(m.diagnostics)}</div>', unsafe_allow_html=True)

    st.write("")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Inlier Count", f"{m.inlier_count} / {m.total_candidates}")
    c2.metric("Inlier Ratio", f"{m.inlier_ratio:.1%}")
    c3.metric("Total 2D RMSE", f"{m.rmse_total_px:.3f} px")
    c4.metric("Physical Error", f"{m.physical_error_m:.2f} m" if m.physical_error_m else "N/A")
    c5.metric("Spatial Coverage", f"{m.grid_coverage_ratio:.1%}")
    c6.metric("Runtime", f"{m.runtime_sec:.2f} s")

    # SIH Judge 11-Step Tour Tabs
    st.markdown("---")
    st.subheader("🏆 SIH Judge Demonstration Walkthrough (11 Verification Stages)")

    tab_titles = [
        "1. GIS & Footprint",
        "2. Sensor Encoding",
        "3. Illumination Phase",
        "4. Correspondences",
        "5. Outlier Rejection",
        "6. Spatial ANMS",
        "7. Sub-Pixel Refinement",
        "8. Model Estimation",
        "9. Registered Warping",
        "10. Quality Inspection",
        "11. Export Artifacts"
    ]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.write("**Stage 1: GIS Footprint & Selenographic Geometry**")
        if gis_info:
            c_g1, c_g2, c_g3 = st.columns(3)
            c_g1.metric("Spatial Overlap", f"{gis_info.overlap_percentage:.1f}%", "Overlap Detected" if gis_info.overlap_detected else "No Overlap")
            c_g2.metric("GSD Scale Ratio", f"{gis_info.scale_ratio:.2f}x", f"{gis_info.source_gsd}m vs {gis_info.reference_gsd}m")
            c_g3.metric("Sun Azimuth Delta", f"{gis_info.sun_geometry['azimuth_delta_deg']:.1f}°" if gis_info.sun_geometry['azimuth_delta_deg'] is not None else "N/A", gis_info.sun_geometry['illumination_status'])

            st.markdown(f"""
            * **Lunar Datum:** IAU-2000 Mean Sphere (Radius $R = 1,737.4\text{{ km}}$)
            * **Source Selenographic Footprint:** Lon bounds `[{gis_info.source_footprint['bounds_geo'][0]:.4f}, {gis_info.source_footprint['bounds_geo'][2]:.4f}]`, Lat bounds `[{gis_info.source_footprint['bounds_geo'][1]:.4f}, {gis_info.source_footprint['bounds_geo'][3]:.4f}]`
            * **Reference Selenographic Footprint:** Lon bounds `[{gis_info.reference_footprint['bounds_geo'][0]:.4f}, {gis_info.reference_footprint['bounds_geo'][2]:.4f}]`, Lat bounds `[{gis_info.reference_footprint['bounds_geo'][1]:.4f}, {gis_info.reference_footprint['bounds_geo'][3]:.4f}]`
            * **Intersection Area:** {gis_info.common_roi.get('intersection_area_km2', 0.0)} km²
            * **Metadata Reliability:** Sun: `{gis_info.metadata_status['sun_geometry']}` | CRS: `{gis_info.metadata_status['spatial_crs']}`
            """)
        else:
            st.info("GIS Pre-registration disabled or metadata not provided.")

    with tabs[1]:
        st.write("**Stage 2: Sensor-Aware Multimodal Processing**")
        st.write("Dispatches specialized image representations acknowledging distinct sensor modalities (OHRC high-resolution panchromatic, TMC-2 stereo bandpass, or IIRS hyperspectral continuum PCA).")
        from src.multimodal.sensor_encoder import encode_sensor_image
        s_rep = encode_sensor_image(s_img)
        r_rep = encode_sensor_image(r_img)
        c_m1, c_m2 = st.columns(2)
        c_m1.image(s_rep.structural_map, caption=f"Source Structural Map ({s_rep.sensor_type})", use_container_width=True)
        c_m2.image(r_rep.structural_map, caption=f"Reference Structural Map ({r_rep.sensor_type})", use_container_width=True)

    with tabs[2]:
        st.write("**Stage 3: Illumination-Robust Phase Congruency & Reliability**")
        st.write("Convolving with 2D Log-Gabor filter bank (3 scales, 6 orientations) extracts illumination-invariant maximum moment maps ($M$) and shadow-reliability masks, making detection immune to sun angle flips.")
        from src.illumination.phase_congruency import LogGaborPhaseCongruency
        from src.illumination.reliability import compute_terrain_reliability
        lg = LogGaborPhaseCongruency()
        pc_r = lg.compute(r_img.normalized)
        rel_r = compute_terrain_reliability(r_img.normalized, pc_r.max_moment)
        c_i1, c_i2 = st.columns(2)
        c_i1.image(pc_r.max_moment, caption="Phase Congruency Max Moment (Crater Rims)", use_container_width=True)
        c_i2.image(rel_r, caption="Terrain Reliability Map (Texture & Shadow Filter)", use_container_width=True)

    with tabs[3]:
        st.write("**Stage 4: Candidate Correspondences**")
        st.write(f"Identified {m.total_candidates} candidate matches across the scene using {feature_method}.")
        if res.matcher_info and "LOFTR" in str(res.matcher_info.get("active_backend", "")):
            if res.matcher_info.get("checkpoint_loaded"):
                st.info(f"🛰️ **Active LoFTR Weights:** Lunar Fine-Tuned Checkpoint (`{res.matcher_info.get('checkpoint_path')}`) | Supervised on calibrated lunar geometry.")
            else:
                st.warning("⚠️ **Active LoFTR Weights:** Pretrained Outdoor Fallback | No fine-tuned lunar checkpoint found.")
        st.image(res.vis_matches, caption="Candidate Matches (Green: Inliers | Red: Filtered Outliers)", use_container_width=True)

    with tabs[4]:
        st.write("**Stage 5: Robust Outlier Rejection (RANSAC / MAGSAC++)**")
        st.write(f"Filtered {len(res.estimation_result.outliers)} false correspondences ({1.0 - m.inlier_ratio:.1%} outlier rate), isolating {m.inlier_count} geometric inliers.")
        st.metric("Outliers Rejected", len(res.estimation_result.outliers))

    with tabs[5]:
        st.write("**Stage 6: Spatially Uniform Control-Point Optimization (ANMS)**")
        st.write(f"Grid-based Adaptive Non-Maximal Suppression (ANMS) spreads control points uniformly across the reference frame, achieving {m.grid_coverage_ratio:.1%} coverage and avoiding single-crater clustering.")
        st.image(res.vis_spatial, caption="Spatial Uniformity: Grid Cell Coverage (Green: Occupied | Yellow: Control Points)", use_container_width=True)

    with tabs[6]:
        st.write("**Stage 7: Sub-Pixel Correspondence Refinement**")
        st.write("Local Normalized Cross-Correlation (NCC) with 2D continuous parabolic surface interpolation refines integer pixel coordinates to sub-pixel floating-point positions.")
        st.image(res.vis_subpixel, caption="Sub-Pixel Correction Vector Field (quiver plot scaled 10x)", use_container_width=True)

    with tabs[7]:
        st.write("**Stage 8: Geometric Model Estimation & Stability**")
        st.write(f"Model Selected: **{m.model_name}** | Reason: {res.estimation_result.model_selection_reason}")
        st.write(f"Matrix Condition Number: **{m.condition_number:.1f}** (Threshold < 2000 for stable homography)")
        st.write(f"Median Residual: **{m.median_residual_px:.3f} px** | 95th Percentile: **{m.p95_residual_px:.3f} px**")

    with tabs[8]:
        st.write("**Stage 9: High-Fidelity Warping & Resampling**")
        st.write("Source image warped into the reference coordinate frame using bicubic spline interpolation.")
        st.image(res.warped_image, caption="Registered Source Image (Transformed into Reference Frame)", use_container_width=True)

    with tabs[9]:
        st.write("**Stage 10: Cartographic Quality Inspection (Checkerboard & Difference)**")
        st.write("Inspect continuous crater rim alignment across checkerboard boundaries to visually verify sub-pixel seam continuity.")
        st.image(res.vis_checkerboard, caption="Checkerboard Verification: Seam alignment between Registered Source & Reference", use_container_width=True)
        st.image(res.vis_diff, caption="Absolute Difference Heatmap (Dark = Perfect Alignment | Bright = Residuals)", use_container_width=True)

    with tabs[10]:
        st.write("**Stage 11: Exportable SIH Artifacts**")
        st.write("Download complete scientific registration deliverables: GeoTIFF, CSV control points, GeoJSON, and Metrics JSON.")
        
        df = res.export_control_points_dataframe()
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        geojson_bytes = json.dumps(res.export_control_points_geojson(), indent=2).encode('utf-8')
        json_bytes = res.metrics.to_json().encode('utf-8')

        c_d1, c_d2, c_d3 = st.columns(3)
        c_d1.download_button("📥 Control Points (CSV)", csv_bytes, file_name="lunar_control_points.csv", mime="text/csv")
        c_d2.download_button("📥 Control Points (GeoJSON)", geojson_bytes, file_name="lunar_control_points.geojson", mime="application/geo+json")
        c_d3.download_button("📥 Registration Metrics (JSON)", json_bytes, file_name="registration_metrics.json", mime="application/json")

        st.dataframe(df.head(10), use_container_width=True)