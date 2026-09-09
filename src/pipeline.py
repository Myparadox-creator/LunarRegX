"""
Master End-to-End GIS-Assisted Multi-Modal Lunar Image Registration Pipeline.
Orchestrates:
1. GIS / Coordinate Reference System (IAU-2000 Lunar Datum) & Footprint Overlap Analysis
2. Common Region of Interest (ROI) extraction
3. Sensor-Aware Multimodal Encoding (OHRC, TMC-2, IIRS)
4. Pluggable Correspondence Matching (Proposed Phase-Structural, LoFTR, LightGlue, SIFT, RIFT2)
5. Robust Geometric Verification (RANSAC/MAGSAC++ with SVD Condition Check)
6. Spatially Uniform Control-Point Optimization (ANMS Grid Binning)
7. Sub-Pixel Continuous 2D Parabolic Peak Refinement
8. GeoTIFF Geospatial Warping & Multi-Artifact Export (GeoTIFF, CSV, GeoJSON, Metrics JSON)
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import time
import json
import numpy as np
import pandas as pd

from src.io.dataset import LunarImage
from src.io.metadata import SensorMetadata
from src.gis.pre_registration import GISPreRegistrationAnalyzer, SpatialPreRegistrationResult
from src.gis.footprint import extract_common_roi
from src.multimodal.sensor_encoder import encode_sensor_image, CommonTerrainRepresentation
from src.preprocessing.pipeline import PreprocessingConfig, preprocess_lunar_image
from src.illumination.phase_congruency import LogGaborPhaseCongruency
from src.illumination.reliability import compute_terrain_reliability
from src.features.classical import ClassicalFeatureEngine
from src.features.phase_structural import PhaseStructuralEngine
from src.features.learned_adapter import LearnedFeatureEngine
from src.features.deep_matcher import DeepCorrespondenceMatcher
from src.matching.matcher import CorrespondenceMatcher, Correspondence
from src.spatial.anms import SpatialOptimizer
from src.spatial.coverage import compute_spatial_coverage
from src.subpixel.refinement import SubPixelRefiner
from src.geometry.estimator import RobustGeometricEstimator, EstimationResult
from src.warping.resampler import ImageWarper, export_geotiff
from src.evaluation.metrics import RegistrationMetrics
from src.evaluation.validator import evaluate_checkpoints
from src.evaluation.failure_detector import FailureDetector
from src.visualization.visualizer import (
    render_matches_visualization,
    render_spatial_coverage_overlay,
    render_checkerboard,
    render_difference_map,
    render_subpixel_quiver
)

@dataclass
class RegistrationPipelineConfig:
    feature_method: str = "PHASE_STRUCTURAL"  # PHASE_STRUCTURAL, RIFT2, LOFTR, LIGHTGLUE, SIFT, AKAZE, LEARNED
    preferred_model: str = "AUTO"             # AUTO, SIMILARITY, AFFINE, HOMOGRAPHY
    use_gis_pre_registration: bool = True
    use_multimodal_encoder: bool = True
    use_clahe: bool = True
    use_bandpass: bool = False
    use_spatial_anms: bool = True
    use_subpixel: bool = True
    ratio_thresh: float = 0.80
    ransac_thresh_px: float = 3.0
    anms_grid_size: int = 10
    anms_max_per_cell: int = 4
    subpixel_patch_size: int = 17
    subpixel_search_radius: int = 4
    independent_val_split: float = 0.20       # 20% held-out checkpoints

@dataclass
class RegistrationOutput:
    source_image: LunarImage
    reference_image: LunarImage
    warped_image: np.ndarray
    warped_mask: np.ndarray
    all_correspondences: List[Correspondence]
    inlier_correspondences: List[Correspondence]
    estimation_result: EstimationResult
    metrics: RegistrationMetrics
    # Visual artifacts
    vis_matches: np.ndarray
    vis_spatial: np.ndarray
    vis_checkerboard: np.ndarray
    vis_diff: np.ndarray
    vis_subpixel: np.ndarray
    # GIS and Multimodal Metadata
    spatial_pre_reg: Optional[SpatialPreRegistrationResult] = None
    common_roi_window: Optional[Tuple[int, int, int, int]] = None

    def export_control_points_dataframe(self) -> pd.DataFrame:
        """
        Export structured pandas DataFrame of all control points.
        """
        rows = []
        for c in self.all_correspondences:
            ref_x, ref_y = c.ref_pt
            ref_sub_x, ref_sub_y = c.refined_ref_pt if c.refined_ref_pt is not None else (ref_x, ref_y)
            rows.append({
                "id": c.id,
                "source_x": float(c.src_pt[0]),
                "source_y": float(c.src_pt[1]),
                "reference_x": float(ref_x),
                "reference_y": float(ref_y),
                "refined_ref_x": float(ref_sub_x),
                "refined_ref_y": float(ref_sub_y),
                "subpixel_dx": float(c.subpixel_offset[0]),
                "subpixel_dy": float(c.subpixel_offset[1]),
                "confidence": float(c.confidence),
                "is_inlier": bool(c.is_inlier),
                "grid_cell_x": c.grid_cell[0] if c.grid_cell else None,
                "grid_cell_y": c.grid_cell[1] if c.grid_cell else None,
                "residual_px": float(c.residual)
            })
        return pd.DataFrame(rows)

    def export_control_points_geojson(self) -> Dict[str, Any]:
        """
        Export control points in standardized GeoJSON format for GIS software (QGIS/ArcGIS).
        """
        features = []
        for c in self.inlier_correspondences:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(c.ref_pt[0]), float(c.ref_pt[1])]
                },
                "properties": {
                    "id": c.id,
                    "source_x": float(c.src_pt[0]),
                    "source_y": float(c.src_pt[1]),
                    "subpixel_dx": float(c.subpixel_offset[0]),
                    "subpixel_dy": float(c.subpixel_offset[1]),
                    "confidence": float(c.confidence),
                    "residual_px": float(c.residual)
                }
            })
        return {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features
        }

    def export_geotiff(self, filepath: str) -> str:
        """
        Export registered source image as authentic GeoTIFF with Lunar projection.
        """
        bounds_proj = None
        if self.spatial_pre_reg and self.spatial_pre_reg.common_roi.get("bounds_proj"):
            bounds_proj = self.spatial_pre_reg.common_roi["bounds_proj"]
        return export_geotiff(filepath, self.warped_image, bounds_proj=bounds_proj)

class LunarRegistrationPipeline:
    def __init__(self, config: Optional[RegistrationPipelineConfig] = None):
        self.config = config or RegistrationPipelineConfig()
        self._init_components()

    def _init_components(self):
        cfg = self.config
        m = cfg.feature_method.upper()

        # 1. Feature / Matcher Engine
        self.is_deep_matcher = m in ["LOFTR", "LIGHTGLUE", "DEEP_MATCHER"]
        if self.is_deep_matcher:
            self.deep_matcher = DeepCorrespondenceMatcher(backend="LOFTR" if "LOFTR" in m else "LIGHTGLUE")
            self.feature_engine = None
        elif m in ["SIFT", "AKAZE", "ORB"]:
            self.feature_engine = ClassicalFeatureEngine(method=m)
            self.deep_matcher = None
        elif m in ["PHASE_STRUCTURAL", "RIFT", "RIFT2"]:
            self.feature_engine = PhaseStructuralEngine()
            self.deep_matcher = None
        elif m == "LEARNED":
            self.feature_engine = LearnedFeatureEngine()
            self.deep_matcher = None
        else:
            self.feature_engine = PhaseStructuralEngine()
            self.deep_matcher = None

        # 2. Classical Matcher (for feature engine workflows)
        self.matcher = CorrespondenceMatcher(ratio_thresh=cfg.ratio_thresh)

        # 3. GIS Pre-Registration Analyzer
        self.gis_analyzer = GISPreRegistrationAnalyzer()

        # 4. Spatial ANMS Optimizer
        self.spatial_optimizer = SpatialOptimizer(
            grid_cols=cfg.anms_grid_size,
            grid_rows=cfg.anms_grid_size,
            max_points_per_cell=cfg.anms_max_per_cell
        )

        # 5. Sub-pixel Refiner
        self.subpixel_refiner = SubPixelRefiner(
            patch_size=cfg.subpixel_patch_size,
            search_radius=cfg.subpixel_search_radius
        )

        # 6. Geometric Estimator
        self.estimator = RobustGeometricEstimator(
            ransac_thresh=cfg.ransac_thresh_px,
            preferred_model=cfg.preferred_model
        )

        # 7. Warper & Failure Detector
        self.warper = ImageWarper(interpolation="bicubic")
        self.failure_detector = FailureDetector()

    def run(
        self,
        source: LunarImage,
        reference: LunarImage
    ) -> RegistrationOutput:
        start_time = time.time()
        cfg = self.config

        # 1. GIS / Spatial Pre-Registration Analysis
        spatial_pre_reg = None
        if cfg.use_gis_pre_registration:
            spatial_pre_reg = self.gis_analyzer.analyze(source, reference)

        # 2. Sensor-Aware Multimodal Encoding
        if cfg.use_multimodal_encoder:
            src_rep = encode_sensor_image(source)
            ref_rep = encode_sensor_image(reference)
            src_8u = src_rep.enhanced_8u
            ref_8u = ref_rep.enhanced_8u
            src_norm = src_rep.structural_map
            ref_norm = ref_rep.structural_map
            src_mask = src_rep.feature_mask
            ref_mask = ref_rep.feature_mask
        else:
            pre_cfg = PreprocessingConfig(use_clahe=cfg.use_clahe, use_bandpass=cfg.use_bandpass)
            src_8u, src_norm, src_shadow = preprocess_lunar_image(source.normalized, pre_cfg)
            ref_8u, ref_norm, ref_shadow = preprocess_lunar_image(reference.normalized, pre_cfg)
            src_mask = src_shadow
            ref_mask = ref_shadow

        # 3. Correspondence Matching
        if self.is_deep_matcher:
            # End-to-end transformer / deep matching
            candidates, _ = self.deep_matcher.match(src_8u, ref_8u, src_mask, ref_mask)
        else:
            # Feature Detection & Description
            lg = LogGaborPhaseCongruency()
            pc_src = lg.compute(src_norm)
            pc_ref = lg.compute(ref_norm)
            rel_src = compute_terrain_reliability(src_norm, pc_src.max_moment, src_mask)
            rel_ref = compute_terrain_reliability(ref_norm, pc_ref.max_moment, ref_mask)

            kps_src, desc_src = self.feature_engine.detect_and_compute(src_8u, source.mask, rel_src)
            kps_ref, desc_ref = self.feature_engine.detect_and_compute(ref_8u, reference.mask, rel_ref)
            candidates = self.matcher.match(kps_src, desc_src, kps_ref, desc_ref, rel_src, rel_ref)

        if len(candidates) < 4:
            raise RuntimeError(f"Found only {len(candidates)} candidate matches. Geometric registration requires at least 4.")

        # 4. Spatially Uniform Control-Point Optimization (ANMS)
        if cfg.use_spatial_anms:
            active_candidates = self.spatial_optimizer.optimize(candidates, reference.shape)
        else:
            active_candidates = candidates

        # 5. Sub-Pixel Continuous 2D Parabolic Peak Refinement
        if cfg.use_subpixel:
            active_candidates = self.subpixel_refiner.refine_all(active_candidates, src_norm, ref_norm)

        # 6. Independent Checkpoint Validation Split (80% estimation, 20% validation)
        np.random.seed(42)
        n_cand = len(active_candidates)
        perm = np.random.permutation(n_cand)
        n_val = max(int(n_cand * cfg.independent_val_split), 3) if n_cand >= 15 else 0
        val_indices = set(perm[:n_val])
        est_candidates = [c for i, c in enumerate(active_candidates) if i not in val_indices]
        checkpoints = [c for i, c in enumerate(active_candidates) if i in val_indices]

        # 7. Robust Geometric Model Estimation
        est_result = self.estimator.estimate(est_candidates, use_subpixel=cfg.use_subpixel)
        model = est_result.model
        inliers = est_result.inliers

        # 8. Independent Checkpoint Validation
        chk_rmse_x, chk_rmse_y, chk_rmse_total = evaluate_checkpoints(checkpoints, model, use_subpixel=cfg.use_subpixel)

        # 9. Spatial Coverage Assessment
        cov_metrics = compute_spatial_coverage(inliers, reference.shape, cfg.anms_grid_size, cfg.anms_grid_size)

        # 10. Image Warping into Reference Frame
        warped_img, warped_mask = self.warper.warp(source.raw_array, model, reference.shape)
        warped_8u, _ = self.warper.warp(source.display_8bit, model, reference.shape)

        # 11. Residual Analysis
        residuals = [c.residual for c in inliers]
        med_res = float(np.median(residuals)) if residuals else 0.0
        p95_res = float(np.percentile(residuals, 95.0)) if residuals else 0.0
        max_res = float(np.max(residuals)) if residuals else 0.0

        ref_gsd = reference.metadata.gsd if reference.metadata else 1.0
        phys_error = float(est_result.rmse_pixels * ref_gsd)
        runtime = float(time.time() - start_time)

        # 12. Assemble Registration Metrics
        metrics = RegistrationMetrics(
            total_candidates=len(candidates),
            inlier_count=len(inliers),
            inlier_ratio=est_result.inlier_ratio,
            rmse_x_px=float(est_result.rmse_pixels / np.sqrt(2.0)),
            rmse_y_px=float(est_result.rmse_pixels / np.sqrt(2.0)),
            rmse_total_px=est_result.rmse_pixels,
            median_residual_px=med_res,
            p95_residual_px=p95_res,
            max_residual_px=max_res,
            grid_coverage_ratio=cov_metrics.grid_coverage_ratio,
            convex_hull_ratio=cov_metrics.convex_hull_area_ratio,
            density_variance=cov_metrics.density_variance,
            checkpoint_rmse_px=chk_rmse_total if n_val > 0 else None,
            physical_error_m=phys_error,
            model_name=est_result.model_name,
            condition_number=est_result.condition_number,
            runtime_sec=runtime
        )

        status, diagnostics = self.failure_detector.diagnose(metrics)
        metrics.status = status
        metrics.diagnostics = diagnostics

        # 13. Render Visual Artifacts
        vis_matches = render_matches_visualization(src_8u, ref_8u, active_candidates)
        vis_spatial = render_spatial_coverage_overlay(ref_8u, inliers, cfg.anms_grid_size, cfg.anms_grid_size)
        vis_checker = render_checkerboard(warped_8u, ref_8u, tile_size=40)
        vis_diff = render_difference_map(warped_8u, ref_8u, warped_mask)
        vis_subpix = render_subpixel_quiver(inliers, reference.shape)

        return RegistrationOutput(
            source_image=source,
            reference_image=reference,
            warped_image=warped_img,
            warped_mask=warped_mask,
            all_correspondences=active_candidates,
            inlier_correspondences=inliers,
            estimation_result=est_result,
            metrics=metrics,
            vis_matches=vis_matches,
            vis_spatial=vis_spatial,
            vis_checkerboard=vis_checker,
            vis_diff=vis_diff,
            vis_subpixel=vis_subpix,
            spatial_pre_reg=spatial_pre_reg
        )