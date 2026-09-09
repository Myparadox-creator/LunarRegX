"""
GIS Pre-Registration Analyzer.
Pre-analyzes spatial metadata, footprint geometry, sun geometry, and camera viewing angle
before executing expensive visual/learned correspondence matching.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
from src.io.dataset import LunarImage
from .lunar_crs import LunarCRS
from .footprint import (
    create_footprint_from_dimensions,
    compute_footprint_intersection,
    LunarFootprint
)

@dataclass
class SpatialPreRegistrationResult:
    overlap_detected: bool
    overlap_percentage: float
    source_gsd: float
    reference_gsd: float
    scale_ratio: float
    source_footprint: Dict[str, Any]
    reference_footprint: Dict[str, Any]
    common_roi: Dict[str, Any]
    sun_geometry: Dict[str, Any]
    camera_geometry: Dict[str, Any]
    metadata_status: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overlap_detected": self.overlap_detected,
            "overlap_percentage": self.overlap_percentage,
            "source_gsd": self.source_gsd,
            "reference_gsd": self.reference_gsd,
            "scale_ratio": self.scale_ratio,
            "source_footprint": self.source_footprint,
            "reference_footprint": self.reference_footprint,
            "common_roi": self.common_roi,
            "sun_geometry": self.sun_geometry,
            "camera_geometry": self.camera_geometry,
            "metadata_status": self.metadata_status
        }

class GISPreRegistrationAnalyzer:
    """
    Evaluates GIS spatial overlap and radiometric geometry between two lunar scenes.
    """
    def __init__(self, default_lat: float = -70.0, default_lon: float = 0.0):
        self.default_lat = default_lat
        self.default_lon = default_lon

    def analyze(self, source: LunarImage, reference: LunarImage) -> SpatialPreRegistrationResult:
        s_meta = source.metadata
        r_meta = reference.metadata

        s_gsd = float(s_meta.gsd) if s_meta and s_meta.gsd else 1.0
        r_gsd = float(r_meta.gsd) if r_meta and r_meta.gsd else 1.0

        # Center selenographic coordinates
        s_lat = s_meta.extra_attributes.get("center_latitude", self.default_lat) if s_meta else self.default_lat
        s_lon = s_meta.extra_attributes.get("center_longitude", self.default_lon) if s_meta else self.default_lon
        r_lat = r_meta.extra_attributes.get("center_latitude", s_lat) if r_meta else s_lat
        r_lon = r_meta.extra_attributes.get("center_longitude", s_lon) if r_meta else s_lon

        # Build footprints
        src_fp = create_footprint_from_dimensions(source.width, source.height, s_lon, s_lat, s_gsd)
        ref_fp = create_footprint_from_dimensions(reference.width, reference.height, r_lon, r_lat, r_gsd)

        # Intersection analysis
        overlap_info = compute_footprint_intersection(src_fp, ref_fp)

        # Sun Geometry
        s_sun_az = s_meta.solar_azimuth_deg if s_meta else None
        s_sun_el = s_meta.solar_elevation_deg if s_meta else None
        r_sun_az = r_meta.solar_azimuth_deg if r_meta else None
        r_sun_el = r_meta.solar_elevation_deg if r_meta else None

        az_delta = abs(s_sun_az - r_sun_az) if (s_sun_az is not None and r_sun_az is not None) else None
        if az_delta is not None and az_delta > 180.0:
            az_delta = 360.0 - az_delta

        sun_geom = {
            "source_sun_azimuth_deg": s_sun_az,
            "source_sun_elevation_deg": s_sun_el,
            "reference_sun_azimuth_deg": r_sun_az,
            "reference_sun_elevation_deg": r_sun_el,
            "azimuth_delta_deg": az_delta,
            "illumination_status": "EXTREME_SHADOW_FLIP" if (az_delta is not None and az_delta >= 120.0) else "NOMINAL"
        }

        # Camera Geometry
        cam_geom = {
            "source_incidence_deg": s_meta.incidence_angle_deg if s_meta else None,
            "reference_incidence_deg": r_meta.incidence_angle_deg if r_meta else None,
            "source_emission_deg": s_meta.emission_angle_deg if s_meta else None,
            "reference_emission_deg": r_meta.emission_angle_deg if r_meta else None
        }

        # Metadata Status Tracking
        meta_status = {
            "sun_geometry": "available" if (s_sun_az is not None and r_sun_az is not None) else "estimated",
            "camera_geometry": "available" if (s_meta.incidence_angle_deg is not None) else "unavailable",
            "spatial_crs": "available" if "crs" in s_meta.extra_attributes else "estimated",
            "gsd": "available" if s_meta.gsd > 0 else "estimated"
        }

        common_roi_info = {
            "has_roi": overlap_info["overlap_detected"],
            "intersection_area_km2": overlap_info["intersection_area_km2"],
            "bounds_proj": overlap_info["intersection_bounds_proj"]
        }

        return SpatialPreRegistrationResult(
            overlap_detected=overlap_info["overlap_detected"],
            overlap_percentage=overlap_info["overlap_percentage_mean"],
            source_gsd=s_gsd,
            reference_gsd=r_gsd,
            scale_ratio=overlap_info["scale_ratio"],
            source_footprint={
                "bounds_geo": src_fp.bounds_geo,
                "area_km2": src_fp.area_km2,
                "crs": src_fp.crs.projection_type
            },
            reference_footprint={
                "bounds_geo": ref_fp.bounds_geo,
                "area_km2": ref_fp.area_km2,
                "crs": ref_fp.crs.projection_type
            },
            common_roi=common_roi_info,
            sun_geometry=sun_geom,
            camera_geometry=cam_geom,
            metadata_status=meta_status
        )