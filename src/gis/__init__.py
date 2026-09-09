"""
GIS / Spatial Intelligence Layer for Lunar Remote Sensing.
Provides authentic Lunar datums, coordinate reference systems, footprint geometry,
spatial overlap analysis, and common Region of Interest (ROI) extraction.
"""
from .lunar_crs import LunarCRS, SelenographicPoint, ProjectedPoint
from .footprint import LunarFootprint, compute_footprint_intersection, extract_common_roi
from .pre_registration import GISPreRegistrationAnalyzer, SpatialPreRegistrationResult

__all__ = [
    "LunarCRS",
    "SelenographicPoint",
    "ProjectedPoint",
    "LunarFootprint",
    "compute_footprint_intersection",
    "extract_common_roi",
    "GISPreRegistrationAnalyzer",
    "SpatialPreRegistrationResult",
]