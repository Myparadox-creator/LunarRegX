"""
Lunar Image Footprint Generation and Spatial Overlap Analysis.
Computes Selenographic polygon boundaries, polygon intersection using Shapely,
overlap percentages, scale ratios, and common ROI pixel windows.
"""
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import transform
from .lunar_crs import LunarCRS

@dataclass
class LunarFootprint:
    polygon_geo: Polygon       # Selenographic coordinates (lon_deg, lat_deg)
    polygon_proj: Polygon      # Projected meters coordinates (x_m, y_m)
    bounds_geo: Tuple[float, float, float, float]   # min_lon, min_lat, max_lon, max_lat
    bounds_proj: Tuple[float, float, float, float]  # min_x, min_y, max_x, max_y
    area_km2: float
    gsd_m: float
    crs: LunarCRS

def create_footprint_from_bounds(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    gsd_m: float = 1.0,
    crs: Optional[LunarCRS] = None
) -> LunarFootprint:
    """
    Generate LunarFootprint from selenographic bounding coordinates.
    """
    l_crs = crs or LunarCRS(projection_type="EQUIRECTANGULAR", center_lon_deg=(min_lon + max_lon) / 2.0, center_lat_deg=(min_lat + max_lat) / 2.0)
    poly_geo = box(min_lon, min_lat, max_lon, max_lat)

    # Project coordinates to meters
    min_x, min_y = l_crs.forward(min_lon, min_lat)
    max_x, max_y = l_crs.forward(max_lon, max_lat)
    poly_proj = box(min(min_x, max_x), min(min_y, max_y), max(min_x, max_x), max(min_y, max_y))

    area_km2 = float(poly_proj.area / 1e6)
    return LunarFootprint(
        polygon_geo=poly_geo,
        polygon_proj=poly_proj,
        bounds_geo=(min_lon, min_lat, max_lon, max_lat),
        bounds_proj=(min(min_x, max_x), min(min_y, max_y), max(min_x, max_x), max(min_y, max_y)),
        area_km2=area_km2,
        gsd_m=gsd_m,
        crs=l_crs
    )

def create_footprint_from_dimensions(
    width_px: int,
    height_px: int,
    center_lon: float,
    center_lat: float,
    gsd_m: float,
    crs: Optional[LunarCRS] = None
) -> LunarFootprint:
    """
    Generate LunarFootprint given image dimensions (px), center coordinate, and GSD (m/px).
    """
    l_crs = crs or LunarCRS(
        projection_type="POLAR_STEREOGRAPHIC_SOUTH" if abs(center_lat) >= 65.0 and center_lat < 0 else (
            "POLAR_STEREOGRAPHIC_NORTH" if center_lat >= 65.0 else "EQUIRECTANGULAR"
        ),
        center_lon_deg=center_lon,
        center_lat_deg=center_lat
    )

    cx_m, cy_m = l_crs.forward(center_lon, center_lat)
    half_w_m = (width_px * gsd_m) / 2.0
    half_h_m = (height_px * gsd_m) / 2.0

    min_x, max_x = cx_m - half_w_m, cx_m + half_w_m
    min_y, max_y = cy_m - half_h_m, cy_m + half_h_m
    poly_proj = box(min_x, min_y, max_x, max_y)

    # Convert corners back to geographic
    c1_lon, c1_lat = l_crs.inverse(min_x, min_y)
    c2_lon, c2_lat = l_crs.inverse(max_x, max_y)
    min_lon, max_lon = min(c1_lon, c2_lon), max(c1_lon, c2_lon)
    min_lat, max_lat = min(c1_lat, c2_lat), max(c1_lat, c2_lat)
    poly_geo = box(min_lon, min_lat, max_lon, max_lat)

    area_km2 = float(poly_proj.area / 1e6)
    return LunarFootprint(
        polygon_geo=poly_geo,
        polygon_proj=poly_proj,
        bounds_geo=(min_lon, min_lat, max_lon, max_lat),
        bounds_proj=(min_x, min_y, max_x, max_y),
        area_km2=area_km2,
        gsd_m=gsd_m,
        crs=l_crs
    )

def compute_footprint_intersection(
    src_footprint: LunarFootprint,
    ref_footprint: LunarFootprint
) -> Dict[str, Any]:
    """
    Calculates spatial overlap percentage, area, and intersection polygon
    by intersecting selenographic boundaries and projecting into the reference lunar CRS.
    """
    inter_geo = src_footprint.polygon_geo.intersection(ref_footprint.polygon_geo)
    has_overlap = not inter_geo.is_empty and inter_geo.area > 0

    if has_overlap and hasattr(inter_geo, "exterior"):
        coords = list(inter_geo.exterior.coords)
        proj_coords = [ref_footprint.crs.forward(lon, lat) for lon, lat in coords]
        inter_proj = Polygon(proj_coords)
        inter_area_km2 = float(inter_proj.area / 1e6)
        inter_bounds_proj = inter_proj.bounds
    else:
        inter_area_km2 = 0.0
        inter_bounds_proj = None

    src_area = src_footprint.area_km2
    ref_area = ref_footprint.area_km2
    overlap_pct_src = min(round((inter_area_km2 / src_area * 100.0), 2), 100.0) if src_area > 0 else 0.0
    overlap_pct_ref = min(round((inter_area_km2 / ref_area * 100.0), 2), 100.0) if ref_area > 0 else 0.0
    scale_ratio = float(src_footprint.gsd_m / ref_footprint.gsd_m) if ref_footprint.gsd_m > 0 else 1.0

    return {
        "overlap_detected": bool(has_overlap and inter_area_km2 > 1e-6),
        "intersection_area_km2": round(inter_area_km2, 4),
        "overlap_percentage_src": overlap_pct_src,
        "overlap_percentage_ref": overlap_pct_ref,
        "overlap_percentage_mean": round((overlap_pct_src + overlap_pct_ref) / 2.0, 2),
        "scale_ratio": round(scale_ratio, 4),
        "source_gsd_m": src_footprint.gsd_m,
        "reference_gsd_m": ref_footprint.gsd_m,
        "intersection_bounds_proj": inter_bounds_proj
    }

def extract_common_roi(
    image: np.ndarray,
    footprint: LunarFootprint,
    target_bounds_proj: Tuple[float, float, float, float]
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crops pixel array to target projected bounding box.
    Returns:
        cropped_array: sub-array of image
        pixel_window: (col_offset, row_offset, width, height)
    """
    h, w = image.shape[:2]
    min_x, min_y, max_x, max_y = footprint.bounds_proj
    t_min_x, t_min_y, t_max_x, t_max_y = target_bounds_proj

    # Compute fractional pixel bounding box
    px_w_m = (max_x - min_x) / w
    px_h_m = (max_y - min_y) / h

    x0 = int(np.clip(round((t_min_x - min_x) / px_w_m), 0, w - 1))
    x1 = int(np.clip(round((t_max_x - min_x) / px_w_m), 1, w))
    # y is inverted in image coordinates (top to bottom)
    y0 = int(np.clip(round((max_y - t_max_y) / px_h_m), 0, h - 1))
    y1 = int(np.clip(round((max_y - t_min_y) / px_h_m), 1, h))

    if x1 <= x0 or y1 <= y0:
        return image, (0, 0, w, h)

    cropped = image[y0:y1, x0:x1]
    return cropped, (x0, y0, x1 - x0, y1 - y0)