"""
Quantitative Spatial Distribution and Coverage Metrics.
Measures spatial uniformity, grid dispersion, and clustering avoidance for SIH evaluation.
"""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.spatial import ConvexHull
from src.matching.matcher import Correspondence

@dataclass
class SpatialCoverageMetrics:
    grid_coverage_ratio: float       # Occupied cells / total valid cells
    num_occupied_cells: int
    total_cells: int
    density_variance: float          # Variance of point counts across cells
    spatial_entropy: float           # Information entropy of spatial distribution
    convex_hull_area_ratio: float    # Hull area / total image area
    max_uncovered_radius_px: float   # Largest circle without control points

def compute_spatial_coverage(
    correspondences: List[Correspondence],
    ref_shape: Tuple[int, int],
    grid_cols: int = 10,
    grid_rows: int = 10
) -> SpatialCoverageMetrics:
    """
    Compute rigorous spatial uniformity metrics across reference image frame.
    """
    h, w = ref_shape[:2]
    total_cells = grid_cols * grid_rows

    if not correspondences:
        return SpatialCoverageMetrics(
            grid_coverage_ratio=0.0,
            num_occupied_cells=0,
            total_cells=total_cells,
            density_variance=0.0,
            spatial_entropy=0.0,
            convex_hull_area_ratio=0.0,
            max_uncovered_radius_px=float(max(h, w))
        )

    cell_w = max(w / grid_cols, 1.0)
    cell_h = max(h / grid_rows, 1.0)

    cell_counts = np.zeros((grid_rows, grid_cols), dtype=np.int32)
    pts = []

    for c in correspondences:
        rx, ry = c.ref_pt
        gx = int(np.clip(rx // cell_w, 0, grid_cols - 1))
        gy = int(np.clip(ry // cell_h, 0, grid_rows - 1))
        cell_counts[gy, gx] += 1
        pts.append([rx, ry])

    pts = np.array(pts, dtype=np.float32)
    occupied = int(np.sum(cell_counts > 0))
    coverage_ratio = float(occupied / total_cells)

    # Density variance
    density_var = float(np.var(cell_counts))

    # Spatial entropy: - sum(p * log2(p))
    total_pts = len(correspondences)
    probs = cell_counts.flatten() / float(total_pts)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))

    # Convex hull ratio
    hull_ratio = 0.0
    if len(pts) >= 3:
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume  # 2D volume is area
            total_area = float(h * w)
            hull_ratio = float(np.clip(hull_area / total_area, 0.0, 1.0))
        except Exception:
            hull_ratio = 0.0

    # Max uncovered radius approximation via grid centers
    max_radius = 0.0
    empty_indices = np.argwhere(cell_counts == 0)
    if empty_indices.size > 0 and pts.size > 0:
        # Distance from center of each empty cell to nearest control point
        for ey, ex in empty_indices:
            center_x = (ex + 0.5) * cell_w
            center_y = (ey + 0.5) * cell_h
            min_dist = np.min(np.hypot(pts[:, 0] - center_x, pts[:, 1] - center_y))
            if min_dist > max_radius:
                max_radius = float(min_dist)
    else:
        max_radius = float(np.hypot(cell_w, cell_h))

    return SpatialCoverageMetrics(
        grid_coverage_ratio=coverage_ratio,
        num_occupied_cells=occupied,
        total_cells=total_cells,
        density_variance=density_var,
        spatial_entropy=entropy,
        convex_hull_area_ratio=hull_ratio,
        max_uncovered_radius_px=max_radius
    )
