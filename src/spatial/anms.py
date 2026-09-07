"""
Spatially Uniform Match Optimization.
Adaptive Non-Maximal Suppression (ANMS) and Grid Binning for Planetary Imagery.
Prevents clustering around a single prominent crater and enforces uniform spatial coverage.
"""
from typing import List, Tuple
import numpy as np
from src.matching.matcher import Correspondence

class SpatialOptimizer:
    def __init__(
        self,
        grid_cols: int = 10,
        grid_rows: int = 10,
        max_points_per_cell: int = 4,
        min_distance_px: float = 12.0
    ):
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.max_points_per_cell = max_points_per_cell
        self.min_distance_px = min_distance_px

    def optimize(
        self,
        correspondences: List[Correspondence],
        ref_shape: Tuple[int, int]
    ) -> List[Correspondence]:
        """
        Enforce spatially uniform control points using grid-binned ANMS.
        """
        if not correspondences:
            return []

        h, w = ref_shape[:2]
        cell_w = max(w / self.grid_cols, 1.0)
        cell_h = max(h / self.grid_rows, 1.0)

        # Group correspondences by reference image grid cell
        grid = {}
        for corr in correspondences:
            rx, ry = corr.ref_pt
            gx = int(np.clip(rx // cell_w, 0, self.grid_cols - 1))
            gy = int(np.clip(ry // cell_h, 0, self.grid_rows - 1))
            cell_key = (gx, gy)
            corr.grid_cell = cell_key
            if cell_key not in grid:
                grid[cell_key] = []
            grid[cell_key].append(corr)

        selected: List[Correspondence] = []

        # In each cell, sort by confidence score and apply minimum distance suppression
        for cell_key, cell_matches in grid.items():
            cell_matches.sort(key=lambda c: c.confidence, reverse=True)
            cell_selected: List[Correspondence] = []

            for cand in cell_matches:
                if len(cell_selected) >= self.max_points_per_cell:
                    break

                # Distance check against already accepted points in this cell
                too_close = False
                cx, cy = cand.ref_pt
                for acc in cell_selected:
                    ax, ay = acc.ref_pt
                    dist = np.hypot(cx - ax, cy - ay)
                    if dist < self.min_distance_px:
                        too_close = True
                        break

                if not too_close:
                    cell_selected.append(cand)

            selected.extend(cell_selected)

        return selected
