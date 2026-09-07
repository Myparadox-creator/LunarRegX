"""
Unit tests for spatial optimization (ANMS) and coverage metrics computation.
"""
import numpy as np
import pytest
from src.matching.matcher import Correspondence
from src.spatial.anms import SpatialOptimizer
from src.spatial.coverage import compute_spatial_coverage

def test_spatial_coverage_uniform_dispersion():
    grid_cols = 10
    grid_rows = 10
    corrs = []
    idx = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            x = (c + 0.5) * 50.0
            y = (r + 0.5) * 50.0
            corrs.append(Correspondence(
                id=idx,
                src_pt=(x, y),
                ref_pt=(x, y),
                confidence=0.9,
                dist=0.1,
                ratio=0.5
            ))
            idx += 1

    metrics = compute_spatial_coverage(corrs, (500, 500), grid_cols=10, grid_rows=10)

    assert metrics.grid_coverage_ratio == 1.0
    assert metrics.num_occupied_cells == 100
    assert metrics.convex_hull_area_ratio > 0.70

def test_spatial_optimizer_clustering_suppression():
    optimizer = SpatialOptimizer(grid_cols=10, grid_rows=10, max_points_per_cell=2, min_distance_px=10.0)

    clustered = []
    for i in range(20):
        clustered.append(Correspondence(
            id=i,
            src_pt=(10.0 + i*0.5, 10.0 + i*0.5),
            ref_pt=(10.0 + i*0.5, 10.0 + i*0.5),
            confidence=0.9 - i*0.01,
            dist=0.1,
            ratio=0.5
        ))

    filtered = optimizer.optimize(clustered, (500, 500))

    assert len(filtered) <= 2
