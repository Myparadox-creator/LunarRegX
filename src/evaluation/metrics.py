"""
Comprehensive Registration Metrics Formulation.
Computes inlier statistics, pixel and physical RMSE (meters), spatial distribution metrics,
and residual error distributions.
"""
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import numpy as np
import json

@dataclass
class RegistrationMetrics:
    total_candidates: int
    inlier_count: int
    inlier_ratio: float
    rmse_x_px: float
    rmse_y_px: float
    rmse_total_px: float
    median_residual_px: float
    p95_residual_px: float
    max_residual_px: float
    grid_coverage_ratio: float
    convex_hull_ratio: float
    density_variance: float
    checkpoint_rmse_px: Optional[float] = None
    physical_error_m: Optional[float] = None  # Converted to meters via GSD
    model_name: str = "UNKNOWN"
    condition_number: float = 1.0
    runtime_sec: float = 0.0
    status: str = "SUCCESS"  # SUCCESS, WARNING, FAILURE
    diagnostics: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
