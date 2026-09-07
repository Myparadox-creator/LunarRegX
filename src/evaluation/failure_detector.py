"""
Scientific Failure Detection Engine.
Automatically detects unphysical, degenerate, or poorly constrained lunar registrations.
Emits clear human-readable SUCCESS, WARNING, or FAILURE status.
"""
from typing import List, Tuple
from src.evaluation.metrics import RegistrationMetrics

class FailureDetector:
    def __init__(
        self,
        min_inliers_fail: int = 8,
        min_inliers_warn: int = 15,
        min_inlier_ratio_fail: float = 0.10,
        min_inlier_ratio_warn: float = 0.20,
        min_coverage_warn: float = 0.35,
        max_rmse_fail: float = 4.0,
        max_rmse_warn: float = 2.0,
        max_condition_warn: float = 5000.0,
        max_condition_fail: float = 1e6
    ):
        self.min_inliers_fail = min_inliers_fail
        self.min_inliers_warn = min_inliers_warn
        self.min_inlier_ratio_fail = min_inlier_ratio_fail
        self.min_inlier_ratio_warn = min_inlier_ratio_warn
        self.min_coverage_warn = min_coverage_warn
        self.max_rmse_fail = max_rmse_fail
        self.max_rmse_warn = max_rmse_warn
        self.max_condition_warn = max_condition_warn
        self.max_condition_fail = max_condition_fail

    def diagnose(self, metrics: RegistrationMetrics) -> Tuple[str, List[str]]:
        """
        Evaluate metrics against criteria and return (status, diagnostic_messages).
        """
        reasons: List[str] = []
        is_failure = False

        # Check inlier count
        if metrics.inlier_count < self.min_inliers_fail:
            reasons.append(f"CRITICAL: Too few geometric inliers ({metrics.inlier_count} < {self.min_inliers_fail}). Geometry cannot be reliably constrained.")
            is_failure = True
        elif metrics.inlier_count < self.min_inliers_warn:
            reasons.append(f"WARNING: Low inlier count ({metrics.inlier_count} < {self.min_inliers_warn}). Transformation has low statistical redundancy.")

        # Check inlier ratio
        if metrics.inlier_ratio < self.min_inlier_ratio_fail:
            reasons.append(f"CRITICAL: Severe outlier contamination (inlier ratio {metrics.inlier_ratio:.1%} < {self.min_inlier_ratio_fail:.1%}).")
            is_failure = True
        elif metrics.inlier_ratio < self.min_inlier_ratio_warn:
            reasons.append(f"WARNING: High outlier ratio (inlier ratio {metrics.inlier_ratio:.1%}).")

        # Check RMSE
        if metrics.rmse_total_px > self.max_rmse_fail:
            reasons.append(f"CRITICAL: Excessive reprojection residual (Total RMSE {metrics.rmse_total_px:.2f} px > {self.max_rmse_fail:.1f} px).")
            is_failure = True
        elif metrics.rmse_total_px > self.max_rmse_warn:
            reasons.append(f"WARNING: Elevated residual error (Total RMSE {metrics.rmse_total_px:.2f} px).")

        # Check spatial coverage
        if metrics.grid_coverage_ratio < self.min_coverage_warn:
            reasons.append(f"WARNING: Poor spatial distribution: control points cover only {metrics.grid_coverage_ratio:.1%} of frame (< {self.min_coverage_warn:.1%}). Edge regions may diverge.")

        # Check condition number
        if metrics.condition_number > self.max_condition_fail:
            reasons.append(f"CRITICAL: Degenerate matrix condition number ({metrics.condition_number:.1e}). Model exhibits extreme projective shearing.")
            is_failure = True
        elif metrics.condition_number > self.max_condition_warn:
            reasons.append(f"WARNING: High matrix condition number ({metrics.condition_number:.1f}). Sensitivity to match noise.")

        if is_failure:
            status = "FAILURE"
        elif len(reasons) > 0:
            status = "WARNING"
        else:
            status = "SUCCESS"
            reasons.append("Optimal registration: High inlier confidence, sub-pixel residual, and uniform spatial coverage.")

        return status, reasons
