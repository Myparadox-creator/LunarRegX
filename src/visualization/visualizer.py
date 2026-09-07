"""
High-Resolution Visualization and Cartographic Quality Inspection.
Produces correspondence plots, spatial distribution heatmaps, checkerboard comparisons,
sub-pixel quiver plots, and difference images.
"""
from typing import List, Tuple, Optional
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.matching.matcher import Correspondence
from src.geometry.models import GeometricModel

def render_matches_visualization(
    src_img_8u: np.ndarray,
    ref_img_8u: np.ndarray,
    correspondences: List[Correspondence],
    max_display: int = 200
) -> np.ndarray:
    """
    Draw side-by-side matches.
    Green lines: Inliers
    Red lines: Rejected Outliers
    """
    h_s, w_s = src_img_8u.shape[:2]
    h_r, w_r = ref_img_8u.shape[:2]

    out_h = max(h_s, h_r)
    out_w = w_s + w_r

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:h_s, :w_s] = cv2.cvtColor(src_img_8u, cv2.COLOR_GRAY2BGR) if src_img_8u.ndim == 2 else src_img_8u
    canvas[:h_r, w_s:w_s + w_r] = cv2.cvtColor(ref_img_8u, cv2.COLOR_GRAY2BGR) if ref_img_8u.ndim == 2 else ref_img_8u

    # Subsample if too many
    display_matches = correspondences
    if len(display_matches) > max_display:
        step = len(display_matches) // max_display
        display_matches = display_matches[::step]

    # Draw outliers first (red)
    for c in display_matches:
        if not c.is_inlier:
            pt1 = (int(round(c.src_pt[0])), int(round(c.src_pt[1])))
            pt2 = (int(round(c.ref_pt[0] + w_s)), int(round(c.ref_pt[1])))
            cv2.line(canvas, pt1, pt2, (0, 0, 220), 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 3, (0, 0, 255), -1)
            cv2.circle(canvas, pt2, 3, (0, 0, 255), -1)

    # Draw inliers on top (green)
    for c in display_matches:
        if c.is_inlier:
            pt1 = (int(round(c.src_pt[0])), int(round(c.src_pt[1])))
            pt2 = (int(round(c.ref_pt[0] + w_s)), int(round(c.ref_pt[1])))
            cv2.line(canvas, pt1, pt2, (0, 230, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 3, (0, 255, 0), -1)
            cv2.circle(canvas, pt2, 3, (0, 255, 0), -1)

    return canvas

def render_spatial_coverage_overlay(
    ref_img_8u: np.ndarray,
    inliers: List[Correspondence],
    grid_cols: int = 10,
    grid_rows: int = 10
) -> np.ndarray:
    """
    Render control points with spatial grid overlay and cell density visualization.
    """
    h, w = ref_img_8u.shape[:2]
    canvas = cv2.cvtColor(ref_img_8u, cv2.COLOR_GRAY2BGR) if ref_img_8u.ndim == 2 else ref_img_8u.copy()

    cell_w = w / float(grid_cols)
    cell_h = h / float(grid_rows)

    # Grid count
    counts = np.zeros((grid_rows, grid_cols), dtype=np.int32)
    for c in inliers:
        rx, ry = c.ref_pt
        gx = int(np.clip(rx // cell_w, 0, grid_cols - 1))
        gy = int(np.clip(ry // cell_h, 0, grid_rows - 1))
        counts[gy, gx] += 1

    # Overlay grid cells
    overlay = canvas.copy()
    for r in range(grid_rows):
        for col in range(grid_cols):
            x1, y1 = int(col * cell_w), int(r * cell_h)
            x2, y2 = int((col + 1) * cell_w), int((r + 1) * cell_h)
            cnt = counts[r, col]
            if cnt > 0:
                # Green tint for occupied cell
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 0), -1)
            else:
                # Slight red tint for vacant cell
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 150), -1)

    # Alpha blend overlay
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    # Grid lines
    for col in range(grid_cols + 1):
        x = int(col * cell_w)
        cv2.line(canvas, (x, 0), (x, h), (180, 180, 180), 1)
    for r in range(grid_rows + 1):
        y = int(r * cell_h)
        cv2.line(canvas, (0, y), (w, y), (180, 180, 180), 1)

    # Draw inlier points
    for c in inliers:
        rx, ry = int(round(c.ref_pt[0])), int(round(c.ref_pt[1]))
        cv2.circle(canvas, (rx, ry), 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (rx, ry), 5, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas

def render_checkerboard(
    img_a: np.ndarray,
    img_b: np.ndarray,
    tile_size: int = 40
) -> np.ndarray:
    """
    Generate checkerboard interleaving of two registered images to inspect crater rim alignment.
    """
    h = min(img_a.shape[0], img_b.shape[0])
    w = min(img_a.shape[1], img_b.shape[1])

    a = img_a[:h, :w]
    b = img_b[:h, :w]

    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    if b.ndim == 2:
        b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

    y_indices, x_indices = np.mgrid[0:h, 0:w]
    checker = ((y_indices // tile_size) + (x_indices // tile_size)) % 2 == 0

    out = np.where(checker[:, :, None], a, b)
    return out

def render_difference_map(
    warped_img: np.ndarray,
    ref_img: np.ndarray,
    valid_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Absolute difference map colored with colormap to expose local registration errors.
    """
    h = min(warped_img.shape[0], ref_img.shape[0])
    w = min(warped_img.shape[1], ref_img.shape[1])

    w_sub = warped_img[:h, :w].astype(np.float32)
    r_sub = ref_img[:h, :w].astype(np.float32)

    diff = np.abs(w_sub - r_sub)
    if valid_mask is not None:
        diff *= valid_mask[:h, :w].astype(np.float32)

    diff_norm = np.clip(diff / (np.percentile(diff[diff > 0], 95) + 1e-6) if np.any(diff > 0) else diff, 0.0, 1.0)
    diff_8u = (diff_norm * 255.0).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_8u, cv2.COLORMAP_MAGMA)

    if valid_mask is not None:
        mask_3c = valid_mask[:h, :w, None]
        diff_color = np.where(mask_3c, diff_color, 0)

    return diff_color

def render_subpixel_quiver(
    inliers: List[Correspondence],
    ref_shape: Tuple[int, int],
    scale_factor: float = 10.0
) -> np.ndarray:
    """
    Render vector field quiver plot of sub-pixel displacement vectors (dx, dy).
    """
    h, w = ref_shape[:2]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")

    rx = [c.ref_pt[0] for c in inliers]
    ry = [c.ref_pt[1] for c in inliers]
    dx = [c.subpixel_offset[0] * scale_factor for c in inliers]
    dy = [c.subpixel_offset[1] * scale_factor for c in inliers]

    ax.scatter(rx, ry, color="#00ffcc", s=15, alpha=0.7, label="Control Points")
    ax.quiver(rx, ry, dx, dy, color="#ffcc00", angles="xy", scale_units="xy", scale=1, width=0.005, label=f"Sub-Pixel Offset ({scale_factor}x)")

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Invert Y for image coordinates
    ax.set_title("Sub-Pixel Correction Vectors", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#222222", edgecolor="gray", labelcolor="white")
    fig.tight_layout()

    fig.canvas.draw()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    plot_img = rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    return plot_img[:, :, :3]
