"""
Photometrically Realistic Lunar Terrain and Challenge Benchmark Generator.
Simulates lunar digital elevation models (DEM) with power-law impact crater distributions,
and renders them using Lommel-Seeliger / Hapke lunar reflectance across varying solar geometries.
Generates test benchmark pairs representing:
1. Baseline (low illumination delta)
2. Illumination Challenge (180 deg solar azimuth flip)
3. Scale Challenge (2x scale delta, OHRC 0.25m vs LROC 0.5m)
4. Viewpoint Challenge (Affine shear and rotation)
5. Polar Crater Challenge (Deep shadow and low contrast)
"""
from pathlib import Path
import numpy as np
import cv2
import tifffile
from scipy.ndimage import gaussian_filter

def generate_lunar_dem(height: int = 512, width: int = 512, seed: int = 42) -> np.ndarray:
    """
    Synthesize fractal lunar elevation terrain with superposed impact craters.
    """
    np.random.seed(seed)
    dem = np.zeros((height, width), dtype=np.float32)

    # Multi-scale fractal background (regolith roughness)
    for oct in range(5):
        freq = 2 ** (oct + 2)
        noise = np.random.randn(freq, freq).astype(np.float32)
        upscaled = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        dem += upscaled * (0.5 ** oct) * 15.0

    # Impact craters (power-law size-frequency distribution)
    # Radii from 8 px up to 90 px
    craters = [
        # (x, y, radius, depth)
        (256, 256, 75, 25.0),
        (130, 140, 45, 16.0),
        (380, 120, 50, 18.0),
        (120, 380, 40, 14.0),
        (370, 370, 55, 20.0),
        (200, 340, 25, 9.0),
        (310, 180, 30, 10.0),
        (170, 220, 18, 6.0),
        (330, 290, 22, 7.5),
        (90, 240, 15, 5.0),
        (430, 250, 20, 6.5),
        (260, 110, 16, 5.5),
        (250, 410, 28, 9.5),
    ]

    # Add 40 micro-craters
    for _ in range(40):
        cx = np.random.randint(20, width - 20)
        cy = np.random.randint(20, height - 20)
        cr = np.random.uniform(5.0, 14.0)
        cd = cr * 0.35
        craters.append((cx, cy, cr, cd))

    y_grid, x_grid = np.mgrid[0:height, 0:width]

    for cx, cy, radius, depth in craters:
        dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
        dist = np.sqrt(dist_sq)
        # Parabolic crater interior
        bowl_mask = dist <= radius
        dem[bowl_mask] -= depth * (1.0 - (dist[bowl_mask] / radius)**2)

        # Raised crater rim
        rim_width = radius * 0.45
        rim_mask = (dist > radius * 0.85) & (dist < radius + rim_width)
        rim_height = depth * 0.28
        dem[rim_mask] += rim_height * np.sin(np.pi * (dist[rim_mask] - radius * 0.85) / rim_width)

    return dem

def render_lunar_surface(
    dem: np.ndarray,
    solar_azimuth_deg: float = 45.0,
    solar_elevation_deg: float = 35.0,
    albedo_noise: float = 0.05
) -> np.ndarray:
    """
    Render lunar surface using Lommel-Seeliger scattering law.
    Lommel-Seeliger reflectance: I = cos(i) / (cos(i) + cos(e))
    """
    h, w = dem.shape
    # Surface normals
    gy, gx = np.gradient(dem)
    # Normal vector N = (-gx, -gy, 1) normalized
    norm = np.sqrt(gx**2 + gy**2 + 1.0)
    nx = -gx / norm
    ny = -gy / norm
    nz = 1.0 / norm

    # Solar vector S
    az_rad = np.radians(solar_azimuth_deg)
    el_rad = np.radians(solar_elevation_deg)
    sx = np.cos(el_rad) * np.sin(az_rad)
    sy = np.cos(el_rad) * np.cos(az_rad)
    sz = np.sin(el_rad)

    # Cosine incidence: dot(N, S)
    cos_i = np.maximum(nx * sx + ny * sy + nz * sz, 0.0)
    # Cosine emission: nadir view (0, 0, 1) -> nz
    cos_e = np.maximum(nz, 0.01)

    # Lommel-Seeliger model
    reflectance = cos_i / (cos_i + cos_e + 1e-6)

    # Add regolith albedo variegation
    np.random.seed(int(solar_azimuth_deg))
    albedo = 1.0 + np.random.randn(h, w) * albedo_noise
    img = reflectance * albedo

    # Cast shadow threshold
    img[cos_i <= 0.02] = 0.005

    # Normalize to [0.0, 1.0]
    img_p1, img_p99 = np.percentile(img, (1.0, 99.0))
    if img_p99 > img_p1:
        img_norm = np.clip((img - img_p1) / (img_p99 - img_p1), 0.0, 1.0)
    else:
        img_norm = np.zeros_like(img)

    return (img_norm * 255.0).astype(np.uint8)

def create_all_benchmarks(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    dem = generate_lunar_dem(512, 512, seed=100)

    # 1. Baseline: identical terrain, minor lighting delta
    ref1 = render_lunar_surface(dem, solar_azimuth_deg=45.0, solar_elevation_deg=40.0)
    M_true1 = cv2.getRotationMatrix2D((256, 256), 2.0, 1.0)
    M_true1[0, 2] += 14.25
    M_true1[1, 2] -= 8.75
    src1_raw = render_lunar_surface(dem, solar_azimuth_deg=50.0, solar_elevation_deg=38.0)
    src1 = cv2.warpAffine(src1_raw, M_true1, (512, 512), flags=cv2.INTER_CUBIC)

    cv2.imwrite(str(out_dir / "pair1_baseline_ref.png"), ref1)
    cv2.imwrite(str(out_dir / "pair1_baseline_src.png"), src1)

    # 2. Illumination challenge: 180 deg solar azimuth flip (shadows completely invert!)
    ref2 = render_lunar_surface(dem, solar_azimuth_deg=45.0, solar_elevation_deg=30.0)
    src2_raw = render_lunar_surface(dem, solar_azimuth_deg=225.0, solar_elevation_deg=30.0)
    M_true2 = cv2.getRotationMatrix2D((256, 256), 4.0, 1.0)
    M_true2[0, 2] += 8.0
    M_true2[1, 2] += 12.0
    src2 = cv2.warpAffine(src2_raw, M_true2, (512, 512), flags=cv2.INTER_CUBIC)

    cv2.imwrite(str(out_dir / "pair2_illumination_ref.png"), ref2)
    cv2.imwrite(str(out_dir / "pair2_illumination_src.png"), src2)

    # 3. Scale challenge: 1.35x scale difference (OHRC vs LROC NAC resolution delta)
    ref3 = render_lunar_surface(dem, solar_azimuth_deg=60.0, solar_elevation_deg=45.0)
    M_true3 = cv2.getRotationMatrix2D((256, 256), 5.0, 1.35)
    M_true3[0, 2] -= 20.0
    M_true3[1, 2] += 15.0
    src3 = cv2.warpAffine(ref3, M_true3, (512, 512), flags=cv2.INTER_CUBIC)

    cv2.imwrite(str(out_dir / "pair3_scale_ref.png"), ref3)
    cv2.imwrite(str(out_dir / "pair3_scale_src.png"), src3)

    # 4. Viewpoint challenge: affine shear + rotation
    ref4 = render_lunar_surface(dem, solar_azimuth_deg=55.0, solar_elevation_deg=35.0)
    pts_src_ref = np.float32([[100, 100], [400, 100], [100, 400]])
    pts_dst_ref = np.float32([[115, 90], [420, 125], [85, 415]])
    M_true4 = cv2.getAffineTransform(pts_src_ref, pts_dst_ref)
    src4 = cv2.warpAffine(ref4, M_true4, (512, 512), flags=cv2.INTER_CUBIC)

    cv2.imwrite(str(out_dir / "pair4_viewpoint_ref.png"), ref4)
    cv2.imwrite(str(out_dir / "pair4_viewpoint_src.png"), src4)

    # 5. Polar crater challenge: deep permanent shadow, low incidence angle (12 deg)
    dem_polar = generate_lunar_dem(512, 512, seed=999)
    ref5 = render_lunar_surface(dem_polar, solar_azimuth_deg=30.0, solar_elevation_deg=14.0)
    src5_raw = render_lunar_surface(dem_polar, solar_azimuth_deg=75.0, solar_elevation_deg=12.0)
    M_true5 = cv2.getRotationMatrix2D((256, 256), 3.0, 1.0)
    M_true5[0, 2] += 10.0
    M_true5[1, 2] -= 6.0
    src5 = cv2.warpAffine(src5_raw, M_true5, (512, 512), flags=cv2.INTER_CUBIC)

    cv2.imwrite(str(out_dir / "pair5_polar_shadow_ref.png"), ref5)
    cv2.imwrite(str(out_dir / "pair5_polar_shadow_src.png"), src5)

    print(f"Successfully generated all 5 challenge benchmark pairs in {out_dir}")

if __name__ == "__main__":
    out = Path(r"C:\Users\ASUS\.gemini\antigravity\scratch\lunar_registration\data\samples")
    create_all_benchmarks(out)
