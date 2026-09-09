"""
Lunar Image Ingestion and Export Layer.
Supports GeoTIFF, TIFF, PNG, JPEG, and NPY with multi-bit-depth preservation.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import cv2
import tifffile
from PIL import Image

from src.io.metadata import SensorMetadata

@dataclass
class LunarImage:
    raw_array: np.ndarray          # Original unscaled data (e.g. 12/16-bit)
    normalized: np.ndarray         # Float32 normalized in [0.0, 1.0]
    display_8bit: np.ndarray       # Uint8 [0, 255] for visualization & CV2
    metadata: SensorMetadata
    mask: Optional[np.ndarray] = None  # Valid data mask (True = valid, False = nodata)
    filepath: Optional[str] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.raw_array.shape

    @property
    def height(self) -> int:
        return self.raw_array.shape[0]

    @property
    def width(self) -> int:
        return self.raw_array.shape[1]

def load_lunar_image(
    filepath: str | Path,
    metadata_path: Optional[str | Path] = None,
    sensor_name: Optional[str] = None,
    gsd_override: Optional[float] = None
) -> LunarImage:
    """
    Load lunar image from disk, preserving bit-depth, nodata masks, and metadata.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Image file is empty (0 bytes): {path}")

    suffix = path.suffix.lower()
    raw = None

    if suffix in [".tif", ".tiff", ".geotiff"]:
        try:
            raw = tifffile.imread(str(path))
        except Exception:
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    elif suffix == ".npy":
        raw = np.load(str(path))
    else:
        raw_read = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw_read is None:
            try:
                pil_img = Image.open(str(path))
                raw = np.array(pil_img)
            except Exception as e:
                raise ValueError(f"Failed to decode image from {path}: {str(e)}")
        else:
            raw = raw_read

    if raw is None or raw.size == 0:
        raise ValueError(f"Failed to decode image from {path} (empty or corrupt array)")

    if raw.ndim < 2:
        raise ValueError(f"Image has insufficient dimensions {raw.shape} (minimum 2D required)")

    if raw.shape[0] < 8 or raw.shape[1] < 8:
        raise ValueError(f"Image resolution {raw.shape[1]}x{raw.shape[0]} is too small (minimum 8x8 required)")

    if raw.ndim == 3:
        if raw.shape[2] == 4:
            raw = raw[:, :, :3]
        if raw.shape[2] == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY if raw.dtype == np.uint8 else cv2.COLOR_RGB2GRAY)
        elif raw.shape[2] == 1:
            raw = raw.squeeze(2)
        else:
            raw = raw[:, :, 0]

    mask = np.ones(raw.shape, dtype=bool)
    if np.isnan(raw).any():
        mask &= ~np.isnan(raw)
        raw = np.nan_to_num(raw, nan=0.0)

    dtype = raw.dtype
    raw_float = raw.astype(np.float32)
    valid_pixels = raw_float[mask] if mask.any() else raw_float

    if valid_pixels.size > 0:
        p1, p99 = np.percentile(valid_pixels, (1.0, 99.0))
        if p99 > p1:
            norm = np.clip((raw_float - p1) / (p99 - p1), 0.0, 1.0)
        else:
            norm = np.zeros_like(raw_float)
    else:
        norm = np.zeros_like(raw_float)

    display_8bit = (norm * 255.0).astype(np.uint8)

    meta = None
    if metadata_path and Path(metadata_path).exists():
        meta = SensorMetadata.from_yaml(Path(metadata_path))
    else:
        bit_depth = 16 if dtype in [np.uint16, np.int16] else (32 if dtype in [np.float32, np.float64] else 8)
        meta = SensorMetadata(
            sensor_name=sensor_name or "UNKNOWN_LUNAR",
            bit_depth=bit_depth,
            gsd=gsd_override or 1.0
        )

    if gsd_override is not None:
        meta.gsd = gsd_override

    return LunarImage(
        raw_array=raw,
        normalized=norm,
        display_8bit=display_8bit,
        metadata=meta,
        mask=mask,
        filepath=str(path)
    )

def save_lunar_image(
    filepath: str | Path,
    image_array: np.ndarray,
    metadata: Optional[SensorMetadata] = None
) -> None:
    """
    Save registered lunar image as TIFF or PNG.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix in [".tif", ".tiff", ".geotiff"]:
        if image_array.dtype in [np.float32, np.float64]:
            tifffile.imwrite(str(path), image_array.astype(np.float32))
        else:
            tifffile.imwrite(str(path), image_array)
    else:
        if image_array.dtype in [np.float32, np.float64]:
            out_8u = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(path), out_8u)
        else:
            cv2.imwrite(str(path), image_array)
