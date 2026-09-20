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

    # 1. Check for companion PDS4 XML metadata file
    xml_meta = _find_and_parse_companion_xml(path)
    if xml_meta:
        if "solar_azimuth_deg" in xml_meta:
            meta.solar_azimuth_deg = xml_meta["solar_azimuth_deg"]
        if "solar_elevation_deg" in xml_meta:
            meta.solar_elevation_deg = xml_meta["solar_elevation_deg"]
        if "incidence_angle_deg" in xml_meta:
            meta.incidence_angle_deg = xml_meta["incidence_angle_deg"]
        if "sensor_name" in xml_meta and (not sensor_name or sensor_name == "UNKNOWN_LUNAR"):
            meta.sensor_name = xml_meta["sensor_name"]
        for k, v in xml_meta.items():
            meta.extra_attributes[k] = v

    # 2. Check for embedded GeoTIFF metadata via rasterio
    if suffix in [".tif", ".tiff", ".geotiff"]:
        try:
            import rasterio
            with rasterio.open(str(path)) as r_src:
                meta.extra_attributes["bounds_proj"] = (r_src.bounds.left, r_src.bounds.bottom, r_src.bounds.right, r_src.bounds.top)
                if r_src.crs:
                    meta.extra_attributes["crs_wkt"] = r_src.crs.to_wkt()
                    meta.extra_attributes["crs"] = str(r_src.crs)
                if r_src.res and len(r_src.res) == 2 and gsd_override is None:
                    # GSD from raster resolution
                    calc_gsd = float(abs(r_src.res[0]))
                    if calc_gsd > 0.0001:
                        meta.gsd = calc_gsd
        except Exception:
            pass

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

def _find_and_parse_companion_xml(img_path: Path) -> Dict[str, Any]:
    """
    Search for companion PDS4 XML metadata label in the same directory and parse solar/spatial parameters.
    """
    import xml.etree.ElementTree as ET
    meta = {}
    
    candidates = [
        img_path.with_suffix(".xml"),
        img_path.with_suffix(".XML"),
        img_path.parent / (img_path.stem + ".xml"),
        img_path.parent / (img_path.stem.lower() + ".xml")
    ]
    xml_path = None
    for c in candidates:
        if c.exists():
            xml_path = c
            break
            
    if not xml_path:
        return meta

    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        for elem in root.iter():
            if "}" in elem.tag:
                elem.tag = elem.tag.split("}", 1)[1]

        az = root.find(".//solar_azimuth_angle") or root.find(".//sun_azimuth_angle")
        el = root.find(".//solar_elevation_angle") or root.find(".//sun_elevation_angle")
        inc = root.find(".//incidence_angle")

        if az is not None and az.text:
            meta["solar_azimuth_deg"] = float(az.text)
        if el is not None and el.text:
            meta["solar_elevation_deg"] = float(el.text)
        if inc is not None and inc.text:
            meta["incidence_angle_deg"] = float(inc.text)

        w_lon = root.find(".//westernmost_longitude") or root.find(".//west_bounding_coordinate")
        e_lon = root.find(".//easternmost_longitude") or root.find(".//east_bounding_coordinate")
        n_lat = root.find(".//northernmost_latitude") or root.find(".//north_bounding_coordinate")
        s_lat = root.find(".//southernmost_latitude") or root.find(".//south_bounding_coordinate")

        if all(x is not None and x.text for x in [w_lon, e_lon, n_lat, s_lat]):
            min_lon = float(w_lon.text)
            max_lon = float(e_lon.text)
            min_lat = float(s_lat.text)
            max_lat = float(n_lat.text)
            meta["bounds_geo"] = (min_lon, min_lat, max_lon, max_lat)
            meta["center_longitude"] = (min_lon + max_lon) / 2.0
            meta["center_latitude"] = (min_lat + max_lat) / 2.0

        # Sensor detection from filename
        fname = xml_path.name.lower()
        if "ohr" in fname:
            meta["sensor_name"] = "OHRC"
        elif "tmc" in fname:
            meta["sensor_name"] = "TMC2"
        elif "iir" in fname:
            meta["sensor_name"] = "IIRS"
        elif "lroc" in fname or "nac" in fname:
            meta["sensor_name"] = "LROC_NAC"
    except Exception:
        pass
        
    return meta

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
