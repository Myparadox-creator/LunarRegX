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

    # 3. Calculate authentic physical GSD from selenographic bounds if available
    b_geo = meta.extra_attributes.get("bounds_geo")
    if b_geo and raw.shape[0] > 0:
        min_lon, min_lat, max_lon, max_lat = b_geo
        lat_span_deg = abs(max_lat - min_lat)
        # Moon mean radius R = 1,737,400 m -> 30,323.0 meters per degree of latitude
        physical_h_m = lat_span_deg * 30323.0
        if physical_h_m > 0:
            calc_gsd = round(float(physical_h_m / raw.shape[0]), 4)
            meta.extra_attributes["calculated_gsd_m"] = calc_gsd
            meta.gsd = calc_gsd
    elif gsd_override is not None:
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
        tags: Dict[str, str] = {}
        for elem in root.iter():
            local_tag = elem.tag.split("}", 1)[1] if "}" in elem.tag else elem.tag
            if elem.text and elem.text.strip():
                tags[local_tag.lower()] = elem.text.strip()

        # Parse solar angles
        for az_k in ["solar_azimuth_angle", "sun_azimuth_angle", "sun_azimuth", "solar_azimuth"]:
            if az_k in tags:
                try:
                    meta["solar_azimuth_deg"] = float(tags[az_k])
                    break
                except ValueError:
                    pass

        for el_k in ["solar_elevation_angle", "sun_elevation_angle", "sun_elevation", "solar_elevation"]:
            if el_k in tags:
                try:
                    meta["solar_elevation_deg"] = float(tags[el_k])
                    break
                except ValueError:
                    pass

        for inc_k in ["incidence_angle", "incidence"]:
            if inc_k in tags:
                try:
                    meta["incidence_angle_deg"] = float(tags[inc_k])
                    break
                except ValueError:
                    pass

        # Check for standard bounding coordinates
        w_lon = tags.get("westernmost_longitude") or tags.get("west_bounding_coordinate")
        e_lon = tags.get("easternmost_longitude") or tags.get("east_bounding_coordinate")
        n_lat = tags.get("northernmost_latitude") or tags.get("north_bounding_coordinate")
        s_lat = tags.get("southernmost_latitude") or tags.get("south_bounding_coordinate")

        if all(x is not None for x in [w_lon, e_lon, n_lat, s_lat]):
            try:
                min_lon = float(w_lon)
                max_lon = float(e_lon)
                min_lat = float(s_lat)
                max_lat = float(n_lat)
                meta["bounds_geo"] = (min_lon, min_lat, max_lon, max_lat)
                meta["center_longitude"] = (min_lon + max_lon) / 2.0
                meta["center_latitude"] = (min_lat + max_lat) / 2.0
            except ValueError:
                pass
        else:
            # Check for Chandrayaan-2 PRADAN corner coordinates
            corner_lats = []
            corner_lons = []
            for tag in ["upper_left_latitude", "upper_right_latitude", "lower_left_latitude", "lower_right_latitude"]:
                if tag in tags:
                    try:
                        corner_lats.append(float(tags[tag]))
                    except ValueError:
                        pass
            for tag in ["upper_left_longitude", "upper_right_longitude", "lower_left_longitude", "lower_right_longitude"]:
                if tag in tags:
                    try:
                        corner_lons.append(float(tags[tag]))
                    except ValueError:
                        pass

            if len(corner_lats) >= 2 and len(corner_lons) >= 2:
                min_lon = min(corner_lons)
                max_lon = max(corner_lons)
                min_lat = min(corner_lats)
                max_lat = max(corner_lats)
                meta["bounds_geo"] = (min_lon, min_lat, max_lon, max_lat)
                meta["center_longitude"] = (min_lon + max_lon) / 2.0
                meta["center_latitude"] = (min_lat + max_lat) / 2.0

        # Pixel resolution / GSD detection
        for res_k in ["pixel_resolution", "spatial_resolution", "detector_pixel_width"]:
            if res_k in tags:
                try:
                    parsed_res = float(tags[res_k])
                    if parsed_res > 0:
                        meta["gsd"] = parsed_res
                        break
                except ValueError:
                    pass

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
