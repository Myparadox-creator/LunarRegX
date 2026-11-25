"""
Lunar Image Ingestion and Export Layer.
Supports GeoTIFF, TIFF, PNG, JPEG, and NPY with multi-bit-depth preservation.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Set
from pathlib import Path
import re
import xml.etree.ElementTree as ET
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
    gsd_override: Optional[float] = None,
    xml_path: Optional[str | Path] = None,
    require_xml: bool = False
) -> LunarImage:
    """
    Load lunar image from disk, preserving bit-depth, nodata masks, and metadata.

    Parameters:
        xml_path: Optional explicit path to a companion PDS4 / ISRO SAC XML label.
        require_xml: If True, raises ValueError unless a valid corresponding XML label
            is present (at ``xml_path`` or alongside ``filepath``) and matches the image.
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

    # 1. Resolve and validate PDS4 / SAC XML metadata file
    xml_meta: Dict[str, Any] = {}
    if xml_path is not None:
        xml_meta = validate_lunar_xml(xml_path, image_filename=path.name)
    elif require_xml:
        companion_xml = _find_companion_xml_path(path)
        if companion_xml is None:
            raise ValueError(
                f"Missing corresponding XML metadata file for '{path.name}'. "
                "Please upload or provide its PDS4 / SAC XML file."
            )
        xml_meta = validate_lunar_xml(companion_xml, image_filename=path.name)
    else:
        companion_xml = _find_companion_xml_path(path)
        if companion_xml is not None:
            try:
                xml_meta = parse_lunar_xml(companion_xml)
            except Exception:
                xml_meta = {}

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
    geotiff_gsd: Optional[float] = None
    if suffix in [".tif", ".tiff", ".geotiff"]:
        try:
            import rasterio
            with rasterio.open(str(path)) as r_src:
                meta.extra_attributes["bounds_proj"] = (r_src.bounds.left, r_src.bounds.bottom, r_src.bounds.right, r_src.bounds.top)
                if r_src.crs:
                    meta.extra_attributes["crs_wkt"] = r_src.crs.to_wkt()
                    meta.extra_attributes["crs"] = str(r_src.crs)
                    if r_src.res and len(r_src.res) == 2:
                        calc_gsd = float(abs(r_src.res[0]))
                        if calc_gsd > 0.0001 and not (calc_gsd == 1.0 and r_src.transform.is_identity):
                            geotiff_gsd = calc_gsd
                elif r_src.res and len(r_src.res) == 2 and gsd_override is None and not r_src.transform.is_identity:
                    calc_gsd = float(abs(r_src.res[0]))
                    if calc_gsd > 0.0001:
                        geotiff_gsd = calc_gsd
        except Exception:
            pass

    # 3. Resolve physical GSD with authentic hierarchy: GeoTIFF CRS res > bounds_geo span > XML resolution > override
    b_geo = meta.extra_attributes.get("bounds_geo")
    if geotiff_gsd is not None:
        meta.gsd = geotiff_gsd
    elif b_geo and raw.shape[0] > 0:
        min_lon, min_lat, max_lon, max_lat = b_geo
        lat_span_deg = abs(max_lat - min_lat)
        # Moon mean radius R = 1,737,400 m -> 30,323.0 meters per degree of latitude
        physical_h_m = lat_span_deg * 30323.0
        if physical_h_m > 0:
            calc_gsd = round(float(physical_h_m / raw.shape[0]), 4)
            meta.extra_attributes["calculated_gsd_m"] = calc_gsd
            meta.gsd = calc_gsd
    elif "gsd" in xml_meta and xml_meta["gsd"] > 0:
        meta.gsd = float(xml_meta["gsd"])
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

def _normalize_sensor_token(text: Optional[str]) -> Optional[str]:
    """Detect canonical lunar sensor code from filename or XML text."""
    if not text:
        return None
    low = text.lower()
    if "ohr" in low:
        return "OHRC"
    if "tmc" in low:
        return "TMC2"
    if "iir" in low:
        return "IIRS"
    if "lroc" in low or "_nac" in low or low.startswith("nac"):
        return "LROC_NAC"
    if "_wac" in low or low.startswith("wac"):
        return "LROC_WAC"
    if "selene" in low or "kaguya" in low:
        return "SELENE"
    return None

def _extract_product_signatures(text: Optional[str]) -> Dict[str, Set[str]]:
    """
    Extract normalized mission product IDs by naming family from filenames or XML tags:
      - ``pradan``: e.g. ``ch2_ohr_ncp_20190906t2241285714``
      - ``sac``: e.g. ``ohrxxd18cho2359602nnnn24342131250969``
    """
    sigs: Dict[str, Set[str]] = {"pradan": set(), "sac": set()}
    if not text:
        return sigs
    low = text.lower()
    for m in re.findall(r"ch2_[a-z]{3}_[a-z0-9]+_\d{8}t\d{4,10}", low):
        sigs["pradan"].add(m)
    for m in re.findall(r"(?:ohr|tmc|iir)[a-z0-9]{2}[a-z0-9]*\d{14,}", low):
        sigs["sac"].add(m)
    return sigs

def _find_companion_xml_path(img_path: Path) -> Optional[Path]:
    """Locate companion XML metadata file alongside an image."""
    candidates = [
        img_path.with_suffix(".xml"),
        img_path.with_suffix(".XML"),
        img_path.parent / (img_path.stem + ".xml"),
        img_path.parent / (img_path.stem.lower() + ".xml")
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def parse_lunar_xml(xml_path: str | Path) -> Dict[str, Any]:
    """
    Parse PDS4 (<Product_Observational>) or ISRO SAC (<product>) lunar XML metadata label.
    Raises ValueError on malformed or empty XML files.
    """
    xp = Path(xml_path)
    if not xp.exists():
        raise FileNotFoundError(f"XML file not found: {xp}")
    if xp.stat().st_size == 0:
        raise ValueError(f"XML file '{xp.name}' is empty (0 bytes).")

    try:
        tree = ET.parse(str(xp))
    except ET.ParseError as e:
        raise ValueError(f"Malformed XML syntax in '{xp.name}': {e}") from e

    root = tree.getroot()
    root_local = root.tag.split("}", 1)[1] if "}" in root.tag else root.tag

    tags: Dict[str, str] = {}
    all_local_tags: Set[str] = set()
    for elem in root.iter():
        local_tag = (elem.tag.split("}", 1)[1] if "}" in elem.tag else elem.tag).lower()
        all_local_tags.add(local_tag)
        if elem.text and elem.text.strip():
            # Keep first occurrence if duplicate blocks exist (e.g. refined/system tags handled separately)
            if local_tag not in tags:
                tags[local_tag] = elem.text.strip()

    recognized_markers = {
        "logical_identifier", "job_id", "level0_dataset", "solar_azimuth_angle",
        "sun_azimuth_angle", "sun_azimuth", "solar_azimuth", "sun_azimuth_in_degree",
        "solar_elevation_angle", "sun_elevation_angle", "sun_elevation", "solar_elevation",
        "sun_elevation_in_degree", "pixel_resolution", "spatial_resolution", "resolution_in_meter",
        "upper_left_latitude", "topleft_latitude", "westernmost_longitude", "west_bounding_coordinate"
    }
    has_recognized_schema = (
        root_local.lower() in {"product_observational", "product"}
        or bool(all_local_tags & recognized_markers)
    )
    if not has_recognized_schema:
        raise ValueError(
            f"XML file '{xp.name}' is not a valid PDS4 or ISRO SAC lunar metadata label "
            "(missing Product_Observational/product schema and telemetry tags)."
        )

    xml_format = "PDS4" if root_local.lower() == "product_observational" else (
        "SAC_PRODUCT" if root_local.lower() == "product" else "LUNAR_XML"
    )
    meta: Dict[str, Any] = {
        "xml_loaded": True,
        "xml_valid": True,
        "xml_format": xml_format,
        "xml_path": str(xp),
        "xml_filename": xp.name,
    }

    # Parse solar angles (PDS4 & SAC formats)
    for az_k in ["solar_azimuth_angle", "sun_azimuth_angle", "sun_azimuth", "solar_azimuth", "sun_azimuth_in_degree"]:
        if az_k in tags:
            try:
                meta["solar_azimuth_deg"] = float(tags[az_k])
                break
            except ValueError:
                pass

    for el_k in ["solar_elevation_angle", "sun_elevation_angle", "sun_elevation", "solar_elevation", "sun_elevation_in_degree"]:
        if el_k in tags:
            try:
                meta["solar_elevation_deg"] = float(tags[el_k])
                break
            except ValueError:
                pass

    for inc_k in ["incidence_angle", "incidence", "solar_incidence", "solar_incidence_angle_in_degree"]:
        if inc_k in tags:
            try:
                meta["incidence_angle_deg"] = float(tags[inc_k])
                break
            except ValueError:
                pass

    # Standard bounding coordinates
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
        # Check PDS4 corner coordinates and SAC corner coordinates
        corner_sets = [
            (
                ["upper_left_latitude", "upper_right_latitude", "lower_left_latitude", "lower_right_latitude"],
                ["upper_left_longitude", "upper_right_longitude", "lower_left_longitude", "lower_right_longitude"],
            ),
            (
                ["topleft_latitude_refined", "topright_latitude_refined", "bottomleft_latitude_refined", "bottomright_latitude_refined"],
                ["topleft_longitude_refined", "topright_longitude_refined", "bottomleft_longitude_refined", "bottomright_longitude_refined"],
            ),
            (
                ["topleft_latitude", "topright_latitude", "bottomleft_latitude", "bottomright_latitude"],
                ["topleft_longitude", "topright_longitude", "bottomleft_longitude", "bottomright_longitude"],
            ),
        ]
        for lat_tags, lon_tags in corner_sets:
            corner_lats = []
            corner_lons = []
            for t in lat_tags:
                if t in tags:
                    try:
                        corner_lats.append(float(tags[t]))
                    except ValueError:
                        pass
            for t in lon_tags:
                if t in tags:
                    try:
                        corner_lons.append(float(tags[t]))
                    except ValueError:
                        pass
            if len(corner_lats) >= 2 and len(corner_lons) >= 2:
                min_lon = min(corner_lons)
                max_lon = max(corner_lons)
                min_lat = min(corner_lats)
                max_lat = max(corner_lats)
                # Avoid creating broken lat/lon boxes when a polar/antimeridian strip spans wide longitudes
                is_polar_wrap = (max_lon - min_lon) >= 60.0
                if not is_polar_wrap:
                    meta["bounds_geo"] = (min_lon, min_lat, max_lon, max_lat)
                    meta["center_longitude"] = (min_lon + max_lon) / 2.0
                    meta["center_latitude"] = (min_lat + max_lat) / 2.0
                break

    # Pixel resolution / GSD detection
    for res_k in ["pixel_resolution", "spatial_resolution", "resolution_in_meter", "detector_pixel_width"]:
        if res_k in tags:
            try:
                parsed_res = float(tags[res_k])
                if parsed_res > 0:
                    meta["gsd"] = parsed_res
                    break
            except ValueError:
                pass

    # Product identifier & sensor detection from XML metadata
    product_id = tags.get("job_id") or tags.get("level0_dataset") or tags.get("logical_identifier") or tags.get("file_name")
    if product_id:
        meta["product_id"] = product_id

    for attr_name in ["projection", "area"]:
        if attr_name in tags:
            meta[attr_name] = tags[attr_name]

    detected_sensor = None
    for cand_text in [tags.get("job_id"), tags.get("logical_identifier"), tags.get("file_name"), tags.get("title"), xp.name]:
        detected_sensor = _normalize_sensor_token(cand_text)
        if detected_sensor:
            break
    if detected_sensor:
        meta["sensor_name"] = detected_sensor

    # Collect product signatures present inside XML for correspondence verification
    xml_sigs: Dict[str, Set[str]] = {"pradan": set(), "sac": set()}
    for cand_text in [tags.get("logical_identifier"), tags.get("job_id"), tags.get("level0_dataset"), tags.get("file_name"), xp.name]:
        sub_sigs = _extract_product_signatures(cand_text)
        xml_sigs["pradan"].update(sub_sigs["pradan"])
        xml_sigs["sac"].update(sub_sigs["sac"])
    meta["product_signatures"] = {k: sorted(v) for k, v in xml_sigs.items()}

    return meta

def validate_lunar_xml(xml_path: str | Path, image_filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate that an XML file is a well-formed PDS4/SAC lunar metadata label
    and verify that it does not conflict with the paired raster image filename.
    """
    meta = parse_lunar_xml(xml_path)
    if not image_filename:
        return meta

    xp_name = Path(xml_path).name
    img_sensor = _normalize_sensor_token(image_filename)
    xml_sensor = meta.get("sensor_name")
    if img_sensor and xml_sensor and img_sensor != xml_sensor:
        raise ValueError(
            f"XML correspondence mismatch: image '{image_filename}' identifies sensor {img_sensor}, "
            f"but XML '{xp_name}' identifies sensor {xml_sensor}. Please upload the matching XML file."
        )

    img_sigs = _extract_product_signatures(image_filename)
    xml_sigs = meta.get("product_signatures", {})
    for family in ("pradan", "sac"):
        img_ids = img_sigs.get(family, set())
        xml_ids = set(xml_sigs.get(family, []))
        if img_ids and xml_ids and not (img_ids & xml_ids):
            raise ValueError(
                f"XML correspondence mismatch: image '{image_filename}' ({sorted(img_ids)[0]}) "
                f"does not match XML '{xp_name}' ({sorted(xml_ids)[0]}). Please upload the corresponding XML file."
            )

    return meta

def _find_and_parse_companion_xml(img_path: Path) -> Dict[str, Any]:
    """
    Backward-compatible helper: locate companion PDS4 / SAC XML label in the same
    directory and parse solar/spatial parameters.
    """
    xml_path = _find_companion_xml_path(img_path)
    if not xml_path:
        return {}
    try:
        return parse_lunar_xml(xml_path)
    except Exception:
        return {}

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
