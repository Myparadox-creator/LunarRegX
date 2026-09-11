"""
ISRO PRADAN & PDS4 Product Ingestion and Optimization Tool.
Unpacks large multi-hundred-megabyte Chandrayaan-2 ZIP archives, extracts 12/16-bit
calibrated GeoTIFF/IMG arrays and XML labels, and generates optimized registration crops.
"""
from pathlib import Path
import argparse
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import tifffile
from PIL import Image

def detect_sensor_from_name(name: str) -> str:
    name_lower = name.lower()
    if "ohr" in name_lower:
        return "OHRC"
    elif "tmc" in name_lower:
        return "TMC2"
    elif "iir" in name_lower:
        return "IIRS"
    elif "nac" in name_lower or "lroc" in name_lower:
        return "LROC_NAC"
    return "UNKNOWN_LUNAR"

def parse_pds4_xml(xml_path: Path) -> dict:
    meta = {}
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
    except Exception as e:
        print(f"[!] Warning: Could not parse XML metadata: {e}")
    return meta

def ingest_product(
    input_path: str | Path,
    target_crop_size: int = 1024,
    center_crop: bool = True
):
    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    project_root = Path(__file__).resolve().parent.parent
    extract_dir = project_root / "data" / "raw" / "extracted_pradan"
    extract_dir.mkdir(parents=True, exist_ok=True)

    xml_file = None
    raster_file = None

    if inp.suffix.lower() == ".zip":
        print(f"[*] Unpacking archive ({inp.stat().st_size / (1024*1024):.1f} MB): {inp.name}...")
        with zipfile.ZipFile(inp, "r") as z:
            z.extractall(extract_dir)

        for f in extract_dir.rglob("*"):
            if f.suffix.lower() == ".xml" and not xml_file:
                xml_file = f
            elif f.suffix.lower() in [".tif", ".tiff", ".img", ".png"] and not f.name.endswith("_b.png") and not raster_file:
                raster_file = f
    else:
        raster_file = inp
        possible_xml = inp.with_suffix(".xml")
        if possible_xml.exists():
            xml_file = possible_xml

    if not raster_file or not raster_file.exists():
        raise FileNotFoundError(f"No valid image raster (.tif/.img/.png) found in {inp}")

    sensor_type = detect_sensor_from_name(inp.name)
    if sensor_type == "UNKNOWN_LUNAR" and xml_file:
        sensor_type = detect_sensor_from_name(xml_file.name)

    print(f"[*] Detected Instrument: {sensor_type}")
    print(f"[*] Reading satellite raster: {raster_file.name}...")

    try:
        arr = tifffile.imread(str(raster_file))
    except Exception:
        arr = cv2.imread(str(raster_file), cv2.IMREAD_UNCHANGED)

    if arr is None:
        raise ValueError(f"Failed to read image array from {raster_file}")

    if arr.ndim == 3:
        arr = arr[:, :, 0]

    h, w = arr.shape
    print(f"[*] Native Dimensions: {w} x {h} (Bit depth: {arr.dtype})")

    dest_sensor_dir = project_root / "data" / "raw" / "ch2" / sensor_type.lower()
    dest_sensor_dir.mkdir(parents=True, exist_ok=True)

    arr_f = arr.astype(np.float32)
    p1, p99 = np.percentile(arr_f[arr_f > 0], (1.0, 99.0)) if np.any(arr_f > 0) else (0, 255)
    norm = np.clip((arr_f - p1) / max(1.0, p99 - p1), 0.0, 1.0)
    arr_8u = (norm * 255.0).astype(np.uint8)

    if min(h, w) > target_crop_size:
        if center_crop:
            cy, cx = h // 2, w // 2
            half = target_crop_size // 2
            crop_8u = arr_8u[cy - half : cy + half, cx - half : cx + half]
        else:
            crop_8u = arr_8u[:target_crop_size, :target_crop_size]
    else:
        crop_8u = arr_8u

    crop_dest = dest_sensor_dir / f"{raster_file.stem}_crop_{target_crop_size}.png"
    cv2.imwrite(str(crop_dest), crop_8u)
    print(f"[+] Saved optimized sub-scene crop to: {crop_dest}")

    sample_dest = project_root / "data" / "samples" / f"real_{sensor_type.lower()}_{raster_file.stem[:12]}.png"
    cv2.imwrite(str(sample_dest), crop_8u)
    print(f"[+] Added to benchmark samples for UI demo: {sample_dest}")

    metadata = {}
    if xml_file and xml_file.exists():
        metadata = parse_pds4_xml(xml_file)
        print(f"[*] Parsed Solar Geometry: Azimuth {metadata.get('solar_azimuth_deg')}°, Elevation {metadata.get('solar_elevation_deg')}°")

    print(f"\n[SUCCESS] Successfully ingested {sensor_type} product!")
    print(f"-> Ready for registration in the Web UI or CLI.")
    return crop_dest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest, extract, and tile large ISRO PRADAN Chandrayaan-2 products")
    parser.add_argument("--file", type=str, required=True, help="Path to downloaded .zip or .tif file")
    parser.add_argument("--crop_size", type=int, default=1024, help="Target square crop size for registration (default 1024)")
    args = parser.parse_args()
    ingest_product(args.file, target_crop_size=args.crop_size)
