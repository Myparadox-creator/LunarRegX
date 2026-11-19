# 🌕 LunarRegX Dataset Transparency & Ingestion Guide
**Smart India Hackathon (SIH 2026) | Problem ID: 26166**

This document provides a transparent, scientifically honest account of all lunar imagery and telemetry in LunarRegX, detailing the origin of synthetic stress-test pairs, authentic Chandrayaan-2 flight data, and instructions for ingesting new ISRO PRADAN products.

---

## 🔬 1. Dataset Classification & Transparency Statement

LunarRegX explicitly partitions all imagery into two categories:

| Category | Storage Path | Description | Tracking Status |
| :--- | :--- | :--- | :--- |
| **Synthetic Challenge Benchmarks** | `data/samples/pair[1-5]_*.png` | Photometrically simulated lunar DEMs with controlled physical stress factors | Tracked in Git |
| **Authentic Flight Imagery & Labels** | `data/raw/ch2/` & `data/samples/real_ch2_*` | Calibrated ISRO Chandrayaan-2 (OHRC, TMC-2, IIRS) PDS4 XML metadata and flight swaths | XMLs tracked; large PNGs (>8 MB) gitignored |

> [!IMPORTANT]
> **Data Honesty:** Synthetic benchmark pairs are strictly labeled as synthetic benchmarks. They are never misrepresented as real flight captures. They serve to provide mathematically verifiable ground-truth geometry for rigorous empirical benchmarking.

---

## 🧪 2. Synthetic Challenge Benchmarks (`data/samples/pair1` to `pair5`)

The 5 benchmark scenario pairs were generated using `scripts/generate_lunar_benchmarks.py`. The simulation pipeline models lunar surface topography and illumination physics:
1. **Fractal Elevation Model:** Synthesizes multi-scale lunar topography using 5 octaves of Perlin/Gaussian fractal noise representing regolith micro-roughness.
2. **Impact Crater Distribution:** Injects impact crater morphology adhering to the lunar power-law size-frequency distribution ($N(D) \propto D^{-2}$), ranging from large impact basins (radius $15\text{--}90\text{ px}$) to micro-craters ($5\text{--}14\text{ px}$) with parabolic floors and raised rims.
3. **Photometric Shading:** Renders digital elevation surfaces using Lommel-Seeliger / Hapke lunar reflectance:
   $$I \propto \frac{\cos(i)}{\cos(i) + \cos(e)}$$
   where $i$ is the solar incidence angle and $e$ is the emission angle, capturing the non-Lambertian backscattering characteristic of lunar regolith.

### Benchmark Scenarios:
* **Pair 1: Baseline Control (`pair1_baseline_{src,ref}.png`)**
  - *Conditions:* Consistent solar illumination (azimuth 135°, elevation 30°).
  - *Purpose:* Validates nominal sub-pixel registration accuracy under standard conditions.
* **Pair 2: 180° Shadow Reversal (`pair2_illumination_{src,ref}.png`)**
  - *Conditions:* Exact 180° solar azimuth flip (Source: 315°, Reference: 135°).
  - *Purpose:* Validates phase-congruency invariance against crater shadow inversion, causing classical gradient methods (SIFT/AKAZE) to fail.
* **Pair 3: Multi-Scale Challenge (`pair3_scale_{src,ref}.png`)**
  - *Conditions:* 2× Ground Sample Distance (GSD) scale disparity (0.25 m/px vs 0.50 m/px).
  - *Purpose:* Tests multi-resolution pyramid and scale-adaptive matching across different orbiter altitudes.
* **Pair 4: Viewpoint Shear (`pair4_viewpoint_{src,ref}.png`)**
  - *Conditions:* Oblique camera viewing angle (affine shear matrix with 15° effective tilt).
  - *Purpose:* Tests affine transformation stability and projective distortion tolerance.
* **Pair 5: Polar Crater Terrain (`pair5_polar_shadow_{src,ref}.png`)**
  - *Conditions:* Low solar elevation (3°), deep permanent shadow, high dynamic range with clipped dark floors.
  - *Purpose:* Tests structural matching in South Pole permanently shadowed regions (PSR).

To regenerate or customize the synthetic benchmark dataset:
```bash
python scripts/generate_lunar_benchmarks.py --output_dir data/samples
```

---

## 🛰️ 3. Authentic Chandrayaan-2 Flight Data

The repository includes authentic flight metadata, PDS4 XML labels, and calibrated flight browse swaths from the ISRO Chandrayaan-2 mission:

### Included Flight Products:
* **Chandrayaan-2 OHRC (Orbiter High Resolution Camera - 0.25 m/px):**
  - `data/raw/ch2/ohrc/ch2_ohr_ncp_20190906T2241285714_d_img_gds.xml`: Authentic PDS4 flight metadata for the Sept 6, 2019 pass over the Chandrayaan-2 Vikram landing site ($70.5^\circ\text{S to } 71.4^\circ\text{S}$, $22.5^\circ\text{E to } 22.9^\circ\text{E}$).
  - `data/raw/ch2/ohrc/ch2_ohr_ncp_20211023T0027462822_d_img_d18.xml`: PDS4 flight metadata for the Oct 23, 2021 South Pole swath ($68.8^\circ\text{S to } 69.7^\circ\text{S}$, $32.1^\circ\text{E to } 32.55^\circ\text{E}$).
  - `data/samples/real_ch2_ohrc_sept2019_vikram.{png,xml}`: 10,107 × 1,200 pixel flight browse product (gitignored per open-source size policy, present in local test tree).
  - `data/samples/real_ch2_ohrc_landing_site.{png,xml}`: 9,369 × 1,200 pixel flight browse product.
* **Chandrayaan-2 IIRS (Imaging Infrared Spectrometer - ~80 m/px):**
  - `data/raw/ch2/iirs/ch2_iir_nri_20231003T2152304115_d_img_d18.xml` & `.hdr`: Authentic hyperspectral spectral cube metadata from Oct 3, 2023 ($66.2^\circ\text{S to } 72.4^\circ\text{S}$, $31.0^\circ\text{E to } 34.0^\circ\text{E}$).
  - `data/samples/real_ch2_iirs_infrared_pass.png`: Calibrated 512 × 512 infrared browse pass (tracked in Git).

### Real Cross-Sensor Overlap:
Notice that `real_ch2_ohrc_landing_site` (Latitude: -69.7° to -68.8°, Longitude: 32.1° to 32.55°) is **geographically nested inside** the coverage swath of `real_ch2_iirs_infrared_pass` (Latitude: -72.4° to -66.2°, Longitude: 31.0° to 34.0°). This authentic cross-sensor overlap serves as the empirical basis for validating the GIS Footprint Intersection and Common ROI extraction module (`src/gis/pre_registration.py`).

---

## 📥 4. Step-by-Step Guide: Ingesting Real Flight Data from ISRO ISSDC PRADAN

To download and register authentic Chandrayaan-2 products directly from ISRO's Planetary Data System:

### Step 1: Access ISRO ISSDC PRADAN
1. Visit the **ISSDC PRADAN portal**: [https://pradan.issdc.gov.in/ch2/](https://pradan.issdc.gov.in/ch2/)
2. Log in with your registered ISRO ISSDC credentials (registration is free and open to researchers).

### Step 2: Search and Select Chandrayaan-2 Products
1. Under **Mission**, select **CHANDRAYAAN-2**.
2. Under **Payload**, choose:
   - **OHRC** (Orbiter High Resolution Camera) for ultra-high resolution ($0.25\text{ m/px}$) optical imagery.
   - **TMC-2** (Terrain Mapping Camera-2) for stereo triplet DEMs ($5.0\text{ m/px}$).
   - **IIRS** (Imaging Infrared Spectrometer) for hyperspectral mineral mapping ($80\text{ m/px}$).
3. Define the geographic search box (e.g., South Pole region: Latitude $-90^\circ$ to $-65^\circ$).
4. Download the product package (`.zip` or `.tar`). A standard download contains:
   - `*_d_img.xml` (PDS4 XML observational and cartographic metadata label).
   - `*_d_img.img` or `*_d_img.tif` (Calibrated radiometric image array).
   - `*_b_brw.png` (Browse preview image).

### Step 3: Run the LunarRegX Automated Ingestion Pipeline
Execute `scripts/ingest_pradan_product.py` pointing to the downloaded archive or unpacked directory:

```bash
python scripts/ingest_pradan_product.py \
    --input /path/to/ch2_ohr_ncp_20211023T0027462822_d_img_d18.zip \
    --crop_size 1024
```

### What the Ingestion Pipeline Does:
1. **PDS4 Metadata Parsing:** Extracts selenographic footprint corners, solar azimuth (`solar_azimuth_angle`), solar elevation (`solar_elevation_angle`), and incidence angles directly from the XML tree.
2. **Sensor Auto-Detection:** Maps product codes (`ohr`, `tmc`, `iir`) to instrument GSD and physical constants.
3. **Radiometric Normalization:** Normalizes 10-bit, 12-bit, or 16-bit DN values to floating-point $[0.0, 1.0]$ with percentile-based histogram stretching.
4. **Sub-Scene Tiling:** Generates registration-ready $1024 \times 1024$ or full-scene crops stored in `data/raw/extracted_pradan/`.
5. **Output Delivery:** Saves paired GeoTIFF and JSON metadata files ready for immediate CLI (`register.py`) or Streamlit UI registration.
