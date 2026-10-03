# 🧠 LunarRegX Core Architectural Brain & Context Reference

> **Single Source of Truth** for developers, researchers, and AI pair-programmers working on **LunarRegX**.
> Read this document first before making architectural changes or extending features to prevent design drift, mathematical errors, or regression bugs.

---

## 🏛️ 1. Core Philosophy & Immutable Invariants

1. **Planetary Cartographic Accuracy (IAU-2000 Datum Invariant)**:
   * The Moon is **NOT** Earth. **Never assume terrestrial WGS-84 / EPSG:4326 datums.**
   * All spatial calculations MUST use the **IAU-2000 / IAU-2015 Lunar Reference Datum**:
     * Mean Lunar Radius: $R = 1,737,400.0\text{ m}$
     * Equatorial Radius: $a = 1,738,140.0\text{ m}$
     * Polar Radius: $b = 1,735,970.0\text{ m}$
   * Projections:
     * Polar Regions ($|\text{lat}| \ge 65^\circ$): **Polar Stereographic** (`+proj=stere +lat_0=±90 +lat_ts=±80 ...`)
     * Equatorial / Mid-Latitudes ($|\text{lat}| < 65^\circ$): **Equirectangular** (`+proj=eqc +lat_ts=lat_0 ...`)

2. **Illumination Invariance (Phase Congruency Invariant)**:
   * Raw intensity gradients (Sobel, SIFT, ORB) fail catastrophically under extreme crater shadow flips ($180^\circ$ solar azimuth reversal).
   * The primary correspondence engine is **Fourier Phase Congruency (Log-Gabor Filter Bank)**:
     * Number of orientations: $n_{\text{orient}} = 6$
     * Number of frequency scales: $n_{\text{scale}} = 3$
     * Maximum Index Map (MIM) captures structural energy invariant to contrast reversals and radiometric gain changes.

3. **Multi-Scale GSD Normalization & Aspect Ratio Invariant**:
   * Chandrayaan-2 payloads have vast Ground Sampling Distance (GSD) disparities:
     * OHRC: $0.25\text{ m/px}$
     * TMC-2: $5.0\text{ m/px}$ ($20\times$ ratio)
     * IIRS: $80.0\text{ m/px}$ ($320\times$ ratio)
     * LRO LROC-NAC: $0.50\text{ m/px}$ ($2\times$ ratio)
   * **Rule:** Never perform non-uniform stretching that distorts crater circularity into oval ellipses. Always preserve physical aspect ratios using isotropic scaling.
   * **Rule for Extreme GSD Ratios ($> 10\times$):** Simulate the lower-resolution sensor's Point Spread Function (PSF) using Gaussian low-pass filtering on the high-resolution image before correspondence detection.

4. **Sub-Pixel Precision Benchmark ($< 0.30\text{ px}$ RMSE)**:
   * Coarse matches must always be refined using **continuous 2D parabolic surface peak fitting** over the Normalized Cross-Correlation (NCC) surface:
     $$\Delta x = \frac{R(x-1, y) - R(x+1, y)}{2[R(x-1, y) - 2R(x, y) + R(x+1, y)]}$$
   * Outlier rejection requires **RANSAC / MAGSAC++** coupled with **SVD Condition Number Guard** ($\kappa(H) < 1000$) to eliminate ill-conditioned degenerate geometries.

5. **Resource Envelope & Memory Safety (< 1 GB RAM Ceiling)**:
   * Streamlit Community Cloud enforces a strict **1 GB RAM** threshold.
   * Never allocate uncompressed gigapixel numpy arrays into memory simultaneously.
   * Use bounding-box windowing (ROI crops) and float32 normalizations.

---

## 📁 2. Complete Codebase Map

```
lunar_registration/
├── app/
│   └── streamlit_app.py         # Aerospace cockpit UI dashboard with 11-step walkthrough
├── src/
│   ├── pipeline.py              # Master 11-stage registration pipeline orchestrator
│   ├── features/
│   │   ├── phase_structural.py  # Log-Gabor Phase Congruency & MIM descriptor engine
│   │   ├── deep_matcher.py      # LoFTR / LightGlue / CNN matcher with multi-tier fallback
│   │   └── classical.py         # SIFT / AKAZE / ORB benchmark engines
│   ├── gis/
│   │   ├── lunar_crs.py         # IAU-2000 Moon CRS & PyProj forward/inverse projections
│   │   ├── footprint.py         # Shapely polygon intersection & common ROI window extraction
│   │   └── pre_registration.py  # Solar/camera geometry & spatial overlap analyzer
│   ├── multimodal/
│   │   └── sensor_encoder.py    # Sensor-specific structural normalizers (OHRC, TMC2, IIRS)
│   ├── subpixel/
│   │   └── refinement.py        # 2D parabolic continuous peak sub-pixel interpolator
│   ├── geometry/
│   │   └── estimator.py         # Robust geometric estimator with SVD condition number check
│   ├── warping/
│   │   └── resampler.py         # Bicubic image warper & authentic GeoTIFF exporter
│   ├── spatial/
│   │   └── anms.py              # Adaptive Non-Maximal Suppression for spatial grid spread
│   ├── evaluation/
│   │   ├── failure_detector.py  # 5-criteria SUCCESS/WARNING/FAILURE classifier
│   │   └── metrics.py           # RegistrationMetrics dataclass (RMSE, inliers, physical error)
│   └── io/
│       ├── dataset.py           # Multi-bit-depth (8/12/16-bit) image loader & normalizer
│       └── metadata.py          # SensorMetadata container & PDS4 XML label parser
├── scripts/
│   ├── ingest_pradan_product.py # Automated ISRO PRADAN & PDS4 ZIP unpacker and cropper
│   └── generate_lunar_benchmarks.py # Benchmark suite generator (Pairs 1-5)
├── data/
│   └── samples/                 # Curated flight swaths and benchmark image pairs
└── tests/                       # 37 automated unit and integration tests (100% passing)
```

---

## 🛡️ 3. Rules & Guidelines for Feature Implementation

1. **Preserve All 37 Automated Tests:**
   * Run `python -m pytest -q` after every code change. All 37 tests MUST pass.
2. **State & Role Swapping Safety:**
   * When modifying `streamlit_app.py`, always ensure session state keys (`current_mode`, `swap_roles`, `reg_result`) remain consistent across reruns.
3. **No External Network Dependencies in Core Algorithms:**
   * Offline-first capability. All feature engines, Phase Congruency filters, and PyProj transformers must operate completely offline.
