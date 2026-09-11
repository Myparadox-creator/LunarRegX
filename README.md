# Robust Multi-Modal Lunar Image Registration (SIH Prototype)

A research-grade, production-quality prototype for registering Chandrayaan-2 (OHRC, TMC-2, IIRS) optical imagery against heterogeneous lunar reference images (LROC NAC/WAC) with sub-pixel accuracy and uniform spatial control-point distribution.

## Core Features
- **Illumination Invariance:** Log-Gabor Phase Congruency (Kovesi 1999 / RIFT Li et al. 2019) moments and Maximum Index Maps (MIM), resilient to 180° solar azimuth flips and crater shadow reversals.
- **Spatially Uniform Control Points:** Adaptive Non-Maximal Suppression (ANMS) and grid-based binning prevent single-crater clustering and maximize frame coverage.
- **Sub-Pixel Refinement:** Local Normalized Cross-Correlation (NCC) with continuous 2D parabolic quadratic peak surface fitting achieves < 0.1 pixel refinement.
- **Stability-Guided Geometric Selection:** Hierarchical model selection (Similarity → Affine → Homography) with SVD condition number check to eliminate degenerate keystone shearing.
- **Independent Validation:** 80/20 train/check point split preventing in-sample overfitting claims.
- **Sensor-Agnostic YAML Configs:** Configurable for Chandrayaan-2 OHRC (0.25 m/px), TMC-2 (5.0 m/px), IIRS (80 m/px), and LRO LROC-NAC (0.50 m/px).
- **Streamlit Web Dashboard:** Includes a dedicated 10-step **SIH Judge Demonstration Mode** for presentations.

## Installation & Setup
```bash
pip install -r requirements.txt
```

## Lunar LoFTR Fine-Tuning Pipeline (SIH26166)

The framework features an end-to-end training and fine-tuning pipeline to specialize LoFTR (Detector-Free Local Feature Matching with Transformers) on planetary lunar imagery (Chandrayaan-2 OHRC, TMC-2, IIRS, and LROC NAC):

- **Data Ingestion & Scanner:** Recursively searches `data/raw/ch2/ohrc`, `tmc2`, `iirs`, `lroc`, `selene`, and falls back to calibrated samples in `data/samples/`.
- **Selenographic Footprint Pairing:** Computes geospatial polygonal intersections (IAU-2000 Lunar Datum) with $\ge 20\%$ overlap filtering.
- **Pseudo-Ground-Truth Derivation:** Consumes multi-spectral representations with structural Log-Gabor phase congruency and strict RANSAC verification ($\kappa(H) < 2000$). Pairs with insufficient inliers are flagged rather than fabricated.
- **Numerically Stable Dual-Supervision Loss:** Implements coarse dual-log-softmax cross-entropy and fine regression loss directly optimizing CNN backbone and transformer attention layers with full autograd gradient propagation.
- **Transparent Model Routing:** The UI correspondence selector retains the 6 exact SIH options. When selecting `LOFTR (Deep Transformer)`, the backend automatically prioritizes `models/loftr/lunar_finetuned/best.ckpt` and gracefully falls back to pretrained weights if checkpoints are absent.

### Dataset Health Diagnostics
```bash
python validate_lunar_dataset.py
```

### LoFTR Training & Fine-Tuning
```bash
python train_loftr_lunar.py --epochs 5 --lr 1e-4 --batch_size 2
```

### Interactive Correspondence Visualizer
```bash
python scripts/verify_correspondences.py --source data/samples/pair1_baseline_src.png --reference data/samples/pair1_baseline_ref.png --save results/inspection_canvas.png
```

## Running the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

## Running the CLI
```bash
python register.py --source data/samples/pair1_baseline_src.png --reference data/samples/pair1_baseline_ref.png --method LOFTR --output_dir results/loftr_run
```

## Running the FastAPI REST API Backend
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Interactive OpenAPI/Swagger documentation is available at:
`http://localhost:8000/docs`

## Running Automated Benchmarks & Comparisons
```bash
# Run benchmark across all 6 methods + LoFTR fine-tuned vs pretrained comparison
python experiments/benchmark.py
# Focus on LoFTR
python experiments/benchmark.py --method LOFTR
```

## Running the Automated Test Suite (35 Unit Tests)
```bash
python -m pytest -v
```

