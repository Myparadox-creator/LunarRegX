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

## Running the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

## Running the CLI
```bash
python register.py --source data/samples/pair1_baseline_src.png --reference data/samples/pair1_baseline_ref.png --output_dir results/pair1
```

## Running Automated Benchmarks & Ablation Studies
```bash
python experiments/benchmark.py
python experiments/ablation.py
```

## Running the Test Suite
```bash
python -m pytest -o pythonpath=. -v
```
