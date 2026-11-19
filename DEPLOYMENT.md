# 🚀 LunarRegX Deployment Guide
**Smart India Hackathon (SIH 2026) | Problem ID: 26166**

This guide provides instructions for deploying the **LunarRegX Prototype Dashboard** to **Streamlit Community Cloud**, as well as hosting via **Docker** or running the **FastAPI REST API**.

---

## 🌕 1. Deploying on Streamlit Community Cloud (100% Free)

Streamlit Community Cloud hosts the full, interactive Python application with all 6 correspondence engines directly from your GitHub repository.

### Step-by-Step 1-Click Deployment:

1. **Verify Latest Code on GitHub:**
   Make sure all changes are pushed to `main`:
   ```bash
   git push origin main
   ```

2. **Access Streamlit Community Cloud:**
   Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with your GitHub account.

3. **Deploy the App:**
   * Click **"New app"** (or **"Create app"**).
   * Fill in the repository parameters:
     * **Repository:** `Myparadox-creator/LunarRegX`
     * **Branch:** `main`
     * **Main file path:** `streamlit_app.py`
   * Click **"Deploy!"**.

4. **Live Access:**
   Streamlit Cloud automatically provisions the container, installs dependencies from `requirements.txt` and `packages.txt`, and generates a public URL:
   ```text
   https://<your-custom-subdomain>.streamlit.app
   ```

### Pre-Configured Cloud Assets:
* **Root Entrypoint:** `streamlit_app.py` dispatches cleanly to `app/streamlit_app.py` with full package path resolution.
* **C-Libraries:** `packages.txt` provides `libgl1` and `libglib2.0-0` for headless OpenCV cartography.
* **Geospatial Stack:** `requirements.txt` includes `pyproj`, `shapely`, `rasterio`, and `affine`.

---

## 🐳 2. Deploying via Docker (Any Cloud / Server / Render)

For containerized deployments on local servers, AWS EC2, GCP Cloud Run, or Render:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8000
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

### Build & Run:
```bash
# Build the container
docker build -t lunarregx .

# Run Streamlit dashboard
docker run -p 8501:8501 lunarregx
```

---

## ⚡ 3. Running the FastAPI REST API Service

To run the standalone, high-performance REST API backend for automated pipeline invocation:

```bash
# Launch FastAPI backend on port 8000
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

* **Interactive Swagger Documentation:** `http://localhost:8000/docs`
* **OpenAPI Schema:** `http://localhost:8000/openapi.json`
* **Health Check:** `http://localhost:8000/health`
* **Supported Sensors:** `http://localhost:8000/sensors`
* **Benchmark Results:** `http://localhost:8000/benchmarks`

---

## 🧠 4. LoFTR Model Weights & Domain Adaptation Status

### Honest Weights Policy
Deep transformer feature matching via LoFTR in LunarRegX operates under an honest weight disclosure policy:
1. **Default Production Deployment:** The deployed Streamlit dashboard and REST API utilize the standard pretrained outdoor LoFTR checkpoint (`loftr_outdoor`), which downloads automatically into PyTorch Hub cache or uses cached weights. Pretrained LoFTR delivers strong zero-shot deep dense correspondence across scale changes (0.34 px on Pair 3) and shear distortions (0.32 px on Pair 4).
2. **Experimental Fine-Tuning Script:** The repository includes `train_loftr_lunar.py`, an experimental PyTorch Lightning / PyTorch training harness for fine-tuning LoFTR on paired lunar surface swaths. A preliminary 1-epoch proof-of-concept run was performed (`models/loftr/lunar_finetuned/training_log.json`, train loss 13.69, val loss 14.07). Because `.ckpt` binary files (~46 MB) are gitignored according to open-source best practices, fresh repository clones and cloud containers operate with pretrained outdoor weights.
3. **Reproducing or Continuing Lunar Training:**
   To train a production-converged lunar LoFTR checkpoint, a workstation or cluster with an NVIDIA GPU (CUDA compute capability >= 7.0, minimum 16 GB VRAM) is required:
   ```bash
   python train_loftr_lunar.py --data_dir data/samples --epochs 50 --batch_size 4 --lr 1e-4
   ```
   Trained checkpoints saved to `models/loftr/lunar_finetuned/best.ckpt` will be automatically detected and loaded by `DeepCorrespondenceMatcher` if present.

---

## 📁 5. Root Entrypoint Architecture (`streamlit_app.py` vs `app.py`)

The repository contains two thin wrapper files at the root:
* **`streamlit_app.py`**: Default entrypoint detected by **Streamlit Community Cloud**. It ensures `ROOT_DIR` is in `sys.path` and dispatches cleanly to `app/streamlit_app.py`.
* **`app.py`**: Default entrypoint detected by **Hugging Face Spaces** (Docker and Streamlit Spaces). It mirrors the same clean dispatch to `app/streamlit_app.py`.

Both wrappers prevent path-resolution issues on cloud platforms without duplicating UI code.

---

## 📊 6. Memory Profile & Cloud Resource Safety (< 1 GB RAM Ceiling)

Streamlit Community Cloud enforces a strict **1.0 GB RAM memory ceiling**. Exceeding this threshold results in silent container termination with an `OOMKilled` status.

### Empirical Peak Memory Measurements:
Memory was profiled during end-to-end registration of $512 \times 512$ lunar scenes using Python's standard `tracemalloc` instrumentation:

| Algorithmic Engine | Tracemalloc Peak Heap | Estimated Process Working Set | 1 GB Free Tier Status |
| :--- | :---: | :---: | :--- |
| **PHASE_STRUCTURAL (Proposed)** | **108.76 MB** | ~180 MB | 🟢 **Safe** (> 800 MB Headroom) |
| **LoFTR (Pretrained Outdoor)** | **32.12 MB** | ~350 MB | 🟢 **Safe** (~650 MB Headroom) |
| **SIFT / AKAZE (Baselines)** | **14.50 MB** | ~95 MB | 🟢 **Safe** (> 900 MB Headroom) |

### Memory Hardening Mechanisms Implemented:
1. **Lazy Deep Imports:** `torch` and `kornia` are never imported at application launch. They are only loaded into memory if the user explicitly switches the correspondence engine to `LOFTR` or `LEARNED`.
2. **Resource Caching (`@st.cache_resource`):** When `LOFTR` is selected, the transformer weights are cached in memory across interactions using Streamlit's resource cache, preventing redundant model re-instantiations and memory leaks.
3. **In-Memory GeoTIFF Export:** Warped cartographic GeoTIFFs are synthesized on demand via `rasterio.io.MemoryFile` with immediate buffer release rather than persisting large intermediate files to disk.

