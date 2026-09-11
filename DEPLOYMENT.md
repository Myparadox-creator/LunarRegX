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