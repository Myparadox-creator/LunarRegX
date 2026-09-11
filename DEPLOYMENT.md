# 🚀 LunarRegX Deployment Guide
**Smart India Hackathon (SIH 2026) | Problem ID: 26166**

This guide provides step-by-step instructions for deploying the **LunarRegX Web Dashboard** and **API Gateway** to **Vercel**, as well as hosting the persistent PyTorch/Streamlit application on **Streamlit Community Cloud** or **HuggingFace Spaces**.

---

## 🌐 1. Deploying to Vercel (Recommended for SIH Submissions)

Vercel provides a lightning-fast, globally distributed edge deployment with zero cold starts. It hosts the **LunarRegX Web Showcase**, the **Interactive SIH 2026 Judge Walkthrough**, and the **Serverless API Gateway**.

### Method A: 1-Click GitHub Import (Easiest)
1. Go to [vercel.com](https://vercel.com) and log in with your GitHub account.
2. Click **"Add New..."** &rarr; **"Project"**.
3. In the list of repositories, select **`LunarRegX`** (or paste `https://github.com/Myparadox-creator/LunarRegX.git`).
4. Vercel will automatically detect `vercel.json`:
   * **Framework Preset:** `Other`
   * **Root Directory:** `./`
   * **Build Command:** Leave blank (handled automatically)
   * **Output Directory:** Leave blank (handled automatically)
5. Click **Deploy**.
6. In **~15 seconds**, your project will be live at:
   ```text
   https://lunarregx.vercel.app
   ```
   *(or your assigned custom `.vercel.app` domain)*

### Method B: Deploying via Vercel CLI
If you have Node.js installed, you can deploy straight from your terminal:
```bash
# 1. Install Vercel CLI globally
npm install -g vercel

# 2. Login to your Vercel account
vercel login

# 3. Deploy to production
vercel --prod
```

### What Vercel Serves:
* **Interactive Web Dashboard:** `https://your-domain.vercel.app/`
* **Serverless Health Check:** `https://your-domain.vercel.app/api/health`
* **Supported Sensors API:** `https://your-domain.vercel.app/api/sensors`
* **Benchmark Metrics API:** `https://your-domain.vercel.app/api/results`
* **OpenAPI Swagger Docs:** `https://your-domain.vercel.app/api/docs`

---

## 🌕 2. Deploying Streamlit on Streamlit Community Cloud (100% Free)

To host the full, stateful Python Streamlit application with all 6 correspondence engines online:

1. Push your latest code to GitHub:
   ```bash
   git push origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New app"**.
4. Configure the deployment:
   * **Repository:** `Myparadox-creator/LunarRegX`
   * **Branch:** `main`
   * **Main file path:** `app/streamlit_app.py`
5. Click **"Deploy!"**.
6. Streamlit Cloud will spin up a persistent Python environment and give you a public URL (e.g. `https://lunarregx.streamlit.app`).

---

## 🐳 3. Deploying via Docker (Any Cloud / Render / Railway)

If you prefer deploying a full containerized REST API or web app:

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8000
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

Build and run locally or on any cloud:
```bash
docker build -t lunarregx .
docker run -p 8501:8501 -p 8000:8000 lunarregx
```

---

## 📋 Architecture Comparison

| Feature | Vercel (Edge & Serverless) | Streamlit Community Cloud | Local Workstation |
| :--- | :---: | :---: | :---: |
| **Primary Use Case** | Public Jury Presentation & API | Full Interactive App | Development & PRADAN Ingestion |
| **Deployment Speed** | **< 15 seconds** | ~2 minutes | Instant |
| **Cold Starts** | **0 seconds (Global CDN)** | ~10-20 seconds | None |
| **PDS4 Zip Ingestion** | Metadata & Tiles | Small tiles (< 50MB) | **Full 1.15 GB rasters** |
| **Cost** | **100% Free** | **100% Free** | Free |