"""
Root Streamlit Entrypoint for 1-Click Cloud Deployment (Streamlit Community Cloud).
Dispatches cleanly to app/streamlit_app.py.
"""
import sys
from pathlib import Path
import runpy

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

app_path = ROOT_DIR / "app" / "streamlit_app.py"
runpy.run_path(str(app_path), run_name="__main__")
