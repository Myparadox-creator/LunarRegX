"""
Streamlit UI verification tests ensuring Custom File Upload locks the registration
pipeline until both raster images and their corresponding XML files are provided.
"""
from pathlib import Path
import pytest


def test_streamlit_file_upload_locks_pipeline_without_xml(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))

    app_path = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception

    # Demo mode must keep EXECUTE REGISTRATION PIPELINE enabled
    demo_exec = [b for b in at.button if "EXECUTE REGISTRATION PIPELINE" in b.label]
    assert len(demo_exec) == 1
    assert demo_exec[0].disabled is False

    # Switch to Custom Upload -> File Upload
    at.sidebar.segmented_control[0].set_value("Custom Upload").run()
    assert not at.exception
    at.sidebar.segmented_control[1].set_value("File Upload").run()
    assert not at.exception

    # Registration pipeline button must be locked (disabled) until images + XMLs are uploaded
    upload_exec = [b for b in at.button if "EXECUTE REGISTRATION PIPELINE" in b.label]
    assert len(upload_exec) == 1
    assert upload_exec[0].disabled is True
    assert any("corresponding XML metadata files" in info.value for info in at.info)
