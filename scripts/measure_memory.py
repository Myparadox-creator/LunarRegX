import sys
import tracemalloc
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.io.dataset import load_lunar_image
from src.pipeline import LunarRegistrationPipeline, RegistrationPipelineConfig

method = sys.argv[1] if len(sys.argv) > 1 else "PHASE_STRUCTURAL"

s_p = Path("data/samples/pair1_baseline_src.png")
r_p = Path("data/samples/pair1_baseline_ref.png")
src = load_lunar_image(s_p)
ref = load_lunar_image(r_p)

tracemalloc.start()
cfg = RegistrationPipelineConfig(feature_method=method)
pipe = LunarRegistrationPipeline(config=cfg)
res = pipe.run(src, ref)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"[{method}] Tracemalloc Peak: {peak / (1024*1024):.2f} MB")
