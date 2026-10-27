"""#5330: cost.py with onnxruntime-gpu from a private target dir loaded ahead of the venv's CPU build."""

import runpy
import sys
import os

sys.path.insert(0, os.environ.get("ORT_GPU_DIR", "/expscratch/sgreenberg/ort-gpu-5330"))
import onnxruntime as ort

getattr(ort, "preload_dlls", lambda: None)()
print("ort", ort.__version__, ort.get_available_providers(), file=sys.stderr)
from vtscore.media.face import imagekit_bridge as b

p = b.get_face_pipeline()
sess = [
    getattr(v, "get_providers", None) and v.get_providers()
    for o in vars(p).values()
    for v in (vars(o).values() if hasattr(o, "__dict__") else [])
]
print("sessions", [s for s in sess if s], file=sys.stderr)
sys.argv = ["cost.py", *sys.argv[1:]]
runpy.run_path("scripts/experiments/fhibe/buffalo_m_cost_5330.py", run_name="__main__")
