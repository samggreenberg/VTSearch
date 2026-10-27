"""#5330: every detection of det_10g (buffalo_l) and det_2.5g (buffalo_m) on FHIBE's one-person photos at 1024/640.

    python buffalo_m_detect_5330.py <imagekit src dir> <shard> <nshards> <out.csv>
One row per detection: uid, size, det, confidence, box x0 y0 x1 y1, 5 landmarks (x,y), in long-side pixels.
"""

import csv
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, sys.argv[1])
from imagekit.face import FaceDetector
from imagekit.runtime import RuntimeOptions

shard, n, out = int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
D = Path("/expscratch/sgreenberg/fhibe/fhibe-full-resolution-674a7dcf-derived")
M = Path("/expscratch/sgreenberg/imagekit-models")
dets = {}
for name, p in (("l", M / "buffalo_l/det_10g.onnx"), ("m", M / "buffalo_m/det_2.5g.onnx")):
    d = FaceDetector()
    d.initialize(RuntimeOptions(num_threads=int(sys.argv[5]) if len(sys.argv) > 5 else 4), str(p))
    dets[name] = d
uids = sorted(f.stem for f in (D / "long1024").glob("*.jpg"))[shard::n]
with out.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(
        ["uid", "size", "det", "ms", "conf", "x0", "y0", "x1", "y1"] + [f"{a}{i}" for i in range(5) for a in "xy"]
    )
    for uid in uids:
        for size in (1024, 640):
            a = np.asarray(Image.open(D / f"long{size}" / f"{uid}.jpg").convert("RGB"))
            for name, d in dets.items():
                t = time.perf_counter()
                r = d.detect(a)
                ms = 1000 * (time.perf_counter() - t)
                if not r:
                    w.writerow([uid, size, name, round(ms, 1), "", "", "", "", ""] + [""] * 10)
                for f in r:
                    lm = np.asarray(f.landmarks).reshape(-1).tolist() if f.landmarks is not None else [""] * 10
                    w.writerow(
                        [uid, size, name, round(ms, 1), round(float(f.confidence), 4)]
                        + [round(float(v), 2) for v in f.bbox]
                        + [round(float(v), 2) for v in lm]
                    )
print("done", len(uids))
