"""#5341: every detection of det_500m (buffalo_s) on FHIBE's one-person photos at 1024/640.

    python buffalo_s_detect_5341.py <imagekit src dir> <shard> <nshards> <out.csv> [threads]
The same rows as ``buffalo_m_detect_5330.py`` (uid, size, det, ms, confidence, box, 5 landmarks, in long-side
pixels) with ``det`` = ``s``, so its shards sit beside #5330's ``l`` / ``m`` shards and the same compare reads them.
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
DET = Path("/expscratch/sgreenberg/imagekit-models/grid-default/imagekit/det_500m.onnx")
det = FaceDetector()
det.initialize(RuntimeOptions(num_threads=int(sys.argv[5]) if len(sys.argv) > 5 else 4), str(DET))
uids = sorted(f.stem for f in (D / "long1024").glob("*.jpg"))[shard::n]
with out.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(
        ["uid", "size", "det", "ms", "conf", "x0", "y0", "x1", "y1"] + [f"{a}{i}" for i in range(5) for a in "xy"]
    )
    for uid in uids:
        for size in (1024, 640):
            a = np.asarray(Image.open(D / f"long{size}" / f"{uid}.jpg").convert("RGB"))
            t = time.perf_counter()
            r = det.detect(a)
            ms = 1000 * (time.perf_counter() - t)
            if not r:
                w.writerow([uid, size, "s", round(ms, 1), "", "", "", "", ""] + [""] * 10)
            for f in r:
                lm = np.asarray(f.landmarks).reshape(-1).tolist() if f.landmarks is not None else [""] * 10
                w.writerow(
                    [uid, size, "s", round(ms, 1), round(float(f.confidence), 4)]
                    + [round(float(v), 2) for v in f.bbox]
                    + [round(float(v), 2) for v in lm]
                )
print("done", len(uids))
