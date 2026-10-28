"""#5330: per-face cost of the imagekit path under buffalo_l and buffalo_m, same photos, same node.

    IMAGEKIT_FACE_CONFIG=<cfg> python buffalo_m_cost_5330.py <n photos> <size> <label>
image2face convert with provider imagekit (SCRFD detect + ArcFace embed + crop), one pass; 5 warm-up photos.
"""

import sys
import time

sys.path.insert(0, "scripts/experiments/pile")
import pile_config as pc
from pilebuild.loaders import fhibe
from vtscore.converters.image2face import CONVERTER as conv

n, size, label = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
photos = fhibe.read_photos(fhibe.data_dir())[: n + 5]
copies = pc.fhibe_derived_dir(size)
medias = [{"media_bytes": (copies / f"{p.uid}.jpg").read_bytes(), "filename": f"{p.uid}.jpg"} for p in photos]
params = {"align": "roll"}
for m in medias[:5]:
    conv.convert(m, params)
t = time.perf_counter()
faces = 0
for m in medias[5:]:
    faces += len(conv.convert(m, params))
s = time.perf_counter() - t
print(f"{label} {size}: {n} photos, {faces} faces; {1000 * s / n:.0f} ms/photo, {1000 * s / max(faces, 1):.0f} ms/face")
