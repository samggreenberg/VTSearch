"""What does a cell's SHIPPED negative set actually look like on coco_better?

The issue quotes #3670's vg_scale figure (evaluable pool ~1,900 larger than
SCALE_N_NEG, ~84% barren). coco_better is a different build, so measure it.
"""

import collections
import statistics
import sys

sys.path.insert(0, "scripts/experiments/pile")
sys.path.insert(0, "scripts/experiments/calibration")
import pile_config as pc  # noqa: E402

pc.setup_env()
from _cells_io import load_medias  # noqa: E402

des = load_medias(pc.EMBEDDINGS / "coco_better__siglip.pkl")
cells = [pc.scale_cell(c, b) for c in pc.SCALE_CLASSES for b in pc.BOX_BANDS]

ev_of = collections.defaultdict(list)
cats_of = {}
for i, d in des.items():
    cats = set(d.get("categories") or [])
    cats_of[i] = cats
    for cell in d.get("evaluable_categories") or []:
        if cell not in cats:
            ev_of[cell].append(i)

barren = {i for i, c in cats_of.items() if not c and des[i].get("evaluable_categories")}
rows = []
for cell in cells:
    neg = ev_of[cell]
    nb = sum(1 for i in neg if i in barren)
    rows.append((cell, len(neg), nb, len(neg) - nb))

n = [r[1] for r in rows]
print("cells: %d   SCALE_N_NEG = %d   barren shared pool = %d" % (len(rows), pc.SCALE_N_NEG, len(barren)))
print("shipped negatives per cell: min %d  median %d  max %d" % (min(n), int(statistics.median(n)), max(n)))
print("  = barren %d + cross-class %d..%d" % (len(barren), min(r[3] for r in rows), max(r[3] for r in rows)))
print("barren share of the shipped negative set: %.0f%%" % (100 * statistics.mean(r[2] / r[1] for r in rows)))
print("cross-class share:                        %.0f%%" % (100 * statistics.mean(r[3] / r[1] for r in rows)))
print()
for cell, tot, nb, nc in rows[:3] + rows[-3:]:
    print("  %-20s %6d  barren %5d (%2.0f%%)  cross-class %5d" % (cell, tot, nb, 100 * nb / tot, nc))
