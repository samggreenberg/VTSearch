#!/usr/bin/env python3
"""Score a finished negative-pass group against COCO.

FIXES A BUG IN THE MANIFEST. `make_negative_slate.py` wrote

    "reference": "present" if medias[i].get("labels_exhaustive") else ""

which records only WHETHER an image is scored, not what the answer is -- every
scored row claimed "clean". The real reference is per-group and comes from COCO:
an image is clean for a group when COCO, which annotates all eighty of its
classes on any image it annotates, lists none of that group's members.

READS THE COMMITTED VERDICTS, NOT THE BANK (#4012). This used to open the
reviewer's `negbank/*.json` labelsets, which were deleted with the study dir on
2026-09-18 (#4001) and exist in no record. The judgements themselves survive in
`docs/experiments/2026-09-06-shipped-pool-3666/verdicts.csv` with the polarity
already applied, so the group scores are computable from the repository alone;
see `negative_pass_verdicts.py`. What is gone is the raw label string, which
nothing here needed once `present` is resolved.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/experiments/pile")
sys.path.insert(0, "scripts/experiments/calibration")
import negative_pass_verdicts as npv  # noqa: E402
import pile_config as pc  # noqa: E402
from _cells_io import load_medias  # noqa: E402
from coco_anchor import coco_truth, ensure_sources  # noqa: E402

rows = npv.load()
ALL = list(pc.SCALE_CANDIDATES_3588) + list(pc.SCALE_CLASSES_ORIGINAL)

m = load_medias(pc.EMBEDDINGS / "vg_scale__siglip.pkl")
anchor = Path(pc.PILE / "coco_anchor")
image_data, instances = ensure_sources(anchor, False)
truth = coco_truth(instances, set(ALL))
meta = json.loads(Path(image_data).read_text())
coco_of = {int(x["image_id"]): int(x["coco_id"]) for x in meta if x.get("coco_id")}


def coco_present(iid):
    cid = coco_of.get(iid)
    if cid is None or not m.get(iid, {}).get("labels_exhaustive"):
        return None
    return {c for c, b in truth.get(cid, {}).items() if b}


for name in npv.GROUP_PASSES:
    group = npv.for_pass(rows, name)
    if not group:
        raise SystemExit(f"{npv.VERDICTS} has no pass named {name!r}")
    members = set(group[0].members)
    said_clean, said_dirty, scored, agree, seeded = 0, 0, 0, 0, 0
    misses, false_alarms = [], []
    for v in group:
        human_clean = not v.present
        said_clean += human_clean
        said_dirty += not human_clean
        if v.seeded:
            # Written from COCO rather than judged, so scoring it against COCO
            # measures nothing. The original pass had no way to tell these apart.
            seeded += 1
            continue
        p = coco_present(v.image_id)
        if p is None:
            continue
        scored += 1
        ref_clean = not (p & members)
        if ref_clean == human_clean:
            agree += 1
        elif human_clean:
            misses.append((v.image_id, sorted(p & members)))
        else:
            false_alarms.append(v.image_id)
    n = len(group)
    print(f"\n=== {name} ({', '.join(sorted(members))}) ===")
    print(f"  {n} labelled: {said_dirty} NOT clean ({100 * said_dirty / n:.1f}%), {said_clean} clean")
    if seeded:
        print(f"  {seeded} COCO-seeded row(s) excluded from the scoring: the reference, not a judgement")
    if scored:
        print(f"  scored against COCO: {scored} rows, agreement {100 * agree / scored:.1f}%")
        print(f"    COCO saw one and the reviewer did not : {len(misses)}")
        print(f"    reviewer saw one and COCO did not     : {len(false_alarms)}")
        if misses:
            print("    missed:", Counter(c for _i, cs in misses for c in cs).most_common())
    print(f"  images the reviewer FOUND one in: {[v.image_id for v in group if v.present and not v.seeded]}")
