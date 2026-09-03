"""Measure `vg_scale`'s true positive supply per class, band-free and per band.

#3547 asks for a DEEPER haystack: `vg_scale_any` gives 300 positives per class
against 3900 negatives, and #3319's 400-click wave harvested 82-85% of the sim
half's ~150.  How deep the rebuilt pile can go is a property of the DATA, so it
has to be measured before a plan quotes a number -- which is the whole point of
`lessons/2026-09-02-one-pilot-cell-cleared-a-hazard-the-full-wave-hit.md`.

Runs the loader's own front half (read -> anchor -> correct -> band) and reports
supply.  Writes nothing; in particular it does NOT touch the roster.
"""

from __future__ import annotations

import json
import sys

import pile_config as pc

sys.path.insert(0, str(pc.Path(__file__).resolve().parent))

from pilebuild.corrections import load_corrections  # noqa: E402
from pilebuild.env import log  # noqa: E402
from pilebuild.loaders.vg_scale import (  # noqa: E402
    anchor_to_coco,
    apply_corrections,
    band_candidates,
    read_vg_labels,
)
from pilebuild.vgsource import vg_image_paths, vg_source  # noqa: E402

import coco_anchor as ca  # noqa: E402


def main() -> None:
    wanted = set(pc.SCALE_CLASSES)
    paths = vg_image_paths()
    _, records, dims = vg_source()

    image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    truth = ca.coco_truth(instances, wanted)
    with image_data.open() as fh:
        coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in json.load(fh) if m.get("coco_id")}

    corrections = load_corrections()
    labels = read_vg_labels(records, paths, dims, wanted)
    box_dims, exhaustive, n_anchored, n_reframed = anchor_to_coco(labels, dims, coco_of, truth, ca.COCO_DIMS, wanted)
    unbanded = apply_corrections(labels, corrections, box_dims, exhaustive)
    log(f"  labels: {len(labels)} VG images, {n_anchored} repaired from COCO, {n_reframed} re-framed")

    supply, _boxes, clean = band_candidates(labels, box_dims, unbanded)

    rows = []
    for c in pc.SCALE_CLASSES:
        per_band = {b: len(supply[c][b]) for b in pc.BOX_BANDS}
        union = len({i for b in pc.BOX_BANDS for i in supply[c][b]})
        rows.append((c, per_band, union))

    print("\n=== per-class positive supply (band-free `union` is what vg_scale_any sees) ===")
    print("%-12s %7s %7s %7s | %7s" % ("class", "small", "medium", "large", "union"))
    for c, pb, union in rows:
        print("%-12s %7d %7d %7d | %7d" % (c, pb["small"], pb["medium"], pb["large"], union))

    unions = sorted(u for _, _, u in rows)
    per_band_min = min(v for _, pb, _ in rows for v in pb.values())
    print("\nclean (negative-eligible) images: %d" % len(clean))
    print(
        "band-free union: min %d (%s), median %d, max %d"
        % (unions[0], min(rows, key=lambda r: r[2])[0], unions[len(unions) // 2], unions[-1])
    )
    print("thinnest single band across all classes: %d" % per_band_min)

    print("\n=== how deep can the pile go? (prevalence held at the designed 7.14%) ===")
    print("%-9s %-9s %-9s %-11s %-9s %s" % ("N_POS", "classes", "N_NEG", "cell medias", "sim pos", "note"))
    for n_pos in (300, 450, 600, 750, 900, 1200, 1500):
        keep = [c for c, _, u in rows if u >= n_pos]
        n_neg = round(n_pos * (1 - 0.0714286) / 0.0714286)
        note = "ALL 12" if len(keep) == 12 else "drops " + ",".join(c for c, _, u in rows if u < n_pos)
        print("%-9d %-9d %-9d %-11d %-9d %s" % (n_pos, len(keep), n_neg, n_pos + n_neg, n_pos // 2, note))


if __name__ == "__main__":
    main()
