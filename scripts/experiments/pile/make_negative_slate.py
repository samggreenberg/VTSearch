#!/usr/bin/env python3
"""Build the negative-pass slate: is the SHARED pool actually clean?

Every per-class slate measured one class against its own negatives. This one
measures the pool itself, and it is a different question. An image sits in the
shared pool because it carries no VG label for any candidate -- but the whole
finding of #3588 is that a missing VG name is not an absent object. The random
stratum of the thirteen class slates put per-class pool error between 0.0% and
7.1%; what nobody has measured is the JOINT rate, the share of pool images that
hold at least one of the thirteen. That number is what a negative in this
benchmark is worth.

Two strata, mirroring `make_class_slate.py` so the two passes stay comparable:

* ``random`` -- uniform from the pool. This is the estimator. Nothing else here
  measures anything; the boundary stratum is chosen to be wrong.
* ``boundary`` -- ranked by the best text score over all thirteen class names,
  so contamination is surfaced cheaply. Biased by design, and the manifest
  records WHICH class drove each row so the bias can be read afterwards.

Images VG already labels with a candidate are evicted first: they are known
contamination, they are counted by `make_class_slate.py` as `evicted.json`, and
asking a reviewer about them measures nothing.

**Polarity is stated positively on purpose.** The reviewer's question is "do you
see NONE of the thirteen?", so Good means clean. A conjunction of thirteen
negatives is not something anyone can hold in their head, which is why the
guide's checklist is ordered by measured hit rate rather than alphabetically --
six classes account for every pool error observed so far.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pile_config as pc  # noqa: E402
from make_class_slate import canonicalise, log  # noqa: E402
from pilebuild.loaders.vg_scale import read_vg_labels, vg_source  # noqa: E402

#: Ordered by measured pool error from the thirteen class slates, so the
#: reviewer scans in the order objects actually turn up rather than
#: alphabetically. The tail contributed ZERO errors in 350 uniform draws.
CHECKLIST = (
    "car",
    "bowl",
    "chair",
    "cup",
    "vase",
    "truck",
    "fork",
    "bottle",
    "spoon",
    "sink",
    "bench",
    "cell phone",
    "fire hydrant",
)

DETECTOR = "none of the 13"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=str(pc.EMBEDDINGS / "vg_scale__siglip.pkl"))
    ap.add_argument("--embedder", default="siglip")
    ap.add_argument("--out", default=str(pc.PILE.parent / "classes-3588" / "slates"))
    ap.add_argument("--boundary", type=int, default=100)
    ap.add_argument("--random", dest="n_random", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260905)
    args = ap.parse_args()

    from vtscore.embedding import embed_text_query  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
    from _cells_io import load_medias  # noqa: PLC0415

    classes = tuple(CHECKLIST)
    # The CANDIDATE table, not the shipped one: these classes are candidates,
    # and `SCALE_VG_NAMES` would silently fall back to the bare class name for
    # every one of them -- reading `cup` while ignoring mug, goblet and the
    # rest, which is exactly the spelling-split blindness this study found.
    vg_names = {c: pc.SCALE_CANDIDATE_VG_NAMES.get(c, (c,)) for c in classes}
    ambiguous = {c: pc.SCALE_CANDIDATE_VG_AMBIGUOUS.get(c, ()) for c in classes}
    wanted = {n for names in vg_names.values() for n in names}
    wanted |= {n for names in ambiguous.values() for n in names}
    log(
        f"reading VG source for {len(wanted)} names over {len(classes)} classes "
        f"({sum(len(v) for v in ambiguous.values())} ambiguous)"
    )
    paths, records, dims = vg_source()
    labels = read_vg_labels(records, paths, dims, wanted)
    canonicalise(labels, dict(vg_names))

    medias = load_medias(Path(args.cell))
    pool = [i for i in sorted(medias) if not medias[i].get("categories")]
    log(f"{len(pool)} shared negatives from {Path(args.cell).name}")

    # Known contamination is not the question -- evict it and say how much.
    holds = {i for i in pool if any(c in labels.get(i, {}) for c in classes)}
    # An ambiguous spelling is evidence of neither presence nor absence, so an
    # image carrying one is barred from the negative pool rather than counted
    # as clean -- the same three-valued treatment `lift_ambiguous` gives bands.
    amb_names = {n for names in ambiguous.values() for n in names}
    amb = {i for i in pool if i not in holds and amb_names & set(labels.get(i, {}))}
    clean_pool = [i for i in pool if i not in holds and i not in amb]
    log(
        f"evicted {len(holds)} images VG already labels with a candidate "
        f"({100 * len(holds) / len(pool):.1f}%) and {len(amb)} carrying an "
        f"ambiguous spelling; {len(clean_pool)} remain"
    )

    mat = np.stack([np.asarray(media_embedding(medias[i]), dtype=np.float32) for i in clean_pool])
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12

    best = np.full(len(clean_pool), -np.inf, dtype=np.float32)
    driver = [""] * len(clean_pool)
    for c in classes:
        tv = embed_text_query(c, "image", embedder_name=args.embedder)
        if tv is None:
            raise SystemExit(f"no text tower for embedder {args.embedder!r}")
        v = np.asarray(tv, dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-12
        s = mat @ v
        hit = s > best
        best = np.where(hit, s, best)
        for n in np.nonzero(hit)[0]:
            driver[n] = c

    order = sorted(range(len(clean_pool)), key=lambda n: -float(best[n]))
    boundary = order[: args.boundary]
    rest = [n for n in order[args.boundary :]]
    rng = random.Random(args.seed)
    uniform = rng.sample(rest, min(args.n_random, len(rest)))

    cdir = Path(args.out) / DETECTOR.replace(" ", "_")
    cdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for stratum, items in (("boundary", boundary), ("random", uniform)):
        for n in items:
            i = clean_pool[n]
            src = paths.get(i)
            if src is None:
                continue
            (cdir / f"{i}.jpg").write_bytes(src.read_bytes())
            # COCO annotates all 80 of its classes on an image it annotates at
            # all, so one exhaustive flag settles the whole conjunction at once
            # -- a scored subset for the negative pass, free.
            rows.append(
                {
                    "image_id": i,
                    "class": "clean",
                    "stratum": stratum,
                    "cell": "",
                    "text_score": round(float(best[n]), 4),
                    "reference": "present" if medias[i].get("labels_exhaustive") else "",
                    "exhaustive": "yes" if medias[i].get("labels_exhaustive") else "no",
                    "n_boxes": 0,
                    "detector": DETECTOR,
                    "driver": driver[n],
                }
            )

    with (cdir / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (Path(args.out) / "negative_pass.json").write_text(
        json.dumps(
            {
                "evicted_known": len(holds),
                "pool": len(pool),
                "clean_pool": len(clean_pool),
                "evicted_ambiguous": len(amb),
                "checklist": list(classes),
                "rows": len(rows),
            },
            indent=1,
        )
    )

    n_exh = sum(1 for r in rows if r["exhaustive"] == "yes")
    log(f"wrote {len(rows)} rows to {cdir} ({n_exh} scored, {100 * n_exh / len(rows):.0f}%)")
    from collections import Counter

    log(f"boundary drivers: {dict(Counter(r['driver'] for r in rows if r['stratum'] == 'boundary').most_common(6))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
