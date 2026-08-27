#!/usr/bin/env python
"""Rank candidate classes so a human can pick a roster from a shortlist.

    python shortlist.py --corpus <dir>                  # ranked table + sheet
    python shortlist.py --corpus <dir> --write-roster    # also draft roster.json

The corpus builder clusters every mark it can find, which on SPODS alone
proposes on the order of a hundred candidate classes.  Two dozen of them are
worth hand-verifying and the rest are not, and the difference is mostly legible
from measurable properties.  This ranks them and renders one contact sheet of
exemplars, so picking a roster is looking at a page rather than reading JSON.

**The ranking is an ordering aid, not a decision.**  Every signal here is a
proxy: they can say a mark is big, common, sharply clustered and unlike its
neighbours, and none of them can say it is *interesting* — that this is the kind
of stamp somebody would actually search for.  Take the table as the order to
review in, not the answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cluster_marks as _cluster  # noqa: E402
import docmarks_config as cfg  # noqa: E402
import roster as _roster  # noqa: E402
from sources._common import Page, read_manifest  # noqa: E402

#: How the signals combine into one score.  Deliberately simple and flat: a
#: cleverer weighting would imply a precision the proxies do not have.
WEIGHTS = {
    "instances": 0.30,  # more instances = more retrievable, more to learn from
    "size": 0.25,  # bigger marks clear the ~32px structural floor
    "tightness": 0.20,  # a sharply clustered class is probably really one mark
    "separation": 0.25,  # far from its nearest neighbour = unambiguous label
}


def _norm(values: np.ndarray) -> np.ndarray:
    """Min-max to ``[0, 1]``; a flat vector scores 0.5 rather than dividing by 0."""
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.full_like(values, 0.5, dtype=float)
    return (values - lo) / (hi - lo)


def score_candidates(
    pages: Sequence[Page],
    classes: dict[str, dict[str, Any]],
    *,
    backend: str = "phash",
) -> list[dict[str, Any]]:
    """Rank every class, returning one record each, best first."""
    class_ids = sorted(classes)
    if not class_ids:
        return []

    by_id = {p.page_id: p for p in pages}
    exemplar_desc: list[np.ndarray] = []
    intra: list[float] = []
    rows: list[dict[str, Any]] = []

    for class_id in class_ids:
        meta = classes[class_id]
        descs = []
        for page_id in meta["page_ids"][:24]:
            page = by_id.get(page_id)
            if page is None:
                continue
            for mark in page.marks:
                if mark.class_id == class_id:
                    from PIL import Image

                    with Image.open(page.path) as im:
                        descs.append(_cluster.phash(_cluster.crop_mark(im.convert("L"), mark.box)))
                    break
        if not descs:
            continue
        block = np.array(descs, dtype=bool)
        exemplar_desc.append(block[0])
        # Mean pairwise Hamming inside the class: low means the instances really
        # do look like each other, which is weak evidence the cluster is one mark.
        if len(block) > 1:
            bits = block.astype(np.uint8)
            inv = 1 - bits
            d = (bits @ inv.T + inv @ bits.T) / block.shape[1]
            intra.append(float(d[np.triu_indices(len(block), k=1)].mean()))
        else:
            intra.append(1.0)
        rows.append(
            {
                "class_id": class_id,
                "source": meta.get("source", class_id.split("/", 1)[0]),
                "kind": meta.get("kind", "?"),
                "n_instances": meta["n_instances"],
                "median_mark_px": meta.get("median_mark_px"),
                "located_by": meta.get("located_by", "box"),
            }
        )

    if not rows:
        return []

    # Nearest-other-class distance: a class that sits far from every other one
    # has an unambiguous label; a close pair is a coin flip until adjudicated.
    exemplars = np.array(exemplar_desc, dtype=bool)
    bits = exemplars.astype(np.uint8)
    inv = 1 - bits
    between = (bits @ inv.T + inv @ bits.T).astype(float) / exemplars.shape[1]
    np.fill_diagonal(between, np.inf)
    nearest = between.min(axis=1)
    nearest_idx = between.argmin(axis=1)

    instances = np.array([r["n_instances"] for r in rows], dtype=float)
    sizes = np.array([r["median_mark_px"] or 0 for r in rows], dtype=float)
    tightness = 1.0 - np.array(intra, dtype=float)

    scores = (
        WEIGHTS["instances"] * _norm(np.log1p(instances))
        + WEIGHTS["size"] * _norm(np.log1p(sizes))
        + WEIGHTS["tightness"] * _norm(tightness)
        + WEIGHTS["separation"] * _norm(nearest)
    )

    for i, row in enumerate(rows):
        row["intra_distance"] = round(float(intra[i]), 4)
        row["nearest_other"] = rows[int(nearest_idx[i])]["class_id"]
        row["nearest_distance"] = round(float(nearest[i]), 4)
        row["score"] = round(float(scores[i]), 4)

    return sorted(rows, key=lambda r: -r["score"])


def render_sheet(candidates: Sequence[dict[str, Any]], classes: dict[str, Any], out_path: Path) -> None:
    """One numbered contact sheet of exemplars, in ranked order."""
    from PIL import Image, ImageDraw

    thumbs = []
    for rank, cand in enumerate(candidates):
        crop = classes[cand["class_id"]].get("query_crop")
        if crop and Path(crop).exists():
            thumbs.append((rank, cand, Image.open(crop)))
    if not thumbs:
        return

    cols, size, pad, cap = 8, 170, 8, 26
    n_rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (size + pad) + pad, n_rows * (size + pad + cap) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 8), "DocMarks roster shortlist — ranked; pick the ones worth verifying", fill="black")

    for i, (rank, cand, im) in enumerate(thumbs):
        r, c = divmod(i, cols)
        thumb = im.convert("RGB").copy()
        thumb.thumbnail((size, size))
        x, y = pad + c * (size + pad), 30 + r * (size + pad + cap)
        sheet.paste(thumb, (x, y))
        draw.rectangle([x, y, x + thumb.width, y + thumb.height], outline="#bbbbbb")
        draw.text((x, y + thumb.height + 2), f"#{rank}  {cand['class_id'].split('/')[-1]}"[:30], fill="#222222")
        draw.text(
            (x, y + thumb.height + 13),
            f"n={cand['n_instances']} px={cand['median_mark_px']} sep={cand['nearest_distance']:.2f}",
            fill="#666666",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--top", type=int, default=48, help="how many candidates to table and sheet")
    ap.add_argument("--roster-size", type=int, default=24)
    ap.add_argument("--write-roster", action="store_true", help="draft roster.json from the top candidates")
    ap.add_argument("--name", default="spods-v1", help="roster name")
    args = ap.parse_args(argv)

    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    pages = list(read_manifest(args.corpus / "corpus.jsonl"))

    candidates = score_candidates(pages, classes)
    if not candidates:
        print("no candidate classes — has the corpus been built?")
        return 1

    print(f"{len(candidates)} candidate class(es); top {min(args.top, len(candidates))}:\n")
    print(f"  {'#':>3}  {'score':>5}  {'n':>4}  {'px':>5}  {'sep':>5}  class / nearest other")
    for rank, c in enumerate(candidates[: args.top]):
        px = c["median_mark_px"] if c["median_mark_px"] is not None else "band"
        print(
            f"  {rank:>3}  {c['score']:>5.3f}  {c['n_instances']:>4}  {str(px):>5}  "
            f"{c['nearest_distance']:>5.2f}  {c['class_id']}  ~ {c['nearest_other'].split('/')[-1]}"
        )

    sheet = args.corpus / "shortlist.png"
    render_sheet(candidates[: args.top], classes, sheet)
    (args.corpus / "shortlist.json").write_text(json.dumps(candidates, indent=2) + "\n", encoding="utf-8")
    print(f"\nsheet  {sheet}\ntable  {args.corpus / 'shortlist.json'}")

    if args.write_roster:
        path = args.corpus / "roster.json"
        if path.exists():
            print(f"\n{path} already exists — not overwriting a roster someone has edited")
            return 0
        _roster.save(_roster.starter(args.name, candidates, size=args.roster_size), path)
        print(f"\ndrafted {path} with the top {args.roster_size} — edit it by hand before verifying")
    else:
        print("\npick from the sheet, then write roster.json (or re-run with --write-roster for a draft)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
