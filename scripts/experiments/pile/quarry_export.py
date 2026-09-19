#!/usr/bin/env python3
"""Cut a study-ready set out of `coco_quarry_full`: any prevalence, per-class negatives.

`SCALE_PREVALENCE`, `SCALE_N_POS` and `SCALE_N_NEG` were fixed when a pile was
built, so moving any of them meant a rebuild and a re-embed (#3987). Over an
exhaustively annotated corpus every ``(image, class)`` pair is answered, and once
the whole corpus is embedded (#4017, #4022) a cell stops being a designation and
becomes a **filter**. This is that filter.

**Nothing here embeds anything.** The output is a manifest of media ids; a study
loads the full-corpus cell it wants and keeps those ids. Prevalence becomes a
sampling parameter rather than a property of the pile.

## Two pools, and the default is the one #3986 argues for

A shared-pool negative holds **no class in C at all**, which is stricter than
scoring needs -- a negative for `book` need only lack `book` -- and it draws an
unrepresentative sample: such an image is barren by construction, so a detector
scored against it answers an easier question than the app poses, and can learn
*an image containing a B cannot contain an A* (#3667 measured that shortcut at
1.88 on its FPR-inflation scale).

Per-class, the negatives for `book@small` are every image the cell can be scored
against that is not one of its positives -- which is what ``evaluable_categories``
already records. Measured on the built cells: **~100,000 negatives per cell
against the designated 9,900, about half of them holding another class in C**,
where the shared pool is 0% by construction.

``--pool shared`` reproduces the old shape for an arm that needs to be compared
against a published number.

## Prevalence is a ratio, so the envelope is two-dimensional

N = P(1-pi)/pi. The corpus caps N at ~100k per cell, so a target pi and a target
positive count cannot both be chosen freely: `car@large` at pi=0.001 would need
1.29M negatives against 96,458 available, while the same cell at 100 positives
needs 99,900 and fits. ``--envelope`` prints what each cell can actually reach,
and an infeasible request is refused with the two numbers that would make it
feasible rather than a silently thinner set.

## The draw reproduces the designated build

Positives are ordered by :func:`~pilebuild.loaders.vg_scale.rank`, the same
hash-of-(cell, image_id) the builder uses, so an export at ``SCALE_N_POS`` with
no other constraint selects the same images `coco_quarry` designated. That is
what makes a query-time cell comparable with a build-time one rather than merely
similar, and it is pinned by a test.

    python quarry_export.py --cell bus@small --positives 100 --prevalence 0.01
    python quarry_export.py --envelope
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "calibration"))

import pile_config as pc  # noqa: E402
from pilebuild.loaders.vg_scale import rank  # noqa: E402

#: The cheapest full-corpus column to read membership from. Cell membership is
#: identical across columns -- it comes from the annotations, not the vectors --
#: so there is no reason to page in a patch shard to answer "which images".
MEMBERSHIP_CELL = "coco_quarry_full__clip.pkl"

POOLS = ("per_class", "shared")


def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def load_membership(path: Path | None = None) -> dict[int, dict]:
    """``{image_id: {"cells": [...], "evaluable": [...]}}`` for the whole corpus."""
    from _cells_io import load_medias  # noqa: PLC0415

    src = path or (pc.EMBEDDINGS / MEMBERSHIP_CELL)
    if not src.exists():
        raise SystemExit(f"missing {src}; build coco_quarry_full first (#4017)")
    return {
        iid: {"cells": list(d.get("categories") or []), "evaluable": list(d.get("evaluable_categories") or [])}
        for iid, d in load_medias(src).items()
    }


def candidates(members: dict[int, dict], cell: str, pool: str) -> tuple[list[int], list[int]]:
    """``(positives, negatives)`` available for *cell*, before any sampling."""
    if pool not in POOLS:
        raise SystemExit(f"unknown pool {pool!r}; known: {', '.join(POOLS)}")
    pos = [i for i, m in members.items() if cell in m["cells"]]
    if pool == "shared":
        # The old shape: an image holding nothing in C at all.
        neg = [i for i, m in members.items() if not m["cells"]]
    else:
        pos_set = set(pos)
        neg = [i for i, m in members.items() if cell in m["evaluable"] and i not in pos_set]
    return pos, neg


def negatives_for(n_pos: int, prevalence: float) -> int:
    """N = P(1-pi)/pi, the count that puts *n_pos* positives at *prevalence*."""
    if not 0 < prevalence < 1:
        raise SystemExit(f"prevalence must be in (0, 1), got {prevalence}")
    return round(n_pos * (1 - prevalence) / prevalence)


def select(members: dict[int, dict], cell: str, n_pos: int, prevalence: float, pool: str = "per_class") -> dict:
    """One study-ready selection, or a refusal naming what would make it fit."""
    pos, neg = candidates(members, cell, pool)
    if n_pos > len(pos):
        raise SystemExit(f"{cell}: asked for {n_pos:,} positives, {len(pos):,} exist")
    want_neg = negatives_for(n_pos, prevalence)
    if want_neg > len(neg):
        # Both ways out, because which one is acceptable is the caller's to judge.
        max_pos = max(1, int(len(neg) * prevalence / (1 - prevalence)))
        max_pi = n_pos / (n_pos + len(neg))
        raise SystemExit(
            f"{cell}: {n_pos:,} positives at pi={prevalence} needs {want_neg:,} negatives and "
            f"{len(neg):,} are available ({pool} pool) -- SHORT {want_neg - len(neg):,}. "
            f"Negatives are the scarce side and every extra positive costs "
            f"{round((1 - prevalence) / prevalence):,} more of them, so adding positives makes "
            f"this worse, not better. Either drop to {max_pos:,} positives at this pi, or raise "
            f"pi to {max_pi:.4g} at this positive count."
        )
    # Hash-ordered, so the draw is stable when the corpus changes and an export
    # at SCALE_N_POS reproduces what `designate_cells` chose (#3726).
    chosen_pos = sorted(sorted(pos, key=lambda i: rank(cell, i))[:n_pos])
    neg_key = "__negatives__" if pool == "shared" else f"{cell}:negatives"
    chosen_neg = sorted(sorted(neg, key=lambda i: rank(neg_key, i))[:want_neg])
    return {
        "cell": cell,
        "pool": pool,
        "prevalence": prevalence,
        "n_positives": len(chosen_pos),
        "n_negatives": len(chosen_neg),
        "available_positives": len(pos),
        "available_negatives": len(neg),
        "positives": chosen_pos,
        "negatives": chosen_neg,
    }


def envelope(members: dict[int, dict], pool: str) -> list[dict]:
    """Per cell: what is available, and the rarest pi reachable at 100 positives."""
    cells = sorted({c for m in members.values() for c in m["cells"]})
    rows = []
    for cell in cells:
        pos, neg = candidates(members, cell, pool)
        rows.append(
            {
                "cell": cell,
                "positives": len(pos),
                "negatives": len(neg),
                "min_pi_at_100": 100 / (100 + len(neg)) if neg else None,
                "min_pi_at_all_positives": len(pos) / (len(pos) + len(neg)) if neg else None,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", help="e.g. bus@small")
    ap.add_argument("--positives", type=int, default=pc.SCALE_N_POS)
    ap.add_argument("--prevalence", type=float, default=pc.SCALE_PREVALENCE)
    ap.add_argument("--pool", default="per_class", choices=POOLS)
    ap.add_argument("--out", type=Path, help="write the manifest here (default: stdout summary only)")
    ap.add_argument("--envelope", action="store_true", help="what every cell can reach, then exit")
    ap.add_argument("--membership", type=Path, help="read membership from this cell instead")
    args = ap.parse_args()

    members = load_membership(args.membership)
    log(f"membership over {len(members):,} images")

    if args.envelope:
        rows = envelope(members, args.pool)
        print(f"\n{'cell':<20}{'pos':>8}{'neg':>10}{'min pi @100':>13}{'min pi @all pos':>17}")
        for r in rows:
            lo = f"{r['min_pi_at_100']:.5f}" if r["min_pi_at_100"] else "--"
            la = f"{r['min_pi_at_all_positives']:.5f}" if r["min_pi_at_all_positives"] else "--"
            print(f"{r['cell']:<20}{r['positives']:>8,}{r['negatives']:>10,}{lo:>13}{la:>17}")
        return 0

    if not args.cell:
        raise SystemExit("--cell is required unless --envelope")
    sel = select(members, args.cell, args.positives, args.prevalence, args.pool)
    log(
        f"{sel['cell']}: {sel['n_positives']:,} positives + {sel['n_negatives']:,} negatives "
        f"at pi={sel['prevalence']} ({sel['pool']} pool; "
        f"{sel['available_positives']:,}/{sel['available_negatives']:,} available)"
    )
    if args.out:
        args.out.write_text(json.dumps(sel, indent=1) + "\n")
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
