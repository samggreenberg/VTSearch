"""Review banded UCSF pages into known negatives for the four UCSF classes (#4088).

At v4.1 every positive of a UCSF roster class sits on a *banded* page -- a
single-page tobacco-company letter the letterhead pull selected -- and the only
other banded pages in its pool are the handful a person rejected.  So a control
that ignores the mark and ranks UCSF Tobacco pages first scores AP 0.53-0.97:
a method can win by recognising "tobacco letter" rather than the mark.  The
anchor classes do not have this, because ``own_verified`` adds their whole
exhaustively checked source as known negatives.

The fix is the same thing done by hand: show a person banded pages, one class
at a time, and ask whether the mark is on the page.

* **Bad** -> the page joins the class's ``reviewed_negative_page_ids``: a
  same-style negative, which is exactly what the pool lacks.
* **Good** -> the page carries the mark and becomes a positive, located by its
  letterhead band (``PROVENANCE_BAND``: the mark is on the page, not boxed).
  The band classes miss most of their own mark (#3922), so this review also
  completes the positives.

The draw is **uniform** over the banded frame, not ranked by any detector, so
the new negatives are not selected for being easy for one.  Each class gets as
many pages as it takes to bring the mark-blind control below
:data:`TARGET_AP` at the tier the frame is drawn from, sized by simulation
(:func:`pages_needed`).  Planted known positives -- about one per
:data:`QUESTIONS_PER_CONTROL` -- check that the reviewer is looking.

Nothing here writes to the corpus.  ``emit`` renders the queues and a verdict
slate in the ``ucsf_classes`` format; ``binary_review.py bank`` fills it; and
``audit_to_corrections.py --task ucsf_classes --audit-dir <banked dir>`` applies
it through the same applier the classes were admitted with.

    python banded_review.py --out <queue root>     # dry: prints the plan
    python banded_review.py --out <queue root> --emit
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fullmarks_config as cfg  # noqa: E402
from binary_review import PREFIX, Question, class_refs, emit  # noqa: E402
from contamination_sample import UCSF_CLASSES, controls_for  # noqa: E402
from sources._common import read_manifest  # noqa: E402

TASK = "ucsf_banded"
#: Where the verdict slate lives, under ``<corpus>/audit/``.
AUDIT_DIR = "ucsf_banded"
#: The ``part`` stamped on every slate row, so an applied row says which pass it came from.
PART = "banded_4088"
#: The mark-blind control must score below this after the review.  #4088 closes
#: at 0.2; aiming lower leaves room for Good votes, which move a page from the
#: negatives it was drawn to be into the positives.
TARGET_AP = 0.15
QUESTIONS_PER_CONTROL = 40
SEED = 20260922


def shortcut_ap(n_pos: int, n_neg: int, reps: int = 200, seed: int = 0) -> float:
    """Expected AP of a control that ranks every banded page first, in random order.

    Only the top block matters: the positives are all banded, so AP depends on
    how many banded negatives share the block with them and nothing else.
    """
    if n_pos == 0:
        return 0.0
    rng = random.Random(seed)
    total = 0.0
    for _ in range(reps):
        slots = sorted(rng.sample(range(n_pos + n_neg), n_pos))
        total += sum((i + 1) / (rank + 1) for i, rank in enumerate(slots)) / n_pos
    return total / reps


def pages_needed(n_pos: int, n_neg: int, target: float = TARGET_AP, cap: Optional[int] = None) -> int:
    """Fewest extra banded negatives that bring :func:`shortcut_ap` below *target*."""
    if n_pos == 0 or shortcut_ap(n_pos, n_neg) < target:
        return 0
    lo, hi = 0, max(1, math.ceil(n_pos / target))
    while shortcut_ap(n_pos, n_neg + hi) >= target:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if shortcut_ap(n_pos, n_neg + mid) < target:
            hi = mid
        else:
            lo = mid + 1
    return lo if cap is None else min(lo, cap)


def band_box(page: Any) -> Optional[list[int]]:
    """The letterhead band the pull recorded on *page*: its unclassed ``candidate`` mark."""
    for m in page.marks:
        if m.class_id is None and m.provenance == "candidate":
            return [int(v) for v in m.box]
    return None


def banded_frame(class_id: str, meta: dict[str, Any], pages: dict[str, Any], tiers: set[str]) -> list[str]:
    """Banded UCSF pages in *tiers* that nobody has ruled on for this class."""
    ruled = (
        set(meta.get("page_ids", []))
        | set(meta.get("reviewed_negative_page_ids", []))
        | set(meta.get("excluded_page_ids", []))
        | {meta.get("query_page_id")}
    )
    return sorted(
        pid
        for pid, page in pages.items()
        if page.source == "ucsf"
        and (page.meta or {}).get("letterhead_author")
        and (page.meta or {}).get("tier") in tiers
        and pid not in ruled
        and band_box(page) is not None
    )


def proposal_of(class_id: str) -> str:
    return class_id.split("logo_", 1)[1]


def queue_name(class_id: str) -> str:
    return f"{PREFIX} {class_id.split('/')[-1]} -- is this mark on this tobacco letter?"


def question(class_id: str, i: int, pid: str, arm: str, page: Any, refs, stem: Optional[str] = None) -> Question:
    """A whole-page "is the mark here?" question.  *stem* names the file; the default is this pass's."""
    fn = f"{stem or 'banded__' + proposal_of(class_id)}__{i:04d}.jpg"
    return Question(
        filename=fn,
        task=TASK,
        question="Is the mark on the left anywhere on this page?",
        refs=refs,
        page_id=pid,
        box=[0, 0, page.width, page.height],
        outline=False,
        # translate_ucsf reads sheet/cell; class and arm are for the report.
        key={"sheet": fn, "cell": 0, "n_cells": 1, "class_id": class_id, "page_id": pid, "arm": arm},
        # Same sheet as the contamination queues, which were labelled at a page a
        # second: portrait canvas, greyscale q80, and a footer with no tell.
        canvas=[1530, 1350],
        greyscale=True,
        quality=80,
        # Crop to the inked region (Sam, 2026-09-22: "this looks uncropped"): a
        # scan's white border is flecked with specks, so trimming solid borders
        # alone left the page at full size.  No captions: the reviewer compares
        # marks, and file names under the references were wasted space.
        trim_border=True,
        crop_to_ink=True,
        ref_labels=False,
        anonymous=True,
        detail="",
    )


def slate_row(class_id: str, q: Question, index: int, page: Any) -> dict[str, Any]:
    """One single-cell ``ucsf_classes`` sheet, so the existing applier takes it unchanged."""
    return {
        "task": "ucsf_classes",
        "proposal": proposal_of(class_id),
        "part": PART,
        "sheet": q.filename,
        "sheet_index": index,
        "n_sheets": None,
        "cells": [{"index": 0, "page_id": q.page_id, "inliers": 0, "box": band_box(page), "located": False}],
        "arm": q.key["arm"],
        "verdict": "",
    }


def relation_rows(corpus: Path) -> list[dict[str, Any]]:
    """The owner's rulings that these four proposals are new classes, carried forward.

    ``apply_ucsf_classes`` needs each proposal's relation to know which class a
    sheet belongs to.  They were ruled on 2026-09-19 and are copied, not re-asked.
    """
    path = corpus / "audit" / "ucsf_classes_banked" / "verdicts.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    want = {proposal_of(c) for c in UCSF_CLASSES}
    out = [dict(r) for r in rows if r.get("task") == "ucsf_classes_relation" and r["proposal"] in want]
    missing = want - {r["proposal"] for r in out}
    if missing or any(r.get("relation") != "new" for r in out):
        raise SystemExit(f"relation rows not all ruled 'new': missing {sorted(missing)}")
    return out


def plan(
    classes: dict[str, Any], pages: dict[str, Any], tier: str, target: float
) -> list[tuple[str, int, int, int, list[str]]]:
    """(class, positives, banded negatives, pages to draw, frame) per class, at *tier*."""
    tiers = set(cfg.TIER_ORDER[: cfg.TIER_ORDER.index(tier) + 1])
    out = []
    for cid in UCSF_CLASSES:
        meta = classes[cid]
        in_tier = {p for p, pg in pages.items() if (pg.meta or {}).get("tier") in tiers}
        pos = {p for p in meta.get("page_ids", []) if p in in_tier} - {meta.get("query_page_id")}
        negs = [
            p
            for p in meta.get("reviewed_negative_page_ids", [])
            if p in in_tier and (pages[p].meta or {}).get("industry") == "Tobacco"
        ]
        frame = banded_frame(cid, meta, pages, tiers)
        n = pages_needed(len(pos), len(negs), target, cap=len(frame))
        out.append((cid, len(pos), len(negs), n, frame))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--out", type=Path, required=True, help="queue root; one directory per class")
    ap.add_argument("--tier", default="m")
    ap.add_argument("--target", type=float, default=TARGET_AP)
    ap.add_argument("--emit", action="store_true", help="render the queues and write the slate (default: plan only)")
    args = ap.parse_args(argv)

    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    pages = {p.page_id: p for p in read_manifest(args.corpus / "corpus.jsonl")}
    rng = random.Random(SEED)

    steps = plan(classes, pages, args.tier, args.target)
    for cid, n_pos, n_neg, n, frame in steps:
        after = shortcut_ap(n_pos, n_neg + n)
        print(
            f"  {cid}: {n_pos} positives, {n_neg} banded negatives, control AP {shortcut_ap(n_pos, n_neg):.2f}"
            f" -> {after:.2f} with {n} of {len(frame)} banded pages"
        )
    if not args.emit:
        print("plan only -- pass --emit to render")
        return 0

    slate_dir = args.corpus / "audit" / AUDIT_DIR
    slate = slate_dir / "verdicts.jsonl"
    if slate.exists():
        raise SystemExit(f"{slate} exists; it may hold answers, so it is never overwritten")
    rows = relation_rows(args.corpus)
    for cid, _n_pos, _n_neg, n, frame in steps:
        drawn = rng.sample(frame, n)
        refs = class_refs(cid, classes, pages)
        n_ctrl = max(2, round(n / QUESTIONS_PER_CONTROL))
        controls = controls_for(cid, classes, pages, refs, rng, n_ctrl)
        items = [(p, "uniform") for p in drawn] + [(p, "control") for p in controls]
        rng.shuffle(items)
        qs = [question(cid, i, pid, arm, pages[pid], refs) for i, (pid, arm) in enumerate(items)]
        rows += [slate_row(cid, q, i, pages[q.page_id]) for i, q in enumerate(qs)]
        qdir = args.out / (cid.replace("/", "_") + "__banded")
        emit(qs, qdir, queue_name(cid), pages=pages, corpus=args.corpus)
        print(f"  {cid}: {len(qs)} question(s), {len(controls)} planted -> {qdir}", flush=True)
    slate_dir.mkdir(parents=True, exist_ok=True)
    slate.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"slate: {slate} ({len(rows)} rows); seed {SEED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
