"""A second completeness pass, proposed by matchers that are not SIFT.

The first pass (#3927) found 108 missing members, every one proposed by SIFT.
The members still missing are therefore the ones SIFT cannot see -- too few
keypoints, a faint or smeared impression, a copy at a scale the detector misses
-- and any benchmark comparison of SIFT against another method inherits that
tilt.  This pass takes proposals from independent techniques, removes everything
already decided, and puts the remainder in front of a person in the same shape
as the first pass, so ``audit_to_corrections.py --task completeness --audit-dir
completeness2`` applies the answers unchanged.

**The interface between proposers and this module is one JSON file per method**::

    <proposals dir>/proposals-<method>.json
    {"method": "siglip_tiles",
     "classes": {"<class_id>": [["<page_id>", <score>, [x, y, w, h] | null], ...]}}

ranked best first, scores higher-is-better and comparable only within one list,
boxes in page pixels.  A proposal without a box cannot be shown in context or
added as a new mark, so it is counted and dropped.

:func:`merge_proposals` drops, per class:

* pages outside the anchor sources (UCSF is distractor-only);
* member pages, and proposals landing on a mark already labelled this class;
* marks already cannot-linked to the class (``adjudications.json`` ``different``);
* pages a previous pass rejected as carrying no box
  (``audit.completeness_checked*.unboxed_rejected_page_ids``);
* candidates a previous slate already put in front of a reviewer
  (its ``verdicts.jsonl`` rows with a verdict), matched on page and mark.

What is left is unioned by (page, overlapping mark) -- or by page for a match on
no mark -- remembering which methods proposed it, and ranked by **how many
methods agree**, then by the best rank any of them gave it, as a fraction of that
method's list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from completeness import ANCHOR_SOURCES, Candidate, ClassCandidates, overlapping_mark
from sources._common import Page

#: Candidates per class; two one-screen sheets at completeness.PER_SHEET.
TOP = 36
AUDIT_DIR = "completeness2"
PASS = "multi"
#: Short names for sheet captions, which have room for about 30 characters.
CODES = {"sift": "SIFT", "siglip_tiles": "SIG", "dinov3_patches": "DINO", "ocr": "OCR", "ncc": "NCC"}


def proposals_path(directory: Path, method: str) -> Path:
    return directory / f"proposals-{method}.json"


def save_proposals(
    method: str, by_class: dict[str, list[tuple[str, float, Optional[Sequence[int]]]]], path: Path
) -> None:
    """Write one method's ranked proposals in the shared format."""
    classes = {
        cid: [[p, float(s), [int(v) for v in box] if box is not None else None] for p, s, box in rows]
        for cid, rows in sorted(by_class.items())
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"method": method, "classes": classes}) + "\n", encoding="utf-8")


def load_proposals(paths: Sequence[Path]) -> dict[str, dict[str, list[list[Any]]]]:
    out: dict[str, dict[str, list[list[Any]]]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        method = data["method"]
        if method in out:
            raise ValueError(f"{path}: method {method!r} given twice")
        out[method] = data["classes"]
    return out


@dataclass
class Decided:
    """What earlier passes settled for one class."""

    rejected_marks: set[tuple[str, int]] = field(default_factory=set)
    rejected_unboxed_pages: set[str] = field(default_factory=set)
    reviewed_marks: set[tuple[str, int]] = field(default_factory=set)
    reviewed_unboxed_pages: set[str] = field(default_factory=set)


def decided(
    classes: dict[str, Any],
    pages: dict[str, Page],
    separations: Sequence[dict[str, Any]],
    reviewed_rows: Sequence[dict[str, Any]] = (),
) -> dict[str, Decided]:
    out = {cid: Decided() for cid in classes}
    for row in separations:
        cid = row.get("right_class_id")
        if cid is None:
            page, idx = pages.get(row.get("right_page_id", "")), row.get("right_mark_index")
            if page is not None and idx is not None and idx < len(page.marks):
                cid = page.marks[idx].class_id
        if cid in out and row.get("left_mark_index") is not None:
            out[cid].rejected_marks.add((row["left_page_id"], int(row["left_mark_index"])))
    for cid, meta in classes.items():
        for key, value in meta.get("audit", {}).items():
            if key.startswith("completeness_checked") and isinstance(value, dict):
                out[cid].rejected_unboxed_pages.update(value.get("unboxed_rejected_page_ids", []))
    for row in reviewed_rows:
        cid = row.get("class_id")
        if cid not in out or not str(row.get("verdict", "")).strip():
            continue
        for cand in row.get("candidates", []):
            if cand.get("mark_index") is None:
                out[cid].reviewed_unboxed_pages.add(cand["page_id"])
            else:
                out[cid].reviewed_marks.add((cand["page_id"], int(cand["mark_index"])))
    return out


@dataclass
class _Merged:
    page_id: str
    mark_index: Optional[int]
    box: tuple[int, int, int, int]
    best_frac: float
    methods: dict[str, float] = field(default_factory=dict)


def merge_proposals(
    classes: dict[str, Any],
    pages: dict[str, Page],
    proposals: dict[str, dict[str, list[list[Any]]]],
    done: dict[str, Decided],
    *,
    top: int = TOP,
    stats: Optional[dict[str, dict[str, int]]] = None,
) -> list[ClassCandidates]:
    """Per roster class, undecided anchor-page candidates ranked by method agreement.

    *stats*, if given, is filled per method with ``proposed`` (rows read),
    ``no_box``, ``decided`` (dropped as members or already ruled on) and
    ``novel`` (kept before the *top* cap).
    """
    out = []
    for method in proposals:
        if stats is not None:
            stats.setdefault(method, {"proposed": 0, "no_box": 0, "decided": 0, "novel": 0})
    for class_id, meta in sorted(classes.items()):
        if not meta.get("on_roster"):
            continue
        members = set(meta.get("page_ids", []))
        known = done.get(class_id, Decided())
        merged: dict[tuple[str, Optional[int]], _Merged] = {}
        for method, by_class in sorted(proposals.items()):
            rows = by_class.get(class_id, [])
            st = stats[method] if stats is not None else {"proposed": 0, "no_box": 0, "decided": 0, "novel": 0}
            for rank, row in enumerate(rows):
                page_id, _score, box = row[0], row[1], row[2]
                st["proposed"] += 1
                page = pages.get(page_id)
                if page is None or page.source not in ANCHOR_SOURCES:
                    st["decided"] += 1
                    continue
                if box is None:
                    st["no_box"] += 1
                    continue
                box = tuple(int(v) for v in box)
                idx = overlapping_mark(page, box)
                if page_id in members:
                    st["decided"] += 1
                    continue
                if idx is not None:
                    mark_key = (page_id, idx)
                    if (
                        page.marks[idx].class_id == class_id
                        or mark_key in known.rejected_marks
                        or mark_key in known.reviewed_marks
                    ):
                        st["decided"] += 1
                        continue
                elif page_id in known.rejected_unboxed_pages or page_id in known.reviewed_unboxed_pages:
                    st["decided"] += 1
                    continue
                st["novel"] += 1
                frac = rank / max(1, len(rows))
                key = (page_id, idx)
                entry = merged.get(key)
                if entry is None:
                    merged[key] = _Merged(page_id, idx, box, frac, {method: frac})
                else:
                    entry.methods[method] = min(frac, entry.methods.get(method, frac))
                    if frac < entry.best_frac:
                        entry.best_frac, entry.box = frac, box
        ranked = sorted(merged.values(), key=lambda m: (-len(m.methods), m.best_frac, m.page_id))[:top]
        cands = []
        for m in ranked:
            page = pages[m.page_id]
            mark = page.marks[m.mark_index] if m.mark_index is not None else None
            cands.append(
                Candidate(
                    page_id=m.page_id,
                    inliers=0,
                    box=m.box,
                    mark_index=m.mark_index,
                    mark_class_id=mark.class_id if mark else None,
                    mark_kind=mark.kind if mark else None,
                    note="+".join(CODES.get(k, k) for k in sorted(m.methods, key=lambda k: m.methods[k])),
                )
            )
        out.append(ClassCandidates(class_id=class_id, candidates=cands))
    return out


def verdict_row(entry: ClassCandidates) -> dict[str, Any]:
    """The completeness row, plus which methods proposed each candidate."""
    from completeness import verdict_row as base  # noqa: PLC0415

    row = base(entry)
    row["pass"] = PASS
    for out, cand in zip(row["candidates"], entry.candidates):
        out["methods"] = cand.note
    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse  # noqa: PLC0415

    import docmarks_config as cfg  # noqa: PLC0415
    from cluster_marks import load_adjudication_rows  # noqa: PLC0415
    from completeness import render, write_template  # noqa: PLC0415
    from sources._common import read_manifest  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--proposals", type=Path, nargs="+", required=True, help="proposals-<method>.json files")
    ap.add_argument(
        "--reviewed",
        type=Path,
        nargs="*",
        default=None,
        help="earlier slates' verdicts.jsonl (default: the SIFT pass, audit/completeness/verdicts.jsonl)",
    )
    ap.add_argument("--top", type=int, default=TOP)
    ap.add_argument("--audit-dir", default=AUDIT_DIR, help="subdirectory of <corpus>/audit to write the slate to")
    args = ap.parse_args(argv)

    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    pages = {p.page_id: p for p in read_manifest(args.corpus / "corpus.jsonl")}
    _same, different = load_adjudication_rows(args.corpus / "adjudications.json")
    reviewed_paths = (
        args.reviewed if args.reviewed is not None else [args.corpus / "audit" / "completeness" / "verdicts.jsonl"]
    )
    reviewed_rows = [
        json.loads(line)
        for path in reviewed_paths
        if path.exists()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    proposals = load_proposals(args.proposals)
    stats: dict[str, dict[str, int]] = {}
    entries = merge_proposals(
        classes, pages, proposals, decided(classes, pages, different, reviewed_rows), top=args.top, stats=stats
    )

    out = args.corpus / "audit" / args.audit_dir
    rows = []
    for entry in entries:
        sheets = render(entry, classes, pages, out) if entry.candidates else []
        rows.append(verdict_row(entry))
        agree = sum(1 for c in entry.candidates if "+" in (c.note or ""))
        print(
            f"  {entry.class_id}: {len(entry.candidates)} candidate(s), {agree} proposed by 2+ methods, {len(sheets)} sheet(s)",
            flush=True,
        )
    write_template(rows, out / "verdicts.jsonl")
    (out / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for method, st in sorted(stats.items()):
        print(f"  {method}: {st}")
    print(f"\n{sum(len(e.candidates) for e in entries)} candidate(s) over {len(entries)} class(es) -> {out}")
    print(
        f"  python audit_to_corrections.py --task completeness --audit-dir {args.audit_dir} --reviewer <name>   # dry run"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
