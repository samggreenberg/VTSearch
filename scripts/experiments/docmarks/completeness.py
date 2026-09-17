"""The completeness pass: which pages carry a roster mark that the labels miss?

``membership`` checks every instance a class *has*; nothing checked for instances
it *lacks*.  #3927 found them on Tobacco800 -- the B&W three-leaf mark on seven
pages filed under six other classes, and marks on pages with no box at all -- so
every such page is scored as a negative for the class it carries.

A strong instance matcher is the right tool to *propose* the missing members,
and a person is the right one to decide.  This module does both halves:

* :func:`candidates` reads an ``inliers.json`` from ``eval_sift_rank.py`` (SIFT
  verified against every page of a tier) and keeps, per roster class, the
  non-member anchor pages that match the query crop most strongly.
* :func:`render` re-verifies just those pages to recover *where* the match is,
  draws each as a crop around it -- with the page's existing mark outlined and
  its current label named -- and writes a ``verdicts.jsonl`` template.
* :func:`apply_completeness` folds the answers back.  An accepted candidate that
  overlaps an existing mark has that mark's ``class_id`` reassigned; one that
  overlaps none gains a new mark with the matched box and
  ``provenance="completeness"``, so a study can tell a hand-added box from a
  shipped one.  Accepted instances are must-linked to the class and rejected
  boxed marks cannot-linked, so a rebuild replays both; a rejected page with no
  mark is recorded on the class as a checked negative.

**A new box must survive a rebuild**, and marks are otherwise re-derived from the
sources, so hand-added marks live in their own store, ``added_marks.json``, which
``build_corpus.py`` replays onto the pages before identity clustering.  They are
appended in store order and unclassed, exactly as they were when the verdict was
applied, so the must-link recorded against ``(page_id, mark_index)`` binds the
same mark after a rebuild.

The candidate pool is **anchor pages only** (SPODS, StaVer, Tobacco800): every
anchor page is in tier ``s``, so one tier-``s`` run covers every page a roster
mark can be boxed on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional, Sequence

from sources._common import Mark, Page

#: Candidates shown per class, strongest first.  Enough to reach past the
#: confusers on round stamps (#3911 found 100+ same-source pages at >= 15 inliers
#: for some) without turning one class into an afternoon.
TOP = 30
#: A candidate needs at least the app's cold-start verification gate.
MIN_INLIERS = 8
#: An accepted match reuses an existing mark when this much of the match box
#: lies inside it; otherwise a new mark is added.
OVERLAP = 0.5
ANCHOR_SOURCES = ("spods", "staver", "tobacco800")
PROVENANCE = "completeness"
ADDED_MARKS = "added_marks.json"


@dataclass
class Candidate:
    page_id: str
    inliers: int
    box: Optional[tuple[int, int, int, int]] = None  # x, y, w, h in page pixels
    mark_index: Optional[int] = None
    mark_class_id: Optional[str] = None
    mark_kind: Optional[str] = None
    #: Caption override for the sheet (the multi-method pass names its proposers).
    note: Optional[str] = None


@dataclass
class ClassCandidates:
    class_id: str
    positive_inliers: list[int] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)


def candidates(
    classes: dict[str, Any],
    pages: dict[str, Page],
    inliers: dict[str, dict[str, list[int]]],
    *,
    top: int = TOP,
    min_inliers: int = MIN_INLIERS,
) -> list[ClassCandidates]:
    """Per roster class, the strongest-matching anchor pages that are not members."""
    out = []
    for class_id, meta in sorted(classes.items()):
        if not meta.get("on_roster") or class_id not in inliers:
            continue
        members = set(meta.get("page_ids", []))
        scores = inliers[class_id]
        ranked = sorted(
            (
                (v[0], p)
                for p, v in scores.items()
                if p not in members and v[0] >= min_inliers and p in pages and pages[p].source in ANCHOR_SOURCES
            ),
            key=lambda t: (-t[0], t[1]),
        )[:top]
        out.append(
            ClassCandidates(
                class_id=class_id,
                positive_inliers=sorted((scores.get(p, [0, 0])[0] for p in members), reverse=True),
                candidates=[Candidate(page_id=p, inliers=n) for n, p in ranked],
            )
        )
    return out


def overlapping_mark(page: Page, box: tuple[int, int, int, int]) -> Optional[int]:
    """Index of the non-signature mark holding most of *box*, if at least OVERLAP of it."""
    bx, by, bw, bh = box
    area = max(1, bw * bh)
    best, best_share = None, 0.0
    for i, mark in enumerate(page.marks):
        if mark.kind == "signature":
            continue
        mx, my, mw, mh = mark.box
        ix = max(0, min(bx + bw, mx + mw) - max(bx, mx))
        iy = max(0, min(by + bh, my + mh) - max(by, my))
        share = ix * iy / area
        if share > best_share:
            best, best_share = i, share
    return best if best_share >= OVERLAP else None


def locate(entry: ClassCandidates, pages: dict[str, Page], verify) -> None:
    """Fill each candidate's pixel box and overlapping mark.

    *verify* maps a page id to the match's normalised inlier box
    ``(x0, y0, x1, y1)`` or ``None`` -- the caller runs the matcher, so this stays
    testable without one.  A candidate whose match cannot be re-fitted is dropped:
    there is nothing to show a reviewer.
    """
    kept = []
    for cand in entry.candidates:
        norm = verify(cand.page_id)
        if norm is None:
            continue
        page = pages[cand.page_id]
        x0, y0, x1, y1 = norm
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        cand.box = (
            int(round(x0 * page.width)),
            int(round(y0 * page.height)),
            max(1, int(round((x1 - x0) * page.width))),
            max(1, int(round((y1 - y0) * page.height))),
        )
        idx = overlapping_mark(page, cand.box)
        if idx is not None:
            cand.mark_index = idx
            cand.mark_class_id = page.marks[idx].class_id
            cand.mark_kind = page.marks[idx].kind
        kept.append(cand)
    entry.candidates = kept


def load_added_marks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("marks", [])


def save_added_marks(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps({"marks": list(rows)}, indent=2) + "\n", encoding="utf-8")


def replay_added_marks(
    pages: Sequence[Page], rows: Sequence[dict[str, Any]], warnings: Optional[list[str]] = None
) -> int:
    """Append stored hand-added marks to their pages, unclassed, in store order.

    Clustering then assigns them, bound by the must-links the completeness pass
    recorded.  A row naming a page that is not in this build is reported, not
    fatal: a smaller ``--limit`` build legitimately lacks it.  A mark already on
    the page with the same box is not added twice.
    """
    by_id = {p.page_id: p for p in pages}
    added = 0
    for row in rows:
        page = by_id.get(row["page_id"])
        if page is None:
            if warnings is not None:
                warnings.append(f"added mark on {row['page_id']}: page not in this build")
            continue
        box = tuple(int(v) for v in row["box"])
        if any(tuple(m.box) == box for m in page.marks):
            continue
        page.marks.append(Mark(row.get("kind", "logo"), box, None, row.get("provenance", PROVENANCE)))
        added += 1
    return added


def verdict_row(entry: ClassCandidates) -> dict[str, Any]:
    return {
        "task": "completeness",
        "class_id": entry.class_id,
        "n_candidates": len(entry.candidates),
        "positive_inliers": entry.positive_inliers,
        "candidates": [
            {
                "index": i,
                "page_id": c.page_id,
                "inliers": c.inliers,
                "box": list(c.box) if c.box else None,
                "mark_index": c.mark_index,
                "mark_class_id": c.mark_class_id,
            }
            for i, c in enumerate(entry.candidates)
        ],
        # "none" = no candidate carries this class's mark. Otherwise the
        # comma-separated candidate numbers that DO (e.g. "0,3,7").
        "verdict": "",
        "notes": "",
    }


def _parse(raw: str, n: int) -> tuple[Optional[list[int]], Optional[str]]:
    raw = raw.strip().lower()
    if raw == "none":
        return [], None
    if not raw:
        return None, None  # not reviewed
    try:
        idx = sorted({int(tok) for tok in raw.replace(" ", "").split(",") if tok})
    except ValueError:
        return None, f"verdict must be 'none' or comma-separated candidate numbers, got {raw!r}"
    bad = [i for i in idx if not 0 <= i < n]
    if bad:
        return None, f"candidate number(s) {bad} outside 0..{n - 1}"
    return idx, None


def apply_completeness(
    pages: list[Page],
    classes: dict[str, Any],
    verdicts: Sequence[dict[str, Any]],
    *,
    reviewer: Optional[str] = None,
) -> tuple[list[str], list[str], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Add accepted candidates to their classes; record rejections.

    Returns ``(changes, problems, merges, separations, added_marks)``: the middle
    two in the shape ``audit_to_corrections`` already writes to
    ``adjudications.json``, the last to append to ``added_marks.json``.
    """
    by_id = {p.page_id: p for p in pages}
    changes: list[str] = []
    problems: list[str] = []
    merges: list[dict[str, Any]] = []
    separations: list[dict[str, Any]] = []
    added_rows: list[dict[str, Any]] = []

    for row in verdicts:
        class_id = row["class_id"]
        meta = classes.get(class_id)
        if meta is None:
            problems.append(f"{class_id}: not in classes.json")
            continue
        cands = row.get("candidates", [])
        accepted, error = _parse(str(row.get("verdict", "")), len(cands))
        if error:
            problems.append(f"{class_id}: {error}")
            continue
        if accepted is None:
            continue

        anchor = next(
            (
                (p, i)
                for p in meta.get("page_ids", [])
                if p in by_id
                for i, m in enumerate(by_id[p].marks)
                if m.class_id == class_id
            ),
            None,
        )
        if anchor is None:
            problems.append(f"{class_id}: no existing instance to link accepted candidates to")
            continue

        added, reassigned, rejected_pages = 0, 0, []
        for i, cand in enumerate(cands):
            page = by_id.get(cand["page_id"])
            if page is None:
                problems.append(f"{class_id}: candidate {i} page {cand['page_id']} is not in the manifest")
                continue
            idx = cand.get("mark_index")
            if i in accepted:
                if idx is not None and idx < len(page.marks):
                    old = page.marks[idx]
                    if old.class_id and old.class_id != class_id and old.class_id in classes:
                        problems.append(
                            f"{class_id}: candidate {i} is an instance of roster class {old.class_id}; "
                            "that is a merge, not a missing member -- use the merge slate"
                        )
                        continue
                    page.marks[idx] = Mark(old.kind, old.box, class_id, old.provenance)
                    reassigned += 1
                else:
                    box = tuple(cand["box"])
                    kind = meta.get("kind", "logo")
                    page.marks.append(Mark(kind, box, class_id, PROVENANCE))
                    idx = len(page.marks) - 1
                    added_rows.append(
                        {
                            "page_id": page.page_id,
                            "kind": kind,
                            "box": list(box),
                            "provenance": PROVENANCE,
                            "class_id": class_id,
                            "note": f"completeness: {class_id}, {cand.get('inliers')} SIFT inliers, reviewer {reviewer}",
                        }
                    )
                    added += 1
                if page.page_id not in meta["page_ids"]:
                    meta["page_ids"].append(page.page_id)
                merges.append(
                    {
                        "left_page_id": anchor[0],
                        "left_mark_index": anchor[1],
                        "right_page_id": page.page_id,
                        "right_mark_index": idx,
                        "note": f"completeness: missing member of {class_id} confirmed by hand",
                    }
                )
            else:
                if idx is not None:
                    separations.append(
                        {
                            "left_page_id": page.page_id,
                            "left_mark_index": idx,
                            "right_page_id": anchor[0],
                            "right_mark_index": anchor[1],
                            "right_class_id": class_id,
                            "note": f"completeness: candidate rejected for {class_id}",
                        }
                    )
                else:
                    rejected_pages.append(page.page_id)

        meta["page_ids"] = sorted(meta["page_ids"])
        meta["n_instances"] = len(meta["page_ids"])
        audit = meta.setdefault("audit", {})
        # One record per pass, so a later pass never erases an earlier one's rejections.
        audit["completeness_checked" + (f"_{row['pass']}" if row.get("pass") else "")] = {
            "reviewed_by": reviewer,
            "reviewed_on": date.today().isoformat(),
            "n_candidates": len(cands),
            "accepted": len(accepted),
            "unboxed_rejected_page_ids": sorted(rejected_pages),
        }
        changes.append(
            f"{class_id}: {len(accepted)} of {len(cands)} candidate(s) accepted "
            f"({reassigned} reassigned, {added} new box(es)); {meta['n_instances']} instance(s)"
        )
    return changes, problems, merges, separations, added_rows


def write_template(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ---------------------------------------------------------------------------
# The slate: matcher and sheets (GRID side; not exercised by the unit tests)
# ---------------------------------------------------------------------------


def sift_verifier(classes: dict[str, Any], pages: dict[str, Page], budget: int):
    """``verify(class_id)(page_id) -> normalised inlier box or None`` with SIFT at *budget*."""
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from vtscore.media.image._image_bulk import _load_pil  # noqa: PLC0415
    from vtscore.media.structural import SiftMatcher  # noqa: PLC0415

    matcher = SiftMatcher()

    def gray(path: str):
        return np.asarray(_load_pil(Path(path)).convert("L"), dtype=np.uint8)

    def for_class(class_id: str):
        with Image.open(classes[class_id]["query_crop"]) as crop:
            crop_feats = matcher.detect_and_describe(np.asarray(crop.convert("L"), dtype=np.uint8), max_features=budget)

        def verify(page_id: str):
            stats = matcher.verify(
                crop_feats, matcher.detect_and_describe(gray(pages[page_id].path), max_features=budget)
            )
            return stats.inlier_box if stats.model_ok and stats.inlier_box else None

        return verify

    return for_class


#: Candidates per sheet. A sheet is answered by eye and must fit one screen with
#: its reference row (Sam, 2026-09-17: a 90-candidate class rendered as three stacked
#: 30-cell sheets meant scrolling up to count and down to answer).
PER_SHEET = 18
COLS = 6


def _font(size: int, bold: bool = False):
    from PIL import ImageFont  # noqa: PLC0415

    for name in (
        ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
        "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def render(
    entry: ClassCandidates, classes: dict[str, Any], pages: dict[str, Page], out: Path, *, per_sheet: int = PER_SHEET
) -> list[Path]:
    """One-screen sheets: the reference row, then up to *per_sheet* numbered candidates.

    The candidate number is the largest thing in its cell -- the reviewer reads it
    and types it -- and the reference row (query crop + members) repeats on every
    sheet, so nothing needs scrolling back to.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    thumb, pad, cap = 250, 10, 22
    big, small, title = _font(40, bold=True), _font(15), _font(18, bold=True)
    meta = classes[entry.class_id]
    refs: list[tuple[str, Any]] = []
    with Image.open(meta["query_crop"]) as im:
        refs.append(("QUERY", im.convert("RGB").copy()))
    for page_id in meta.get("page_ids", [])[: COLS - 1]:
        page = pages.get(page_id)
        mark = next((m for m in page.marks if m.class_id == entry.class_id), None) if page else None
        if mark is None:
            continue
        x, y, w, h = mark.box
        with Image.open(page.path) as im:
            refs.append(("member", im.convert("RGB").crop((x, y, x + w, y + h))))

    cells: list[tuple[int, str, Any]] = []
    for i, cand in enumerate(entry.candidates):
        page = pages[cand.page_id]
        x, y, w, h = cand.box
        if cand.mark_index is not None:
            mx, my, mw, mh = page.marks[cand.mark_index].box
            x0, y0, x1, y1 = min(x, mx), min(y, my), max(x + w, mx + mw), max(y + h, my + mh)
        else:
            x0, y0, x1, y1 = x, y, x + w, y + h
        margin = int(0.25 * max(x1 - x0, y1 - y0))
        with Image.open(page.path) as im:
            im = im.convert("RGB")
            left, top = max(0, x0 - margin), max(0, y0 - margin)
            crop = im.crop((left, top, min(im.width, x1 + margin), min(im.height, y1 + margin)))
        draw = ImageDraw.Draw(crop)
        lw = max(2, crop.width // 120)
        draw.rectangle([x - left, y - top, x - left + w, y - top + h], outline="#1f77b4", width=lw)
        if cand.mark_index is not None:
            mx, my, mw, mh = page.marks[cand.mark_index].box
            draw.rectangle([mx - left, my - top, mx - left + mw, my - top + mh], outline="#d62728", width=lw)
            label = cand.mark_class_id.split("/")[-1][:20] if cand.mark_class_id else "unclassed mark"
        else:
            label = "NO BOX"
        if cand.note:
            cells.append((i, f"{cand.note} · {label[:12]}", crop))
        else:
            cells.append((i, f"{cand.inliers} inl · {label}", crop))

    paths = []
    name = entry.class_id.replace("/", "__")
    chunks = [cells[j : j + per_sheet] for j in range(0, len(cells), per_sheet)] or [[]]
    pos = entry.positive_inliers
    for sheet_no, chunk in enumerate(chunks):
        rows = 1 + (len(chunk) + COLS - 1) // COLS
        width = COLS * (thumb + pad) + pad
        sheet = Image.new("RGB", (width, 44 + rows * (thumb + cap + pad)), "white")
        draw = ImageDraw.Draw(sheet)
        first, last = (chunk[0][0], chunk[-1][0]) if chunk else (0, -1)
        draw.text(
            (pad, 10),
            f"{entry.class_id}   sheet {sheet_no + 1}/{len(chunks)}   candidates {first}-{last}   "
            + (f"(members' inliers: median {pos[len(pos) // 2]})" if pos else ""),
            fill="black",
            font=title,
        )
        for c, (label, img) in enumerate(refs[:COLS]):
            t = img.copy()
            t.thumbnail((thumb, thumb))
            x, y = pad + c * (thumb + pad), 44
            sheet.paste(t, (x, y + cap))
            draw.text((x, y), label, fill="#b00000", font=small)
        draw.line(
            [(pad, 44 + thumb + cap + pad // 2), (width - pad, 44 + thumb + cap + pad // 2)], fill="#999999", width=2
        )
        for k, (index, caption, img) in enumerate(chunk):
            r, c = divmod(k, COLS)
            x, y = pad + c * (thumb + pad), 44 + (r + 1) * (thumb + cap + pad)
            t = img.copy()
            t.thumbnail((thumb, thumb))
            sheet.paste(t, (x, y + cap))
            draw.rectangle([x, y + cap, x + thumb - 1, y + cap + thumb - 1], outline="#cccccc")
            draw.text((x, y + 2), caption, fill="#333333", font=small)
            # The number the reviewer types: large, boxed, top-left over the crop.
            num = str(index)
            nb = draw.textbbox((0, 0), num, font=big)
            bw, bh = nb[2] - nb[0] + 16, nb[3] - nb[1] + 14
            draw.rectangle([x, y + cap, x + bw, y + cap + bh], fill="#111111")
            draw.text((x + 8 - nb[0], y + cap + 7 - nb[1]), num, fill="#ffffff", font=big)
        path = out / f"{name}_{sheet_no:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path)
        paths.append(path)
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse  # noqa: PLC0415

    import docmarks_config as cfg  # noqa: PLC0415
    from sources._common import read_manifest  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument(
        "--inliers",
        type=Path,
        required=True,
        help="inliers.json from eval_sift_rank.py (tier s covers every anchor page)",
    )
    ap.add_argument(
        "--budget", type=int, default=8192, help="SIFT max_features; use the budget the inliers were computed at"
    )
    ap.add_argument("--top", type=int, default=TOP)
    ap.add_argument("--min-inliers", type=int, default=MIN_INLIERS)
    ap.add_argument(
        "--classes",
        default="",
        help="space-separated class ids to (re)render, e.g. to extend --top where the first slate's cap cut off "
        "real copies; numbering is by inlier rank, so the first N candidates keep their numbers",
    )
    args = ap.parse_args(argv)

    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    pages = {p.page_id: p for p in read_manifest(args.corpus / "corpus.jsonl")}
    inliers = json.loads(args.inliers.read_text(encoding="utf-8"))
    out = args.corpus / "audit" / "completeness"
    only = set(args.classes.split())
    verifier = sift_verifier(classes, pages, args.budget)

    rows = []
    template = out / "verdicts.jsonl"
    if only and template.exists():
        # Keep the other classes' rows; replace only the re-rendered ones.
        rows = [
            r
            for r in (json.loads(line) for line in template.read_text(encoding="utf-8").splitlines() if line)
            if r["class_id"] not in only
        ]
    for entry in candidates(classes, pages, inliers, top=args.top, min_inliers=args.min_inliers):
        if only and entry.class_id not in only:
            continue
        locate(entry, pages, verifier(entry.class_id))
        sheets = render(entry, classes, pages, out)
        rows.append(verdict_row(entry))
        on_mark = sum(1 for c in entry.candidates if c.mark_index is not None)
        print(
            f"  {entry.class_id}: {len(entry.candidates)} candidate(s), {on_mark} on an existing mark, {len(sheets)} sheet(s)",
            flush=True,
        )
    write_template(sorted(rows, key=lambda r: r["class_id"]), template)
    print(f"\n{sum(r['n_candidates'] for r in rows)} candidate(s) over {len(rows)} class(es) -> {out}")
    print("Fill each row's verdict ('none' or candidate numbers), then:")
    print("  python audit_to_corrections.py --task completeness --reviewer <name>          # dry run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
