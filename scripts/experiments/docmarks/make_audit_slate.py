#!/usr/bin/env python
"""Render the contact sheets for the DocMarks human passes.

    python make_audit_slate.py --task merge        # which of these classes are one mark?
    python make_audit_slate.py --task membership   # is each crop really this mark?
    python make_audit_slate.py --task letterhead   # is a mark there at all?
    python make_audit_slate.py --task cluster      # is this class one mark?
    python make_audit_slate.py --task confusable   # are these two the same mark?
    python make_audit_slate.py --task distinctive  # is this a mark or a shape?

Each task writes PNG sheets plus a ``verdicts.jsonl`` template into
``<out>/audit/<task>/``.  Fill in the verdict field, then fold the answers back
with ``audit_to_corrections.py``.

The corpus stores **both directions** of the ground truth, because an eval needs
both: a shared class id says what the detector must find together, and a
recorded separation says what it must tell apart.  Clustering can only ever
propose the first.  These passes supply the second and confirm the first.

``merge``
    The pairwise pass in a shape that finishes.  Every class on a handful of
    numbered contact sheets in similarity order, plus explicit side-by-side
    sheets for the closest pairs; the answer is a list of index sets ("12 37 41
    are the same mark") in ``merges.txt``, which compiles straight back into
    ``confusable``'s same/different verdicts.  Use this instead of ``confusable``
    on anything past a couple of dozen classes -- 60 classes is 1,770 pairs, and
    a pass that hands a reviewer 1,770 PNGs does not get run.

``letterhead``
    Runs first, on UCSF candidate pages.  Not "are these crops one mark" but "is
    any of this a mark" — an author query returns letters, and some are plain
    paper, carbon copies, or the second page of something.  A band with nothing
    printed on it must be a distractor, and no amount of clustering discovers
    that on its own.

``cluster``
    SPODS and StaVer ship mark locations without identities, and UCSF ships
    neither, so every class in those strata is derived and is a hypothesis until
    looked at.  A previous study published per-class results on exactly this
    kind of unchecked inventory.  One sheet per class, all instances; single
    linkage's failure mode (two marks bridged by one ambiguous crop) is obvious
    on sight.  ``split`` is productive here — it re-clusters that class alone at
    a tighter threshold and re-sheets the pieces.

``confusable``
    The nearest class pairs, side by side, ranked by descriptor distance so the
    genuinely ambiguous ones come first.  ``different`` records a permanent
    separation that every future re-cluster honours; ``same`` sends you to
    ``merge_into:`` on the cluster task.  Without this the corpus can only ever
    assert similarity, and the pairs a detector most needs to distinguish are
    precisely the ones a threshold decided by a hair.

``distinctive``
    A plain warning triangle or a ruled box is a *shape*, not an *instance*: no
    amount of geometry makes "find this rectangle" a well-posed retrieval query.
    The prior study's worst SPODS/StaVer classes (``warning_diamond`` at 17
    keypoints, ``hospital_cross``) are this failure, and averaging them into a
    headline AP measures the dataset's junk rather than the method.  Splitting
    them out as a labelled stratum — not deleting them — lets both numbers be
    reported.

Deliberately **not** an annotation pass: exhaustively checking the distractor
pool for unlabelled positives.  That is unfixable by hand at 200k pages and is
instead prevented by construction — see ``CONTAMINATES`` in
``docmarks_config.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cluster_marks as _cluster  # noqa: E402
import docmarks_config as cfg  # noqa: E402
from sources._common import Page, read_manifest  # noqa: E402

THUMB = 180
COLS = 8
PAD = 6


def _sheet(images: Sequence[Any], title: str, out_path: Path, captions: Optional[Sequence[str]] = None) -> None:
    from PIL import Image, ImageDraw

    rows = (len(images) + COLS - 1) // COLS
    cap_h = 14 if captions else 0
    width = COLS * (THUMB + PAD) + PAD
    height = rows * (THUMB + PAD + cap_h) + PAD + 24
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD, 6), title, fill="black")

    for i, im in enumerate(images):
        r, c = divmod(i, COLS)
        thumb = im.convert("RGB").copy()
        thumb.thumbnail((THUMB, THUMB))
        x = PAD + c * (THUMB + PAD)
        y = 24 + PAD + r * (THUMB + PAD + cap_h)
        sheet.paste(thumb, (x, y))
        draw.rectangle([x, y, x + thumb.width, y + thumb.height], outline="#cccccc")
        if captions:
            draw.text((x, y + thumb.height + 2), captions[i][:28], fill="#444444")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _crop(page: Page, box: tuple[int, int, int, int], pad_frac: float = 0.08) -> Any:
    from PIL import Image

    x, y, w, h = box
    pad = int(round(max(w, h) * pad_frac))
    with Image.open(page.path) as im:
        return im.convert("RGB").crop(
            (max(0, x - pad), max(0, y - pad), min(im.width, x + w + pad), min(im.height, y + h + pad))
        )


def task_cluster(pages: list[Page], classes: dict[str, Any], out: Path, *, max_per_class: int = 24) -> list[dict]:
    """One sheet per derived class: every instance, so a bad merge is obvious."""
    by_id = {p.page_id: p for p in pages}
    verdicts = []
    derived = {k: v for k, v in classes.items() if any(p.startswith("clustered") for p in v.get("provenance", []))}

    for class_id, meta in sorted(derived.items(), key=lambda kv: -kv[1]["n_instances"]):
        crops, caps = [], []
        for page_id in meta["page_ids"][:max_per_class]:
            page = by_id.get(page_id)
            if page is None:
                continue
            for mark in page.marks:
                if mark.class_id == class_id:
                    crops.append(_crop(page, mark.box))
                    caps.append(page_id.split("/")[-1])
                    break
        if not crops:
            continue
        name = class_id.replace("/", "__")
        _sheet(crops, f"{class_id}  —  {meta['n_instances']} instances  —  all one mark?", out / f"{name}.png", caps)
        verdicts.append(
            {
                "task": "cluster",
                "class_id": class_id,
                "n_instances": meta["n_instances"],
                "sheet": f"{name}.png",
                # one of:
                #   ok                     every crop on the sheet is one mark
                #   merge_into:<class_id>  this and that class are the same mark
                #   split                  the sheet holds more than one mark;
                #                          re-clusters this class alone at a
                #                          tighter threshold and re-sheets it
                #   drop                   not a usable mark at all
                "verdict": "",
                "notes": "",
            }
        )
    return verdicts


def task_membership(pages: list[Page], classes: dict[str, Any], out: Path) -> list[dict]:
    """Every instance of every roster class, numbered, for an in/out call.

    This is the pass that makes a roster worth having.  ``cluster`` asks whether
    a class is *one* mark; this asks, of each individual crop, whether it really
    is *that* mark.  On a curated roster the two dozen classes come to a few
    hundred crops, which is an afternoon — and afterwards no positive in the
    eval is unexamined, so a false negative is unambiguously the detector's
    fault rather than possibly the label's.

    Verdicts are recorded as **indices to reject** rather than one row per crop,
    because the sheets are numbered and "3,17" is the whole answer for a class
    that is 30-for-32 correct.  An empty verdict on a reviewed class means every
    crop is good, so the row must still be marked — ``ok`` says so explicitly.
    """
    by_id = {p.page_id: p for p in pages}
    roster_classes = {k: v for k, v in classes.items() if v.get("on_roster")}
    if not roster_classes:
        return []

    verdicts = []
    for class_id, meta in sorted(roster_classes.items()):
        crops, caps, page_ids = [], [], []
        for page_id in meta["page_ids"]:
            page = by_id.get(page_id)
            if page is None:
                continue
            for mark in page.marks:
                if mark.class_id == class_id:
                    crops.append(_crop(page, mark.box))
                    caps.append(f"[{len(crops) - 1}] {page_id.split('/')[-1]}")
                    page_ids.append(page_id)
                    break
        if not crops:
            continue

        name = class_id.replace("/", "__")
        for start in range(0, len(crops), COLS * 4):
            _sheet(
                crops[start : start + COLS * 4],
                f"{class_id} — instances {start}..{start + len(crops[start : start + COLS * 4]) - 1} "
                f"of {len(crops)} — which are NOT this mark?",
                out / f"{name}_{start // (COLS * 4):02d}.png",
                caps[start : start + COLS * 4],
            )
        verdicts.append(
            {
                "task": "membership",
                "class_id": class_id,
                "n_instances": len(crops),
                # Index -> page id, so a verdict of "3,17" resolves without the
                # reviewer ever handling a page id.
                "page_ids": page_ids,
                # "ok" = every crop is this mark. Otherwise a comma-separated
                # list of the numbered crops that are NOT (e.g. "3,17").
                "verdict": "",
                "notes": "",
            }
        )
    return verdicts


def _font(size: int) -> Any:
    """A truetype face at *size* if one is installed, else PIL's bitmap default.

    The slate's whole answer format is the reviewer reading an index off a cell
    and typing it, so the index has to be unmistakable at a glance.  PIL's
    built-in font is 11 px and renders "37" and "87" nearly alike on a scan-grey
    background; DejaVu is present on every machine this has run on so far, and
    the fallback keeps the sheets rendering rather than crashing where it is not.
    """
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def seriate(dist: np.ndarray) -> list[int]:
    """Order classes so that near-identical ones land next to each other.

    A slate in class-id order is a slate the reviewer has to scan quadratically:
    the two crops of the same stamp are wherever the alphabet put them.  Ordered
    by similarity, an over-split shows up as two adjacent cells that look the
    same, which is the thing a person is genuinely good at spotting.

    Greedy nearest-neighbour from the most isolated class, which is a cheap
    approximation of the shortest Hamiltonian path.  It is *not* a claim about
    global structure -- a 1-D order cannot preserve a metric that isn't 1-D, and
    the row wrap breaks adjacency every ``MERGE_SLATE_COLS`` cells anyway.  That
    is why the near-pair appendix exists: the ordering is an aid to scanning,
    and the appendix is what guarantees every risky pair is actually seen.

    Deterministic, including ties, so a re-rendered slate has the same numbering
    and a half-finished answer file stays valid.
    """
    n = dist.shape[0]
    if n == 0:
        return []
    work = dist.copy()
    np.fill_diagonal(work, np.inf)
    # Start at the most isolated class: it is an endpoint of the path, so the
    # chain runs through the crowd rather than starting inside it.
    start = int(np.argmax(np.where(np.isinf(work), -np.inf, work).min(axis=1)))
    order = [start]
    unvisited = set(range(n)) - {start}
    while unvisited:
        last = order[-1]
        nxt = min(unvisited, key=lambda j: (float(work[last, j]), j))
        order.append(nxt)
        unvisited.discard(nxt)
    return order


def near_pairs(dist: np.ndarray, k: int) -> list[tuple[float, int, int]]:
    """The *k* closest class pairs, nearest first, ties broken by index."""
    n = dist.shape[0]
    pairs = [(float(dist[i, j]), i, j) for i in range(n) for j in range(i + 1, n)]
    pairs.sort(key=lambda t: (t[0], t[1], t[2]))
    return pairs[:k]


def _cell(strip: Sequence[Any], index: int, label: str, *, thumb: int = 120) -> Any:
    """One class as a numbered horizontal strip of its own instances."""
    from PIL import Image, ImageDraw

    slots = max(1, len(strip))
    head = 22
    width = slots * (thumb + 4) + 4
    cell = Image.new("RGB", (width, head + thumb + 6), "white")
    draw = ImageDraw.Draw(cell)
    draw.rectangle([0, 0, width - 1, cell.height - 1], outline="#999999")
    draw.rectangle([0, 0, width - 1, head], fill="#eef1f5", outline="#999999")
    draw.text((5, 3), f"[{index}]", fill="#0b3d91", font=_font(16))
    draw.text((52, 5), label[:46], fill="#333333", font=_font(11))

    for i, im in enumerate(strip[:slots]):
        t = im.convert("RGB").copy()
        t.thumbnail((thumb, thumb))
        x = 4 + i * (thumb + 4) + (thumb - t.width) // 2
        y = head + 3 + (thumb - t.height) // 2
        cell.paste(t, (x, y))
    return cell


def _paste_grid(cells: Sequence[Any], title: str, out_path: Path, *, cols: int) -> None:
    from PIL import Image, ImageDraw

    if not cells:
        return
    cw = max(c.width for c in cells)
    ch = max(c.height for c in cells)
    rows = (len(cells) + cols - 1) // cols
    font = _font(14)
    # Widen for the heading if the grid is narrower than it: the heading is what
    # says which sheet this is and what the reviewer is being asked, and a
    # two-column pairs sheet is narrower than its own instructions.
    title_w = int(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(title, font=font)) + 2 * PAD
    sheet = Image.new("RGB", (max(cols * (cw + PAD) + PAD, title_w), rows * (ch + PAD) + PAD + 26), "white")
    ImageDraw.Draw(sheet).text((PAD, 6), title, fill="black", font=font)
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        sheet.paste(cell, (PAD + c * (cw + PAD), 26 + PAD + r * (ch + PAD)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _class_strip(page_ids: Sequence[str], by_id: dict[str, Page], class_id: str, limit: int) -> list[Any]:
    """Up to *limit* instance crops of *class_id*, evenly spread over the class.

    Spread rather than the first *limit*: page ids sort by source and number, so
    the head of the list is whatever the scanner did first, and a class whose
    later instances drifted (a re-inked stamp, a second printing) would look
    homogeneous on the slate for no better reason than alphabetical order.
    """
    crops: list[Any] = []
    resolved = [pid for pid in page_ids if pid in by_id]
    if not resolved:
        return crops
    step = max(1, len(resolved) // max(1, limit))
    for page_id in resolved[::step][:limit]:
        page = by_id[page_id]
        for mark in page.marks:
            if mark.class_id == class_id:
                crops.append(_crop(page, mark.box))
                break
    return crops


def task_merge(
    pages: list[Page],
    classes: dict[str, Any],
    out: Path,
    *,
    top_pairs: int = cfg.MERGE_SLATE_NEAR_PAIRS,
) -> dict[str, Any]:
    """Every class on a few numbered sheets; the answer is a list of index sets.

    This is ``confusable`` asked in a shape a person can finish.  The pairwise
    pass is correct and, past a couple of dozen classes, unusable: a 60-class
    corpus is 1,770 pairs, so adjudicating the matrix means opening 1,770 PNGs
    to type ``different`` 1,750 times.  The information a reviewer actually has
    is not a stream of independent binary answers -- it is a *partition*, and
    almost all of it is the trivial part.

    So the slate elicits the partition directly.  Classes are laid out in
    similarity order and numbered; the reviewer writes one line per group of
    classes that are the same mark, and nothing at all for the overwhelming
    majority that are already right.  A 60-class slate whose clustering
    over-split three times is four sheets and three lines of answer.

    Two things keep that cheap answer honest:

    * **The near-pair appendix.**  A 1-D ordering cannot keep every close pair
      adjacent, and the row wrap breaks adjacency every few cells regardless.
      The ``top_pairs`` closest pairs therefore also get an explicit
      side-by-side sheet, so no pair where a wrong call costs anything depends
      on the layout having been kind.
    * **A closed world that only covers what was looked at.**  ``REVIEWED-ALL``
      in the answer file records every *appendix* pair the reviewer did not
      merge as a permanent separation -- the half of the ground truth clustering
      cannot produce, bought from one sitting.  It does not touch the pairs that
      appear nowhere but the far end of the distance ranking: those were never
      compared, and claiming them would put a decision nobody made into the file
      that every future re-cluster is bound by.
    """
    from PIL import Image

    by_id = {p.page_id: p for p in pages}
    pool = {k: v for k, v in classes.items() if v.get("on_roster")} or classes

    exemplars: list[tuple[str, Any]] = []
    for class_id, meta in sorted(pool.items()):
        crop = meta.get("query_crop")
        if crop and Path(crop).exists():
            exemplars.append((class_id, Image.open(crop)))
    if len(exemplars) < 2:
        return {"classes": [], "near_pairs": [], "sheets": []}

    desc = np.array([_cluster.phash(im) for _cid, im in exemplars], dtype=bool)
    bits = desc.astype(np.uint8)
    inv = 1 - bits
    dist = (bits @ inv.T + inv @ bits.T).astype(float) / desc.shape[1]
    np.fill_diagonal(dist, 0.0)

    order = seriate(dist)
    slate_index = {orig: slot for slot, orig in enumerate(order)}

    cells, index_rows = [], []
    for slot, orig in enumerate(order):
        class_id, exemplar = exemplars[orig]
        meta = pool[class_id]
        strip = _class_strip(meta.get("page_ids", []), by_id, class_id, cfg.MERGE_SLATE_INSTANCES) or [exemplar]
        cells.append(_cell(strip, slot, f"{class_id}  ({int(meta.get('n_instances', len(strip)))}x)"))
        index_rows.append(
            {
                "index": slot,
                "class_id": class_id,
                "n_instances": int(meta.get("n_instances", 0)),
                "sheet": f"slate_{slot // (cfg.MERGE_SLATE_COLS * cfg.MERGE_SLATE_ROWS):02d}.png",
            }
        )

    per_sheet = cfg.MERGE_SLATE_COLS * cfg.MERGE_SLATE_ROWS
    sheets = []
    for start in range(0, len(cells), per_sheet):
        name = f"slate_{start // per_sheet:02d}.png"
        _paste_grid(
            cells[start : start + per_sheet],
            f"DocMarks merge slate — classes [{start}]..[{start + len(cells[start : start + per_sheet]) - 1}] "
            f"of {len(cells)} — which of these are the SAME mark?",
            out / name,
            cols=cfg.MERGE_SLATE_COLS,
        )
        sheets.append(name)

    pair_rows = []
    pairs = near_pairs(dist, top_pairs)
    for rank, (d, i, j) in enumerate(pairs):
        li, ri = slate_index[i], slate_index[j]
        pair_rows.append(
            {
                "rank": rank,
                "left_index": li,
                "right_index": ri,
                "left_class_id": exemplars[i][0],
                "right_class_id": exemplars[j][0],
                "distance": round(float(d), 4),
                "sheet": f"pairs_{rank // cfg.MERGE_PAIRS_PER_SHEET:02d}.png",
            }
        )

    for start in range(0, len(pairs), cfg.MERGE_PAIRS_PER_SHEET):
        chunk = pairs[start : start + cfg.MERGE_PAIRS_PER_SHEET]
        pair_cells = []
        for d, i, j in chunk:
            li, ri = slate_index[i], slate_index[j]
            pair_cells.append(_cell([exemplars[i][1], exemplars[j][1]], li, f"vs [{ri}]   distance {d:.3f}", thumb=150))
        _paste_grid(
            pair_cells,
            f"DocMarks — the {len(pairs)} closest pairs, {start}..{start + len(chunk) - 1} — "
            "merge these on the slate, or they are recorded as DIFFERENT",
            out / f"pairs_{start // cfg.MERGE_PAIRS_PER_SHEET:02d}.png",
            cols=2,
        )

    payload = {"classes": index_rows, "near_pairs": pair_rows, "sheets": sheets}
    (out / "index.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_merge_template(out / "merges.txt", index_rows, len(pair_rows))
    return payload


MERGE_TEMPLATE_HEADER = """\
# DocMarks merge slate — {n} classes, {p} near pairs on the appendix sheets.
#
# One line per GROUP of classes that are THE SAME MARK, as slate indices:
#
#     12 37 41
#     3 8        # same elephant stamp, blue and red ink
#
# That is the whole answer format. Classes named on no line are left alone,
# which is the expected outcome for most of them — you are correcting an
# over-split, not re-labelling the corpus. Text after '#' is kept as a note.
# Groups that share a class are unioned, so you cannot contradict yourself by
# writing the same merge twice.
#
# When you have worked every slate sheet AND every pairs_*.png sheet, add:
#
#     REVIEWED-ALL
#
# on a line of its own. That records the {p} appendix pairs you did NOT merge as
# permanently DIFFERENT — a cannot-link every future re-cluster honours. It is
# the half of the ground truth clustering cannot produce, and it is why the
# appendix is worth working through. Leave the line off and only your merges are
# recorded; nothing else is assumed about what you looked at.
#
# Then:  python audit_to_corrections.py --task merge --apply
"""


def _write_merge_template(path: Path, index_rows: Sequence[dict[str, Any]], n_pairs: int) -> None:
    lines = [MERGE_TEMPLATE_HEADER.format(n=len(index_rows), p=n_pairs), "\n"]
    path.write_text("".join(lines), encoding="utf-8")


def task_confusable(
    pages: list[Page],
    classes: dict[str, Any],
    out: Path,
    *,
    top_n: int = 60,
) -> list[dict]:
    """Side-by-side sheets of the *nearest* class pairs — same mark, or not?

    This is the half of the ground truth clustering cannot produce.  A shared
    class id already says "these must be found together"; nothing yet says
    "these must be told apart", and the pairs where that matters are exactly the
    ones a threshold decided by a hair.  Ranking pairs by descriptor distance
    puts the genuinely ambiguous ones first, so the adjudication effort lands
    where the labels are actually undetermined rather than being spread evenly
    over pairs no one could confuse.

    A ``different`` verdict is recorded as a permanent separation and is
    enforced on every future re-cluster, so this decision is made once.
    """
    from PIL import Image

    # On a roster the full matrix is small enough to adjudicate exhaustively:
    # 24 classes is 276 pairs, and every pair judged means the "different"
    # half of the ground truth is complete rather than sampled. top_n only
    # bites on a candidate pool, where the count would otherwise be quadratic
    # in a few hundred classes.
    pool = {k: v for k, v in classes.items() if v.get("on_roster")} or classes
    exemplars: list[tuple[str, Any]] = []
    for class_id, meta in sorted(pool.items()):
        crop = meta.get("query_crop")
        if crop and Path(crop).exists():
            exemplars.append((class_id, Image.open(crop)))
    if len(exemplars) < 2:
        return []

    desc = np.array([_cluster.phash(im) for _cid, im in exemplars], dtype=bool)
    bits = desc.astype(np.uint8)
    inv = 1 - bits
    dist = (bits @ inv.T + inv @ bits.T).astype(float) / desc.shape[1]
    np.fill_diagonal(dist, 1.0)

    all_pairs = sorted(
        ((dist[i, j], i, j) for i in range(len(exemplars)) for j in range(i + 1, len(exemplars))),
        key=lambda t: (t[0], t[1], t[2]),
    )
    pairs = all_pairs if len(all_pairs) <= top_n else all_pairs[:top_n]

    verdicts = []
    for rank, (d, i, j) in enumerate(pairs):
        left_id, left_im = exemplars[i]
        right_id, right_im = exemplars[j]
        name = f"pair_{rank:03d}"
        _sheet(
            [left_im, right_im],
            f"{left_id}   vs   {right_id}   (distance {d:.3f}) — same mark, or different?",
            out / f"{name}.png",
            [left_id.split("/")[-1], right_id.split("/")[-1]],
        )
        verdicts.append(
            {
                "task": "confusable",
                "left_class_id": left_id,
                "right_class_id": right_id,
                "distance": round(float(d), 4),
                "sheet": f"{name}.png",
                # one of: same | different
                # "different" is recorded permanently and blocks these two from
                # ever being merged by a later re-cluster.
                "verdict": "",
                "notes": "",
            }
        )
    return verdicts


def task_distinctive(pages: list[Page], classes: dict[str, Any], out: Path) -> list[dict]:
    """One sheet of every class's exemplar: is this an instance or a shape?"""
    from PIL import Image

    exemplars, caps, verdicts = [], [], []
    for class_id, meta in sorted(classes.items()):
        crop_path = meta.get("query_crop")
        if not crop_path or not Path(crop_path).exists():
            continue
        exemplars.append(Image.open(crop_path))
        caps.append(class_id.split("/")[-1])
        verdicts.append(
            {
                "task": "distinctive",
                "class_id": class_id,
                # one of: distinctive | generic
                # "generic" = a shape anyone could draw (plain box, warning
                # triangle, ruled line). Kept in the corpus, excluded from the
                # headline stratum.
                "verdict": "",
                "notes": "",
            }
        )

    for start in range(0, len(exemplars), COLS * 6):
        chunk = exemplars[start : start + COLS * 6]
        _sheet(
            chunk,
            f"distinctive vs generic — classes {start}..{start + len(chunk) - 1}",
            out / f"exemplars_{start // (COLS * 6):03d}.png",
            caps[start : start + COLS * 6],
        )
    return verdicts


def task_letterhead(pages: list[Page], out: Path, *, sample_per_author: int = 60, seed: int = 7) -> list[dict]:
    """Sampled letterhead bands per candidate author — is a mark there at all?

    This runs *before* clustering can be trusted, and it answers a different
    question from ``cluster``: not "are these crops one mark" but "is any of
    this a mark".  An author query returns letters; some are printed on plain
    paper, some are carbon copies, some are the second page of something.  A
    band with no mark on it is a page that must be a distractor, not a class
    member, and no amount of clustering discovers that on its own.
    """
    from PIL import Image

    rng = random.Random(seed)
    by_author: dict[str, list[Page]] = {}
    for page in pages:
        author = page.meta.get("letterhead_author")
        if author and any(m.provenance in ("candidate", "clustered_band") for m in page.marks):
            by_author.setdefault(author, []).append(page)

    verdicts = []
    for author, group in sorted(by_author.items()):
        sample = rng.sample(group, min(sample_per_author, len(group)))
        name = re.sub(r"[^a-z0-9]+", "_", author.lower()).strip("_")
        thumbs = []
        for page in sample:
            with Image.open(page.path) as im:
                band = next((m.box for m in page.marks if m.area() > 0), (0, 0, im.width, im.height // 4))
                x, y, w, h = band
                thumbs.append(im.convert("RGB").crop((x, y, x + w, y + h)))
        for start in range(0, len(thumbs), COLS * 5):
            _sheet(
                thumbs[start : start + COLS * 5],
                f"{author} — letterhead bands — how many carry a printed mark?",
                out / f"{name}_{start // (COLS * 5):03d}.png",
            )
        verdicts.append(
            {
                "task": "letterhead",
                "author": author,
                "sampled": len(sample),
                "population": len(group),
                # Count of sampled bands that DO carry a printed mark. Divided
                # by `sampled` this is the candidate pool's yield, which decides
                # whether the UCSF stratum is worth clustering at all.
                "verdict": "",
                "notes": "",
            }
        )
    return verdicts


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--task", required=True, choices=("merge", "membership", "cluster", "confusable", "distinctive", "letterhead")
    )
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--sample-per-author", type=int, default=60)
    ap.add_argument(
        "--near-pairs",
        type=int,
        default=cfg.MERGE_SLATE_NEAR_PAIRS,
        help="merge: how many of the closest class pairs get an explicit side-by-side sheet, "
        "and so become eligible for a REVIEWED-ALL separation",
    )
    ap.add_argument(
        "--top-pairs",
        type=int,
        default=400,
        help="confusable: cap on pairs; a 24-class roster's full 276-pair matrix fits under the default",
    )
    args = ap.parse_args(argv)

    pages = list(read_manifest(args.corpus / "corpus.jsonl"))
    classes_path = args.corpus / "classes.json"
    classes = json.loads(classes_path.read_text(encoding="utf-8")) if classes_path.exists() else {}

    out = args.corpus / "audit" / args.task
    out.mkdir(parents=True, exist_ok=True)

    if args.task == "merge":
        payload = task_merge(pages, classes, out, top_pairs=args.near_pairs)
        print(f"merge: {len(payload['classes'])} class(es) on {len(payload['sheets'])} slate sheet(s)")
        print(f"  slate    {out}/slate_*.png")
        print(f"  pairs    {out}/pairs_*.png   ({len(payload['near_pairs'])} nearest pairs)")
        print(f"  answer   {out}/merges.txt  (one line per group of same-mark indices)")
        return 0

    if args.task == "membership":
        verdicts = task_membership(pages, classes, out)
    elif args.task == "cluster":
        verdicts = task_cluster(pages, classes, out)
    elif args.task == "confusable":
        verdicts = task_confusable(pages, classes, out, top_n=args.top_pairs)
    elif args.task == "distinctive":
        verdicts = task_distinctive(pages, classes, out)
    else:
        verdicts = task_letterhead(pages, out, sample_per_author=args.sample_per_author)

    path = out / "verdicts.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for v in verdicts:
            fh.write(json.dumps(v, sort_keys=True) + "\n")

    print(f"{args.task}: {len(verdicts)} item(s) to review")
    print(f"  sheets   {out}")
    print(f"  verdicts {path}  (fill in the 'verdict' field, then run audit_to_corrections.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
