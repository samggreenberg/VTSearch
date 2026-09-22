"""DocMarks review as Good/Bad questions in VTSearch.

The review sheets asked the owner to read a grid and type the numbers that
match: one command, but a slow cognitive load, and a whole letterhead shrunk to a
thumbnail left its logo unreadable with no way to zoom.  VTSearch already is an
annotation tool whose only gesture is Good/Bad, so every audit that can be put as
a binary goes there instead:

* each question is **one image**: the references on the left (the class's query
  crop and a few instances), **one** candidate on the right, cropped to its mark
  and enlarged, with the question written on the image;
* a queue is a VTSearch dataset plus a detector **of the same name**, and the
  name is the instruction (the dashboard is the owner's worklist);
* :func:`emit` writes a queue's images and a ``manifest.json`` mapping each image
  back to the audit row it answers; ``bank`` reads the votes off the live app
  and rewrites them into that audit's own verdict format, beside the original,
  so ``audit_to_corrections.py`` applies them unchanged.

Sub-commands::

    python binary_review.py emit  --task ucsf_classes|query_crops|box_tighten|completeness2
    python binary_review.py load  --queue <dir> [--queue <dir> ...]
    python binary_review.py bank  [--date YYYY-MM-DD]

``bank`` only ever reads detectors whose name starts with ``docmarks``, and
nothing here deletes a dataset or a detector: the vg_scale campaign shares the
dashboard, and a detector is often the only copy of a human review.

**``bank`` merges; it never rewrites a verdict file from the app alone.** A
finished queue is cleared off the dashboard once its votes are banked, so the
app is a record of the *unfinished* work only.  Banking again from ``--api``
after a clear therefore found no votes for the cleared queues and wrote their
verdicts back empty -- 1,725 answered questions in one afternoon.  Votes are now
read from three places and merged, in increasing precedence: the ``.cleared``
detector backups taken when a queue was removed, the ``votes_*.json`` archive
this command writes beside every queue on every run, and the live app.  A write
that would answer fewer items than the file already on disk is refused unless
``--allow-loss`` is passed.

Nothing here clears a queue -- that is done by hand against the registry API --
but the order matters, and is why the backups exist: **copy the detector JSON to**
:data:`CLEARED` ``/detectors-cleared-<date>/<name>.json.cleared`` **before deleting
the pair**, because the labels in that file are the votes themselves, where a
verdict file holds only what they were translated into.

Other passes (``completeness_multi.py``) build :class:`Question` objects and
call :func:`emit`; their ``bank`` translation is a :data:`TRANSLATORS` entry.

**One queue per class** whenever a class has more than a few questions (the
owner answers a class's questions in a row, then moves on): UCSF members per
proposal, query crops per roster class, completeness per class.  Queue names
start ``docmarks <class> -- <question>`` so the dashboard groups by class.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Every queue name starts with this; ``bank`` ignores any other detector.
PREFIX = "docmarks"
ROOT = Path("/expscratch/sgreenberg/docmarks/binary")
#: A regrouped queue's directory; kept apart so the per-pass dirs stay readable as vote archives.
MERGED_SUFFIX = "__merged"
#: Detector JSONs copied out of the app before a finished queue was cleared, under
#: ``detectors-cleared-<date>/``.  A cleared queue is gone from the dashboard, so
#: these are the only record of its votes and ``bank`` reads them as a source.
CLEARED = Path("/expscratch/sgreenberg/keep")
CANVAS = (1400, 700)
LEFT_W = 540
HEADER_H = 78
FOOTER_H = 34
#: The candidate's mark is enlarged to at least this many pixels on its long side
#: when the panel allows it.
MIN_MARK_PX = 400
MARGIN = 0.25
#: Extra query crops kept per class, in inlier-rank order.
QUERY_CROP_CAP = 4

Box = Sequence[int]


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


@dataclass
class Ref:
    """A reference image: a crop file, or a box on a corpus page."""

    label: str
    path: Optional[str] = None
    page_id: Optional[str] = None
    box: Optional[list[int]] = None


@dataclass
class Question:
    filename: str
    task: str
    question: str
    refs: list[Ref]
    page_id: str
    box: list[int]
    #: For box questions: the current box, drawn grey beside the red proposal.
    old_box: Optional[list[int]] = None
    #: Where the answer goes back to: whatever the task's translator needs.
    key: dict[str, Any] = field(default_factory=dict)
    #: Claude's first pass: "good", "bad" or None.  Recorded, never voted.
    suggestion: Optional[str] = None
    item: str = ""
    #: Page context around the box, as a share of its long side.  A located SIFT
    #: box can sit inside the mark, so "same logo?" questions show more.
    margin: float = MARGIN
    #: False when the box is only a region the mark is somewhere inside (a SigLIP
    #: tile): drawing it would read as "this is the box".
    outline: bool = True
    #: Small print under the image (e.g. which methods proposed the candidate).
    detail: str = ""
    #: Shave the near-solid margin off the candidate before fitting it to the
    #: panel.  A scanned page is mostly white paper, so the mark renders small
    #: while the reviewer pays for margin -- twice, since the sheet is fetched
    #: whole and then again as a thumbnail.  Uses
    #: :func:`vtscore.media.image.edge_trim.solid_edge_box`, the same detector
    #: the thumbnailer and the embedding cleaner use, so a page whose border is
    #: not near-solid (a skewed scan frame, a black ring that is not uniform)
    #: simply does not trim rather than trimming wrongly.
    #:
    #: **Only for questions that draw no box.**  ``outline`` coordinates are
    #: relative to the untrimmed region, so trimming under a drawn box would
    #: move the box off the thing it points at.
    trim_border: bool = False
    #: Render greyscale and at this JPEG quality.  A scanned page IS greyscale,
    #: so three colour channels at q90 buy nothing and cost a lot: the review
    #: sheet is fetched whole over a tunnel for every vote, and the centre panel
    #: goes black until it lands.  Measured on these sheets, RGB q90 is a 198 KB
    #: median / 450 KB p90; greyscale q80 at 0.9 scale is 111 KB / 245 KB --
    #: lighter than the 124 KB median of the passes that reviewed at a page a
    #: second.  Only for sheets whose colour carries nothing; a task that draws
    #: the red proposal box needs the default.
    greyscale: bool = False
    quality: int = 90
    #: Suppress the page id in the footer.  A planted control is only a check
    #: on attention while it is indistinguishable from the rest of the queue,
    #: and a footer reading ``spods/00882`` among twenty ``ucsf/...`` pages
    #: announces it.
    anonymous: bool = False
    #: Canvas override, ``(width, height)``.  The default is landscape, which
    #: letterboxes a *portrait* page into a third of the panel -- fine for a box
    #: on a page, useless for "is the mark anywhere on this page", where the
    #: page itself has to be legible.  A contamination question renders the page
    #: 2.3x wider this way, which is the difference between a 35 px letterhead
    #: crest and an 87 px one.
    canvas: Optional[list[int]] = None
    #: Crop the candidate to its inked region (:func:`ink_box`) after any
    #: ``trim_border``.  For whole-page questions a scan's white border is often
    #: flecked with specks, so the near-solid-border trimmer finds nothing to
    #: trim and the page renders at its full size with a wide white margin.
    #: Only for questions that draw no box, like ``trim_border``.
    crop_to_ink: bool = False
    #: Caption each reference and title the column.  The captions are file names,
    #: which the reviewer does not need; without them the column is narrower
    #: (:attr:`left_w`) and each reference is larger.
    ref_labels: bool = True
    #: Width of the reference column; ``None`` is :data:`LEFT_W`.
    left_w: Optional[int] = None


#: :func:`ink_box`: a pixel darker than this is ink.
INK_LEVEL = 128
#: :func:`ink_box`: ink closer than this many pixels is one clump, so the broken
#: strokes of a faint stamp count together while an isolated speck stays alone.
INK_JOIN_PX = 6
#: :func:`ink_box`: a clump with fewer ink pixels than this is a speck.
INK_MIN_PX = 30
#: :func:`ink_box`: margin kept around the inked region, as a share of its long side.
INK_PAD = 0.03


def ink_box(im: Any) -> Optional[tuple[int, int, int, int]]:
    """``(left, top, right, bottom)`` of every clump of ink on *im*, specks ignored.

    Ink within :data:`INK_JOIN_PX` is merged before clumps are measured, so a
    mark made of many faint fragments survives as one clump.  ``None`` when the
    page holds no clump at all (a blank page), which callers leave uncropped.
    """
    import numpy as np  # noqa: PLC0415
    from scipy import ndimage  # noqa: PLC0415

    ink = np.asarray(im.convert("L")) < INK_LEVEL
    if not ink.any():
        return None
    k = 2 * INK_JOIN_PX + 1
    labels, n = ndimage.label(ndimage.binary_dilation(ink, structure=np.ones((k, k), bool)))
    if n == 0:
        return None
    counts = ndimage.sum(ink, labels, index=np.arange(1, n + 1))
    keep = np.isin(labels, np.flatnonzero(counts >= INK_MIN_PX) + 1) & ink
    if not keep.any():
        return None
    rows, cols = np.flatnonzero(keep.any(axis=1)), np.flatnonzero(keep.any(axis=0))
    top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    pad = int(INK_PAD * max(bottom - top, right - left))
    h, w = ink.shape
    return (max(0, left - pad), max(0, top - pad), min(w, right + pad), min(h, bottom + pad))


def short_class(class_id: str) -> str:
    """``tobacco800/logo_ajj10e00_1`` -> ``t800 logo_ajj10e00_1``: the class first, so the dashboard groups by it."""
    source, _, local = class_id.partition("/")
    return f"{ {'tobacco800': 't800'}.get(source, source) } {local}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def suggested_verdicts(audit: Path) -> dict[str, str]:
    path = audit / "verdicts.suggested.jsonl"
    return {row["class_id"]: row.get("verdict", "") for row in read_jsonl(path)} if path.exists() else {}


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def expand(
    box: Box, other: Optional[Box], width: int, height: int, margin: float = MARGIN
) -> tuple[int, int, int, int]:
    """The page region shown for a candidate: its box (and the old one), plus a margin, clipped."""
    x0, y0, x1, y1 = box[0], box[1], box[0] + box[2], box[1] + box[3]
    if other:
        x0, y0 = min(x0, other[0]), min(y0, other[1])
        x1, y1 = max(x1, other[0] + other[2]), max(y1, other[1] + other[3])
    pad = int(margin * max(x1 - x0, y1 - y0, 1))
    return max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)


def panel_scale(region: tuple[int, int, int, int], mark_long: int, panel: tuple[int, int]) -> float:
    """Enlarge so the mark reaches MIN_MARK_PX, but never past the panel."""
    rw, rh = max(1, region[2] - region[0]), max(1, region[3] - region[1])
    fit = min(panel[0] / rw, panel[1] / rh)
    want = MIN_MARK_PX / max(1, mark_long)
    return min(fit, max(want, 1.0))


def sheet_size(q: "Question") -> tuple[int, int]:
    """Canvas for *q* -- its override, else the shared landscape default."""
    return (int(q.canvas[0]), int(q.canvas[1])) if q.canvas else CANVAS


def footer_text(q: "Question") -> str:
    """The small print under the sheet.

    An ``anonymous`` question shows only its ``detail``: a planted attention
    control is a check on whether the reviewer looked, and a footer naming
    ``spods/00882`` among twenty ``ucsf/...`` pages answers the question for
    them.  The same applies to the arm a question was drawn from -- knowing a
    page was ranked highly by SigLIP is a reason to look harder at it.
    """
    if q.anonymous:
        return q.detail
    return f"{q.item}   ·   {q.page_id}" + (f"   ·   {q.detail}" if q.detail else "")


def _open_page(page: Any, corpus: Path):
    from PIL import Image  # noqa: PLC0415

    p = Path(page.path)
    return Image.open(p if p.is_absolute() else corpus / p)


def _fit(im: Any, size: tuple[int, int]) -> Any:
    from PIL import Image  # noqa: PLC0415

    w, h = im.size
    s = min(size[0] / w, size[1] / h)
    return im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS)


def render(q: Question, pages: dict[str, Any], corpus: Path, out: Path) -> Path:
    from PIL import Image, ImageDraw  # noqa: PLC0415

    from completeness import _font  # noqa: PLC0415

    size = sheet_size(q)
    W, H = size
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, HEADER_H], fill="#111111")
    draw.text((20, 16), q.question, fill="white", font=_font(38, bold=True))

    # left: references in a 2x2 grid (1 ref fills the panel)
    body_top, body_h = HEADER_H + 8, H - HEADER_H - FOOTER_H - 16
    left_w = q.left_w or LEFT_W
    title_h, caption_h = (34, 24) if q.ref_labels else (0, 0)
    if q.ref_labels:
        draw.text((16, body_top), "REFERENCE", fill="#1f5fbf", font=_font(22, bold=True))
    refs = q.refs[:4]
    grid = 1 if len(refs) == 1 else 2
    cell_w, cell_h = (left_w - 24) // grid, (body_h - title_h) // grid
    for i, ref in enumerate(refs):
        if ref.path:
            with Image.open(ref.path) as im:
                crop = im.convert("RGB")
        else:
            page = pages[ref.page_id]
            with _open_page(page, corpus) as im:
                crop = im.convert("RGB").crop(expand(ref.box, None, page.width, page.height, 0.3))
        thumb = _fit(crop, (cell_w - 12, cell_h - 6 - caption_h))
        cx = 12 + (i % grid) * cell_w
        cy = body_top + title_h + (i // grid) * cell_h
        img.paste(thumb, (cx + (cell_w - thumb.width) // 2, cy + (cell_h - caption_h - thumb.height) // 2))
        if q.ref_labels:
            label, small = ref.label, _font(16)
            while len(label) > 1 and draw.textlength(label, font=small) > cell_w - 10:
                label = label[:-1]
            draw.text((cx + 4, cy + cell_h - 24), label, fill="#555555", font=small)
    draw.line([left_w, HEADER_H, left_w, H - FOOTER_H], fill="#999999", width=3)

    # right: one candidate, enlarged
    page = pages[q.page_id]
    region = expand(q.box, q.old_box, page.width, page.height, q.margin)
    with _open_page(page, corpus) as im:
        crop = im.convert("RGB").crop(region)
    panel = (W - left_w - 32, body_h)
    if (q.trim_border or q.crop_to_ink) and not q.outline:
        from vtscore.media.image.edge_trim import solid_edge_box  # noqa: PLC0415

        # A solid scan frame first: its ink would otherwise hold the ink box open.
        trimmed = solid_edge_box(crop)
        if trimmed:
            crop = crop.crop(trimmed)
        inked = ink_box(crop) if q.crop_to_ink else None
        if inked:
            crop = crop.crop(inked)
        # The mark-size enlargement below is keyed to the untrimmed region, so
        # a trimmed crop just fills the panel: there is no box to keep legible,
        # only the page.
        s = min(panel[0] / crop.width, panel[1] / crop.height)
    else:
        s = panel_scale(region, max(q.box[2], q.box[3]), panel)
    crop = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))), Image.Resampling.LANCZOS)
    ox = left_w + 16 + (panel[0] - crop.width) // 2
    oy = body_top + (panel[1] - crop.height) // 2
    img.paste(crop, (ox, oy))
    lw = 4

    def outline(box: Box, colour: str) -> None:
        x, y, w, h = box
        draw.rectangle(
            [
                ox + (x - region[0]) * s,
                oy + (y - region[1]) * s,
                ox + (x - region[0] + w) * s,
                oy + (y - region[1] + h) * s,
            ],
            outline=colour,
            width=lw,
        )

    if q.old_box:
        outline(q.old_box, "#8a8a8a")
    if q.outline:
        outline(q.box, "#e0201c")
    draw.rectangle([0, H - FOOTER_H, W, H], fill="#eeeeee")
    footer = footer_text(q)
    draw.text((16, H - FOOTER_H + 6), footer, fill="#333333", font=_font(18))
    path = out / q.filename
    if q.greyscale:
        img = img.convert("L")
    img.save(path, quality=q.quality, optimize=True)
    return path


def emit(
    questions: Sequence[Question], out_dir: Path, dataset_name: str, *, pages: dict[str, Any], corpus: Path
) -> Path:
    """Render *questions* into *out_dir*/images and write *out_dir*/manifest.json.

    The manifest is what ``bank`` reads: ``{"dataset_name", "task", "questions":
    {filename: Question-as-dict}}``.  *dataset_name* must start with
    :data:`PREFIX`; it becomes both the VTSearch dataset and its detector.
    """
    if not dataset_name.startswith(PREFIX):
        raise ValueError(f"queue names must start with {PREFIX!r}: {dataset_name!r}")
    names = [q.filename for q in questions]
    if len(set(names)) != len(names):
        raise ValueError("duplicate question filenames")
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    for q in questions:
        render(q, pages, corpus, images)
    tasks = sorted({q.task for q in questions})
    manifest = {
        "dataset_name": dataset_name,
        "tasks": tasks,
        "created": datetime.date.today().isoformat(),
        "questions": {q.filename: asdict(q) for q in questions},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return out_dir


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------


#: :func:`class_refs` skips an instance whose box covers more of its page than
#: this: it is a region the mark is somewhere inside, not the mark, and as a
#: reference thumbnail it shrinks to an unreadable strip of letterhead.
REF_MAX_PAGE_FRAC = 0.10


def class_refs(class_id: str, classes: dict[str, Any], pages: dict[str, Any], k: int = 3) -> list[Ref]:
    """The class's query crop, then its *k* largest other instances boxed on the mark itself.

    A band-located instance (``*_band`` provenance: the mark is somewhere in the
    letterhead band) and any box over :data:`REF_MAX_PAGE_FRAC` of its page are
    skipped -- ranked by size they would win, and they show the reviewer a page
    strip instead of the mark.
    """
    meta = classes[class_id]
    refs = [Ref(label=f"query · {class_id.split('/')[-1]}", path=meta["query_crop"])]
    inst = []
    for pid in meta.get("page_ids", []):
        if pid == meta.get("query_page_id") or pid not in pages:
            continue
        page = pages[pid]
        for m in page.marks:
            if m.class_id != class_id:
                continue
            if (
                str(m.provenance).endswith("_band")
                or m.box[2] * m.box[3] > REF_MAX_PAGE_FRAC * page.width * page.height
            ):
                continue
            inst.append((m.box[2] * m.box[3], pid, list(m.box)))
            break
    for _, pid, box in sorted(inst, reverse=True)[:k]:
        refs.append(Ref(label=pid.split("/")[-1], page_id=pid, box=box))
    return refs


def _cells_accepted(suggestion: str, n: int) -> Optional[set[int]]:
    s = (suggestion or "").strip().lower()
    if not s:
        return None
    if s == "all":
        return set(range(n))
    if s == "none":
        return set()
    neg = s.startswith("all but")
    body = s[len("all but") :] if neg else s
    idx = {int(t) for t in body.replace(" ", "").split(",") if t}
    return set(range(n)) - idx if neg else idx


def emit_ucsf(
    corpus: Path, audit: Path, cross: Optional[dict[str, Any]], classes, pages
) -> list[tuple[str, list[Question]]]:
    rows = read_jsonl(audit / "verdicts.jsonl")
    rels = {r["proposal"]: r for r in rows if r.get("task") == "ucsf_classes_relation"}
    sheets = [r for r in rows if r.get("task") == "ucsf_classes"]
    queues: list[tuple[str, list[Question]]] = []
    relation_qs: list[Question] = []
    for name, rel in rels.items():
        sugg = rel.get("suggested_relation", "")
        target = sugg.split(" ", 1)[1] if sugg.startswith("extends ") else None
        if target is None and cross and name in cross:
            target = max(cross[name].items(), key=lambda kv: max(kv[1]))[0]
        crop_ref = Ref(label=f"{name} crop", path=rel["query_crop"])
        refs = [crop_ref]
        if sugg.startswith("extends ") and target in classes:
            refs.append(Ref(label=f"roster · {target.split('/')[-1]}", path=classes[target]["query_crop"]))
        # strongest located member cells Claude's first pass accepted
        strong = []
        for r in sheets:
            if r["proposal"] != name or not r["part"].startswith("a_"):
                continue
            ok = _cells_accepted(r.get("suggestion", ""), len(r["cells"])) or set()
            strong += [
                (c["inliers"], c["page_id"], c["box"]) for c in r["cells"] if c.get("located") and c["index"] in ok
            ]
        for inl, pid, box in sorted(strong, reverse=True)[: 4 - len(refs)]:
            if pid in pages:
                refs.append(Ref(label=f"{pid.split('/')[-1]} ({inl} inl)", page_id=pid, box=box))
        qs = []
        for r in sheets:
            if r["proposal"] != name:
                continue
            ok = _cells_accepted(r.get("suggestion", ""), len(r["cells"]))
            for c in r["cells"]:
                if c["page_id"] not in pages:
                    continue
                located = c.get("located", True)
                qs.append(
                    Question(
                        filename=f"{slug(r['sheet'].rsplit('.', 1)[0])}__c{c['index']:02d}.jpg",
                        task="ucsf_classes",
                        question="Same logo as the left?" if located else "Is the left logo anywhere in this band?",
                        refs=refs,
                        page_id=c["page_id"],
                        box=list(c["box"]),
                        key={"sheet": r["sheet"], "cell": c["index"], "n_cells": len(r["cells"])},
                        suggestion=None if ok is None else ("good" if c["index"] in ok else "bad"),
                        item=f"{name} {r['part']} sheet {r['sheet_index'] + 1} cell {c['index']} · {c['inliers']} inliers",
                        margin=0.6,
                    )
                )
        queues.append((f"{PREFIX} {name} -- right logo same as left?", qs))
        if target and target in classes and rel.get("query_page_id") in pages:
            relation_qs.append(
                Question(
                    filename=f"relation__{slug(name)}.jpg",
                    task="ucsf_classes_relation",
                    question="Same mark as this roster class?",
                    refs=class_refs(target, classes, pages),
                    page_id=rel["query_page_id"],
                    box=list(rel["query_box"]),
                    key={"proposal": name, "target": target},
                    suggestion="good" if sugg.startswith("extends ") else "bad",
                    item=f"UCSF proposal {name} vs roster {target}",
                )
            )
    queues.append((f"{PREFIX} ucsf relations -- UCSF logo (right) is the left roster class?", relation_qs))
    return queues


def emit_query_crops(audit: Path, classes, pages, cap: int = QUERY_CROP_CAP) -> list[tuple[str, list[Question]]]:
    rows = read_jsonl(audit / "verdicts.jsonl")
    sugg = suggested_verdicts(audit)
    queues = []
    for r in rows:
        cid = r["class_id"]
        qs = []
        picked = _cells_accepted(sugg.get(cid, ""), len(r["candidates"]))
        for c in r["candidates"]:
            if c["page_id"] not in pages:
                continue
            qs.append(
                Question(
                    filename=f"qcrop__{slug(cid)}__c{c['index']:02d}.jpg",
                    task="query_crops",
                    question="Good extra query? (whole mark, tight red box, clean)",
                    refs=[Ref(label=f"current query · {cid.split('/')[-1]}", path=classes[cid]["query_crop"])],
                    page_id=c["page_id"],
                    box=list(c["box"]),
                    key={"class_id": cid, "index": c["index"], "n": len(r["candidates"])},
                    suggestion=None if picked is None else ("good" if c["index"] in picked else "bad"),
                    item=f"{cid} candidate {c['index']} · {c['inliers']} inliers",
                )
            )
        # one queue per class: the owner answers one class at a time
        queues.append((f"{PREFIX} {short_class(cid)} -- good extra query crop?", qs))
    return queues


def box_questionable(member: dict[str, Any]) -> bool:
    flags = member.get("flags") or []
    return "unchanged" not in flags and "no fit" not in flags and member.get("new_box") is not None


def emit_box_tighten(audit: Path, classes, pages) -> list[tuple[str, list[Question]]]:
    rows = read_jsonl(audit / "verdicts.jsonl")
    sugg = suggested_verdicts(audit)
    qs = []
    for r in rows:
        cid = r["class_id"]
        picked = _cells_accepted(sugg.get(cid, ""), len(r["members"]))
        for m in r["members"]:
            if not box_questionable(m) or m["page_id"] not in pages:
                continue
            qs.append(
                Question(
                    filename=f"box__{slug(cid)}__m{m['index']:02d}.jpg",
                    task="box_tighten",
                    question="Red box right? (grey = current box)",
                    refs=[Ref(label=f"query · {cid.split('/')[-1]}", path=classes[cid]["query_crop"])],
                    page_id=m["page_id"],
                    box=list(m["new_box"]),
                    old_box=list(m["old_box"]),
                    key={"class_id": cid, "index": m["index"]},
                    suggestion=None if picked is None else ("good" if m["index"] in picked else "bad"),
                    item=f"{cid} member {m['index']} · {m.get('inliers', 0)} inliers",
                )
            )
    return [(f"{PREFIX} staver boxes -- red box right?", qs)]


COMPLETENESS2 = Path("/expscratch/sgreenberg/docmarks/completeness2/verdicts.suggested.jsonl")


def tile_box(candidate: dict[str, Any]) -> bool:
    """Proposed by SigLIP tiles alone: its box is the whole tile, not the mark."""
    return str(candidate.get("methods", "")).strip() == "SIG"


def emit_completeness2(source: Path, classes, pages) -> list[tuple[str, list[Question]]]:
    """The multi-proposer completeness slate (#3963), one queue per class."""
    queues = []
    for r in read_jsonl(source):
        cid = r["class_id"]
        if cid not in classes:
            continue
        refs = class_refs(cid, classes, pages)
        qs = []
        for c in r["candidates"]:
            if c["page_id"] not in pages or not c.get("box"):
                continue
            tile = tile_box(c)
            sugg = {"yes": "good", "no": "bad"}.get(str(c.get("suggestion", "")).lower())
            qs.append(
                Question(
                    filename=f"compl2__{slug(cid)}__c{c['index']:02d}.jpg",
                    task="completeness2",
                    question="Same mark as left?" + ("  (mark somewhere in this tile)" if tile else ""),
                    refs=refs,
                    page_id=c["page_id"],
                    box=list(c["box"]),
                    key={"class_id": cid, "index": c["index"], "tile_box": tile},
                    suggestion=sugg,
                    item=f"{cid} candidate {c['index']}",
                    margin=0.5 if tile else 0.6,  # a tile can clip the mark: show around it too
                    outline=not tile,
                    detail=f"found by {c.get('methods', '?')}",
                )
            )
        queues.append((f"{PREFIX} {short_class(cid)} -- right mark same as left? (completeness 2)", qs))
    return queues


def merged_name(class_id: str) -> str:
    """The one pair a class gets once its passes are regrouped."""
    return f"{PREFIX} {short_class(class_id)} -- read the question on each image"


def regroup(
    root: Path,
    cleared: Path = CLEARED,
    base: Optional[str] = None,
    skip_tasks: Sequence[str] = ("box_tighten",),
) -> list[tuple[str, Path, int]]:
    """Rebuild what is left of every pass as ONE queue per class, dropping answered questions.

    A class reaches the dashboard twice -- a 4-12 question ``query_crops`` pair and
    a 36-question ``completeness2`` one -- for what is a single sitting, and every
    image already carries its own question in large type.  This merges what is
    still unanswered, **copying the rendered images rather than re-rendering
    them**, so a question seen after the regroup is pixel-identical to the one
    before it.

    Answered questions are dropped, which is the point: re-asking one wastes the
    reviewer's time and invites a second, contradicting vote.  The per-pass
    directories are left in place because their ``votes_*.json`` archives are
    what :func:`collect_votes` reads for a queue that is no longer on the
    dashboard.

    ``box_tighten`` is skipped by default: "is this red box right?" is a
    different act from "is this the same mark", and mixing the two is the
    confusion this is meant to remove.
    """
    by_queue, _ = collect_votes(base, root, cleared)
    # by filename, not by queue: a question answered in some other detector is
    # still answered, and re-asking it is the thing this is meant to avoid
    votes, _conflicts, _unplaceable = vote_index(by_queue)
    remaining: dict[str, list[tuple[Path, str, dict[str, Any]]]] = defaultdict(list)
    for mf in sorted(root.glob("*/manifest.json")):
        if mf.parent.name.endswith(MERGED_SUFFIX):
            continue
        m = json.loads(mf.read_text(encoding="utf-8"))
        for fn, q in sorted(m["questions"].items()):
            if fn in votes or q["task"] in skip_tasks:
                continue
            class_id = (q.get("key") or {}).get("class_id")
            if not class_id:
                raise SystemExit(f"{mf}: {fn} has no key.class_id, so it cannot be grouped by class")
            remaining[class_id].append((mf.parent / "images" / fn, fn, q))
    out = []
    for class_id, items in sorted(remaining.items()):
        name = merged_name(class_id)
        qdir = root / (slug(name.split(" -- ")[0]) + MERGED_SUFFIX)
        images = qdir / "images"
        images.mkdir(parents=True, exist_ok=True)
        wanted = {fn for _src, fn, _q in items}
        for stale in sorted(images.glob("*.jpg")):
            # a question answered since the last regroup: leaving the file would re-ask it
            if stale.name not in wanted:
                stale.unlink()
                print(f"  {name}: dropped {stale.name}, answered since the last regroup")
        for src, fn, _q in items:
            dest = images / fn
            if dest.exists():
                continue
            try:
                os.link(src, dest)  # same filesystem: no second copy of the bytes
            except OSError:
                shutil.copy2(src, dest)
        manifest = {
            "dataset_name": name,
            "tasks": sorted({q["task"] for _src, _fn, q in items}),
            "created": datetime.date.today().isoformat(),
            "merged_from": sorted({src.parent.parent.name for src, _fn, _q in items}),
            "questions": {fn: q for _src, fn, q in items},
        }
        (qdir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        out.append((name, qdir, len(items)))
    return out


# ---------------------------------------------------------------------------
# Banking: votes -> each audit's verdict rows
# ---------------------------------------------------------------------------


def docmarks_detectors(detectors: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only our queues.  vg_scale shares the dashboard; its detectors are never read."""
    return [d for d in detectors if str(d.get("name", "")).startswith(PREFIX + " ")]


def cell_list(accepted: set[int], n: int) -> str:
    if len(accepted) == n:
        return "all"
    if not accepted:
        return "none"
    return ",".join(str(i) for i in sorted(accepted))


def translate_ucsf(
    rows: list[dict[str, Any]], questions: dict[str, dict[str, Any]], votes: dict[str, str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fill ``verdict`` on every sheet whose cells are all answered, ``relation`` on answered proposals."""
    by_sheet: dict[str, dict[int, Optional[str]]] = defaultdict(dict)
    rel_vote: dict[str, tuple[str, str]] = {}
    for fn, q in questions.items():
        if q["task"] in ("ucsf_classes", "ucsf_banded"):
            by_sheet[q["key"]["sheet"]][q["key"]["cell"]] = votes.get(fn)
        elif q["task"] == "ucsf_classes_relation" and fn in votes:
            rel_vote[q["key"]["proposal"]] = (votes[fn], q["key"]["target"])
    out, unanswered = [], []
    for r in rows:
        r = dict(r)
        if r.get("task") == "ucsf_classes":
            cells = by_sheet.get(r["sheet"], {})
            n = len(r["cells"])
            if len(cells) < n or any(v is None for v in cells.values()):
                missing = n - sum(1 for v in cells.values() if v is not None)
                unanswered.append(f"{r['sheet']}: {missing} of {n} cells unanswered; verdict left blank")
            else:
                r["verdict"] = cell_list({i for i, v in cells.items() if v == "good"}, n)
                r["verdict_source"] = "vtsearch"
        elif r.get("task") == "ucsf_classes_relation":
            if r["proposal"] in rel_vote:
                vote, target = rel_vote[r["proposal"]]
                r["relation"] = f"extends {target}" if vote == "good" else "new"
                r["verdict_source"] = "vtsearch"
            elif not str(r.get("relation", "")).strip():
                # A relation ruled earlier and carried into this slate is answered.
                unanswered.append(f"{r['proposal']}: relation unanswered")
        out.append(r)
    return out, unanswered


def _per_class(
    questions: dict[str, dict[str, Any]], votes: dict[str, str], task: str
) -> dict[str, dict[int, Optional[str]]]:
    got: dict[str, dict[int, Optional[str]]] = defaultdict(dict)
    for fn, q in questions.items():
        if q["task"] == task:
            got[q["key"]["class_id"]][q["key"]["index"]] = votes.get(fn)
    return got


def translate_query_crops(rows, questions, votes, cap: int = QUERY_CROP_CAP):
    """Accepted candidates in inlier-rank order (candidate index), at most *cap*; 'none' if all rejected."""
    got = _per_class(questions, votes, "query_crops")
    out, unanswered = [], []
    for r in rows:
        r = dict(r)
        asked = got.get(r["class_id"], {})
        if not asked or any(v is None for v in asked.values()):
            unanswered.append(
                f"{r['class_id']}: {sum(v is None for v in asked.values())} of {len(asked)} candidates unanswered"
            )
        else:
            keep = sorted(i for i, v in asked.items() if v == "good")[:cap]
            r["verdict"] = ",".join(map(str, keep)) if keep else "none"
            r["verdict_source"] = "vtsearch"
        out.append(r)
    return out, unanswered


def translate_box_tighten(rows, questions, votes):
    """Members voted Good accept the red box; members never asked keep their box."""
    got = _per_class(questions, votes, "box_tighten")
    out, unanswered = [], []
    for r in rows:
        r = dict(r)
        asked = got.get(r["class_id"], {})
        if any(v is None for v in asked.values()):
            unanswered.append(
                f"{r['class_id']}: {sum(v is None for v in asked.values())} of {len(asked)} boxes unanswered"
            )
        else:
            keep = sorted(i for i, v in asked.items() if v == "good")
            r["verdict"] = ",".join(map(str, keep)) if keep else "none"
            r["verdict_source"] = "vtsearch"
        out.append(r)
    return out, unanswered


def needs_drawn_box(tile: bool, candidate: Optional[dict[str, Any]]) -> bool:
    """A tile-box Good that no existing mark can stand in for.

    A tile says "the mark is somewhere in here", so it cannot become a mark's box.
    But when a boxed mark already sits under the tile, ``apply_completeness``
    reassigns *that* mark and never reads the tile box -- nothing needs drawing.
    Only a tile over bare page does (#4040).
    """
    return bool(tile) and (candidate is None or candidate.get("mark_index") is None)


def translate_completeness2(rows, questions, votes):
    """Good candidates become the completeness verdict -- except undrawn tile boxes.

    A Good vote on a SigLIP-tile candidate says the mark is on that page, but its
    box is the tile.  Where no mark is already boxed under that tile, passing it
    through would add a tile-sized mark, so those indices go to ``needs_tight_box``
    and stay out of ``verdict`` until a box is drawn.
    """
    got: dict[str, dict[int, tuple[Optional[str], bool]]] = defaultdict(dict)
    for fn, q in questions.items():
        if q["task"] == "completeness2":
            got[q["key"]["class_id"]][q["key"]["index"]] = (votes.get(fn), bool(q["key"].get("tile_box")))
    out, unanswered = [], []
    for r in rows:
        r = dict(r)
        asked = got.get(r["class_id"], {})
        missing = sum(v is None for v, _ in asked.values())
        if not asked or missing:
            unanswered.append(
                f"{r['class_id']}: {missing} of {len(asked)} candidates unanswered; verdict left as it was"
            )
            out.append(r)
            continue
        # The row's own candidates, not the manifest, say whether a mark is already
        # boxed under a tile: a queue built before #4040 has no such key to read.
        cands = {int(c["index"]): c for c in r.get("candidates", []) if c.get("index") is not None}
        good = [i for i, (v, _tile) in asked.items() if v == "good"]
        keep = sorted(i for i in good if not needs_drawn_box(asked[i][1], cands.get(i)))
        tight = sorted(i for i in good if needs_drawn_box(asked[i][1], cands.get(i)))
        r["verdict"] = ",".join(map(str, keep)) if keep else "none"
        r["needs_tight_box"] = tight
        r["verdict_source"] = "vtsearch"
        if tight:
            unanswered.append(
                f"{r['class_id']}: candidate(s) {tight} carry the mark but have only a tile box; draw a box before apply"
            )
        out.append(r)
    return out, unanswered


def _translate_surprise(rows, questions, votes):
    from surprise_review import translate_surprise  # noqa: PLC0415

    return translate_surprise(rows, questions, votes)


#: task (as written in a manifest) -> (verdict source for a corpus, translator)
TRANSLATORS: dict[str, tuple[Callable[[Path], Path], Callable[..., Any]]] = {
    "ucsf_classes": (lambda corpus: corpus / "audit" / "ucsf_classes" / "verdicts.jsonl", translate_ucsf),
    "ucsf_classes_relation": (lambda corpus: corpus / "audit" / "ucsf_classes" / "verdicts.jsonl", translate_ucsf),
    # Banded pages reviewed into known negatives (#4088): single-cell ucsf_classes
    # sheets, in a slate of their own so the admission slate is never rewritten.
    "ucsf_banded": (lambda corpus: corpus / "audit" / "ucsf_banded" / "verdicts.jsonl", translate_ucsf),
    "query_crops": (lambda corpus: corpus / "audit" / "query_crops" / "verdicts.jsonl", translate_query_crops),
    # Top-ranked presumed negatives of a run (#4089); the translator lives with its pass.
    "surprise": (lambda corpus: corpus / "audit" / "surprise" / "verdicts.jsonl", _translate_surprise),
    "box_tighten": (lambda corpus: corpus / "audit" / "box_tighten" / "verdicts.jsonl", translate_box_tighten),
    "completeness2": (lambda corpus: COMPLETENESS2, translate_completeness2),
}


def votes_from_labels(detail: dict[str, Any]) -> dict[str, str]:
    """``labels-detail`` -> {filename: "good"|"bad"}.  A file voted both ways is a conflict and raises."""
    votes: dict[str, str] = {}
    for label in ("good", "bad"):
        for row in detail.get(label, []):
            fn = row.get("filename") or row.get("name")
            if fn in votes and votes[fn] != label:
                raise ValueError(f"{fn} is both good and bad")
            votes[fn] = label
    return votes


# ---------------------------------------------------------------------------
# The live app
# ---------------------------------------------------------------------------


def app_base() -> str:
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    squeue = shutil.which("squeue") or "/usr/bin/squeue"
    got = subprocess.run(  # noqa: S603 - fixed argv
        [squeue, "-u", "sgreenberg", "-h", "-n", "vtsearch", "-o", "%N"], capture_output=True, text=True, check=False
    ).stdout.split()
    if not got:
        raise SystemExit("no running vtsearch job")
    return f"http://{got[0]}:11850"


def api(base: str, path: str, payload=None, method: str = "GET", timeout: int = 180) -> dict[str, Any]:
    req = urllib.request.Request(  # noqa: S310 - our own app on the cluster
        base.rstrip("/") + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:  # noqa: S310
            body = fh.read().decode()
    except urllib.error.HTTPError as exc:
        return {"_error": f"{exc.code}: {exc.read().decode()[:300]}"}
    return json.loads(body) if body.strip() else {}


def _count(d: dict[str, Any]) -> int:
    fc = d.get("file_type_counts") or {}
    return sum(int(v) for v in fc.values()) if fc else int(d.get("num_items") or d.get("count") or 0)


def load_queue(base: str, queue: Path, wait: int = 1800) -> str:
    """Import a queue's images as a dataset and create its same-named detector (both idempotent)."""
    manifest = json.loads((queue / "manifest.json").read_text(encoding="utf-8"))
    name = manifest["dataset_name"]
    n = len(list((queue / "images").glob("*.jpg")))
    have = {d.get("name"): d for d in api(base, "/api/datasets/registry").get("datasets", [])}
    if name not in have:
        resp = api(
            base,
            "/api/dataset/import/server_folder",
            {
                "path": str(queue / "images"),
                "media_type": "image",
                "recursive": "false",
                "dig_archives": "false",
                "dataset_name": name,
            },
            method="POST",
        )
        if "_error" in resp:
            return f"{name}: IMPORT FAILED {resp['_error'][:120]}"
        t0 = time.time()
        while time.time() - t0 < wait:
            time.sleep(5)
            d = {x.get("name"): x for x in api(base, "/api/datasets/registry").get("datasets", [])}.get(name)
            if d and _count(d) >= n:
                break
        else:
            return f"{name}: import still running after {wait}s"
    got = _count({x.get("name"): x for x in api(base, "/api/datasets/registry").get("datasets", [])}.get(name, {}))
    dets = {d["name"] for d in api(base, "/api/detectors/registry").get("detectors", [])}
    if name not in dets:
        resp = api(
            base,
            "/api/detectors/registry",
            {
                "name": name,
                "media_type": "image",
                "embedder_type": "semantic",
                "text_query": "logo",
                "examples": [{"type": "text", "value": "logo"}],
            },
            method="POST",
        )
        if "_error" in resp or not resp.get("ok"):
            return f"{name}: dataset {got}/{n}, DETECTOR FAILED {str(resp)[:120]}"
    return f"{name}: dataset {got}/{n} items, detector ok"


def labels_of(detector: dict[str, Any]) -> dict[str, str]:
    """A detector JSON -- a ``.cleared`` backup or a live export -- to {filename: "good"|"bad"}."""
    votes: dict[str, str] = {}
    for row in (detector.get("labelset") or {}).get("labels") or []:
        fn = row.get("filename") or row.get("origin_name")
        if fn and row.get("label") in ("good", "bad"):
            votes[fn] = row["label"]
    return votes


def collect_votes(base: Optional[str], root: Path, cleared: Path) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    """Every vote ever cast, per queue, whether or not the queue is still on the dashboard.

    Later sources win, so a live answer overrides an archived one for the same
    image.  ``base`` of ``None`` reads the on-disk sources only, which is what a
    test (and a run against a dead app) needs.
    """
    votes: dict[str, dict[str, str]] = defaultdict(dict)
    seen: dict[str, int] = defaultdict(int)
    for path in sorted(cleared.glob("detectors-cleared-*/*.json.cleared")):
        backup = json.loads(path.read_text(encoding="utf-8"))
        name, got = backup.get("name"), labels_of(backup)
        if name and got:
            votes[name].update(got)
            seen["cleared backup"] += 1
    for path in sorted(root.glob("*/votes_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        name = (d.get("detector") or {}).get("name")
        got = votes_from_labels(d.get("labels") or {})
        if name and got:
            votes[name].update(got)
            seen["archive"] += 1
    if base:
        for d in docmarks_detectors(api(base, "/api/detectors/registry").get("detectors", [])):
            got = votes_from_labels(api(base, f"/api/detectors/{urllib.parse.quote(d['name'])}/labels-detail"))
            if got:
                votes[d["name"]].update(got)
                seen["live detector"] += 1
    return dict(votes), dict(seen)


def question_index(manifests: dict[str, tuple[Path, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """``{filename: question}`` across every queue.

    A filename is claimed by several manifests as a matter of course: a regrouped
    queue hardlinks its predecessors' images, so the same question appears in the
    pass that first asked it and in the merged pair that still asks it.  That is
    benign only while the two describe the *same* question, so a filename claimed
    twice with differing entries raises rather than letting one silently win.
    """
    index: dict[str, dict[str, Any]] = {}
    claimed: dict[str, list[str]] = defaultdict(list)
    for name, (_qdir, m) in sorted(manifests.items()):
        for fn, q in (m.get("questions") or {}).items():
            claimed[fn].append(name)
            prev = index.get(fn)
            if prev is not None and json.dumps(prev, sort_keys=True) != json.dumps(q, sort_keys=True):
                raise SystemExit(
                    f"{fn} is claimed by {claimed[fn]} with different questions; "
                    "a vote on it cannot be routed, so nothing was banked"
                )
            index[fn] = q
    return index


def vote_index(
    by_queue: dict[str, dict[str, str]], questions: Optional[dict[str, dict[str, Any]]] = None
) -> tuple[dict[str, str], list[str], list[str]]:
    """``{filename: label}`` for every vote, whatever detector it was cast in.

    A vote is an answer to the QUESTION its filename names, not to the dataset
    folder it happens to sit in: loading a second dataset into one detector puts
    a foreign queue's votes there, and keying by folder drops every one of them.
    Returns the index plus what could not be resolved -- ``conflicts`` (one
    filename voted both ways in different queues, where no rule can say which
    click was later) and ``unplaceable`` (a vote naming a question no manifest
    has).  Both are reported rather than guessed at or dropped in silence.
    """
    per_file: dict[str, dict[str, str]] = defaultdict(dict)
    for queue, votes in by_queue.items():
        for fn, label in votes.items():
            per_file[fn][queue] = label
    index: dict[str, str] = {}
    conflicts: list[str] = []
    unplaceable: list[str] = []
    for fn, per in sorted(per_file.items()):
        if questions is not None and fn not in questions:
            unplaceable.append(f"{fn}: voted in {sorted(per)} but no manifest asks it")
            continue
        labels = set(per.values())
        if len(labels) > 1:
            conflicts.append(f"{fn}: {', '.join(f'{q}={v}' for q, v in sorted(per.items()))}")
            continue
        index[fn] = next(iter(labels))
    return index, conflicts, unplaceable


def archive_votes(qdir: Path, date: str, payload: dict[str, Any]) -> Path:
    """Keep this run's votes beside the queue, without shrinking an earlier archive.

    A re-imported queue can answer with fewer votes than the archive already
    holds, and that archive may be the only copy once the queue is cleared.
    """
    n = sum(len((payload.get("labels") or {}).get(k) or []) for k in ("good", "bad"))
    dest = qdir / f"votes_{date}.json"
    if dest.exists():
        old = json.loads(dest.read_text(encoding="utf-8")).get("labels") or {}
        have = sum(len(old.get(k) or []) for k in ("good", "bad"))
        if have > n:
            dest = qdir / f"votes_{date}T{datetime.datetime.now():%H%M%S}.json"
            print(f"  WARNING {qdir.name}: app has {n} vote(s), archive has {have}; wrote {dest.name} instead")
    dest.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return dest


def answered(rows: Sequence[dict[str, Any]]) -> int:
    """Rows a translator has filled in from votes."""
    return sum(1 for r in rows if r.get("verdict_source") == "vtsearch")


def guard_write(dest: Path, out_rows: Sequence[dict[str, Any]], cleared: Path, allow_loss: bool) -> None:
    """Refuse a write that answers fewer items than the file already on disk.

    The same shape as ``pilebuild.corrections.dropped_rows``: the guard exists
    because the thing being overwritten is human work, and the flag that skips it
    has to be typed on purpose.
    """
    before = answered(read_jsonl(dest)) if dest.exists() else 0
    now = answered(out_rows)
    if now < before and not allow_loss:
        raise SystemExit(
            f"refusing to write {dest}: it answers {before} item(s) and this run answers {now}.\n"
            f"  Votes for a cleared queue live only in its votes_*.json archive or a .cleared backup "
            f"under {cleared} -- check those are readable before overwriting, or pass --allow-loss."
        )


def bank(
    base: str,
    root: Path,
    corpus: Path,
    date: str,
    cleared: Path = CLEARED,
    allow_loss: bool = False,
) -> int:
    manifests = {}
    for mf in sorted(root.glob("*/manifest.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        manifests[m["dataset_name"]] = (mf.parent, m)
    # archive what the app holds now, before reading anything back
    for d in docmarks_detectors(api(base, "/api/detectors/registry").get("detectors", [])) if base else []:
        name = d["name"]
        if name not in manifests:
            print(f"  {name}: no manifest under {root}, skipped")
            continue
        detail = api(base, f"/api/detectors/{urllib.parse.quote(name)}/labels-detail")
        archive_votes(manifests[name][0], date, {"detector": d, "labels": detail})
    by_queue, sources = collect_votes(base, root, cleared)
    print(f"  vote sources: {', '.join(f'{v} {k}(s)' for k, v in sorted(sources.items())) or 'none'}")
    questions = question_index(manifests)
    votes, conflicts, unplaceable = vote_index(by_queue, questions)
    for line in conflicts:
        print(f"  CONFLICT {line}")
    for line in unplaceable:
        print(f"  UNPLACEABLE {line}")
    by_task: dict[str, tuple[dict[str, Any], dict[str, str]]] = defaultdict(lambda: ({}, {}))
    for name, (_qdir, m) in sorted(manifests.items()):
        got = sum(1 for fn in m["questions"] if fn in votes)
        print(f"  {name}: {got} of {len(m['questions'])} answered")
        for fn, q in m["questions"].items():
            src = TRANSLATORS[q["task"]][0](corpus)
            qs, vs = by_task[str(src)]
            qs[fn] = q
            if fn in votes:
                vs[fn] = votes[fn]
    for src_name, (qs, vs) in by_task.items():
        translator = TRANSLATORS[next(iter(qs.values()))["task"]][1]
        src = Path(src_name)
        rows = read_jsonl(src)
        out_rows, unanswered = translator(rows, qs, vs)
        dest = src.with_name("verdicts.from_vtsearch.jsonl")
        guard_write(dest, out_rows, cleared, allow_loss)
        dest.write_text("".join(json.dumps(r) + "\n" for r in out_rows), encoding="utf-8")
        print(f"{src.parent.name}: wrote {dest} ({len(vs)} votes); {len(unanswered)} item(s) not fully answered")
        for u in unanswered[:40]:
            print(f"    {u}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse  # noqa: PLC0415

    import docmarks_config as cfg  # noqa: PLC0415
    from sources._common import read_manifest  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit")
    e.add_argument("--task", choices=["ucsf_classes", "query_crops", "box_tighten", "completeness2"], required=True)
    e.add_argument("--corpus", type=Path, default=cfg.OUT)
    e.add_argument("--root", type=Path, default=ROOT)
    e.add_argument("--cross", type=Path, default=Path("/expscratch/sgreenberg/docmarks/ucsf-3921/cross.json"))
    e.add_argument("--cap", type=int, default=QUERY_CROP_CAP)
    ld = sub.add_parser("load")
    ld.add_argument("--queue", type=Path, action="append", required=True)
    ld.add_argument("--api", default=None)
    rg = sub.add_parser("regroup")
    rg.add_argument("--root", type=Path, default=ROOT)
    rg.add_argument("--api", default=None, help="read live votes too, so an in-flight answer is not re-asked")
    rg.add_argument("--cleared", type=Path, default=CLEARED)
    b = sub.add_parser("bank")
    b.add_argument("--corpus", type=Path, default=cfg.OUT)
    b.add_argument("--root", type=Path, default=ROOT)
    b.add_argument("--api", default=None)
    b.add_argument("--date", default=datetime.date.today().isoformat())
    b.add_argument("--cleared", type=Path, default=CLEARED, help="root holding detectors-cleared-*/ backups")
    b.add_argument(
        "--allow-loss",
        action="store_true",
        help="write even when the result answers fewer items than the file on disk (it is human work)",
    )
    args = ap.parse_args(argv)

    if args.cmd == "load":
        base = args.api or app_base()
        for q in args.queue:
            print(load_queue(base, q), flush=True)
        return 0
    if args.cmd == "bank":
        return bank(args.api or app_base(), args.root, args.corpus, args.date, args.cleared, args.allow_loss)
    if args.cmd == "regroup":
        for name, qdir, n in regroup(args.root, args.cleared, args.api or app_base()):
            print(f"  {name}: {n} question(s) -> {qdir}", flush=True)
        return 0

    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    pages = {p.page_id: p for p in read_manifest(args.corpus / "corpus.jsonl")}
    audit = args.corpus / "audit" / args.task
    if args.task == "ucsf_classes":
        cross = json.loads(args.cross.read_text()) if args.cross.exists() else None
        queues = emit_ucsf(args.corpus, audit, cross, classes, pages)
    elif args.task == "query_crops":
        queues = emit_query_crops(audit, classes, pages, cap=args.cap)
    elif args.task == "completeness2":
        queues = emit_completeness2(COMPLETENESS2, classes, pages)
    else:
        queues = emit_box_tighten(audit, classes, pages)
    # a class can have queues from several passes: the pass keeps their dirs apart
    suffix = "__completeness2" if args.task == "completeness2" else ""
    for name, qs in queues:
        d = args.root / (slug(name.split(" -- ")[0]) + suffix)
        emit(qs, d, name, pages=pages, corpus=args.corpus)
        print(f"  {name}: {len(qs)} questions -> {d}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
