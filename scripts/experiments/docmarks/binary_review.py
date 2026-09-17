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

    python binary_review.py emit  --task ucsf_classes|query_crops|box_tighten
    python binary_review.py load  --queue <dir> [--queue <dir> ...]
    python binary_review.py bank  [--date YYYY-MM-DD]

``bank`` only ever reads detectors whose name starts with ``docmarks``, and
nothing here deletes a dataset or a detector: the vg_scale campaign shares the
dashboard, and a detector is often the only copy of a human review.

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
import re
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

    W, H = CANVAS
    img = Image.new("RGB", CANVAS, "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, HEADER_H], fill="#111111")
    draw.text((20, 16), q.question, fill="white", font=_font(38, bold=True))

    # left: references in a 2x2 grid (1 ref fills the panel)
    body_top, body_h = HEADER_H + 8, H - HEADER_H - FOOTER_H - 16
    draw.text((16, body_top), "REFERENCE", fill="#1f5fbf", font=_font(22, bold=True))
    refs = q.refs[:4]
    grid = 1 if len(refs) == 1 else 2
    cell_w, cell_h = (LEFT_W - 24) // grid, (body_h - 34) // grid
    for i, ref in enumerate(refs):
        if ref.path:
            with Image.open(ref.path) as im:
                crop = im.convert("RGB")
        else:
            page = pages[ref.page_id]
            with _open_page(page, corpus) as im:
                crop = im.convert("RGB").crop(expand(ref.box, None, page.width, page.height, 0.3))
        thumb = _fit(crop, (cell_w - 12, cell_h - 30))
        cx = 12 + (i % grid) * cell_w
        cy = body_top + 34 + (i // grid) * cell_h
        img.paste(thumb, (cx + (cell_w - thumb.width) // 2, cy + (cell_h - 24 - thumb.height) // 2))
        label, small = ref.label, _font(16)
        while len(label) > 1 and draw.textlength(label, font=small) > cell_w - 10:
            label = label[:-1]
        draw.text((cx + 4, cy + cell_h - 24), label, fill="#555555", font=small)
    draw.line([LEFT_W, HEADER_H, LEFT_W, H - FOOTER_H], fill="#999999", width=3)

    # right: one candidate, enlarged
    page = pages[q.page_id]
    region = expand(q.box, q.old_box, page.width, page.height, q.margin)
    with _open_page(page, corpus) as im:
        crop = im.convert("RGB").crop(region)
    panel = (W - LEFT_W - 32, body_h)
    s = panel_scale(region, max(q.box[2], q.box[3]), panel)
    crop = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))), Image.Resampling.LANCZOS)
    ox = LEFT_W + 16 + (panel[0] - crop.width) // 2
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
    outline(q.box, "#e0201c")
    draw.rectangle([0, H - FOOTER_H, W, H], fill="#eeeeee")
    draw.text((16, H - FOOTER_H + 6), f"{q.item}   ·   {q.page_id}", fill="#333333", font=_font(18))
    path = out / q.filename
    img.save(path, quality=90)
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


def class_refs(class_id: str, classes: dict[str, Any], pages: dict[str, Any], k: int = 3) -> list[Ref]:
    """The class's query crop, then its *k* largest other boxed instances."""
    meta = classes[class_id]
    refs = [Ref(label=f"query · {class_id.split('/')[-1]}", path=meta["query_crop"])]
    inst = []
    for pid in meta.get("page_ids", []):
        if pid == meta.get("query_page_id") or pid not in pages:
            continue
        for m in pages[pid].marks:
            if m.class_id == class_id:
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
        if q["task"] == "ucsf_classes":
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
            else:
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


#: task (as written in a manifest) -> (audit dir name, translator)
TRANSLATORS: dict[str, tuple[str, Callable[..., Any]]] = {
    "ucsf_classes": ("ucsf_classes", translate_ucsf),
    "ucsf_classes_relation": ("ucsf_classes", translate_ucsf),
    "query_crops": ("query_crops", translate_query_crops),
    "box_tighten": ("box_tighten", translate_box_tighten),
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


def bank(base: str, root: Path, corpus: Path, date: str) -> int:
    manifests = {}
    for mf in sorted(root.glob("*/manifest.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        manifests[m["dataset_name"]] = (mf.parent, m)
    dets = docmarks_detectors(api(base, "/api/detectors/registry").get("detectors", []))
    by_task: dict[str, tuple[dict[str, Any], dict[str, str]]] = defaultdict(lambda: ({}, {}))
    for d in dets:
        name = d["name"]
        if name not in manifests:
            print(f"  {name}: no manifest under {root}, skipped")
            continue
        qdir, m = manifests[name]
        detail = api(base, f"/api/detectors/{urllib.parse.quote(name)}/labels-detail")
        (qdir / f"votes_{date}.json").write_text(
            json.dumps({"detector": d, "labels": detail}, indent=1) + "\n", encoding="utf-8"
        )
        votes = votes_from_labels(detail)
        print(f"  {name}: {len(votes)} of {len(m['questions'])} answered")
        for fn, q in m["questions"].items():
            audit_dir = TRANSLATORS[q["task"]][0]
            qs, vs = by_task[audit_dir]
            qs[fn] = q
            if fn in votes:
                vs[fn] = votes[fn]
    for audit_dir, (qs, vs) in by_task.items():
        translator = TRANSLATORS[next(iter(qs.values()))["task"]][1]
        src = corpus / "audit" / audit_dir / "verdicts.jsonl"
        rows = read_jsonl(src)
        out_rows, unanswered = translator(rows, qs, vs)
        dest = src.with_name("verdicts.from_vtsearch.jsonl")
        dest.write_text("".join(json.dumps(r) + "\n" for r in out_rows), encoding="utf-8")
        print(f"{audit_dir}: wrote {dest} ({len(vs)} votes); {len(unanswered)} item(s) not fully answered")
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
    e.add_argument("--task", choices=["ucsf_classes", "query_crops", "box_tighten"], required=True)
    e.add_argument("--corpus", type=Path, default=cfg.OUT)
    e.add_argument("--root", type=Path, default=ROOT)
    e.add_argument("--cross", type=Path, default=Path("/expscratch/sgreenberg/docmarks/ucsf-3921/cross.json"))
    e.add_argument("--cap", type=int, default=QUERY_CROP_CAP)
    ld = sub.add_parser("load")
    ld.add_argument("--queue", type=Path, action="append", required=True)
    ld.add_argument("--api", default=None)
    b = sub.add_parser("bank")
    b.add_argument("--corpus", type=Path, default=cfg.OUT)
    b.add_argument("--root", type=Path, default=ROOT)
    b.add_argument("--api", default=None)
    b.add_argument("--date", default=datetime.date.today().isoformat())
    args = ap.parse_args(argv)

    if args.cmd == "load":
        base = args.api or app_base()
        for q in args.queue:
            print(load_queue(base, q), flush=True)
        return 0
    if args.cmd == "bank":
        return bank(args.api or app_base(), args.root, args.corpus, args.date)

    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))
    pages = {p.page_id: p for p in read_manifest(args.corpus / "corpus.jsonl")}
    audit = args.corpus / "audit" / args.task
    if args.task == "ucsf_classes":
        cross = json.loads(args.cross.read_text()) if args.cross.exists() else None
        queues = emit_ucsf(args.corpus, audit, cross, classes, pages)
    elif args.task == "query_crops":
        queues = emit_query_crops(audit, classes, pages, cap=args.cap)
    else:
        queues = emit_box_tighten(audit, classes, pages)
    for name, qs in queues:
        d = args.root / slug(name.split(" -- ")[0])
        emit(qs, d, name, pages=pages, corpus=args.corpus)
        print(f"  {name}: {len(qs)} questions -> {d}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
