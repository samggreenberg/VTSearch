#!/usr/bin/env python
"""Render the contact sheets for the DocMarks human passes.

    python make_audit_slate.py --task cluster      # are the derived classes real?
    python make_audit_slate.py --task distinctive  # is this mark an instance at all?
    python make_audit_slate.py --task letterhead   # does the weak label hold?

Each task writes PNG sheets plus a ``verdicts.jsonl`` template into
``<out>/audit/<task>/``.  Fill in the verdict field, then fold the answers back
with ``audit_to_corrections.py``.

**These are the only three annotation passes the corpus asks for**, and each one
exists because a specific number is otherwise unknowable rather than merely
unverified:

``cluster``
    SPODS and StaVer ship mark locations without identities, so their class
    inventory is derived by :mod:`cluster_marks` and is currently a hypothesis.
    A previous study published per-class results on exactly this kind of derived
    inventory without checking it.  Reviewing the largest clusters catches the
    single-linkage failure mode (two classes bridged by one ambiguous crop) in
    about a minute per class.

``distinctive``
    A plain warning triangle or a ruled box is a *shape*, not an *instance*: no
    amount of geometry makes "find this rectangle" a well-posed retrieval query.
    The prior study's worst SPODS/StaVer classes (``warning_diamond`` at 17
    keypoints, ``hospital_cross``) are this failure, and averaging them into a
    headline AP measures the dataset's junk rather than the method.  Splitting
    them out as a labelled stratum — not deleting them — lets both numbers be
    reported.

``letterhead``
    The UCSF stratum's labels come from metadata: ``author:"PHILIP MORRIS"``
    implies a Philip Morris letterhead.  Nobody has measured how often that is
    true.  If it is 90% the stratum is usable with a noise model; if it is 40%
    it is not usable at all.  ~100 sampled pages per author settles it, and
    nothing else can.

Deliberately **not** an annotation pass: exhaustively checking the distractor
pool for unlabelled positives.  That is unfixable by hand at 200k pages and is
instead prevented by construction — see ``CONTAMINATES`` in
``docmarks_config.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    derived = {k: v for k, v in classes.items() if "clustered" in v.get("provenance", [])}

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
                # one of: ok | split | merge_into:<class_id> | drop
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


def task_letterhead(pages: list[Page], out: Path, *, sample_per_author: int = 100, seed: int = 7) -> list[dict]:
    """Sampled full pages per weak-label author: does the letterhead exist?"""
    from PIL import Image

    rng = random.Random(seed)
    by_class: dict[str, list[Page]] = {}
    for page in pages:
        for mark in page.marks:
            if mark.provenance == "weak" and mark.class_id:
                by_class.setdefault(mark.class_id, []).append(page)

    verdicts = []
    for class_id, group in sorted(by_class.items()):
        sample = rng.sample(group, min(sample_per_author, len(group)))
        name = class_id.replace("/", "__")
        thumbs = []
        for page in sample:
            with Image.open(page.path) as im:
                # Only the top third: a letterhead lives at the top of the page,
                # and a full-page thumbnail at this size shows nothing useful.
                thumbs.append(im.convert("RGB").crop((0, 0, im.width, im.height // 3)))
        for start in range(0, len(thumbs), COLS * 5):
            _sheet(
                thumbs[start : start + COLS * 5],
                f"{class_id} — page tops — which carry the letterhead?",
                out / f"{name}_{start // (COLS * 5):03d}.png",
            )
        verdicts.append(
            {
                "task": "letterhead",
                "class_id": class_id,
                "sampled": len(sample),
                "population": len(group),
                # Fill in the count of sampled pages that DO carry the mark.
                # That count divided by `sampled` is the stratum's label
                # precision, which is the number the whole layer hangs on.
                "verdict": "",
                "notes": "",
            }
        )
    return verdicts


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=("cluster", "distinctive", "letterhead"))
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--sample-per-author", type=int, default=100)
    args = ap.parse_args(argv)

    pages = list(read_manifest(args.corpus / "corpus.jsonl"))
    classes_path = args.corpus / "classes.json"
    classes = json.loads(classes_path.read_text(encoding="utf-8")) if classes_path.exists() else {}

    out = args.corpus / "audit" / args.task
    out.mkdir(parents=True, exist_ok=True)

    if args.task == "cluster":
        verdicts = task_cluster(pages, classes, out)
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
