#!/usr/bin/env python
"""Render the contact sheets for the DocMarks human passes.

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
        "--task", required=True, choices=("membership", "cluster", "confusable", "distinctive", "letterhead")
    )
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--sample-per-author", type=int, default=60)
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
