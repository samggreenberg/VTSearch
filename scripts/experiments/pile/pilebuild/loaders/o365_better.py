"""``o365_better``: `coco_better`'s construction on a second source, Objects365 v2 val (#4815).

Every object-class method comparison in this repo is read off `coco_better`. Its
per-cell voting/test split guards a detector against its own votes, but not a
*method* against COCO: its curation, its annotators' habits, and the rulings
built on top of it (#4056, #4119, #3985, #4179) are the same on both halves of
every split. A second bench drawn from another annotation source is the check
on that, and it only works as a check if nothing is tuned on it -- see the
``o365_better`` entry in :data:`pile_config.DATASETS`.

**What is the same is the point.** *C* is :data:`pile_config.SCALE_CLASSES`,
banded by the same :func:`~pilebuild.scale_core.band_candidates` on the largest
instance, designated by the same :func:`~pilebuild.scale_core.designate_cells`
and drawn by the same :func:`~pilebuild.scale_core.draw_negatives`, into the
same media dict (:func:`~pilebuild.scale_core.scale_media`). A cell here and a
`coco_better` cell of the same name mean the same thing by construction, so a
method contrast that holds on one and not the other is a fact about the
source, not about the harness.

**What differs is the vocabulary, and it is mapped through the class rules.**
Objects365 names 365 categories in its own words. :data:`pile_config.O365_NAMES`
says which ARE a class of *C* under its :data:`pile_config.SCALE_CLASS_RULES`
entry -- the rule is the definition, COCO's label set only one reading of it --
and :data:`pile_config.O365_AMBIGUOUS` which might be. The reader refuses a
mapped name its file does not carry, because a misspelt name maps nothing and
says nothing: Objects365's own spellings carry typos (``Dinning Table``), and
"correcting" one would be exactly that silent failure.

**A box makes an image a positive only if it is ONE real object.** The release
flags three kinds of box, and none of them is a region a user would drag:

* ``iscrowd`` -- a group the annotators could not separate. 14% of the split's
  boxes, against ~1% in COCO. This is Objects365's own answer to the lump
  problem `coco_better` needs LVIS for (#3985): a bunch of bananas drawn as one
  box is flagged rather than passed off as one banana.
* ``isfake`` -- a depiction: a dog on a poster, a car on a screen. The class
  rules call these Bad (`bicycle`'s pictogram, `bird`'s figurine).
* ``isreflected`` -- an object seen in a mirror.

A flagged box, a box under an ambiguous name, or a box with no area makes its
(image, class) pair **doubtful**: the class is held, so the image is never a
negative for it, and the pair is never a positive either. When the same image
also holds an unflagged box of the class, that box is what bands it and the
doubtful ones are dropped. This is the door `scale_core` already has for a COCO
box that is not ONE object (``excluded``), not a new one.

`coco_better` drops ``iscrowd`` regions outright, so a class held ONLY by a crowd
box would read as absent there. On COCO that never happens -- over 2017
train+val, no image holds a class of *C* only as a crowd region, because COCO
boxes the individual instances beside a crowd -- so the shortcut costs it
nothing. On Objects365 val it happens for 15,073 (image, class) pairs, and
taking the shortcut would put crowds of a class into that class's negatives.


**Every image is ``coco_scored``.** The field means "the source answered for
every class at once", which is what makes a negative provable (#3670), and
Objects365 annotates all 365 categories on every image. The name is
`coco_better`'s; the meaning carries over, and the media's ``origin`` records
the real source (``labels: objects365``).

**Designated only.** There is no full-corpus mode and no ``relabel``: the split
is 80,000 images, a designated build embeds about a third of them, and a rule
change is a rebuild.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import pile_config as pc

from pilebuild.env import log
from pilebuild.scale_core import band_candidates, designate_cells, draw_negatives, scale_label_fields, scale_media

#: The release's per-box flags, any of which makes a box doubtful.
FLAGS = ("iscrowd", "isfake", "isreflected")


class O365Labels(NamedTuple):
    """What the reader hands the build: labels plus the pairs that may not be positives.

    *labels* is ``{image_id: {class: [[x0, y0, x1, y1], ...]}}`` in pixel space,
    with an entry for **every** image -- ``{}`` is "Objects365 looked and there
    is nothing of *C* here", the distinction #3667 turned on. A class's boxes are
    its unflagged ones when it has any, else its doubtful ones. *doubtful* is the
    (image, class) pairs whose boxes are all doubtful; the build passes it as
    ``excluded``. *reasons* counts, per (image, class) doubtful pair, what made
    it so -- the census prices each one.
    """

    labels: dict[int, dict[str, list[list[float]]]]
    doubtful: set[tuple[int, str]]
    box_dims: dict[int, tuple[int, int]]
    filenames: dict[int, str]
    reasons: dict[tuple[int, str], set[str]]


def read_o365_labels(path: Path, classes: tuple[str, ...]) -> O365Labels:
    """Read one Objects365 annotation file into :class:`O365Labels` for *classes*."""
    if not path.exists():
        raise SystemExit(f"o365_better: missing {path}; stage Objects365 v2 val (see pile_config.O365_ROOT)")
    with path.open() as fh:
        data = json.load(fh)
    cat_name = {c["id"]: c["name"] for c in data["categories"]}

    definite = {name: cls for cls in classes for name in pc.o365_names_for(cls)}
    ambiguous: dict[str, list[str]] = defaultdict(list)
    for cls in classes:
        for name in pc.O365_AMBIGUOUS.get(cls, ()):
            ambiguous[name].append(cls)
    unknown = sorted((set(definite) | set(ambiguous)) - set(cat_name.values()))
    if unknown:
        raise SystemExit(f"o365_better: {path.name} has no category named {unknown}; check the spelling")

    plain: dict[int, dict[str, list[list[float]]]] = defaultdict(dict)
    doubt: dict[int, dict[str, list[list[float]]]] = defaultdict(dict)
    reasons: dict[tuple[int, str], set[str]] = defaultdict(set)
    dims: dict[int, tuple[int, int]] = {}
    filenames: dict[int, str] = {}
    for img in data["images"]:
        iid = int(img["id"])
        dims[iid] = (int(img["width"]), int(img["height"]))
        filenames[iid] = img["file_name"]

    for ann in data["annotations"]:
        name = cat_name.get(ann["category_id"])
        cls = definite.get(name)
        maybe = ambiguous.get(name, ())
        if cls is None and not maybe:
            continue
        iid = int(ann["image_id"])
        x, y, w, h = ann["bbox"]
        box = [x, y, x + w, y + h]
        why = {f for f in FLAGS if ann.get(f)} | ({"no_area"} if w <= 0 or h <= 0 else set())
        if cls is not None:
            if why:
                doubt[iid].setdefault(cls, []).append(box)
                reasons[(iid, cls)] |= why
            else:
                plain[iid].setdefault(cls, []).append(box)
        for other in maybe:
            doubt[iid].setdefault(other, []).append(box)
            reasons[(iid, other)].add(f"name:{name}")

    labels: dict[int, dict[str, list[list[float]]]] = {iid: dict(plain.get(iid, {})) for iid in dims}
    doubtful: set[tuple[int, str]] = set()
    for iid, by_cls in doubt.items():
        for cls, boxes in by_cls.items():
            if cls not in labels[iid]:
                labels[iid][cls] = boxes
                doubtful.add((iid, cls))
    reasons = {pair: why for pair, why in reasons.items() if pair in doubtful}
    log(f"  o365_better: {len(labels):,} images, {len(doubtful):,} (image, class) pairs held but doubtful")
    return O365Labels(labels, doubtful, dims, filenames, reasons)


class _Plan(NamedTuple):
    """Everything :func:`load` decides before it reads a single pixel."""

    src: O365Labels
    boxes_for: dict[tuple[int, str], list[list[float]]]
    cells: list[str]
    neg_set: set[int]
    positive_in: dict[int, list[str]]
    answered: set[int]
    emit_ids: set[int]


def banded(
    src: O365Labels,
) -> tuple[dict[str, dict[str, list[int]]], dict[tuple[int, str], list[list[float]]], list[int]]:
    """``(supply, boxes_for, clean)`` for *src*, by the shipped band rule.

    The one call both the build and ``o365_supply.py`` make, so the census
    measures the supply the build will draw from rather than a restatement of it.
    """
    return band_candidates(
        src.labels,
        src.box_dims,
        unbanded=set(),
        classes=pc.SCALE_CLASSES,
        excluded=src.doubtful,
        largest=pc.SCALE_BAND_ON_LARGEST,
    )


def _plan(dataset: str) -> _Plan:
    """Which images the cell holds and what each is labelled -- the whole selection."""
    return plan_from(read_o365_labels(pc.O365_VAL_ANNOTATIONS, pc.SCALE_CLASSES))


def plan_from(src: O365Labels) -> _Plan:
    """:func:`_plan` from labels already read -- the census plans without the staged tree."""
    supply, boxes_for, clean = banded(src)
    answered = set(src.labels)  # Objects365 answered for every category on every image
    chosen = designate_cells(supply, corrections={}, roster={}, dropped=pc.O365_DROPPED_CELLS)
    negatives, spares = draw_negatives(clean, roster={}, coco_scored=answered, coco_fraction=1.0)

    cells = sorted(chosen)
    short = [c for c in cells if len(chosen[c]) < pc.SCALE_N_POS]
    log(
        f"  o365_better: {len(cells)} cells, {sum(len(v) for v in chosen.values()):,} positives, "
        f"{len(negatives):,} negatives (+{len(spares):,} spares) from {len(clean):,} clean"
    )
    if short:
        log(f"  o365_better: WARNING {len(short)} cells under SCALE_N_POS and not in O365_DROPPED_CELLS: {short}")

    positive_in: dict[int, list[str]] = defaultdict(list)
    for cell, ids in chosen.items():
        for iid in ids:
            positive_in[iid].append(cell)
    neg_set = set(negatives)
    emit_ids = set(positive_in) | neg_set | set(spares)
    return _Plan(src, boxes_for, cells, neg_set, positive_in, answered, emit_ids)


def label_fields(plan: _Plan, iid: int) -> dict:
    """The label half of *iid*'s media, exactly as :func:`load` writes it, without its pixels."""
    return scale_label_fields(
        iid=iid,
        box_dims=plan.src.box_dims[iid],
        cats=sorted(plan.positive_in.get(iid, [])),
        boxes_for=plan.boxes_for,
        cells=plan.cells,
        neg_set=plan.neg_set,
        labels=plan.src.labels,
        coco_scored=plan.answered,
        exhaustive=plan.answered,
        reviewed_absent=set(),
        reviewed_present=set(),
    )


def load(dataset: str, medias: dict[int, dict], embedder_name: str) -> None:
    """Populate *medias* with the designated positives, the pool, and its spares."""
    plan = _plan(dataset)
    src = plan.src
    missing = 0
    log(f"  o365_better: emitting {len(plan.emit_ids):,} medias from {pc.O365_VAL_DIR}")
    for iid in sorted(plan.emit_ids):
        path = pc.O365_VAL_DIR / src.filenames[iid]
        if not path.exists():
            missing += 1
            continue
        media = scale_media(
            iid=iid,
            data=path.read_bytes(),
            filename=path.name,
            origin_name=str(path),
            box_dims=src.box_dims[iid],
            cats=sorted(plan.positive_in.get(iid, [])),
            boxes_for=plan.boxes_for,
            cells=plan.cells,
            neg_set=plan.neg_set,
            labels=src.labels,
            coco_scored=plan.answered,
            exhaustive=plan.answered,
            reviewed_absent=set(),
            reviewed_present=set(),
            embedder_name=embedder_name,
            importer="o365_better",
            labels_source="objects365",
        )
        if media is not None:
            medias[iid] = media
    if missing:
        log(f"  o365_better: WARNING {missing} designated images absent from {pc.O365_VAL_DIR}")


def check(dataset: str) -> str:
    """What an ``o365_better`` rebuild reads: the annotation file and all 44 patch directories.

    Every patch, because the designated draw is spread across all of them by
    hash rank, and a canary that checks a sample of what the build opens is a
    canary that passes on a half-staged tree (#3299).
    """
    paths = [pc.O365_VAL_ANNOTATIONS] + [pc.O365_VAL_DIR / d for d in pc.O365_VAL_PATCH_DIRS]
    for path in paths:
        if not path.exists():
            raise SystemExit(f"{dataset}: missing {path}")
    return f"Objects365 v2 val annotations + {len(pc.O365_VAL_PATCH_DIRS)} patch directories present"
