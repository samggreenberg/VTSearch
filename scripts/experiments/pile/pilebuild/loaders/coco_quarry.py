"""``coco_quarry``: `vg_scale`'s question, asked of COCO 2017 with no Visual Genome.

The same 25 classes, the same three bands, the same shared negative pool — drawn
from COCO 2017 train+val (123,287 images) instead of the half of Visual Genome
COCO happens to have sourced. #3983 measured that this supplies every one of the
75 cells at the shipped ``SCALE_N_POS``, with the thinnest at 1.8x the floor.

**What this loader does NOT do is the point of it.** `vg_scale` spends most of
its length repairing an annotation source that cannot answer the question asked
of it: :func:`~pilebuild.loaders.vg_scale.canonicalise` and
:data:`pile_config.SCALE_VG_NAMES` guess which free-text spellings mean the
class, :func:`~pilebuild.loaders.vg_scale.lift_ambiguous` withholds the ones that
might not, :func:`~pilebuild.loaders.vg_scale.anchor_to_coco` replaces VG's
labels with COCO's wherever COCO annotates, and 4,709 correction rows patch what
is left. COCO annotates all eighty classes exhaustively on every image in its own
vocabulary, so **none of that apparatus has a question to answer here**: no name
tables, no fold-in, no ambiguity lifting, no anchoring, no corrections file.

**What it reuses is also the point.** :func:`~pilebuild.loaders.vg_scale.band_for`,
:func:`~pilebuild.loaders.vg_scale.band_candidates`,
:func:`~pilebuild.loaders.vg_scale.designate_cells` and
:func:`~pilebuild.loaders.vg_scale.draw_negatives` are imported unchanged, and the
media dict is built by the shared :func:`~pilebuild.loaders.vg_scale.scale_media`.
A cell here and a cell there are comparable only if the rule that banded them is
the same object rather than the same intention — the reason `band_for` was split
out in the first place, and the reason `coco_only_supply.py` imports it rather
than restating it.

**``iscrowd`` regions are dropped, explicitly.** A crowd box is a region, not an
instance: admitting one would let a single annotation band an image by an extent
no user would ever drag, which is #3985's defect arriving through the front door.
#3983's census made the same choice, so the supply this builds matches the supply
that was measured. VG has no equivalent concept, so this has no counterpart in
the older loader — it is a decision COCO makes available, not a divergence.

**A patch column is sharded, and that is a memory limit rather than a taste.**
``build_pile.py`` assembles a whole cell in RAM and writes it with one
``pickle.dump``, and every consumer reads it back through ``load_medias`` into a
single dict. Measured: the full corpus holds 20.9 GB of pixel bytes at peak, and
a full-corpus ``dinov3_patch`` cell is a further ~37 GB of grids (123,287 x
301,056 bytes, read off a built cell) plus a ~37 GB thinned copy at dump. Cut
into shards each is ~4.7 GB -- the size of the ``vg_scale`` patch cell that
already works -- and a study loads only the shards it needs. The single-vector
columns need none of this and stay whole.

**Every image is ``coco_scored``.** In `vg_scale` that flag separates images
where "holds none of *C*" is a fact from those where it is VG's silence, and the
negative pool is stratified on it (#3670). Here the distinction is empty: COCO
answered for all eighty classes on every image in the corpus, so the pool is
provable by construction and ``coco_fraction`` is 1.0 rather than a composition
to choose.
"""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from pathlib import Path

import pile_config as pc

from pilebuild.env import log
from pilebuild.loaders.vg_scale import band_candidates, designate_cells, draw_negatives, scale_media

#: The two splits, in the order the census read them.
SPLITS = ("val2017", "train2017")

#: ``(anchor, classes) -> (labels, box_dims, filenames)``. The builder calls
#: :func:`load` once per embedder in one process, and re-reading 490 MB of
#: annotations and re-banding 123,287 images five times is 2.5 minutes of the
#: designated build and proportionally more of a full-corpus one. The inputs are
#: files on disk that do not change mid-run, so the answer is cached rather than
#: recomputed.
_CORPUS: dict[tuple[str, tuple[str, ...]], tuple[dict, dict, dict]] = {}


def read_coco_labels(
    anchor: Path, classes: tuple[str, ...]
) -> tuple[dict[int, dict[str, list[list[float]]]], dict[int, tuple[int, int]], dict[int, str]]:
    """``(labels, box_dims, filenames)`` over both 2017 splits, in pixel space.

    *labels* is ``{image_id: {class: [[x0, y0, x1, y1], ...]}}`` and carries an
    entry for **every** image in the corpus, including those holding none of
    *classes*: an image with no entry is "no answer", while an image with an
    empty one is "COCO looked and there is nothing here". `vg_scale`'s
    :func:`_evaluable` turns on exactly that difference, and the two were
    spellable as the same value for one commit (#3667).

    ``iscrowd`` regions are skipped — see this module's docstring.
    """
    key = (str(anchor), tuple(classes))
    if key in _CORPUS:
        return _CORPUS[key]

    labels: dict[int, dict[str, list[list[float]]]] = {}
    dims: dict[int, tuple[int, int]] = {}
    filenames: dict[int, str] = {}
    wanted = set(classes)
    crowd = 0

    for split in SPLITS:
        path = anchor / f"instances_{split}.json"
        if not path.exists():
            raise SystemExit(f"coco_quarry: missing {path}; run coco_anchor.py --fetch")
        with path.open() as fh:
            data = json.load(fh)
        cats = {c["id"]: c["name"] for c in data["categories"]}
        for img in data["images"]:
            iid = int(img["id"])
            dims[iid] = (int(img["width"]), int(img["height"]))
            filenames[iid] = img["file_name"]
            labels.setdefault(iid, {})
        for ann in data["annotations"]:
            if ann.get("iscrowd"):
                crowd += 1
                continue
            name = cats.get(ann["category_id"])
            if name not in wanted:
                continue
            x, y, w, h = ann["bbox"]
            labels[int(ann["image_id"])].setdefault(name, []).append([x, y, x + w, y + h])

    log(f"  coco_quarry: {len(labels):,} images, {crowd:,} iscrowd regions dropped")
    _CORPUS[key] = (labels, dims, filenames)
    return labels, dims, filenames


def _cells_of(cls: str, supply: dict[str, dict[str, list[int]]]) -> dict[str, list[int]]:
    """``{"class@band": ids}`` for one class, straight off the banded supply.

    `designate_cells` caps each cell at ``SCALE_N_POS`` and prefers reviewed
    membership; neither applies when the whole corpus is embedded, because the
    cap is the thing being deferred to export time.
    """
    return {f"{cls}@{band}": ids for band, ids in supply.get(cls, {}).items() if ids}


def _zip_members() -> dict[str, tuple[Path, str]]:
    """``{basename: (zip, member)}`` across both staged archives.

    Both are read in place. The train archive's 118,287 members are STORED, not
    deflated, so a member read is a seek and a raw read; extracting would cost
    19.3 GB and ~11 minutes to save about five minutes of a six-hour embed
    (#3991). A name resolves in exactly one archive — the splits are disjoint.
    """
    members: dict[str, tuple[Path, str]] = {}
    for zip_path in (pc.COCO_VAL_ZIP, pc.COCO_TRAIN_ZIP):
        if not zip_path.exists():
            raise SystemExit(f"coco_quarry: missing {zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".jpg"):
                    members[Path(name).name] = (zip_path, name)
    return members


def load(dataset: str, medias: dict[int, dict], embedder_name: str) -> None:
    """Populate *medias* with the designated positives, the pool, and its spares."""
    classes = pc.SCALE_CLASSES
    labels, box_dims, filenames = read_coco_labels(pc.COCO_ANCHOR_DIR, classes)

    # No corrections and no roster: there is nothing to repair and no prior
    # review whose membership has to be preserved. Both helpers take the empty
    # case rather than a separate code path.
    supply, boxes_for, clean = band_candidates(labels, box_dims, unbanded=set(), classes=classes)
    coco_scored = set(labels)  # every image; COCO answered for all eighty

    full = bool(pc.DATASETS.get(dataset, {}).get("full_corpus"))
    if full:
        # Every image in the corpus, and no draw. A designated cell freezes
        # SCALE_N_POS and SCALE_N_NEG into the embedding, so changing either --
        # or the prevalence #3987 wants down to 0.1%, which needs ~100k
        # negatives against 100 positives -- costs a re-embed of everything.
        # Embedded whole, a cell is a FILTER over a fixed set and those become
        # export-time queries, which is the point of the quarry.
        chosen = {cell: list(ids) for cls in supply for cell, ids in _cells_of(cls, supply).items()}
        negatives, spares = sorted(clean), []
    else:
        chosen = designate_cells(supply, corrections={}, roster={})
        negatives, spares = draw_negatives(clean, roster={}, coco_scored=coco_scored, coco_fraction=1.0)

    cells = sorted(chosen)
    short = [c for c in cells if len(chosen[c]) < pc.SCALE_N_POS]
    log(
        f"  coco_quarry: {len(cells)} cells, {sum(len(v) for v in chosen.values()):,} positives, "
        f"{len(negatives):,} negatives (+{len(spares):,} spares)"
    )
    if short:
        log(f"  coco_quarry: WARNING {len(short)} cells under SCALE_N_POS: {', '.join(short[:6])}")

    members = _zip_members()
    positive_in: dict[int, list[str]] = defaultdict(list)
    for cell, ids in chosen.items():
        for iid in ids:
            positive_in[iid].append(cell)
    neg_set = set(negatives)

    # Resolve every id to its archive FIRST, so an image absent from both is
    # counted once rather than once per archive, and each zip is opened once.
    by_zip: dict[Path, list[tuple[int, str]]] = defaultdict(list)
    missing = 0
    # In full-corpus mode the emit set is the CORPUS, not the union of the draws.
    # `band_candidates` returns banded supply and the clean pool; an image that
    # holds a class in no valid band -- scattered, or oversize -- is in neither,
    # so taking the union here would drop it and call the result "everything".
    emit_ids = set(labels) if full else (set(positive_in) | neg_set | set(spares))

    # A shard is a slice of the EMIT set, never of the question. Supply, banding
    # and the clean pool above are computed over the whole corpus and only then
    # filtered, because a cell computed within a shard would be a different cell:
    # `band_for` is per image, but "holds none of C" and every count are not.
    # Modulo rather than a contiguous range so each shard carries every class and
    # band in proportion, which makes one shard a usable sample on its own.
    shard = pc.DATASETS.get(dataset, {}).get("shard")
    if shard:
        index, count = shard
        emit_ids = {iid for iid in emit_ids if iid % count == index}
    where = f" (FULL CORPUS shard {shard[0]}/{shard[1]})" if shard else (" (FULL CORPUS)" if full else "")
    log(f"  coco_quarry: emitting {len(emit_ids):,} medias{where}")
    for iid in sorted(emit_ids):
        found = members.get(Path(filenames[iid]).name)
        if found is None:
            missing += 1
            continue
        by_zip[found[0]].append((iid, found[1]))

    for zip_path, items in sorted(by_zip.items()):
        with zipfile.ZipFile(zip_path) as zf:
            for iid, member in items:
                media = scale_media(
                    iid=iid,
                    data=zf.read(member),
                    filename=Path(filenames[iid]).name,
                    origin_name=f"{zip_path}::{member}",
                    box_dims=box_dims[iid],
                    cats=sorted(positive_in.get(iid, [])),
                    boxes_for=boxes_for,
                    cells=cells,
                    neg_set=neg_set,
                    labels=labels,
                    coco_scored=coco_scored,
                    exhaustive=coco_scored,
                    reviewed_absent=set(),
                    reviewed_present=set(),
                    embedder_name=embedder_name,
                    importer="coco_quarry",
                )
                if media is not None:
                    medias[iid] = media

    if missing:
        log(f"  coco_quarry: WARNING {missing} designated images absent from the archives")


def check(dataset: str) -> str:
    """What a ``coco_quarry`` rebuild reads, named as the loader names them.

    The annotations and **both** image archives. A canary that checks a path the
    build does not open is not a canary (#3299), and this build opens the train
    archive that #3991 found staged and unreferenced.
    """
    paths = [pc.COCO_ANCHOR_DIR / f"instances_{s}.json" for s in SPLITS]
    paths += [pc.COCO_VAL_ZIP, pc.COCO_TRAIN_ZIP]
    for path in paths:
        if not path.exists():
            raise SystemExit(f"{dataset}: missing {path}")
    return "COCO 2017 annotations + both image zips present"
