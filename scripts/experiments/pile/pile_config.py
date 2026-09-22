"""The shared pre-embedded pile: which ``(dataset, embedder)`` cells exist and where.

A *cell* is one ``<dataset>__<embedder>.pkl`` under the pile's
``embeddings/`` dir — the per-pair artifact every study loads instead of
re-embedding. Studies point ``VTSEARCH_DATA_DIR`` at the pile and read the
cells in place; nothing here is study-specific.

**Reproducibility.** The pile lives on scratch, which is treated as purgeable,
so every cell must be rebuildable from sources that are *not* on scratch:

* ``visual_genome_m`` / ``caltech101_m`` are VTSearch demo datasets, downloaded
  into the shared demo cache (``DEMO_CACHE``) and loaded by ``load_demo_dataset``.
* ``coco_val`` is not a demo dataset; it is assembled from the COCO-2017-val
  images and the flattened annotations staged under ``COCO_ROOT``.

Because ``_cells_io.dump_medias`` drops ``media_bytes``, a cell holds vectors
(plus ``patch_grid`` for patch embedders) and no pixels — so the pile is small
relative to its sources and a rebuild always re-reads the staged originals.

**Region voting.** Only patch embedders emit ``patch_grid``; a boxed dataset
paired with a single-vector embedder silently degrades to binary voting. That
mis-specification has burned three studies (#2877, #2897, #2905), so
:func:`region_capable` states it per *cell* rather than per dataset, and
``build_pile.py --verify`` asserts the geometry is actually present.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import NamedTuple

USER = os.environ.get("USER", "sgreenberg")

#: Root of the shared pile. Everything below is derived from it.
PILE = Path(os.environ.get("VTS_PILE", f"/expscratch/{USER}/vts-cache"))
DATADIR = PILE / "datadir"
EMBEDDINGS = DATADIR / "embeddings"
MODELS = PILE / "models"

#: Shared, non-scratch sources the pile is rebuilt from.
DEMO_CACHE = Path(os.environ.get("VTS_DEMO_CACHE", "/exp/scale26/datasets/external/vtsearch-demos"))
COCO_ROOT = Path(os.environ.get("VTS_COCO_ROOT", "/exp/scale26/datasets/external/COCO"))
#: The zip the builder reads pixels out of. This, not an extracted directory,
#: is what a `coco_val` rebuild actually depends on -- the staging area holds
#: `val2017.zip` and has never held `val2017/`. Named here because it was
#: previously spelled inline in the builder while :data:`COCO_IMAGES` named the
#: directory, and the rebuild canary checked the directory: it reported
#: `coco_val` REBUILD-BROKEN against a source that was present and fine (#3299).
COCO_VAL_ZIP = COCO_ROOT / "images" / "val2017.zip"
#: Where the images live *if* somebody extracts them. Optional, and not part of
#: the rebuild path: nothing depends on this directory existing. `box_sheets.py`
#: prefers it (a loose JPEG is cheaper than a zip member) but falls back to
#: :data:`COCO_VAL_ZIP`, which is where the pixels have always actually been --
#: it used to read only this path and drew empty sheets instead (#3305).
COCO_IMAGES = COCO_ROOT / "images" / "val2017"
COCO_ANNOTATIONS = COCO_ROOT / "derived" / "objects_flat_val2017.jsonl.gz"

#: COCO 2017 **train** pixels -- 118,287 images, 94% of `coco_quarry`'s corpus
#: (#3983, #3991). Staged in the same shared tree as the val zip and referenced
#: nowhere in this repo until now, which is why #3991 was filed believing an
#: 18 GB fetch was still owed: it is not, the archive has been there since July.
#:
#: **Read members in place; do not extract.** Measured 2026-09-18: all 118,287
#: members are STORED, not deflated (ratio 1.000), so a read is a seek and a raw
#: read with no inflate. Read + decode costs 6.4 ms/image from the zip against
#: 4.0 ms from an extracted tree -- about 13 min versus 8 min over the whole
#: corpus, against an embed run of ~6 hours (#3988). Extraction would pay ~11
#: minutes and **19.3 GB** of scratch to save ~5 minutes, i.e. 1.4% of the run,
#: on the filesystem that run needs for its own 36 GB of output.
#:
#: `zipfile.ZipFile` is not thread-safe, so a parallel reader wants one handle
#: per worker. That is cheap: parsing the 118k-entry central directory costs
#: well under a tenth of a second.
COCO_TRAIN_ZIP = COCO_ROOT / "images" / "train2017.zip"
#: Where the train images live *if* somebody extracts them anyway. Optional and
#: not part of any rebuild path, exactly like :data:`COCO_IMAGES` -- and carrying
#: the same warning, because #3299 was a canary checking this kind of directory
#: while the builder opened the zip.
COCO_TRAIN_IMAGES = COCO_ROOT / "images" / "train2017"

#: The COCO 2017 *annotations* -- `instances_train2017.json` and
#: `instances_val2017.json` -- as `coco_anchor.py --fetch` stages them. Named
#: here because `coco_quarry` reads them at build time and a build input spelled
#: only in an `--anchor-dir` argument is a build input nobody can check (#3299).
#:
#: Not the same file as :data:`COCO_ANNOTATIONS`, which is the val-only flattened
#: `.jsonl.gz` the `coco_val` loader reads. Two sources, two names, on purpose.
COCO_ANCHOR_DIR = Path(os.environ.get("VTS_COCO_ANCHOR_DIR", str(PILE / "coco_anchor")))

#: How many shards the full-corpus patch column is cut into. Chosen so each cell
#: lands near 4.7 GB, which is what `vg_scale__dinov3_patch.pkl` already is and
#: therefore a size the harness is known to load.
COCO_QUARRY_SHARDS = 8

#: Datasets in the pile. ``boxed`` means the medias carry ground-truth region
#: boxes, which is what a region-voting arm drags — necessary but not
#: sufficient (the embedder must also be patch-capable; see region_capable).
#: ``source_dir`` is the demo extraction dir the loader treats as "already
#: downloaded" (vtscore/datasets/downloader/*.py). It must be present in the
#: datadir before a demo cell is built — see :func:`require_demo_source`.
DATASETS: dict[str, dict] = {
    "visual_genome_m": {"boxed": True, "kind": "demo", "source_dir": "visual_genome"},
    "caltech101_m": {"boxed": False, "kind": "demo", "source_dir": "caltech-101"},
    "coco_val": {"boxed": True, "kind": "coco"},
    # `vg_scale`'s question asked of COCO 2017 with no Visual Genome at all
    # (#3983): the same 25 classes and three bands, drawn from 123,287 images
    # instead of the half of VG that COCO sourced. A separate dataset rather
    # than a rebuild of `vg_scale`, because rebuilding a cell silently changes
    # what it is and five-plus shipped studies are conditioned on the VG-built
    # ones -- they stay readable under their own name, and nothing new is built
    # on them.
    "coco_quarry": {"boxed": True, "kind": "coco_quarry"},
    # The same corpus with NO designation draw: every COCO 2017 image embedded,
    # so a cell is a filter over a fixed set rather than a build-time choice.
    # That is what lets `SCALE_N_POS`/`SCALE_N_NEG` and the prevalence axis
    # (#3987) become export-time queries instead of re-embeds. `on_request`
    # because it is ~6.8x the designated build and nobody wants it by default.
    #
    # Single-vector columns only, and that is a measured limit rather than a
    # preference: `build_pile.py` holds the whole cell in RAM and writes it in
    # one `pickle.dump`, and every consumer reads it back through `load_medias`
    # into one dict. A full-corpus `dinov3_patch` cell is ~37 GB of patch grids
    # -- unwritable at any sane `--mem` once the thinned copy is counted, and
    # unreadable afterwards. The patch column stays on `coco_quarry`.
    "coco_quarry_full": {"boxed": True, "kind": "coco_quarry", "full_corpus": True, "on_request": True},
    # The patch column over the same full corpus, in shards. One cell would be
    # ~37 GB of grids: `build_pile.py` holds a whole cell in RAM and writes it
    # with one `pickle.dump`, and every consumer reads it back through
    # `load_medias` into one dict, so a single cell is unwritable at any sane
    # `--mem` and unreadable afterwards. At COCO_QUARRY_SHARDS each is ~4.7 GB --
    # the size of the `vg_scale` patch cell studies already load.
    #
    # The shard is a slice of the EMIT set only. Supply, banding and the clean
    # pool are computed over the whole corpus first, so a cell means the same
    # thing in every shard; the split is by `image_id % n`, so each carries every
    # class and band in proportion rather than an arbitrary contiguous range.
    **{
        f"coco_quarry_full_s{i}": {
            "boxed": True,
            "kind": "coco_quarry",
            "full_corpus": True,
            "on_request": True,
            "shard": (i, COCO_QUARRY_SHARDS),
        }
        for i in range(COCO_QUARRY_SHARDS)
    },
    # Box-size-banded VG, drawn from the WHOLE source (all 108k images, full
    # free-text vocabulary) rather than the demo pipeline's 100 curated
    # categories on a 4% slice.  The `_s`/`_m`/`_l` on `visual_genome_*` is a
    # dataset *size* tier and says nothing about boxes; these are the box bands.
    # The same-class-across-bands set (#3156). One pickle, one class list, one
    # negative pool; the band lives on the category name (`bus@small`). Not a
    # replacement for `vg_box_*` -- those measured what they measured and stay
    # reproducible -- but the two are not comparable: disjoint vocabularies
    # against a fixed one.
    #
    # Drawn from the half of VG that COCO sourced, and labelled from COCO's
    # exhaustive annotation rather than VG's free text, because VG's own labels
    # cannot support the construction: measured on this pool its recall over C
    # is 0.76, and 1.4% of the images it calls negative actually hold the object
    # (`coco_anchor.py`). At 80 positives per cell that would be ~54 hidden
    # positives sitting in the negatives.
    # `vg_scale` with the box-size band collapsed away (#3115): the same images,
    # boxes and corrections, keyed on the bare class.  A calibration study wants
    # uniform prevalence across cells and does not care how big the box is;
    # `visual_genome_m` gives neither (25 to 1645 positives, and its thin
    # categories produce cells with no trainable step at all).
    #
    # DERIVED from the built `vg_scale` pickle, so it must be listed AFTER it -
    # and so it inherits whatever that cell currently holds.  #3252 changed how
    # `vg_scale` selects and corrects its cells, which means a `vg_scale_any`
    # built before that commit is NOT the same dataset as one built after it.
    # Rebuild it whenever `vg_scale` is rebuilt.  That used to be a rule nobody
    # could check: `--force` on `vg_scale` alone left this cell holding the old
    # labels with the right media count and the right vectors, so it looked
    # healthy (#3281 shipped a box repair to one study and not the other).  It
    # is now enforced twice -- `build_pile.py` pulls this dataset into any run
    # that rebuilds its parent, and `--verify` compares the parent-label digest
    # stamped on each derived media against the parent's live one.
    # `vg_scale_any`'s construction with the BAND DROPPED FROM SELECTION rather
    # than from the key, and sized for a long labelling session (#3547).
    #
    # `vg_scale_any` collapses `class@band` after the fact, so it inherits
    # `vg_scale`'s per-band designation and is capped by the THINNEST band:
    # `bus@small` has 138 candidates, which is why 100/band was the ceiling and
    # 300 positives per class the result.  A study that never asks about box
    # size does not need that cap.  Designating band-free off the same
    # COCO-anchored labels takes the binding class from 414 candidates to 1006
    # (`stop sign`), which is what makes a 400-click horizon measurable at all:
    # #3319's deep wave harvested 82-85% of its ~150 sim positives.
    #
    # Prevalence is held at `vg_scale`'s designed 7.14% BY CONSTRUCTION rather
    # than inherited by accident -- `SCALE_DEEP_N_NEG` is derived from
    # `SCALE_DEEP_N_POS`, not set beside it.  That is the whole point: the
    # optimum this dataset exists to locate is `k* = -log2((1-pi)/pi)`, so
    # adding positives against a FIXED negative pool would move the answer
    # (300->900 against 3900 shifts pi to 18.8% and k* by a full bit) while
    # appearing to be nothing but "a deeper haystack".
    #
    # `on_request`: this one is 3x `vg_scale`'s media count, so it stays OUT of
    # the default sweep. A bare `build_pile.py` would otherwise quietly add five
    # cells nobody asked for, one of them a ~7 GB `dinov3_patch` grid. Name it
    # to build it: `--datasets vg_scale_deep --embedders siglip`.
}

#: Box-size bands, as a fraction of image area, anchored to the patch
#: embedder's geometry (the same anchors the calibration harness bands on):
#: one DINOv3 patch is 1/196 of the image and the smallest HAC leaf is 1/12.
#: ``small`` is therefore "below what the patch grid can resolve at all".
#: Whether an image that is a positive for one class may serve as a NEGATIVE
#: for a class it does not hold (#3667).
#:
#: The construction originally said: positive for its own cells, negative only
#: if it holds nothing in *C*, excluded otherwise. The "excluded otherwise" is
#: right for the SAME class at another size -- scoring a large-bus image as a
#: small-bus negative penalises a detector for finding a real bus -- but it was
#: applied to every OTHER class too, where the reason does not hold. The cost
#: was 41.9% of the pile dropped from every class's evaluation, and negatives
#: that contain none of the classes in *C* while positives contain one, so a
#: detector could score by learning "is this a scene with stuff in it". Measured
#: against twelve; the shortcut can only sharpen at twenty-five, since every
#: class added to *C* is one more thing a shared negative may not hold.
#:
#: Gated on ``labels_exhaustive``: COCO annotates all eighty of its classes on
#: any image it annotates, so absence is a fact there. On the other half absence
#: is VG's silence, measured wrong 0.5-2.5% of the time per class (#3588), and
#: importing that into the negatives is a different decision from this one.
SCALE_CROSS_CLASS_NEGATIVES = True

#: The upper cut mirrors ``MAX_VOTED_AREA``: a box covering >80% of the image
#: is not a region, it is the image.
#:
#: **The band is a VIEW over every instance, not a replacement for them, and
#: that is a decision** (2026-09-07).
#: :func:`~pilebuild.loaders.vg_scale.band_for` summarises a class's boxes in an
#: image by their **union**, which is what the harness's SIMULATED Good vote
#: drags and is what #3156 measured -- and the build keeps every instance box
#: behind it
#: (``band_candidates`` stores them all; ``_emit_medias`` writes one region per
#: box). Keep it that way. The union is derivable from the instances and the
#: instances are not derivable from the union, so a summary chosen at eval time
#: -- largest, smallest, count, density -- stays available at no cost, while
#: collapsing the set at build time would end that permanently and silently.
#:
#: **The union is a modelling choice, not a reproduction of the app** (2026-09-18).
#: A real Good vote carries at most ONE box -- ``MediaVoteRequest.region_box`` is
#: four numbers and ``good_region_boxes`` holds one box per media -- drawn by a
#: person around one object, plausibly the most prominent. The union is what
#: :func:`vtscore.eval.labels.region_box_for_category` returns when it has to
#: invent that drag from *N* ground-truth boxes, and its own docstring justifies
#: it only against picking one **arbitrarily** ("would depend on annotation
#: order"), which does not rule out picking the **largest**. So the union is an
#: assumption about the simulated user that nothing has validated against a real
#: one. It is not eval/app drift -- there is no app function to mirror, because
#: the app has a human -- but it is not a fact about the app either, and this
#: file used to say it was.
#:
#: The shipped band statistic does **not** move as a result: changing it now
#: would restate what #3156 measured under its own name. What the decision buys
#: is the option, and what it costs is one invariant to defend -- a review that
#: replaces a class's instance set with one drawn box spends the option without
#: anyone deciding to (#3726).
PATCH_AREA = 1 / 196
LEAF_AREA = 1 / 12
MAX_VOTED_AREA = 0.80
BOX_BANDS: dict[str, tuple[float, float]] = {
    "small": (0.0, PATCH_AREA),
    "medium": (PATCH_AREA, LEAF_AREA),
    "large": (LEAF_AREA, MAX_VOTED_AREA),
}

#: How many categories each banded dataset draws, and the image cap.  Categories
#: are stratified *within* the band so a band is not silently all one size.
BAND_N_CATEGORIES = int(os.environ.get("VTS_BAND_N_CATEGORIES", "40"))
BAND_MAX_IMAGES = int(os.environ.get("VTS_BAND_MAX_IMAGES", "12000"))
#: Categories whose union box is much larger than a single instance are
#: scattered instances, not a region a user would drag.
BAND_MAX_INFLATION = float(os.environ.get("VTS_BAND_MAX_INFLATION", "1.5"))
BAND_MIN_IMAGES = int(os.environ.get("VTS_BAND_MIN_IMAGES", "50"))

#: LVIS v1 annotations, BOTH splits (#3985). LVIS re-annotated COCO's own images
#: with one box per instance, so it is the referee for a COCO box drawn round a
#: pile. Val alone is only 16% of COCO and left every fruit cell at 3-54 of 100
#: positives; train and val together cover 119,979 of 123,287 images.
LVIS_DIR = Path(os.environ.get("VTS_LVIS_DIR", str(PILE / "lvis")))
LVIS_SPLITS = ("train", "val")

#: Band each (image, class) on its LARGEST instance, not the union of all of them,
#: and record only that box as the cell's region, so the simulated user drags one
#: object (#4096). Under the union rule an image whose instances are spread out
#: (union > ``BAND_MAX_INFLATION`` x the largest) was SCATTERED and never a
#: positive: 111,324 of 287,303 image-class pairs in C, 39%, and 58% of `book`.
#: Positives then leaned toward images holding one isolated instance and left
#: out busy scenes -- a street of cars, a table of cups. Owner ruling, 2026-09-22:
#: use the most obvious instance, which is the largest.
SCALE_BAND_ON_LARGEST = True

#: Classes whose picked COCO box must be ONE object by LVIS's reckoning before
#: the image may be a positive, mapped to the LVIS names that count as the same
#: object (#3985).
#:
#: COCO draws one box round a bunch of bananas, a bowl of apples or a stack of
#: books, and the band is then the size of the pile, not of ONE object -- the
#: unit the owner ruled (`ClassRule.unit`). The test is on the box the simulated
#: user drags (:data:`SCALE_BAND_ON_LARGEST`): keep it only if EXACTLY ONE LVIS
#: instance lies inside it (:data:`SCALE_LUMP_CONTAIN`). None means nothing
#: vouches for the box; two or more means LVIS drew several objects where COCO
#: drew one. An excluded image keeps its label, so it is never scored as a
#: negative for its own class -- it is simply not a positive.
#:
#: Scored on the owner's 72 fruit votes, which were cast on exactly this box:
#: 28 of 29 piles caught and 5 of 43 Good boxes lost. The per-image mean-area
#: ratio it replaces (cut 1.8, #4085) caught the same 28 and lost 6, and needed a
#: tuned cut; this needs none.
#:
#: NOT `skis` or `potted plant`, although their ratios are as high: there the
#: owner ruled COCO's box IS the object (a pair; a pot with its plant) and 23 of
#: 24 high-ratio images were voted Good. The name sets are
#: `coco_box_granularity.SAME`'s curated ones: `apple` without `pear`, `orange`
#: without `lemon`.
SCALE_LUMP_FILTER: dict[str, tuple[str, ...]] = {
    "banana": ("banana",),
    "apple": ("apple",),
    "orange": ("orange_(fruit)", "mandarin_orange"),
}
#: Share of an LVIS box that must lie inside the picked COCO box for the LVIS
#: instance to count as inside it.
SCALE_LUMP_CONTAIN = 0.5

#: Cells not built at all, because no honest supply reaches ``SCALE_N_POS``.
#: Built short, their prevalence -- and so their AP -- would not be comparable
#: with any other cell.
SCALE_DROPPED_CELLS: frozenset[str] = frozenset()

#: VG is annotated with free text, so its vocabulary is not a list of objects.
#: A detector asked to find "red" or "front" is measuring nothing, so these are
#: excluded from the banded datasets. The policy is **concrete countable
#: objects only**, which drops three kinds of name:
#:
#: * colours and other attributes -- properties, not things;
#: * frame relations and abstractions -- "front", "group", "object": either a
#:   position in the image or a placeholder the annotator reached for;
#: * mass nouns and unbounded surfaces -- "grass", "sky", "floor": real, but
#:   *stuff* rather than an object with an extent a user would drag a box around.
#:
#: The third group is the aggressive part of the policy and it costs coverage
#: in the large band specifically, because scene-scale stuff is exactly what
#: large boxes are made of. Countable landforms and structures ("tree",
#: "mountain", "building") are deliberately kept.
NON_OBJECT_CATEGORIES: frozenset[str] = frozenset(
    # attributes
    """red blue green yellow orange purple pink brown black white gray grey tan beige
    silver gold golden dark light bright colorful clear blurry shiny""".split()
    # frame relations, abstractions, placeholders
    + """front back side top bottom left right middle center centre corner edge end
    part section area region spot place row line lines stripe stripes pattern design
    shape size distance background foreground surface object objects thing things item
    items stuff group bunch pile set collection image picture photo photograph view
    scene display something other""".split()
    # mass nouns, unbounded surfaces and scene regions
    + """water snow sand dirt mud grass gravel concrete pavement asphalt sky smoke steam
    fog haze shade shadow shadows reflection glare sunlight ice foam liquid air weather
    ground floor flooring wall walls ceiling road roadway street sidewalk pathway path
    field beach ocean sea lake river land terrain lawn grassy lot traffic""".split()
)


def is_object_category(name: str) -> bool:
    """False for VG names that are not concrete countable objects.

    Matches on the **head noun** (the last token), not the whole string, because
    VG's vocabulary is full of modified compounds. Whole-string matching lets
    ``blue sky`` and ``table top`` through while head-noun matching drops them;
    matching *any* token would wrongly drop ``blue jeans``, ``tennis ball`` and
    ``left eye``, whose heads are perfectly good objects.
    """
    tokens = name.replace("-", " ").split()
    if not tokens:
        return False
    return tokens[-1] not in NON_OBJECT_CATEGORIES and name not in NON_OBJECT_CATEGORIES


#: Extra exclusions for a **same-class-across-scale-bands** study, keyed by head
#: noun and carrying the reason. Deliberately separate from
#: :func:`is_object_category`, which defines the published ``vg_box_*`` sets and
#: must keep defining them; this is a stricter policy layered on top for the new
#: construction, so the old numbers stay reproducible.
#:
#: The extra bar exists because a scale study asks two things of a class that
#: mere objecthood does not:
#:
#: * **Its size must be its own.** A part's box is set by its host, so a "small
#:   nose" is just a distant face -- banding it measures the host's distance,
#:   not the object's scale, and the arm silently becomes a different experiment.
#: * **Its absence must be checkable.** The negative pool is ~95% of the images
#:   and rests on "no instance here". For a part that is unverifiable at any
#:   scale: every image with a person has a nose whether or not VG annotated
#:   one, so the negatives are poisoned by construction and no amount of review
#:   fixes it. That is the worst case for the correction pass, not a candidate
#:   for it.
#:
#: Curated, not inferred -- and reported rather than applied silently, because
#: silent automated judgements about VG's vocabulary are what #3156 is about.
SCALE_STUDY_EXCLUSIONS: dict[str, str] = {
    # Individuated only by a host object. Size tracks the host; absence is
    # unverifiable wherever the host appears.
    **dict.fromkeys(
        """nose ear ears eye eyes face head hair mouth lip lips chin cheek forehead eyebrow
        eyebrows neck chest shoulder shoulders arm arms hand hands finger fingers thumb leg legs
        foot feet knee elbow wrist ankle waist hip tail paw paws hoof hooves horn horns tusk tusks
        beak snout mane fur skin tooth teeth tongue mustache moustache beard sideburns""".split(),
        "part",
    ),
    # Parts of artefacts: same two failures, non-anatomical.
    **dict.fromkeys(
        """collar sleeve sleeves cuff pocket zipper hem waistband strap straps buckle handle knob
        spout lid rim brim blade tread stem tip base""".split(),
        "part",
    ),
    # A location rather than a thing: the box has no principled extent (where
    # does an intersection begin?), so its area is an annotator choice and the
    # band it lands in is noise.
    **dict.fromkeys(
        """court courtyard intersection station runway walkway crossing crosswalk driveway alley
        parking lot yard park playground platform entrance exit doorway hallway corridor stairway
        staircase kitchen bathroom bedroom room office restaurant store shop market""".split(),
        "place",
    ),
    # Parts of a plant or structure: same failure as anatomy.
    **dict.fromkeys("""trunk branch twig root roof chimney railing banister step steps""".split(), "part"),
}

#: One string, several objects: "find the trunk in the middleground" is not one
#: question, so the class cannot be scored as one. Matched on the **whole name**
#: rather than the head noun, because a modifier is precisely what resolves the
#: ambiguity -- bare ``bat`` is unusable, ``baseball bat`` is a perfectly good
#: class. (Head-noun matching would reject both, and misreport the reason for
#: ``tree trunk``, which is unfit for being a *part*, not for being ambiguous.)
POLYSEMOUS_NAMES: frozenset[str] = frozenset(
    """trunk bat mouse pitcher crane tie nail bow plate glass iron seal pen""".split()
)

#: A class annotated on more than this share of all images is treated as
#: pervasive: its negative pool is both thin and least trustworthy, since a
#: ubiquitous thing is exactly what an annotator stops bothering to mark. `sky`
#: is the worked example -- 18.8% prevalent as annotated, plainly higher in
#: truth (`docs/experiments/2026-08-12-overview-bench/REPORT.md`). Measured, not listed,
#: because which names are pervasive is a property of the corpus.
PERVASIVE_PREVALENCE = float(os.environ.get("VTS_PERVASIVE_PREVALENCE", "0.10"))


def scale_study_exclusion(name: str) -> str | None:
    """Why *name* is unfit for a scale-band study, or ``None`` if it is fit.

    Head-noun matched, like :func:`is_object_category`, so ``left eye`` and
    ``bus station`` are caught while ``eyeglasses`` and ``gas station wall``
    are judged on their own heads.
    """
    if not is_object_category(name):
        return "non_object"
    if name in POLYSEMOUS_NAMES:
        return "polysemous"
    tokens = name.replace("-", " ").split()
    return SCALE_STUDY_EXCLUSIONS.get(tokens[-1]) or SCALE_STUDY_EXCLUSIONS.get(name)


#: The class list *C* for the same-class-across-bands study (issue #3156).
#:
#: Chosen by the owner on 2026-08-17 from the measured shortlist
#: (``shortlist_scale_classes.py --compact --floor 100``), out of the 24
#: candidates that were simultaneously: supported at >= 100 images in all three
#: bands, free of a measured alias partner and of plural-form ambiguity, and
#: **also a COCO-2017 class**. That last property is what makes the correction
#: pass affordable: COCO val2017 is exhaustively annotated over these names, so
#: VG's miss rate -- and our own annotators' accuracy -- can be scored against it
#: with no extra human review.
#:
#: Deliberately *not* derived at build time from the scan. Which classes a human
#: can annotate consistently is a judgement, and re-deriving it would silently
#: change what the study measures whenever the scan is re-run.
#:
#: **Fifty-three since #4056**, and the widening was made on a rule rather
#: than a judgement: a class is in if it clears ``SCALE_N_POS`` in all three
#: bands. #3983 measured that at 54 of COCO's 80; `wine glass` is held out by
#: owner ruling because :data:`SCALE_CLASS_MERGES` folds it into `cup`, which
#: leaves 53. That is the ONLY exception, and it is about class identity.
#:
#: **Rebuilding across the widening costs a published cell -0.03 AP, and
#: almost all of that is prevalence.** Measured on the rebuilt set against the
#: preserved 25-class build, over the 75 cells that existed before: paired dAP
#: **-0.030 +- 0.004** (`siglip`) and **-0.028 +- 0.004** (`siglip2_l`) as
#: shipped, but **-0.003 / +0.001** once the negative count is held still, and
#: dAUC **+0.002** -- and AUC cannot see prevalence. A rebuilt cell simply
#: carries 23,891 negatives against 16,535.
#:
#: The shared pool IS drawn as *holds none of C*, so a wider C leaves fewer and
#: emptier candidates -- 49,503 clean images at 25 against 16,091 at 53, and the
#: mean count of C-classes held by a clean pool image falls 1.163 -> 0.000.
#: **That mechanism is real and does not reach the benchmark**, because the
#: barren draw is capped at :data:`SCALE_N_NEG` in both builds while #3667's
#: cross-class negatives more than double (6,635 -> 13,991). An earlier
#: measurement varied the barren component alone and read +0.24 AP; that is a
#: pool `coco_quarry` never ships, and the #4056 report records the correction.
#: A design that let the barren pool scale with its candidate set WOULD inherit
#: it, so this constrains any future change to how the pool is sized.
#:
#: :data:`SCALE_CLASSES_25` still freezes the old roster, and the 25-class build
#: is preserved at ``/expscratch/sgreenberg/keep/coco-quarry-25-20260920/``, so
#: a published number can be reproduced rather than merely disclaimed.
#:
#: **Twenty-five since #3588**, and the thirteen were added on the same terms as
#: the first twelve: measured supply, a measured name audit, and a human review
#: of every class before it shipped. What #3588 bought is the *context* axis --
#: the original twelve sampled it by accident, with only `kite` and `boat`
#: scene-exclusive, so within-band cost spanned 9.4x while the band effect it was
#: built to measure spanned 4x. The thirteen add same-scene partners on purpose
#: (`truck`/`car` beside `bus`, `fork`/`spoon` beside `knife`) and widen the hard
#: end; the easy end could not be widened at all, because a class that owns its
#: scene is photographed filling the frame and so fails the small band (#3603).
#:
#: The pre-#3588 twelve are kept as :data:`SCALE_CLASSES_ORIGINAL` -- not for the
#: build, which reads this tuple, but because the published #3618 / #3635 / #3636
#: / #3666 numbers are conditioned on that list, and re-running those scripts
#: against twenty-five would silently restate what they measured.
SCALE_CLASSES: tuple[str, ...] = (
    # #3588 added `truck` beside `car`; #4056 merged them, owner ruling
    # 2026-09-20. COCO carries the boundary well at the IMAGE level -- 0.01%
    # contradictory negatives -- but not at the BOX level, which is the unit
    # REGION VOTING uses: 9.6% of car/truck boxes carry the MINORITY COCO label
    # for their own LVIS object type (107 sedans called `truck`, 50 minivans
    # called `truck`, 23 pickups called `car`). Merging reaches all 203 and needs
    # no LVIS, which covers 16% of the corpus; excluding minivans would have
    # fixed 50 of 203.
    "enclosed road vehicle",
    "clock",
    "bird",
    "boat",
    "umbrella",
    "kite",
    "book",
    "dog",
    "backpack",
    "knife",
    "bicycle",
    "bus",
    "stop sign",
    # Added by #3588, in the order that issue ranked them: the four same-scene
    # partners first, then the nine whose surroundings ARE their negative pool.
    "fork",
    "spoon",
    # `cup` U `wine glass`, owner ruling 2026-09-20 (#4056, resolving #4074).
    # COCO's stem branch WORKS -- mugs and teacups are `cup` 100% of the time,
    # real stemware is `wine glass` 97% -- but the STEMLESS GLASS is not
    # classifiable: `glass_(drink_container)`, 817 boxes and the largest type,
    # is called `cup` 75% and `wine glass` 25%. 36% of the pair's boxes sit in a
    # 20-80% split type and 10.7% carry the minority label, which is the
    # region-voting rate and worse than the vehicles' 9.6%. Holding `wine glass`
    # OUT did not protect `cup` -- `cup` already held 613 plain glasses and 17
    # wineglasses -- it only left 204 identical glasses outside the roster, free
    # to serve as negative regions against positives they cannot be told from.
    "single serving drinking vessel",
    "bowl",
    "bottle",
    "vase",
    "bench",
    "chair",
    "sink",
    "cell phone",
    "fire hydrant",
    # Added by #4056. The rule is the count and nothing else: a class is in if
    # it clears SCALE_N_POS in all three bands, which #3983 measured as 54 of
    # COCO's 80 -- 28 here, because `wine glass` is held out below.
    # Selection deliberately ignores scatter and purity -- both
    # correlate with how hard a class is to detect, so filtering on either
    # would make the benchmark easier and bias every result optimistically.
    # Alphabetical, because provenance is the comment's job, not the order's.
    "airplane",
    "apple",
    "banana",
    "baseball bat",
    "dining table",
    "frisbee",
    "handbag",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parking meter",
    "person",
    "potted plant",
    "remote",
    "scissors",
    "skateboard",
    "skis",
    "snowboard",
    "suitcase",
    "surfboard",
    "tennis racket",
    "tie",
    "toothbrush",
    "traffic light",
    "tv",
    # `wine glass` clears the count rule and is DELIBERATELY NOT HERE (#4056,
    # owner ruling 2026-09-20). SCALE_CLASS_MERGES folds it into `cup`, so
    # admitting it would redefine `cup` and cost every published `cup` number
    # its meaning. The purity measurement says the same from the other side:
    # `cup` is 39% glass_(drink_container) and `wine glass` is 24% of that
    # same synset, so the two are not disjoint classes waiting to be split.
    # This is the ONLY exception to count-only selection, and it is an
    # exception about class IDENTITY, not difficulty -- selecting on scatter
    # or purity stays forbidden.
)

#: The twenty-five *C* held before #4056, frozen as a historical roster.
#:
#: Every published `coco_quarry` measurement is conditioned on a shared
#: negative pool drawn as *holds none of these twenty-five* -- 49,503 clean
#: candidates, where the widened roster leaves 16,058. The pool DRAWN is
#: ``SCALE_N_NEG`` either way, so no cell changes size, but the images in it
#: come from a different candidate set. A script that compares against a
#: published number reads this tuple; the builders read
#: :data:`SCALE_CLASSES`. Same reason :data:`SCALE_CLASSES_ORIGINAL` exists.
SCALE_CLASSES_25: tuple[str, ...] = (
    "clock",
    "bird",
    "boat",
    "umbrella",
    "kite",
    "book",
    "dog",
    "backpack",
    "knife",
    "bicycle",
    "bus",
    "stop sign",
    "truck",
    "car",
    "fork",
    "spoon",
    "cup",
    "bowl",
    "bottle",
    "vase",
    "bench",
    "chair",
    "sink",
    "cell phone",
    "fire hydrant",
)

#: The twelve *C* held before #3588, frozen as a historical roster.
#:
#: Every published measurement of the shared negative pool -- #3618's name
#: coverage, #3635's contamination rates, #3636's pooled families, #3666's pool
#: error for "the shipped twelve" -- is conditioned on a pool drawn as *holds
#: none of these twelve*. Reading those scripts off :data:`SCALE_CLASSES` after
#: the promotion would not re-measure them, it would answer a different question
#: under the old numbers' names. So the scripts that compare *shipped* against
#: *candidate* read this, and the builders read :data:`SCALE_CLASSES`.
#:
#: Not a table anyone should add to. A class joining *C* joins the tuple above.
SCALE_CLASSES_ORIGINAL: tuple[str, ...] = (
    "clock",
    "bird",
    "boat",
    "umbrella",
    "kite",
    "book",
    "dog",
    "backpack",
    "knife",
    "bicycle",
    "bus",
    "stop sign",
)

#: VG spellings that ARE a class in *C*, beyond the class name itself.
#:
#: VG's vocabulary is free text and :func:`pilebuild.vgsource.vg_boxes_by_name`
#: matches an object's PRIMARY name only, so a class built from one spelling
#: silently drops every other. That is not merely a supply loss: on the ~52% of
#: VG that COCO does not annotate, VG's silence is the only evidence of absence,
#: so an instance annotated under an unlisted spelling becomes a **negative** for
#: its own class (#3605). There is no cheaper fix available in the reader --
#: every one of the 2,516,939 objects in this release of VG carries a ``names``
#: list of length **one**, and of the 18,897 objects named by an entry in either
#: table below, **zero** carry the class name further down that list (#3618).
#:
#: Every entry is measured, and folding one makes two claims, so two things are
#: measured (``name_evidence.py``):
#:
#: 1. **the class is present** when this name is its only evidence -- the
#:    *repair precision*: over the VG-COCO overlap, the share of images carrying
#:    the name and NOT the class name where COCO says the class is there anyway.
#:    Read it as a price, in the right units: ``1 / precision - 1`` is how many
#:    **good hard negatives are destroyed per contaminated negative retired**.
#:    Not pool membership -- 77,119 images are eligible against a 4,200-image
#:    draw -- but the images a name withholds are the ones hardest to tell from
#:    the class, which is what makes the ratio the thing to cut on (#3635). The
#:    cut is 1/3 -- two destroyed per repair -- taken on the **Wilson lower
#:    bound**, so a name measured on five images cannot outrank one measured on
#:    two thousand.
#: 2. **this box is the object**: at least half of the name's boxes land on a
#:    COCO box of the class, over at least 20 boxes. A band is a claim about one
#:    object's size (#3616), so a name that passes (1) and fails (2) goes to
#:    :data:`SCALE_VG_AMBIGUOUS` instead -- the safe side, since a wrong
#:    ambiguous costs a few pool images and a wrong alias injects a mis-banded
#:    positive.
#:
#: **Box overlap between two names is not the instrument here** and cannot be.
#: ``scan_name_overlap.py`` needs the two names on one image, and an annotator
#: who writes `back pack` does not also write `backpack`: 846 of 1,740
#: class-vs-candidate pairs never co-occur at all, and where a singular and its
#: plural do co-occur they are deliberately different boxes -- `backpack` /
#: `backpacks` scores **0.000 both ways** over 10 co-images and is called
#: *distinct*. That test keeps its own job, which is the case where two names
#: really do sit on one box (`clock`/`clock face`, 0.562/0.701 over 286) and
#: refuting a lookalike, which is how `bus` survived matching 80 images
#: annotated `bush`.
SCALE_VG_NAMES: dict[str, tuple[str, ...]] = {
    "backpack": ("back pack",),
    # Two names; `benches` clears the box floor exactly, at 50% over 144 boxes.
    "bench": ("benches", "wooden bench"),
    "bicycle": ("bicycles",),
    # COCO annotates ducks, geese and gulls as `bird`, and so does VG under its
    # own species names: each of these is above the cut on both tests.
    "bird": ("duck", "goose", "ostrich", "owl", "parrot", "pigeon", "seagull", "swan"),
    "boat": ("boats", "canoe", "kayak", "raft", "sailboats", "ship"),
    # COCO has no magazine class and annotates magazines as `book`, which is the
    # reading this dataset already took -- see SCALE_CLASS_RULES["book"].
    "book": ("magazine",),
    # +375 repaired. `dish soap`, `handsoap` and `waterbottle` fold on the same
    # reading as the shipped `bottle incl jars` rule: the vessel is a bottle
    # whatever it holds.
    "bottle": (
        "beer bottle",
        "bottles",
        "dish soap",
        "dishsoap",
        "handsoap",
        "plastic bottle",
        "water bottle",
        "waterbottle",
        "wine bottle",
        "wine bottles",
        "winebottle",
    ),
    # `bowls` folds where `books` and `knives` do not: it clears the box test at
    # 59% over 112 boxes, so the plural here is a second bowl rather than a pile.
    # #3636 measures a plural; it does not assume one.
    "bowl": (
        "beige bowl",
        "black bowl",
        "bowls",
        "brown bowl",
        "colorful bowls",
        "gray bowl",
        "green bowl",
        "grey bowl",
        "orange bowl",
        "red bowl",
        "silver bowl",
        "white bowls",
    ),
    # The subtype family pools to 83% over 18 sole images and 82% over 74 boxes,
    # which carries the four route/deck spellings no one of which reached the
    # five-image floor on its own (#3636).
    "bus": ("buses", "city bus", "double-decker bus", "passenger bus", "school bus", "tour bus"),
    # The largest repair in the study: +1,588 images on the non-COCO half against
    # the 5,122 the class could already see, 1,320 of them landing in a band.
    # Overlap coverage 23.2% -> 31.4%.
    #
    # `van` and `vehicle` are NOT here, and both cleared the cuts. They are in
    # SCALE_VG_AMBIGUOUS because a shipped ruling already settled them and the
    # audit cannot see it: COCO splits `van` 261 truck / 318 car / 37 bus, which
    # is the measurement SCALE_CLASS_RULES["truck"] was written from, so 72%
    # precision against COCO `car` is a coin-flip reported as an alias.
    # `vehicle` is a superordinate covering `truck`, `bus` and `bicycle` -- three
    # other classes in C -- so folding it would make their images `car`
    # positives. The two score 51% and 55% box agreement, the weakest here.
    "car": (
        "automobile",
        "black car",
        "blue car",
        "bluecar",
        "car's",
        "cars",
        "dark car",
        "gray car",
        "grey car",
        "jeep",
        "minivan",
        "purple car",
        "red car",
        "sedan",
        "silver car",
        "suv",
        "tan car",
        "taxi",
        "white car",
        "whitecar",
        "yellow car",
    ),
    # The biggest proportional repair anywhere in C: +1,280 images against the 535
    # the class could already see (+239%), and overlap coverage 15.5% -> 44.9%.
    # `phone` is the riskiest entry in this table and stays on measurement rather
    # than comfort -- 68% precision over 884 sole images and 54% box agreement
    # over 1,035 boxes, both above the cut -- against a class `coco_folds.py`
    # scores at 46% definition risk, because VG's `phone` is nearly half
    # landlines. What settles a landline is SCALE_CLASS_RULES["cell phone"], not
    # this table.
    "cell phone": ("cellphone", "cellphone.", "phone"),
    # +361 repaired, 297 banded. `bar stool` folds because the reviewed rule is
    # `chair incl stools not couches`, and the audit agrees at 93% over 15 sole
    # images. `chair back` is NOT here: it is a PART, and a part's box is not the
    # object -- the same reading that put `beak` (86%) and `knife block` (79%) in
    # the ambiguous table at higher precision than this name's 62%.
    "chair": (
        "arm chair",
        "armchair",
        "bar stool",
        "barstool",
        "beach chair",
        "brown chair",
        "chairs",
        "chairs.",
        "folding chair",
        "green chair",
        "grey chair",
        "lawn chair",
        "lawnchair",
        "orange chair",
        "pink chair",
        "purple chair",
        "red chairs",
        "yellow chair",
    ),
    # The face is the clock: 89% of `clock face` boxes land on COCO's clock box,
    # over 184 of them -- the best-supported fold in the study. `clockface` is
    # the same word without the space, and pooled with it scores 89% over 19
    # sole images and 89% over 194 boxes (#3636).
    "clock": ("clock face", "clockface", "clocks"),
    # +386 repaired on the non-COCO half, 345 of them banded.
    #
    # The stemware half is the CLASS MERGE (see SCALE_CLASS_MERGES). It was
    # carried by hand at the #3588 promotion because the audit scored every
    # candidate against COCO `cup` alone -- `wine glass` read 38% precision and
    # 2% box agreement and landed in `neither`, a refusal produced by the scorer
    # looking at the wrong half of the class.
    #
    # #3700 made the audit merge-aware, and it now CONFIRMS the two spellings
    # that carry the mass: `wine glass` at **98%** precision over 151 sole images
    # and **85%** box agreement over 253 boxes, `wine glasses` at 95% / 57%.
    # The other four stay on #3588's direct measurement against COCO
    # `wine glass` boxes, because they sit below the audit's own floors rather
    # than being refused by it: `goblet` clears precision at 70% with 12 boxes
    # against a floor of 20, and `wineglass`, `champagne glass` and
    # `champagne flute` fall under `--min-sole` entirely. `unmeasured` is not
    # `refuted`.
    #
    # `mug` is the cup half's own missing spelling and was never affected: 88%
    # over 193 sole images, 82% over 290 boxes, before and after.
    #
    # `glass` is NOT here, and the audit now proposes that it should be. See
    # SCALE_VG_AMBIGUOUS["cup"] for why it is refused anyway.
    "cup": (
        "black cup",
        "brown cup",
        "champagne flute",
        "champagne glass",
        "coffee cup",
        "coffee mug",
        "coffeemug",
        "goblet",
        "green cup",
        "mug",
        "paper cup",
        "pink cup",
        "plastic cup",
        "plasticcup",
        "white cups",
        "wine glass",
        "wine glasses",
        "wineglass",
        "yellow cup",
    ),
    # `dalmation` (VG's spelling) and `lab` come from the breed group, `white dog`
    # from the colour one: 74% and 91% pooled, both folding on box agreement (#3636).
    "dog": ("black dog", "brown dog", "dalmation", "dogs", "lab", "puppy", "white dog"),
    # The single largest coverage gain in the study: 44.7% -> 73.8% of COCO's
    # `fire hydrant` boxes, and +282 repaired against 411 own on the non-COCO
    # half. `hydrant` alone carries 314 boxes at 87% on class, so the long
    # spelling by itself throws away a third of the class (#3588, #3618).
    "fire hydrant": ("hydrant", 'hydrant"'),
    # The thinnest alias row in C: one name. `silverware` and `utensils` are
    # withheld instead -- their box is a whole place setting, which is the rule
    # `fork` and `knife` already share.
    "fork": ("forks",),
    # COCO's `kite` covers parasails and parachutes, and VG names them so.
    # `para sail` is `parasail` with a space; the pair scores 83% over 24 sole
    # images and 84% over 74 boxes (#3636).
    "kite": ("kites", "para sail", "parachute", "parasail"),
    "knife": ("butter knife",),
    # The colour family carries four spellings nobody could measure alone; the two
    # that do the work are `bathroom sink` (100% over 27 sole images) and
    # `kitchen sink` (88% over 16).
    "sink": (
        "bathroom sink",
        "black sink",
        "blue sink",
        "kitchen sink",
        "pink sink",
        "red sink",
        "silver sink",
    ),
    # `ladle` folds on the same reading the review wrote into the rule: a ladle is
    # a spoon with a deep bowl (82% over 17 sole images).
    "spoon": ("blue spoon", "ladle", "silver spoon", "spoons", "white spoon"),
    # +191 repaired on the non-COCO half; overlap coverage 25.6% -> 29.4%. The
    # subtypes the review's ruling names are here where they carry 20 boxes and in
    # SCALE_VG_AMBIGUOUS where they do not, which splits one object across two
    # tables by sample size: `fire truck` (41 boxes, 85% on class) folds, while
    # `firetruck` (16), `fire engine` (12) and `tow truck` (9) are withheld. All
    # four are trucks under SCALE_CLASS_RULES; the floor is about what a box test
    # can carry, not about what the object is (#3588).
    "truck": (
        "ambulance",
        "black truck",
        "brown truck",
        "dump truck",
        "dumptruck",
        "fire truck",
        "food truck",
        "gray truck",
        "green truck",
        "grey truck",
        "orange truck",
        "pick up",
        "pick up truck",
        "pick-up",
        "pickup",
        "pickup truck",
        "pink truck",
        "red truck",
        "silver truck",
        "white truck",
        "yellow truck",
    ),
    # Two groups, both folding (#3636). The colour family -- eight spellings, one
    # hypothesis -- pools to 97% over 35 sole images and 80% over 122 boxes, so
    # the four nobody could measure alone join the two that could. The subtype
    # family (`beach`, `patio`, `closed`, `open`) pools to 88% and 69%.
    "umbrella": (
        "beach umbrella",
        "blue umbrella",
        "closed umbrella",
        "green umbrella",
        "open umbrella",
        "orange umbrella",
        "parasol",
        "patio umbrella",
        "red umbrella",
        "white umbrella",
        "yellow umbrella",
    ),
    # `urn` at 67% over 12 sole images is the marginal fold here -- Wilson lower
    # bound 0.39 against the 1/3 cut -- and it is the one the class name misses
    # most.
    "vase": (
        "black vase",
        "brown vase",
        "dark vase",
        "flower vase",
        "green vase",
        "orange vase",
        "pink vase",
        "red vase",
        "urn",
        "white vase",
        "yellow vase",
    ),
}

#: What :func:`pilebuild.loaders.vg_scale.canonicalise` does when an alias box
#: lands on an image where the class already has one of its own -- see that
#: function's ``FOLD_MODES``.
#:
#: ``fold`` is the reading #3637 measured and kept, and the margin is not close:
#: over the VG-COCO overlap, on the 225 images where the three modes disagree,
#: COCO's exhaustive boxes say the class is **not** a single-band positive on 199
#: of them, and folding names the right answer 88% of the time against 6.7% for
#: keeping the class's own band. The structural reason is bigger than that
#: number: the COCO half is already banded off COCO's own exhaustive box set, so
#: ``anchor_to_coco`` un-bands 17% of the cleanly-banded images there on evidence
#: of exactly the same kind. A mode that protected the un-anchored half from it
#: would make the two halves of one dataset disagree about what a positive is.
#:
#: The alternatives are kept so the arm can be re-measured (``band_fold.py``),
#: not because either is a candidate default.
SCALE_FOLD_MODE = os.environ.get("VTS_SCALE_FOLD_MODE", "fold")

#: VG spellings that are evidence the class MAY be present, and cannot be its box.
#:
#: `bike` is the case that named this table. Over the 51,411-image VG-COCO
#: overlap it carries **638 of COCO's 3,683 `bicycle` boxes** against the
#: `bicycle` spelling's 775 -- so `bicycle` built from one spelling is missing
#: roughly half its positives on the non-COCO half. It cannot simply be merged:
#: on the images where it is the only evidence, COCO finds a bicycle **47%** of
#: the time, and `bike` is a measured alias of `motorcycle` too (box IoU 0.38
#: over 388 co-images).
#:
#: A box under one of these names is treated as **evidence of neither presence
#: nor absence**: it is not a positive (we cannot say the object is there) and it
#: bars the image from the shared negative pool (we cannot say it is not). That
#: is the ``excluded`` third state the construction already has -- see
#: :func:`pilebuild.loaders.vg_scale.lift_ambiguous`. It removes the contaminated
#: negatives; recovering the missing positives needs a human pass.
#:
#: **Three kinds of name land here, and they share one treatment because they
#: share one answer** -- *this image cannot serve as a negative, and this box
#: cannot serve as a positive* (#3618):
#:
#: * a spelling that may denote something else: `bike` (47% precision),
#:   `tricycle`, `silverware`.
#: * a **collective**: `books` (89% precision, 34% box agreement), `birds`,
#:   `umbrellas`, `knives`. The box is a pile; a band is a claim about one
#:   object's size.
#: * a **part or container**, whose box is not the object at all: `beak` (86%),
#:   `bookshelf` (81%), `knife block` (79%), `stop` (70% -- the lettering on the
#:   sign). Named apart in the report because they cost differently: a spelling
#:   withholds the images that spell one class oddly, while a scene word
#:   withholds a whole scene type from **every** class's pool.
#:
#: Suppression applies only where it is the sole evidence. On an image COCO
#: annotates, or one a reviewer has ruled on, the answer is already known and
#: the ambiguous spelling is ignored.
SCALE_VG_AMBIGUOUS: dict[str, tuple[str, ...]] = {
    "backpack": ("black backpack", "black bag", "bookbag", "duffle bag"),
    "bench": ("bleacher", "park bench", "picnic bench", "picnic table", "picnic tables"),
    "bicycle": ("bicyclist", "bike", "bike tire", "bikes", "tricycle"),
    # `black bird` / `white bird` pool to 100% over 6 sole images but only 15
    # boxes -- above the precision cut, below the box floor, which is the safe
    # side. `pigeons` inherits from `pigeon`, not from the species family:
    # a plural is a collective whatever its singular does (#3636).
    "bird": (
        "beak",
        "birds",
        "black bird",
        "dove",
        "ducks",
        "feather",
        "feathers",
        "geese",
        "peacock",
        "pigeons",
        "seagulls",
        "white bird",
    ),
    "boat": ("barge", "bouy", "sail boat", "sailboat"),
    "book": (
        "binder",
        "black book",
        "book case",
        "book shelf",
        "bookcase",
        "books",
        "bookshelf",
        "dvd",
        "dvds",
        "games",
        "library",
        "magazines",
        "notebook",
        "white book",
    ),
    "bottle": (
        "beer bottles",
        "beverages",
        "glass bottle",
        "glass bottles",
        "green bottle",
        "greenbottle",
        "hand soap",
        "jar",
        "ketchup bottle",
        "ketchup bottle.",
        "plastic bottles",
        "shaker",
        "shakers",
        "soap bottle",
        "soda bottle",
        "water bottles",
    ),
    "bowl": ("blue bowl", "blue bowls", "casserole dish", "dishes", "dog bowl", "mixing bowl"),
    # `busses` is VG's other plural of `bus`. `buses` folds on its own measured
    # box agreement; `busses` has none of its own, and a plural with no
    # measurement behind it is a collective until shown otherwise (#3636).
    "bus": ("blue bus", "busses"),
    # `van` and `vehicle` are demoted aliases, not refuted names -- see the note
    # in SCALE_VG_NAMES["car"]. Both are evidence the class MAY be present, so
    # they still bar their image from the shared negative pool.
    #
    # `truck`, `pick up`, `pick up truck` and `pickup truck` are NOT here, and
    # leaving them here would have been the most expensive line in this file.
    # The audit adjudicates one class at a time, so it offered every one of them
    # as ambiguous evidence for `car` -- reasonable in isolation, and fatal in
    # combination, because `lift_ambiguous` POPS the box rather than merely
    # declining to band it. Listing `truck` as ambiguous for `car` therefore
    # deletes every VG `truck` box in the corpus: measured, `truck` supply went
    # to **0 in all three bands** from 3,386. A name another class in C already
    # owns is settled by that class; it is never a neighbour's ambiguous entry.
    "car": (
        "automobil",
        "automobiles",
        "blue cars",
        "city street",
        "golf cart",
        "green car",
        "intersection",
        "lot",
        "mini van",
        "parked car",
        "parked cars",
        "parkedcar",
        "parkedcars",
        "parking lot",
        "pick-up truck",
        "police car",
        "police cars",
        "sedans",
        "taxi cab",
        "taxicab",
        "taxis",
        "traffic",
        "van",
        "vehicle",
        "vehicle?",
        "vehicles",
        "white van",
    ),
    "cell phone": (
        "black phone",
        "blue phone",
        "cell",
        "cell phones",
        "cellphon",
        "cellphones",
        "flip phone",
        "flipphone",
        "grey phone",
        "iphone",
        "ipod",
        "mobile",
        "mobile phone",
        "mobile phones",
        "mp3 player",
        "phones",
        "smart phone",
        "smartphone",
        "white phone",
    ),
    # `chair back` is a part: its box is the back, and a band is a claim about one
    # object's size.
    "chair": (
        "armchairs",
        "bar stools",
        "beach chairs",
        "black chair",
        "blackchair",
        "blue chair",
        "blue chairs",
        "bluechair",
        "brown chairs",
        "chair back",
        "classroom",
        "computer chair",
        "desk chair",
        "folding chairs",
        "high chair",
        "highchair",
        "lawn chairs",
        "office chair",
        "rocking chair",
        "seats",
        "stool",
        "white chairs",
        "wicker chair",
    ),
    # `numerals` and `clock faces` each inherit from their own singular (#3636).
    "clock": ("alarm clock", "clock faces", "numeral", "numerals", "roman numerals"),
    # `glass` is the entry that named this table, and it is the one entry the
    # audit actively DISAGREES with.
    #
    # It is 62.2% good by fold-out (1,146 of 3,224 VG boxes on COCO `cup`, 861
    # on `wine glass`), well clear of `bike`'s 40.1%, which is why it was first
    # tried as a plain alias. A fold-out rate is not a positive-precision rate:
    # fold-out is measured only on the COCO-annotated half, while POSITIVES are
    # drawn from all of VG, and there the ~35% of `glass` boxes that land on no
    # COCO class -- windowpanes, eyeglasses -- arrive as positives unexamined.
    # The review measured the damage: the merged `cup` slate rejected **9 of 30**
    # boxed positives (30%) against 0-17% for every other class of the thirteen,
    # and worst in the LARGE band at 40%, which is where a windowpane lands.
    #
    # **#3700 raised the case for folding it, and changed nothing that matters.**
    # Once the scorer counted COCO `wine glass` too, `glass`, `glass.` and
    # `clear glass` all cleared the cuts and `name_evidence.py` began proposing
    # them as aliases. That is the same evidence measured better on the half
    # where `glass` was never the problem -- repair precision is computed on the
    # overlap, and the windowpanes are on the half with no reference at all. The
    # measurement that decides is a human one and no script produces it, so the
    # exclusion is a hand exclusion with a test behind it, exactly like `sign`
    # for `stop sign` (`test_glass_is_withheld_from_cup_rather_than_folded`).
    "cup": (
        "beer",
        "beer glass",
        "beer glasses",
        "beverage",
        "blue cup",
        "coaster",
        "coffee",
        "coffee cups",
        "cups",
        "dishes",
        "drink",
        "drinking glass",
        "drinking glasses",
        "glass",
        "glass of water",
        "glass.",
        "juice",
        "liquid",
        "measuring cup",
        "measuring cups",
        "mugs",
        "mugs.",
        "napkin",
        "napkin holder",
        "napkins",
        "orange juice",
        "paper cups",
        "pepper shaker",
        "plastic cups",
        "salt",
        "salt shaker",
        "saucer",
        "shaker",
        "straw",
        "tea",
        "tea cup",
        "teapot",
        "tumbler",
        "tumblers",
        "water glass",
        "water glasses",
    ),
    "dog": ("bulldog", "poodle"),
    "fire hydrant": ("fire hydrant.", "fire-hydrant", "firehydrant", "hydrants"),
    "fork": ("silver fork", "silver forks", "silver ware", "silverware", "white fork"),
    "knife": ("butterknife", "knife block", "knives", "silverware"),
    "sink": (
        "basin",
        "basins",
        "dishwasher",
        "double sink",
        "double sinks",
        "faucet",
        "white sink",
        "white sinks",
    ),
    "spoon": (
        "cooking utensils",
        "kitchen utensil",
        "kitchen utensils",
        "ladles",
        "plastic spoon",
        "plastic spoons",
        "silver ware",
        "silverware",
        "utensil",
        "utensils",
    ),
    # `sign` is the largest fold-in column anywhere in C -- 473 of COCO's 1,016
    # `stop sign` boxes, 46.6% -- and it is NOT here, because a VG `sign` box is
    # a stop sign 7.9% of the time: listing it would withhold 12.7 images from
    # the pool per contaminated negative removed. This class's missing positives
    # need a human pass, not a name (#3618).
    "stop sign": ("octagon", "stop"),
    # The truck subtypes that fall below the 20-box floor, plus the three-way
    # names. `van` is withheld here and from `car` too, which is the honest
    # reading of a name COCO splits 261 / 318 / 37.
    "truck": (
        "blue truck",
        "box truck",
        "camper",
        "campers",
        "delivery truck",
        "fire engine",
        "fire trucks",
        "firetruck",
        "firetrucks",
        "flatbed",
        "food trucks",
        "lorry",
        "luggage cart",
        "pick-up truck",
        "rv",
        "rvs",
        "semi truck",
        "semi-truck",
        "tow truck",
        "tractor",
        "trailer",
        "trucks",
        "van",
        "vehicles",
        "white van",
    ),
    "umbrella": ("an umbrella", "black umbrella", "pink umbrella", "umbrellas"),
    "vase": ("artifact", "blue vase", "glass vase", "glass vases", "urns", "vas", "vases"),
}


#: How a name is written, once, so ``vg_name_families.py`` and the grouping
#: below cannot drift apart on what a name's head noun is.
#:
#: Trailing characters VG annotators leave on a name (`umbrella.`, `"clock"`).
NAME_PUNCT = ".,;:!?'\"()[]"


def name_head(name: str) -> str:
    """The final token of *name*, stripped of punctuation and a possessive.

    `umbrella's` and `umbrella.` are the same word as `umbrella` with an
    annotator's typing on the end, and there is no sense in which they denote
    something else.
    """
    tokens = name.replace("-", " ").split()
    if not tokens:
        return ""
    tok = tokens[-1].strip(NAME_PUNCT)
    return tok[:-2] if tok.endswith("'s") else tok


def name_singulars(token: str) -> set[str]:
    """Candidate singular forms of *token*, over-generating on purpose.

    Over-generation is safe here because the result is only ever used to *test*
    membership against one known class name -- `buses` proposing both `bus` and
    `buse` costs nothing, and missing `bus` would cost a whole spelling.
    """
    out = {token}
    if token.endswith("ies"):
        out.add(token[:-3] + "y")
    if token.endswith("ves"):
        out.update({token[:-3] + "fe", token[:-3] + "f"})
    if token.endswith("ses"):  # busses -> bus
        out.add(token[:-3])
    if token.endswith("es"):
        out.add(token[:-2])
    if token.endswith("s"):
        out.add(token[:-1])
    return out


def name_skeleton(name: str) -> str:
    """*name* with whitespace, hyphens and annotator punctuation removed.

    Two names with one skeleton are one word typed two ways -- `back pack` /
    `backpack`, `clock face` / `clockface`, `row boat` / `rowboat`. This is what
    the ``spelling`` construction groups on, and it is an equivalence relation
    rather than a lexicon, so it needs no vocabulary to maintain.
    """
    return "".join(ch for ch in name.lower() if ch.isalnum())


class Construction(NamedTuple):
    """A productive way of writing a name that does not change what it denotes.

    ``modifiers`` is the vocabulary that may stand in front of the class's head
    noun. It is a **declared list, not a regex**: which words leave a denotation
    alone is exactly the judgement #3636 is about, so it is written down here
    beside the tables it fills rather than inferred at run time.

    ``foldable`` is whether a member may reach :data:`SCALE_VG_NAMES` at all. A
    ``count`` compound never can, however well it scores: `two birds` names a
    *set* of that many, and a band is a claim about one object's size -- the
    same reason `books` and `umbrellas` sit in the ambiguous table (#3618).
    """

    key: str
    modifiers: frozenset[str]
    foldable: bool
    why: str


#: The constructions ``name_evidence.py --pooled`` adjudicates as one hypothesis.
#:
#: #3618 scored every candidate name alone, against a floor of five images where
#: the name is the class's only evidence, and **76 of 626 fell below it** --
#: recorded `unmeasured`, neither acted on nor refuted. They are not noise: they
#: carry 312 non-COCO images between them.
#:
#: Most are not independent hypotheses. `blue umbrella`, `red umbrella`,
#: `green umbrella`, `orange umbrella` and `yellow umbrella` are one hypothesis
#: five times over -- *a colour word in front of the class name does not change
#: what the name denotes* -- and that hypothesis is testable at the sample size
#: of the whole family rather than one colour at a time.
#:
#: A name joins a construction for class *c* when its head noun is *c*'s head
#: noun (a plural allowed) and every remaining token is in the vocabulary. So
#: `black clock` is a `clock` colour compound and `black face` is not, which is
#: the distinction that keeps the group measuring one thing.
SCALE_VG_CONSTRUCTIONS: tuple[Construction, ...] = (
    Construction(
        key="colour",
        modifiers=frozenset(
            {
                "beige",
                "black",
                "blue",
                "brown",
                "colorful",
                "colourful",
                "dark",
                "gold",
                "golden",
                "gray",
                "green",
                "grey",
                "multicolored",
                "orange",
                "pink",
                "purple",
                "red",
                "silver",
                "tan",
                "white",
                "yellow",
            }
        ),
        foldable=True,
        why="a colour word does not change what the name denotes",
    ),
    Construction(
        key="size",
        modifiers=frozenset({"big", "giant", "huge", "large", "little", "long", "small", "tall", "tiny"}),
        foldable=True,
        why="a size word does not change what the name denotes; the band is read off the box, not the word",
    ),
    Construction(
        key="typing",
        modifiers=frozenset({"a", "an", "the"}),
        foldable=True,
        why="a determiner is the annotator typing, not a distinction",
    ),
    Construction(
        key="count",
        modifiers=frozenset(
            {"two", "three", "four", "five", "six", "several", "many", "some", "multiple", "group", "bunch", "pair"}
        ),
        foldable=False,
        why="a numeral names a SET of that many: the box is a pile, so a member can be evidence but never a band",
    ),
    #: Not a lexicon -- membership is the equivalence class under
    #: :func:`name_skeleton`, so `back pack`/`backpack` and `clock face`/`clockface`
    #: group with no vocabulary to keep current.
    Construction(
        key="spelling",
        modifiers=frozenset(),
        foldable=True,
        why="one word typed two ways: same letters, different whitespace",
    ),
    #: Also not a lexicon: one group per singular form, so `pigeon`/`pigeons`
    #: is a hypothesis and `pigeons`/`ducks` is not. That is the pairwise
    #: question worth asking -- *does this plural denote what its own singular
    #: denotes* -- and for `pigeons` it is far better evidence than the species
    #: group, which knows only that a pigeon is a bird.
    #:
    #: **Never foldable**, and that is the shipped rule rather than caution:
    #: `books`, `birds`, `umbrellas`, `knives`, `ducks`, `geese` and `seagulls`
    #: are all in :data:`SCALE_VG_AMBIGUOUS` because the box is a pile. The
    #: plurals #3618 *did* fold (`boats`, `clocks`, `dogs`, `kites`) each earned
    #: it on their own measured box agreement, which an individual measurement
    #: still delivers -- inheritance is for names that have none, and a plural
    #: with no measurement of its own is a collective until shown otherwise.
    #: `buses` keeps the alias it earned; `busses` inherits withheld.
    Construction(
        key="plural",
        modifiers=frozenset(),
        foldable=False,
        why="a plural names a SET: the box is a pile, so a member can be evidence but never a band",
    ),
)


class NameGroup(NamedTuple):
    """A set of names a human asserts denote the same kind of thing.

    Where a :class:`Construction` is productive -- a vocabulary that applies to
    every class -- a group is a judgement about *this* class's vocabulary, and
    the judgement is the ``criterion``. That string is load-bearing, not a
    comment: it is what makes membership auditable, and it is the only defence
    against fitting the group to the answer. **Every candidate name meeting the
    criterion is listed, including the ones known to score badly** -- `crane`
    (the machine) is in `bird`/`species` and `jet ski` is in `boat`/`vessel`,
    because a group whose losers were quietly left out is not a measurement.

    ``foldable=False`` marks a group whose members may be evidence the class is
    present but can never carry a band, exactly as for a construction.
    """

    key: str
    criterion: str
    names: tuple[str, ...]
    foldable: bool = True


#: Hand-declared groups, per class, each with the criterion that defines it.
#:
#: These reach what a construction cannot. #3618's residue is mostly *hyponyms*
#: -- `yacht`, `ferry`, `flamingo`, `grandfather clock` -- and no modifier
#: vocabulary groups those, because the whole name is different. What groups
#: them is a person saying "these all name a kind of watercraft", which is a
#: hypothesis with a sample size like any other.
#:
#: This is **not** the head-noun fold #3618 refuted. That was mechanical: every
#: name sharing a head noun, which puts `hot dog` (405 images, 0 of 181) in with
#: `puppy`. Three things separate a group from it: the criterion is stated and
#: excludes `hot dog` on meaning rather than on its score; the group is
#: adjudicated before anything is inherited; and a member measurable on its own
#: keeps its own verdict either way (``name_evidence.py``).
#: **Members with a common non-class sense are excluded** (#3662, criterion
#: pre-registered on the issue 2026-09-07 before any re-run). A name earns its
#: place in a declared group by what it *denotes*, and `vessel` in VG is a
#: container or a blood vessel, `crane` is the machine, `lab` is a laboratory
#: and `tip` is a gratuity. #3636 would not declare a group afresh after seeing
#: which member sank it -- that is how a study fits its own answer -- so the
#: rule was written down first, applied to EVERY group including the ones that
#: passed, and each exclusion is recorded on the issue with the other sense it
#: carries.
SCALE_VG_GROUPS: dict[str, tuple[NameGroup, ...]] = {
    "bicycle": (
        NameGroup(
            key="part",
            criterion="a VG name for a part of a bicycle",
            names=(
                "bicycle tire",
                "bike tire",
                "front wheel",
            ),
        ),
    ),
    "bird": (
        NameGroup(
            key="species",
            criterion="a VG name denoting a species or kind of bird",
            names=(
                "chickens",
                "duck",
                "ducks",
                "eagle",
                "flamingo",
                "geese",
                "goose",
                "hen",
                "ostrich",
                "owl",
                "parrot",
                "peacock",
                "pelican",
                "penguin",
                "pigeon",
                "pigeons",
                "rooster",
                "seagull",
                "seagulls",
                "swan",
            ),
        ),
    ),
    "boat": (
        NameGroup(
            key="vessel",
            criterion="a VG name denoting a kind of watercraft",
            names=(
                "barge",
                "boats",
                "canoe",
                "cruise ship",
                "ferry",
                "jet ski",
                "kayak",
                "motorboat",
                "raft",
                "row boat",
                "rowboat",
                "sail boat",
                "sailboat",
                "sailboats",
                "yacht",
            ),
        ),
        NameGroup(
            key="mooring",
            criterion="a VG name for a place where vessels are moored (not open water or a shoreline)",
            names=("marina",),
        ),
        NameGroup(
            key="part",
            criterion="a VG name for a part of a vessel",
            names=(
                "hull",
                "mast",
                "oar",
                "sail",
                "sails",
            ),
        ),
    ),
    "book": (
        NameGroup(
            key="part",
            criterion="a VG name for a part of a book",
            names=(
                "binding",
                "book cover",
                "pages",
            ),
        ),
    ),
    "bus": (
        NameGroup(
            key="subtype",
            criterion="a VG name for a kind of bus, by its route or its deck",
            names=("city bus", "double decker", "double-decker bus", "passenger bus", "school bus", "tour bus"),
        ),
        NameGroup(
            key="part",
            criterion="a VG name for a part of a bus specifically (not a part any vehicle has)",
            names=("bus front",),
        ),
    ),
    "clock": (
        NameGroup(
            key="subtype",
            criterion="a VG name for a kind of clock, by its mechanism or its mounting",
            names=("alarm clock", "digital clock", "grandfather clock"),
        ),
        NameGroup(
            key="dial",
            criterion="a VG name for a clock's dial taken as a whole (not a marking on it)",
            names=(
                "clock face",
                "clock faces",
                "clockface",
                "dials",
            ),
        ),
        NameGroup(
            key="marking",
            criterion="a VG name for a marking on a clock face",
            names=("roman numerals",),
        ),
        NameGroup(
            key="part",
            criterion="a VG name for a part of a clock other than its dial or its markings",
            names=("clock frame",),
        ),
    ),
    "dog": (
        NameGroup(
            key="breed",
            criterion="a VG name for a dog breed or life stage",
            names=(
                "bulldog",
                "dalmation",
                "poodle",
                "puppy",
            ),
        ),
        NameGroup(
            key="part",
            criterion="a VG name for a part of a dog",
            names=("dog's head",),
        ),
    ),
    "kite": (
        NameGroup(
            key="part",
            criterion="a VG name for a part of a kite",
            names=("kite tail",),
        ),
    ),
    "knife": (
        NameGroup(
            key="subtype",
            criterion="a VG name for a kind of knife, by what it cuts",
            names=(
                "butter knife",
                "butterknife",
                "cake server",
            ),
        ),
        NameGroup(
            key="part",
            criterion="a VG name for a part of a knife",
            names=("knife blade",),
        ),
    ),
    "stop sign": (
        NameGroup(
            key="sign-type",
            criterion="a VG name for a kind of road or street sign, by what it says",
            names=(
                "arrow sign",
                "construction sign",
                "direction sign",
                "electric sign",
                "handicapped sign",
                "no parking sign",
                "one way sign",
                "street sign",
            ),
        ),
    ),
    "umbrella": (
        NameGroup(
            key="subtype",
            criterion="a VG name for a kind of umbrella, by its use or its state",
            names=("beach umbrella", "closed umbrella", "open umbrella", "parasol", "patio umbrella"),
        ),
    ),
}


def scale_vg_groups_for(cls: str, candidates: list[str]) -> dict[str, list[str]]:
    """Which pooled groups *cls*'s *candidates* fall into, keyed by group.

    Constructions are matched here rather than in the caller so that the
    vocabularies above are the only place the rule is written. A name may belong
    to at most one construction (the first that accepts it) and to any declared
    group, so `parasol` is both `umbrella`/`subtype` and -- were it spelled two
    ways -- a ``spelling`` member.
    """
    head_wanted = name_head(cls)
    skeletons: dict[str, list[str]] = {}
    for n in candidates:
        skeletons.setdefault(name_skeleton(n), []).append(n)

    out: dict[str, list[str]] = {}
    for con in SCALE_VG_CONSTRUCTIONS:
        if con.key == "spelling":
            continue
        members = []
        for n in candidates:
            tokens = n.replace("-", " ").split()
            if len(tokens) < 2 or head_wanted not in name_singulars(name_head(n)):
                continue
            if all(t.strip(NAME_PUNCT) in con.modifiers for t in tokens[:-1]):
                members.append(n)
        if members:
            out[con.key] = sorted(members)

    # `spelling` is ONE GROUP PER SKELETON, not one per class. The hypothesis is
    # "`clockface` denotes what `clock face` denotes", and it is pairwise: pooling
    # every respelt name in a class would ask instead whether bookcase-ish images
    # are book images, which is a different question with a different answer. A
    # skeleton shared with the class name alone is dropped -- there is no second
    # rate to pool with, since the class name is never its own sole evidence.
    for skel, ns in sorted(skeletons.items()):
        if len(ns) > 1:
            out[f"spelling:{skel}"] = sorted(ns)

    # `plural`: group by singular form, the same shape as `spelling`. A name's
    # key is the shortest of its own singular forms that is itself a candidate
    # (or the class name), so `ducks` keys on `duck`, `buses` and `busses` both
    # key on `bus`, and `clock faces` on `clock face`.
    known = set(candidates) | {cls}
    by_singular: dict[str, list[str]] = {}
    for n in candidates:
        forms = sorted(name_singulars(n) & known, key=len)
        by_singular.setdefault(forms[0] if forms else n, []).append(n)
    for key, ns in sorted(by_singular.items()):
        if len(ns) > 1:
            out[f"plural:{key}"] = sorted(ns)

    for grp in SCALE_VG_GROUPS.get(cls, ()):
        members = sorted(set(grp.names) & set(candidates))
        if members:
            out[grp.key] = members
    return out


def scale_vg_group_foldable(cls: str, key: str) -> bool:
    """Whether a member of group *key* may reach :data:`SCALE_VG_NAMES`."""
    base = key.split(":", 1)[0]
    for con in SCALE_VG_CONSTRUCTIONS:
        if con.key == base:
            return con.foldable
    for grp in SCALE_VG_GROUPS.get(cls, ()):
        if grp.key == key:
            return grp.foldable
    return True


def scale_vg_group_why(cls: str, key: str) -> str:
    """The declared reason group *key* is one hypothesis -- printed with it."""
    base = key.split(":", 1)[0]
    for con in SCALE_VG_CONSTRUCTIONS:
        if con.key == base:
            return con.why
    for grp in SCALE_VG_GROUPS.get(cls, ()):
        if grp.key == key:
            return grp.criterion
    return ""


#: Classes in *C* whose VG-name coverage has actually been measured.
#:
#: Written down because "no alternate spelling is listed" and "no alternate
#: spelling exists" are the same empty table, and the first is what shipped:
#: `bicycle` was built from one spelling for the whole of #3156 with every
#: structural check passing. A class is added here once its names have been
#: adjudicated against COCO -- ``coco_folds.py`` for the fold-in column,
#: ``vg_name_families.py`` for the spellings COCO's half barely sees, and
#: ``name_evidence.py`` for the verdict.
#:
#: All twenty-five: the first twelve at #3618, the #3588 thirteen at their
#: promotion. Listed one by one rather than derived from :data:`SCALE_CLASSES`,
#: because deriving it would mark a *newly added* class audited without anyone
#: having looked -- which is the exact failure this flag exists to make visible.
#: :func:`pilebuild.loaders.vg_scale.load` names the unaudited classes on every
#: build, since the rebuild is when this stops being cheap to fix (#3605).
#:
#: **A cleared human review is not this flag.** #3588 reviewed its thirteen at
#: 300 images each and cleared all thirteen before a single one of their names
#: had been adjudicated -- the review asks whether a human can label the class
#: consistently, and this asks which VG spellings the class is BUILT from, and
#: nothing about the first answers the second. The audit run at the promotion is
#: what earned the thirteen their place here, and it was worth running: it
#: repaired **+1,280** images for `cell phone` against the 535 it could already
#: see, and took `fire hydrant` from 44.7% to **73.8%** of COCO's boxes.
SCALE_VG_NAMES_AUDITED: frozenset[str] = frozenset(
    {
        "backpack",
        "bench",
        "bicycle",
        "bird",
        "boat",
        "book",
        "bottle",
        "bowl",
        "bus",
        "car",
        "cell phone",
        "chair",
        "clock",
        "cup",
        "dog",
        "fire hydrant",
        "fork",
        "kite",
        "knife",
        "sink",
        "spoon",
        "stop sign",
        "truck",
        "umbrella",
        "vase",
    }
)


class ClassRule(NamedTuple):
    """One class's review definition: the ``name`` a reviewer sees, and the ``test``."""

    #: What the slate's dataset/detector is called. This is the ONLY thing a
    #: reviewer sees while voting -- files are named by image id alone -- so it
    #: has to carry the discrimination on its own, in a few words.
    name: str
    #: The full wording the name abbreviates: what counts as Good, what counts
    #: as Bad, and the near-miss the short name does not settle. Read by whoever
    #: builds the slate and whoever adjudicates it, so both apply one definition.
    #:
    #: Empty means *not written down yet*, not *no boundary case*: the rule was
    #: measured as a name before :class:`ClassRule` existed and its wording has
    #: never been recorded. Fill one in when a slate of that class is issued --
    #: an unwritten test is the state #3612 exists to end.
    test: str = ""
    #: What ONE object of the class is, for a question about the BOX rather than
    #: the membership (#3985's "is the red box around ONE object?"). ``test``
    #: says which objects belong; this says how many of them one box may hold.
    #: They differ: a bunch of bananas is all `banana` and still not one banana.
    #: Written into the queue's name, because "one object" alone left a pair of
    #: skis undecidable. Empty means the class name's own count noun is the unit.
    #: The count is of objects, not pieces: one apple cut into slices is ONE
    #: apple, so Good. Slices plainly from several fruit are Bad; slices whose
    #: source can't be told are cannot-tell, so Good.
    #: Owner rulings, 2026-09-22.
    unit: str = ""


#: Per-class review definitions, for the classes whose plain English name is not
#: the whole question.
#:
#: A class whose meaning differs between the halves of a dataset is not noisy, it
#: is two classes wearing one name (``make_definition_reslate.py``). The rule
#: that separates them travels in the **dataset name**, because a reviewer
#: cannot see a manifest while voting -- and until this table existed the rule
#: was typed by hand at slate time and written down nowhere, so the wording a
#: re-review used was whatever the next person remembered.
#:
#: Two things are therefore recorded, not one. The ``name`` is what the reviewer
#: reads; the ``test`` is what the name abbreviates, and is what settles the
#: near-misses a two-word name cannot. A rule whose ``test`` lives only in a
#: session transcript is a rule that will be re-derived differently.
#:
#: Classes absent from this table are their own definition, and
#: :func:`review_name` falls back to the bare class name for them. A class need
#: not be in :data:`SCALE_CLASSES` to appear here: `cell phone` is a #3588
#: candidate whose first slate is already voted.
#: What each class in *C* is actually MADE OF, measured rather than asserted.
#:
#: :data:`SCALE_CLASS_RULES` says what a reviewer should count as Good. This says
#: what COCO's annotators *did* count -- which is a different fact, and under a
#: pure-COCO build it is the one that governs, because nobody reviews COCO's
#: labels. A result on `cup` is a result on whatever COCO put in `cup`, and that
#: turns out to be three things.
#:
#: Measured by matching COCO boxes to LVIS boxes on the same image (mutual best
#: match, IoU >= 0.7) and reading off what LVIS's 1,203 *defined* synsets call
#: them -- ``coco_class_purity.py``, study
#: ``docs/experiments/2026-09-18-coco-only-supply-3983/``. COCO's own vocabulary
#: cannot answer this: it is curated and disjoint, so an annotator records one
#: label and no disagreement, and two COCO classes almost never box the same
#: pixels (3.4% collision, with no separation between the classes whose reviews
#: split and the rest). A finer vocabulary over the same pixels is the only way
#: to see inside a class without a review pass.
#:
#: **Read these as composition, not as error.** Three different things appear:
#:
#: * **Subtype spread** -- `bottle` is wine/water/beer bottles, `bird` is
#:   duck/gull/pigeon. Harmless; the class is what it says, at finer grain.
#: * **A genuine boundary contest** -- 17% of `truck` is what LVIS calls a car,
#:   against 2% the other way, and both are in *C*. That asymmetry is real and is
#:   the one open decision the measurement surfaced.
#: * **A minority a reader would not expect** -- `stop sign` is 20% generic
#:   `street_sign`, `book` 6% magazines, `cell phone` 4% landlines. Small, and
#:   exactly the cases that split #3612's review: a 4% minority splits two
#:   reviewers as surely as a 40% one when neither has a rule to consult.
#:
#: So **share does not rank difficulty** and no threshold should be read off this
#: table. Every class whose review actually split is homogeneous -- `book` 85%,
#: `cell phone` 89%, `knife` 93%, `bench` 93%. The table's job is to be quotable
#: beside a published number, and to be the source text when a rule is written.
#:
#: Percentages are of matched boxes and members under 1% are dropped, so a row
#: need not sum to 100. Regenerate with ``coco_class_purity.py --out``.
SCALE_CLASS_CONTENTS: dict[str, str] = {
    "single serving drinking vessel": "glass_(drink_container) 34%, wineglass 24%, cup 20%, mug 12%, bowl 1%, Dixie_cup 1%, candle 1%, teacup 1%, vase 1%, pitcher_(vessel_for_liquid) 1%, bucket 1% -- the union of COCO `cup` and `wine glass` (#4056), n=2196",
    "enclosed road vehicle": "car_(automobile) 63%, truck 12%, minivan 9%, pickup_truck 6%, cab_(taxi) 2%, trailer_truck 2%, fire_engine 2%, bus_(vehicle) 1%, police_cruiser 1% -- the union of COCO `car` and `truck` (#4056), n=1935",
    "dining table": "tablecloth 34%, table 31%, dining_table 10%, place_mat 5%, plate 5%, coffee_table 3%, tray 3%, desk 3%, chopping_board 1%, pizza 1%, kitchen_table 1%, cabinet 1%, bench 1%",
    "tv": "television_set 50%, monitor_(computer_equipment) computer_monitor 47%, signboard 1%, fireplace 1%",
    "remote": "remote_control 56%, control 41%, cellular_telephone 1%, telephone 1%",
    "handbag": "handbag 63%, suitcase 8%, backpack 7%, shoulder_bag 5%, tote_bag 4%, plastic_bag 4%, shopping_bag 3%, strap 2%, duffel_bag 1%, briefcase 1%, basket 1%",
    "person": "person 64%, wet_suit 8%, jacket 5%, dress 3%, coat 2%, shirt 2%, sweater 1%, jersey 1%, suit_(clothing) 1%, jean 1%, statue_(sculpture) 1%, trousers 1%, sweatshirt 1%, polo_shirt 1%, pajamas 1%",
    "potted plant": "flower_arrangement 75%, flowerpot 21%, Christmas_tree 2%, vase 1%, jar 1%",
    "orange": "orange_(fruit) 77%, mandarin_orange 7%, lemon 6%, peach 3%, carrot 2%, lime 1%, apple 1%, egg 1%",
    "apple": "apple 80%, pear 5%, peach 4%, lime 2%, radish 2%, orange_(fruit) 1%, tomato 1%, basket 1%, crate 1%, lemon 1%",
    "tie": "necktie 89%, bow-tie 7%, bow_(decorative_ribbons) 2%, scarf 1%",
    "motorcycle": "motorcycle 89%, motor_scooter 8%, dirt_bike 1%, bicycle 1%, tarp 1%",
    "airplane": "airplane 91%, fighter_jet 7%, jet_plane 2%",
    "suitcase": "suitcase 93%, trunk 2%, duffel_bag 2%, backpack 1%, box 1%, briefcase 1%",
    "snowboard": "snowboard 94%, ski 5%",
    "microwave": "microwave_oven 96%, toaster_oven 2%, coffee_maker 1%, stove 1%",
    "laptop": "laptop_computer 97%, monitor_(computer_equipment) computer_monitor 2%",
    "skis": "ski 97%, snowboard 2%",
    "traffic light": "traffic_light 98%, street_sign 1%, streetlight 1%",
    "scissors": "scissors 98%, shears 1%",
    "tennis racket": "tennis_racket 98%, racket 1%",
    "keyboard": "computer_keyboard 99%, laptop_computer 1%",
    "baseball bat": "baseball_bat 99%, toy 1%",
    "toothbrush": "toothbrush 99%, handle 1%, screwdriver 1%",
    "surfboard": "surfboard 99%",
    "frisbee": "frisbee 99%, toy 1%",
    "mouse": "mouse_(computer_equipment) 99%, control 1%",
    "skateboard": "skateboard 99%",
    "banana": "banana 99%",
    "parking meter": "parking_meter 100%",
    "fire hydrant": "fireplug 100% -- the only class in C with no second member at all",
    "fork": "fork 97%",
    "bicycle": "bicycle 96%, wheel 2%",
    "umbrella": "umbrella 94%, awning 4% -- an awning is not an umbrella; small but not zero",
    "dog": "dog 94%, puppy 2%, pet 1%",
    "boat": "boat 93%, canoe 2%, raft 1%",
    "knife": "knife 93%, handle 3%, spatula 2%",
    "bench": "bench 93%, table 2%, desk 1%, chair 1%, pew 1%",
    "kite": "kite 92%, parasail 4%, flag 2%",
    "clock": "clock 90%, wall_clock 3%, watch 2%, alarm_clock 2%, clock_tower 2% -- a"
    " wristwatch and a clock tower are both `clock`",
    "cell phone": "cellular_telephone 89%, telephone 4%, camera 3%, iPod 1% -- the 4% landlines"
    " are what split the first slate (#3612), at 4%",
    "bus": "bus_(vehicle) 88%, school_bus 5%, car_(automobile) 4%",
    "vase": "vase 87%, flowerpot 6%, pitcher 1%, pottery 1% -- planters are in, though #3784"
    " retired 21 of them under the reviewer rule",
    "book": "book 85%, magazine 6%, notebook 2%, binder 2% -- the 6% magazines are the whole of #3612's 21-vs-49 split",
    "backpack": "backpack 82%, suitcase 8%, duffel_bag 3%, handbag 2%",
    "stop sign": "stop_sign 79%, street_sign 20%, signboard 1% -- a fifth of this class is a"
    " sign that is not a stop sign, the largest unexpected minority in C",
    "spoon": "spoon 78%, ladle 6%, fork 5%, wooden_spoon 4%, soupspoon 3%, spatula 2%, knife 2%",
    "sink": "sink 76%, kitchen_sink 20%, bathtub 2% -- a subtype split, not a boundary",
    "bird": "bird 76%, duck 7%, gull 5%, pigeon 3%, goose 2%, pelican 1% -- subtype spread",
    "bowl": "bowl 72%, plate 5%, basket 3%, pot 3%, dish 3%, cup 2%, bucket 2%, pan 1%",
    "chair": "chair 66%, armchair 11%, deck_chair 8%, stool 5%, bench 2%, folding_chair 1%,"
    " sofa 1%, rocking_chair 1% -- a stool has no back, and is still `chair` here",
    "bottle": "bottle 42%, wine_bottle 18%, water_bottle 9%, beer_bottle 7%, soap 4%,"
    " condiment 2%, jar 2%, alcohol 2%, soda 2%, shampoo 2% -- mostly subtype spread,"
    " but the soap/shampoo/jar tail is ~8% of non-drink containers",
}

SCALE_CLASS_RULES: dict[str, ClassRule] = {
    "single serving drinking vessel": ClassRule(
        name="single serving drinking vessel any stem",
        test=(
            "Good: anything a person drinks a single serving from -- mugs, teacups, "
            "paper and plastic cups, tumblers and highballs, wine glasses, champagne "
            "flutes, martini glasses, goblets, beer glasses. STEM OR NO STEM IS "
            "IRRELEVANT, which is the point of the class. Bad: BOWLS and BOTTLES, their "
            "own classes in C; jugs, pitchers, carafes and teapots, which serve more than "
            "one; vases; and a candle that happens to sit in a cup. "
            "THE CUP/WINE-GLASS DISTINCTION IS DELIBERATELY GONE (#4056, owner ruling "
            "2026-09-20, resolving #4074). COCO's stem branch works where a stem is "
            "visible -- `mug` and `teacup` are `cup` 100% of the time, `wineglass` is "
            "`wine glass` 97% -- but the STEMLESS GLASS defeats it: "
            "`glass_(drink_container)`, 817 boxes and the pair's largest type, is called "
            "`cup` 75% and `wine glass` 25%. 10.7% of the pair's boxes carry the minority "
            "COCO label for their own object type, which is the rate REGION VOTING reads "
            "and worse than the vehicles' 9.6%."
        ),
    ),
    "enclosed road vehicle": ClassRule(
        name="enclosed road vehicle not bus or bike",
        test=(
            "Good: cars, taxis, minivans, SUVs, pickups, vans, box trucks, semis and "
            "tractor units, flatbeds, tow, fire and garbage trucks -- anything that is a "
            "SELF-PROPELLED, ENCLOSED road vehicle carrying people or goods. Bad: BUSES "
            "and MOTORCYCLES, their own classes in C; bicycles; a detached trailer, which "
            "is not self-propelled; and plant machinery that WORKS AT A SITE rather than "
            "carrying down a road -- cranes on tracks, tractors, forklifts, bulldozers. "
            "THE CAR/TRUCK DISTINCTION IS DELIBERATELY GONE (#4056, owner ruling "
            "2026-09-20). COCO draws it well at the IMAGE level and badly at the BOX "
            "level, and region voting reads boxes: 9.6% of its car/truck boxes carry the "
            "MINORITY COCO label for their own LVIS object type -- 107 sedans called "
            "`truck`, 50 minivans called `truck`, 23 pickups called `car` -- so a reviewer "
            "applying goods-versus-people would contradict the ground truth about once in "
            "ten. `enclosed` excludes motorcycles without a size rule, `road` excludes "
            "boats and aircraft, and `bus` stays out because COCO annotates it separately "
            "and consistently (88% `bus_(vehicle)`)."
        ),
    ),
    # Candidates from #3588, each rule measured with `coco_folds.py` before it
    # was written: the fold-in names the boundary case a reviewer will actually
    # meet. Long form, with the counts, in the annotation guide.
    "fork": ClassRule(
        name="fork incl sporks not strainers",
        test=(
            "Good: metal, plastic and disposable forks, serving, carving and fondue forks, "
            "and SPORKS. THE TEST IS THE TINES: prongs meant to PIERCE OR HOLD FOOD make a "
            "fork, and how many there are does not matter -- a carving fork has two. A "
            "spork has them, so it is a Fork and not a Spoon; where an object could be "
            "read as either, the tines decide. "
            "Bad: spatulas, tongs, whisks, skewers, and a PASTA STRAINER even the kind with "
            "protruding fingers, because those fingers separate pasta from water and are "
            "not meant to pierce anything. Vote Good only when the boxed "
            "object IS a fork, not when a fork sits somewhere inside a `silverware` or "
            "`utensil` box covering a whole place setting. When only the handle shows and "
            "the food gives nothing away, read the GRIP: a fist closed to stab is a fork, "
            "a spoon is never held that way. Known bad positive: 2322780 boxes a steam "
            "locomotive's cow-catcher as a fork (fork@medium, so rejectable)."
        ),
    ),
    "spoon": ClassRule(
        name="spoon incl plastic not spatulas or sporks",
        test=(
            "Good: teaspoons, tablespoons, soup, wooden, plastic, disposable and serving "
            "spoons, and ladles -- a ladle is a spoon with a deep bowl. Bad: spatulas, "
            "slotted turners, scoops, whisks, tongs, and a SPORK, which is a Fork -- "
            "tines decide, and no single object is both. Judge the object, not the drawer. "
            "When only the HANDLE shows, read the food: a handle out of cereal is a "
            "spoon, a handle out of a salad is a fork. The one rule here that infers "
            "from surroundings rather than the object, because the alternative deletes "
            "every partly buried spoon."
        ),
    ),
    "bowl": ClassRule(
        name="bowl incl plates not planters or wrappers",
        test=(
            "Good: bowls, plates (a paper plate is a plate), saucers, dishes, serving "
            "pots, baskets that hold food, disposable food containers, and a dog's water "
            "bowl. A paper food boat with turned-up sides is a bowl, flimsy or not. "
            "CONTAINING food does not make something a food container: a 5-gallon bucket "
            "of apples is not a bowl, nor is a shopping cart, a grocery store, or a car "
            "boot with the shopping in it. It has to be MADE to hold food. "
            "Bad: flat wrappers and sleeves, cups and mugs (`cup`), sink basins (`sink`), "
            "toilet bowls, feed troughs, planters (out of C, not `vase`), ashtrays and carafes. "
            "Judge the vessel, not the food -- which answers WHAT TO BOX. Contents "
            "answer WHICH CLASS when the vessel alone is ambiguous: full of soup is a "
            "Bowl whatever its shape; empty, the ladder decides (Bowl 0.66 h/w, Cup 1.26). "
            "No single object is both a Cup and a Bowl, though an image may hold one of each."
        ),
    ),
    "bottle": ClassRule(
        name="bottle incl jars",
        test=(
            "Good: water, wine, beer, soda and spirit bottles, jars ALWAYS and whatever is "
            "in them (a jar of flowers is a bottle, not a vase), jugs and pitchers -- a "
            "pouring vessel serving more than one is a bottle, not a cup -- soap and "
            "shampoo dispensers, shakers, spray bottles, baby bottles, condiment bottles, "
            "vacuum flasks, and the seasoning shelf -- shakers including SALT (16) and "
            "PEPPER (11) shakers, condiment, ketchup, mustard and oil bottles "
            "(181 boxes for the family), and a SQUEEZABLE TUBE -- toothpaste, suntan "
            "lotion, shower gel -- which is reasoned rather than measured (the toiletries "
            "family is 110 boxes but the tube shape itself only ~6). Bad: cans, cartons, boxes, a stemmed "
            "glass of wine, and a "
            "FUEL TANK. An integral component of a larger object is not an instance of a "
            "container class: a mouth is not a food container and a stomach is not a "
            "bottle. Same test as the feed trough in `bench`. "
            "Judge the container, not its contents; do not judge it by its neck, since "
            "`jar` (120) and `jug` (28) fold in and barely have one."
        ),
    ),
    # Until 2026-09-09 `vase` claimed "flower pots, planters" and "a potted
    # plant's pot is a vase", and `bowl` pointed planters AT `vase`. The reviewer
    # rejected planters through the whole finished vase slate and reported the
    # line as "a little weak", judging on proportions and ornateness instead --
    # which is the shape of a rule nobody can apply twice the same way.
    #
    # COCO disagrees with the old wording in two independent ways. `potted plant`
    # is a class of its own: 69% of its images carry no `vase` at all, and only
    # 82 of 2,500 potted plants reach IoU>0.5 with a vase box. And of 445 COCO
    # `vase` crops taken from images with NO potted plant, 82.0% look like a vase
    # and 4.1% like a flower pot or potted plant. (The first pass of that
    # measurement normalised COCO's boxes by the VG file's dimensions and read
    # 47.4%/3.6%; 88% of anchored pairs are a uniform ~1.28x rescale of each
    # other, which aspect-ratio gating cannot see -- the trap `anchor_to_coco`
    # documents and this analysis walked into anyway.)
    #
    # So a planter is not a vase, and `bowl` already refused it on its own
    # principle -- "it has to be MADE to hold food" -- which a planter fails
    # whatever its shape. Both doors shut, and `potted plant` is not in C, so the
    # object is out of C. The plant is not the operative fact; being made to hold
    # one is, which is why an EMPTY planter is out too.
    #
    # Sized before ruling: 167 of the 278 vase rejects score `potted plant` or
    # `flower pot` above `vase`, against 18 of the 116 accepts. Applying the old
    # wording literally would have roughly doubled vase's positives, so this
    # keeps the 394 verdicts as cast rather than voiding them.
    "vase": ClassRule(
        name="vase not planters",
        test=(
            "Good: only a vessel MADE as one -- vases, urns, decorative pottery. "
            "Against an ornamental BOWL, use the box: a vase is TALLER THAN "
            "WIDE (median h/w 1.58, 84% of boxes) and a bowl is wider than tall (0.66, "
            "13%); the middle halves do not overlap. Size does not help -- bowl's median "
            "box is the larger. "
            "A VESSEL MADE TO HOLD A GROWING PLANT is neither a vase nor a bowl -- a "
            "flower pot, planter or window box is out of C entirely, whether or not a "
            "plant is in it, because the plant is not what decides it. If you cannot "
            "tell a planter from a vase and it is empty, use the shape test above. "
            "Bad: a cooking pot on a stove, a plain bowl, and any BORROWED vessel however "
            "it is used -- a jar of cut flowers is a `bottle`, a glass of them a `cup`. "
            "A pitcher or jug of them is a `bottle`: COCO split them (pitcher to cup 30, "
            "jug to bottle 28), so the call is made on portion instead. "
            "Costs 192 boxes, 8.2% of COCO vase, which is the largest narrowing here."
        ),
    ),
    # The guide first named `chair` (53 boxes) as this class's confusion. It is
    # third: `seat` (64) and `table` (58) both outrank it, and each turns on a
    # question COCO's annotators do not ask.
    # `bench` and `chair` share one principle about vehicles, and until 2026-09-08
    # they contradicted each other: `chair` ruled a car seat out ("a component is
    # not an instance"; "A PART INHERITS THE RULING OF ITS WHOLE") while `bench`
    # ruled a rowboat's thwart IN. The reviewer found it mid-pass. `chair`'s
    # principle wins because it generalises -- one line covers cars, motorcycles,
    # buses and boats for both classes -- and because the alternative forces
    # `chair` to admit bus seats, which it refuses for a practical reason rather
    # than a principled one.
    #
    # The cost is real and worth naming: a swan boat's passenger bench is
    # unambiguously built as seating for two or more, and it is now Bad. The line
    # is "part of a vehicle", not "not really seating".
    #
    # **This ruling rests on consistency, NOT on a measurement**, and that is a
    # weaker footing than `chair`'s. The annotation guide settled `chair` with a
    # rate test -- every vehicle class holds a chair BELOW COCO's base rate
    # (airplane 0.14x, train 0.17x, bus 0.20x, boat 0.56x), which reads as
    # annotators treating a vehicle interior as containing none. Run for `bench`
    # the same test says the opposite -- boat 1.42x, train 1.40x, car 1.43x --
    # and it is confounded rather than contradicting: benches and vehicles share
    # outdoor scenes, so a waterfront bench behind a moored boat and a station
    # bench beside a train both count. The guide itself rules a platform bench a
    # Bench. Neither that test nor a box-containment test separates "in the
    # vehicle" from "near it" for this class, so what COCO does with built-in
    # boat seating is UNMEASURED. Rendered cases show genuine ones exist.
    "bench": ClassRule(
        name="bench not chairs",
        test=(
            "Good: any backed or backless seat BUILT AS SEATING for two or more -- park, "
            "bus-stop and station benches, church pews, picnic-table benches. Bad: a "
            "single chair, a sofa, a judge's bench (that is a table; the seating is the "
            "chairs behind it), a concrete planter wall or ledge people merely sit on, "
            "and SEATING BUILT INTO A VEHICLE -- a rowboat's thwart, a boat's built-in "
            "hull seating, a bus's bench seat. A bench standing loose on a deck or in a "
            "bus IS one, exactly as a car seat out of the car is a Chair. Three tests: "
            "seating or surface, built as seating or merely sittable, and free-standing "
            "or part of a vehicle."
        ),
    ),
    "chair": ClassRule(
        name="chair incl stools not couches",
        test=(
            "Good: dining, office, folding and deck chairs, armchairs, high chairs, "
            "stools and bar stools, and one seat within a row of stadium or theatre "
            "seating; `seat`/`seats` (269) is the third largest fold-in here. One seat "
            "is a Chair, two or more a couch -- upholstered is NOT the test, so a club "
            "chair or recliner counts. A lifeguard station counts (built as seating for "
            "one). Bad: couches and sofas (`couch`), benches, a TOILET (separate COCO "
            "class, zero confusions), and a CAR SEAT -- a component is not an instance, "
            "and counting them would fire on every street scene. A car seat REMOVED from "
            "the car is free-standing, so it counts; so do a motorcycle's seat and a "
            "saddle NOT (both are part of the vehicle or the tack, and both are zero "
            "boxes in COCO). Someone clearly sitting on an INVISIBLE chair: vote Good and "
            "draw NO box -- present, no size measured, which excludes the image from the "
            "negative pool without making it a positive. A PART INHERITS THE RULING OF ITS "
            "WHOLE: a chair back or leg is evidence of a Chair and you box the Chair, but "
            "a headrest in a car is part of a car seat, so it is not one."
        ),
    ),
    "sink": ClassRule(
        name="sink basin not counter",
        test=(
            "Good: kitchen sinks, bathroom sinks, pedestal basins, utility sinks, vessel "
            "basins; a double sink in one unit is one sink. Bad: bathtubs, showers, "
            "toilets, urinals. This class's risk is the BOX, not membership: box the "
            "basin and its tap, never the vanity or the run of counter."
        ),
    ),
    "fire hydrant": ClassRule(
        name="fire hydrant not standpipes",
        test=(
            "Good: street fire hydrants in any colour or design, including ones wrapped, "
            "repainted or half-buried in snow. Bad: building standpipes and wall-mounted "
            "siamese connections, bollards, water valves, parking meters, utility posts. "
            "Also Bad: a FIRE TRUCK (that is a `Truck`, another of the thirteen), and "
            "busted street plumbing whose break is hidden under water -- you cannot "
            "confirm what you cannot see. Good-with-no-box requires certainty of "
            "PRESENCE; unsure whether anything is there at all is Bad. "
            "The cleanest class measured -- a call that feels hard here usually means the "
            "object is something else."
        ),
    ),
    # COCO has no magazine class, so its annotators put magazines in `book`
    # while the human pass applied the narrower English reading -- leaving 21
    # verdicts on one definition and 49 on another. The dataset takes COCO's,
    # since that is the half with an exhaustive reference.
    # The class's whole risk is landlines: VG `phone` lands on no COCO class
    # 46.2% of the time, worse than `book`'s 43.3%. The first slate's test read
    # "anything with a cord or a base station is Bad", which discriminates on a
    # base being PRESENT when what it means is that the handset is not itself
    # the whole device -- so it rejected 2387021, a mobile phone in a charging
    # dock (#3612).
    "cell phone": ClassRule(
        name="cell phone not landlines",
        test=(
            "Bad if the handset needs the base to work -- landline handsets, desk phones, "
            "payphones, wall phones, intercoms. A mobile phone resting in a charging dock "
            "or cradle is still Good."
        ),
    ),
    # ------------------------------------------------------------------
    # The SHIPPED twelve (#3673). #3666 measured what their absence costs:
    # six of the nine pool-error finds in the negative pass were boundary
    # calls on rules that did not exist, and at a 1% rate one ruling moves a
    # class further than 3,000 extra uniform draws would.
    #
    # Every entry below was measured before it was written, with BOTH tests,
    # because the cheaper one gets two of them wrong. `coco_folds.py` gives the
    # box test -- which VG names land on a COCO box of the class -- and it says
    # COCO's annotators call a wristwatch a `clock` 35 times and a `canopy` or
    # `tent` an `umbrella` 58 times, more than `parasol`. Read alone it would
    # have folded both in. `name_evidence.py` gives the image test the pool
    # actually asks -- where the name is the SOLE evidence, does COCO find the
    # class? -- and refutes both: `watch` 11% against a 4.5% base, `canopy` 7%
    # and `tent` 10% against 3.7%, all under the 1/3 cut, all verdict
    # `neither`. A fold-in tail is COCO's inconsistency; it is not a definition.
    "clock": ClassRule(
        name="clock not watches",
        test=(
            "Good: a device whose job is showing the time and which stands, hangs or is "
            "mounted -- wall, tower, station, mantel, alarm and desk clocks, analogue or "
            "digital, and a bare clock face on a building. Bad: a WRISTWATCH or a watch on "
            "a table (`watch` was measured for this class and refused: over the 970 overlap "
            "images where it is the only evidence COCO finds a clock 11% of the time "
            "against a 4.5% base, with 3% box agreement -- the 35 COCO clock boxes a VG "
            "`watch` box lands on are a tail, and admitting them would define the class one "
            "way on the COCO half and another way on the half VG names alone). Also Bad: a "
            "departure board or scoreboard that happens to show the time (`display` scores "
            "2%), a clock drawn on a screen or printed on a page -- the depiction rule "
            "applies to every class -- and a sundial. The near-miss this settles is a "
            "wristwatch worn by a bystander, which is what the negative pass found (#3666)."
        ),
    ),
    "umbrella": ClassRule(
        name="umbrella incl parasols not canopies",
        test=(
            "Good: a hand-held umbrella open or furled, a parasol, and a beach or patio "
            "umbrella -- one central pole carrying a round canopy. Bad: a pop-up CANOPY or "
            "market stall, a tent, an awning over a shopfront, a sunshade sail. The test is "
            "the frame, not the shade it casts: ONE POLE AND A ROUND TOP is an umbrella, "
            "FOUR LEGS OR A WALL FIXING is not. Measured, and the box test disagrees with "
            "the image test here: COCO's annotators land `canopy` on a COCO umbrella box 32 "
            "times and `tent` 26, together more than `parasol`'s 38 -- but over the images "
            "where those names are the only evidence COCO finds an umbrella 7% and 10% of "
            "the time against a 3.7% base (`awning` 4%, `shade` 1%), all verdict `neither`. "
            "The near-miss this settles is a rank of pop-up canopies at a skate park (#3666)."
        ),
    ),
    "backpack": ClassRule(
        name="backpack not handbags or luggage",
        test=(
            "Good: a bag made to be carried on the back on shoulder straps -- rucksacks, "
            "daypacks, school bags, hiking packs -- whether worn, held or set down. Bad: a "
            "handbag, a shoulder or messenger bag, a suitcase, a duffel, a camera bag. COCO "
            "carries `handbag` and `suitcase` as their own classes, so this line is COCO's "
            "too. Two straps over two shoulders is the cue; a single diagonal strap is a "
            "shoulder bag. `bookbag` is on the ambiguous list (85% precision, 88% box) and "
            "`pack` is not a name for anything (38%). The near-miss this settles is the "
            "hump under a motorcyclist's leathers, which the pass could not call (#3666)."
        ),
    ),
    "stop sign": ClassRule(
        name="stop sign not other signs",
        test=(
            "Good: the octagonal red STOP sign, on a post, on a school bus arm, or held; "
            "from behind ONLY when the octagon is readable in the silhouette. Bad: every "
            "other traffic and street sign -- yield, one way, speed limit, street names -- "
            "a stop sign painted on the road, a pictogram, and a blank sign back whose "
            "shape you cannot read. `sign` is deliberately NOT a name for this class even "
            "though it carries 46.6% of COCO's stop-sign boxes: a VG `sign` box is a stop "
            "sign 7.9% of the time, which would withhold 12.7 pool images per contaminated "
            "negative retired (#3618, #3635). The near-miss this settles is the blank "
            "aluminium back of a sign on a street-name pole (#3666)."
        ),
    ),
    "book": ClassRule(
        name="book incl magazines",
        test=(
            "Good: a bound book, and also a magazine, a notebook and a bound pamphlet -- "
            "COCO has no magazine class and annotates magazines as `book`, which is the "
            "reading this dataset uses. Bad: newspapers, loose paper, letters, posters, "
            "menus, printouts, and a screen showing text. ONE TEST: IS IT BOUND ALONG A "
            "SPINE? Bound is a book; folded or loose sheets are not. Measured on the "
            "overlap, and read against `book`'s own 13% self-match rather than against "
            "100%: `magazines` 11% and `magazine` 10% land on a COCO book box at the same "
            "rate as `book` itself, `newspaper` 3% at a quarter of it, `menu` and `paper` "
            "at ~1%. This class is the study's calibration failure -- 43% of its VG boxes "
            "land on no COCO class, the worst of the twenty-five -- and the bound test "
            "narrows it without repairing that."
        ),
        # A stack or a shelf is several books however tightly packed; a book is
        # one volume, never one page (owner, 2026-09-22).
        unit="ONE book, not a stack or a shelf of them",
    ),
    # Ruled 2026-09-10 (#3789). The rule said "any LIVE bird" and the reviewer hit
    # a case it did not reach: a dead robin held in another bird's beak, at least
    # twice in the queue. Not live, so it failed the Good clause; not cooked, not a
    # part, not a depiction, so nothing in the Bad list caught it either.
    #
    # "live" was doing accidental work. Every entry on the Bad list carries a
    # principle -- it is food, it is a part not a whole, it is a representation not
    # an instance -- and prey satisfies none of them. The rule's own rationale is
    # entirely about VG naming (`chicken`, `turkey` usually mean dinner), which is
    # about food and never about death.
    #
    # Owner's ruling: whole dead birds are Birds, and taxidermy mounts are too. A
    # mount is the animal, not a figurine of one. "Prepared as food" replaces
    # "cooked", which also closes a hole nobody had hit: a raw plucked chicken in a
    # butcher's window is not cooked and is plainly not wanted here.
    #
    # Costs nothing to apply: the bird pass had not started (0 of 645), so no
    # verdict was cast under either reading.
    "bird": ClassRule(
        name="bird incl dead not food",
        test=(
            "Good: any WHOLE bird, wild or domestic, of any species, ALIVE OR DEAD -- a bird "
            "carried as prey, one lying dead, and a taxidermy mount, which is the animal "
            "itself and not a depiction of it. Ducks, geese, gulls, "
            "pigeons, swans, parrots, ostriches, owls, eagles, flamingos, peacocks, hens "
            "and roosters. Bad: a bird PREPARED AS FOOD -- cooked, plucked or butchered, on "
            "a plate, in a pan or in a display case -- a feather or a wing on its own, a "
            "bird figurine, a bird on a sign or a logo. The food clause is not "
            "hypothetical: `chicken` names 428 overlap images and COCO finds a bird on 10% "
            "of them, `turkey` 53 images at 12% -- in VG both words are usually food. "
            "`crane` is the other trap and it is a machine: 308 images, 2%. None of the "
            "three can be a name for this class, but a reviewer looking at an actual "
            "bird should vote Good. DEATH IS NOT THE TEST, food is: the exclusion exists "
            "because VG uses bird words to mean dinner, not because a dead bird stops "
            "being one."
        ),
    ),
    "dog": ClassRule(
        name="dog not wolves",
        test=(
            "Good: any live DOMESTIC dog, of any breed, size or life stage -- on a lead, "
            "loose, working, being carried, in a vehicle, or partly hidden. A puppy is a "
            "dog. Breed appearance is NOT the test: a husky, malamute or shepherd is a Dog "
            "however wolf-like it looks. Bad: an OBVIOUSLY WILD canid -- a wolf, coyote, "
            "fox, jackal or dingo -- and a HOT DOG, which is a trap for a name rather than "
            "for an eye (405 images on this name's head-noun family, 8 boxes in the whole "
            "COCO overlap). One test: wild or domestic, not what the coat looks like. "
            "`Obviously` is the operative word, exactly as in the toy rule: a wolf-like "
            "canid you have to squint at is somebody's husky, so vote Good. Excluding wild "
            "canids is COCO's own reading -- it annotates domestic dogs alone -- and it is "
            "the whole of what this rule adds. A dog FIGURINE, ornament or soft toy, and a "
            "dog in a photo, on a sign, a logo or a screen, are already Bad under the "
            "protocol's `a depiction is not the object` and `an obvious toy is not the "
            "object`, which bind every class here; they are deliberately not restated, "
            "because a generic rule restated per class is a generic rule that will drift."
        ),
    ),
    "boat": ClassRule(
        name="boat any watercraft",
        test=(
            "Good: anything built to travel on water and carrying its own hull -- ships, "
            "ferries, yachts, sailboats, canoes, kayaks, rafts, rowboats, gondolas, barges, "
            "tugs, pedal boats, jet skis. On a trailer or in dry dock still counts. Bad: a "
            "surfboard or paddleboard (COCO carries `surfboard` separately), a sail or a "
            "mast on its own, a dock, a buoy, a boat on a sign. `sailboat`, `canoe`, "
            "`kayak`, `raft` and `ship` are already folded in; `yacht`, `ferry`, `rowboat` "
            "and `barge` all measure 88-100% precision on small samples and are candidates "
            "for the same treatment."
        ),
    ),
    "bus": ClassRule(
        name="bus incl coaches not trams",
        test=(
            "Good: a road vehicle built to carry passengers in rows and boarded through "
            "its own door -- city buses, coaches, school buses, double-deckers, minibuses, "
            "tour buses, trolleybuses on tyres. Bad: a TRAM or train on rails, a cargo van, "
            "a truck, an RV or camper, a bus shelter. The boundary that costs verdicts is "
            "the van: `van` names three different vehicles and COCO splits it 261 truck / "
            "318 car / 37 bus, so read the BODY -- rows of seats and a passenger door is a "
            "Bus, a cargo box is a Truck. Two VG words are traps for a NAME and not for a "
            "reviewer: `coach` is usually a person -- 0% precision over its 50 sole images -- "
            "and `trolley` is usually a shopping cart (31% over 32)."
        ),
    ),
    "bicycle": ClassRule(
        name="bicycle incl trikes not motorcycles",
        test=(
            "Good: a human-powered pedal cycle -- road, mountain, BMX, folding, cargo and "
            "children's bicycles, ridden, parked or on a rack; a tricycle counts (71% "
            "precision, and COCO boxes six as `bicycle`). Bad: a MOTORCYCLE, moped or "
            "scooter (COCO carries `motorcycle` separately, and `motorcycle` measured as an "
            "alias of `bike` at 0.38 box IoU), an exercise bike, a wheel or a bike rack "
            "alone, and a bicycle PICTOGRAM on a road sign -- three of the ten "
            "`bicycle@small` positives in #3156 are exactly that, boxed as `bicycle` by "
            "COCO, and the depiction rule wins over COCO's box (#3614). The class is built "
            "from the spelling `bicycle` alone while `bike` carries 638 of COCO's 3,683 "
            "boxes against `bicycle`'s 775, so it is missing roughly half its positives on "
            "the non-COCO half (#3605)."
        ),
    ),
    "kite": ClassRule(
        name="kite incl parasails and parachutes",
        test=(
            "Good: a kite flown on a line, and -- this is the surprise, and it is COCO's "
            "reading, not ours -- a PARASAIL, a paraglider and a PARACHUTE: `parasail` "
            "lands on a COCO kite box 57 times and `parachute` 26, and both are already "
            "folded into this class. Bad: a flag, a banner, a balloon, a bird, a windsock, "
            "a kite tail or string on its own. A kite lying on the ground still counts."
        ),
    ),
    "knife": ClassRule(
        name="knife incl butter knives and servers",
        test=(
            "Good: a bladed cutting or spreading implement at the table or in the kitchen "
            "-- table, steak, butter, bread, chef's, paring and pocket knives, cleavers, "
            "and cake or pizza servers. Bad: SCISSORS (COCO carries its own class, and VG "
            "`scissors` finds a COCO knife on 3% of its 196 sole images), a spatula, a "
            "peeler, a knife block or a drawer with nothing visible, and a whole "
            "`silverware` or `utensil` box covering a place setting -- vote Good only when "
            "the boxed object IS the knife, the same rule `fork` carries. Where only the "
            "handle shows, read the blade line, not the food."
        ),
    ),
    # The remaining rules were measured as names before ``test`` existed
    # (#3588): `coco_folds.py` asks which VG names land on a COCO class's boxes
    # over the ~51k-image overlap, which enumerates a class's boundary cases
    # before a human meets one -- run against `book` it prints `magazine` (79)
    # and `magazines` (30). Each name below states the boundary case that
    # measurement found, and the long form now lives in each entry's ``test``
    # above -- filled in as each class was slated (#3588).
    # --- Added by #4056 ------------------------------------------------------
    # The twenty-five above were each measured with `coco_folds.py` before the
    # rule was written. That script was deleted by the VG retirement (#4038) and
    # is not coming back: fold-in asked which VG NAME lands on a COCO box, and
    # there are no VG names any more (#4060 tracks the guides that still name
    # it). Under a pure-COCO build the question changed -- nobody reviews COCO's
    # labels, so what a brief must be written against is what COCO's annotators
    # actually PUT in the class. That is `coco_class_purity.py` against LVIS's
    # defined synsets, the same measurement :data:`SCALE_CLASS_CONTENTS` carries,
    # quoted below wherever it names a boundary a reviewer will really meet.
    #
    # Five are cross-class pairs INSIDE C, which is the `truck`/`car` situation
    # and the one a reviewer gets wrong silently, because each side looks correct
    # alone: `skis`/`snowboard`, `backpack`/`handbag`/`suitcase`, `apple`/
    # `orange`, `tv`/`laptop`, `potted plant`/`vase`.
    "parking meter": ClassRule(
        name="parking meter head not post",
        test=(
            "Good: on-street parking meters, single and twin head, and multi-space pay "
            "stations. Bad: ticket machines that are not for parking, post boxes, "
            "bollards, utility pillars. 100% pure over 89 LVIS matches -- the cleanest "
            "class in C, so membership is never the question. The risk is the BOX: take "
            "the head and its housing, not the run of pole down to the pavement."
        ),
    ),
    "banana": ClassRule(
        name="banana fruit",
        test=(
            "Good: single bananas, hands and whole bunches, green or ripe, on a stall, a "
            "tree or a plate. Bad: plantains presented as plantains, and banana in a cut "
            "fruit mix where no whole fruit survives. 99% pure; pineapple, pear and "
            "carrot at 0.2% each are the only competing names, so the BOX is the whole "
            "question and membership almost never is. A bunch belongs to the class but "
            "is not ONE object: see ``unit``."
        ),
        unit="ONE banana, not a bunch or a hand",
    ),
    "skateboard": ClassRule(
        name="skateboard deck",
        test=(
            "Good: skateboards, longboards and cruisers -- ridden, carried, or lying on "
            "the ground; box the deck with its trucks and wheels. Bad: kick scooters, "
            "which have a handlebar and are not in C at all. 99% pure; `toy` at 0.4% is "
            "the only competing name."
        ),
    ),
    "mouse": ClassRule(
        name="mouse computer not animal",
        test=(
            "Good: computer mice, wired or wireless. Bad: the ANIMAL, which is the error "
            "the bare class name invites and which a reviewer will admit by reflex. Also "
            "bad: trackpads and trackballs built into another device. 99.3% of this "
            "class's boxes are `mouse_(computer_equipment)` and the remaining 0.7% is "
            "`control`; not one is an animal, so an animal here is an annotation error, "
            "not a boundary case."
        ),
    ),
    "frisbee": ClassRule(
        name="frisbee disc",
        test=(
            "Good: throwing discs of any colour -- in flight, held, or on the ground, "
            "including golf discs. Bad: plates, pan lids and toy rings. 99% pure; `toy` "
            "at 0.7% is the same object read more generically, not a different one."
        ),
    ),
    "surfboard": ClassRule(
        name="surfboard not snowboard",
        test=(
            "Good: surfboards, longboards, bodyboards and paddleboards -- in the water, "
            "under an arm, or stacked on a rack. Bad: SNOWBOARDS, which are their own "
            "class in C, and water skis. 99% pure, but `snowboard` and `water_ski` each "
            "appear at 0.2% and the confusion is one a reviewer makes from the outline "
            "alone once the scene is cropped away. Use the scene: water or beach is a "
            "surfboard, snow is a snowboard."
        ),
    ),
    "toothbrush": ClassRule(
        name="toothbrush manual or electric",
        test=(
            "Good: manual and electric toothbrushes, single or in a holder, including a "
            "brush head on its handle. Bad: hairbrushes, paintbrushes, and the bare "
            "charging base of an electric brush with no brush on it. 99% pure; `handle` "
            "and `screwdriver` at 0.6% each are the only competitors, and the "
            "screwdriver reading is a real one on a long plain electric handle."
        ),
    ),
    "baseball bat": ClassRule(
        name="baseball bat not pipe",
        test=(
            "Good: baseball and softball bats -- swung, held, racked, or on the ground; "
            "wood or metal. Bad: cricket bats, hockey sticks and tennis rackets, which "
            "are a different implement, and a bare pipe or post. 99% pure; `toy` 0.6% "
            "and `pipe` 0.3% are the only competing names."
        ),
    ),
    "keyboard": ClassRule(
        name="keyboard computer not piano",
        test=(
            "Good: computer keyboards, standalone or wireless. Bad: MUSICAL keyboards "
            "and pianos -- the error the bare class name invites, and `piano` does "
            "appear on 0.3% of these boxes. Also bad: on-screen keyboards. A laptop's "
            "own keyboard belongs here only where COCO boxed it apart from the machine; "
            "otherwise the object is `laptop`, which is its own class in C. 98.5% is "
            "`computer_keyboard`."
        ),
    ),
    "tennis racket": ClassRule(
        name="tennis racket not other rackets",
        test=(
            "Good: tennis rackets -- held, swung, bagged, or on the ground. Bad: "
            "badminton and squash rackets and table-tennis bats, which LVIS calls the "
            "generic `racket` and which appear at 1.3%. Head size and the court settle "
            "it. 98% pure."
        ),
    ),
    "scissors": ClassRule(
        name="scissors not shears or pliers",
        test=(
            "Good: scissors of any size, open or closed, including kitchen and craft "
            "scissors. Bad: garden shears and secateurs (`shears`, 1.0%), pliers (0.5%) "
            "and tongs (0.5%). The test is the pivot and TWO RING HANDLES; a sprung tool "
            "with no rings is not scissors. 98% pure."
        ),
        # Plural name, one tool. COCO and LVIS agree box for box (count ratio 1.00).
        unit="ONE pair of scissors, i.e. one tool",
    ),
    "traffic light": ClassRule(
        name="traffic light not street sign",
        test=(
            "Good: vehicle and pedestrian signal heads -- on a pole, gantry or wire, lit "
            "or unlit, any aspect count. Bad: street signs (1.5%) and plain street "
            "lights (1.0%), which share the pole and the silhouette. The test is whether "
            "it SIGNALS, not whether it is mounted high. 97.5% pure. VG could not carry "
            "this class at all -- its head noun `light` is not an object, so "
            "`scale_study_exclusion` barred it; COCO's vocabulary has no head nouns, "
            "which is the only reason it is admissible here."
        ),
    ),
    "skis": ClassRule(
        name="skis pair not poles",
        test=(
            "Good: skis -- on feet, carried, planted in the snow, or racked; a pair is "
            "one object wherever COCO boxed it as one. Bad: SNOWBOARDS, their own class "
            "in C, at 2.3%; and ski poles and boots (`ski_pole`, `ski_boot`, 0.5% each), "
            "which are never this class however tightly they sit beside it. 96.8% is "
            "`ski`."
        ),
        # Plural like `scissors`: COCO boxes the pair (count ratio 2.09 against
        # LVIS's single `ski`), and ruling one ski would reject ~62% of its boxes.
        unit="ONE pair on one skier or one loose ski, not a rack",
    ),
    "laptop": ClassRule(
        name="laptop not a monitor",
        test=(
            "Good: laptops and notebooks, open or closed, including one sitting on a "
            "dock. Bad: desktop MONITORS, which appear here at 2.2% and which `tv` -- "
            "also in C -- takes 47% of the time. The test is the HINGE: a screen joined "
            "to its own keyboard is a laptop, a screen on a stand is not. 96.7% is "
            "`laptop_computer`."
        ),
    ),
    "microwave": ClassRule(
        name="microwave oven not toaster",
        test=(
            "Good: microwave ovens, counter-top or built-in, door open or shut. Bad: "
            "toaster ovens (2.4%), conventional ovens and stoves (0.6%) and coffee "
            "makers (0.6%) -- all a box with a door at about the same height. The "
            "turntable and the control panel down one side are the cue. 96.5% pure."
        ),
    ),
    "snowboard": ClassRule(
        name="snowboard not skis",
        test=(
            "Good: snowboards -- ridden, carried, planted in the snow, or on a rack. "
            "Bad: SKIS, their own class in C, at 4.6%; and surfboards at 0.4%. One board "
            "under both feet is a snowboard; two narrow boards are skis. 94% pure, the "
            "lowest of the board classes, and the ski confusion is nearly all of it."
        ),
    ),
    "suitcase": ClassRule(
        name="suitcase not bag",
        test=(
            "Good: wheeled and unwheeled suitcases, hard and soft, upright or flat, on a "
            "carousel or a rack; an old cabin `trunk` (1.8%) is Good. Bad: BACKPACKS "
            "(1.1%) and HANDBAGS (0.1%), both their own classes in C, duffel bags (1.6%) "
            "and briefcases (0.5%). The test is the LID: a case opens on a hinge into "
            "two halves, a bag opens at the top. 93% pure."
        ),
    ),
    "airplane": ClassRule(
        name="airplane fixed wing",
        test=(
            "Good: fixed-wing aircraft of any size -- airliners, light aircraft, "
            "military jets (`fighter_jet`, 6.7%) and gliders, on the ground or in the "
            "air. Bad: HELICOPTERS (0.2%) and anything rotary, which the class name does "
            "not cover. 91% pure, and the 9% is almost entirely other names for the same "
            "object rather than other objects."
        ),
    ),
    "motorcycle": ClassRule(
        name="motorcycle incl scooters",
        test=(
            "Good: motorcycles, mopeds and motor scooters (`motor_scooter`, 8.1%) and "
            "dirt bikes (0.9%) -- ridden or parked. Bad: BICYCLES, their own class in C, "
            "at 0.7%; and a machine under a cover where only the tarp is visible "
            "(`tarp`, 0.5%). The test is the ENGINE, not the size or the step-through "
            "frame: a scooter with a motor is this class, a pedal cycle with a battery "
            "is `bicycle`. 89% pure."
        ),
    ),
    "tie": ClassRule(
        name="tie necktie incl bow ties",
        test=(
            "Good: neckties, worn or hung -- and BOW TIES, which are In, at 7.3% of this "
            "class's boxes. Bad: scarves (0.5%), necklaces (0.2%) and lanyards. COCO "
            "puts `bow-tie` in `tie`, which a reviewer reading the bare word narrowly "
            "will reject, so the bow-tie case is settled here rather than left to taste. "
            "89% is `necktie`."
        ),
    ),
    "apple": ClassRule(
        name="apple not other round fruit",
        test=(
            "Good: apples -- whole, on a tree, in a bowl, or on a stall, any colour. "
            "Bad: pears (5.1%), peaches (4.1%), tomatoes (1.0%) and ORANGES, their own "
            "class in C, at 1.4%. 80% pure, and the 20% is almost all other round fruit, "
            "so a mixed fruit bowl is where a reviewer's accuracy goes. Judge each "
            "FRUIT, never the bowl."
        ),
        unit="ONE apple, not a pile or bowl",
    ),
    "orange": ClassRule(
        name="orange citrus fruit",
        test=(
            "Good: oranges and mandarins (`mandarin_orange`, 6.6%) -- whole, on a tree "
            "or a stall. Bad: LEMONS (6.5%) and limes (1.4%), a different fruit however "
            "similar the shape; peaches (2.7%); carrots (1.8%); and APPLES, their own "
            "class in C, at 1.4%. 77% pure, second-lowest in C. Colour alone is not the "
            "test -- a green orange and a lime look alike."
        ),
        unit="ONE orange, not a pile or bowl",
    ),
    "potted plant": ClassRule(
        name="potted plant incl cut arrangements",
        test=(
            "Good: plants in a pot or planter (`flowerpot`, 20.7%) AND cut-flower "
            "arrangements, which are 75.1% of this class's boxes under LVIS's "
            "`flower_arrangement`. Bad: Christmas trees (2.4%), and a bare VASE with "
            "nothing in it -- its own class in C, at 1.2%. This is the least intuitive "
            "name in C: the class is overwhelmingly cut flowers rather than potted "
            "plants, so reading the name literally would reject three quarters of it. "
            "Where a vase holds flowers both classes can be right on one image, and each "
            "is judged on its own object."
        ),
        # Container WITH its contents, so COCO's box is right and LVIS's pot-only
        # `flowerpot` box is the odd one out (area ratio 5.92, count 1.40).
        unit="ONE pot or vase with its plant or flowers, not several",
    ),
    "person": ClassRule(
        name="person whole not garment",
        test=(
            "Good: people -- whole or partly occluded, any pose, any scale, including "
            "someone in the background. Bad: statues and figurines (1.2% together), "
            "depictions in a painting or on a sign, and mannequins. Read the composition "
            "note with care: the class is 64% `person` and most of the rest is CLOTHING "
            "(`wet_suit` 7.8%, `jacket` 5.4%, `dress` 2.7%, `coat` 2.4%, `shirt` 2.2%). "
            "That is not a definitional split -- LVIS boxes the garment where COCO boxes "
            "the wearer, and mutual best match pairs the two. The object is always the "
            "PERSON, never the garment."
        ),
    ),
    "handbag": ClassRule(
        name="handbag not backpack or suitcase",
        test=(
            "Good: handbags, shoulder bags (4.8%), tote bags (4.3%), clutches and purses "
            "-- carried, worn, or set down. Bad: BACKPACKS (6.5%) and SUITCASES (7.8%), "
            "both their own classes in C; and plastic or paper shopping bags "
            "(`plastic_bag` 4.2%, `shopping_bag` 2.7%). 63% pure and the confusion is "
            "with two classmates, so this is the `truck`/`car` situation: decide on the "
            "STRAPS and the CLOSURE, not the size. Two straps over both shoulders is a "
            "backpack; a rigid hinged case is a suitcase; everything else carried in the "
            "hand or on one shoulder is this."
        ),
    ),
    "remote": ClassRule(
        name="remote control handheld",
        test=(
            "Good: handheld remote controls for a television, projector or console, and "
            "game controllers. Bad: cell phones (0.7%), their own class in C; telephones "
            "(0.7%); calculators (0.4%). The 56% / 41% split between `remote_control` "
            "and `control` in the composition note is ONE object under two LVIS names, "
            "not two kinds of thing -- effective purity is about 97%, and membership is "
            "rarely the question."
        ),
    ),
    "tv": ClassRule(
        name="tv incl computer monitors",
        test=(
            "OWNER RULED 2026-09-20 over 40 questions: monitors are IN, at the same rate "
            "as televisions -- `television_set` 20/21 Good, `monitor_(computer_equipment)` "
            "18/19. The question put was the retrieval one a cell actually scores (the "
            "query is `a tv`), not a definitional one about objects, because the honest "
            "answer to `is this a tv` depends on USE -- the same panel is a tv with a "
            "console on it and arguably not one with a spreadsheet on it, and COCO's "
            "annotators never conditioned on that. In practice the distinction did not "
            "survive the images. "
            "Good: televisions AND desktop computer monitors -- both, deliberately. The "
            "class is 50.4% `television_set` and 46.9% "
            "`monitor_(computer_equipment)`, so it is a SCREEN class, not a television "
            "class, and reading the name literally would reject half of it. Bad: "
            "LAPTOPS, their own class in C, whose screen is never boxed here; projected "
            "images; framed pictures (`painting` 0.4%); signboards (0.9%). The hinge "
            "settles it again: a screen attached to its own keyboard is `laptop`, a "
            "standalone screen on a stand or a wall is this."
        ),
    ),
    "dining table": ClassRule(
        name="dining table surface in dining use",
        test=(
            "THE RULE IS FUNCTION, NOT FURNITURE. Owner ruled 2026-09-20 over 48 "
            "questions, and the answer separates sharply by what the surface is DOING: "
            "`tray` 4/4 Good, `plate` 4/4, `place_mat` 4/4, `dining_table` 4/4, "
            "`tablecloth` 14/15 -- against generic `table` 7/13 and `coffee_table` 1/4. "
            "Good: any surface in dining use and whatever covers it -- a laid table, a "
            "cloth-covered table, a tray or place setting shot close enough that the "
            "surface fills the frame. A close-up whose best LVIS match is the `plate` is "
            "still the surface, and is Good; an earlier version of this rule called "
            "those Bad and the ruling overturned it. "
            "Bad: coffee tables and other living-room furniture with no meal on them "
            "(3 of 4 rejected); CHAIRS and BENCHES, their own classes in C. A bare "
            "generic table is genuinely a coin-flip (7/13) -- if nothing says dining, it "
            "is not this class. "
            "The class is 34.2% `tablecloth`, 30.5% generic `table` and only 10.4% "
            "`dining_table` by LVIS, the lowest purity in C, admitted because the "
            "selection rule is the count and nothing else. Quote the composition note "
            "beside any `dining table` number; the name does not describe the class."
        ),
    ),
}


def review_name(cls: str, suffix: str = "") -> str:
    """The dataset/detector name a slate of *cls* is reviewed under.

    The class's rule name where it has one, else the bare class name, plus an
    optional *suffix* naming the pass (``positives``, ``audit``). Every slate
    maker builds its ``detector`` column from this, so a class's rule reaches
    the reviewer whichever pass they are voting -- the first pass included,
    which is where a definition split does its damage.
    """
    rule = SCALE_CLASS_RULES.get(cls)
    return f"{rule.name if rule else cls}{f' {suffix}' if suffix else ''}"


#: The pass suffixes :func:`review_name` appends. Listed so the join can be
#: undone: a slate's ``detector`` column is the only place a *past* review's rule
#: name survives, and reading it back is what lets a verdict be stamped with the
#: wording its reviewer saw rather than the wording in force today.
REVIEW_SUFFIXES = ("positives", "audit", "reviewed")


def rule_of_review_name(detector: str) -> str:
    """The rule name inside a slate's ``detector``, with the pass suffix removed.

    The inverse of :func:`review_name`, and it has to be an inverse rather than a
    guess: a detector name is *evidence about the past*. ``make_class_recheck.py``
    additionally appends a ``" -- ..."`` tail naming the question, which
    ``bank_verdicts.py`` already strips the same way.

    ``tests_lib/meta/test_pile_rule_version.py`` pins the round trip over every
    class and every suffix, so a new suffix that is not listed above fails there
    rather than silently reading as part of the rule.
    """
    name = detector.split(" -- ")[0]
    for suffix in REVIEW_SUFFIXES:
        if name.endswith(f" {suffix}"):
            return name[: -len(suffix) - 1]
    return name


def is_scale_review(detector: str, text_query: str) -> bool:
    """Whether a dashboard detector is a vg_scale review at all.

    The dashboard is shared: other projects load their own review queues onto the
    same app (DocMarks does, 2026-09-17). ``bank_verdicts.py`` exported EVERY voted
    detector, so a foreign queue would have been banked into vg_scale's human
    record as a ``slate`` labelset -- and ``retire_finished.py``, finding that file,
    would then have deleted the other project's finished pairs. Both scripts ask
    this first and leave anything else alone.

    A vg_scale review is queried by one of its classes, and its name, once the
    question suffix and any bracketed tail are stripped, is that class's rule --
    which always begins with the class name. The rule itself may be an OLD one (a
    detector left over from before a ruling still banks, without a digest), so the
    name is tested by its prefix rather than against the rule in force.
    """
    if text_query not in SCALE_CLASSES:
        return False
    rule = rule_of_review_name(detector.split(" [")[0].split(" (")[0])
    return rule == text_query or rule.startswith(f"{text_query} ")


def detector_kind(detector: str) -> str:
    """Which question a dashboard detector asks, and so which labelset file it banks to.

    The single home for a mapping ``bank_verdicts.py`` and ``retire_finished.py``
    used to copy inline. They must agree: ``retire_finished.py`` re-verifies the
    live votes against the file ``bank_verdicts.py`` wrote, so if the two ever
    derived different kinds it would compare against the wrong file.

    A class-named detector with no marker is the pass's ``slate`` ("is this box
    one?"). **Every other question must carry a marker, or it banks over that
    class's slate labelset** -- a ten-image audit named ``chair incl stools not
    couches`` would overwrite the 455-Good chair slate.

    ``-- prominent check`` asks whether the boxed instance is the most prominent
    one, and is distinct from ``-- box check`` so a triage of a class cannot
    overwrite that class's box-check audit either. ``-- seat check`` reviews the
    positives a build actually SEATED in a class's cells (#3926) -- whatever their
    source -- and asks both questions at once, so it needs its own file beside the
    prominence triage of the same class. The markers do not overlap, so their
    order does not matter.
    """
    if "-- recheck" in detector:
        return "recheck"
    if "-- box check" in detector:
        return "boxcheck"
    if "-- prominent check" in detector:
        return "prominent"
    if "-- seat check" in detector:
        return "seatcheck"
    if ("any in image" in detector) or ("below-cut" in detector):
        return "belowcut"
    return "slate"


def rule_digest(cls: str) -> str:
    """A short hash of *cls*'s rule **as written** -- the name and the test together.

    The name answers "was the reviewer shown different words?". It does not
    answer "did the rule move?", because a rule can be edited in the body while
    the name stands still: #3756 rewrote `bench`'s Bad list and kept its name.
    Both questions are worth asking and only one of them is readable in a diff,
    so both are recorded -- the name because it is the wording a reviewer saw and
    a human can check it at a glance, the digest because it is the only thing
    that can see an edit the name hides.

    Twelve hex characters, which is a hash to compare rather than a hash to
    defend: the adversary here is a forgotten edit, not a forger.
    """
    rule = SCALE_CLASS_RULES.get(cls)
    payload = f"{rule.name if rule else cls}\n{rule.test if rule else ''}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def rule_stamp(cls: str) -> dict[str, str]:
    """The two fields a row records about the rule it was answered under (#3814).

    Merge into a verdict or a correction row at the moment the answer is
    *given*, never at the moment a file is regenerated: a row cast in August and
    re-derived in September must carry August's rule, or the stamp asserts the
    one thing it exists to disprove. Every writer here therefore takes its name
    from what the reviewer was shown -- the slate's ``detector`` column, or a
    labelset's recorded ``rule`` -- and falls back to this only when the two
    already agree.

    **Absence means unknown and must never be read as current.** The 872 rows
    predating this field cannot be back-filled honestly: nothing on disk says
    which wording they were cast under, which is the whole of #3814. See
    :func:`pilebuild.corrections.rule_state`.
    """
    return {"rule": review_name(cls), "rule_digest": rule_digest(cls)}


def scale_vg_wanted() -> set[str]:
    """Every VG name the ``vg_scale`` read must match, spellings included.

    The class names themselves plus both name tables. Read with this rather than
    ``set(SCALE_CLASSES)``: a spelling absent from the read is invisible later,
    since an image holding it then looks like an image holding nothing.
    """
    wanted = set(SCALE_CLASSES)
    for table in (SCALE_VG_NAMES, SCALE_VG_AMBIGUOUS):
        for names in table.values():
            wanted.update(names)
    return wanted


#: Images per ``(class, band)`` cell, and the shared negative pool every cell
#: draws from. The pool is the whole of VG, labelled by VG and repaired from
#: COCO where an exhaustive reference exists, so the binding supply is the union
#: of both halves; the builder logs any cell it cannot fill.
#:
#: Cells are **designated**, not inferred: each is exactly these positives plus
#: this negative pool, and every other image in the pickle is excluded from it.
#: Prevalence is therefore identical in all 36 cells by construction, which is
#: what makes small-vs-large a paired comparison rather than two datasets with
#: different difficulty. Unequal prevalence between arms is what made wave 1 and
#: wave 2 of the overview benchmark non-comparable.
SCALE_N_POS = int(os.environ.get("VTS_SCALE_N_POS", "100"))
SCALE_N_NEG = int(os.environ.get("VTS_SCALE_N_NEG", "9900"))
#: Extra negatives drawn into the pickle but designated into no cell. A human
#: verdict can retire a contaminated negative later; re-designating from a spare
#: is a relabel, while drawing a fresh one would mean re-embedding every cell.
#:
#: **1,000 since the #3588 promotion, up from 300 — and the reason is no longer
#: the one that raised it.** It was raised to survive a correction pass over a
#: contaminated pool: a shared negative is evaluable in *every* cell
#: (:func:`pilebuild.loaders.vg_scale._evaluable` returns the whole cell list for
#: an image with no categories), so a verdict finding a car in a pool image does
#: not retire it from `car` alone — the image stops being clean and leaves all
#: twenty-five. Against the negative pass's measured **14% ± 7** joint rate that
#: is ~550 retirements per 3,900 negatives, and 300 would be spent by one pass.
#:
#: #3670 landed in the same window and made that arithmetic moot: the pool is now
#: drawn entirely from the COCO-scored half, where "holds none of *C*" is a fact,
#: so contamination is **0 by construction** and a verdict retires a negative only
#: where a human finds what COCO's exhaustive annotation missed. What the number
#: buys today is therefore headroom rather than a budget — for `vg_scale_deep`,
#: whose composition #3690 pinned separately, and for any future pass over either.
#: It is kept at 1,000 because 1,000 images out of ~14,000 medias is noise against
#: a rebuild, while running out mid-pass turns a relabel into a re-embed of every
#: cell. Do not read the size as evidence that the pool is dirty.
SCALE_N_NEG_SPARE = int(os.environ.get("VTS_SCALE_N_NEG_SPARE", "1000"))

#: What the shared negative pool is made of (#3670).
#:
#: ``provable``
#:     Every designated negative is COCO-scored, so "holds no bus" is a FACT --
#:     COCO annotates all eighty of its classes on any image it touches -- rather
#:     than VG's silence, which #3666 measured wrong **1.40%** [0.68, 2.86] of
#:     the time pooled over the shipped twelve, and #3635 predicts between
#:     **0.28% and 2.87%** depending on the class.
#: ``matched``
#:     The negatives' COCO share is matched to the positives' own (~57%), so
#:     provenance carries no information about the label. Keeps the
#:     contamination, and keeps the whole negative review (#3670 measured that
#:     price: `provable` rules 513 of 743 reviewed negatives ineligible).
#:
#: **THE ARGUMENT THIS WAS SHIPPED ON DOES NOT HOLD. The setting stands; its
#: justification does not, and the choice is open (#3702).**
#:
#: What was claimed: both compositions distort, the magnitudes are not separable,
#: but the provenance shortcut is *uniform across classes* while contamination is
#: not -- so the uniform one cancels in the class-vs-class contrast this dataset
#: exists to make. Two errors underneath it, both in data already on disk:
#:
#: * the "two independent routes agreeing on 1.1x" were **one route counted
#:   twice**. Contamination inflates the forward arm and depresses the reverse
#:   arm; under pure contamination they must sum to 2. See
#:   `provenance_shortcut.py`, which carries the derivation and what the sum is
#:   good for instead.
#: * **the artifact is not uniform.** On the contamination-free diagnostic
#:   (`forward + reverse - 2`, which is 0 for pure contamination) the classes run
#:   `bus` 0.65, `bicycle` 0.60, `umbrella` 0.57 against `knife` 0.07, `boat`
#:   0.13, `book` 0.14 -- a near-tenfold spread, concentrated in street scenes,
#:   where COCO and YFCC100M differ most. It does not track contamination either:
#:   `backpack` is the dirtiest class at 2.8% and sits mid-table, while `bus` is
#:   among the cleanest and is the worst affected.
#:
#: So both distortions vary per class, and the spread argument selects neither
#: composition. What survives is that contamination is 1.18x [1.09, 1.37] at
#: #3666's measured 1.40%, and that the provenance artifact is real -- the sums
#: are 2.35 and 2.36, well clear of 2 -- but **unsized**, since `1/reverse` is
#: only a lower bound. Sizing it needs ~400 human-cleared off-COCO negatives per
#: class; the existing labels give ~53, at which the ratio's standard error is
#: +/-0.60. (#3667's cross-class shortcut, for scale, was 1.88x and justified
#: rebuilding eleven cells.)
#:
#: Switching back to ``matched`` needs the off-COCO stratum EMBEDDED, which the
#: all-provable build does not carry -- it is a rebuild, not a relabel.
SCALE_NEG_COMPOSITION = os.environ.get("VTS_SCALE_NEG_COMPOSITION", "provable")

#: The **designed** prevalence of a `vg_scale` cell: 100 positives per band x 3
#: bands against the shared negatives, which #3670 took from 3,900 to 9,900
#: so a cell's prevalence is 1%. Named because `vg_scale_deep` has to
#: REPRODUCE it rather than re-derive it, and because every k* this family of
#: studies quotes is computed from it (`-log2((1-pi)/pi) = -3.71`).
#:
#: DESIGNED, not realised, and since #3667 the two differ. The harness scores
#: the *evaluable pool*, which grew ~45% when each class gained the other
#: eleven's COCO-exhaustive positives as negatives -- no positive was added, so
#: prevalence fell: `vg_scale_any` to **4.99%** and `vg_scale_deep` to **5.09%**
#: (k* -4.25 and -4.22, against the -3.70 this constant gives). The two cells
#: moved TOGETHER, 0.03 bits apart, so the "only depth changed" premise below
#: survives; what does not survive is quoting -3.71 as the dataset's own
#: optimum. See #3681 and
#: `docs/experiments/2026-09-06-cross-class-negatives-3667/REPORT.md`.
SCALE_PREVALENCE = (3 * SCALE_N_POS) / (3 * SCALE_N_POS + SCALE_N_NEG)

#: `vg_scale_deep`'s positives per class (#3547). 900 is the deepest value all
#: twenty-five classes support band-free -- `stop sign`, the thinnest, has 1006
#: candidates (`measure_supply.py`) -- and it is chosen against `preflight.sh`
#: check 16b, which clears only when the sim half holds MORE positives than the
#: horizon has steps: at `SIM_FRACTION` 0.5 that is 450 against 400.
#:
#: **The #3588 promotion did not move it, which was not a foregone conclusion.**
#: Thirteen classes joined *C* and the binding constraint stayed exactly where it
#: was: the thinnest of the thirteen band-free is `fire hydrant` at 1138, still
#: clear of `stop sign`'s 1006, so depth and class list did not have to trade
#: against each other. Had one of the thirteen come in under 900, the choice
#: would have been between a deep set that carries fewer classes than the shallow
#: one and a depth change that restates #3547's published optimum.
#:
#: Going deeper costs classes, not money: 1200 drops `kite`, `stop sign` and
#: `fire hydrant`, and a class list that differs from #3319's would confound the
#: horizon axis with a vocabulary axis in the one comparison this dataset exists
#: to make.
SCALE_DEEP_N_POS = int(os.environ.get("VTS_SCALE_DEEP_N_POS", "900"))
#: `vg_scale_deep` does NOT follow #3670's expansion, and the pin is deliberate.
#: Deriving its pool from the live `SCALE_PREVALENCE` would have taken it from
#: 11,700 negatives to 29,700 as a silent side effect of a change to a DIFFERENT
#: dataset. Deep exists for one comparison -- the #3319/#3547 acquisition horizon
#: -- and moving its prevalence mid-stream would confound that axis with a
#: prevalence axis, which is the argument `SCALE_DEEP_N_POS` already makes about
#: holding the class list fixed. Whether deep should follow is #3690.
#:
#: Still DERIVED, never set: a negative pool written as a literal beside a
#: positive count is how prevalence drifts. What is pinned is the `vg_scale`
#: pool size deep's prevalence refers to, not the pool itself.
SCALE_DEEP_PIN_N_NEG = int(os.environ.get("VTS_SCALE_DEEP_PIN_N_NEG", "3900"))
SCALE_DEEP_PREVALENCE = (3 * SCALE_N_POS) / (3 * SCALE_N_POS + SCALE_DEEP_PIN_N_NEG)
SCALE_DEEP_N_NEG = round(SCALE_DEEP_N_POS * (1 - SCALE_DEEP_PREVALENCE) / SCALE_DEEP_PREVALENCE)
SCALE_DEEP_N_NEG_SPARE = int(os.environ.get("VTS_SCALE_DEEP_N_NEG_SPARE", "300"))


#: How far a VG copy's aspect ratio may drift from the COCO original before its
#: boxes are considered untransferable, as a fraction of the COCO ratio.
#: Normalised coordinates survive a rescale but not a re-crop or a rotation, and
#: 49 of the 51,497 overlaps are one of those -- small enough to ignore by
#: accident, which is why it is a constant with a check rather than an
#: assumption.
#:
#: The threshold is *relative*, and only :func:`aspect_transferable` may read
#: it: see that function for why a bare comparison against this number is a bug
#: rather than a shorthand (#3657).
MAX_ASPECT_DRIFT = float(os.environ.get("VTS_MAX_ASPECT_DRIFT", "0.01"))


def aspect_drift(vg_wh: tuple[int, int], coco_wh: tuple[int, int]) -> float:
    """How far a VG copy's aspect ratio has drifted from its COCO original.

    Relative to COCO's ratio, because COCO's is the reference: the question is
    whether COCO's normalised box still describes VG's pixels, so the drift is
    measured as a fraction of the framing the box was recorded in.
    """
    coco_ratio = coco_wh[0] / coco_wh[1]
    return abs((vg_wh[0] / vg_wh[1]) - coco_ratio) / coco_ratio


def aspect_transferable(vg_wh: tuple[int, int], coco_wh: tuple[int, int]) -> bool:
    """May a normalised COCO box transfer to the VG copy of the same image?

    One implementation, so that anything asking "do these two copies frame the
    same thing?" asks it the same way -- the treatment
    :func:`pilebuild.loaders.vg_scale.band_for` got for the banding question.

    It is a function rather than a bare ``> MAX_ASPECT_DRIFT`` because the two
    readings of that constant are indistinguishable at a glance and were both
    live: the loader divided by COCO's ratio, three analysis scripts compared an
    absolute difference of ratios, and the disagreement ran in *both* directions
    depending on orientation. A landscape 4:3 original (ratio 1.333) let the
    loader tolerate |delta| up to 0.0133 where the scripts stopped at 0.0100; a
    portrait 3:4 (0.75) had the loader stop at 0.0075 while the scripts still
    allowed 0.0100. So the set of images the *measurement* called adjudicable
    was not the set the *build* anchored, and it was skewed by orientation
    rather than by a uniform margin (#3657).

    The relative reading is the one that shipped the dataset, so it is the one
    that survives; the absolute call sites moved to it.
    """
    return aspect_drift(vg_wh, coco_wh) <= MAX_ASPECT_DRIFT


#: The coordinate space a correction box is recorded in. VG's and COCO's boxes
#: arrive in **pixels**; a correction box comes from the app's ``region_box``,
#: which is already **normalised** to [0, 1]. The builder divides every box by
#: (W, H) on the way into the pickle, so a correction box merged in unconverted
#: is normalised twice: it lands on the frame origin, sub-pixel, and takes its
#: band with it (#3281 -- 130 boxes, and 97 images filed in ``@small`` whose
#: object is medium or large). The space is therefore *declared* in the file and
#: converted once at read, never inferred: the two spaces are indistinguishable
#: for a box in the top-left corner of a 1x1 image, which is exactly the shape
#: the bug produced.
CORRECTION_BOX_SPACE = "normalised"

#: Below this normalised side length a box is sub-pixel on any image the pile
#: holds -- VG's largest copy is 1280 px wide -- so it cannot describe anything
#: that was observed. Zero legitimate boxes are anywhere near it; the 130
#: double-normalised ones were all under 1e-3.
MIN_BOX_SIDE = float(os.environ.get("VTS_MIN_BOX_SIDE", "0.000244"))  # 1/4096

#: "Crushed to the origin": both corners inside the top-left square holding this
#: fraction of the frame area. Unlike the sub-pixel rule this one has genuine
#: hits -- a small object really can sit in the top-left corner, 43 of 3470
#: healthy boxes do -- so it gates on the *rate*, not on any single box.
CORNER_AREA_FRAC = float(os.environ.get("VTS_CORNER_AREA_FRAC", "0.01"))

#: The share of a cell's boxes that may be crushed to the origin before the
#: build is refused. The measured healthy rate is 1.2% and the defect put it at
#: 100% of the affected images, so anything in between separates them.
MAX_CORNER_RATE = float(os.environ.get("VTS_MAX_CORNER_RATE", "0.05"))


#: Which images each cell currently holds. Selection is hash-stable, but a
#: roster is what carries membership across a CHANGE of selection rule -- and
#: across the corrections that are the whole point of the review, since a review
#: is only worth what it still covers after the next rebuild.
ROSTER = Path(os.environ.get("VTS_SCALE_ROSTER", str(PILE / "vg_scale_roster.json")))

#: `vg_scale_deep`'s own roster. Separate from `ROSTER` on purpose: the two
#: datasets designate different cells from the same candidates, and one file
#: holding both would let a `vg_scale_deep` rebuild retire images `vg_scale`'s
#: review is pinned to.
DEEP_ROSTER = Path(os.environ.get("VTS_SCALE_DEEP_ROSTER", str(PILE / "vg_scale_deep_roster.json")))


#: Where the `vg_scale` review has always kept its slates, verdict files and
#: adjudications. On scratch, like the pile.
SCALE_WORK = Path(os.environ.get("VTS_SCALE_WORK", str(PILE.parent / "vgscale-3156")))

#: The ``source`` of a verdict that changes nothing: a reviewer looked at a
#: designated positive and agreed with the box it already had.
#:
#: It exists because "nothing to correct" and "nobody looked" were the same
#: absence until #3727 -- `verdicts_to_corrections.py` counted a confirmation
#: and wrote no row, so `designate_cells` could not give a confirmed image the
#: seat priority that keeps a rebuild from reshuffling it away. The exhaustive
#: pass (#3720) makes that the *modal* outcome: 3,391 images whose answer is
#: mostly "yes, and here is what else is in it".
#:
#: **A confirmation carries no boxes, and must never be read as a boxless
#: `present`.** That other row means "it is here but I could not draw it", and
#: `apply_corrections` answers it by popping the class -- which on a confirmed
#: positive would delete the very thing being confirmed. Both sides therefore
#: branch on this constant rather than on the emptiness of ``boxes``.
CORRECTION_SOURCE_CONFIRMED = "human_confirmed"

#: Where labelsets exported from the app were parked. Off scratch -- which is
#: why part of the record survived, and #3729 is the observation that this was
#: luck rather than a policy.
SCALE_LABELSETS = Path(os.environ.get("VTS_SCALE_LABELSETS", f"/exp/{USER}/vgscale-3156-labelsets"))

#: The #3588 promotion campaign's working directory -- the review that cleared
#: the thirteen classes now in *C*. A separate campaign with its own verdicts,
#: its own slates and its own corrections output, and the live
#: `corrections.json` is the UNION of it and the #3156 one (#3732).
SCALE_WORK_3588 = Path(os.environ.get("VTS_SCALE_WORK_3588", str(PILE.parent / "classes-3588")))

#: The roots :data:`HUMAN_RECORD` sources are written against, so an inventory
#: row is portable between users and machines instead of embedding one person's
#: home directory the way the review scripts' defaults did.
RECORD_ROOTS: dict[str, Path] = {
    "PILE": PILE,
    "WORK": SCALE_WORK,
    "LABELSETS": SCALE_LABELSETS,
    "WORK3588": SCALE_WORK_3588,
    "EXP": Path(os.environ.get("VTS_EXP_HOME", f"/exp/{USER}")),
}


class HumanArtifact(NamedTuple):
    """One entry in the record of what people actually answered.

    ``source`` is ``{ROOT}/pattern``, resolved against :data:`RECORD_ROOTS`; the
    pattern may glob, because the slate manifests and labelsets arrive one per
    class rather than as a single file.
    """

    source: str
    #: ``human`` cannot be regenerated at any price -- it is a record of someone
    #: having looked. ``support`` is machine-written but is what makes a human
    #: answer *interpretable* (which class, which cell, which stratum a row was
    #: about), so losing it turns verdicts into anonymous votes. ``derived`` is
    #: rebuilt by every build from the two above and costs only compute.
    tier: str
    #: Why this file is in the inventory. Read by whoever wonders whether it can
    #: be deleted; the answer has to travel with the row rather than in someone's
    #: memory, which is the failure #3729 describes.
    why: str

    def root_token(self) -> str:
        return self.source.split("/", 1)[0].strip("{}")

    def root(self) -> Path:
        return RECORD_ROOTS[self.root_token()]

    def pattern(self) -> str:
        return self.source.split("/", 1)[1]

    def resolve(self) -> list[Path]:
        """Every existing file this row names on this machine (possibly none)."""
        base = self.root()
        if not base.exists():
            return []
        if any(ch in self.pattern() for ch in "*?["):
            return [p for p in base.glob(self.pattern()) if p.is_file()]
        one = base / self.pattern()
        return [one] if one.is_file() else []


#: **What no rebuild can bring back** (#3729).
#:
#: The pile is purgeable by design and every cell is rebuildable from off-scratch
#: sources -- see this module's docstring. Human answers are not, and until this
#: table existed nothing said which files were which: the verdicts, the app
#: labelsets behind them and the adjudications sat on the same purgeable mount as
#: the cells, with ad-hoc dated snapshots of *some* of them in two unrelated
#: directories and none of the September adjudication.
#:
#: `scripts/experiments/pile/verdict_store.py` copies these into `data/vg_scale/`
#: and checks that the copy is current. Add a row when a new kind of human
#: judgement starts being recorded -- that is the moment it is cheap.
HUMAN_RECORD: tuple[HumanArtifact, ...] = (
    HumanArtifact(
        "{WORK}/verdicts_*.json",
        "human",
        "the reviewer's per-(image, class) Good/Bad, with the stratum it was asked under; "
        "`verdicts_to_corrections.py` reads one of these by default",
    ),
    HumanArtifact(
        "{EXP}/verdicts_snapshot_*.json",
        "human",
        "an earlier ad-hoc snapshot, kept because nothing else records what it holds",
    ),
    HumanArtifact(
        "{LABELSETS}/*.json",
        "human",
        "app labelset exports -- the clicks themselves, which `ingest_slate.py` turns into verdicts -- "
        "plus the verdict files parked here by hand, one of which is hardcoded in "
        "`verdicts_to_corrections.py`'s default",
    ),
    HumanArtifact(
        "{LABELSETS}/audit-raw/*.json",
        "human",
        "the audit pass's raw labelsets",
    ),
    HumanArtifact(
        "{WORK}/adjudicat*.json",
        "human",
        "second opinions on reviewer/reference disagreements, with the reasoning that decided them",
    ),
    HumanArtifact(
        "{WORK3588}/verdicts_*.json",
        "human",
        "the #3588 promotion campaign's verdicts -- a second review, in its own directory, and "
        "absent from this inventory until #3732 noticed the live corrections file depends on it",
    ),
    HumanArtifact(
        "{WORK3588}/corrections_13.json",
        "human",
        "that campaign's corrections output, merged into the live `corrections.json` and not "
        "reproducible without the campaign's own verdicts",
    ),
    HumanArtifact(
        "{WORK3588}/slates*/*/manifest.csv",
        "support",
        "which (image, class, cell) each of the promotion campaign's slate rows was",
    ),
    HumanArtifact(
        "{WORK}/slates*/*/manifest.csv",
        "support",
        "which (image, class, cell) each slate row was: without it a labelset is a pile of "
        "anonymous votes and no verdict can be reconstructed from it",
    ),
    HumanArtifact(
        "{PILE}/corrections.json",
        # Reproducible again as of #3732, and kept `human` anyway. The recipe
        # that produces it is now this script's defaults -- three verdict files
        # UNION the #3588 campaign's own corrections output -- and it was
        # recovered from a session transcript, not from anything in the repo.
        # A file whose reproduction depends on a second campaign's scratch
        # directory surviving is not `derived` in the sense that word is used
        # here: losing that directory makes this the only copy again.
        "human",
        "the verdicts the build actually reads; reproducible only from the recovered two-campaign "
        "recipe in `verdicts_to_corrections.py`, and only while `{WORK3588}` survives",
    ),
    HumanArtifact(
        "{PILE}/vg_scale_roster.json",
        "derived",
        "the membership a review was carried out against; a build rewrites it, and losing it "
        "means the next rebuild reshuffles reviewed images away",
    ),
    HumanArtifact(
        "{PILE}/vg_scale_deep_roster.json",
        "derived",
        "the deep sibling's roster, same argument",
    ),
)


def scale_cell(category: str, band: str) -> str:
    """The band-suffixed category name a harness cell is keyed on.

    One pickle carries all three bands, distinguished by this suffix, because a
    cell is already ``(dataset, category)`` -- so the bands need no harness
    change, embedding is done once instead of three times, and the bands are
    paired on identical negatives.
    """
    return f"{category}@{band}"


#: Embedders in the pile. ``patch`` embedders attach ``patch_grid`` and are the
#: only ones that can carry a region-voting arm. ``batch`` is the GPU forward
#: batch size (``VTSEARCH_EMBED_BATCH_SIZE``); the app's default of 32 is sized
#: for a modest card and wastes a build GPU on a base-sized encoder, while a
#: SO400M/384 model at 32 is already the heavy end. Sizes are per model, not per
#: run, so a fatter card only means the whole table can move up.
#:
#: Batch size does not change *which* images are embedded, but it does change
#: their vectors, by more than this comment used to claim. #3683 rebuilt
#: ``siglip2_l`` at 31 instead of 32 -- same images, same node, everything else
#: equal -- and 27 of 7,746 vectors moved, by up to **1.6e-04**: the batched
#: GEMM's reduction order is not independent of what an image was batched with,
#: even in fp32. Median 0, and 7,719 of 7,746 bit-identical, so the effect is a
#: few images rather than a shift; but 1.6e-04 is 400x the same-node rebuild
#: floor and larger than the fp16 difference #3143 measured and rejected. Which
#: is why the size a cell was built at is now in its provenance sidecar
#: (``embed_batch_size``): changing an entry below silently redefines every cell
#: it is rebuilt into, and the sidecar is the only place that says so.
#: Deliberately three, not five. ``siglip`` is the shipped default and
#: ``siglip2_l`` the premium end; the middles (``siglip_l``, ``siglip2``) were
#: dropped because a study learns little from interpolating between them, and
#: the compute is better spent on more runs of the endpoints.
#:
#: The cost of that: ``siglip`` -> ``siglip2_l`` moves generation (1 -> 2) and
#: capacity (base -> SO400M) at the same time, so a difference between them
#: cannot be attributed to either alone. Rebuild a middle column if a result
#: ever needs that split -- ``build_pile.py --embedders siglip2`` restores one.
#: The two CLIP columns are **evaluation only** (#3292) and exist to test whether
#: #3287's `calibration_fraction` optimum follows single-vector geometry or just
#: the SigLIP lineage.  Both are run, not one, because a single CLIP arm cannot
#: separate the two things that change when you leave SigLIP:
#:
#:   `clip`   ViT-B/32, 512-d - the checkpoint the app already ships
#:   `clip_l` ViT-L/14, 768-d - dimension-matched to `siglip`, so a difference
#:                              cannot be "CLIP's vectors are narrower"
#:
#: Agreement between them is what licenses reading their verdict as CLIP's
#: lineage rather than CLIP's capacity.  Neither is selectable in the app
#: (`MediaEmbedder.eval_only`); `clip_l` is not a production candidate at all.
EMBEDDERS: dict[str, dict] = {
    "siglip": {"patch": False, "batch": 128},
    "siglip2_l": {"patch": False, "batch": 32},
    "clip": {"patch": False, "batch": 128},
    # ViT-L/14 at 224px: ~3x the base encoder's activation, so half the batch.
    "clip_l": {"patch": False, "batch": 64},
    # Patch embedders hold an (N, H, W, D) grid per image, not one vector, so
    # they carry far more activation memory per item than their backbone size
    # alone suggests.
    "dinov3_patch": {"patch": True, "gated": True, "batch": 64},
}


def embed_batch_size(embedder: str) -> int | None:
    """This embedder's ``VTSEARCH_EMBED_BATCH_SIZE``, or ``None`` for the default."""
    val = EMBEDDERS.get(embedder, {}).get("batch")
    return int(val) if val else None


def cells() -> list[tuple[str, str]]:
    """Every ``(dataset, embedder)`` cell in the full grid."""
    return [(ds, emb) for ds in DATASETS for emb in EMBEDDERS]


def pickle_name(dataset: str, embedder: str) -> str:
    return f"{dataset}__{embedder}.pkl"


def cell_path(dataset: str, embedder: str) -> Path:
    return EMBEDDINGS / pickle_name(dataset, embedder)


def provenance_path(dataset: str, embedder: str) -> Path:
    """Sidecar recording *which machine* produced this cell (#3160).

    Beside the pickle rather than inside it: a cell built before this existed
    stays loadable, and the sidecar can be read (or backfilled) without paying
    to unpickle a 900 MB file.
    """
    return EMBEDDINGS / f"{dataset}__{embedder}.provenance.json"


def is_patch_embedder(embedder: str) -> bool:
    return bool(EMBEDDERS.get(embedder, {}).get("patch"))


def region_capable(dataset: str, embedder: str) -> bool:
    """True when this *cell* can actually region-vote.

    Both halves are required: ground-truth boxes to drag (dataset) and a patch
    grid to pool them over (embedder). Stated per cell precisely because the
    per-dataset flag alone reads as "this arm region-votes" and does not.
    """
    return bool(DATASETS.get(dataset, {}).get("boxed")) and is_patch_embedder(embedder)


def require_demo_source(dataset: str) -> None:
    """Fail loudly if a demo dataset's source is not staged in the datadir.

    The demo downloaders treat a *missing* extraction dir as "not downloaded
    yet" and go fetch it. On a datadir that lost its symlink into the shared
    demo cache, that silently substitutes a partial re-download for the real
    dataset: the build still succeeds, but the cell holds a truncated subset
    and disagrees with its sibling cells. Cheaper to block than to detect.
    """
    name = DATASETS.get(dataset, {}).get("source_dir")
    if not name:
        return
    src = DATADIR / name
    if not src.exists():
        raise SystemExit(
            f"{dataset}: demo source {src} is missing, so the loader would re-download it.\n"
            f"  Link the shared cache in first, e.g.\n"
            f"    ln -s {DEMO_CACHE}/{name} {src}"
        )
    if not any(src.iterdir()):
        raise SystemExit(f"{dataset}: demo source {src} is empty (an empty dir reads as 'download complete')")


def setup_env() -> None:
    """Point vtscore + HF at the pile. Call before importing anything vtscore."""
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import _expcommon  # noqa: PLC0415

    # Default to the checkout this file lives in, rather than requiring VTS_REPO.
    # Depending on the env var is a live hazard: with it unset, ``import vtscore``
    # falls through to the venv's editable install, which points at the *main*
    # checkout -- 592 commits stale at the time of writing, and missing embedders
    # this pile uses. A build that resolved there would embed against different
    # code with no error. (This is how the shadow-module trap actually bites:
    # `VAR=x cmd1 && cmd2` applies VAR to cmd1 only, so the second command
    # silently ran against the wrong tree.)
    repo = os.environ.get("VTS_REPO") or str(Path(__file__).resolve().parents[3])
    os.environ["VTS_REPO"] = repo  # so calibration's common.py agrees with us
    # `_expcommon.setup_env` puts `repo` first on sys.path and drops the venv's
    # editable-install finder, so `import vtscore` resolves to this checkout
    # rather than whichever clone that install points at.
    _expcommon.setup_env(repo=repo, datadir=DATADIR, models_dir=MODELS, hf_home=MODELS)


# --------------------------------------------------------------------------
# #3588: the thirteen classes added to C, and the definition each was reviewed
# under. PROMOTED -- every name below now lives in the shipped tables, and this
# section is what the promotion was made of rather than a queue of pending work.
# --------------------------------------------------------------------------

#: The thirteen classes #3588 added to *C*, measured rather than proposed.
#:
#: Issue #3588 asks for the class list to sample *context exclusivity* on
#: purpose instead of by accident. Its own proposal does not survive the gate
#: it correctly specifies: of the classes it names, `airplane` (85 small),
#: `train` (34), `zebra` (27), `elephant` (20), `giraffe` (14), `cat` (44),
#: `suitcase` (54) and `potted plant` (1) all miss the 100-per-band floor, and
#: `motorcycle`, `surfboard`, `snowboard` and `skateboard` each carry a
#: measured alias partner (`bike` 0.38; `board` 0.45/0.50/0.52 --
#: `scan_name_overlap.py`). `traffic light` fails twice: 58 in the large band,
#: and its head noun `light` is already barred by :func:`scale_study_exclusion`.
#:
#: What survives is listed here. It is *not* the symmetric design the issue
#: asked for, and the asymmetry is structural rather than a sampling accident:
#: every scene-exclusive class the issue wanted (train, zebra, giraffe,
#: elephant) fails on the SMALL band specifically, because an animal or vehicle
#: that owns its scene is photographed filling the frame -- `giraffe` has 14
#: small-band images against 1,279 large. Context exclusivity and small-band
#: supply are anti-correlated in VG, so the easy end of the axis cannot be
#: widened with this source at this floor. These additions widen the hard end
#: and add same-scene partners; see the report for what that costs the design.
#:
#: **All thirteen are now in :data:`SCALE_CLASSES`** -- reviewed at 300 images
#: each, cleared, and promoted on the owner's 12-vs-13 call (#3604). The tuple
#: survives the promotion because the #3588-era analyses are conditioned on the
#: split: `shipped_pool_error.py` and the negative pass both ask *shipped versus
#: candidate*, and that question needs the two rosters to stay nameable after
#: they became one class list. Read it as "the thirteen #3588 added", never as
#: "classes still under consideration" -- nothing here is pending.
SCALE_CANDIDATES_3588: tuple[str, ...] = (
    # Tier A -- a habitat partner of a class already in C, so the negative pool
    # is shared and the contrast is same-scene, different-object.
    "truck",  # partner of `bus`      -- street, large vehicle
    "car",  # partner of `bus`/`truck` -- street, and the generic-clutter end
    "fork",  # partner of `knife`     -- table setting
    "spoon",  # partner of `knife`    -- table setting
    # Tier C -- objects whose surroundings ARE the negative pool.
    "cup",
    "bowl",
    "bottle",
    "vase",
    "bench",
    "chair",
    "sink",
    "cell phone",
    "fire hydrant",
)

#: VG names that denote a class NOT YET in *C*, held apart from the shipped read.
#:
#: **Empty since the #3588 promotion, and that is the table working, not rotting.**
#: The two tables cannot be merged because they are read at different times by
#: different code: :data:`SCALE_VG_NAMES` widens the ``vg_scale`` READ and is
#: folded by :func:`pilebuild.loaders.vg_scale.canonicalise` on every build, so an
#: entry there for a class outside *C* changes the built dataset on the strength
#: of a name nobody has reviewed. This one is read only by the slate makers, which
#: band a candidate without touching the pickle. When the candidate is promoted
#: its row moves across, **minus the class name itself** -- entries here list the
#: class name too, because the slate builder has no separate class-name read to
#: add it to, and forgetting to strip it on the way over would list `cup` as one
#: of `cup`'s own alternate spellings.
#:
#: The three rows that lived here -- `fire hydrant`/`hydrant`, `cell phone`/`phone`,
#: and the `cup` stemware merge -- are now in :data:`SCALE_VG_NAMES`, carrying
#: their measurements with them. Read a candidate's names through
#: :func:`scale_names_for`, never off this table directly: after a promotion the
#: row is in the other table, and a bare ``.get(c, (c,))`` silently answers "this
#: class has no alternate spellings" for a class that has seven.
SCALE_CANDIDATE_VG_NAMES: dict[str, tuple[str, ...]] = {}

#: Candidate-class spellings that MAY denote the class but also denote something
#: else -- :data:`SCALE_VG_AMBIGUOUS` for classes not yet in *C*.
#:
#: Same three-valued treatment, for the same reason: a box under one of these is
#: evidence in neither direction, so it is dropped from the bands *and* bars its
#: image from the shared negative pool. :func:`pilebuild.loaders.vg_scale.lift_ambiguous`
#: does both, and exempts any image COCO annotates exhaustively or a reviewer has
#: ruled on -- there the question is already answered and the spelling is ignored.
#:
#: Empty since the #3588 promotion; `glass`, the entry that named this table, is
#: now :data:`SCALE_VG_AMBIGUOUS`\ ``["cup"]``.
SCALE_CANDIDATE_VG_AMBIGUOUS: dict[str, tuple[str, ...]] = {}


def scale_names_for(cls: str) -> tuple[str, ...]:
    """Every VG spelling that IS *cls*, whichever side of promotion it sits on.

    The class name first, then its measured alternate spellings from the shipped
    table or the candidate one. Written because the slate makers read the
    candidate table with a ``(cls,)`` default, which answers "no alternate
    spellings" and "this class was promoted last week" identically -- and the
    second answer is wrong in the expensive direction, since a slate built
    without `hydrant` silently drops a third of `fire hydrant`'s boxes.
    """
    extra = SCALE_CANDIDATE_VG_NAMES.get(cls) or SCALE_VG_NAMES.get(cls) or ()
    return (cls, *(n for n in extra if n != cls))


def scale_ambiguous_for(cls: str) -> tuple[str, ...]:
    """*cls*'s ambiguous spellings, whichever side of promotion it sits on.

    The companion to :func:`scale_names_for`, and it matters for the same reason:
    `glass` moved tables at the promotion, and a slate that stopped withholding
    it would put windowpanes back among `cup`'s positives.
    """
    return SCALE_CANDIDATE_VG_AMBIGUOUS.get(cls) or SCALE_VG_AMBIGUOUS.get(cls) or ()


#: Classes this project defines as the UNION of several COCO classes.
#:
#: Distinct from every other table here, and the distinction is the whole point:
#: an alias merge (:data:`SCALE_VG_NAMES`) says two *names* denote one
#: object, which is a measurement. This says we are choosing a class boundary
#: COCO did not draw, which is a decision.
#:
#: It is available at all only because both halves are COCO classes. The scored
#: subset -- the fifth of each slate carrying a COCO answer, and the only reason
#: a reviewer's residual error is a number rather than a hope -- survives a union
#: of exhaustively annotated classes, since "COCO annotated a cup or a wine glass
#: here" is as well defined as either half. That is NOT true of a category COCO
#: lacks entirely, which is what the toy and fuel-tank rulings turn on; running
#: the two together is a mistake this file made for one commit.
#:
#: `cup` ∪ `wine glass` buys +8,180 boxes (+38%), +1,469 images, and **+35% in
#: the small band** -- the binding constraint on class supply everywhere here
#: (#3603). It costs a negative-pool redraw at build time for those 1,469, and
#: it makes `cup` the first class in this study that is not a plain COCO class.
SCALE_CLASS_MERGES: dict[str, tuple[str, ...]] = {
    # `car` U `truck`, owner ruling 2026-09-20 (#4056). See the roster comment.
    "enclosed road vehicle": ("car", "truck"),
    # `cup` U `wine glass`, owner ruling 2026-09-20. Declared here for months and
    # never applied (#4074): measured on the built cell, 0 of 300 `cup` positives
    # had been admitted on a wine glass, so the union its own docstring priced
    # was simply absent. Now applied, and the class renamed -- `cup` was never a
    # fair name for a set that is 27% plain glasses and 26% stemware.
    "single serving drinking vessel": ("cup", "wine glass"),
}


def coco_classes_for(cls: str) -> set[str]:
    """The COCO classes whose boxes count as *cls*, merges applied."""
    return set(SCALE_CLASS_MERGES.get(cls, (cls,)))


#: Rule names for classes that have LEFT *C*, frozen so the committed human
#: record stays readable.
#:
#: #3814 made a labelset carry the rule it was voted under, and the readers
#: REFUSE a file whose rule is not the one in force -- which is the right
#: behaviour, because a verdict cast under a different definition is not a
#: verdict about this class. But #4056 merged `car` + `truck` and `cup` +
#: `wine glass`, and `scripts/experiments/pile/human_record/` holds thousands of
#: vg_scale-era verdicts voted under the retired names. Without this table every
#: one of them fails the check at once, which reads as corruption rather than as
#: history.
#:
#: This is a record of what WAS, exactly like :data:`SCALE_CLASSES_ORIGINAL` and
#: :data:`SCALE_CLASSES_25`. Nothing here licenses voting under a retired name
#: again: :func:`review_name` still returns only the current rule.
SCALE_CLASS_RULES_RETIRED: dict[str, str] = {
    "car": "car incl SUVs and minivans",
    "truck": "truck incl vans not SUVs",
    "cup": "cup incl mugs glasses and stemware",
}


def rule_names_ever(cls: str) -> set[str]:
    """Every rule name *cls* has been voted under -- the current one, then retired."""
    names = {review_name(cls)}
    retired = SCALE_CLASS_RULES_RETIRED.get(cls)
    if retired:
        names.add(retired)
    return names


def scale_class_dataset_name(category: str) -> str:
    """Deprecated alias for :func:`review_name`.

    A thin spelling of :func:`review_name` with no pass suffix, kept because
    ``make_class_slate.py`` bands a *candidate* rather than issuing a voted
    slate and so has no pass to name.

    This used to read a second ``SCALE_CLASS_RULES`` of its own, declared later
    in this module and therefore shadowing the first: #3588 and #3612 each gave
    the same table the same name from opposite ends of one branch, and the
    survivor -- the string one -- left :func:`review_name` reading ``.name`` off
    a ``str``. Both rule sets now live in the one table above.
    """
    return review_name(category)
