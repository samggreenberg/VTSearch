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

import os
from pathlib import Path

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
    # Box-size-banded VG, drawn from the WHOLE source (all 108k images, full
    # free-text vocabulary) rather than the demo pipeline's 100 curated
    # categories on a 4% slice.  The `_s`/`_m`/`_l` on `visual_genome_*` is a
    # dataset *size* tier and says nothing about boxes; these are the box bands.
    "vg_box_small": {"boxed": True, "kind": "vg_band", "band": "small"},
    "vg_box_medium": {"boxed": True, "kind": "vg_band", "band": "medium"},
    "vg_box_large": {"boxed": True, "kind": "vg_band", "band": "large"},
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
    "vg_scale": {"boxed": True, "kind": "vg_scale", "labels": "coco"},
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
    "vg_scale_any": {"boxed": True, "kind": "vg_scale_any"},
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
    "vg_scale_deep": {"boxed": True, "kind": "vg_scale_deep", "on_request": True},
}

#: Box-size bands, as a fraction of image area, anchored to the patch
#: embedder's geometry (the same anchors the calibration harness bands on):
#: one DINOv3 patch is 1/196 of the image and the smallest HAC leaf is 1/12.
#: ``small`` is therefore "below what the patch grid can resolve at all".
#: The upper cut mirrors ``MAX_VOTED_AREA``: a box covering >80% of the image
#: is not a region, it is the image.
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
SCALE_CLASSES: tuple[str, ...] = (
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
SCALE_N_NEG = int(os.environ.get("VTS_SCALE_N_NEG", "3900"))
#: Extra negatives drawn into the pickle but designated into no cell. A human
#: verdict can retire a contaminated negative later; re-designating from a spare
#: is a relabel, while drawing a fresh one would mean re-embedding every cell.
SCALE_N_NEG_SPARE = int(os.environ.get("VTS_SCALE_N_NEG_SPARE", "300"))

#: The designed prevalence of a `vg_scale` cell: 100 positives per band x 3
#: bands against 3900 shared negatives. Named because `vg_scale_deep` has to
#: REPRODUCE it rather than re-derive it, and because every k* this family of
#: studies quotes is computed from it (`-log2((1-pi)/pi) = -3.71`).
SCALE_PREVALENCE = (3 * SCALE_N_POS) / (3 * SCALE_N_POS + SCALE_N_NEG)

#: `vg_scale_deep`'s positives per class (#3547). 900 is the deepest value all
#: twelve classes support band-free -- `stop sign`, the thinnest, has 1006
#: candidates (`measure_supply.py`) -- and it is chosen against `preflight.sh`
#: check 16b, which clears only when the sim half holds MORE positives than the
#: horizon has steps: at `SIM_FRACTION` 0.5 that is 450 against 400.
#:
#: Going deeper costs classes, not money: 1200 drops `kite` and `stop sign`,
#: and a class list that differs from #3319's would confound the horizon axis
#: with a vocabulary axis in the one comparison this dataset exists to make.
SCALE_DEEP_N_POS = int(os.environ.get("VTS_SCALE_DEEP_N_POS", "900"))
#: DERIVED, never set: see the `vg_scale_deep` note in DATASETS. A negative pool
#: written as a literal beside a positive count is how prevalence drifts.
SCALE_DEEP_N_NEG = round(SCALE_DEEP_N_POS * (1 - SCALE_PREVALENCE) / SCALE_PREVALENCE)
SCALE_DEEP_N_NEG_SPARE = int(os.environ.get("VTS_SCALE_DEEP_N_NEG_SPARE", "300"))


#: How far a VG copy's aspect ratio may drift from the COCO original before its
#: boxes are considered untransferable. Normalised coordinates survive a rescale
#: but not a re-crop or a rotation, and 49 of the 51,497 overlaps are one of
#: those -- small enough to ignore by accident, which is why it is a constant
#: with a check rather than an assumption.
MAX_ASPECT_DRIFT = float(os.environ.get("VTS_MAX_ASPECT_DRIFT", "0.01"))


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
#: Batch size does not change what is embedded: in fp32 it shifts vectors by
#: ~1e-7 through kernel selection, orders of magnitude below anything the
#: studies resolve.
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
# #3588: candidate additions to C, and the definition each is reviewed under
# --------------------------------------------------------------------------

#: Candidates for an expanded *C*, measured rather than proposed.
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

#: VG names that denote one of the candidates. VG's vocabulary is free text and
#: :func:`pilebuild.vgsource.vg_boxes_by_name` matches the PRIMARY name only, so
#: a class built from one spelling silently drops the others.
#:
#: Only merges measured as aliases are listed. `fire hydrant` / `hydrant` is one
#: object under two spellings (box IoU 0.77/0.74, `scan_name_overlap.py`), and
#: `hydrant` accounts for 266 of the 835 COCO `fire hydrant` boxes on the
#: overlap -- taking `fire hydrant` alone would throw away a third of the class.
#: `phone` is listed for `cell phone` on the same evidence (541 boxes) **and is
#: the single riskiest entry here**: see `SCALE_CLASS_RULES`.
SCALE_VG_NAMES: dict[str, tuple[str, ...]] = {
    "fire hydrant": ("fire hydrant", "hydrant"),
    "cell phone": ("cell phone", "phone", "cellphone"),
}

#: The definition each class is reviewed under, and the name that definition
#: travels on.
#:
#: **This is the `book` failure made structural.** COCO has no magazine class,
#: so COCO's annotators put magazines in `book`; the human pass applied the
#: narrower English reading; and the class became two definitions wearing one
#: name -- 21 verdicts on one, 49 on the other, with every structural check
#: passing (`make_definition_reslate.py`). A reviewer cannot see a manifest
#: while voting, so the rule has to travel on the one string the app shows:
#: the dataset and detector name.
#:
#: The rules are **measured, not drafted**. `coco_folds.py` asks which VG names
#: land on a COCO class's boxes over the ~51k-image overlap, which enumerates
#: the boundary cases before a human meets one -- run against `book` it prints
#: `magazine` (79) and `magazines` (30). Each entry below names the boundary
#: case that measurement found, and the long form is in the annotation guide.
#:
#: Keyed by class; the value is the dataset/detector name. Deliberately a
#: literal rather than a formatting of the class name: the whole point is that
#: it says something the class name does not.
SCALE_CLASS_RULES: dict[str, str] = {
    "truck": "truck incl vans not SUVs",
    "car": "car incl SUVs and minivans",
    "fork": "fork incl plastic",
    "spoon": "spoon incl plastic not spatulas",
    "cup": "cup incl mugs and glasses not stemware",
    "bowl": "bowl incl plates and dishes",
    "bottle": "bottle incl jars",
    "vase": "vase incl pots and planters",
    "bench": "bench not chairs",
    "chair": "chair incl stools not couches",
    "sink": "sink basin not counter",
    "cell phone": "cell phone not landlines",
    "fire hydrant": "fire hydrant not standpipes",
}


def scale_class_dataset_name(category: str) -> str:
    """The dataset/detector name *category* is reviewed under.

    Falls back to the bare class name, which is correct for a class with no
    boundary case worth stating -- and wrong to invent one for, since a rule
    nobody needs is a rule a reviewer will misread.
    """
    return SCALE_CLASS_RULES.get(category, category)
