"""Pre-registered grid for the embedding-precision study (issue #3143).

The question is narrow and the failure mode is expensive: fp16 makes the
`siglip2_l` forward 4.2x faster, but it *changes the vectors*, and the whole
pre-embedded pile plus every published result (#3129, the `vg_box` re-run) is
fp32.  A pile with some cells fp16 and some fp32 is a confound that would
surface months later as an unexplained arm difference, so this measures the
change on a side pile and touches nothing shared.

**The arm table is the design.**  Two things are varied, deliberately crossed:

* **precision** — the treatment (``fp32`` / ``fp16`` / ``autocast_fp16`` / ``bf16``).
* **GPU type** — the *control*.  Same code, same fp32 math on two different
  cards drifts by ~1e-7 through kernel selection alone (the claim #3144 leaned
  on, never measured here).  Without that arm, "fp16 moves cosines by 1e-3" has
  no denominator: the number to report is the ratio of the precision effect to
  the irreducible one, not the precision effect on its own.

That is also why ``fp16`` runs on **both** card types.  If half-precision drift
were card-specific, a single-GPU measurement would generalise no better than
#2877's single-environment acquisition result did.

Every arm gets its own pile root so cell filenames (``<dataset>__<embedder>.pkl``
carries no arm) cannot collide, and every arm writes a ``provenance.json``
recording the precision the process *actually* resolved, the card it ran on, and
the loaded model's parameter dtype.  Asserting the premise rather than the
parameter is the #2877/#2897/#2905 lesson: a knob that silently did nothing
looks exactly like a treatment with no effect.
"""

from __future__ import annotations

import os
from pathlib import Path

USER = os.environ.get("USER", "sgreenberg")

#: Where this study's side piles live.  Scratch, not ``/exp`` (50G quota), and
#: never the shared pile at ``/expscratch/$USER/vts-cache``.
STUDY = Path(os.environ.get("VTS_PRECISION_STUDY", f"/expscratch/{USER}/precision-3143"))

#: The shared pile: read-only here.  Its ``models`` dir is reused so no arm
#: re-downloads weights, and its cells are the published fp32 reference the
#: rebuild is checked against.
SHARED_PILE = Path(os.environ.get("VTS_PILE", f"/expscratch/{USER}/vts-cache"))
SHARED_MODELS = SHARED_PILE / "models"
SHARED_EMBEDDINGS = SHARED_PILE / "datadir" / "embeddings"

#: One dataset, on purpose.  ``visual_genome_m`` is the cheapest non-saturated
#: cell (4193 medias) and is what the issue names; the axis under test is
#: precision, and adding datasets would buy generality on an axis nobody
#: suspects while multiplying a run that has to stay cheap enough to redo.
DATASET = os.environ.get("VTS_PRECISION_DATASET", "visual_genome_m")

#: Both single-vector embedders, not just the expensive one.  ``siglip2_l`` is
#: where the speedup is (4.2x) and ``siglip`` is the **shipped default**, so it
#: is the one a precision flip would actually reach most users through.  Sizing
#: a decision on the premium arm alone is how #2877's -3 got over-fitted to one
#: environment.
EMBEDDERS = [e for e in os.environ.get("VTS_PRECISION_EMBEDDERS", "siglip,siglip2_l").split(",") if e]

#: ``dinov3_patch`` is deliberately absent, and not for cost reasons: its patch
#: grids are **already stored float16** (``vtscore/datasets/stages/embedding.py``
#: casts them), so its region path is quantised to half before any of this. That
#: makes it the arm where fp16 compute is least likely to matter and the one
#: where a null result would be least informative. It is also licence-gated
#: (needs ``HF_TOKEN``). Worth a follow-up, not a blocker for the default flip.
EXCLUDED_EMBEDDERS = {"dinov3_patch": "patch grids are already stored float16; needs HF_TOKEN"}

#: ``(precision, gpu_type)`` per arm.  ``fp32_l40s`` is the reference every
#: other arm is differenced against; ``fp32_v100`` is the noise floor.
ARMS: dict[str, dict[str, str]] = {
    "fp32_l40s": {
        "precision": "fp32",
        "gpu": "l40s",
        "role": "reference — the shipped default, and the base for every difference",
    },
    "fp32_v100": {
        "precision": "fp32",
        "gpu": "v100",
        "role": "control — same math, different card: the irreducible drift floor (#3144)",
    },
    "fp16_l40s": {
        "precision": "fp16",
        "gpu": "l40s",
        "role": "candidate — weight cast, the fast implementation",
    },
    "fp16_v100": {
        "precision": "fp16",
        "gpu": "v100",
        "role": "candidate on a second card — does the drift travel?",
    },
    "autocast_l40s": {
        "precision": "autocast_fp16",
        "gpu": "l40s",
        "role": "candidate — fp32 weights, per-op autocast: safer, slower",
    },
    "bf16_l40s": {
        "precision": "bf16",
        "gpu": "l40s",
        "role": "candidate — wider exponent; needs sm_80+, so no V100 twin",
    },
    # Added mid-run, to explain a result the first six arms produced rather than
    # to test half precision at all.  `fp32_v100` vs `fp32_l40s` — the same fp32
    # code on two cards — came out at median 1-cos 1.5e-4 on `siglip2_l` while
    # `siglip` agreed to 7.6e-13.  A 1e-4 disagreement between two *fp32* runs is
    # the size of the fp16 effect this study exists to measure, so it had to be
    # explained before any of it could be read.
    #
    # The suspect is TF32: `torch.backends.cudnn.allow_tf32` defaults to **True**,
    # so convolutions (SigLIP's patch embedding is a conv2d) run at 10 mantissa
    # bits on a TF32-capable card.  An L40S is sm_89 and has TF32; a V100 is
    # sm_70 and does not, so it runs true fp32.  That makes "fp32" mean two
    # different things across the two cards, which is not a precision effect and
    # not a kernel-selection rounding difference — it is a third format nobody
    # asked for.
    #
    # This arm is fp32 with TF32 turned OFF on the L40S.  If the hypothesis holds
    # it lands on `fp32_v100`, not on `fp32_l40s`.
    "fp32_notf32_l40s": {
        "precision": "fp32",
        "gpu": "l40s",
        "tf32": "off",
        "role": "diagnostic — fp32 with TF32 disabled; does it match the V100?",
    },
    # TF32 was REFUTED: `fp32_notf32_l40s` came out **bit-identical** to
    # `fp32_l40s` (median 1-cos exactly 0.0, mean 4e-17), so TF32 was never
    # active in the fp32 arms and cannot explain anything.  Recorded here rather
    # than deleted, because a refuted hypothesis with a measurement behind it is
    # what stops the report asserting a plausible wrong cause.
    #
    # Next suspect: **cuDNN algorithm selection**.  SigLIP's patch embedding is a
    # conv2d, and cuDNN picks an algorithm per (shape, card) from heuristics.
    # Winograd carries visibly more fp32 error than implicit-GEMM, so two cards
    # choosing differently for the SO400M/384 geometry would disagree at ~1e-4
    # while agreeing at ~1e-13 wherever they happen to pick the same one — which
    # is exactly the split observed (`siglip2_l` 1.5e-4, `siglip` 7.6e-13).
    #
    # `deterministic` restricts cuDNN to reproducible algorithms.  If the two
    # cards agree under it and disagree without it, algorithm selection is the
    # mechanism.
    "fp32_det_l40s": {
        "precision": "fp32",
        "gpu": "l40s",
        "deterministic": "on",
        "role": "diagnostic — fp32, cuDNN restricted to deterministic algorithms",
    },
    "fp32_det_v100": {
        "precision": "fp32",
        "gpu": "v100",
        "deterministic": "on",
        "role": "diagnostic — its V100 twin; do the two cards now agree?",
    },
    # cuDNN algorithm selection REFUTED too: both `det` arms came out
    # bit-identical to their own card's fp32 (exactly 0), so the 1.5e-4 split is
    # stable and reproducible, not a nondeterministic algorithm choice.
    #
    # Which leaves the sharpest remaining clue.  The published cell was built by
    # job 495266 on **rack7n03**; both V100 arms here landed on **rack5n03**.
    # SLURM calls both `gres/gpu:v100`, but that is a *type* label, not a device:
    # this cluster's V100s include SXM2 and PCIE parts with different SM counts,
    # and a different SM count means different kernel tiling and a different
    # accumulation order.  If `gres/gpu:v100` is not one device, then "the card"
    # was never the right axis — the node is.
    #
    # This arm pins the node the published cell was built on.  If it lands on the
    # L40S cluster, the pile is reproducible per *node*, and #3144's auto-pick —
    # which requests a type — cannot guarantee a reproducible rebuild.
    "fp32_v100_rack7n03": {
        "precision": "fp32",
        "gpu": "v100",
        "node": "rack7n03",
        "role": "diagnostic — the exact node the published cell was built on",
    },
}

#: The arm every drift figure is measured against.
REFERENCE_ARM = "fp32_l40s"

#: Arms whose *only* difference from the reference is the card.  Their drift is
#: the floor; a treatment arm is only interesting above it.
FLOOR_ARMS = ["fp32_v100"]

#: The margin the calibration studies resolve (see docs/experiments/overview-bench).
#: A precision change is adoptable only if the benchmark moves by less than this.
DECISION_MARGIN = float(os.environ.get("VTS_PRECISION_MARGIN", "0.005"))


def arm_pile(arm: str) -> Path:
    """This arm's pile root (its own, so cell filenames cannot collide)."""
    return STUDY / "piles" / arm


def arm_cell(arm: str, embedder: str, dataset: str | None = None) -> Path:
    return arm_pile(arm) / "datadir" / "embeddings" / f"{dataset or DATASET}__{embedder}.pkl"


def shared_cell(embedder: str, dataset: str | None = None) -> Path:
    """The published fp32 cell in the shared pile, for the reproduction check."""
    return SHARED_EMBEDDINGS / f"{dataset or DATASET}__{embedder}.pkl"


def provenance_path(arm: str) -> Path:
    return arm_pile(arm) / "provenance.json"


def results_dir() -> Path:
    return STUDY / "results"
