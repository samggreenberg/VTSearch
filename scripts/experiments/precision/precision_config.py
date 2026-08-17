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
