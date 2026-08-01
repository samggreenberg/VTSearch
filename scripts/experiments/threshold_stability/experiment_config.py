"""Pre-registered grid for the threshold-stability study (issue #2790).

One place the runner and the analyzer agree on. See
``docs/plans/threshold-stability-experiment.md`` (on ``dev``) for the design and
``scripts/sod/THRESHOLD_STABILITY_STATUS.md`` for the plan-vs-reality reframe (the
sweep's default ``argmin`` rule is the *infidelity*; the ``conformal`` rule is what
production Autopilot actually runs).

Six arms A/B the whole/box-pool path's threshold rule + smoothing + fold count, on
the #2790 config (COCO, SigLIP 2, ``whole``, ``--neg-multiple 100``,
``--min-box-frac 0.03``). Each arm is a set of ``sweep.py`` flags; a cell is one
``(class, seed)`` and runs **every arm** inside it, sharing the embedding cache so
the arms are paired on identical data and the same startup exemplar. CPU-only —
the embeddings come from the #2790 cache, and the MLPs are tiny.
"""

from __future__ import annotations

import os
import re

# --- Fixed #2790 repro config (pre-registered) ---
DATASET = os.environ.get("THRSTAB_DATASET", "coco")
EMBEDDER = os.environ.get("THRSTAB_EMBEDDER", "siglip2")  # user OK'd SigLIP 1 <-> 2 either way
PROPOSAL = "whole"
MAX_LABELS = int(os.environ.get("THRSTAB_MAX_LABELS", "60"))
NEG_MULTIPLE = int(os.environ.get("THRSTAB_NEG_MULTIPLE", "100"))
MIN_BOX_FRAC = os.environ.get("THRSTAB_MIN_BOX_FRAC", "0.03")
INCLUSION = int(os.environ.get("THRSTAB_INCLUSION", "0"))

# --- Sizing (variance is the measurand, so more seeds than the plots' 3) ---
CLASSES = os.environ.get("THRSTAB_CLASSES", "stop sign,traffic light,fire hydrant,parking meter,bus").split(",")
SEEDS = list(range(int(os.environ.get("THRSTAB_N_SEEDS", "10"))))

#: Each arm: (name, threshold_rule, threshold_smooth, calibrate_count). ``argmin-k2``
#: is the status quo baseline whose trace Stage A replays; the rest isolate one
#: factor each (rule = S1, fold count = S2, med3 = temporal hysteresis, and
#: rank-transfer = the S3 fold->final scale probe, applied by the replay tool).
ARMS: list[tuple[str, str, str, int]] = [
    ("argmin-k2", "argmin", "none", 2),
    ("argmin-k8", "argmin", "none", 8),
    ("conformal-k2", "conformal", "none", 2),
    ("conformal-k8", "conformal", "none", 8),
    ("conformal-k2-med3", "conformal", "med3", 2),
    ("rank-transfer-k2", "rank-transfer", "none", 2),
]

#: The baseline arm whose recorded --labeling-trace Stage A replays (frozen votes).
BASELINE_ARM = "argmin-k2"

#: Stage A replay depth (fold-split seeds × trainer seeds), per the plan.
REPLAY_FOLD_SEEDS = int(os.environ.get("THRSTAB_REPLAY_FOLD_SEEDS", "10"))
REPLAY_TRAINER_SEEDS = int(os.environ.get("THRSTAB_REPLAY_TRAINER_SEEDS", "10"))


def class_slug(cls: str) -> str:
    """Filesystem-safe class slug **identical to** ``scripts/sod/features.slugify``.

    The exemplar cache dir and the labeling-trace path are keyed on this exact
    slug (e.g. ``stop sign`` -> ``stop-sign``); a mismatch means the replay tool
    can't find a class's exemplars. Replicated here (rather than imported) so the
    config stays importable without the sod package on the path.
    """
    return re.sub(r"[^a-z0-9]+", "-", cls.strip().lower()).strip("-")


def array_cells() -> list[dict]:
    """Enumerate ``(class, seed)`` cells for the SLURM array (stable order)."""
    return [{"cls": cls, "seed": seed} for cls in CLASSES for seed in SEEDS]
