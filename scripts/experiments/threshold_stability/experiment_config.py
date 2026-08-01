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

#: Each arm: (name, threshold_rule, threshold_smooth, calibrate_count). Conformal is
#: the only calibration rule now (argmin retired), so arms are named by what they
#: vary: fold count (``k2`` = the app default of calibrate_count=2, ``k8`` = 8),
#: temporal smoothing (``k2-med3``), and the S3 fold->final scale probe
#: (``rank-transfer``, evaluated in Stage-A replay). ``k2`` is the baseline.
ARMS: list[tuple[str, str, str, int]] = [
    ("k2", "conformal", "none", 2),
    ("k8", "conformal", "none", 8),
    ("k2-med3", "conformal", "med3", 2),
    ("rank-transfer", "rank-transfer", "none", 2),
]

#: The baseline arm whose recorded --labeling-trace Stage A replays (frozen votes).
BASELINE_ARM = "k2"

#: Stage A replay depth (fold-split seeds × trainer seeds). The plan's 10×10 is
#: 100 refits per (step, rule); default to a lighter 5×3 that still resolves the
#: split-vs-fit variance split and finishes overnight (override up for the final).
REPLAY_FOLD_SEEDS = int(os.environ.get("THRSTAB_REPLAY_FOLD_SEEDS", "5"))
REPLAY_TRAINER_SEEDS = int(os.environ.get("THRSTAB_REPLAY_TRAINER_SEEDS", "3"))


def class_slug(cls: str) -> str:
    """Filesystem-safe class slug **identical to** ``scripts/sod/features.slugify``.

    The exemplar cache dir and the labeling-trace path are keyed on this exact
    slug (e.g. ``stop sign`` -> ``stop-sign``); a mismatch means the replay tool
    can't find a class's exemplars. Replicated here (rather than imported) so the
    config stays importable without the sod package on the path.
    """
    return re.sub(r"[^a-z0-9]+", "-", cls.strip().lower()).strip("-")


def array_cells() -> list[dict]:
    """Enumerate SLURM-array cells — **one per class**.

    Each cell runs every arm once with ``--iterations len(SEEDS)`` (so all seeds
    for an arm come from a single sweep, sharing the cache) and replays each
    seed's baseline trace. One-per-class (not per (class, seed)) avoids re-running
    the low seeds that ``--iterations N`` = seeds ``0..N-1`` would duplicate."""
    return [{"cls": cls} for cls in CLASSES]
