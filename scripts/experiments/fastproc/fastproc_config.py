"""Pre-registered grid for the image-processor study (issue #3146).

The issue's premise is that every image embedder is on the slow PIL/numpy
resize path because none of them passes ``use_fast``, and that flipping to
``use_fast=True`` is the win.  **Measured against the installed transformers
(5.12.1) the premise is false**, and the arm table is built around what is
actually true instead:

* transformers v5 **removed the ``Fast`` suffix**.  ``SiglipImageProcessor`` is
  now the torchvision implementation; the PIL one was renamed
  ``SiglipImageProcessorPil``.  Passing nothing already selects torchvision.
* ``use_fast`` is itself deprecated — the current spelling is ``backend=``, and
  ``use_fast=True`` on an explicitly-named concrete class is a no-op.
* So the flip #3146 proposes has *already happened*, silently, through a
  dependency range (`transformers>=4.49`) that spans the version where the
  default changed.

That leaves two questions worth a study, and they are not the one the issue
asked.  **The perf question** is the issue's *other* proposed fix — moving
resize/normalise onto the GPU, which the torchvision backend supports through a
``device=`` call kwarg.  **The reproducibility question** is the one the
premise-check surfaced: the PIL path is still reachable (an older transformers,
an explicit ``backend="pil"``), it produces *different pixels*, and the entire
pre-embedded pile is one backend's output with nothing recording which.

Arms are therefore backend x device, all on **one pinned node**.  #3160
established that ``gres/gpu:v100`` is two different devices and that a card
swap alone moves ``siglip2_l`` fp32 by 1.5e-4 — the size of the effect a study
like this exists to measure.  Pinning the node is cheaper than measuring around
it, and ``tv_cpu_rep`` is the arm that proves the pinning worked: it is the
reference arm run a second time, so anything it drifts by is the floor and
every other number is read against it.

``published_pile`` is not an arm here but it is the adjudicator.  If the
reference rebuild reproduces the shared pile's cell, the pile is torchvision-
built and the premise-check is confirmed end to end rather than by reading a
class name.
"""

from __future__ import annotations

import os
from pathlib import Path

USER = os.environ.get("USER", "sgreenberg")

#: This study's side piles.  Scratch, never the shared pile.
STUDY = Path(os.environ.get("VTS_FASTPROC_STUDY", f"/expscratch/{USER}/fastproc-3146"))

#: The shared pile: read-only here.  Its weights are reused so no arm
#: re-downloads, and its cells are what the reference rebuild is checked against.
SHARED_PILE = Path(os.environ.get("VTS_PILE", f"/expscratch/{USER}/vts-cache"))
SHARED_MODELS = SHARED_PILE / "models"
SHARED_EMBEDDINGS = SHARED_PILE / "datadir" / "embeddings"

#: One dataset, matching #3143 so the two studies' drift numbers are comparable
#: on the same medias.  ``visual_genome_m`` is 4193 medias of real photographs at
#: mixed sizes, which is what makes it the right corpus for a *resize* study:
#: every image is resampled by a different factor.
DATASET = os.environ.get("VTS_FASTPROC_DATASET", "visual_genome_m")

#: ``siglip`` is the shipped default and ``siglip2_l`` is where the processor
#: share is largest (68% of wall clock once #3143/#3145 land, per the issue).
EMBEDDERS = [e for e in os.environ.get("VTS_FASTPROC_EMBEDDERS", "siglip,siglip2_l").split(",") if e]

#: ``dinov3_patch`` is excluded, and the reason is a measurement rather than a
#: budget: **transformers ships no PIL implementation for DINOv3**.  Asked for
#: ``backend="pil"`` it warns "Requested pil backend is not available" and hands
#: back torchvision, so a `pil` arm there would be the reference arm wearing a
#: different label — the exact unasserted-premise failure of #2877.  Its
#: ``device="cuda"`` arm is a real follow-up; its `pil` arm cannot exist.
EXCLUDED_EMBEDDERS = {
    "dinov3_patch": "transformers has no PIL backend for DINOv3; a pil arm would silently be the tv arm",
}

#: One node for every arm.  Not a GPU *type* — a node.  See the module docstring.
PIN_NODE = os.environ.get("VTS_FASTPROC_NODE", "rack4n01")
PIN_GPU = os.environ.get("VTS_FASTPROC_GPU", "l40s")

#: ``(backend, device)`` per arm.
ARMS: dict[str, dict[str, str]] = {
    "tv_cpu": {
        "backend": "torchvision",
        "device": "cpu",
        "role": "reference — what ships today, named explicitly instead of resolved",
    },
    "tv_cpu_rep": {
        "backend": "torchvision",
        "device": "cpu",
        "role": "floor — the reference arm run twice on the same node; its drift is the noise",
    },
    "pil_cpu": {
        "backend": "pil",
        "device": "cpu",
        "role": "the path #3146 believed was shipped, and what transformers<5 resolves to",
    },
    "tv_cuda": {
        "backend": "torchvision",
        "device": "cuda",
        "role": "candidate — resize/normalise on the GPU, the issue's other proposed fix",
    },
}

#: Every difference is taken against the shipped path, not against PIL.  A
#: candidate is adoptable or not relative to what users have today; measuring
#: against PIL would quote a drift nobody is currently exposed to.
REFERENCE_ARM = "tv_cpu"

#: Arms running the *same* code as the reference.  One, deliberately: with the
#: node pinned there is only one way for the reference to differ from itself.
FLOOR_ARMS = ["tv_cpu_rep"]

#: The margin the calibration studies resolve (docs/experiments/overview-bench).
#: Same value #3143 adopted; a processor change is adoptable only if the
#: benchmark moves by less than this.
DECISION_MARGIN = float(os.environ.get("VTS_FASTPROC_MARGIN", "0.005"))


def arm_pile(arm: str) -> Path:
    """This arm's pile root (its own — ``<dataset>__<embedder>.pkl`` carries no arm)."""
    return STUDY / "piles" / arm


def arm_cell(arm: str, embedder: str, dataset: str | None = None) -> Path:
    return arm_pile(arm) / "datadir" / "embeddings" / f"{dataset or DATASET}__{embedder}.pkl"


def shared_cell(embedder: str, dataset: str | None = None) -> Path:
    """The published cell in the shared pile — the adjudicator, not an arm."""
    return SHARED_EMBEDDINGS / f"{dataset or DATASET}__{embedder}.pkl"


def provenance_path(arm: str) -> Path:
    return arm_pile(arm) / "provenance.json"


def results_dir() -> Path:
    return STUDY / "results"
