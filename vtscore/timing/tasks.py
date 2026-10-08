"""Canonical registry of long-running task families and their ordered steps.

Every task that drives a progress bar with a ``step``/``total_steps`` structure
registers here, and paces its bar with a weight vector, one entry per tracker
step, built from the terms declared below.

Adding a new long-running task means adding a :class:`TaskSpec` here, then
calling :func:`vtscore.timing.step_weights` at the task's entry point instead of
writing a literal vector.

**Phases vs tracker steps.** Usually they are the same thing and
``step_index`` is just ``(1, 2, 3, …)``. A task may model a step as several
cost *phases* that scale differently — ``dataset_load``'s step 1 covers both the
network transfer (scales with archive bytes) and the archive unpack (scales with
archive bytes at a very different rate) — in which case several phases share one
tracker step and :func:`vtscore.timing.step_weights` sums their terms back into
that step's slot.

**Default terms** reproduce the hand-tuned vectors these tasks shipped with.
They are *pseudo-seconds*: only their ratios are meaningful. (An
admin-measured per-environment profile could once replace them with real
seconds; it was retired in #4667, so these are now the only terms.) One vector
is no longer a transcription — ``dataset_stage``'s was re-derived from measured rows once its
step boundary was corrected (#3593); its comment below says from which.

**Per-media defaults.** A task whose split genuinely differs by media type may
carry :attr:`TaskSpec.media_default_terms`, an override vector per media type,
read with the same pseudo-second semantics. ``dataset_open`` is the one that
does: an audio pickle's read is a far larger share of an open than an image
pickle's (#4105). A media type without an override falls back to
:attr:`TaskSpec.default_terms`, so an unmeasured media type paces exactly as it
did before any override existed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskSpec:
    """Declares one long-running task family's step structure.

    Attributes:
        name: Stable identifier, the key callers pass to
            :func:`vtscore.timing.step_weights`.
        steps: Ordered cost-phase names. Profile coefficients are keyed by these.
        step_index: 1-based tracker step each phase reports against, parallel to
            ``steps``. Several phases may share one step (see module docstring).
        tracker_steps: How many step numbers the task reports — the length of
            the weight vector ``set_step_weights`` expects.
        scale: Human description of what the ``n`` scale variable counts.
        default_terms: Shipped fallback pseudo-seconds, parallel to ``steps``.
            Empty means "this task has its own richer default model" — only
            ``dataset_load``, whose calibrated table lives in
            :mod:`vtscore.datasets.stages._load_cost_model`.
        media_default_terms: Per-media-type overrides of ``default_terms``,
            keyed by media type id (``"audio"``, ``"image"``, …), each parallel
            to ``steps``, consulted only for the media types it names; every
            other media type keeps
            ``default_terms``. Read it through :meth:`defaults_for`.
        byte_scaled: Steps whose cost tracks downloaded **bytes** rather than
            item count. Descriptive only since the fitter that priced them as a
            per-MB rate was retired (#4667).
        loads_encoder: Whether a run of this task can pay a **cold encoder
            load** (the first run in a process to need a given
            ``(media_type, embedder)`` downloads and instantiates the model).
            Descriptive only since the recorder that kept a residency ledger
            on it was retired (#4667).
    """

    name: str
    steps: tuple[str, ...]
    step_index: tuple[int, ...]
    tracker_steps: int
    scale: str
    default_terms: tuple[float, ...] = ()
    byte_scaled: tuple[str, ...] = ()
    loads_encoder: bool = False
    # ``hash=False`` keeps the spec hashable, as it was before this field
    # existed: a mapping cannot be hashed, and the flat fields already
    # identify a spec.
    media_default_terms: Mapping[str, tuple[float, ...]] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if len(self.step_index) != len(self.steps):
            raise ValueError(f"{self.name}: step_index must be parallel to steps")
        if self.default_terms and len(self.default_terms) != len(self.steps):
            raise ValueError(f"{self.name}: default_terms must be parallel to steps")
        if self.media_default_terms and not self.default_terms:
            # An override needs something to override: a task with no flat
            # default has its own richer model (``dataset_load``), and a
            # per-media vector beside it would silently replace that model for
            # one media type only.
            raise ValueError(f"{self.name}: media_default_terms requires default_terms")
        for media_type, terms in self.media_default_terms.items():
            if len(terms) != len(self.steps):
                raise ValueError(f"{self.name}: media_default_terms[{media_type!r}] must be parallel to steps")

    def defaults_for(self, media_type: str = "") -> tuple[float, ...]:
        """The shipped fallback terms for *media_type*.

        Its entry in :attr:`media_default_terms` when it has one, otherwise
        :attr:`default_terms` — so an empty or unrecognised media type gets the
        task-wide vector, never nothing.
        """
        return self.media_default_terms.get(media_type, self.default_terms)


def _linear(
    name: str,
    steps: tuple[str, ...],
    scale: str,
    terms: tuple[float, ...],
    *,
    loads_encoder: bool = False,
    media_terms: Mapping[str, tuple[float, ...]] | None = None,
) -> TaskSpec:
    """Build a spec whose phases map 1:1 onto tracker steps (the common case)."""
    return TaskSpec(
        name=name,
        steps=steps,
        step_index=tuple(range(1, len(steps) + 1)),
        tracker_steps=len(steps),
        scale=scale,
        default_terms=terms,
        loads_encoder=loads_encoder,
        media_default_terms=dict(media_terms or {}),
    )


#: Every registered task family, keyed by :attr:`TaskSpec.name`.
#:
#: The default terms below are transcribed from the literal vectors these tasks
#: carried before they were centralised here; each site's original reasoning
#: is preserved in the comments here rather than in six scattered constants.
TASKS: dict[str, TaskSpec] = {
    # Importing a dataset: acquire the source, read/convert it into medias,
    # embed every item, then dedup + coverage-atlas + registry save. Deliberately
    # carries no default terms — its shipped model is the measured affine table
    # in ``_load_cost_model``, which is already ``n``-aware per (device, media,
    # embedder) and far better than any flat vector could be here.
    "dataset_load": TaskSpec(
        name="dataset_load",
        steps=("download", "extract", "load", "embed", "finalize"),
        step_index=(1, 1, 2, 3, 4),
        tracker_steps=4,
        scale="media items embedded",
        byte_scaled=("download", "extract"),
        loads_encoder=True,
    ),
    # Re-opening an already-imported dataset from its pkl. Step 1 (pickle read +
    # convert + the near-instant exact-dedup) is seconds at most; step 2 is the
    # coverage atlas, ~10 ms when the cached atlas restores and a hierarchical
    # k-means rebuild when it does not.
    #
    # That rebuild is linear in n and measured, not extrapolated, to 36 500
    # items: on a V100 with cuML active it costs 0.0027 s/item for image
    # (r^2 0.999 over n = 412..36 497; 1.3 s, 13 s at 5110, 26 s at 10 220,
    # 100 s at 36 497) and 0.0025 s/item for audio (n = 1960..8732). A rebuild
    # is therefore seconds below ~20 000 items and reaches minutes only near
    # COVERAGE_ATLAS_AUTO_THRESHOLD (50 000), where the fit gives ~140 s
    # (docs/experiments/2026-09-22-atlas-rebuild-3595/REPORT.md, #3595).
    #
    # The 0.85 below is checked for image: a rebuilding open spends 0.81-0.94
    # of its time in the coverage step at every n from 838 to 36 497. Audio
    # gets its own vector because reading and converting an audio pickle is a
    # much larger part of the open (3.5-16 s against a 6-22 s rebuild), so its
    # measured rebuild share is 0.52-0.63 (n = 1960..8732, mean 0.58). At 0.85
    # an audio open's bar crawled through the read, reaching 15 % after ~40 %
    # of the wait, then raced through the atlas (#4105). Its 0.60 sits a
    # little above that mean on purpose: every row is V100 + cuML, and on a
    # CPU host sklearn's k-means should make the rebuild heavier while the
    # read, which no GPU was speeding up, stays about the same. Video and text
    # are unmeasured and keep the task-wide vector. On a restore the share is
    # <= 0.01 for both media; a flat default cannot tell the two apart, so a
    # restoring open's bar jumps across the coverage slice when the atlas
    # restores.
    "dataset_open": _linear(
        "dataset_open",
        ("items", "coverage"),
        "media items in the pkl",
        (0.15, 0.85),
        media_terms={"audio": (0.40, 0.60)},
    ),
    # Promoting a staged subset into a real dataset. The atlas's hierarchical
    # k-means dominates, then embedding serialization; the registry write is
    # trivial.
    "dataset_promote": _linear(
        "dataset_promote",
        ("coverage", "serialize", "registry"),
        "media items in the promoted subset",
        (6.0, 3.5, 0.5),
    ),
    # Staging an import: run the importer, embed what it produced, serialize the
    # result to a staging pkl. Deliberately *not* modelled as a ``dataset_load``:
    # staging stops before dedup, the coverage atlas, and the registry write (a
    # later promote pays those), so folding its runs into the load fit would
    # teach that finalize is free. No byte-scaled phases here — the staging path
    # is never told an archive size, so a per-MB rate would have nothing to
    # divide by.
    #
    # These are the one set of default terms not transcribed from a pre-profile
    # hand-tuned vector. The shipped ``(0.30, 0.60, 0.10)`` budgeted 60 % of the
    # bar to a step that measured 0.000–0.002 s on every run ever recorded,
    # because the importer's embedding was landing under ``acquire``
    # (``_STAGE_STATUS_TO_STEP`` in ``vtscore/datasets/load_pipeline.py`` is the
    # fix). Re-derived from #3521 §5's image rows, reading the old ``acquire``
    # slope as the embed it actually was: embed ``0.0136 s/item``, serialize
    # ``~0.0042 s/item`` — 76:24, which is the 0.72:0.23 below. Acquire measured
    # near zero there (a demo's acquisition is a cached local read, and its fresh
    # rows leave no residual once the embed line is subtracted); it gets 0.05
    # rather than 0.02 because the only importer these rows cover is the demo
    # one, and a server-folder or upload import spends real I/O in that step.
    "dataset_stage": _linear(
        "dataset_stage",
        ("acquire", "embed", "serialize"),
        "media items staged",
        (0.05, 0.72, 0.23),
        loads_encoder=True,
    ),
    # Loading a saved detector: read its labelset, pull the label examples back
    # into the active dataset, retrain the MLP. Training dominates; the other
    # two are quick I/O.
    "detector_load": _linear(
        "detector_load",
        ("restore_labels", "seed_examples", "train"),
        "labels in the detector's labelset",
        (0.15, 0.15, 0.70),
        loads_encoder=True,
    ),
    # Text search: load the embedder, embed the one-line query, score every
    # media by cosine similarity. The model load dominates on a cold start
    # (seconds to pull CLAP / SigLIP weights); embedding one short query is
    # trivial; scoring scales with the dataset but is vectorised.
    "text_sort": _linear(
        "text_sort",
        ("load_model", "embed_query", "score"),
        "medias scored",
        (0.75, 0.05, 0.20),
        loads_encoder=True,
    ),
    # Running saved detectors across saved datasets. Scoring dominates; loading
    # datasets from pkl is moderate; preparing detector configs is quick.
    "find": _linear(
        "find",
        ("prepare", "load", "score"),
        "medias scored across all selected datasets",
        (0.10, 0.30, 0.60),
        loads_encoder=True,
    ),
    # Train-and-score against the active dataset: resolve the detector, train
    # its MLP, score every media, apply the resulting labels. Train + score
    # carry the cost; resolve/apply are quick.
    "train_and_score": _linear(
        "train_and_score",
        ("resolve", "train", "score", "apply"),
        "medias scored",
        (0.10, 0.45, 0.40, 0.05),
        loads_encoder=True,
    ),
}


#: Branch names meaning "this run took the step's **cheap** path" — a cached
#: artefact stood in for work a first run has to do. Deprecated (#4667): the
#: recorder that stamped them and the fitter that read them are retired, and
#: nothing in this repository consults either set.
#:
#: ``cached``    a demo import satisfied itself from the embeddings pkl, so it
#:               downloaded nothing, embedded nothing, and loaded no encoder.
#: ``restored``  a dataset open adopted the coverage atlas cached in its pickle
#:               instead of rebuilding the hierarchical k-means.
#: ``deferred``  a dataset open past ``COVERAGE_ATLAS_AUTO_THRESHOLD`` skipped
#:               the atlas entirely, leaving it to the on-demand endpoint.
CHEAP_BRANCHES = frozenset({"cached", "restored", "deferred"})

#: Branch names meaning "this run did the work" — the branch somebody waits on.
#: Deprecated with :data:`CHEAP_BRANCHES` (#4667).
#:
#: ``fresh``     the import really downloaded, embedded, and finalised.
#: ``rebuilt``   the coverage atlas was built from scratch.
DEAR_BRANCHES = frozenset({"fresh", "rebuilt"})


def task_spec(task: str) -> TaskSpec | None:
    """Return the :class:`TaskSpec` for *task*, or ``None`` if unregistered.

    Unregistered is not an error: a caller that passes an unknown task simply
    gets no weights and keeps whatever fallback it supplied.
    """
    return TASKS.get(task)
