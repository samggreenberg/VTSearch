"""``DetectorContext``: all mutable state that belongs to one loaded detector."""

from __future__ import annotations

from typing import Any

from vtscore.config.runtime import DEFAULT_BETA


class DetectorContext:
    """All mutable state that belongs to a single loaded detector.

    Bundles per-detector vote state, training artifacts, and cached in-memory
    data (head, threshold, training media with embeddings).  Multiple detectors
    can be loaded simultaneously; one is "active" (feeding the labeling UI).
    """

    __slots__ = (
        "detector_id",
        "name",
        "media_type",
        # The concrete space the label cache / head is *currently* built in (an
        # adaptive marker, re-stamped when the active dataset's space changes).
        # Distinct from ``embedder_type``: it names whichever concrete embedder
        # of the detector's type the active dataset supplied, so the
        # cache-invalidation compare stays honest across same-type swaps.
        "embedder",
        # The detector's *locked* embedder type - ``"semantic"`` /
        # ``"patch_semantic"`` / ``"structural"`` - persisted on the detector
        # JSON and loaded here at detector load.  Immutable in memory (never
        # re-stamped).  The detector scores in whatever concrete embedder of
        # this type the active dataset binds.  Empty for a legacy detector,
        # where routing falls back to the dataset score precedence.  See
        # ``docs/ARCHITECTURE.md`` → "Per-detector embedder type".
        "embedder_type",
        # Vote state
        "good_votes",
        "bad_votes",
        "label_history",
        # The votes in force before ``label_history[0]``, as ``(good ids, bad
        # ids)``: the labels restored silently from the detector's labelset
        # when its votes were derived for this dataset, which the history never
        # records.  The progress cache replays the history on top of them, so
        # its label sets are the ones the learned sort trained on (#5358).
        "label_history_base",
        "vote_click_times",
        "vote_region_boxes",
        # Per-vote surfacing provenance (which flow / phase / sort / rank
        # surfaced the item).  Recorded at click time because none of it is
        # re-derivable later: the ranking is client-side ephemeral state and
        # the score's model is overwritten by the next retrain.  See
        # ``vtscore/datasets/vote_provenance.py``.
        "vote_provenance",
        "click_counter",
        # True when this detector's in-memory votes are find/scoring output
        # (set by /api/find-label) rather than genuine training labels.  While
        # set, ``sync_labels_to_loaded_detector`` refuses to persist the votes
        # so a scoring pass can't overwrite the detector's saved labelset.
        # Per-detector (not a process global) so a find pass on one detector
        # never blocks vote syncing on another.  Cleared whenever the votes are
        # re-derived from the on-disk labelset (detector load / dataset switch).
        "find_mode",
        # Training artifacts
        "last_learned_scores",
        "textsort_suggestions",
        "find_initial_labels",
        # Find-session verification state:
        # ``verified_ids`` are the ids the human has explicitly verified this Find
        # session (a dict used as an ordered set, like ``good_votes``);
        # ``find_scores`` is the frozen per-item detector score from the single
        # scoring pass, so a balance (cutoff) change re-thresholds without
        # re-scoring.  Both are in-memory only and never persisted.
        "verified_ids",
        "find_scores",
        # True when the on-disk labelset has been changed (e.g. Find corrections
        # folded in + retrain) *since* this frozen Find evaluation was scored, so
        # the still-displayed find_initial_labels / find_scores reflect the
        # previous detector version.  Drives the "out of date" note in
        # ``GET /api/find/stats``.  Cleared on a fresh find-label scoring pass and
        # on any session reset (clear votes / dataset switch).
        "find_eval_stale",
        # The detector's balance (#4413): F-beta's beta, per detector and seeded
        # from the user's setting on first read (#3416), tracked in its own flag.
        "beta",
        "beta_seeded",
        # Cached in-memory data (never exported)
        "training_medias",  # voted media items with embeddings
        "label_embeddings",  # str → np.ndarray, keyed by stable_element_id
        # Region box the cached ``label_embeddings`` entry is final for, keyed
        # by stable_element_id: the element's box when resolving it again
        # would give the same vector (pooled from it, or a box the embedder
        # can't use), ``None`` for no box or a failed pool to retry.  Lets
        # ``populate_label_embeddings`` detect a region→none (or any region
        # edit) transition and re-resolve instead of returning a stale
        # region-pooled vector keyed to an element that no longer has a
        # region (logical-bug-audit finding M4), while an unchanged box
        # skips re-resolution (#4192).
        "label_embedding_regions",
        # Cross-dataset local features (StructuralFeatures) for the labelset's
        # elements, keyed by stable_element_id.  Re-derived from each element's
        # origin so a saved structural detector can build templates + train its
        # verification classifier against datasets that aren't currently loaded;
        # the full (unfiltered) features are cached and the region_box is applied
        # downstream at template-build time.  In-memory only, never persisted.
        # See docs/plans/structural-embedder.md.
        "label_local_features",  # str → StructuralFeatures
        # Region-flooded negatives for the labelset's Bad elements on patch
        # datasets: str → list[np.ndarray], keyed by stable_element_id, holding
        # the element's image-level vector + every raw patch so a saved detector
        # re-sorted cross-dataset floods the same rows the live vote path does.
        # In-memory only, re-derived from origins, never persisted; cleared on
        # embedder switch alongside ``label_embeddings``.
        "label_negative_regions",
        # Full score-row stacks for the labelset's elements on patch datasets:
        # str → list[np.ndarray], keyed by stable_element_id, holding the
        # image-level vector + every raw patch - the rows the scorer max-pools
        # that element's media over (identical to the flood above under
        # MaxPatch).  Lets threshold calibration collapse a Good bag and a Bad
        # bag the same way inference collapses any image, instead of comparing
        # a max-over-1 against a max-over-197.  In-memory only, re-derived from
        # origins, never persisted; cleared on embedder switch alongside
        # ``label_embeddings``.
        "label_score_regions",
        "model",  # nn.Sequential | None (current trained head)
        # Signature of the labelset ``model`` was trained from
        # (:func:`~vtscore.detectors.model_loading.labelset_signature`), or
        # None when unknown.  Stamped by every writer of ``model``; Find reuses
        # the cached head only while it still matches the detector's saved
        # labelset, because nothing that changes the labels drops ``model``
        # (issue #4204).  In-memory only, never persisted.
        "model_labels_sig",  # tuple | None
        # Structural detectors on a tiled (document) dataset keep every
        # (template, page) geometric fit here across retrains, so a vote verifies
        # one new template instead of the whole shortlist (#3928).  In-memory
        # only; entries record the features they were computed on, so a dataset
        # reload misses rather than reusing them.
        "structural_verification_cache",  # VerificationCache | None
        "threshold",  # decision threshold
        # Cross-dataset training-corpus counts (from on-disk labelset).  These
        # are independent of ``good_votes``/``bad_votes``, which only count
        # labels for media in the *currently loaded* dataset.  They drive the
        # frontend's "Sort by Learned" gating so a detector trained on dataset
        # A stays trainable when the user switches to dataset B.
        "labelset_good_count",
        "labelset_bad_count",
        # Dataset ID for which the cid-keyed vote state above is valid.
        # Media IDs are only meaningful within a single dataset, so when the
        # active dataset changes for a loaded detector we must clear the cid
        # dicts and re-derive them from the on-disk labelset against the new
        # dataset's medias.  See ``ensure_votes_match_active_dataset``.
        "votes_dataset_id",
        # Cached parsed labelset + mtime of the on-disk detector JSON the cache
        # was derived from.  Lets ``ensure_votes_match_active_dataset`` skip the
        # rehydrate (read+parse) when neither the active dataset nor the file
        # has changed, and lets ``learned_sort`` reuse the parsed labelset
        # instead of re-reading the JSON from disk on every click.
        "cached_labelset",  # LabelSet | None
        "cached_labelset_mtime",  # float
        "cached_labelset_media_type",  # str
        # Sync source
        "labelset_source",  # dict | None: {"source_name": "...", "field_values": {...}}
        # Calibration folds cache.  Holds ``(key, CalibrationFolds,
        # holdout_rows)`` where *key* is a deterministic fingerprint of the
        # **inclusion-independent** calibration inputs (training vectors,
        # labels, calibrate_count, calibration_fraction, hidden_dim), the
        # folds carry the per-fold held-out ``(scores, labels)``, the
        # fallback sentinel, and the trained fold models, and *holdout_rows*
        # names the training row behind each held-out score, for a
        # ``holdout_sink`` a cache hit must still fill.  Because the operating point
        # is deliberately absent from *key*, a re-cut at another inclusion
        # (the acquisition cut, Smart's pricing) hits the cache and only
        # re-runs the cheap quantile rule (no fold refit); a label/embedder
        # change rotates *key* and falls through to a fresh calibration.
        "calibration_cache",  # tuple[Any, CalibrationFolds, tuple[tuple[int, ...], ...]] | None
        # The fold-anchored population estimator the last retrain fitted
        # (``FoldAnchoredCut``), or None when it degenerated or had no haystack.
        # A re-cut answers an inclusion: the ``mid_tilt`` rule anchors the
        # measured midpoint cut at inclusion 0 and tilts monotonically away
        # from it (issue #2865).  Written on every retrain; read by
        # ``recut_detector_threshold`` and ``detector_acquisition_threshold`` so
        # every re-cut at an inclusion (the line with no class model or no
        # balance, the acquisition offset, Smart's pricing) uses this estimator
        # instead of the raw cross-calibration one.  Holds fitted Gaussians and
        # sorted score samples - process-scoped, never serialised.
        "anchored_cut_cache",  # FoldAnchoredCut | None
        # Retired (#4362): always ``None``.  It held the #4220 precision
        # estimate a retrain parked for the Find Stats curve; the curve went
        # with #4360 and nothing builds the estimate any more.  Kept so an
        # out-of-tree reader of the attribute sees "no estimate" rather than
        # an ``AttributeError``.
        "precision_floor_cache",  # None
        # The ranking the line is drawn over (``LineRanking``, #4272): the
        # haystack the last retrain scored (or that a head the label quota
        # gives drew its line on, #5048), sorted, with the items the trainer
        # held as voted.  The balance keeps the top *count* unvoted items of it;
        # a re-cut (``recut_detector_threshold``) and the spot check's
        # candidate both read it.  Ids and scores only - never serialised.
        "line_ranking",  # LineRanking | None
        # The items the structural verification gate passed on the last
        # re-rank (#4505): a structural detector's line is the gate's boundary,
        # so ``line_ranking`` is dropped and this is what the balance counts.
        # An unordered set, never a ranking a check could walk; cleared
        # wherever ``line_ranking`` is written or dropped.  Ids only.
        "gate_passed",  # frozenset[int] | None
        # The spot check (``SpotCheck``, #4272) that last finished on this
        # detector - its fixed candidate, labels, verdict and the fingerprint
        # of the set it left the line on - and the one running now, if any.
        # A finished result is kept across retrains and goes stale rather than
        # vanishing; a running check keeps the line's state as it was until it
        # ends.  Ids and labels only - never serialised.
        "precision_check",  # SpotCheck | None (finished)
        "precision_check_run",  # SpotCheck | None (running)
        # Test mode's test of the line (``LineTest``, #4524): the one running
        # on this detector's Find session, or the last finished one, over the
        # frozen Find scores.  Picks, labels and band counts only - never
        # serialised itself: a finished test's verdict is kept on the
        # detector's JSON instead (``vtscore.detectors.line_verdicts``,
        # #4526).  Goes wherever the spot check goes: a vote clear, a dataset
        # switch, a fresh Find pass.
        "line_test",  # LineTest | None
        # How many votes the detector held when its last spot check ended,
        # finished or cancelled: the weak-separation check never prompts once
        # it is set (``weak_check_due``, #4909), and the check before Done's
        # cooldown counts from it (``done_check_due``).  Goes with the check.
        "check_ended_votes",  # int | None
        # The line the labels imply (``LabelsLine``, #4452): the class model
        # the last retrain's calibration folds fitted from the labels, and the
        # prevalence it estimated on the corpus scored last (a Find pass over a
        # new dataset re-estimates it there).  A balance change re-cuts it.
        # Goes with the head.
        "labels_line",  # LabelsLine | None
        # The Goods' centroid's line (``CentroidLine``, #4732): the corpus
        # cosines it was drawn on, so a balance change redraws it on the same
        # scores.  Read only while the model is a centroid head.
        "centroid_line",  # CentroidLine | None
    )

    def __init__(
        self,
        detector_id: str = "",
        *,
        name: str = "",
        media_type: str = "",
        embedder: str = "",
        embedder_type: str = "",
    ) -> None:
        self.detector_id: str = detector_id
        self.name: str = name
        self.media_type: str = media_type
        self.embedder: str = embedder
        self.embedder_type: str = embedder_type
        # Vote state
        self.good_votes: dict[int, None] = {}
        self.bad_votes: dict[int, None] = {}
        self.label_history: list[tuple[int, str, float]] = []
        # Replaced whole, never mutated, so the progress cache can read it without the state lock.
        self.label_history_base: tuple[frozenset[int], frozenset[int]] = (frozenset(), frozenset())
        self.vote_click_times: dict[int, int] = {}
        # Per-good-vote region boxes (normalised x0, y0, x1, y1).  Only set when
        # the user drew a region as part of a yes-vote; absent for image-level
        # yes-votes and for every no-vote.  Patch-embedder v2.
        self.vote_region_boxes: dict[int, tuple[float, float, float, float]] = {}
        # Per-vote surfacing provenance, keyed by media id.  See the slot
        # comment and :mod:`vtscore.datasets.vote_provenance`.
        self.vote_provenance: dict[int, dict[str, object]] = {}
        self.click_counter: int = 0
        # See the slot comment: True while these votes are find/scoring output.
        self.find_mode: bool = False
        # Training artifacts
        self.last_learned_scores: dict[int, float] = {}
        self.textsort_suggestions: list[str] = []
        self.find_initial_labels: dict[int, str] = {}
        # Find-session verification state.
        self.verified_ids: dict[int, None] = {}
        self.find_scores: dict[int, float] = {}
        # See the slot comment: the displayed Find evaluation is for the detector
        # version that scored this pass; flipped True when its labelset changes
        # underneath (corrections folded in + retrain).
        self.find_eval_stale: bool = False
        self.beta: float | None = DEFAULT_BETA
        self.beta_seeded: bool = False
        # Cached in-memory data (never exported)
        self.training_medias: dict[int, dict[str, Any]] = {}
        # Embeddings for every saved labelset element, keyed by
        # stable_element_id.  Populated at detector load (resolve_file +
        # embed_file) and topped up when new votes come in.  Lets head
        # training and learned-sort use *all* saved labels, including
        # those whose underlying media isn't part of the active dataset.
        self.label_embeddings: dict[str, Any] = {}
        self.label_embedding_regions: dict[str, tuple[float, float, float, float] | None] = {}
        self.label_local_features: dict[str, Any] = {}
        self.label_negative_regions: dict[str, list[Any]] = {}
        self.label_score_regions: dict[str, list[Any]] = {}
        self.model: Any = None  # nn.Sequential | None
        self.model_labels_sig: tuple | None = None
        self.structural_verification_cache: Any = None  # VerificationCache | None
        self.threshold: float = 0.5
        self.labelset_good_count: int = 0
        self.labelset_bad_count: int = 0
        self.votes_dataset_id: str = ""
        self.cached_labelset: Any = None  # LabelSet | None
        self.cached_labelset_mtime: float = 0.0
        self.cached_labelset_media_type: str = ""
        # Sync source
        self.labelset_source: dict[str, Any] | None = None
        self.calibration_cache: tuple[Any, ...] | None = None
        self.anchored_cut_cache: Any = None  # FoldAnchoredCut | None
        self.precision_floor_cache: Any = None  # retired (#4362): always None
        self.line_ranking: Any = None  # LineRanking | None
        self.gate_passed: frozenset[Any] | None = None
        self.precision_check: Any = None  # SpotCheck | None
        self.precision_check_run: Any = None  # SpotCheck | None
        self.line_test: Any = None  # LineTest | None
        self.check_ended_votes: int | None = None
        self.labels_line: Any = None  # LabelsLine | None
        self.centroid_line: Any = None  # CentroidLine | None
