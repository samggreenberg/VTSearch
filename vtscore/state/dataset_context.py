"""``DatasetContext``: all mutable state that belongs to one loaded dataset, and its ``MediasDict``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vtscore.state.primitives import _state_lock


# ---------------------------------------------------------------------------
# DatasetContext: bundles all per-dataset mutable state
# ---------------------------------------------------------------------------


class MediasDict(dict):
    """A ``dict`` of media items that bumps a revision counter on mutation.

    Backs :attr:`DatasetContext.medias`.  The embedding-matrix and
    region-matrix caches key their validity on ``ctx.media_revision``
    (see :mod:`vtscore.embedding.matrix`); routing every add / remove /
    replace through this subclass means those *structural* mutations bump
    the revision automatically, so no call site has to remember to
    invalidate the derived caches after changing which media are loaded.

    Limitation (important): a ``dict`` subclass only observes changes to
    the *key → value* mapping.  An in-place edit of a value dict - e.g.
    ``medias[cid]["embedding"] = vec`` while re-embedding or clipping -
    never calls ``__setitem__`` on this container, so it does **not** bump
    the counter.  Code that rewrites a media's vector in place must bump it
    itself via :meth:`DatasetContext.bump_media_revision`; the embed / clip
    load stages do this through ``invalidate_embedding_matrix``.  This is
    the ``media_revision`` follow-up (logical-bug-audit root-cause
    Pattern #4).

    Over-bumping (bumping when nothing actually changed) is always safe: it
    only forces an unnecessary cache rebuild, never serves a stale one.
    """

    __slots__ = ("_on_mutate",)

    def __init__(self, on_mutate: Callable[[], None], *args: Any, **kwargs: Any) -> None:
        # dict's C-level constructor / update populate directly without
        # calling our Python __setitem__, so seeding from *args* does not
        # fire _on_mutate - the owner counts construction as one bump when it
        # assigns the dict, not once per seeded item.
        object.__setattr__(self, "_on_mutate", on_mutate)
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        self._on_mutate()

    def __delitem__(self, key: Any) -> None:
        super().__delitem__(key)  # raises before the bump if key is absent
        self._on_mutate()

    def pop(self, *args: Any) -> Any:
        n = len(self)
        result = super().pop(*args)
        if len(self) != n:
            self._on_mutate()
        return result

    def popitem(self) -> Any:
        result = super().popitem()  # raises on empty before the bump
        self._on_mutate()
        return result

    def clear(self) -> None:
        had_items = bool(self)
        super().clear()
        if had_items:
            self._on_mutate()

    def update(self, *args: Any, **kwargs: Any) -> None:
        super().update(*args, **kwargs)
        self._on_mutate()

    def setdefault(self, key: Any, default: Any = None) -> Any:
        existed = key in self
        result = super().setdefault(key, default)
        if not existed:
            self._on_mutate()
        return result


class DatasetContext:
    """All mutable state that belongs to a single loaded dataset.

    Vote-related state (``good_votes``, ``bad_votes``, ``label_history``, etc.)
    lives in :class:`DetectorContext`, not here.  ``DatasetContext`` holds only
    dataset-intrinsic state: the media items, coverage atlas, and display name.
    """

    __slots__ = (
        "dataset_id",
        # Backing store for the ``medias`` property (a :class:`MediasDict`)
        # and the monotonic revision counter it bumps.  ``media_revision``
        # advances on every structural mutation of the medias (and on every
        # in-place vector rewrite that calls ``bump_media_revision``); the
        # embedding / region matrix caches key their validity on it instead
        # of comparing sorted id lists (logical-bug-audit root-cause
        # Pattern #4).
        "_medias",
        "_media_revision",
        "coverage_atlas",
        "dataset_display_name",
        # Cached contiguous (N, D) float32 embedding matrix, the sorted
        # media-id list it corresponds to, and the ``media_revision`` it was
        # built at.  Built lazily on first access by
        # ``vtscore.embedding.matrix.get_embedding_matrix`` and reused
        # across cosine sort, head scoring, and coverage-atlas construction so
        # we don't rebuild a 10k-row matrix per call.  Cache validity is the
        # revision match; the id list is kept for the returned tuple / region
        # snap-key match.
        "_emb_matrix_ids",
        "_emb_matrix",
        "_emb_matrix_revision",
        # Memoised answer to "do *all* these medias share one primary
        # embedder, and which?" (``None`` = they disagree, or some media has
        # no primary), plus the ``media_revision`` it was computed at.
        # ``vtscore.embedding.matrix._collapse_to_primary_for_ctx`` asks this
        # before the matrix-cache check to decide whether an explicitly routed
        # embedder name is the whole dataset's primary and so may reuse the
        # cached primary matrix (issue #3650).  Memoised because that question
        # is an O(N) Python scan on a path that otherwise touches no media.
        "_uniform_primary",
        "_uniform_primary_revision",
        # One-way latch: set the first time ``invalidate_embedding_matrix``
        # fires for this context (an in-place vector rewrite - re-embed /
        # clip). Once set, ``get_embedding_matrix`` never again considers the
        # on-disk mmap sidecar (see S1, docs/plans/scalability.md) for this
        # context's lifetime, even though the id list alone can't detect a
        # same-id in-place rewrite. A fresh load gets a fresh ``DatasetContext``
        # (and so a fresh, unset latch), so this never needs resetting.
        "_emb_sidecar_disabled",
        # Cached *flattened region matrix* for patch-region (e.g. DINOv3)
        # datasets, mirroring ``_emb_matrix`` but expanded to one row per
        # (media, region) pair.  ``_region_matrix`` is the ``(R, D)`` float32
        # matrix; ``_region_media_index`` / ``_region_index_per_row`` are the
        # parallel ``int64`` arrays mapping each row back to its media index
        # (into the sorted id list) and its region index within that media.
        # Built lazily by ``vtscore.embedding.matrix.get_region_matrix_for_snap``
        # and reused across votes so online retraining never rebuilds the
        # multi-hundred-thousand-row matrix per vote.  Keyed, like
        # ``_emb_matrix``, on the sorted media-id list in ``_region_matrix_ids``
        # plus the ``media_revision`` it was built at (``_region_matrix_revision``).
        "_region_matrix_ids",
        "_region_matrix",
        "_region_matrix_revision",
        "_region_media_index",
        "_region_index_per_row",
        # Cached secondary media lookups (S14): the ``(origin_key, md5, name)``
        # triple ``build_media_lookup`` produces, keyed - like the embedding
        # matrix above - on the ``media_revision`` it was built at
        # (``_lookup_index_revision``).  Many routes resolve label entries
        # against the active dataset (label import/export, find-stats,
        # add-to-pile, learned sort); rebuilding these O(N) tables (a
        # ``json.dumps`` per origin) on every request is what S14 removes.
        # Built + reused by ``vtscore.state.media_lookup.cached_media_lookups``;
        # ``None`` until first access, invalidated automatically because the
        # revision advances on every structural medias mutation.
        "_origin_key_index",
        "_md5_index",
        "_name_index",
        "_lookup_index_revision",
        # VTSBrowse: cached projection (frozen at ingest) + per-bin-shape
        # pyramids derived from it. The projection (UMAP coords) is shared
        # across bin shapes; ``_pyramids`` maps "hex"/"square" -> Pyramid so the
        # browse hex/square toggle can keep both binnings cached at once.
        # ``_full_job_id`` tracks the in-flight background UMAP build for status
        # polling — the projection runner is a single app-wide slot shared with
        # every other dataset's full build and every subset build, so a poll
        # must look up *this* dataset's own job by id rather than asking the
        # runner what it happens to be running right now (which may be a
        # different dataset's job while this one sits parked pending).
        "_projection",
        "_pyramids",
        "_full_job_id",
        # VTSBrowse region signposts (the "street sign" name layer; see
        # docs/plans/vtsbrowse-toponymy.md).  A RegionLabelSet computed by the
        # labeling pipeline for the *current* frozen layout, or None.  The set
        # carries the projection_id it was fit against, and the labels route
        # refuses to serve it over a different layout — so a stale set is
        # inert, never wrong.  Text + 2-D anchors only, no vectors.
        "_region_labels",
        # In-flight background relabel job id (issue #2404).  When the serve
        # path finds a persisted label set whose ``labeler_signature`` no
        # longer matches the active pipeline, it kicks a background rebuild so
        # the stale signs self-heal; this tracks that job so repeated polls
        # coalesce onto the one rebuild instead of queueing a fresh one each
        # time.  Shares the ``signpost_relabel_jobs`` runner across datasets.
        "_relabel_job_id",
        # VTSBrowse subset projection: an ephemeral UMAP fit over just a subset
        # of this dataset's media ids (e.g. the positives of a Find run),
        # computed on demand and never persisted.  Held alongside the full
        # projection so a user can browse the whole dataset and a subset
        # independently.  ``_subset_ids`` is the sorted id list the cached
        # subset layout was fit against (cache key); ``_subset_job_id`` tracks
        # the in-flight background UMAP build for status polling.
        "_subset_projection",
        "_subset_pyramids",
        "_subset_ids",
        "_subset_job_id",
        "_subset_content_version",
        # Region signposts for the subset layout (mirror of ``_region_labels``).
        "_subset_region_labels",
        # The ids in 1-D similarity order (#5465): a SimilarityOrder the
        # example picker lists a Face dataset in, cached here once read from
        # the container or fit.  Ids only, no vectors; checked against the live
        # id set on every read (vtscore.projection.similarity_order).
        "_similarity_order",
        # Role-typed embedder binding (v3 "three-slot" trio; see
        # docs/ARCHITECTURE.md).  A dataset binds up to one
        # text-capable embedder, up to one patch-capable embedder, and up to
        # one structural (geometric-verification) embedder.  By default the
        # binding is *derived* on demand from the dataset's medias (the
        # single embedder a pre-v3 dataset was loaded with);
        # ``bind_embedders`` overrides that with an explicit, validated
        # triple for genuinely multi-embedder datasets.  The values are
        # embedder *names*, never vectors - no embeddings live here.
        "_text_embedder",
        "_patch_embedder",
        "_structural_embedder",
        "_binding_explicit",
        # Transient create-time flag: when set, the finalize stage runs an
        # extra near-duplicate collapse (images + text) after exact MD5
        # dedup.  Never persisted - the *result* is baked into origins, so
        # reloads default this off.
        "merge_near_duplicates",
    )

    # ------------------------------------------------------------------
    # Derived-cache families (deliberately adjacent to ``__slots__``)
    # ------------------------------------------------------------------
    # Every slot above is either a *source of truth* for this dataset or a
    # *derived cache* - something rebuildable, at some cost, from the medias
    # or from the persisted layout.  The four families below name the caches
    # and :meth:`reset_derived_caches` drops them by family; every call site
    # that used to hand-clear a list of private slots goes through it now.
    #
    # ``_NON_DERIVED_SLOTS`` names the rest explicitly so the two sets can be
    # asserted to *partition* ``__slots__`` (see
    # ``tests_lib/state/test_derived_cache_reset.py``).  That partition is the
    # whole point: a slot added to ``__slots__`` fails the test until it is
    # filed as a cache or as state, instead of being quietly missed the way
    # every hand-written clear-list here had already missed eleven slots by
    # the time issue #3377 counted them.
    _DERIVED_CACHE_SLOTS: dict[str, frozenset[str]] = {
        # Contiguous embedding / flattened-region matrices and their keys.
        "matrices": frozenset(
            {
                "_emb_matrix_ids",
                "_emb_matrix",
                "_emb_matrix_revision",
                "_uniform_primary",
                "_uniform_primary_revision",
                "_region_matrix_ids",
                "_region_matrix",
                "_region_matrix_revision",
                "_region_media_index",
                "_region_index_per_row",
            }
        ),
        # Secondary media lookups (origin key / md5 / name) and their key.
        "lookups": frozenset(
            {
                "_origin_key_index",
                "_md5_index",
                "_name_index",
                "_lookup_index_revision",
            }
        ),
        # VTSBrowse full-dataset layout: projection, per-shape pyramids,
        # signposts, and the in-flight build / relabel job ids that would
        # otherwise keep pointing at a discarded layout.
        "projection": frozenset(
            {
                "_projection",
                "_pyramids",
                "_full_job_id",
                "_region_labels",
                "_relabel_job_id",
            }
        ),
        # VTSBrowse subset layout: the mirror of the above, plus the id list
        # the cached subset was fit against and its tile-cache token.
        "subset": frozenset(
            {
                "_subset_projection",
                "_subset_pyramids",
                "_subset_ids",
                "_subset_job_id",
                "_subset_content_version",
                "_subset_region_labels",
            }
        ),
        # The 1-D similarity order; persisted beside the dataset, so dropping
        # it costs a container read, not a re-fit.
        "order": frozenset({"_similarity_order"}),
    }

    #: Slots that are state, not cache - never touched by
    #: :meth:`reset_derived_caches`.  Each is here for a reason:
    #:
    #: * ``dataset_id`` / ``dataset_display_name`` / ``merge_near_duplicates``
    #:   and the four embedder-binding slots are dataset identity + config.
    #: * ``_medias`` / ``_media_revision`` are the source of truth the caches
    #:   are derived *from*.
    #: * ``coverage_atlas`` is derived, but it is persisted in the dataset
    #:   pickle and restored by the load pipeline, which owns its lifetime.
    #: * ``_emb_sidecar_disabled`` is a one-way latch, not a cache: it records
    #:   that an in-place vector rewrite happened, which no rebuild undoes.
    _NON_DERIVED_SLOTS: frozenset[str] = frozenset(
        {
            "dataset_id",
            "_medias",
            "_media_revision",
            "coverage_atlas",
            "dataset_display_name",
            "_emb_sidecar_disabled",
            "_text_embedder",
            "_patch_embedder",
            "_structural_embedder",
            "_binding_explicit",
            "merge_near_duplicates",
        }
    )

    def __init__(self, dataset_id: str = "") -> None:
        self.dataset_id: str = dataset_id
        # ``_media_revision`` must exist before ``_medias`` so the MediasDict's
        # mutate callback can safely bump it.  Assign the backing slot directly
        # (not via the ``medias`` setter) so construction leaves the revision
        # at 0 rather than counting itself as a mutation.
        self._media_revision: int = 0
        self._medias: MediasDict = MediasDict(self.bump_media_revision)
        self.coverage_atlas: Any = None  # CoverageAtlas | None
        self.dataset_display_name: str | None = None
        self._emb_matrix_ids: list[int] | None = None
        self._emb_matrix: Any = None  # np.ndarray | None
        self._emb_matrix_revision: int | None = None  # media_revision the matrix was built at
        self._uniform_primary: str | None = None  # primary embedder shared by every media, else None
        self._uniform_primary_revision: int | None = None  # media_revision that answer was computed at
        self._emb_sidecar_disabled: bool = False  # latched True by invalidate_embedding_matrix
        self._region_matrix_ids: list[int] | None = None
        self._region_matrix: Any = None  # np.ndarray | None, shape (R, D)
        self._region_matrix_revision: int | None = None  # media_revision the region matrix was built at
        self._region_media_index: Any = None  # np.ndarray | None, int64 (R,)
        self._region_index_per_row: Any = None  # np.ndarray | None, int64 (R,)
        # Cached secondary media lookups (S14); see the __slots__ comment.
        self._origin_key_index: dict[str, list[int]] | None = None
        self._md5_index: dict[str, list[int]] | None = None
        self._name_index: dict[str, list[int]] | None = None
        self._lookup_index_revision: int | None = None  # media_revision the lookups were built at
        self._projection: Any = None  # Projection | None
        self._pyramids: dict[str, Any] = {}  # bin_shape -> Pyramid
        self._full_job_id: str | None = None  # in-flight full-dataset build job id
        self._region_labels: Any = None  # RegionLabelSet | None (signposts, full layout)
        self._relabel_job_id: str | None = None  # in-flight signpost-relabel job id (#2404)
        self._subset_projection: Any = None  # Projection | None (ephemeral subset UMAP)
        self._subset_pyramids: dict[str, Any] = {}  # bin_shape -> Pyramid (subset)
        self._subset_ids: list[int] | None = None  # sorted ids the subset layout is fit on
        self._subset_job_id: str | None = None  # in-flight subset build job id
        self._subset_region_labels: Any = None  # RegionLabelSet | None (signposts, subset)
        # Bumped on each in-place edit of the subset layout (e.g. removing
        # false-positives from a Find browse).  The layout/``projection_id`` is
        # kept stable so the canvas preserves the viewport; this counter changes
        # only the tile cache key/URL so stale tiles aren't served (the tile URL
        # is otherwise cached ``immutable``).  Reset to 0 on a fresh subset fit.
        self._subset_content_version: int = 0
        self._similarity_order: Any = None  # SimilarityOrder | None (#5465)
        # Role-typed embedder binding (see __slots__ comment).  ``None``
        # until either explicitly bound or derived from the medias.
        self._text_embedder: str | None = None
        self._patch_embedder: str | None = None
        self._structural_embedder: str | None = None
        self._binding_explicit: bool = False
        self.merge_near_duplicates: bool = False

    # ------------------------------------------------------------------
    # Medias + revision counter (root-cause Pattern #4)
    # ------------------------------------------------------------------

    @property
    def medias(self) -> MediasDict:
        """The dataset's ``{id: media}`` map, as a revision-tracking dict.

        Structural mutations (add / remove / replace an entry) bump
        :attr:`media_revision` automatically via :class:`MediasDict`.
        """
        return self._medias

    @medias.setter
    def medias(self, value: dict[int, dict[str, Any]]) -> None:
        # Wrap any assigned mapping in a fresh MediasDict bound to *this*
        # context, then count the wholesale replacement as one mutation so a
        # cached matrix built against the old contents is invalidated even
        # when the new id set happens to match the old one.
        self._medias = MediasDict(self.bump_media_revision, value)
        self.bump_media_revision()

    @property
    def media_revision(self) -> int:
        """Monotonic counter, advanced on every mutation of the medias.

        The embedding-matrix and region-matrix caches compare this single
        int instead of two sorted id lists, so a mutation that changes
        vectors without changing the id set still invalidates them
        (logical-bug-audit root-cause Pattern #4).
        """
        return self._media_revision

    def bump_media_revision(self) -> None:
        """Advance :attr:`media_revision` by one.

        Called automatically by :class:`MediasDict` on structural changes.
        Call it directly after an *in-place* rewrite of a media's embedding
        vector (which a dict subclass can't observe) so the derived matrix
        caches rebuild; the embed / clip stages do so via
        ``invalidate_embedding_matrix``.
        """
        self._media_revision += 1

    # ------------------------------------------------------------------
    # Derived-cache invalidation
    # ------------------------------------------------------------------

    def reset_derived_caches(
        self,
        *,
        matrices: bool = True,
        lookups: bool = True,
        projection: bool = True,
        subset: bool = True,
        order: bool = True,
    ) -> None:
        """Drop this context's derived caches, by family.

        The single place any of the ``_DERIVED_CACHE_SLOTS`` above are
        cleared.  Before issue #3377 four call sites each hand-wrote their
        own list of private slots to ``None``, and all four had drifted from
        ``__slots__``: ``clear_medias`` alone was missing the four lookup
        slots, the two full-layout job ids, and all six subset slots, so a
        reload left a stale subset layout that
        ``POST /api/projection/subset`` would serve verbatim whenever the new
        id set happened to match the old one - the exact failure the
        ``clear_medias`` docstring already called out for ``_projection`` and
        ``_pyramids``.

        Each keyword drops one family; pass ``False`` to keep one. The
        default (everything) is what a wholesale medias clear wants.

        Does **not** bump :attr:`media_revision`. Dropping a cache is not a
        change to the medias, and the revision-keyed caches distinguish the
        two: a caller signalling an in-place vector rewrite (
        :func:`vtscore.embedding.matrix.invalidate_embedding_matrix`) bumps
        it itself.
        """
        with _state_lock:
            if matrices:
                self._emb_matrix_ids = None
                self._emb_matrix = None
                self._emb_matrix_revision = None
                self._uniform_primary = None
                self._uniform_primary_revision = None
                self._region_matrix_ids = None
                self._region_matrix = None
                self._region_matrix_revision = None
                self._region_media_index = None
                self._region_index_per_row = None
            if lookups:
                self._origin_key_index = None
                self._md5_index = None
                self._name_index = None
                self._lookup_index_revision = None
            if projection:
                self._projection = None
                self._pyramids = {}
                self._full_job_id = None
                self._region_labels = None
                self._relabel_job_id = None
            if subset:
                self._subset_projection = None
                self._subset_pyramids = {}
                self._subset_ids = None
                self._subset_job_id = None
                self._subset_content_version = 0
                self._subset_region_labels = None
            if order:
                self._similarity_order = None

    # ------------------------------------------------------------------
    # Role-typed embedder binding
    # ------------------------------------------------------------------

    def bind_embedders(
        self,
        *,
        text_embedder: str | None = None,
        patch_embedder: str | None = None,
        structural_embedder: str | None = None,
    ) -> None:
        """Explicitly bind role-typed embedders to this dataset.

        Validates that each slot points at an embedder with the matching
        capability (text / patch / structural) and then stores the triple,
        overriding the default media-derived binding.  Use this for genuinely
        multi-embedder datasets; a single-embedder dataset can rely on the
        derived binding instead.

        Stores embedder *names* only - never vectors or models.
        """
        from vtscore.embedding.binding import validate_binding  # noqa: PLC0415

        validate_binding(text_embedder, patch_embedder, structural_embedder)
        self._text_embedder = text_embedder
        self._patch_embedder = patch_embedder
        self._structural_embedder = structural_embedder
        self._binding_explicit = True

    def _resolve_binding(self) -> tuple[str | None, str | None, str | None]:
        """Return the ``(text_embedder, patch_embedder, structural_embedder)`` triple.

        An explicit binding (set via :meth:`bind_embedders`) wins; otherwise
        the triple is derived from the dataset's medias - the single embedder
        a pre-v3 dataset was loaded with, role-typed by its capabilities.
        """
        if self._binding_explicit:
            return (self._text_embedder, self._patch_embedder, self._structural_embedder)
        if not self.medias:
            return (None, None, None)
        from vtscore.embedding.binding import derive_binding_from_names  # noqa: PLC0415
        from vtscore.embedding.media_vectors import media_embedder_names  # noqa: PLC0415

        first = next(iter(self.medias.values()))
        return derive_binding_from_names(media_embedder_names(first))

    @property
    def text_embedder(self) -> str | None:
        """Name of the bound text-capable embedder, or ``None``."""
        return self._resolve_binding()[0]

    @property
    def patch_embedder(self) -> str | None:
        """Name of the bound patch-capable embedder, or ``None``."""
        return self._resolve_binding()[1]

    @property
    def structural_embedder(self) -> str | None:
        """Name of the bound structural (geometric-verification) embedder, or ``None``."""
        return self._resolve_binding()[2]

    def routed_embedder(self, role: str) -> str | None:
        """Resolve which bound embedder serves *role* (the v3 routing table).

        Roles (see "Routing" in ``docs/ARCHITECTURE.md`` → "Embedder binding and routing"):

        * ``"text"`` - text queries (``POST /api/sort``): the text slot, or
          ``None`` (the caller 400s with ``supports_text=False``).
        * ``"patch"`` - region similarity / voting (``/api/find-label`` region
          overlays, region votes): the patch slot, or ``None`` (the caller 400s;
          region ops need a patch embedder).
        * ``"structural"`` - instance retrieval / geometric verification: the
          structural slot, or ``None`` (the caller 400s; structural ops need a
          structural embedder).
        * ``"score"`` - cosine example sort, the detector head (train + score),
          and the coverage atlas: the structural slot if bound, else the patch
          slot, else the text slot, else ``None``.  ``None`` means a slot-less
          single-vector dataset (e.g.  ``dinov2_single``); the matrix layer then
          reads each media's primary vector, so cosine sort / head scoring keep
          working rather than 400-ing.

        Returns an embedder *name* (or ``None``).  Pass the result straight to
        the embedder-aware matrix layer: a name equal to the dataset's primary
        embedder collapses to the cached primary path there, so the
        single-embedder hot path is unchanged byte-for-byte.
        """
        text, patch, structural = self._resolve_binding()
        if role == "text":
            return text
        if role == "patch":
            return patch
        if role == "structural":
            return structural
        if role == "score":
            return structural or patch or text
        raise ValueError(f"unknown embedder routing role: {role!r}")

    @property
    def supports_text(self) -> bool:
        """Whether this dataset can answer text queries (text slot is bound)."""
        return self.text_embedder is not None

    @property
    def supports_patch_regions(self) -> bool:
        """Whether this dataset has region overlays / voting (patch slot is bound)."""
        return self.patch_embedder is not None

    @property
    def supports_geometric_verification(self) -> bool:
        """Whether this dataset can do instance retrieval (structural slot is bound)."""
        return self.structural_embedder is not None
