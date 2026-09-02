"""Fit Toponymy over a frozen projection and extract region signposts.

The build-time half of the sign pipeline (see
``docs/plans/vtsbrowse-toponymy.md``): given a frozen 2-D layout, the media
embedding matrices, and one text per media (``signpost_texts``), run the
Tutte Institute ``toponymy`` library — multiresolution clustering →
contrastive keyphrases → naming — and flatten its fitted topic tree into a
:class:`~vtscore.projection.labels.RegionLabelSet` the canvas can letter.

Two vector spaces feed the fit, and they are deliberately allowed to differ:

* ``clusterable_vectors`` — a dedicated ~5-D cosine UMAP of the **score**
  embedder's matrix, the same space the frozen 2-D layout was fit in, so the
  named clusters are compact regions *on the map* (an anchor computed from a
  cluster scattered across the layout would land in noise);
* ``embedding_vectors`` — the **text-capable** embedder's matrix, the space
  the keyphrase strings are embedded into, so Toponymy's keyphrase↔cluster
  alignment is meaningful.

For the common single cross-modal embedder (CLAP/SigLIP) the two are the same
matrix, exactly the configuration the signpost studies validated.

Everything here is build-time-only: the topic model, keyphrase vectors, and
clusterable UMAP are dropped on the floor; only derived text + 2-D anchors +
scalar scores leave this module (the No-Persisted-Vectors rule).

``toponymy`` is an optional dependency (installed ``--no-deps``; see
``docs/plans/vtsbrowse-toponymy.md`` for why) and is imported lazily —
:func:`signposting_available` is the cheap probe callers gate on.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata, util
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from vtscore.projection.labels import RegionLabel, RegionLabelSet, make_label_set, medoid

if TYPE_CHECKING:
    from vtscore.projection.umap_projection import Projection

logger = logging.getLogger(__name__)

#: Zoom-level gap between adjacent topic layers — the same zoom-band step the
#: ground-truth demo signposts use (see ``demo_signposts._LEVEL_STEP``), so the
#: canvas hand-off feel matches whichever source lettered the map.
_LEVEL_STEP = 1.8

#: Below this many points a topic tree is degenerate (min_clusters=4 with a
#: base min cluster size of 10 leaves nothing to name); serve no signs.
_MIN_POINTS = 40

#: Toponymy's example configuration (docs/plans/vtsbrowse-toponymy.md locks
#: "follow their examples"): a ~5-D cosine UMAP as the clusterable space.
_CLUSTER_DIM = 5

#: Progress callback shape: ``(current, total, message)``.
ProgressFn = Callable[[int, int, str], None]


def signposting_available() -> bool:
    """Whether the ``toponymy`` library is importable in this environment.

    The silent probe: the serve / signature paths gate on this without
    complaint, because those run on every projection poll and a ``None``
    result there is routine.  Build paths use :func:`require_signposting`
    instead, which treats a missing install as the error it is.
    """
    return util.find_spec("toponymy") is not None


#: One-time latch so a missing-toponymy install shouts exactly once per
#: process, not on every build attempt (subset re-fits, relabel jobs, …).
_missing_toponymy_warned = False


def require_signposting() -> bool:
    """Gate a signpost *build*, logging loudly if ``toponymy`` is missing.

    Unlike :func:`signposting_available` (the quiet probe), ``toponymy`` is not
    a soft optional at build time: ``scripts/install.sh`` installs it
    unconditionally (alongside ``apricot-select``), so its absence when a build
    actually runs means a broken or incomplete install, not a dataset that
    legitimately can't be lettered.  We still degrade gracefully — the caller
    returns ``None`` and the map stays unlettered rather than crashing — but we
    surface the misconfiguration with a one-time ``error`` log so it can't hide
    behind the genuinely-quiet no-embedder / no-provider skips.  Returns whether
    signposting can proceed.
    """
    if signposting_available():
        return True
    global _missing_toponymy_warned
    if not _missing_toponymy_warned:
        _missing_toponymy_warned = True
        logger.error(
            "VTSBrowse signposts are disabled: the required 'toponymy' library "
            "is not importable. It is installed by scripts/install.sh (which "
            "also installs its apricot-select dependency); re-run that script to "
            "repair this environment. Maps will render unlettered until then."
        )
    return False


def toponymy_version() -> str:
    """The installed toponymy distribution version, or ``""``."""
    try:
        return metadata.version("toponymy")
    except metadata.PackageNotFoundError:
        return ""


class EmbedderTextEncoder:
    """Toponymy ``TextEmbedderProtocol`` adapter over ``MediaEmbedder.embed_text``.

    Keyphrase strings are embedded into the same space as the media matrix, so
    cross-modal keyphrase→cluster alignment is meaningful (CLAP/SigLIP).  A
    string the embedder can't encode maps to a zero vector of the right width
    rather than poisoning the batch.
    """

    def __init__(self, embedder: Any, dim: int):
        self._embedder = embedder
        self._dim = dim

    def encode(self, texts: Any, show_progress_bar: bool = False, **kwargs: Any) -> np.ndarray:
        out = []
        for text in texts:
            vec = None
            try:
                if str(text).strip():
                    vec = self._embedder.embed_text(str(text))
            except Exception:
                vec = None
            if vec is None:
                vec = np.zeros(self._dim, dtype=np.float32)
            out.append(np.asarray(vec, dtype=np.float32))
        return np.stack(out) if out else np.empty((0, self._dim), dtype=np.float32)


#: Matches one topic header in a Toponymy disambiguation prompt. The
#: ``combined`` prompt format (the one a no-system-prompt namer gets) lists
#: each colliding topic as a bare ``"N. topic name":`` line at column 0; the
#: capture groups are the 1-based index and the topic name. Anchoring at line
#: start (no leading ``\s*``) and requiring the line to *end* right after the
#: ``":`` keeps the JSON output-format example (``{<1. OLD_NAME1>: …}``,
#: indented and continuing past the colon) from being mistaken for a header.
_DISAMBIGUATION_HEADER_RE = re.compile(r'^"(\d+)\.\s(.+?)":\s*$', re.MULTILINE)


#: What :func:`_keyphrase_topic_name` falls back to when the naming prompt has
#: no keyword line to read — a user-visible sign on the browse canvas, so its
#: rate is counted and logged alongside the suppressed-warning count.
_UNNAMED = "unnamed"


def _keyphrase_topic_name(prompt: str) -> str:
    """The top keyphrase for a single-topic naming prompt (or ``"unnamed"``)."""
    match = re.search(r"Keywords for this group include:\s*([^\n]+)", prompt)
    name = match.group(1).split(",")[0].strip() if match else ""
    return name or _UNNAMED


def _keyphrase_disambiguation(prompt: str) -> dict[str, Any]:
    """Echo old names back for a duplicate-name disambiguation prompt.

    A no-LLM namer cannot invent new distinctions, so every colliding topic
    keeps the name Toponymy already gave it. The one thing that *must* be
    right is the mapping's length: Toponymy's ``default_extract_topic_names``
    raises ``ValueError`` unless the mapping has exactly one entry per topic,
    and that exception drives three ``wait_random_exponential(4, 10)`` retries
    per cluster before a ``UserWarning`` is emitted — the console flood and
    multi-second-per-cluster stall of issue #2558. Parsing every
    ``"N. name":`` header (keyed as ``f"{i}. {name}"`` so the library maps it
    back to the original topic) guarantees the one-per-topic count and lets
    the fix succeed on the first attempt, silently.
    """
    mapping = {f"{index}. {name}": name for index, name in _DISAMBIGUATION_HEADER_RE.findall(prompt)}
    return {"new_topic_name_mapping": mapping, "topic_specificities": [0.5] * len(mapping)}


def make_keyphrase_namer(on_name: Callable[[], None] | None = None, counts: Counter[str] | None = None) -> Any:
    """The no-LLM namer: answer every naming prompt with the top keyphrase.

    A trivial in-process ``LLMWrapper`` whose "LLM calls" parse the prompt
    Toponymy built and return its first (top ``information_weighted``)
    keyphrase as the topic name — an honest fallback that keeps the library's
    contrastive clustering + keyphrase machinery and skips only the LLM's
    phrasing.  Duplicate-name disambiguation passes echo the old names back
    (a no-LLM namer cannot invent new distinctions).

    ``on_name`` fires once per *topic-naming* call (not the batched
    disambiguation passes), which is how :func:`_fit_topic_layers` drives a
    determinate naming progress bar: one topic named ≈ one tick.

    ``counts`` is an optional tally the namer writes its own accounting into:
    ``"named"`` (topic-naming calls), ``"unnamed"`` (of those, the ones whose
    keyword line the prompt regex missed, so the canvas gets a literal
    ``"unnamed"`` sign) and ``"disambiguation"`` (duplicate-name passes).  It
    is an out-parameter rather than an attribute on the returned wrapper so
    nothing here has to reach into Toponymy's ``LLMWrapper`` layout; the
    ``"unnamed"`` rate is what :func:`_fit_topic_layers` logs beside the
    suppressed-warning count.
    """
    from toponymy.llm_wrappers import LLMWrapper  # noqa: PLC0415

    tally = counts if counts is not None else Counter()

    class KeyphraseNamer(LLMWrapper):  # type: ignore[misc]
        @property
        def supports_system_prompts(self) -> bool:
            return False

        def _respond(self, prompt: str) -> str:
            if "new_topic_name_mapping" in prompt:
                tally["disambiguation"] += 1
                return json.dumps(_keyphrase_disambiguation(prompt))
            if on_name is not None:
                on_name()
            name = _keyphrase_topic_name(prompt)
            tally["named"] += 1
            if name == _UNNAMED:
                tally["unnamed"] += 1
            return json.dumps({"topic_name": name, "topic_specificity": 0.5})

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            return self._respond(prompt)

        def _call_llm_with_system_prompt(
            self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
        ) -> str:
            return self._respond(user_prompt)

    return KeyphraseNamer()


#: Matches the *filename* a warning was raised from when it came out of the
#: ``toponymy`` package (``.../site-packages/toponymy/naming.py``, or the
#: package ``__init__`` itself).  ``showwarning`` is handed a filename, not a
#: module name, so this is how the shim in
#: :func:`_counted_toponymy_warnings` tells a library warning apart from one
#: it must pass through untouched.
_TOPONYMY_WARNING_FILE_RE = re.compile(r"(^|/)toponymy(/|\.py$)")


@contextmanager
def _counted_toponymy_warnings() -> Iterator[Counter[str]]:
    """Swallow Toponymy's warnings for the duration, tallying them by message.

    Toponymy narrates naming hiccups through ``warnings.warn`` (naming
    fallbacks, disambiguation retries).  Those must stay off the CLI — a
    per-topic flood was the actual complaint in issue #2558 — but *discarding*
    them, as a blind ``filterwarnings("ignore")`` did, leaves nothing anywhere
    able to tell zero warnings from a flood.  Since the flood's root cause
    (the ``KeyphraseNamer`` prompt parse, issue #2567) is fixed by code that
    a library bump could silently invalidate, the count is the only signal a
    regression would ever produce: the retry path costs three
    ``wait_random_exponential(4, 10)`` sleeps per colliding cluster and says
    nothing while it burns them.

    So: promote toponymy-origin warnings to ``"always"`` (defeating any
    ambient dedup, so a flood counts as a flood) and route them into a
    ``showwarning`` shim that tallies and drops them.  Every *other* module's
    warnings keep the ambient filters and are forwarded to the real
    ``showwarning`` untouched — the one guarantee a blanket
    ``catch_warnings(record=True)`` could not make.

    Yields the tally, keyed ``"{Category}: {message}"``; it is filled in by
    the time the block exits.  Like any ``catch_warnings`` use this mutates
    process-global state for the duration (the caller is the build thread).
    """
    counts: Counter[str] = Counter()
    with warnings.catch_warnings():
        forward = warnings.showwarning

        def _tally(
            message: Any,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: Any = None,
            line: str | None = None,
        ) -> None:
            if _TOPONYMY_WARNING_FILE_RE.search(str(filename).replace("\\", "/")):
                counts[f"{getattr(category, '__name__', category)}: {message}"] += 1
                return
            forward(message, category, filename, lineno, file, line)

        warnings.filterwarnings("always", module=r"toponymy(\..*)?")
        warnings.showwarning = _tally
        yield counts


def _log_fit_diagnostics(suppressed: Counter[str], naming: Counter[str]) -> None:
    """Put one line in the record for what the fit swallowed.

    ``warning`` level when anything is off-nominal (a suppressed warning, or a
    topic the prompt regex could not name), ``debug`` when the fit was clean —
    so the healthy case stays silent on the CLI while a regression announces
    itself instead of only showing up as an unexplained slow build.  The
    counts also ride on the record as attributes, which is what the slow smoke
    test asserts against.

    The disambiguation-pass count is there to keep a zero warning count
    honest: the collision path is the one #2567 fixed, so "no warnings" only
    means "the parse still works" on a fit that actually took it.
    """
    total = int(sum(suppressed.values()))
    named, unnamed = int(naming["named"]), int(naming["unnamed"])
    disambiguations = int(naming["disambiguation"])
    breakdown = "; ".join(f"{count}x {message}" for message, count in suppressed.most_common(5)) or "none"
    logger.log(
        logging.WARNING if (total or unnamed) else logging.DEBUG,
        "toponymy fit: %d suppressed warning(s) [%s]; %d/%d topics fell back to %r; %d disambiguation pass(es)",
        total,
        breakdown,
        unnamed,
        named,
        _UNNAMED,
        disambiguations,
        extra={
            "toponymy_suppressed_warnings": total,
            "toponymy_named_topics": named,
            "toponymy_unnamed_topics": unnamed,
            "toponymy_disambiguations": disambiguations,
        },
    )


def _base_min_cluster_size(n: int) -> int:
    """Scale Toponymy's base min-cluster-size with corpus size.

    Topic count ≈ ``n / base_min_cluster_size`` and naming costs one namer
    call per topic, so the library default of 10 explodes past ~20k items
    (the audio study's cost model: 50k clips → ~5,000 calls).  ``n/300``
    tracks the study's recommendation of ~50–100 for the 20k regime while
    leaving small datasets on the library default.
    """
    return max(10, round(n / 300))


def _fit_topic_layers(
    texts: list[str],
    embedding_vectors: np.ndarray,
    clusterable_vectors: np.ndarray,
    text_encoder: Any,
    *,
    object_description: str,
    corpus_description: str,
    on_progress: ProgressFn | None = None,
) -> list[tuple[list[str], np.ndarray]]:
    """Run the Toponymy fit; return ``(topic_names, cluster_labels)`` per layer.

    Layers come back in Toponymy's order — index 0 is the **finest** layer —
    and each ``cluster_labels`` array assigns every input row a topic index
    (or ``-1`` for noise).  Isolated as the seam tests stub: everything above
    is our glue, everything inside (including the namer, so stubbed callers
    never import toponymy) is the library's responsibility.

    Progress: naming is the long pole (one namer call per topic across every
    layer), so we turn it into a *determinate* bar.  A clusterer subclass
    records each layer's topic count the moment Toponymy finishes clustering
    (finest layer first), giving us the total; the no-LLM namer then ticks the
    bar once per topic, tagged with the 0-based layer it's naming — the
    ``Layer 0 / Layer 1 / …`` breakdown the library used to dump to stdout,
    now surfaced through the UI instead.
    """
    from toponymy import Toponymy  # noqa: PLC0415
    from toponymy.clustering import ToponymyClusterer  # noqa: PLC0415

    # ``named`` counts topic-naming calls so far; ``cumulative`` is the running
    # topic-count boundary per layer (filled in once clustering completes), so
    # ``searchsorted`` maps the current tick to the layer being named.
    naming = {"named": 0, "total": 0, "cumulative": np.empty(0, dtype=np.int64)}
    # The namer's own accounting ("named" / "unnamed" / "disambiguation"),
    # logged beside the suppressed-warning count when the fit returns.
    namer_counts: Counter[str] = Counter()

    def _on_layers(cluster_labels: list[np.ndarray]) -> None:
        sizes = [int(labels.max()) + 1 if labels.size else 0 for labels in cluster_labels]
        naming["cumulative"] = np.cumsum(sizes)
        naming["total"] = int(sum(sizes))

    def _on_name() -> None:
        naming["named"] += 1
        if on_progress is None or not naming["total"]:
            return
        done = min(naming["named"], naming["total"])
        n_layers = len(naming["cumulative"])
        layer = min(int(np.searchsorted(naming["cumulative"], done, side="left")), n_layers - 1)
        on_progress(done, naming["total"], f"Naming regions (layer {layer} of {n_layers - 1})…")

    class _ProgressClusterer(ToponymyClusterer):  # type: ignore[misc]
        """A ``ToponymyClusterer`` that reports its layer sizes as it finds them.

        Wrapping the clusterer (rather than pre-fitting one) keeps Toponymy in
        charge of passing the naming layer_kwargs (prompt template/format,
        exemplar delimiters), so the naming pass is byte-for-byte what an
        unwrapped fit would produce — we only observe the layer counts.
        """

        def fit_predict(self, *args: Any, **kwargs: Any) -> Any:
            layers, tree = super().fit_predict(*args, **kwargs)
            _on_layers([np.asarray(layer.cluster_labels) for layer in layers])
            return layers, tree

    model = Toponymy(
        llm_wrapper=make_keyphrase_namer(_on_name, namer_counts),
        text_embedding_model=text_encoder,
        clusterer=_ProgressClusterer(
            min_clusters=4,
            base_min_cluster_size=_base_min_cluster_size(len(texts)),
            verbose=False,
        ),
        object_description=object_description,
        corpus_description=corpus_description,
        # ``verbose=False`` (not the deprecated ``show_progress_bars=False``) is
        # load-bearing: passing only the legacy flag leaves the library's
        # unified ``verbose`` defaulting to True, which both prints
        # "Layer N found M clusters" to stdout and — because a True ``verbose``
        # overrides a False ``show_progress_bar`` downstream — flashes tqdm bars
        # through exemplar/keyphrase selection (issue: signpost stdout noise).
        # A single ``verbose=False`` silences both and skips the deprecation
        # warnings the legacy flags would emit.
        verbose=False,
    )
    # The KeyphraseNamer above keeps Toponymy's naming warnings from firing at
    # all, but the library is a moving target, so they are suppressed rather
    # than trusted to stay absent (issue #2558) — and counted rather than
    # discarded, so a library bump that re-breaks the prompt parse shows up as
    # a number instead of an unexplained slow build.
    with _counted_toponymy_warnings() as suppressed:
        model.fit(
            texts,
            embedding_vectors.astype(np.float32),
            clusterable_vectors.astype(np.float32),
        )
    _log_fit_diagnostics(suppressed, namer_counts)
    return [
        (list(names), np.asarray(layer.cluster_labels))
        for names, layer in zip(model.topic_names_, model.cluster_layers_)
    ]


def _region_is_covered(members: np.ndarray, adjacent: np.ndarray | None) -> bool:
    """Whether a topic's *members* fall mostly inside a named cluster of *adjacent*.

    *adjacent* is a coarser- or finer-layer per-row topic assignment (``-1`` =
    noise), or ``None`` when there is no such layer (this topic sits at the
    pyramid's edge — the coarsest or finest layer).  Returns ``True`` when a
    majority of *members* are named (topic index ``>= 0``) there, i.e. some
    coarser/finer sign covers this region and the canvas has a name to hand off
    to.  ``None`` or an all-noise adjacent layer means nothing covers the region
    → the sign is terminal on that side and must not fade there.
    """
    if adjacent is None or members.size == 0:
        return False
    named = int(np.count_nonzero(adjacent[members] >= 0))
    return named * 2 >= members.size


def _clusterable_vectors(matrix: np.ndarray, on_progress: ProgressFn | None = None) -> np.ndarray:
    """The dedicated ~5-D cosine UMAP Toponymy clusters on (not the 2-D layout)."""
    import umap  # noqa: PLC0415

    if on_progress is not None:
        on_progress(0, 0, "Clustering regions…")
    # Force an owned, writable copy: *matrix* may be a read-only mmap view
    # (S1's embedding-matrix sidecar, docs/plans/scalability.md), and UMAP
    # doesn't guarantee it never writes to its input in place.
    matrix = np.array(matrix, dtype=np.float32, copy=True, order="C")
    n_components = min(_CLUSTER_DIM, matrix.shape[1], max(2, matrix.shape[0] - 2))
    reducer = umap.UMAP(n_components=n_components, metric="cosine")
    return np.asarray(reducer.fit_transform(matrix), dtype=np.float32)


def build_region_labels(
    proj: "Projection",
    score_matrix: np.ndarray,
    embed_matrix: np.ndarray,
    texts: list[str],
    embedder: Any,
    *,
    object_description: str,
    corpus_description: str,
    on_progress: ProgressFn | None = None,
    source: str = "keyphrase",
) -> RegionLabelSet:
    """Fit Toponymy and flatten its topic tree into signs for *proj*'s layout.

    All three per-item inputs (``score_matrix`` / ``embed_matrix`` rows and
    ``texts``) must align with ``proj.ids``.  Per layer, each topic becomes one
    :class:`RegionLabel`: text = the topic name, anchor = the medoid of the
    topic's members in the frozen 2-D layout, level = the layer's depth mapped
    to a zoom band (coarsest layer at level 0, like the ground-truth demo
    signs), score = member count (the canvas de-clutter tiebreak).

    Returns an id-pinned — possibly empty — set; too-small corpora
    (< ``_MIN_POINTS``) short-circuit to empty rather than fitting a
    degenerate tree.
    """
    empty = make_label_set(proj.projection_id, [])
    n = len(proj.ids)
    if n < _MIN_POINTS or n != len(texts) or score_matrix.shape[0] != n or embed_matrix.shape[0] != n:
        return empty

    clusterable = _clusterable_vectors(score_matrix, on_progress)
    if on_progress is not None:
        on_progress(0, 0, "Naming regions…")

    encoder = EmbedderTextEncoder(embedder, dim=embed_matrix.shape[1])
    layers = _fit_topic_layers(
        texts,
        embed_matrix,
        clusterable,
        encoder,
        object_description=object_description,
        corpus_description=corpus_description,
        on_progress=on_progress,
    )
    if not layers:
        return empty

    coords = np.asarray(proj.coords, dtype=np.float32)
    n_layers = len(layers)
    labels: list[RegionLabel] = []
    for i, (names, cluster_labels) in enumerate(layers):
        level = (n_layers - 1 - i) * _LEVEL_STEP
        # Layers come back finest-first (index 0), so a *finer* sign lives at
        # ``i-1`` and a *coarser* one at ``i+1``.  A topic whose members are
        # noise (unnamed) in an adjacent layer has no sign to hand off to on
        # that side, and the canvas keeps such terminal signs visible there.
        finer = layers[i - 1][1] if i > 0 else None
        coarser = layers[i + 1][1] if i + 1 < n_layers else None
        for topic_idx, name in enumerate(names):
            text = str(name).strip()
            if not text:
                continue
            members = np.nonzero(cluster_labels == topic_idx)[0]
            if members.size == 0:
                continue
            anchor = medoid(coords[members])
            labels.append(
                RegionLabel(
                    level=level,
                    x=float(anchor[0]),
                    y=float(anchor[1]),
                    text=text,
                    score=float(members.size),
                    source=source,
                    has_coarser=_region_is_covered(members, coarser),
                    has_finer=_region_is_covered(members, finer),
                )
            )

    labels.sort(key=lambda lab: (lab.level, -lab.score))
    return make_label_set(proj.projection_id, labels)


__all__ = [
    "EmbedderTextEncoder",
    "build_region_labels",
    "make_keyphrase_namer",
    "require_signposting",
    "signposting_available",
    "toponymy_version",
]
