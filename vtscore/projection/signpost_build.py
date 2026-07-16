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


def _keyphrase_topic_name(prompt: str) -> str:
    """The top keyphrase for a single-topic naming prompt (or ``"unnamed"``)."""
    match = re.search(r"Keywords for this group include:\s*([^\n]+)", prompt)
    name = match.group(1).split(",")[0].strip() if match else ""
    return name or "unnamed"


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


def make_keyphrase_namer() -> Any:
    """The no-LLM namer: answer every naming prompt with the top keyphrase.

    A trivial in-process ``LLMWrapper`` whose "LLM calls" parse the prompt
    Toponymy built and return its first (top ``information_weighted``)
    keyphrase as the topic name — an honest fallback that keeps the library's
    contrastive clustering + keyphrase machinery and skips only the LLM's
    phrasing.  Duplicate-name disambiguation passes echo the old names back
    (a no-LLM namer cannot invent new distinctions).
    """
    from toponymy.llm_wrappers import LLMWrapper  # noqa: PLC0415

    class KeyphraseNamer(LLMWrapper):  # type: ignore[misc]
        @property
        def supports_system_prompts(self) -> bool:
            return False

        def _respond(self, prompt: str) -> str:
            if "new_topic_name_mapping" in prompt:
                return json.dumps(_keyphrase_disambiguation(prompt))
            return json.dumps({"topic_name": _keyphrase_topic_name(prompt), "topic_specificity": 0.5})

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            return self._respond(prompt)

        def _call_llm_with_system_prompt(
            self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
        ) -> str:
            return self._respond(user_prompt)

    return KeyphraseNamer()


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
) -> list[tuple[list[str], np.ndarray]]:
    """Run the Toponymy fit; return ``(topic_names, cluster_labels)`` per layer.

    Layers come back in Toponymy's order — index 0 is the **finest** layer —
    and each ``cluster_labels`` array assigns every input row a topic index
    (or ``-1`` for noise).  Isolated as the seam tests stub: everything above
    is our glue, everything inside (including the namer, so stubbed callers
    never import toponymy) is the library's responsibility.
    """
    import warnings  # noqa: PLC0415

    from toponymy import Toponymy  # noqa: PLC0415
    from toponymy.clustering import ToponymyClusterer  # noqa: PLC0415

    model = Toponymy(
        llm_wrapper=make_keyphrase_namer(),
        text_embedding_model=text_encoder,
        clusterer=ToponymyClusterer(
            min_clusters=4,
            base_min_cluster_size=_base_min_cluster_size(len(texts)),
            show_progress_bar=False,
        ),
        object_description=object_description,
        corpus_description=corpus_description,
        show_progress_bars=False,
    )
    # Toponymy narrates naming hiccups through ``warnings.warn`` (naming
    # fallbacks, disambiguation retries). The KeyphraseNamer above keeps those
    # from firing, but the library is a moving target and its per-topic
    # warnings would flood the CLI (issue #2558) if any slipped through; they
    # are advisory noise we can't act on, so suppress toponymy-origin warnings
    # for the duration of the fit (other libraries' warnings still surface).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module=r"toponymy(\..*)?")
        model.fit(
            texts,
            embedding_vectors.astype(np.float32),
            clusterable_vectors.astype(np.float32),
        )
    return [
        (list(names), np.asarray(layer.cluster_labels))
        for names, layer in zip(model.topic_names_, model.cluster_layers_)
    ]


def _clusterable_vectors(matrix: np.ndarray, on_progress: ProgressFn | None = None) -> np.ndarray:
    """The dedicated ~5-D cosine UMAP Toponymy clusters on (not the 2-D layout)."""
    import umap  # noqa: PLC0415

    if on_progress is not None:
        on_progress(0, 0, "Clustering regions…")
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
    )
    if not layers:
        return empty

    coords = np.asarray(proj.coords, dtype=np.float32)
    n_layers = len(layers)
    labels: list[RegionLabel] = []
    for i, (names, cluster_labels) in enumerate(layers):
        level = (n_layers - 1 - i) * _LEVEL_STEP
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
