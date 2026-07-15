"""Per-media signpost texts — the ``object_to_text`` layer of the sign pipeline.

Toponymy needs *some text per object*: its contrastive (``information_weighted``)
keyphrase extraction mines the texts of **every** object in the fitted set, not
just the sampled exemplars it shows the LLM (see
``docs/plans/vtsbrowse-toponymy.md``).  So the text is a full-corpus,
per-media artifact — expensive for captioner-class providers, embarrassingly
cacheable, and independent of any particular clustering.  We therefore compute
it once and **cache it on the media dict** (``signpost_text`` +
``signpost_text_source``): media dicts are the dataset pickle's payload, so a
text computed during ingest persists with the dataset, and every later browse
or Find→Browse subset re-fit reuses it instead of re-running a model.  Cached
strings are derived *text*, which the No-Persisted-Vectors rule explicitly
allows.

Providers are registered per media type.  Phase 1 ships the no-new-models
tier resolved by the audio/image signpost studies
(``docs/reports/2026-07-12-toponymy-{audio,image}-signposts.html``):

* **audio** — CLAP zero-shot tags against the AudioSet-527 vocabulary
  (template ``"The sound of {}"``, top-5), the audio study's locked default;
* **image** — SigLIP zero-shot tags against the OpenImages-600 vocabulary
  (template ``"a photo of {}"``, top-5), the image study's no-VLM fallback
  (the instructed ~3B VLM captioner default is a planned follow-up and slots
  in as a replacement provider here);
* **text** — the media's own content, truncated (Toponymy's native case).

Zero-shot tagging costs one text-embed of the vocabulary per (embedder,
vocabulary) pair — cached process-scoped, never persisted — plus one matrix
product, so it is effectively free next to the embed pass that precedes it.

The no-new-models tier above is the default.  A media type can additionally
opt into a **generative captioner** (image VLM / audio captioner; see
:mod:`vtscore.projection.signpost_captioners`) via the
``browse_signpost_captioner`` setting: :func:`provider_for` then returns the
captioner wrapped in a :class:`FallbackTextProvider` over the tag provider, so
a missing model or an undecodable item quietly degrades to tags.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from importlib import resources
from typing import Any, Callable, Protocol

import numpy as np

logger = logging.getLogger(__name__)

#: Media-dict field holding the cached signpost text (persisted in the pickle).
TEXT_FIELD = "signpost_text"
#: Media-dict field stamping which provider+embedder produced the cached text;
#: a mismatch with the active provider's signature invalidates the cache.
SOURCE_FIELD = "signpost_text_source"

#: Progress callback shape: ``(current, total, message)``.
ProgressFn = Callable[[int, int, str], None]


class SignpostTextProvider(Protocol):
    """One way of turning a media item into a short text for Toponymy."""

    @property
    def name(self) -> str: ...

    def signature(self, embedder: Any) -> str:
        """Cache key for texts produced by this provider under *embedder*."""
        ...

    def build_texts(
        self,
        ids: list[int],
        medias: dict[int, dict[str, Any]],
        matrix: np.ndarray,
        embedder: Any,
        on_progress: ProgressFn | None = None,
    ) -> dict[int, str]:
        """Return a text per id.  ``matrix`` rows align with ``ids``."""
        ...


@dataclass(frozen=True)
class ZeroShotTagProvider:
    """Top-k vocabulary terms by similarity in a cross-modal embedder's space.

    The embedder's text branch embeds every vocabulary term once (through
    *template*), and each media's tag list is the top-*top_k* terms by dot
    product against its stored media vector — both sides are L2-normalized at
    ingest, so the dot product is cosine similarity.  The contrastive work of
    deciding which tags *distinguish* a cluster is Toponymy's
    ``information_weighted`` keyphrase stage, not ours; this provider only has
    to be honest per item.
    """

    name: str
    vocab_asset: str
    template: str
    top_k: int = 5

    def signature(self, embedder: Any) -> str:
        return f"{self.name}:{getattr(embedder, 'name', '')}"

    def build_texts(
        self,
        ids: list[int],
        medias: dict[int, dict[str, Any]],
        matrix: np.ndarray,
        embedder: Any,
        on_progress: ProgressFn | None = None,
    ) -> dict[int, str]:
        vocab, vocab_vecs = _vocab_vectors(self.vocab_asset, self.template, embedder, on_progress)
        if vocab_vecs.size == 0 or matrix.size == 0:
            return {}
        sims = matrix.astype(np.float32) @ vocab_vecs.T  # (n, v)
        k = min(self.top_k, sims.shape[1])
        top = np.argsort(-sims, axis=1)[:, :k]
        return {mid: ", ".join(vocab[j] for j in top[i]) for i, mid in enumerate(ids)}


@dataclass(frozen=True)
class ContentTextProvider:
    """The media's own text content, truncated — Toponymy's native case."""

    name: str = "content"
    field: str = "content"
    max_chars: int = 2000

    def signature(self, embedder: Any) -> str:
        return self.name

    def build_texts(
        self,
        ids: list[int],
        medias: dict[int, dict[str, Any]],
        matrix: np.ndarray,
        embedder: Any,
        on_progress: ProgressFn | None = None,
    ) -> dict[int, str]:
        out: dict[int, str] = {}
        for mid in ids:
            content = medias.get(mid, {}).get(self.field)
            if isinstance(content, str) and content.strip():
                out[mid] = content.strip()[: self.max_chars]
        return out


@dataclass(frozen=True)
class FallbackTextProvider:
    """A *primary* provider with a *fallback* for whatever it can't produce.

    Used to letter a media type with its generative captioner while keeping the
    zero-shot tag provider as a safety net: if the captioner's model can't load
    at all its ``build_texts`` raises and we caption *everything* from the
    fallback; if it loads but a particular item won't decode, only that item's
    text comes from the fallback.  The composite signature encodes both, so
    turning the captioner on or off (or swapping either side) invalidates the
    cached texts.
    """

    primary: SignpostTextProvider
    fallback: SignpostTextProvider

    @property
    def name(self) -> str:
        return f"{self.primary.name}+{self.fallback.name}"

    def signature(self, embedder: Any) -> str:
        return f"{self.primary.signature(embedder)}|fallback={self.fallback.signature(embedder)}"

    def build_texts(
        self,
        ids: list[int],
        medias: dict[int, dict[str, Any]],
        matrix: np.ndarray,
        embedder: Any,
        on_progress: ProgressFn | None = None,
    ) -> dict[int, str]:
        try:
            produced = self.primary.build_texts(ids, medias, matrix, embedder, on_progress)
        except Exception:
            logger.warning(
                "Signpost captioner %r failed to run; falling back to %r.",
                self.primary.name,
                self.fallback.name,
                exc_info=True,
            )
            produced = {}
        missing = [mid for mid in ids if not produced.get(mid)]
        if missing:
            row_of = {mid: i for i, mid in enumerate(ids)}
            sub = matrix[[row_of[mid] for mid in missing]] if matrix.size else matrix
            filled = self.fallback.build_texts(missing, medias, sub, embedder, on_progress)
            produced = {**filled, **produced}
        return produced


# ---------------------------------------------------------------------------
# Provider registry (per media type)
# ---------------------------------------------------------------------------

#: Base (always-available, no-new-model) providers.
_PROVIDERS: dict[str, SignpostTextProvider] = {
    "audio": ZeroShotTagProvider(
        name="tags:audioset527",
        vocab_asset="audioset527_labels.txt",
        template="The sound of {}",
    ),
    "image": ZeroShotTagProvider(
        name="tags:openimages600",
        vocab_asset="openimages600_labels.txt",
        template="a photo of {}",
    ),
    "text": ContentTextProvider(),
}

#: Generative captioner providers (opt-in per media type via the
#: ``browse_signpost_captioner`` setting).  Populated at import from
#: :mod:`vtscore.projection.signpost_captioners`; the heavy model deps stay
#: lazy inside each provider, so importing the captioners here is cheap.
_CAPTIONERS: dict[str, SignpostTextProvider] = {}


def register_signpost_text_provider(media_type: str, provider: SignpostTextProvider) -> None:
    """Register (or replace) the base signpost text provider for *media_type*."""
    _PROVIDERS[media_type] = provider


def register_signpost_captioner(media_type: str, provider: SignpostTextProvider) -> None:
    """Register (or replace) the generative captioner for *media_type*."""
    _CAPTIONERS[media_type] = provider


def _captioner_enabled(media_type: str) -> bool:
    """Whether the user opted this media type into the generative captioner.

    Reads the ``signpost_captioner`` map off the library-tier
    :class:`~vtscore.config.CoreConfig` (the app populates it from the
    ``browse_signpost_captioner`` setting).  Any failure — no builder
    installed, missing field — reads as "off", so the safe tag default holds.
    """
    try:
        from vtscore.config import CoreConfig  # noqa: PLC0415

        return bool(CoreConfig.from_settings().signpost_captioner.get(media_type, False))
    except Exception:
        return False


def provider_for(media_type: str) -> SignpostTextProvider | None:
    """The active provider for *media_type*, or ``None`` (no signposts).

    Returns the base (tag / content) provider unless the user enabled the
    generative captioner for this media type, in which case the captioner is
    returned wrapped in a :class:`FallbackTextProvider` over the base — so a
    model-load or per-item decode failure quietly degrades to tags.
    """
    base = _PROVIDERS.get(media_type)
    if _captioner_enabled(media_type):
        captioner = _CAPTIONERS.get(media_type)
        if captioner is not None:
            return FallbackTextProvider(captioner, base) if base is not None else captioner
    return base


# ---------------------------------------------------------------------------
# Vocabulary embedding cache (process-scoped; vectors are never persisted)
# ---------------------------------------------------------------------------

_vocab_cache: dict[tuple[str, str, str], tuple[list[str], np.ndarray]] = {}
_vocab_lock = threading.Lock()


def _load_vocab(asset: str) -> list[str]:
    text = (resources.files("vtscore.projection") / "assets" / asset).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _vocab_vectors(
    asset: str,
    template: str,
    embedder: Any,
    on_progress: ProgressFn | None = None,
) -> tuple[list[str], np.ndarray]:
    """Embed a vocabulary through *embedder*'s text branch, cached per process.

    The cache key includes the embedder name so multi-embedder processes keep
    one entry per space.  Terms whose embedding fails are dropped from both the
    returned term list and the matrix, keeping the two aligned.
    """
    key = (asset, template, getattr(embedder, "name", ""))
    with _vocab_lock:
        cached = _vocab_cache.get(key)
    if cached is not None:
        return cached

    terms = _load_vocab(asset)
    kept: list[str] = []
    vecs: list[np.ndarray] = []
    for i, term in enumerate(terms):
        if on_progress is not None and i % 50 == 0:
            on_progress(i, len(terms), "Preparing signpost vocabulary…")
        try:
            vec = embedder.embed_text(template.format(term))
        except Exception:
            vec = None
        if vec is None:
            continue
        kept.append(term)
        vecs.append(np.asarray(vec, dtype=np.float32))
    matrix = np.stack(vecs) if vecs else np.empty((0, 0), dtype=np.float32)
    with _vocab_lock:
        _vocab_cache[key] = (kept, matrix)
    return kept, matrix


# ---------------------------------------------------------------------------
# The cache-aware entry point
# ---------------------------------------------------------------------------


def ensure_signpost_texts(
    medias: dict[int, dict[str, Any]],
    ids: list[int],
    matrix: np.ndarray,
    embedder: Any,
    on_progress: ProgressFn | None = None,
) -> dict[int, str] | None:
    """Return a signpost text per id, computing and caching what's missing.

    Texts cached on the media dicts under a matching provider signature are
    reused verbatim; only the misses are computed (all of them on the first
    run, none on a Find→Browse re-fit of an ingest-prepped dataset).  Newly
    computed texts are stamped back onto the media dicts, so a text computed
    before the dataset pickle is written persists with the dataset.

    Returns ``None`` when no provider is registered for the medias' type, or
    when the provider yields no usable text at all.  ``matrix`` rows must
    align with ``ids`` (the same contract as the embedding submatrix helpers).
    """
    if not ids:
        return None
    first = medias.get(ids[0]) or {}
    provider = provider_for(first.get("media_type", ""))
    if provider is None:
        return None
    signature = provider.signature(embedder)

    texts: dict[int, str] = {}
    missing: list[int] = []
    for mid in ids:
        media = medias.get(mid)
        if media is None:
            continue
        cached = media.get(TEXT_FIELD)
        if isinstance(cached, str) and cached and media.get(SOURCE_FIELD) == signature:
            texts[mid] = cached
        else:
            missing.append(mid)

    if missing:
        row_of = {mid: i for i, mid in enumerate(ids)}
        sub = matrix[[row_of[mid] for mid in missing]] if matrix.size else matrix
        built = provider.build_texts(missing, medias, sub, embedder, on_progress)
        for mid, text in built.items():
            media = medias.get(mid)
            if media is None or not text:
                continue
            media[TEXT_FIELD] = text
            media[SOURCE_FIELD] = signature
            texts[mid] = text

    return texts or None


def texts_signature(medias: dict[int, dict[str, Any]], embedder: Any) -> str | None:
    """The active provider signature for *medias*, or ``None`` when unsupported."""
    if not medias:
        return None
    first = next(iter(medias.values()))
    provider = provider_for(first.get("media_type", ""))
    if provider is None:
        return None
    return provider.signature(embedder)


def _register_default_captioners() -> None:
    """Populate ``_CAPTIONERS`` with the bundled image/audio captioners.

    Import-time and cheap: the captioner module keeps torch/transformers lazy,
    so this only wires up the (opt-in) instances — nothing loads a model until
    a captioner-enabled build actually runs.
    """
    from vtscore.projection.signpost_captioners import AUDIO_CAPTIONER, IMAGE_CAPTIONER  # noqa: PLC0415

    _CAPTIONERS.setdefault("image", IMAGE_CAPTIONER)
    _CAPTIONERS.setdefault("audio", AUDIO_CAPTIONER)


_register_default_captioners()


__all__ = [
    "ContentTextProvider",
    "FallbackTextProvider",
    "SignpostTextProvider",
    "TEXT_FIELD",
    "SOURCE_FIELD",
    "ZeroShotTagProvider",
    "ensure_signpost_texts",
    "provider_for",
    "register_signpost_captioner",
    "register_signpost_text_provider",
    "texts_signature",
]
