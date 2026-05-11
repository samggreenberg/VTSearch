"""Media embedder ABC and shared model-loading helpers."""

from __future__ import annotations

import contextlib
import io
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np

from vtsearch.media.base import ProgressCallback, _noop_progress

__all__ = [
    "MediaEmbedder",
    "embedder_load_setup",
    "extract_tensor",
    "intercept_tqdm_progress",
    "intercept_weight_loading_progress",
    "load_pretrained_local_first",
    "media_from_path",
    "timed_progress",
]


def media_from_path(file_path: Any, origin: dict | None = None) -> dict:
    """Build a minimal media dict suitable for :meth:`MediaEmbedder.embed_media`.

    Convenience helper for callers that only have a local file path (uploaded
    files, converter outputs, seed data, CLI utilities).  File-based embedders
    read ``media["media_path"]``; service-based embedders can also inspect
    *origin* when supplied.
    """
    from pathlib import Path  # noqa: PLC0415

    p = Path(file_path)
    return {
        "media_path": str(p.resolve()),
        "origin": origin,
        "origin_name": p.name,
        "filename": p.name,
        "custom_metadata": None,
    }


# ---------------------------------------------------------------------------
# Shared embedder helpers
# ---------------------------------------------------------------------------


def extract_tensor(output: object):
    """Extract a plain tensor from a model output.

    Depending on the ``transformers`` version, methods like
    ``get_image_features()`` / ``get_text_features()`` /
    ``get_video_features()`` may return either a raw :class:`torch.Tensor`
    or a ``BaseModelOutputWithPooling`` dataclass.  This helper handles
    both cases transparently.
    """
    import torch  # noqa: PLC0415

    if isinstance(output, torch.Tensor):
        return output
    for attr in ("image_embeds", "text_embeds", "video_embeds", "pooler_output"):
        val = getattr(output, attr, None)
        if isinstance(val, torch.Tensor):
            return val
    # Final fallback: treat as tuple-like and return first element
    return output[0]  # type: ignore[index]


@contextlib.contextmanager
def timed_progress(
    on_progress: ProgressCallback,
    status: str,
    message: str,
    current: int = 0,
    total: int = 0,
) -> Any:
    """Show elapsed time in the progress message while a block executes.

    Wraps a long-running blocking operation (typically a heavy ``import``)
    so that the progress callback is updated every second with an elapsed-
    time suffix, e.g. ``"Importing torch… (3s)"``.  This prevents the UI
    from appearing frozen during operations that cannot report incremental
    progress themselves.

    The initial progress update is sent immediately (without a time suffix).
    After the first second the background ticker appends ``(1s)``, ``(2s)``,
    etc. until the ``with`` block exits.
    """
    stop = threading.Event()

    def _ticker() -> None:
        start = time.monotonic()
        while not stop.wait(timeout=1.0):
            elapsed = int(time.monotonic() - start)
            on_progress(status, f"{message} ({elapsed}s)", current, total)

    on_progress(status, message, current, total)
    t = threading.Thread(target=_ticker, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=2)


def embedder_load_setup(on_progress: ProgressCallback, message: str) -> str:
    """Common setup ceremony shared by all embedder ``_load_models_impl()`` methods.

    1. Calls :func:`ensure_torch_configured`.
    2. Runs ``gc.collect()`` to free memory before loading a large model.
    3. Reports initial progress via *on_progress*.
    4. Returns the model cache directory as a string.
    """
    import gc  # noqa: PLC0415

    from vtsearch.config import MODELS_CACHE_DIR  # noqa: PLC0415
    from vtsearch.models.loader import ensure_torch_configured  # noqa: PLC0415

    ensure_torch_configured()
    gc.collect()
    on_progress("loading", message, 0, 0)
    return str(MODELS_CACHE_DIR)


_log = logging.getLogger(__name__)

# Retry settings for transient HuggingFace Hub HTTP errors.
_HF_RETRY_COUNT = 3
_HF_RETRY_BACKOFF_BASE = 2  # seconds; delays will be 2, 4, 8, …


def _is_transient_hf_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a retryable HuggingFace Hub HTTP error."""
    cls_names = {type(exc).__name__} | {c.__name__ for c in type(exc).__mro__}
    if "HfHubHTTPError" in cls_names or "HTTPStatusError" in cls_names:
        msg = str(exc)
        if any(f"{code}" in msg for code in range(500, 600)):
            return True
        lower = msg.lower()
        if any(kw in lower for kw in ("timeout", "timed out", "connection reset", "connection aborted")):
            return True
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return False


def load_pretrained_local_first(load_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call *load_fn* preferring cached model files, falling back to download.

    HuggingFace ``from_pretrained`` (and ``SentenceTransformer()``) contact the
    Hub to check for updates **even when model files are already cached**.  If
    the network is unreachable or slow this HTTP request hangs indefinitely,
    making the UI appear frozen on "Loading … model weights".

    This helper tries ``local_files_only=True`` first.  If that raises
    ``OSError`` (model not yet cached), it retries without the flag so the
    model can be downloaded normally.  Transient HTTP errors from the
    HuggingFace Hub (5xx, timeouts) are retried up to ``_HF_RETRY_COUNT``
    times with exponential backoff.
    """
    try:
        return load_fn(*args, local_files_only=True, **kwargs)
    except (OSError, TypeError, ValueError):
        last_exc: Exception | None = None
        for attempt in range(_HF_RETRY_COUNT):
            try:
                return load_fn(*args, **kwargs)
            except Exception as exc:
                if not _is_transient_hf_error(exc):
                    raise
                last_exc = exc
                delay = _HF_RETRY_BACKOFF_BASE * (2**attempt)
                _log.warning(
                    "Transient HuggingFace Hub error (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1,
                    _HF_RETRY_COUNT,
                    delay,
                    exc,
                )
                time.sleep(delay)
        raise last_exc  # type: ignore[misc]


@contextlib.contextmanager
def intercept_tqdm_progress(callback: ProgressCallback) -> Any:
    """Temporarily hook tqdm progress bars to forward updates to *callback*.

    HuggingFace ``transformers`` and ``huggingface_hub`` use :mod:`tqdm` for
    download and weight-loading progress bars.  Those bars write to stderr,
    which the GUI never sees.  This context manager monkey-patches the base
    ``tqdm.std.tqdm`` class so that every ``update()`` call also pushes
    ``(status, message, current, total)`` to *callback*.

    All tqdm subclasses (``tqdm.auto.tqdm``, ``huggingface_hub.utils.tqdm``,
    etc.) resolve ``update`` through MRO to ``tqdm.std.tqdm.update``, so a
    single patch covers the entire hierarchy.

    Only bars with a known *total* are forwarded; indeterminate spinners are
    silently ignored.
    """
    import tqdm.std

    _orig_init = tqdm.std.tqdm.__init__
    _orig_update = tqdm.std.tqdm.update
    _orig_close = tqdm.std.tqdm.close

    _bars: list[tqdm.std.tqdm] = []

    def _primary_bar() -> tqdm.std.tqdm | None:
        if not _bars:
            return None
        return max(_bars, key=lambda b: getattr(b, "total", 0) or 0)

    def _report(bar: tqdm.std.tqdm) -> None:
        total = getattr(bar, "total", None)
        if not total or total <= 0:
            return
        current = int(getattr(bar, "n", 0))
        desc = (getattr(bar, "desc", "") or "Loading…").rstrip(": ")
        callback("loading", desc, current, int(total))

    _sink = io.StringIO()

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs["file"] = _sink
        _orig_init(self, *args, **kwargs)
        total = getattr(self, "total", None)
        if total and total > 0 and not getattr(self, "disable", False):
            _bars.append(self)
            if _primary_bar() is self:
                _report(self)

    def _patched_update(self: Any, n: int = 1) -> None:
        _orig_update(self, n)
        if _primary_bar() is self:
            _report(self)

    def _patched_close(self: Any) -> None:
        _orig_close(self)
        if self in _bars:
            _bars.remove(self)

    tqdm.std.tqdm.__init__ = _patched_init  # type: ignore[assignment]
    tqdm.std.tqdm.update = _patched_update  # type: ignore[assignment]
    tqdm.std.tqdm.close = _patched_close  # type: ignore[assignment]
    try:
        yield
    finally:
        tqdm.std.tqdm.__init__ = _orig_init  # type: ignore[assignment]
        tqdm.std.tqdm.update = _orig_update  # type: ignore[assignment]
        tqdm.std.tqdm.close = _orig_close  # type: ignore[assignment]


@contextlib.contextmanager
def intercept_weight_loading_progress(callback: ProgressCallback, label: str = "Loading model weights…") -> Any:
    """Track tensor-level progress during model weight loading.

    HuggingFace ``transformers`` with ``low_cpu_mem_usage=True`` dispatches
    tensors one-by-one via ``set_module_tensor_to_device`` from ``accelerate``.
    PyTorch's ``load_state_dict`` (used by ``sentence-transformers``) loads
    tensors via ``__getitem__`` on the state dict.

    This context manager monkey-patches both paths to count tensor operations
    and report ``(current, total)`` progress via *callback*.  The total is
    discovered by also intercepting ``safetensors.torch.load_file`` and
    ``torch.load`` to count keys in loaded state dicts.
    """
    _counter = [0]
    _total = [0]
    _patches: list[tuple] = []

    def _report() -> None:
        if _total[0] > 0:
            callback("loading", label, min(_counter[0], _total[0]), _total[0])

    # --- Intercept safetensors.torch.load_file to learn total tensor count ---
    try:
        import safetensors.torch as _st  # noqa: PLC0415

        _orig_lf = _st.load_file

        def _tracked_lf(*a: Any, **kw: Any) -> Any:
            r = _orig_lf(*a, **kw)
            _total[0] += len(r)
            return r

        _st.load_file = _tracked_lf
        _patches.append((_st, "load_file", _orig_lf))
    except ImportError:
        pass

    # --- Intercept torch.load for .bin weight files ---
    try:
        import torch as _torch  # noqa: PLC0415

        _orig_tl = _torch.load

        def _tracked_tl(*a: Any, **kw: Any) -> Any:
            r = _orig_tl(*a, **kw)
            if isinstance(r, dict) and r:
                sample = next(iter(r.values()))
                if isinstance(sample, _torch.Tensor):
                    _total[0] += len(r)
            return r

        _torch.load = _tracked_tl
        _patches.append((_torch, "load", _orig_tl))
    except ImportError:
        pass

    # --- Intercept set_module_tensor_to_device (HF with low_cpu_mem_usage) ---
    try:
        import transformers.modeling_utils as _tm  # noqa: PLC0415

        _orig_smttd = _tm.set_module_tensor_to_device

        def _tracked_smttd(*a: Any, **kw: Any) -> Any:
            r = _orig_smttd(*a, **kw)
            _counter[0] += 1
            _report()
            return r

        _tm.set_module_tensor_to_device = _tracked_smttd
        _patches.append((_tm, "set_module_tensor_to_device", _orig_smttd))
    except (ImportError, AttributeError):
        pass

    # --- Intercept Module.load_state_dict (PyTorch / SentenceTransformers) ---
    try:
        import torch.nn as _nn  # noqa: PLC0415

        _orig_lsd = _nn.Module.load_state_dict

        class _CountingStateDict(dict):
            """Dict wrapper that counts unique key accesses for progress."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._seen: set = set()

            def __getitem__(self, key: Any) -> Any:
                val = super().__getitem__(key)
                if key not in self._seen:
                    self._seen.add(key)
                    _counter[0] += 1
                    _report()
                return val

        def _tracked_lsd(self_model: Any, state_dict: Any, *a: Any, **kw: Any) -> Any:
            if isinstance(state_dict, dict) and not isinstance(state_dict, _CountingStateDict):
                if _total[0] == 0:
                    _total[0] = len(state_dict)
                state_dict = _CountingStateDict(state_dict)
            return _orig_lsd(self_model, state_dict, *a, **kw)

        _nn.Module.load_state_dict = _tracked_lsd  # type: ignore[assignment]
        _patches.append((_nn.Module, "load_state_dict", _orig_lsd))
    except ImportError:
        pass

    try:
        yield
    finally:
        for obj, attr, orig in _patches:
            setattr(obj, attr, orig)


class MediaEmbedder(ABC):
    """Abstract base class for media embedders.

    A *media embedder* takes a media file (or a text description) and produces
    a fixed-size vector embedding.  Each embedder is associated with exactly one
    :class:`MediaType` (via :attr:`media_type_id`), but a single media type may
    have multiple embedders (e.g. different CLIP variants for images).

    Subclasses must implement:

    * :attr:`name` — unique human-readable identifier (also used as the
      registry key).
    * :attr:`media_type_id` — which media type this embedder works with.
    * :meth:`load_models` — load (and cache) the embedding model.
    * :meth:`embed_media` — embed a media file from disk.
    * :meth:`embed_text` — embed a text query in the same vector space.
    """

    _model_load_lock: threading.Lock

    # Global lock that serialises all ``embed_media`` calls across every
    # embedder type.
    _embed_lock = threading.Lock()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._model_load_lock = threading.Lock()

    def __init__(self) -> None:
        self._on_progress: ProgressCallback = _noop_progress

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this embedder, e.g. ``"clap"``, ``"siglip"``."""

    @property
    @abstractmethod
    def media_type_id(self) -> str:
        """The ``type_id`` of the media type this embedder works with."""

    @property
    def is_default(self) -> bool:
        """Whether this embedder is the default for its media type.

        Exactly one embedder per media type should override this to ``True``.
        :func:`vtsearch.media.embedders_for_type` returns defaults first so
        callers using ``embedders_for_type(t)[0]`` get the default.
        """
        return False

    @property
    def supports_text(self) -> bool:
        """Whether this embedder can embed text queries into the same vector space.

        Cross-modal embedders (CLIP, SigLIP, CLAP, X-CLIP) return ``True`` so
        features like text search and description-enrichment are offered.
        Vision-only or patch-based encoders (DINOv3, Perception Encoder) return
        ``False`` — :meth:`embed_text` will not produce meaningful vectors and
        the UI should hide text-search affordances for datasets using them.
        """
        return True

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load (and cache) the embedding model.

        Called lazily the first time this embedder needs to produce a vector.
        Implementations must be idempotent — a second call should be a no-op.

        Subclasses should override :meth:`_load_models_impl` (not this method).
        This wrapper catches :class:`ImportError` and re-raises with an
        actionable message so that missing dependencies surface clearly.

        A per-class lock serialises concurrent callers so that only one
        thread performs the actual load; others wait and then return
        immediately (the subclass ``_load_models_impl`` checks
        ``self._model is not None``).
        """
        if getattr(self, "_model", None) is not None:
            return
        with self._model_load_lock:
            try:
                self._load_models_impl()
            except ImportError as exc:
                raise ImportError(
                    f"{exc} — required by the '{self.name}' embedder. "
                    f"Install dependencies with: pip install -e '.[cpu,dev]'"
                ) from exc

    @abstractmethod
    def _load_models_impl(self) -> None:
        """Subclass hook: load the embedding model.

        Override this instead of :meth:`load_models`.
        """

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_media(self, media: dict) -> Optional[np.ndarray]:
        """Return a fixed-size embedding vector for *media*.

        *media* is a media dict (the same shape produced by the dataset
        loader).  File-based embedders pull ``Path(media["media_path"])``;
        service-based embedders can use ``media["origin"]``,
        ``media["origin_name"]``, ``media.get("custom_metadata")`` etc. to
        look the content up remotely without touching disk.

        Acquires :attr:`_embed_lock` so that only one forward pass runs at a
        time across all embedder types.  Subclasses must override
        :meth:`_embed_media_impl` (not this method).

        Returns ``None`` if the media cannot be embedded.
        """
        with self._embed_lock:
            return self._embed_media_impl(media)

    @abstractmethod
    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        """Subclass hook: embed a single media item.

        Override this instead of :meth:`embed_media`.
        """

    # ------------------------------------------------------------------
    # Bulk embedding
    # ------------------------------------------------------------------

    def embed_media_bulk(self, medias: list[dict]) -> list[Optional[np.ndarray]]:
        """Embed every item in *medias* and return a same-length list of vectors.

        Positions where an item could not be embedded contain ``None``.

        The default implementation dispatches to :meth:`embed_media` per
        item — each call acquires :attr:`_embed_lock` individually so
        concurrent callers can interleave — and emits per-item progress
        via :attr:`_on_progress` so long runs stay visible in the UI.

        Subclasses backed by a service that natively accepts many items
        per request should override :meth:`_embed_media_bulk_impl`.  If
        they chunk internally (batching), they are responsible for
        emitting their own progress updates through :attr:`_on_progress`.
        """
        if not medias:
            return []
        return self._embed_media_bulk_impl(medias)

    def _embed_media_bulk_impl(self, medias: list[dict]) -> list[Optional[np.ndarray]]:
        """Subclass hook: embed a list of media items.

        Default: loop over :meth:`embed_media`, emitting per-item progress.
        Override to replace the per-item loop with a single bulk request,
        or to batch internally in chunks sized for a remote API.
        """
        total = len(medias)
        results: list[Optional[np.ndarray]] = []
        for i, m in enumerate(medias):
            self._on_progress("embedding", f"Embedding {i + 1}/{total}...", i + 1, total)
            results.append(self.embed_media(m))
        return results

    def embed_medias(self, medias: dict[int, dict]) -> dict[int, Optional[np.ndarray]]:
        """Bulk-embed an id→media dict; return id→vector (or ``None``) dict.

        Convenience wrapper around :meth:`embed_media_bulk` for callers that
        already have medias keyed by ID — typically dataset importers that
        have built the medias dict before embedding.  IDs whose embedding
        failed map to ``None`` in the returned dict, mirroring the position-
        based ``None`` contract of :meth:`embed_media_bulk`.
        """
        if not medias:
            return {}
        keys = list(medias.keys())
        values = [medias[k] for k in keys]
        vectors = self.embed_media_bulk(values)
        return dict(zip(keys, vectors))

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Return an embedding of *text* in the **same vector space** as :meth:`embed_media`.

        The default implementation returns ``None`` (text sorting unavailable).
        """
        return None

    @property
    def description_wrappers(self) -> list[str]:
        """Wrapper templates for enriching sort descriptions.

        Each template is a format string containing ``{text}``.  Override in
        subclasses to provide media-specific wrappers that improve embedding quality.
        """
        return []

    def embed_text_enriched(self, text: str) -> Optional[np.ndarray]:
        """Embed *text* using the average over all description wrappers.

        Falls back to :meth:`embed_text` if no wrappers are defined or all fail.
        """
        wrappers = self.description_wrappers
        if not wrappers:
            return self.embed_text(text)

        embeddings = []
        for wrapper in wrappers:
            wrapped = wrapper.format(text=text)
            vec = self.embed_text(wrapped)
            if vec is not None:
                embeddings.append(vec)

        if not embeddings:
            return self.embed_text(text)

        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        return avg

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable summary of this embedder."""
        return {
            "name": self.name,
            "media_type_id": self.media_type_id,
            "supports_text": self.supports_text,
        }
