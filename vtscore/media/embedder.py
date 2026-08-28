"""Media embedder ABC and shared model-loading helpers."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from vtscore.media.base import ProgressCallback, _noop_progress

if TYPE_CHECKING:
    from vtscore.media.patch_embed import PatchEmbedOutput
    from vtscore.media.structural import StructuralFeatures

__all__ = [
    "DEFAULT_EMBED_BATCH_SIZE",
    "IMPORT_MODULE_ESTIMATES",
    "MediaEmbedder",
    "embed_autocast",
    "embedder_load_setup",
    "extract_tensor",
    "intercept_tqdm_progress",
    "intercept_weight_loading_progress",
    "load_pretrained_local_first",
    "media_from_path",
    "resolve_embed_batch_size",
    "timed_progress",
    "to_compute_device",
    "to_float32",
    "to_model_inputs",
]


DEFAULT_EMBED_BATCH_SIZE = 32

#: Rough module-count estimates used only to *pace* the "Importing …" progress
#: bars: :func:`timed_progress` divides the live ``sys.modules`` delta by the
#: matching value to fill the bar.  Python reports no advance count of how many
#: submodules an ``import`` will pull in, and the delta is heavily
#: context-dependent — a bare ``import torch`` is ~1000 modules, but
#: ``from transformers import CLIPModel`` *after* torch is ~2600, while the
#: startup logging-bridge import of transformers is only ~170.  So these are
#: deliberately per-call-site approximations, not exact totals: the bar is
#: clamped just below 100 % and only snapped to 100 % once the import actually
#: returns (by the next phase message or ``cb.flush()``), so an off estimate
#: merely paces the bar and can never report a still-running import as finished.
IMPORT_MODULE_ESTIMATES: dict[str, int] = {
    "torch": 1100,
    "torch_hub": 50,
    "torchvision": 1100,
    "transformers": 2700,
    "transformers_logging": 190,
    "sentence_transformers": 3300,
    "sklearn": 1700,
    "soundfile": 160,
}


def resolve_embed_batch_size(default: int = DEFAULT_EMBED_BATCH_SIZE) -> int:
    """Return the configured GPU embed batch size.

    Reads ``VTSEARCH_EMBED_BATCH_SIZE`` from the environment; non-positive
    or unparseable values fall back to *default*.  Subclasses with tighter
    VRAM constraints (e.g. video models with per-clip frame stacks) can
    pass a smaller *default* without touching the env var.
    """
    raw = os.environ.get("VTSEARCH_EMBED_BATCH_SIZE", "").strip()
    if not raw:
        return max(1, default)
    try:
        val = int(raw)
    except ValueError:
        return max(1, default)
    return val if val > 0 else max(1, default)


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


def to_compute_device(model: Any, allow_half: bool = False) -> Any:
    """Move a freshly loaded embedding *model* onto the resolved compute device.

    Replaces the hardcoded ``model.to("cpu")`` every embedder used to run at
    load time.  The target device comes from
    :func:`vtscore.config.resolve_device`, which honours ``VTSEARCH_DEVICE`` and
    smoke-tests CUDA before returning a CUDA device - falling back to ``"cpu"``
    when the installed torch wheel can't actually launch a kernel on the visible
    GPU.  The move is therefore always safe:

    * On a CPU-only host (or one whose GPU the wheel can't drive) this is exactly
      the old ``.to("cpu")``, still materialising any ``meta``-device tensors
      left behind by ``low_cpu_mem_usage=True``.
    * On a working CUDA / MPS host the model lands on the accelerator, and every
      embedder's forward pass follows it automatically: each reads
      ``next(self._model.parameters()).device`` and copies its inputs there,
      pulling results back with ``.detach().cpu().numpy()``.

    Returns the moved model so callers can write ``self._model = to_compute_device(self._model)``.

    Pass ``allow_half=True`` to additionally honour ``VTSEARCH_EMBED_PRECISION``
    by casting the weights (see :func:`vtscore.config.embed_weight_dtype`).  It
    is opt-in per embedder rather than global because the precision measurement
    behind it (#3143) covers the **image** encoders only: the audio, video and
    face backbones share this helper, and casting a model whose numerics nobody
    has measured would be a silent change to what it produces.  Half precision
    is off by default in any case, so an embedder that has not opted in is
    unaffected either way.
    """
    from vtscore.config import embed_weight_dtype, resolve_device  # noqa: PLC0415

    model = model.to(resolve_device())
    if allow_half:
        dtype = embed_weight_dtype()
        if dtype is not None:
            model = model.to(dtype)
    return model


@contextlib.contextmanager
def embed_autocast() -> Any:
    """Wrap an embedding forward in ``torch.autocast`` when so configured.

    A no-op unless ``VTSEARCH_EMBED_PRECISION`` names an ``autocast_*`` mode.
    Unlike a weight cast, autocast keeps fp32 master weights and lets torch
    choose per op, holding the reduction-heavy ones (softmax, layer norm) in
    fp32 — numerically the safer half of the two, and the slower one.

    Both paths still return **fp32** vectors: only the compute is half, never
    the stored embedding (see :func:`to_float32`).  Storing half vectors would
    change every downstream matrix's dtype, which is a different change with a
    different blast radius than the one #3143 measured.
    """
    from vtscore.config import embed_autocast_dtype, resolve_device  # noqa: PLC0415

    dtype = embed_autocast_dtype()
    if dtype is None:
        yield
        return
    import torch  # noqa: PLC0415

    device_type = resolve_device().split(":")[0]
    with torch.autocast(device_type=device_type, dtype=dtype):
        yield


def to_model_inputs(inputs: Any, model: Any) -> dict:
    """Move processor *inputs* onto *model*'s device **and** floating dtype.

    Replaces the ``{k: v.to(device) for ...}`` every embedder wrote by hand.
    The dtype half is what a weight cast needs: fp16 weights fed fp32
    ``pixel_values`` raise ``expected scalar type Half but found Float`` in the
    patch-embedding conv, so the pixels have to follow the weights.

    Only floating tensors are cast.  ``input_ids`` / ``attention_mask`` are
    integer and casting them to half would corrupt token ids outright — quietly,
    for ids above 2048, which is most of a real vocabulary.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    out = {}
    for key, val in inputs.items():
        if hasattr(val, "is_floating_point") and val.is_floating_point():
            out[key] = val.to(device=device, dtype=dtype)
        elif hasattr(val, "to"):
            out[key] = val.to(device)
        else:
            out[key] = val
    return out


def to_float32(tensor: Any) -> Any:
    """Upcast a possibly-half tensor back to fp32 before it leaves the GPU.

    Half precision is a *compute* choice; the embedding contract is fp32.
    Numpy would otherwise carry a ``float16`` array straight into the dataset
    pickles and the embedding matrices, silently halving the precision of
    everything stored rather than only of the forward pass.
    """
    return tensor.float() if hasattr(tensor, "float") else tensor


@contextlib.contextmanager
def timed_progress(
    on_progress: ProgressCallback,
    status: str,
    message: str,
    current: int = 0,
    total: int = 0,
    est_modules: int | None = None,
    tick_interval: float = 1.0,
) -> Any:
    """Show elapsed time in the progress message while a block executes.

    Wraps a long-running blocking operation (typically a heavy ``import``)
    so that the progress callback is updated every second with an elapsed-
    time suffix, e.g. ``"Importing torch… (3s, 247 modules)"``.  This
    prevents the UI from appearing frozen during operations that cannot
    report incremental progress themselves.  The module-count delta
    against ``sys.modules`` at entry gives the user a concrete "still
    making progress" signal when a fresh import is grinding through
    hundreds of submodules; a static elapsed counter doesn't distinguish
    a busy import from a stuck network call.

    The initial progress update is sent immediately (without a time suffix).
    After the first second the background ticker appends the elapsed-time
    and module-count suffix until the ``with`` block exits.

    When *est_modules* is given (a rough count of how many modules the wrapped
    import pulls in, see :data:`IMPORT_MODULE_ESTIMATES`), the bar is *driven*
    by that live module delta: it starts at 0 %, climbs as submodules load, and
    is clamped to one below *est_modules* so it can never read 100 % while the
    import is still running.  Completion is signalled by the surrounding layers
    (the next phase message or ``cb.flush()`` on the console, the next job step
    on the web), so a too-small or too-large estimate only paces the bar — it
    never claims a still-running import has finished.  Without *est_modules* the
    passed *current*/*total* are forwarded unchanged (e.g. a fixed step counter
    or an indeterminate ``0/0`` phase message).

    *tick_interval* is the seconds between ticker updates.  Production call
    sites keep the 1-second default; tests pass a small value so they can
    observe ticks without real multi-second sleeps.
    """
    import sys  # noqa: PLC0415

    stop = threading.Event()
    baseline_modules = len(sys.modules)
    est_total = est_modules if (est_modules is not None and est_modules > 0) else 0
    use_est = est_total > 0
    if use_est:
        # The module-count delta is the progress signal; start the bar empty.
        current, total = 0, est_total

    def _bar_current(loaded: int) -> int:
        if not use_est:
            return current
        # Clamp one below the estimate so a running import never fills the bar;
        # the snap to 100 % happens when the block exits and the next message
        # (or flush) completes the bar.
        return max(0, min(loaded, est_total - 1))

    def _ticker() -> None:
        start = time.monotonic()
        while not stop.wait(timeout=tick_interval):
            elapsed = int(time.monotonic() - start)
            loaded = len(sys.modules) - baseline_modules
            suffix = f"({elapsed}s, {loaded} modules)" if loaded > 0 else f"({elapsed}s)"
            on_progress(status, f"{message} {suffix}", _bar_current(loaded), total)

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
    2. Reports initial progress via *on_progress*.
    3. Returns the model cache directory as a string.
    """
    from vtscore.config import MODELS_CACHE_DIR  # noqa: PLC0415
    from vtscore.media.torch_setup import ensure_torch_configured  # noqa: PLC0415

    ensure_torch_configured()
    on_progress("loading", message, 0, 0)
    return str(MODELS_CACHE_DIR)


_log = logging.getLogger(__name__)

# Retry settings for transient HuggingFace Hub HTTP errors.
_HF_RETRY_COUNT = 3
_HF_RETRY_BACKOFF_BASE = 2  # seconds; delays will be 2, 4, 8, …


def hf_token() -> str | bool:
    """Token for HuggingFace Hub requests, else ``False``.

    Resolution order: an interactive OAuth credential (from "Sign in with
    HuggingFace") takes precedence, then the ``HF_TOKEN`` env var.  Signing in
    therefore unlocks gated model weights (e.g. DINOv3) the same way it unlocks
    gated demo datasets.

    All bundled models are public, so no token is *required*; ``False``
    explicitly tells the HF libraries not to look for (or warn about) a
    missing one.  An authenticated token also matters behind shared egress IPs
    (e.g. clusters where many users NAT through one address): the Hub
    rate-limits anonymous requests per IP and can silently delay the first
    metadata call by minutes, while authenticated requests are limited per
    account.
    """
    from vtscore.security.hf_auth import get_token  # noqa: PLC0415

    return get_token() or os.environ.get("HF_TOKEN") or False


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


def load_pretrained_local_first(
    load_fn: Callable[..., Any],
    *args: Any,
    on_progress: Optional[ProgressCallback] = None,
    **kwargs: Any,
) -> Any:
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

    When *on_progress* is given, the network fallback is preceded by a
    metadata preflight (one tiny ``config.json`` fetch) under a
    :func:`timed_progress` ticker labelled "Contacting HuggingFace Hub…".
    The Hub's first metadata request is where pre-download stalls land (DNS
    trouble, Hub latency, anonymous per-IP rate-limit waits — observed at
    minutes on shared egress IPs), and previously they masqueraded as model
    loading.  The preflight absorbs that wait under an honest, ticking label
    while the subsequent real download keeps its clean tqdm progress.
    *on_progress* is consumed here and never forwarded to *load_fn*.
    """
    try:
        return load_fn(*args, local_files_only=True, **kwargs)
    except (OSError, TypeError, ValueError):
        if on_progress is not None:
            with timed_progress(on_progress, "loading", "Contacting HuggingFace Hub…"):
                _hub_metadata_preflight(args, kwargs)
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


def _hub_metadata_preflight(args: tuple, kwargs: dict) -> None:
    """Best-effort fetch of the repo's ``config.json`` to absorb Hub waits.

    Downloads (and thereby caches) the smallest standard file of the model
    repo so that any per-IP anonymous rate-limit sleep or slow first
    connection happens here, inside the caller's "Contacting HuggingFace
    Hub…" ticker, rather than silently inside ``from_pretrained``.  All
    failures are swallowed: the real ``load_fn`` call that follows performs
    its own requests and produces the user-facing error if the Hub is
    genuinely unreachable.
    """
    repo_id = args[0] if args and isinstance(args[0], str) else None
    if repo_id is None:
        return
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        hf_hub_download(
            repo_id,
            "config.json",
            cache_dir=kwargs.get("cache_dir") or kwargs.get("cache_folder"),
            token=kwargs.get("token"),
        )
    except Exception:
        pass


# ----------------------------------------------------------------------
# Global monkey-patch interception (thread-safe)
# ----------------------------------------------------------------------
#
# Both interception context managers below patch *process-wide* globals: tqdm's
# class methods, ``torch.load``, ``nn.Module.load_state_dict``, and friends.
# Model loading is only serialised per embedder *class* (``_model_load_lock`` is
# created per subclass in ``MediaEmbedder.__init_subclass__``), so two different
# embedders can be inside these context managers at the same time — a background
# preload plus a user-triggered embed, say.
#
# Naive save/patch/restore corrupts the process in that case: the second entrant
# saves the *patched* functions as its "originals", and restoring them on exit
# leaves the globals permanently bound to the first entrant's dead closures, so
# every later tqdm bar in the process is forwarded to a stale progress callback.
#
# ``_InterceptRegistry`` fixes that with reference counting: each patch is
# installed exactly once (when the first session enters) and removed exactly once
# (when the last session leaves), all under one re-entrant module-level lock.
# Patched code then resolves *which* session to report to from the calling
# thread, so concurrent loads keep their progress separate instead of sharing a
# single set of counters.

_patch_lock = threading.RLock()


class _InterceptRegistry:
    """Reference-counted, thread-attributed registry of interception sessions.

    :meth:`enter` and :meth:`exit` return whether the caller is responsible for
    installing / removing the global patches.  :meth:`current` maps the calling
    thread to a session: its own if it has one (nested sessions resolve
    innermost-first), otherwise the sole active session when exactly one is
    active — which keeps events raised on helper threads (e.g. huggingface_hub's
    parallel download workers) attributed the way they always were.  With
    several sessions active such events are genuinely ambiguous, so they are
    dropped rather than misreported into an unrelated load's progress bar.
    """

    def __init__(self) -> None:
        self._by_thread: dict[int, list[Any]] = {}
        self._active: list[Any] = []

    def enter(self, session: Any) -> bool:
        """Register *session*; return ``True`` if patches must be installed."""
        with _patch_lock:
            self._by_thread.setdefault(threading.get_ident(), []).append(session)
            self._active.append(session)
            return len(self._active) == 1

    def exit(self, session: Any) -> bool:
        """Unregister *session*; return ``True`` if patches must be removed."""
        with _patch_lock:
            ident = threading.get_ident()
            stack = self._by_thread.get(ident)
            if stack is not None:
                for i, entry in enumerate(stack):
                    if entry is session:
                        del stack[i]
                        break
                if not stack:
                    del self._by_thread[ident]
            for i, entry in enumerate(self._active):
                if entry is session:
                    del self._active[i]
                    break
            return not self._active

    def current(self) -> Any | None:
        """Session the calling thread's events belong to, if unambiguous."""
        with _patch_lock:
            stack = self._by_thread.get(threading.get_ident())
            if stack:
                return stack[-1]
            if len(self._active) == 1:
                return self._active[0]
            return None


def _remove_patches(patches: list[tuple]) -> None:
    """Restore every ``(obj, attr, original)`` triple and empty *patches*."""
    for obj, attr, orig in patches:
        setattr(obj, attr, orig)
    patches.clear()


class _DiscardingSink(io.StringIO):
    """A :class:`io.StringIO` that throws writes away.

    Intercepted bars are redirected here so they never reach the console.  The
    sink is process-wide and lives for the lifetime of the process, so unlike a
    per-context buffer it must not accumulate the text written to it.
    """

    def write(self, s: str) -> int:  # type: ignore[override]
        return len(s)


_tqdm_sink = _DiscardingSink()
_tqdm_registry = _InterceptRegistry()
_tqdm_patches: list[tuple] = []


class _TqdmSession:
    """One active :func:`intercept_tqdm_progress` block."""

    def __init__(self, callback: ProgressCallback) -> None:
        self.callback = callback
        self.bars: list[Any] = []

    def primary_bar(self) -> Any | None:
        if not self.bars:
            return None
        return max(self.bars, key=lambda b: getattr(b, "total", 0) or 0)

    def report(self, bar: Any) -> None:
        total = getattr(bar, "total", None)
        if not total or total <= 0:
            return
        current = int(getattr(bar, "n", 0))
        desc = (getattr(bar, "desc", "") or "Loading…").rstrip(": ")
        self.callback("loading", desc, current, int(total))


def _install_tqdm_patches() -> None:  # noqa: C901
    """Patch ``tqdm.std.tqdm`` so bars report to the active session."""
    import tqdm.std  # noqa: PLC0415

    orig_init = tqdm.std.tqdm.__init__
    orig_update = tqdm.std.tqdm.update
    orig_close = tqdm.std.tqdm.close

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        # Redirect every bar created while interception is live, even one we
        # cannot attribute, so nothing leaks onto the console.
        session = _tqdm_registry.current()
        kwargs["file"] = _tqdm_sink
        orig_init(self, *args, **kwargs)
        if session is None:
            return
        total = getattr(self, "total", None)
        if total and total > 0 and not getattr(self, "disable", False):
            self._vt_session = session
            session.bars.append(self)
            if session.primary_bar() is self:
                session.report(self)

    def _patched_update(self: Any, n: int = 1) -> None:
        orig_update(self, n)
        session = getattr(self, "_vt_session", None)
        if session is not None and session.primary_bar() is self:
            session.report(self)

    def _patched_close(self: Any) -> None:
        orig_close(self)
        session = getattr(self, "_vt_session", None)
        if session is None:
            return
        self._vt_session = None
        for i, bar in enumerate(session.bars):
            if bar is self:
                del session.bars[i]
                break

    tqdm.std.tqdm.__init__ = _patched_init  # type: ignore[assignment]
    tqdm.std.tqdm.update = _patched_update  # type: ignore[assignment]
    tqdm.std.tqdm.close = _patched_close  # type: ignore[assignment]
    _tqdm_patches.extend(
        [
            (tqdm.std.tqdm, "__init__", orig_init),
            (tqdm.std.tqdm, "update", orig_update),
            (tqdm.std.tqdm, "close", orig_close),
        ]
    )


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

    Nesting and concurrent use are safe: the patch is installed once and removed
    once (see :class:`_InterceptRegistry`), and each bar reports to the session
    that was active on its creating thread.
    """
    session = _TqdmSession(callback)
    with _patch_lock:
        if _tqdm_registry.enter(session):
            _install_tqdm_patches()
    try:
        yield
    finally:
        with _patch_lock:
            for bar in session.bars:
                with contextlib.suppress(Exception):
                    bar._vt_session = None
            session.bars.clear()
            if _tqdm_registry.exit(session):
                _remove_patches(_tqdm_patches)


_weight_registry = _InterceptRegistry()
_weight_patches: list[tuple] = []


class _WeightSession:
    """One active :func:`intercept_weight_loading_progress` block."""

    def __init__(self, callback: ProgressCallback, label: str) -> None:
        self.callback = callback
        self.label = label
        self.counter = 0
        self.total = 0
        self.active = True

    def report(self) -> None:
        if self.active and self.total > 0:
            self.callback("loading", self.label, min(self.counter, self.total), self.total)


class _CountingStateDict(dict):
    """Dict wrapper that counts unique key accesses for a weight session."""

    def __init__(self, source: dict, session: _WeightSession) -> None:
        super().__init__(source)
        self._session = session
        self._seen: set = set()

    def __getitem__(self, key: Any) -> Any:
        val = super().__getitem__(key)
        if key not in self._seen:
            self._seen.add(key)
            self._session.counter += 1
            self._session.report()
        return val


def _patch_safetensors_load_file() -> tuple | None:
    """Wrap ``safetensors.torch.load_file`` to count returned tensors."""
    try:
        import safetensors.torch as _st  # noqa: PLC0415
    except ImportError:
        return None
    orig = _st.load_file

    def tracked(*a: Any, **kw: Any) -> Any:
        r = orig(*a, **kw)
        session = _weight_registry.current()
        if session is not None:
            session.total += len(r)
        return r

    _st.load_file = tracked
    return (_st, "load_file", orig)


def _patch_torch_load() -> tuple | None:
    """Wrap ``torch.load`` to count tensors in returned state dicts (.bin weights)."""
    try:
        import torch as _torch  # noqa: PLC0415
    except ImportError:
        return None
    orig = _torch.load

    def tracked(*a: Any, **kw: Any) -> Any:
        r = orig(*a, **kw)
        session = _weight_registry.current()
        if session is not None and isinstance(r, dict) and r:
            sample = next(iter(r.values()))
            if isinstance(sample, _torch.Tensor):
                session.total += len(r)
        return r

    _torch.load = tracked
    return (_torch, "load", orig)


def _patch_set_module_tensor_to_device() -> tuple | None:
    """Wrap ``set_module_tensor_to_device`` (HF low_cpu_mem_usage path)."""
    try:
        import transformers.modeling_utils as _tm  # noqa: PLC0415

        # pyright: ignore[reportAttributeAccessIssue]; set_module_tensor_to_device
        # is re-exported from accelerate at runtime but isn't in the transformers
        # stubs. The AttributeError catch handles missing-attribute drift.
        orig = _tm.set_module_tensor_to_device  # pyright: ignore[reportAttributeAccessIssue]
    except (ImportError, AttributeError):
        return None

    def tracked(*a: Any, **kw: Any) -> Any:
        r = orig(*a, **kw)
        session = _weight_registry.current()
        if session is not None:
            session.counter += 1
            session.report()
        return r

    _tm.set_module_tensor_to_device = tracked  # pyright: ignore[reportAttributeAccessIssue]
    return (_tm, "set_module_tensor_to_device", orig)


def _patch_load_state_dict() -> tuple | None:
    """Wrap ``nn.Module.load_state_dict`` (PyTorch / SentenceTransformers path)."""
    try:
        import torch.nn as _nn  # noqa: PLC0415
    except ImportError:
        return None
    orig = _nn.Module.load_state_dict

    def tracked(self_model: Any, state_dict: Any, *a: Any, **kw: Any) -> Any:
        session = _weight_registry.current()
        if session is not None and isinstance(state_dict, dict) and not isinstance(state_dict, _CountingStateDict):
            if session.total == 0:
                session.total = len(state_dict)
            state_dict = _CountingStateDict(state_dict, session)
        return orig(self_model, state_dict, *a, **kw)

    _nn.Module.load_state_dict = tracked  # type: ignore[assignment]
    return (_nn.Module, "load_state_dict", orig)


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

    Nesting and concurrent use are safe: the patches are installed once and
    removed once (see :class:`_InterceptRegistry`), and each counted tensor is
    charged to the session active on the calling thread.
    """
    session = _WeightSession(callback, label)
    with _patch_lock:
        if _weight_registry.enter(session):
            for installer in (
                _patch_safetensors_load_file,
                _patch_torch_load,
                _patch_set_module_tensor_to_device,
                _patch_load_state_dict,
            ):
                result = installer()
                if result is not None:
                    _weight_patches.append(result)
    try:
        yield
    finally:
        with _patch_lock:
            session.active = False
            if _weight_registry.exit(session):
                _remove_patches(_weight_patches)


#: Key under which an embedder instance stashes its :class:`_ProgressSlot`.
_PROGRESS_SLOT_KEY = "_progress_slot"


class _ProgressSlot:
    """Per-embedder progress state: a process-wide default + a per-thread override.

    *default* is what :func:`vtscore.media.set_progress_callback` wires in once
    at startup (the host application's progress sink); every thread sees it.
    *local* carries the per-thread override installed by
    :meth:`MediaEmbedder.progress_scope` (or a plain ``emb._on_progress = cb``
    assignment) for the duration of one embed / model-load pass.
    """

    __slots__ = ("default", "local")

    def __init__(self, default: ProgressCallback) -> None:
        self.default = default
        self.local = threading.local()


def _progress_slot(emb: "MediaEmbedder") -> _ProgressSlot:
    """Return *emb*'s :class:`_ProgressSlot`, creating it on first use.

    Created lazily (rather than in ``__init__``) so an embedder subclass that
    never calls ``super().__init__()`` still gets one.  ``dict.setdefault`` is
    atomic under the GIL, so two threads racing to create the slot agree on a
    single winner instead of one silently discarding the other's callback.
    """
    state = vars(emb)
    slot = state.get(_PROGRESS_SLOT_KEY)
    if slot is None:
        slot = state.setdefault(_PROGRESS_SLOT_KEY, _ProgressSlot(_noop_progress))
    return slot


class _ThreadLocalProgress:
    """Data descriptor backing :attr:`MediaEmbedder._on_progress`.

    Embedders are process-wide singletons (``vtscore.media._embedder_registry``),
    so a plain instance attribute made the progress callback shared mutable
    state: two concurrent dataset loads on the same embedder would each assign
    their own tracker callback, and the second assignment re-routed the *first*
    load's still-running ``embed_media_bulk`` into the second load's tracker.
    That mis-drew the progress bar and — because tracker callbacks call
    ``check_cancelled()`` — let cancelling one load raise ``CancelledError``
    inside the other's embed pass, aborting the wrong dataset.  Each pass's
    ``finally`` then restored a callback captured before the other's assignment,
    silencing whichever load was still running.

    Reads and writes are therefore scoped to the calling thread: a write only
    ever redirects the progress of embed / model-load calls made *by that
    thread*, which is exactly what every save-and-restore call site wants.  A
    thread that never assigned anything reads the process-wide default
    (:meth:`MediaEmbedder.set_default_progress_callback`), so background work
    with no explicit callback still reports into the host application's
    progress sink.  That fallback narrates work the sink cannot see the end of,
    so a model load taken through it is terminated explicitly by
    :meth:`MediaEmbedder._orphan_progress`; background warm-ups that want no
    progress surface at all should say so with :meth:`MediaEmbedder.silent_progress`.
    """

    def __get__(self, obj: "MediaEmbedder | None", objtype: type | None = None) -> Any:
        if obj is None:
            return self
        slot = _progress_slot(obj)
        cb = getattr(slot.local, "cb", None)
        return slot.default if cb is None else cb

    def __set__(self, obj: "MediaEmbedder", value: ProgressCallback | None) -> None:
        _progress_slot(obj).local.cb = value


class MediaEmbedder(ABC):
    """Abstract base class for media embedders.

    A *media embedder* takes a media file (or a text description) and produces
    a fixed-size vector embedding.  Each embedder is associated with exactly one
    :class:`MediaType` (via :attr:`media_type_id`), but a single media type may
    have multiple embedders (e.g. different CLIP variants for images).

    Subclasses must implement:

    * :attr:`name`: unique human-readable identifier (also used as the
      registry key).
    * :attr:`media_type_id`: which media type this embedder works with.
    * :meth:`load_models`: load (and cache) the embedding model.
    * :meth:`embed_media`: embed a media file from disk.
    * :meth:`embed_text`: embed a text query in the same vector space.
    """

    _model_load_lock: threading.Lock

    # Global lock that serialises all ``embed_media`` calls across every
    # embedder type.
    _embed_lock = threading.Lock()

    #: Progress sink for this embedder's model loads and bulk passes.  Reads
    #: and writes are **per-thread** over a process-wide default; see
    #: :class:`_ThreadLocalProgress` for why, and prefer :meth:`progress_scope`
    #: over assigning it directly.
    _on_progress: ProgressCallback = _ThreadLocalProgress()  # type: ignore[assignment]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._model_load_lock = threading.Lock()

    def set_default_progress_callback(self, callback: ProgressCallback) -> None:
        """Set the process-wide fallback progress sink for this embedder.

        This is the callback every thread sees when it has not installed a
        :meth:`progress_scope` of its own; :func:`vtscore.media.set_progress_callback`
        calls it once at application startup.  Unlike assigning
        :attr:`_on_progress` (which is thread-scoped by design), this is
        deliberately visible across threads.
        """
        _progress_slot(self).default = callback

    @contextlib.contextmanager
    def progress_scope(self, callback: ProgressCallback):
        """Route this embedder's progress to *callback* for the calling thread.

        Restores whatever the calling thread had installed before (usually
        nothing, meaning the process-wide default) on exit, including on
        exception.  Other threads are unaffected for the whole scope, so two
        concurrent dataset loads sharing this singleton embedder each keep
        their own tracker.
        """
        slot = _progress_slot(self)
        prev = getattr(slot.local, "cb", None)
        slot.local.cb = callback
        try:
            yield
        finally:
            slot.local.cb = prev

    def silent_progress(self):
        """Suppress this embedder's progress for the calling thread.

        Sugar over :meth:`progress_scope` for background warm-ups that have no
        progress surface of their own (the smart-preload threads, the
        post-import embedder warm-up).  Without it those calls fall through to
        the process-wide default sink, which in the app is the dataset-import
        channel — see :meth:`_orphan_progress`.
        """
        return self.progress_scope(_noop_progress)

    @contextlib.contextmanager
    def _orphan_progress(self):
        """Publish a terminal ``idle`` for a model load nobody is watching.

        A thread that installed no :meth:`progress_scope` still reports through
        the process-wide default sink, which the app wires to the global
        ``dataset_progress`` tracker — the SSE ``dataset`` channel.  That sink
        has no idea when the work it is narrating ends, so an unscoped
        ``load_models`` left the channel parked on its last "Loading … processor…"
        message forever: an import that had *succeeded* looked exactly like a
        wedged one, and only a profiler could tell them apart (#3167).

        The load itself is the boundary that knows when the work ends, so it is
        where the terminal state belongs.  For the duration of an unscoped load
        this pins the default sink as the thread's own callback (so a nested
        ``load_models`` doesn't re-arm the same wrapper) and, in a ``finally``,
        sends one ``idle`` tick to say the phase is over.

        A caller that installed a scope owns its own channel and is left alone;
        so is a sink that is already the no-op default.
        """
        slot = _progress_slot(self)
        if getattr(slot.local, "cb", None) is not None or slot.default is _noop_progress:
            yield
            return
        sink = slot.default
        with self.progress_scope(sink):
            try:
                yield
            finally:
                sink("idle", "", 0, 0)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this embedder, e.g. ``"clap"``, ``"siglip"``."""

    @property
    def display_name(self) -> str:
        """Human-readable label for this embedder, shown in pickers.

        Defaults to :attr:`name` so legacy embedders keep working unchanged.
        Subclasses should override to surface a friendlier label (e.g.
        ``"SigLIP (general images)"``) while the raw :attr:`name` stays
        available as a secondary line for power users.
        """
        return self.name

    @property
    def model_id(self) -> Optional[str]:
        """Concrete identifier of the pretrained model this embedder loads.

        Usually the HuggingFace Hub repo id of the checkpoint (e.g.
        ``"google/siglip-base-patch16-224"``).  Unlike :attr:`name` (a VTSearch
        slug) and :attr:`display_name` (a friendly label), this is the *exact*
        model a third party would download to reproduce the embedding space, so
        the portable-detector export surfaces it in the bundle manifest/README to
        make the bundle fully actionable.

        A direct weights URL is acceptable where there is no plain repo id (e.g.
        EUPE, loaded from a ``.pt`` URL via ``torch.hub``).  ``None`` (the
        default) means the embedder has no single downloadable model id worth
        surfacing — the classical SIFT/VLAD structural embedder, or FaceNet whose
        weights ship inside ``facenet-pytorch``.
        """
        return None

    @property
    def embedding_dim(self) -> Optional[int]:
        """Output dimensionality of the vectors this embedder produces.

        Descriptor metadata, not derived from a loaded model: it lets tooling
        (the generated docs inventories, UI hints) report the dimension without
        downloading weights.  ``None`` means the dimension is unknown or
        variable; built-in embedders all declare a concrete value.
        """
        return None

    @property
    @abstractmethod
    def media_type_id(self) -> str:
        """The ``type_id`` of the media type this embedder works with."""

    @property
    def is_default(self) -> bool:
        """Whether this embedder is the default for its media type.

        Exactly one embedder per media type should override this to ``True``.
        :func:`vtscore.media.embedders_for_type` returns defaults first so
        callers using ``embedders_for_type(t)[0]`` get the default.
        """
        return False

    @property
    def eval_only(self) -> bool:
        """Whether this embedder exists for measurement rather than for users.

        An eval-only embedder is a *research arm*: it is registered, resolvable
        by name (:func:`vtscore.media.get_embedder`), and usable by the eval
        harness and the pre-embedded pile, but it is withheld from every
        app-facing listing -- the pickers, the per-media-type default, and the
        serialised inventory the frontend reads.

        The distinction is not cosmetic.  A study arm is chosen because it
        *differs* from the shipped embedder in one controlled way; nothing in
        that choice says it is good, supported, or licensed for the app.  A
        deployment can already hide a plugin (``hidden_plugins``), but that is a
        *setting* someone has to apply -- this is a property of the code, so an
        eval arm cannot reach a picker by a deployment forgetting.

        Resolution by name stays open on purpose: a pile cell embedded with an
        eval-only embedder must still load, or the study could not read its own
        vectors back.

        Deliberately **not** in :meth:`to_dict`. The one serialised listing,
        :func:`vtscore.media.all_embedders_dict`, filters eval-only embedders
        out, so the field could only ever serialise as ``False`` -- a constant
        in the API contract, and eighteen exact-equality assertions to carry it.
        Ask the embedder, not its dict.
        """
        return False

    @property
    def supports_text(self) -> bool:
        """Whether this embedder can embed text queries into the same vector space.

        Cross-modal embedders (CLIP, SigLIP, CLAP, X-CLIP) return ``True`` so
        features like text search and description-enrichment are offered.
        Vision-only or patch-based encoders (DINOv3, EUPE) return ``False`` -
        :meth:`embed_text` will not produce meaningful vectors and the UI
        should hide text-search affordances for datasets using them.
        """
        return True

    @property
    def supports_patch_regions(self) -> bool:
        """Whether this embedder produces patch-level vectors and a region tree.

        Patch-based image encoders (DINOv2, DINOv3, EUPE) return ``True``; the
        dataset loader then asks them for a :class:`PatchEmbedOutput` per image
        and stores a hierarchical region set plus the raw patch grid alongside
        the usual ``media["embeddings"]`` vector.  Single-vector embedders return
        ``False`` and the patch-region pipeline is skipped entirely.
        """
        return False

    @property
    def supports_geometric_verification(self) -> bool:
        """Whether this embedder produces local features for instance matching.

        Structural embedders (SIFT/VLAD, and learned-local-feature variants
        later) return ``True``; the dataset loader then asks them for a
        :class:`~vtscore.media.structural.StructuralFeatures` per image and
        stores it as ``media["local_features"]`` alongside the VLAD
        ``media["embeddings"]`` vector, enabling the geometric re-rank + match-stat
        verification paths.  All other embedders return ``False`` and the
        structural pipeline is skipped entirely.

        The flag is deliberately media-agnostic (not ``supports_*_image_*``)
        so an audio constellation-fingerprint backend can reuse it without an
        interface change.
        """
        return False

    @property
    def embed_batch_size(self) -> int:
        """How many items to forward through the model in one GPU call.

        Default reads :envvar:`VTSEARCH_EMBED_BATCH_SIZE` (falling back to
        :data:`DEFAULT_EMBED_BATCH_SIZE`).  Subclasses with tighter VRAM
        budgets (typically video models that stack frames per clip)
        should override to pass a smaller default to
        :func:`resolve_embed_batch_size`.
        """
        return resolve_embed_batch_size()

    @property
    def license_notice(self) -> Optional[str]:
        """User-facing licence warning shown before a user selects this embedder.

        ``None`` (the default) means the embedder has no special licensing
        constraints worth surfacing.  Embedders distributed under a research-
        only or otherwise-restrictive licence (e.g. facebookresearch/EUPE under
        the FAIR Noncommercial Research Licence) return a short human-readable
        string the UI shows on the embedder picker so users know before they
        produce any outputs.  This is advisory; there is no acceptance
        click; users who object pick a different embedder.
        """
        return None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load (and cache) the embedding model.

        Called lazily the first time this embedder needs to produce a vector.
        Implementations must be idempotent; a second call should be a no-op.

        Subclasses should override :meth:`_load_models_impl` (not this method).
        This wrapper catches :class:`ImportError` and re-raises with an
        actionable message so that missing dependencies surface clearly.

        A per-class lock serialises concurrent callers so that only one
        thread performs the actual load; others wait and then return
        immediately (the subclass ``_load_models_impl`` checks
        ``self._model is not None``).

        A load whose caller installed no :meth:`progress_scope` reports through
        the process-wide default sink and is terminated there on the way out;
        see :meth:`_orphan_progress`.
        """
        if getattr(self, "_model", None) is not None:
            return
        with self._orphan_progress(), self._model_load_lock:
            try:
                self._load_models_impl()
            except ImportError as exc:
                raise ImportError(
                    f"{exc}. Required by the '{self.name}' embedder. "
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

        The returned vector is **L2-normalized** here (via
        :func:`vtscore.embedding.normalize.l2_normalize`) so that every
        embedding stored in ``medias`` is unit-norm regardless of which
        embedder produced it; subclasses must not (and need not) normalize
        themselves.

        Returns ``None`` if the media cannot be embedded.
        """
        from vtscore.embedding.normalize import l2_normalize  # noqa: PLC0415

        with self._embed_lock:
            vec = self._embed_media_impl(media)
        return None if vec is None else l2_normalize(vec)

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
        item; each call acquires :attr:`_embed_lock` individually so
        concurrent callers can interleave, and emits per-item progress
        via :attr:`_on_progress` so long runs stay visible in the UI.

        Subclasses backed by a service that natively accepts many items
        per request should override :meth:`_embed_media_bulk_impl`.  If
        they chunk internally (batching), they are responsible for
        emitting their own progress updates through :attr:`_on_progress`.

        Every returned vector is **L2-normalized** here so the stored-as-
        unit-norm invariant holds for the bulk path too (the default impl
        already routes through :meth:`embed_media`, so re-normalizing is a
        harmless no-op; overriding impls that batch raw outputs are covered
        here).
        """
        if not medias:
            return []
        from vtscore.embedding.normalize import l2_normalize  # noqa: PLC0415

        vectors = self._embed_media_bulk_impl(medias)
        return [None if v is None else l2_normalize(v) for v in vectors]

    def _embed_media_bulk_impl(self, medias: list[dict]) -> list[Optional[np.ndarray]]:
        """Subclass hook: embed a list of media items.

        Default: loop over :meth:`embed_media`, emitting per-item progress.
        Override to replace the per-item loop with a single bulk request,
        or to batch internally in chunks sized for a remote API.
        """
        total = len(medias)
        results: list[Optional[np.ndarray]] = []
        for i, m in enumerate(medias):
            self._on_progress("embedding", "Embedding...", i + 1, total)
            results.append(self.embed_media(m))
        return results

    def embed_medias(self, medias: dict[int, dict]) -> dict[int, Optional[np.ndarray]]:
        """Bulk-embed an id→media dict; return id→vector (or ``None``) dict.

        Convenience wrapper around :meth:`embed_media_bulk` for callers that
        already have medias keyed by ID, typically dataset importers that
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

        The result is **L2-normalized** here so query vectors are unit-norm
        just like stored media embeddings; this is what lets
        :mod:`vtscore.training.region_similarity` score with a plain dot
        product instead of re-normalizing on every comparison.  Subclasses
        override :meth:`_embed_text_impl` (not this method) and need not
        normalize themselves.

        Returns ``None`` when this embedder cannot embed text.
        """
        vec = self._embed_text_impl(text)
        if vec is None:
            return None
        from vtscore.embedding.normalize import l2_normalize  # noqa: PLC0415

        return l2_normalize(vec)

    def _embed_text_impl(self, text: str) -> Optional[np.ndarray]:
        """Subclass hook: embed a text query.

        Override this instead of :meth:`embed_text`.  The default returns
        ``None`` (text sorting unavailable).
        """
        return None

    def patch_forward(self, media: dict) -> Optional["PatchEmbedOutput"]:  # noqa: F821
        """Return per-patch features for one image.

        Patch-based image encoders (DINOv2, DINOv3, EUPE) override this to
        return a :class:`~vtscore.media.patch_embed.PatchEmbedOutput`
        carrying the CLS vector, the per-patch grid, and a per-patch saliency
        map.  Single-vector embedders leave the default in place and the
        loader pipeline skips the patch-region step for their datasets.

        The dataset loader gates calls on :attr:`supports_patch_regions`:
        if you set that flag ``True``, you must override this method.

        Acquires :attr:`_embed_lock` so the patch forward pass interleaves
        with single-vector embedders' forward passes on the same lock.
        Subclasses override :meth:`_patch_forward_impl` (not this method).

        Returns ``None`` if the media can't be loaded.
        """
        from vtscore.media.patch_embed import PatchEmbedOutput  # noqa: F401, PLC0415

        with self._embed_lock:
            return self._patch_forward_impl(media)

    def _patch_forward_impl(self, media: dict) -> Optional["PatchEmbedOutput"]:  # noqa: F821
        """Subclass hook for :meth:`patch_forward`.

        Default returns ``None``.  Patch-capable embedders override this.
        """
        return None

    def patch_forward_bulk(self, medias: list[dict]) -> list[Optional["PatchEmbedOutput"]]:  # noqa: F821
        """Return per-patch features for every image in *medias*.

        Patch-capable embedders override :meth:`_patch_forward_bulk_impl`
        to batch the forward pass through their backbone. The default
        loops :meth:`patch_forward` per item and emits per-item progress
        via :attr:`_on_progress`, matching the contract of
        :meth:`embed_media_bulk`.

        Positions where patch-forward returned ``None`` (failed decode,
        unsupported, etc.) contain ``None``.
        """
        if not medias:
            return []
        return self._patch_forward_bulk_impl(medias)

    def _patch_forward_bulk_impl(self, medias: list[dict]) -> list[Optional["PatchEmbedOutput"]]:  # noqa: F821
        """Subclass hook: bulk patch-forward.

        Default: loop over :meth:`patch_forward`, emitting per-item
        progress.  Override to fuse the per-image forward into a single
        batched GPU call.
        """
        total = len(medias)
        results: list[Optional["PatchEmbedOutput"]] = []  # noqa: F821
        for i, m in enumerate(medias):
            self._on_progress("embedding", "Patch-embedding...", i + 1, total)
            results.append(self.patch_forward(m))
        return results

    def local_features_forward(self, media: dict) -> Optional["StructuralFeatures"]:  # noqa: F821
        """Return local instance-matching features for one image.

        Structural embedders (SIFT/VLAD, and learned-local-feature variants
        later) override this to return a
        :class:`~vtscore.media.structural.StructuralFeatures` carrying the
        per-image keypoints and descriptors used by the geometric re-rank and
        the match-statistic verification classifier.  All other embedders
        leave the default in place and the loader skips the structural pass.

        The dataset loader gates calls on
        :attr:`supports_geometric_verification`: if you set that flag
        ``True``, you must override this method.

        Acquires :attr:`_embed_lock` so the feature-detection pass interleaves
        with other embedders' forward passes on the same lock.  Subclasses
        override :meth:`_local_features_forward_impl` (not this method).

        Returns ``None`` if the media can't be loaded.
        """
        with self._embed_lock:
            return self._local_features_forward_impl(media)

    def _local_features_forward_impl(self, media: dict) -> Optional["StructuralFeatures"]:  # noqa: F821
        """Subclass hook for :meth:`local_features_forward`.

        Default returns ``None``.  Structural embedders override this.
        """
        return None

    def local_features_forward_bulk(self, medias: list[dict]) -> list[Optional["StructuralFeatures"]]:  # noqa: F821
        """Return local features for every image in *medias*.

        Structural embedders override :meth:`_local_features_forward_bulk_impl`
        to batch the feature-detection pass.  The default loops
        :meth:`local_features_forward` per item and emits per-item progress
        via :attr:`_on_progress`, matching the contract of
        :meth:`patch_forward_bulk`.

        Positions where feature detection returned ``None`` (failed decode,
        unsupported, etc.) contain ``None``.
        """
        if not medias:
            return []
        return self._local_features_forward_bulk_impl(medias)

    def _local_features_forward_bulk_impl(self, medias: list[dict]) -> list[Optional["StructuralFeatures"]]:  # noqa: F821
        """Subclass hook: bulk local-feature detection.

        Default: loop over :meth:`local_features_forward`, emitting per-item
        progress.  Override to fuse the per-image detection into a batched call.
        """
        total = len(medias)
        results: list[Optional["StructuralFeatures"]] = []  # noqa: F821
        for i, m in enumerate(medias):
            self._on_progress("embedding", "Detecting features...", i + 1, total)
            results.append(self.local_features_forward(m))
        return results

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
            "display_name": self.display_name,
            "model_id": self.model_id,
            "media_type_id": self.media_type_id,
            "is_default": self.is_default,
            "supports_text": self.supports_text,
            "supports_patch_regions": self.supports_patch_regions,
            "supports_geometric_verification": self.supports_geometric_verification,
            "license_notice": self.license_notice,
        }
