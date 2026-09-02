"""Model-load progress reporting and resilient HuggingFace fetching.

Loading an embedding backbone is the longest blocking thing VTSearch does, and
none of the libraries doing it report progress anywhere the GUI can see: tqdm
bars go to stderr, ``from_pretrained`` gives no callback, and a heavy ``import``
gives nothing at all.  This module is the machinery that makes that wait
legible, and it is entirely independent of the :class:`MediaEmbedder` ABC that
calls it:

* :func:`timed_progress` paces a blocking ``import`` off the live
  ``sys.modules`` delta (see :data:`IMPORT_MODULE_ESTIMATES`).
* :func:`intercept_tqdm_progress` and :func:`intercept_weight_loading_progress`
  monkey-patch *process-wide* globals — tqdm's class methods, ``torch.load``,
  ``nn.Module.load_state_dict`` — behind a reference-counted, thread-attributed
  registry so concurrent loads keep their progress separate.
* :func:`load_pretrained_local_first` and :func:`hf_token` make the fetch itself
  robust: cached weights first, then a preflighted download with backoff.
* :func:`embedder_load_setup` is the setup ceremony every ``_load_models_impl``
  opens with.

Split out of :mod:`vtscore.media.embedder`, which re-exports every public name
here for third-party embedders.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import threading
import time
from typing import Any, Callable, Optional

from vtscore.media.base import ProgressCallback

__all__ = [
    "IMPORT_MODULE_ESTIMATES",
    "embedder_load_setup",
    "hf_token",
    "intercept_tqdm_progress",
    "intercept_weight_loading_progress",
    "load_pretrained_local_first",
    "timed_progress",
]


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
