"""Model loading and initialisation — delegates to the media type registry.

All embedding models are now owned by their respective
:class:`~vtsearch.media.base.MediaType` instances and are loaded **lazily**
on first use (the first call to ``embed_media``, ``embed_text``, or a getter
function such as ``get_clap_model``).

This module keeps its original public API (``initialize_models``,
``get_clap_model``, etc.) as thin wrappers so that existing callers continue
to work unchanged.
"""

import gc
import sys

from vtsearch.config import MODELS_CACHE_DIR, TORCH_THREADS, resolve_device


_torch_configured = False


def ensure_torch_configured() -> None:
    """Set ``torch.set_num_threads(TORCH_THREADS)`` the first time torch is used.

    Thread count comes from :data:`vtsearch.config.TORCH_THREADS`, which in
    turn reads ``VTSEARCH_TORCH_THREADS`` (default ``1``).  Safe to call
    multiple times — the configuration is applied only once.  Call this from
    any code path that imports torch before doing work (e.g. ``load_models``,
    ``train_model``).
    """
    global _torch_configured
    if _torch_configured:
        return
    if "torch" not in sys.modules:
        return
    import torch  # noqa: PLC0415

    torch.set_num_threads(TORCH_THREADS)
    _torch_configured = True


def get_torch_device():
    """Return the preferred ``torch.device`` for MLP training / scoring.

    Resolves :data:`vtsearch.config.DEVICE` (``VTSEARCH_DEVICE``, default
    ``"auto"``) to a concrete device — ``cuda`` when available, ``mps`` on
    Apple silicon, or ``cpu``.  Imports torch lazily.
    """
    import torch  # noqa: PLC0415

    return torch.device(resolve_device())


def initialize_models() -> None:
    """Prepare the runtime environment for embedding models.

    Creates the model cache directory and configures PyTorch thread count
    **if torch is already imported**.  When torch has not been imported yet
    (e.g. during fast test startup) the thread-count configuration is
    deferred until ``ensure_torch_configured`` is called by the first code
    path that actually imports torch.

    Models themselves are **not** loaded here.
    """
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ensure_torch_configured()
    gc.collect()


def _make_console_progress(original_callback):
    """Wrap *original_callback* to also print status to the terminal.

    Used during startup preloading so the user sees intermediate status
    messages and download progress bars in the console while models are
    being loaded.
    """
    _last_msg: list[str | None] = [None]
    _on_progress_line: list[bool] = [False]

    def _callback(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
        original_callback(status, message, current, total)

        if total > 0:
            # Measurable progress (download / weight loading) — overwrite same line
            pct = min(100, current * 100 // total)
            filled = pct * 30 // 100
            bar = "#" * filled + "." * (30 - filled)
            line = f"\r    {message} [{bar}] {pct:>3}%"
            sys.stdout.write(line)
            sys.stdout.flush()
            _on_progress_line[0] = True
            if current >= total:
                sys.stdout.write("\n")
                sys.stdout.flush()
                _on_progress_line[0] = False
                _last_msg[0] = None
        elif message and message != _last_msg[0]:
            # New phase message — print on its own line
            if _on_progress_line[0]:
                sys.stdout.write("\n")
                _on_progress_line[0] = False
            sys.stdout.write(f"    {message}\n")
            sys.stdout.flush()
            _last_msg[0] = message

    return _callback


def preload_autoload_media_types() -> list[str]:
    """Eagerly load embedding models for autoload embedders (and legacy media types).

    Reads ``autoload_media_embedders`` from persisted settings and calls
    :meth:`~vtsearch.media.base.MediaEmbedder.load_models` on each one so
    that models are warm before the user opens the GUI.

    Prints intermediate status messages and download progress bars to
    the console so that the user can see what is happening during the
    (potentially long) model loading phase.

    Returns the list of embedder names that were preloaded.
    """
    from vtsearch.media import get_embedder
    from vtsearch.settings import get_autoload_media_embedders

    preloaded: list[str] = []

    for emb_name in get_autoload_media_embedders():
        try:
            emb = get_embedder(emb_name)
            print(f"  Preloading {emb_name} embedder...", flush=True)
            original_cb = emb._on_progress
            emb._on_progress = _make_console_progress(original_cb)
            try:
                emb.load_models()
            finally:
                emb._on_progress = original_cb
            preloaded.append(emb_name)
        except Exception as exc:
            print(f"  Warning: failed to preload {emb_name}: {exc}", flush=True)
    return preloaded


# ---------------------------------------------------------------------------
# Backward-compatible getter functions
#
# These return the model instances held by their respective embedder objects.
# Existing callers that import these functions directly continue to work.
# ---------------------------------------------------------------------------


def get_clap_model():
    """Return ``(clap_model, clap_processor)`` from the CLAP embedder."""
    from vtsearch.media import get_embedder

    emb = get_embedder("clap")
    return emb._get_model_and_processor()


def get_xclip_model():
    """Return ``(xclip_model, xclip_processor)`` from the X-CLIP embedder."""
    from vtsearch.media import get_embedder

    emb = get_embedder("xclip")
    return emb._get_model_and_processor()


def get_e5_model():
    """Return the E5 ``SentenceTransformer`` from the E5 embedder."""
    from vtsearch.media import get_embedder

    emb = get_embedder("e5")
    return emb._get_model()
