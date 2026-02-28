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

from vtsearch.config import MODELS_CACHE_DIR


_torch_configured = False


def ensure_torch_configured() -> None:
    """Set ``torch.set_num_threads(1)`` once, the first time torch is used.

    Safe to call multiple times — the configuration is applied only once.
    Call this from any code path that imports torch before doing work
    (e.g. ``load_models``, ``train_model``).
    """
    global _torch_configured
    if _torch_configured:
        return
    if "torch" not in sys.modules:
        return
    import torch  # noqa: PLC0415

    torch.set_num_threads(1)
    _torch_configured = True


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
    """Eagerly load embedding models for all autoload media types.

    Reads ``autoload_media_types`` from persisted settings and calls
    :meth:`~vtsearch.media.base.MediaType.load_models` on each one so
    that models are warm before the user opens the GUI.

    Prints intermediate status messages and download progress bars to
    the console so that the user can see what is happening during the
    (potentially long) model loading phase.

    Returns the list of type IDs that were preloaded.
    """
    from vtsearch.media import get as media_get
    from vtsearch.settings import get_autoload_media_types

    preloaded: list[str] = []
    for type_id in get_autoload_media_types():
        try:
            mt = media_get(type_id)
            print(f"  Preloading {mt.name} embedder...", flush=True)
            # Temporarily wrap the media type's progress callback so
            # that status updates are also printed to the console.
            original_cb = mt._on_progress
            mt._on_progress = _make_console_progress(original_cb)
            try:
                mt.load_models()
            finally:
                mt._on_progress = original_cb
            preloaded.append(type_id)
        except Exception as exc:
            print(f"  Warning: failed to preload {type_id}: {exc}", flush=True)
    return preloaded


# ---------------------------------------------------------------------------
# Backward-compatible getter functions
#
# These return the model instances held by their respective MediaType objects.
# Existing callers that import these functions directly continue to work.
# ---------------------------------------------------------------------------


def get_clap_model():
    """Return ``(clap_model, clap_processor)`` from the audio media type."""
    from vtsearch.media import get as media_get

    return media_get("audio")._get_model_and_processor()


def get_xclip_model():
    """Return ``(xclip_model, xclip_processor)`` from the video media type."""
    from vtsearch.media import get as media_get

    return media_get("video")._get_model_and_processor()


def get_clip_model():
    """Return ``(clip_model, clip_processor)`` from the image media type."""
    from vtsearch.media import get as media_get

    return media_get("image")._get_model_and_processor()


def get_e5_model():
    """Return the E5 ``SentenceTransformer`` from the text media type."""
    from vtsearch.media import get as media_get

    return media_get("paragraph")._get_model()
