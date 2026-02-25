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

from vtsearch.config import MODELS_CACHE_DIR


def initialize_models() -> None:
    """Prepare the runtime environment for embedding models.

    Creates the model cache directory and configures PyTorch thread count.
    Models themselves are **not** loaded here — each
    :class:`~vtsearch.media.base.MediaType` loads its model lazily the
    first time it is needed (e.g. when ``embed_media`` or ``embed_text``
    is called).
    """
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    import torch

    torch.set_num_threads(1)
    gc.collect()


def preload_favorite_media_types() -> list[str]:
    """Eagerly load embedding models for all favorite media types.

    Reads ``favorite_media_types`` from persisted settings and calls
    :meth:`~vtsearch.media.base.MediaType.load_models` on each one so
    that models are warm before the user opens the GUI.

    Returns the list of type IDs that were preloaded.
    """
    from vtsearch.media import get as media_get
    from vtsearch.settings import get_favorite_media_types

    preloaded: list[str] = []
    for type_id in get_favorite_media_types():
        try:
            mt = media_get(type_id)
            print(f"  Preloading {mt.name} embedder...", flush=True)
            mt.load_models()
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
