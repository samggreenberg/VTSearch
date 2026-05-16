"""Embedding façades: text/file embedders, runtime loader, cached matrix.

The actual embedder implementations live in :mod:`vtsearch.media`; this
package exposes thin call-sites used throughout the rest of VTSearch
(:mod:`vtsearch.datasets.loader`, :mod:`vtsearch.routes.sorting`,
:mod:`vtsearch.eval.runner`, etc.).
"""

from vtsearch.embedding.helpers import (
    embed_audio_file,
    embed_image_file,
    embed_paragraph_file,
    embed_text_query,
    embed_video_file,
)
from vtsearch.embedding.loader import (
    get_clap_model,
    get_e5_model,
    get_torch_device,
    get_xclip_model,
    initialize_models,
    predict_embedders_to_preload,
    preload_predicted_embedders,
    smart_preload_in_background,
)
from vtsearch.embedding.matrix import (
    get_embedding_matrix,
    get_embedding_matrix_for_snap,
    invalidate_embedding_matrix,
)

__all__ = [
    "embed_audio_file",
    "embed_video_file",
    "embed_image_file",
    "embed_paragraph_file",
    "embed_text_query",
    "initialize_models",
    "predict_embedders_to_preload",
    "preload_predicted_embedders",
    "smart_preload_in_background",
    "get_clap_model",
    "get_xclip_model",
    "get_e5_model",
    "get_torch_device",
    "get_embedding_matrix",
    "invalidate_embedding_matrix",
    "get_embedding_matrix_for_snap",
]
