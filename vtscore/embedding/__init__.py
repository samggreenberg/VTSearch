"""Embedding façades: text/file embedders, runtime loader, cached matrix.

The actual embedder implementations live in :mod:`vtscore.media`; this
package exposes thin call-sites used throughout the rest of VTSearch
(:mod:`vtscore.datasets.loader`, :mod:`vtsearch.routes.sorting`,
:mod:`vtscore.eval.runner`, etc.).
"""

from vtscore.embedding.helpers import (
    clear_text_query_cache,
    embed_audio_file,
    embed_image_file,
    embed_paragraph_file,
    embed_text_query,
    embed_video_file,
)
from vtscore.embedding.loader import (
    get_clap_model,
    get_e5_model,
    get_torch_device,
    get_xclip_model,
    initialize_models,
    predict_embedder_for_dataset,
    predict_embedders_to_preload,
    preload_embedder_for_dataset,
    preload_predicted_embedders,
    smart_preload_in_background,
)
from vtscore.embedding.matrix import (
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
    "clear_text_query_cache",
    "initialize_models",
    "predict_embedder_for_dataset",
    "predict_embedders_to_preload",
    "preload_embedder_for_dataset",
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
