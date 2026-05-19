"""vtscore - trainable media search library.

The Flask-free, app-free core of VTSearch: dataset origins, MediaSources,
clippers/croppers, embedders, MLP/detector training and scoring, evaluation.
The :mod:`vtsearch` package wraps this with the HTTP / SPA / settings layer.

See ``docs/plans/extract-library.md`` Phase 8 for the move's history and
``docs/vtscore-api.md`` for the documented public surface.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
