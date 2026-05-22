"""vtscore - trainable media search library.

The Flask-free, app-free core of VTSearch: dataset origins, MediaSources,
clippers/croppers, embedders, MLP/detector training and scoring, evaluation.
The :mod:`vtsearch` package wraps this with the HTTP / SPA / settings layer.

See ``README.md`` for the quickstart, ``CHANGELOG.md`` for per-release notes,
``docs/`` for the developer documentation tree (architecture, concepts,
per-package guides, plugin authoring), and ``../docs/vtscore-api.md`` for
the canonical public-API inventory.

The version below is independent semver, bumped by hand on each release;
unlike :mod:`vtsearch`, it is *not* derived from the git HEAD timestamp.
Bump it when cutting a release and add a matching entry to
``vtscore/CHANGELOG.md``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
