"""A chunk of a chunked import, walked but not yet built (#5314).

A chunked loader does two kinds of work per chunk: the walk (which items
belong to it, in what order) and the build (reading each item's file,
hashing it, making its thumbnail).  :class:`DeferredChunk` hands the caller
the first without the second, so a caller that only needs to pass a chunk by
- a resumed CLI run reaching the chunks it already wrote - never pays for the
build.  The importer hook that yields these is
:meth:`~vtscore.datasets.importers.base.ImporterBase.run_chunked_cli_deferred`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeferredChunk:
    """One chunk of media, its size and contents known, its media not built yet.

    ``count`` is how many media :attr:`build` makes; a caller numbers the
    chunks after this one by it without building this one.  ``build`` makes
    the chunk's medias dict (ids from 1, as a chunked importer yields), and
    is called at most once, before the next chunk is pulled.  ``key``, when
    set, returns what the chunk holds without reading any file's contents -
    paths, sizes and modification times, say - as JSON-serialisable data that
    changes when the chunk's items do.  A chunk with no ``key`` can be told
    apart from a changed one only by its built media, so a caller that needs
    to check it builds it.
    """

    count: int
    build: Callable[[], dict[int, dict[str, Any]]]
    key: Callable[[], Any] | None = None

    @classmethod
    def built(cls, medias: dict[int, dict[str, Any]]) -> DeferredChunk:
        """Wrap a chunk that is already built: no key, so a caller checks it by its media."""
        return cls(count=len(medias), build=lambda: medias)
