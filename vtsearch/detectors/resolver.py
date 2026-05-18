"""Resolve label entries to embeddings by following their origin trails.

When a detector's training labels don't match a target dataset (cross-dataset
scenario), we need to find the original media files, embed them, and use those
embeddings for training.  This module handles that resolution:

1. Given a label entry's origin info, resolve to an actual file on disk.
2. Embed the file using the appropriate embedder for the media type.
3. Return resolved embeddings with availability stats.

File resolution is split into two pluggable resolvers, auto-wired on
first use:

- **Source resolver** — delegates to
  :func:`~vtsearch.datasets.sources.get_source_for_origin` and calls
  :meth:`~MediaSource.resolve_path`.  Registers the source's
  :meth:`~MediaSource.cleanup` on the caller's :class:`ExitStack` so
  per-call temp storage is held alive for the duration of the
  :func:`resolve_file_context` block (otherwise GC of the source can
  delete the path before the caller embeds the file).
- **Importer resolver** — delegates to
  :func:`~vtsearch.datasets.importers.get_importer` and calls
  :meth:`~DatasetImporter.resolve_file`.

External code can replace or extend these via
:func:`register_source_resolver` and :func:`register_importer_resolver`.

Two synthetic origin types (``dupe_set`` and ``converter``) are handled inline
because they are not importers in the registry — they delegate to real
importers.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FileResolver protocol — the contract for pluggable file resolution
# ---------------------------------------------------------------------------


@runtime_checkable
class SourceResolver(Protocol):
    """Callable that resolves an origin to a file via a :class:`MediaSource`.

    Implementations receive an :class:`~contextlib.ExitStack` (so they can
    register :meth:`~MediaSource.cleanup` to fire when the caller's
    :func:`resolve_file_context` exits) plus the origin dict,
    ``origin_name``, and ``filename``.
    """

    def __call__(
        self,
        stack: ExitStack,
        origin: dict[str, Any],
        origin_name: str,
        filename: str,
    ) -> Path | None: ...


@runtime_checkable
class ImporterResolver(Protocol):
    """Callable that resolves an origin via its dataset importer.

    Importer-based dispatch does not allocate a per-call source — importers
    that need to cache materialised content (e.g. ``http_archive``) are
    expected to write to a stable directory and never call ``cleanup()``,
    so no :class:`ExitStack` is threaded through.
    """

    def __call__(
        self,
        origin: dict[str, Any],
        origin_name: str,
        filename: str,
    ) -> Path | None: ...


# ---------------------------------------------------------------------------
# Pluggable resolver registry
# ---------------------------------------------------------------------------

_source_resolver: SourceResolver | None = None
_importer_resolver: ImporterResolver | None = None
_auto_wired = False


def register_source_resolver(fn: SourceResolver) -> None:
    """Register a source-based file resolver.

    The resolver is called with ``(stack, origin, origin_name, filename)``
    and should return a :class:`~pathlib.Path` or ``None``.  Register the
    backing source's :meth:`~MediaSource.cleanup` on *stack* so its temp
    storage is only released once the caller is done with the path.
    """
    global _source_resolver
    _source_resolver = fn


def register_importer_resolver(fn: ImporterResolver) -> None:
    """Register an importer-based file resolver.

    The resolver is called with ``(origin, origin_name, filename)`` and
    should return a :class:`~pathlib.Path` or ``None``.
    """
    global _importer_resolver
    _importer_resolver = fn


def _auto_wire_resolvers() -> None:
    """Auto-wire the default resolvers on first use.

    Imports the datasets source and importer packages lazily and registers
    them as the default resolvers.  This avoids hard-coding cross-package
    imports throughout :func:`resolve_file_from_origin` while still
    providing zero-config behaviour.
    """
    global _auto_wired
    if _auto_wired:
        return
    _auto_wired = True

    # Source-based resolver
    if _source_resolver is None:
        try:
            from vtsearch.datasets.sources import get_source_for_origin

            def _default_source_resolver(
                stack: ExitStack,
                origin: dict[str, Any],
                origin_name: str,
                filename: str,
            ) -> Path | None:
                source = get_source_for_origin(origin)
                if source is None:
                    return None
                # Keep the source alive — and its temp dir, if any — until
                # the caller's resolve_file_context exits.
                stack.callback(source.cleanup)
                return source.resolve_path(origin_name, filename)

            register_source_resolver(_default_source_resolver)
        except ImportError:
            pass

    # Importer-based resolver
    if _importer_resolver is None:
        try:
            from vtsearch.datasets.importers import get_importer

            def _default_importer_resolver(
                origin: dict[str, Any],
                origin_name: str,
                filename: str,
            ) -> Path | None:
                importer_name = origin.get("importer", "")
                importer = get_importer(importer_name)
                if importer is not None:
                    return importer.resolve_file(origin, origin_name, filename)
                return None

            register_importer_resolver(_default_importer_resolver)
        except ImportError:
            pass


@dataclass
class ResolvedLabels:
    """Result of resolving label entries to embeddings."""

    embeddings: list[np.ndarray] = field(default_factory=list)
    labels: list[float] = field(default_factory=list)
    resolved_count: int = 0
    total_count: int = 0
    missing_entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def available_fraction(self) -> float:
        return self.resolved_count / self.total_count if self.total_count else 0.0

    @property
    def has_good_and_bad(self) -> bool:
        return any(v == 1.0 for v in self.labels) and any(v == 0.0 for v in self.labels)


@contextmanager
def resolve_file_context(
    origin: dict[str, Any] | None,
    origin_name: str = "",
    filename: str = "",
) -> Iterator[Path | None]:
    """Resolve a media file from its origin and keep the backing source alive.

    Some :class:`~vtsearch.datasets.sources.base.MediaSource` implementations
    (e.g. PullWrest) materialise the file inside a
    :class:`tempfile.TemporaryDirectory` they own.  If the source is dropped
    before the caller accesses the file, the temp dir is finalized by GC and
    the path goes stale — ``embed_file`` then crashes with
    ``FileNotFoundError``.

    This context manager owns the source for the duration of the ``with``
    block and only invokes :meth:`~MediaSource.cleanup` on exit, so the path
    is guaranteed to remain valid while the caller embeds or reads bytes
    from it.

    Yields the resolved :class:`~pathlib.Path`, or ``None`` if no resolver
    could locate the file.
    """
    with ExitStack() as stack:
        path = _resolve_with_stack(stack, origin, origin_name, filename)
        yield path


def resolve_file_from_origin(
    origin: dict[str, Any] | None,
    origin_name: str = "",
    filename: str = "",
) -> Path | None:
    """Resolve a media file from its origin information.

    Returns the file path if found, or ``None``.  Convenience wrapper over
    :func:`resolve_file_context` for callers that only need to test
    existence or read the file immediately (synchronously, in the same
    expression).  Callers that hold the returned path across an operation
    that re-enters Python — embedding, opening with PIL, etc. — must use
    :func:`resolve_file_context` instead, because some media sources own
    temporary directories that are garbage-collected as soon as this
    function returns.
    """
    with resolve_file_context(origin, origin_name, filename) as p:
        return p


def _resolve_with_stack(  # noqa: C901
    stack: ExitStack,
    origin: dict[str, Any] | None,
    origin_name: str = "",
    filename: str = "",
) -> Path | None:
    """Resolve *origin* to a :class:`Path` while registering any source's
    :meth:`~MediaSource.cleanup` on *stack*.

    Shared core for :func:`resolve_file_context` and (transitively)
    :func:`resolve_file_from_origin`.  The two synthetic origin types are
    handled inline because they delegate to real importers:

    - ``dupe_set``: tries each member until one resolves.
    - ``converter``: reconstructs the parent origin and delegates.
    """
    if origin is None:
        log.debug("resolve_file: origin is None — cannot resolve")
        return None

    importer_name = origin.get("importer", "")
    params = origin.get("params", {})

    log.debug(
        "resolve_file: importer=%r, origin_name=%r, filename=%r, params=%r",
        importer_name,
        origin_name,
        filename,
        params,
    )

    # -- Synthetic origins that delegate to real importers --

    if importer_name == "dupe_set":
        result = _resolve_dupe_set(stack, origin)
        if result is None:
            members = origin.get("members", [])
            log.debug("resolve_file: dupe_set with %d members — none resolved", len(members))
        return result

    if importer_name == "converter":
        result = _resolve_converter(stack, params)
        if result is None:
            log.debug(
                "resolve_file: converter origin failed — source_file=%r, parent_importer=%r",
                params.get("source_file", ""),
                params.get("parent_importer", ""),
            )
        return result

    # -- Source-based dispatch (preferred) --

    _auto_wire_resolvers()

    if _source_resolver is not None:
        result = _source_resolver(stack, origin, origin_name, filename)
        if result is not None:
            log.debug("resolve_file: source-based dispatch succeeded → %s", result)
            return result
        log.debug(
            "resolve_file: source resolver returned None for origin_name=%r, filename=%r",
            origin_name,
            filename,
        )

    # -- Registry-based dispatch (fallback for importers without a source) --

    if _importer_resolver is not None:
        result = _importer_resolver(origin, origin_name, filename)
        if result is not None:
            log.debug("resolve_file: importer dispatch (%s) succeeded → %s", importer_name, result)
            return result
        log.debug(
            "resolve_file: importer resolver returned None (importer=%r, origin_name=%r, filename=%r, params=%r)",
            importer_name,
            origin_name,
            filename,
            params,
        )
    else:
        log.debug("resolve_file: no importer resolver registered")

    # -- Generic fallback for unregistered origins with a path param --
    # Handles synthetic origins like "pdf" that store a direct file path.
    path = params.get("path", "")
    if path:
        p = Path(path)
        if p.is_file():
            log.debug("resolve_file: generic path fallback succeeded → %s", p)
            return p
        log.debug("resolve_file: generic path fallback — %r is not a file", path)

    log.debug(
        "resolve_file: ALL dispatch methods failed for importer=%r, origin_name=%r, filename=%r",
        importer_name,
        origin_name,
        filename,
    )
    return None


def _resolve_dupe_set(stack: ExitStack, origin: dict[str, Any]) -> Path | None:
    """Try each member of a dupe_set until one resolves."""
    for m in origin.get("members", []):
        result = _resolve_with_stack(
            stack,
            m.get("origin"),
            m.get("origin_name", ""),
            m.get("filename", ""),
        )
        if result is not None:
            return result
    return None


def _resolve_converter(stack: ExitStack, params: dict[str, str]) -> Path | None:
    """Resolve a converter origin by rebuilding its parent origin."""
    source_file = params.get("source_file", "")
    parent_importer = params.get("parent_importer", "")
    if not source_file or not parent_importer:
        return None

    # Reconstruct a parent origin dict from the converter's stored params
    parent_params: dict[str, str] = {}
    if params.get("parent_path"):
        parent_params["path"] = params["parent_path"]
    if params.get("parent_url"):
        parent_params["url"] = params["parent_url"]

    parent_origin = {"importer": parent_importer, "params": parent_params}
    return _resolve_with_stack(stack, parent_origin, origin_name=source_file)


def embed_file(file_path: Path, media_type: str, embedder_name: str = "") -> np.ndarray | None:
    """Embed a media file using the appropriate embedder for the media type.

    If *embedder_name* is given, that specific embedder is used (matching by
    name); if it is not found or not given, the first registered embedder for
    the media type is used as a fallback.
    """
    from vtsearch.media import embedders_for_type, get_embedder

    if embedder_name:
        try:
            embedder = get_embedder(embedder_name)
        except (KeyError, ValueError):
            log.warning(
                "embed_file: embedder %r not found, falling back to default for media_type=%r",
                embedder_name,
                media_type,
            )
            embedder = None
    else:
        embedder = None

    if embedder is None:
        avail = embedders_for_type(media_type)
        if not avail:
            log.warning(
                "embed_file: no embedders registered for media_type=%r — cannot embed %s",
                media_type,
                file_path,
            )
            return None
        embedder = avail[0]
    from vtsearch.media.embedder import media_from_path  # noqa: PLC0415

    try:
        result = embedder.embed_media(media_from_path(file_path))
    except Exception:
        log.warning(
            "embed_file: %s.embed_media(%s) raised an exception",
            type(embedder).__name__,
            file_path,
            exc_info=True,
        )
        return None
    if result is None:
        log.warning(
            "embed_file: %s.embed_media(%s) returned None",
            type(embedder).__name__,
            file_path,
        )
    else:
        log.debug(
            "embed_file: embedded %s with %s → shape %s",
            file_path.name,
            type(embedder).__name__,
            result.shape,
        )
    return result


def _apply_clip_and_embed(  # noqa: C901
    file_path: Path,
    media_type: str,
    origin: dict[str, Any],
    embedder_name: str = "",
) -> np.ndarray | None:
    """Apply clip params from *origin* to a resolved file and embed the result.

    If the origin contains clip parameters (``clip_start``/``clip_end`` for
    audio, ``clip_box`` for images, or a text sentence clipper), the file is
    clipped first and the clipped content is embedded.  Falls back to
    :func:`embed_file` when no clip params are present.
    """
    import os
    import tempfile

    params = origin.get("params", {})

    # Chain replay takes precedence over the legacy single-clipper path.
    chain_raw = params.get("clipper_chain")
    if chain_raw:
        from vtsearch.datasets.clipper_chain import parse_trail, replay_chain_on_file

        steps = parse_trail(chain_raw)
        if steps:
            try:
                embedding = replay_chain_on_file(file_path, steps, embedder_name)
            except Exception:
                log.debug("_apply_clip_and_embed: chain replay failed, falling back", exc_info=True)
                embedding = None
            if embedding is not None:
                return embedding
            # Fall through to legacy/full-file embed.

    clipper_name = params.get("clipper", "")

    if not clipper_name:
        return embed_file(file_path, media_type, embedder_name)

    clip_start = params.get("clip_start")
    clip_end = params.get("clip_end")
    clip_box = params.get("clip_box")

    # --- Audio clips: slice WAV bytes ---
    if clip_start is not None and clip_end is not None and media_type == "audio":
        try:
            from vtsearch.media.audio.clipper import _wav_slice

            wav_bytes = file_path.read_bytes()
            sliced = _wav_slice(wav_bytes, float(clip_start), float(clip_end))
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            try:
                os.write(fd, sliced)
                os.close(fd)
                return embed_file(Path(tmp), media_type, embedder_name)
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        except Exception:
            log.debug("_apply_clip_and_embed: audio clip failed, falling back", exc_info=True)
            return embed_file(file_path, media_type, embedder_name)

    # --- Image clips: crop to clip_box ---
    if clip_box is not None and media_type == "image":
        try:
            import io as _io

            from PIL import Image

            parts = [int(float(v)) for v in clip_box.split(",")]
            if len(parts) != 4:
                raise ValueError(f"clip_box must have 4 values, got {len(parts)}")
            box_values: tuple[int, int, int, int] = (parts[0], parts[1], parts[2], parts[3])
            with Image.open(file_path) as img:
                cropped = img.crop(box_values)
                buf = _io.BytesIO()
                cropped.save(buf, format=img.format or "PNG")
            crop_bytes = buf.getvalue()
            fd, tmp = tempfile.mkstemp(suffix=".png")
            try:
                os.write(fd, crop_bytes)
                os.close(fd)
                return embed_file(Path(tmp), media_type, embedder_name)
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        except Exception:
            log.debug("_apply_clip_and_embed: image clip failed, falling back", exc_info=True)
            return embed_file(file_path, media_type, embedder_name)

    # --- Text clips: extract the sentence by clip_index ---
    if media_type == "text":
        try:
            clip_index = params.get("clip_index")
            if clip_index is not None:
                import re

                text = file_path.read_text(encoding="utf-8")
                sentence_re = re.compile(r"(?<=[.!?])\s+")
                sentences = [s.strip() for s in sentence_re.split(text) if s.strip()]
                idx = int(clip_index)
                if 0 <= idx < len(sentences):
                    fd, tmp = tempfile.mkstemp(suffix=".txt")
                    try:
                        os.write(fd, sentences[idx].encode("utf-8"))
                        os.close(fd)
                        return embed_file(Path(tmp), media_type, embedder_name)
                    finally:
                        try:
                            os.unlink(tmp)
                        except OSError:
                            pass
        except Exception:
            log.debug("_apply_clip_and_embed: text clip failed, falling back", exc_info=True)
            return embed_file(file_path, media_type, embedder_name)

    # Video clips and unrecognised clippers: embed the full file.
    return embed_file(file_path, media_type, embedder_name)


def resolve_label_embeddings(  # noqa: C901
    labels: list[dict[str, Any]],
    media_type: str,
    progress_callback: Any | None = None,
) -> ResolvedLabels:
    """Resolve label entries to embeddings by following their origin trails.

    For each label entry, attempts to:
    1. Resolve the original media file from its origin info
    2. Embed it using the appropriate embedder for *media_type*
    3. Collect the embedding and label value

    Args:
        labels: List of label dicts (with origin, origin_name, filename, label keys).
        media_type: The media type for embedding (e.g. "audio", "image").

    Returns:
        A :class:`ResolvedLabels` with resolved embeddings, stats, and missing entries.
    """
    result = ResolvedLabels()

    # Track failure reasons for the summary log
    _no_origin = 0
    _file_not_found = 0
    _embed_failed = 0

    log.info(
        "resolve_label_embeddings: starting resolution of %d label entries for media_type=%r",
        len(labels),
        media_type,
    )

    _total_entries = len(labels)

    for i, entry in enumerate(labels):
        label_val = entry.get("label", "")
        if label_val not in ("good", "bad"):
            if progress_callback is not None:
                progress_callback(current=i + 1, total=_total_entries)
            continue

        result.total_count += 1

        origin = entry.get("origin")
        origin_name = entry.get("origin_name", "")
        filename = entry.get("filename", "")

        with resolve_file_context(origin, origin_name, filename) as file_path:
            if file_path is None:
                result.missing_entries.append(entry)
                if origin is None:
                    _no_origin += 1
                    log.info(
                        "  label[%d] FAILED (no origin): md5=%s, origin_name=%r, "
                        "filename=%r — this label has no origin trail and cannot "
                        "be resolved to a file",
                        i,
                        entry.get("md5", "?")[:12],
                        origin_name,
                        filename,
                    )
                else:
                    _file_not_found += 1
                    log.info(
                        "  label[%d] FAILED (file not found): importer=%r, origin_name=%r, filename=%r, params=%r",
                        i,
                        origin.get("importer", "?"),
                        origin_name,
                        filename,
                        origin.get("params", {}),
                    )
                if progress_callback is not None:
                    progress_callback(current=i + 1, total=_total_entries)
                continue

            # Use clip-aware embedding when the label has clip params in its
            # origin (e.g. from a clipped dataset).  This ensures cross-dataset
            # resolution embeds the clipped content, not the whole parent file.
            origin_params = origin.get("params", {}) if origin is not None else {}
            if origin is not None and (origin_params.get("clipper") or origin_params.get("clipper_chain")):
                embedding = _apply_clip_and_embed(file_path, media_type, origin)
            else:
                embedding = embed_file(file_path, media_type)
            if embedding is None:
                result.missing_entries.append(entry)
                _embed_failed += 1
                log.info(
                    "  label[%d] FAILED (embed): file resolved to %s but embedding returned None for media_type=%r",
                    i,
                    file_path,
                    media_type,
                )
                if progress_callback is not None:
                    progress_callback(current=i + 1, total=_total_entries)
                continue

            result.embeddings.append(embedding)
            result.labels.append(1.0 if label_val == "good" else 0.0)
            result.resolved_count += 1
            log.debug(
                "  label[%d] OK: %s → %s (label=%s)",
                i,
                origin_name or filename,
                file_path.name,
                label_val,
            )
            if progress_callback is not None:
                progress_callback(current=i + 1, total=_total_entries)

    # --- Summary ---
    n_good = sum(1 for v in result.labels if v == 1.0)
    n_bad = sum(1 for v in result.labels if v == 0.0)
    _summary = (
        f"resolve_label_embeddings: {result.resolved_count} of "
        f"{result.total_count} labels resolved "
        f"({n_good} good, {n_bad} bad)"
    )
    if result.missing_entries:
        _summary += (
            f" | {len(result.missing_entries)} FAILED: "
            f"{_no_origin} had no origin, "
            f"{_file_not_found} file not found, "
            f"{_embed_failed} embed failed"
        )

    if result.total_count > 0 and result.resolved_count == 0:
        log.warning(
            "%s. This usually means the importer's resolve_file() method "
            "is missing or the source files are no longer on disk.",
            _summary,
        )
        if result.missing_entries:
            first = result.missing_entries[0]
            log.warning(
                "First unresolved label: origin=%r, origin_name=%r, filename=%r",
                first.get("origin"),
                first.get("origin_name", ""),
                first.get("filename", ""),
            )
    elif result.missing_entries:
        log.warning("%s", _summary)
    else:
        log.info("%s", _summary)

    return result
