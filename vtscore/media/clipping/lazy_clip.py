"""Lazy clip resolution: derive a clip's bytes from its source on demand.

A *lazy clip* is a derived sub-media that stores **no** materialized
``media_bytes`` of its own.  Instead it keeps a reference to the original
source file (``media_path``) plus a *recipe* (recorded in ``origin.params``)
and reproduces its bytes on demand.  This is how reference (thin) imports avoid
duplicating the source bytes once per derived item in the dataset pickle: a
5-minute recording tiled into 30 clips stores the recipe 30 times (a few bytes
each) instead of 30 copies of the audio.

Three recipe families participate:

* **clip slices** - the media type's clipper re-slices the source bytes:

  * **audio** - ``clip_start`` / ``clip_end`` seconds, sliced via
    :func:`~vtscore.media.audio.wav.wav_slice`.
  * **image** - ``clip_box`` pixel box, cropped via Pillow.

  Text clips keep their (tiny) ``media_string`` materialized, and video clips
  are already metadata-only (they share the parent's bytes and the player
  seeks via ``clip_start`` / ``clip_end``), so neither needs lazy resolution.

* **converter output** - a converter (``document2image``, ``video2image``, …)
  rendered a derived media (a PDF page PNG, an extracted video frame) from the
  *source* file.  Keyed by the ``converter`` entry in ``origin.params`` (not by
  target media type), the recipe re-runs the converter and re-selects the exact
  sub-output recorded at import time (``converter_out_index`` /
  ``converter_n_out`` / ``converter_content_hash``) via the shared
  :func:`~vtscore.datasets.clipper_chain._select_chain_output` selector.

* **chain replays** - some :class:`~vtscore.media.clipping.cleaner.MediaCleaner`
  rewrote the payload (a silence trim, an edge trim, a whitespace pass), which
  no slice of the source reproduces, or a clipper cut a converter's output (a
  rendered page tiled, a document's text split into sentences), which the
  converter alone does not.  The recipe is the unit's whole chain trail
  (``origin.params['clipper_chain']``), led for a converter output by the
  converter itself (:func:`~vtscore.media.clipping.clip_recipe.converter_trail`): the
  source is loaded the way the import hydrated it (:func:`load_reference_source`)
  or handed it to the converter, every recorded step is re-run via
  :func:`~vtscore.datasets.clipper_chain.replay_chain_on_media`, and the
  result is served only if its content hash still equals the one the trail
  recorded.  A cleaned unit's "Original" is the same trail cut before its
  first cleaner, checked against the trail's ``original_hash``.

Chain and converter recipes are *exclusive* (:func:`recipe_is_exclusive`):
the unit's ``media_path`` names its source - the uncleaned file, the PDF a
page came from - so when a replay no longer matches (the source file was
edited, or a library upgrade changed a re-encode) the unit serves **no**
bytes, never the source file in their place, and a warning names the file.

Resolved bytes are held in process-scoped LRU caches.  Per the
no-persisted-bytes rule, these caches are purely in-memory: nothing here writes
a materialized clip back to disk or into a pickle.  Clip slices use a
count-bounded cache (payloads are small and uniform - one tile / crop);
converter output and cleaned units use a *byte-bounded* cache (a rendered page,
a video frame or a whole trimmed recording is 1-8 MB and varies by an order of
magnitude, so a count bound that is safe for clips would be unsafe here).
"""

from __future__ import annotations

import io
import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from vtscore.media.clipping.clip_recipe import converter_trail, parse_clip_box, parse_converter_recipe

log = logging.getLogger(__name__)

__all__ = [
    "CHAIN_REPLAY_TYPES",
    "LAZY_CLIP_TYPES",
    "clip_recipe",
    "embed_view",
    "has_lazy_original",
    "lazy_clip_bytes",
    "lazy_original_bytes",
    "load_reference_source",
    "recipe_is_exclusive",
]

#: Media types whose clippers re-slice bytes and therefore support lazy clips.
#: The converter branch is keyed by ``converter`` in ``origin.params`` instead,
#: so it is *not* gated on this list (a converter can target any media type).
LAZY_CLIP_TYPES = ("audio", "image")

#: Media types a reference (thin) parent is hydrated for, so that its clipper
#: and cleaner steps see a payload, and whose cleaned units can be replayed
#: from their trail.  Video is absent on purpose: its units share the parent's
#: bytes, and its cleaners only narrow ``clip_start`` / ``clip_end`` /
#: ``clip_box``, so a thin video import never needed bytes to begin with.
CHAIN_REPLAY_TYPES = ("audio", "image", "text")

# Process-scoped LRU of resolved clip bytes, keyed by (source, recipe).  Bounded
# by entry count; clip payloads are small (one tile / crop) so a few hundred
# entries stays well within memory.  Never persisted.
_CACHE_MAX = 256
_cache: "OrderedDict[tuple, bytes]" = OrderedDict()
_cache_lock = threading.Lock()


class _ByteBoundedLRU:
    """Process-scoped LRU bounded by *total held bytes*, not entry count.

    Converter output (a rendered PDF page, a decoded video frame) is large and
    non-uniform - 1-8 MB per entry, varying by an order of magnitude across
    documents - so a count bound that keeps clip slices safe could hold 1-2 GB
    here.  Bounding by summed payload size keeps memory predictable regardless
    of output size.  Never persisted; purely an in-memory recompute cache.
    """

    def __init__(self, max_bytes: int) -> None:
        self._max = max_bytes
        self._d: "OrderedDict[tuple, tuple[Any, int]]" = OrderedDict()
        self._total = 0
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Any:
        with self._lock:
            hit = self._d.get(key)
            if hit is None:
                return None
            self._d.move_to_end(key)
            return hit[0]

    def put(self, key: tuple, value: Any, nbytes: int | None = None) -> None:
        """Insert *value*, charging *nbytes* against the ceiling (``len(value)`` by default)."""
        size = len(value) if nbytes is None else nbytes
        with self._lock:
            if key in self._d:
                self._total -= self._d[key][1]
            self._d[key] = (value, size)
            self._d.move_to_end(key)
            self._total += size
            # Evict oldest until under the ceiling, but never evict the entry we
            # just inserted (a single output larger than the ceiling is kept so
            # the immediate fetch still hits; it is dropped on the next insert).
            while self._total > self._max and len(self._d) > 1:
                _, (_evicted, evicted_size) = self._d.popitem(last=False)
                self._total -= evicted_size


#: Byte ceiling for the converter-output cache (~256 MB).  Large enough to keep
#: a hot set of rendered pages / frames resident across HTTP range/scrub bursts
#: without an unbounded memory risk.
_CONV_CACHE_MAX_BYTES = 256 * 1024 * 1024
_conv_cache = _ByteBoundedLRU(_CONV_CACHE_MAX_BYTES)

#: Byte ceiling for the chain-replay memo (~128 MB).  It holds one clipper
#: step's *whole* output list per source (every tile of a recording), so that
#: the N cleaned siblings of one tiled source re-tile it once, not N times;
#: without it the dry run that gates re-lazification would be quadratic in the
#: tile count.
_REPLAY_MEMO_MAX_BYTES = 128 * 1024 * 1024
_replay_memo = _ByteBoundedLRU(_REPLAY_MEMO_MAX_BYTES)

# Replays already reported as no longer matching, so a grid of broken items
# logs each once rather than on every request.  Cleared when it grows large.
_warned: set[tuple] = set()
_WARNED_MAX = 1024


def _converter_recipe(params: dict[str, Any]) -> tuple | None:
    """Build a hashable converter recipe from ``origin.params``, or ``None``.

    Parsing is shared with the resolver's replay path via
    :func:`~vtscore.media.clipping.clip_recipe.parse_converter_recipe`; what stays local
    is the *gate*.  A recipe with no sub-output disambiguator cannot pick the
    right output, so it is treated here as "not a lazy converter media"
    (``None``) and the caller falls through to its normal resolution order,
    rather than running the converter only to have the selector refuse to
    guess.
    """
    recipe = parse_converter_recipe(params)
    if recipe is None or not recipe.is_replayable:
        return None
    return recipe.cache_key()


def _converter_output_recipe(params: dict[str, Any], media_type: Any) -> tuple | None:
    """Return the recipe of a unit whose ``params`` record a flat converter, or ``None``.

    A unit that *is* the converter's output - nothing after the converter cut
    or rewrote it - replays the converter alone (:func:`_converter_recipe`).
    One a clipper cut or a cleaner rewrote replays the converter and then the
    rest of its trail (:func:`~vtscore.media.clipping.clip_recipe.converter_trail`),
    checked like any chain replay.  ``None`` when that trail cannot be read or
    checked (a type outside :data:`CHAIN_REPLAY_TYPES`, no recorded content
    hash): the converter alone would rebuild the whole page, not the tile.
    """
    from vtscore.datasets.clipper_chain import _trail_cleaned  # noqa: PLC0415

    trail = converter_trail(params)
    if trail is None:
        return None
    after = trail[1:]
    if not _trail_cleaned(after) and all(entry["kind"] == "cleaner" for entry in after):
        return _converter_recipe(params)
    if media_type not in CHAIN_REPLAY_TYPES or not trail[-1].get("content_hash"):
        return None
    return ("chain", media_type, _trail_key(trail))


def _origin_params(media: dict[str, Any]) -> dict[str, Any]:
    """Return *media*'s ``origin.params`` (empty when it has none).

    A ``dupe_set`` representative reads through to its own origin, which every
    collapse records as the group's first member: the merge rewrote the origin
    but not the item, so its ``media_path`` and its recipe still go together.
    """
    origin = media.get("origin")
    if isinstance(origin, dict) and origin.get("importer") == "dupe_set":
        members = origin.get("members") or []
        first = members[0] if members and isinstance(members[0], dict) else {}
        origin = first.get("origin")
    params = origin.get("params", {}) if isinstance(origin, dict) else {}
    return params if isinstance(params, dict) else {}


def _is_converter_output(media: dict[str, Any]) -> bool:
    """True iff *media* came out of a flat converter, so its ``media_path`` names the file it was converted from."""
    return bool(_origin_params(media).get("converter"))


def clip_recipe(media: dict[str, Any]) -> tuple | None:
    """Return a hashable recipe describing *media*'s derivation, or ``None``.

    The recipe is read from ``origin.params`` (the channel that survives the
    pickle round-trip), not the top-level ``clip_*`` fields, so it resolves
    identically for a freshly derived media and one reopened from disk.

    Returns one of:

    * ``("converter", name, params_items, out_index, n_out, content_hash)`` -
      a converter output re-rendered from the source file.
    * ``("chain", media_type, trail_json)`` - a unit replayed from its whole
      chain trail: a cleaned unit (see :func:`_chain_trail`), or a converter
      output some clipper or cleaner then changed, whose trail the converter
      leads (see :func:`_converter_output_recipe`).
    * ``("audio", clip_start, clip_end)`` - seconds, both floats.
    * ``("image", (x1, y1, x2, y2))`` - integer pixel box.

    or ``None`` when *media* is not a recognised lazy media (no recipe, or a
    media type that doesn't re-slice bytes).

    The converter branch is checked first and is keyed by the ``converter``
    param, not the target media type: a ``document2image`` output has
    ``media_type == "image"`` but is a converter output, not an image *clip* -
    and a tile cut from it carries a ``clip_box`` relative to the rendered
    page, not to the PDF.  The chain branch comes next: a cleaned audio tile
    also carries the tile's ``clip_start`` / ``clip_end``, but slicing them out
    of the source would yield the *uncleaned* tile.

    **Archive members never lazy-slice.** A ``local_archive_member`` media may
    carry ``clip_start`` / ``clip_end`` (a windowed import), but its window is
    *display-only*: the byte routes serve the whole member and the player seeks
    within the window.  These corpora are AAC/MP4, which the WAV-only audio
    slicer cannot cut and which we deliberately do not decode server-side, so we
    return ``None`` here and let the caller fall through to whole-member byte
    serving.
    """
    from vtscore.datasets.archive_stream import archive_member_ref  # noqa: PLC0415

    if archive_member_ref(media) is not None:
        return None

    params = _origin_params(media)
    media_type = media.get("media_type")

    if params.get("converter"):
        return _converter_output_recipe(params, media_type)

    trail = _chain_trail(params, media_type)
    if trail is not None:
        return ("chain", media_type, _trail_key(trail))

    if media_type == "audio":
        cs = params.get("clip_start")
        ce = params.get("clip_end")
        if cs is None or ce is None:
            return None
        try:
            return ("audio", float(cs), float(ce))
        except (TypeError, ValueError):
            return None

    if media_type == "image":
        box = parse_clip_box(params.get("clip_box"))
        if box is None:
            return None
        return ("image", box)

    return None


def _chain_trail(params: dict[str, Any], media_type: Any) -> list[dict[str, Any]] | None:
    """Return the chain trail a cleaned unit replays from, or ``None``.

    Only a unit some cleaner actually changed needs one - an uncleaned clip is
    served by its slice recipe or the whole source - and only a payload-carrying
    type (:data:`CHAIN_REPLAY_TYPES`) can have its payload changed.  The final
    entry must carry a ``content_hash``, since a replay that cannot be checked
    is never served.
    """
    from vtscore.datasets.clipper_chain import _trail_cleaned, parse_trail  # noqa: PLC0415

    if media_type not in CHAIN_REPLAY_TYPES:
        return None
    trail = parse_trail(params.get("clipper_chain"))
    if not trail or not _trail_cleaned(trail) or not trail[-1].get("content_hash"):
        return None
    return trail


def _trail_key(trail: list[dict[str, Any]]) -> str:
    """Return a canonical, hashable encoding of *trail* for cache keys."""
    return json.dumps(trail, sort_keys=True, separators=(",", ":"))


def _payload_bytes(media: dict[str, Any] | None, media_type: str) -> bytes | None:
    """Return *media*'s embeddable payload as bytes (text UTF-8 encoded), or ``None``."""
    if media is None:
        return None
    if media_type == "text":
        text = media.get("media_string")
        return text.encode("utf-8") if isinstance(text, str) else None
    blob = media.get("media_bytes")
    return bytes(blob) if isinstance(blob, (bytes, bytearray)) else None


def _payload_size(media: dict[str, Any]) -> int:
    """Approximate the bytes a media dict holds, for the byte-bounded memo."""
    return sum(len(v) for v in media.values() if isinstance(v, (bytes, bytearray, str)))


def load_reference_source(media_type: str, path: Path) -> dict[str, Any]:
    """Return the fields a reference parent at *path* is hydrated with.

    The one loader shared by the import (which hydrates a thin parent so its
    chain can run, see ``_hydrate_reference_parents`` in
    :mod:`vtscore.datasets.stages.clipper`) and the chain recipe here (which
    replays that chain later).  Sharing it is what lets a replay reproduce the
    import byte for byte: the same ``load_media_data`` call on the same bytes,
    including text's decode and strip.  Callers must have confined *path*.
    """
    import vtscore.media as media_registry  # noqa: PLC0415

    return media_registry.get(media_type).load_media_data(path, media_bytes=path.read_bytes())


def _source_signature(media: dict[str, Any]) -> tuple[Path, tuple] | None:
    """Return a lazy unit's confined source path and its ``(path, mtime, size)`` key.

    The stat fields make an edited source a cache miss, so an edit is noticed
    on the next request rather than whenever the old bytes happen to be evicted.
    """
    from vtscore.security.path_validation import resolve_media_file_path  # noqa: PLC0415

    media_path = media.get("media_path")
    if not media_path:
        return None
    path = resolve_media_file_path(media_path)
    if path is None:
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    return path, (str(path), st.st_mtime_ns, st.st_size)


def _replay_from_source(
    path: Path,
    source_key: tuple,
    media_type: str,
    steps: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Replay *steps* on the reference source at *path*; memoise shared step outputs.

    A clipper's output list is the same for every sibling of one source, so it
    is memoised under the source plus everything recorded before it.  When the
    first step's outputs are already memoised the source is not even loaded.
    """
    from vtscore.datasets.clipper_chain import (  # noqa: PLC0415
        _run_recorded_step,
        _select_chain_output,
        replay_chain_on_media,
    )

    def step_key(index: int) -> tuple:
        entry = steps[index]
        own = _trail_key([{"kind": entry["kind"], "name": entry["name"], "params": entry.get("params") or {}}])
        return ("step", source_key, media_type, _trail_key(steps[:index]), own)

    def run_step(index: int, entry: dict[str, Any], media: dict[str, Any]) -> list[dict[str, Any]] | None:
        key = step_key(index)
        outputs = _replay_memo.get(key)
        if outputs is None:
            outputs = _run_recorded_step(index, entry, media)
            if outputs is None:
                return None
            _replay_memo.put(key, outputs, nbytes=sum(_payload_size(o) for o in outputs))
        # Copies, so a later step that edits its input in place cannot reach
        # the memo every sibling replays from.
        return [dict(o) for o in outputs]

    if steps and steps[0]["kind"] != "cleaner":
        memoised = _replay_memo.get(step_key(0))
        if memoised is not None:
            picked = _select_chain_output(memoised, steps[0])
            if picked is None:
                return None
            return replay_chain_on_media(dict(picked), steps[1:], run_step=lambda i, e, m: run_step(i + 1, e, m))

    return replay_chain_on_media(_replay_source(path, media_type, steps), steps, run_step=run_step)


def _replay_source(path: Path, media_type: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild, from the reference source at *path*, what the first of *steps* saw at import.

    A converter-led trail (a flat converter output, see
    :func:`~vtscore.media.clipping.clip_recipe.converter_trail`) starts from what a
    reference import's ``run_converters_on_folder`` handed the converter: the
    file's name and path, which the converter reads itself.  Any other trail
    starts from the parent the import hydrated (:func:`load_reference_source`).
    """
    if steps and steps[0]["kind"] == "converter":
        return {"filename": path.name, "media_path": str(path)}
    source = {"media_type": media_type, "filename": path.name, "media_path": str(path)}
    source.update(load_reference_source(media_type, path))
    return source


def _checked_replay(
    media: dict[str, Any],
    media_type: str,
    steps: list[dict[str, Any]],
    expected_hash: str | None,
    what: str,
) -> bytes | None:
    """Replay *steps* for *media* and return the payload iff it hashes to *expected_hash*.

    Results are cached (keyed by the source's stat signature, so an edited
    source misses).  A replay that runs but no longer matches is reported once
    and yields ``None``: the caller serves nothing rather than other bytes.
    """
    from vtscore.datasets.clipper_chain import _content_hash  # noqa: PLC0415

    if not expected_hash:
        return None
    sig = _source_signature(media)
    if sig is None:
        return None
    path, source_key = sig
    cache_key = ("chain", source_key, media_type, _trail_key(steps), expected_hash)
    hit = _conv_cache.get(cache_key)
    if hit is not None:
        return hit
    try:
        final = _replay_from_source(path, source_key, media_type, steps)
    except Exception:
        log.warning("lazy_clip: replaying the %s of %s failed", what, path, exc_info=True)
        return None
    payload = _payload_bytes(final, media_type)
    if final is None or payload is None or _content_hash(final) != expected_hash:
        if cache_key not in _warned:
            if len(_warned) >= _WARNED_MAX:
                _warned.clear()
            _warned.add(cache_key)
            log.warning(
                "lazy_clip: replaying %s no longer reproduces the %s recorded at import "
                "(the file was edited, or a library upgrade changed the output); serving nothing",
                path,
                what,
            )
        return None
    _conv_cache.put(cache_key, payload)
    return payload


def _original_steps(media: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str] | None:
    """Return ``(media_type, pre-clean steps, original hash)`` for a lazy Original, or ``None``."""
    from vtscore.datasets.clipper_chain import original_hash, pre_clean_steps  # noqa: PLC0415

    recipe = clip_recipe(media)
    if recipe is None or recipe[0] != "chain":
        return None
    _, media_type, trail_json = recipe
    trail = json.loads(trail_json)
    expected = original_hash(trail)
    if expected is None:
        return None
    return media_type, pre_clean_steps(trail), expected


def has_lazy_original(media: dict[str, Any]) -> bool:
    """True iff *media* is a lazy cleaned unit whose Original can be replayed.

    A unit holding its payload inline answers through its stored snapshot
    instead (see :func:`~vtscore.datasets.clipper_chain.has_original_payload`).
    """
    if media.get("media_bytes") is not None or media.get("media_string") is not None:
        return False
    if not media.get("media_path"):
        return False
    return _original_steps(media) is not None


def lazy_original_bytes(media: dict[str, Any]) -> bytes | None:
    """Return a lazy cleaned unit's *Original* payload, or ``None``.

    The Original is the trail replayed up to (not including) its first
    cleaner - for a sub-clip, the pre-clean clip rather than its source file -
    and is served only if it still hashes to the trail's ``original_hash``.
    Text comes back UTF-8 encoded.
    """
    found = _original_steps(media)
    if found is None:
        return None
    media_type, steps, expected = found
    return _checked_replay(media, media_type, steps, expected, "Original")


def recipe_is_exclusive(media: dict[str, Any]) -> bool:
    """True iff *media*'s recipe is the only legitimate source of its payload.

    A cleaned unit's ``media_path`` names the *uncleaned* source, and a
    converter output's names the file it was converted from (the PDF a page's
    text came from, the video a frame was cut out of), so when the recipe
    fails the resolvers must stop there rather than fall through to reading
    that file: no bytes, never other bytes.  A converter output whose recipe
    cannot be replayed at all is no exception - its source file still is not
    its payload.
    """
    recipe = clip_recipe(media)
    if recipe is not None:
        return recipe[0] in ("chain", "converter")
    return _is_converter_output(media)


class _LazyPayloadView(dict):
    """A copy of a lazy media whose payload is resolved from its recipe on read.

    Handed to embedders in place of the media itself (see :func:`embed_view`).
    The copy drops ``media_path`` / ``media_url``, which name the *source*,
    so an embedder can only see the unit's own payload or nothing.  The
    payload is resolved on each read rather than stored, so a bulk pass over a
    large reference dataset never holds every unit's payload at once; the LRU
    caches absorb repeat reads.  A plugin that copies the view with
    ``dict(view)`` gets no payload and no path, so it embeds nothing rather
    than the wrong file.
    """

    _PAYLOAD_KEYS = ("media_bytes", "media_string")

    def __init__(self, media: dict[str, Any]) -> None:
        super().__init__({k: v for k, v in media.items() if k not in ("media_path", "media_url")})
        self._source = media

    def _payload(self, key: str) -> Any:
        is_text = self._source.get("media_type") == "text"
        if (key == "media_string") != is_text:
            return None
        payload = lazy_clip_bytes(self._source)
        if payload is None:
            return None
        return payload.decode("utf-8", errors="replace") if is_text else payload

    def get(self, key: Any, default: Any = None) -> Any:
        if key in self._PAYLOAD_KEYS:
            value = self._payload(key)
            return default if value is None else value
        return super().get(key, default)

    def __getitem__(self, key: Any) -> Any:
        if key in self._PAYLOAD_KEYS:
            return self._payload(key)
        return super().__getitem__(key)


def embed_view(media: dict[str, Any]) -> dict[str, Any]:
    """Return what an embedder should be handed for *media*.

    *media* itself, unless it is a lazy unit (no inline payload, a recipe in
    its origin) or a converter output: then a view whose payload resolves from
    that recipe, if any.  Every embedder falls back to ``media_path`` when it
    finds no payload, and for these that names the whole source - the
    recording a tile was cut from, the photo before its edge trim, the PDF a
    page came from - so embedding the media directly would embed the wrong
    content.
    """
    if media.get("media_bytes") is not None or media.get("media_string") is not None:
        return media
    if clip_recipe(media) is None and not _is_converter_output(media):
        return media
    return _LazyPayloadView(media)


def _apply_recipe(source_bytes: bytes, recipe: tuple, media: dict[str, Any]) -> bytes | None:
    """Reproduce a derived media's bytes by applying *recipe* to *source_bytes*."""
    kind = recipe[0]
    if kind == "converter":
        return _apply_converter_recipe(source_bytes, recipe, media)
    if kind == "audio":
        from vtscore.media.audio.wav import wav_slice  # noqa: PLC0415

        return wav_slice(source_bytes, recipe[1], recipe[2])
    if kind == "image":
        from vtscore.media.image.decode import open_upright  # noqa: PLC0415

        box = recipe[1]
        # Upright decode, matching the clipper that produced the recorded box:
        # re-cropping the raw sensor bitmap would rebuild a rotated photo's clip
        # from the wrong region.
        with open_upright(io.BytesIO(source_bytes)) as img:
            fmt = img.format or "PNG"
            cropped = img.crop(box)
            buf = io.BytesIO()
            cropped.save(buf, format=fmt)
        return buf.getvalue()
    return None


def _apply_converter_recipe(source_bytes: bytes, recipe: tuple, media: dict[str, Any]) -> bytes | None:
    """Re-run a converter on the source file and re-select the recorded output.

    Rebuilds the converter's ``params`` from the recipe, hands it a minimal
    source-media dict (the whole source bytes plus filename / path), and
    selects the exact sub-output via the shared
    :func:`~vtscore.datasets.clipper_chain._select_chain_output` so the
    content-hash-first, drift-aware, refuse-to-guess semantics match the
    cross-dataset resolver.  Returns ``None`` (cache nothing, fall through)
    when the converter is unknown or no output matches - the same "better no
    bytes than wrong bytes" stance the clip branches take.
    """
    from vtscore.converters import get_converter  # noqa: PLC0415
    from vtscore.datasets.clipper_chain import _select_chain_output  # noqa: PLC0415

    _, converter_name, params_items, out_index, n_out, content_hash = recipe
    conv = get_converter(converter_name)
    if conv is None:
        log.warning("lazy_clip: unknown converter %r; cannot resolve output", converter_name)
        return None

    source_path = media.get("media_path")
    filename = Path(source_path).name if source_path else media.get("filename", "")
    source_media: dict[str, Any] = {"media_bytes": source_bytes, "filename": filename}
    if source_path:
        source_media["media_path"] = source_path

    outputs = conv.convert_normalized(source_media, dict(params_items))

    entry: dict[str, Any] = {"kind": "converter", "name": converter_name, "out_index": out_index, "n_out": n_out}
    if content_hash is not None:
        entry["content_hash"] = content_hash
    picked = _select_chain_output(outputs, entry)
    if picked is None:
        return None
    payload = picked.get("media_bytes")
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    text = picked.get("media_string")
    if isinstance(text, str):
        return text.encode("utf-8")
    return None


def _read_source_bytes(media: dict[str, Any]) -> bytes | None:
    """Read the whole source file (or URL) backing a lazy clip.

    The source reference comes off the media, so it goes through the same
    per-user confinement as a plain ``media_path`` read (see
    :func:`~vtscore.security.path_validation.resolve_media_file_path`) —
    otherwise a clip recipe would be a way around it.
    """
    from vtscore.security.path_validation import resolve_media_file_path  # noqa: PLC0415

    media_path = media.get("media_path")
    if media_path:
        path = resolve_media_file_path(media_path)
        if path is not None and path.exists():
            return path.read_bytes()
    media_url = media.get("media_url")
    if media_url:
        from vtscore.media.base import _fetch_media_url  # noqa: PLC0415

        return _fetch_media_url(media_url)
    return None


def lazy_clip_bytes(media: dict[str, Any]) -> bytes | None:
    """Return *media*'s clip bytes derived from its source, or ``None``.

    ``None`` means "not a lazy clip" (no recipe) or "source unavailable" - in
    both cases the caller falls back to its existing resolution order
    (``media_path`` whole-file read, then ``media_url``) - except for a recipe
    :func:`recipe_is_exclusive` reports, where ``None`` also means "the replay
    no longer matches" and the caller must stop.  A media that already carries
    inline ``media_bytes`` is never lazy, so callers should check that first;
    this function does not.  A cleaned text unit's payload comes back UTF-8
    encoded.
    """
    recipe = clip_recipe(media)
    if recipe is None:
        return None
    if recipe[0] == "chain":
        _, media_type, trail_json = recipe
        trail = json.loads(trail_json)
        return _checked_replay(media, media_type, trail, trail[-1].get("content_hash"), "cleaned payload")

    is_converter = recipe[0] == "converter"
    source_key = media.get("media_path") or media.get("media_url")
    cache_key = (source_key, recipe) if source_key else None

    if cache_key is not None:
        hit = _conv_cache.get(cache_key) if is_converter else _count_cache_get(cache_key)
        if hit is not None:
            return hit

    source = _read_source_bytes(media)
    if source is None:
        return None

    try:
        out = _apply_recipe(source, recipe, media)
    except Exception:
        log.warning("lazy_clip: failed to apply recipe %r to %s", recipe, source_key, exc_info=True)
        return None
    if out is None:
        return None

    if cache_key is not None:
        if is_converter:
            _conv_cache.put(cache_key, out)
        else:
            _count_cache_put(cache_key, out)
    return out


def _count_cache_get(cache_key: tuple) -> bytes | None:
    """Look up a clip slice in the count-bounded LRU."""
    with _cache_lock:
        hit = _cache.get(cache_key)
        if hit is not None:
            _cache.move_to_end(cache_key)
        return hit


def _count_cache_put(cache_key: tuple, value: bytes) -> None:
    """Insert a clip slice into the count-bounded LRU, evicting the oldest."""
    with _cache_lock:
        _cache[cache_key] = value
        _cache.move_to_end(cache_key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
