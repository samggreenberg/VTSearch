"""Clipper chain: ordered list of converter/clipper steps applied at load time.

See ``docs/plans/clipper-chain.md`` for the design. A chain is a list of
dicts of the form::

    [
      {"kind": "converter", "name": "document2text", "params": {}},
      {"kind": "clipper",   "name": "text_token_window", "params": {"window": 512}},
    ]

Each step's input media type must match the previous step's output media
type (or, for step 0, the dataset's source media type). The final step's
output type becomes the dataset's media type.

Single-clipper loads are normalised to a length-1 chain so the runner has
one code path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from marshmallow import ValidationError

log = logging.getLogger(__name__)

ChainStep = dict[str, Any]


def _content_hash(clip: dict[str, Any]) -> str | None:
    """Return a short md5 of a clip's payload, or None if no payload is present.

    Used by the chain trail to disambiguate sub-clips when the runtime
    output count or order drifts from the load-time recording (e.g. a
    source file changed, a converter library version changed).
    """
    s = clip.get("media_string")
    if isinstance(s, str) and s:
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]
    b = clip.get("media_bytes")
    if isinstance(b, (bytes, bytearray)) and b:
        return hashlib.md5(bytes(b)).hexdigest()[:12]
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def normalise_chain(steps: list[ChainStep] | None) -> list[ChainStep]:
    """Return a list of step dicts, dropping pure ``*_default`` clipper no-ops.

    Accepts ``None`` (treated as empty) and tolerates ``params`` missing
    on individual steps. Raises ``ValueError`` on a step missing ``kind``
    or ``name``.
    """
    if not steps:
        return []
    out: list[ChainStep] = []
    for i, raw in enumerate(steps):
        if not isinstance(raw, dict):
            raise ValueError(f"chain step {i} is not a dict: {raw!r}")
        kind = raw.get("kind")
        name = raw.get("name")
        if kind not in ("clipper", "converter"):
            raise ValueError(f"chain step {i}: kind must be 'clipper' or 'converter', got {kind!r}")
        if not name:
            raise ValueError(f"chain step {i}: missing 'name'")
        params = raw.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(f"chain step {i}: 'params' must be a dict, got {type(params).__name__}")
        # Drop pure pass-through clipper steps — same convention used
        # elsewhere in the codebase (`*_default` is a no-op).
        if kind == "clipper" and isinstance(name, str) and name.endswith("_default"):
            continue
        out.append({"kind": kind, "name": str(name), "params": dict(params)})
    return out


def validate_chain(steps: list[ChainStep], source_media_type: str) -> str:
    """Validate a chain and return its final output media type.

    Each step's input type must match the previous step's output type.
    Raises ``ValueError`` on unknown plugin names or type mismatches.
    Accepts the empty chain (returns *source_media_type* unchanged).
    """
    from vtscore.converters import get_converter
    from vtscore.media import get_clipper

    current = source_media_type
    for i, step in enumerate(steps):
        kind = step["kind"]
        name = step["name"]
        if kind == "clipper":
            try:
                clipper = get_clipper(name)
            except KeyError as e:
                raise ValueError(f"chain step {i}: unknown clipper {name!r}") from e
            in_type = clipper.media_type
            out_type = clipper.media_type
        else:
            conv = get_converter(name)
            if conv is None:
                raise ValueError(f"chain step {i}: unknown converter {name!r}")
            try:
                conv.validate_params(step.get("params") or {})
            except ValidationError as e:
                raise ValueError(f"chain step {i} (converter {name!r}): invalid params: {e.messages}") from e
            in_type = conv.source_type
            out_type = conv.target_type
        if in_type != current:
            raise ValueError(
                f"chain step {i} ({kind} {name!r}): expects input type {in_type!r} "
                f"but previous step produces {current!r}"
            )
        current = out_type
    return current


# ---------------------------------------------------------------------------
# Apply (dataset-load path)
# ---------------------------------------------------------------------------


_CLIPPER_BASE_KEYS = {
    "name",
    "display_name",
    "media_type",
    "parameters",
    "description",
    "creation_questions",
}


def _resolved_clipper_params(clipper) -> dict[str, Any]:
    """Return the clipper's *effective* parameter dict (post ``with_params``).

    Reads from ``to_dict()`` and strips the base/metadata keys, matching the
    convention already used by ``_apply_clipper`` in ``load_pipeline.py``.
    """
    d = clipper.to_dict()
    return {k: v for k, v in d.items() if k not in _CLIPPER_BASE_KEYS}


def _run_clipper_step(media: dict[str, Any], step: ChainStep) -> tuple[list[dict[str, Any]], list[ChainStep]]:
    """Run a clipper step on one media. Returns (outputs, per-output trail entries)."""
    from vtscore.media import get_clipper

    base = get_clipper(step["name"])
    if step.get("params"):
        base = base.with_params(step["params"])
    resolved = base.resolve_for_media(media)
    effective = _resolved_clipper_params(resolved)
    outputs = resolved.clip(media)
    n_out = len(outputs)
    trail_entries: list[ChainStep] = []
    for idx, clip in enumerate(outputs):
        entry: ChainStep = {
            "kind": "clipper",
            "name": resolved.name,
            "params": dict(effective),
            "out_index": idx,
            "n_out": n_out,
        }
        if clip.get("clip_start") is not None:
            entry["clip_start"] = str(clip["clip_start"])
        if clip.get("clip_end") is not None:
            entry["clip_end"] = str(clip["clip_end"])
        if clip.get("clip_box") is not None:
            entry["clip_box"] = ",".join(str(v) for v in clip["clip_box"])
        if clip.get("clip_index") is not None:
            entry["clip_index"] = str(clip["clip_index"])
        ch = _content_hash(clip)
        if ch is not None:
            entry["content_hash"] = ch
        trail_entries.append(entry)
    return outputs, trail_entries


def _run_converter_step(media: dict[str, Any], step: ChainStep) -> tuple[list[dict[str, Any]], list[ChainStep]]:
    """Run a converter step on one media. Returns (outputs, per-output trail entries)."""
    from vtscore.converters import get_converter

    conv = get_converter(step["name"])
    if conv is None:
        raise ValueError(f"unknown converter {step['name']!r}")
    outputs = conv.convert_normalized(media, step.get("params") or {})
    target = conv.target_type
    n_out = len(outputs)
    trail_entries: list[ChainStep] = []
    for idx, clip in enumerate(outputs):
        clip.setdefault("type", target)
        entry: ChainStep = {
            "kind": "converter",
            "name": conv.name,
            "params": dict(step.get("params") or {}),
            "out_index": idx,
            "n_out": n_out,
        }
        ch = _content_hash(clip)
        if ch is not None:
            entry["content_hash"] = ch
        trail_entries.append(entry)
    return outputs, trail_entries


def _stamp_origin(  # noqa: C901
    clip: dict[str, Any],
    parent_origin: dict[str, Any] | None,
    parent_origin_name: str,
    chain_trail: list[ChainStep],
    *,
    is_sub_item: bool,
) -> None:
    """Stamp ``origin.params['clipper_chain']`` and legacy single-clipper keys.

    Mirrors the encoding documented in ``docs/plans/clipper-chain.md``.
    ``is_sub_item`` matches the legacy ``is_real_clip`` flag: ``True`` when
    the parent produced more than one final clip (or any clip whose chain
    crosses media types). Controls whether the legacy ``clip_index`` field
    is stamped.
    """
    if isinstance(parent_origin, dict):
        clip["origin"] = dict(parent_origin)
        clip["origin"]["params"] = dict(parent_origin.get("params", {}))
    else:
        clip["origin"] = {"params": {}}
    if not clip.get("origin_name"):
        clip["origin_name"] = parent_origin_name

    params = clip["origin"]["params"]
    params["clipper_chain"] = json.dumps(chain_trail, separators=(",", ":"))

    # Legacy keys describe the *last clipper step* in the chain so existing
    # readers (input_spec, the legacy _apply_clip_and_embed branches, the
    # registry's `clipper` column) keep working. If the chain has no
    # clipper steps (converter-only chain), we leave the legacy keys
    # unstamped — the chain field is the only record.
    last_clipper: ChainStep | None = None
    for entry in reversed(chain_trail):
        if entry["kind"] == "clipper":
            last_clipper = entry
            break
    if last_clipper is None:
        return
    params["clipper"] = last_clipper["name"]
    for pk, pv in (last_clipper.get("params") or {}).items():
        params[f"clipper_{pk}"] = str(pv)
    if "clip_start" in last_clipper:
        params["clip_start"] = last_clipper["clip_start"]
    if "clip_end" in last_clipper:
        params["clip_end"] = last_clipper["clip_end"]
    if "clip_box" in last_clipper:
        params["clip_box"] = last_clipper["clip_box"]
    if is_sub_item:
        params["clip_index"] = str(last_clipper["out_index"])


def apply_chain_to_clips(  # noqa: C901
    clips_dict: dict[int, dict[str, Any]],
    steps: list[ChainStep],
    *,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> tuple[str, list[bool]] | None:
    """Apply *steps* to every media in *clips_dict*, replacing them in place.

    Returns ``(final_media_type, needs_recompute)`` where ``needs_recompute``
    is a per-clip boolean (parallel to ``clips_dict.values()``) marking
    clips whose MD5/embedding/thumbnail should be recomputed (any clip
    descended from a parent that produced more than one final output, or
    any clip whose chain crosses media types). Returns ``None`` if
    *steps* is empty (in which case ``clips_dict`` is left untouched).

    Each clip carries a trail of ``ChainStep`` entries (one per chain step)
    in its origin under ``params['clipper_chain']`` as a JSON string. The
    last clipper step is additionally stamped under the legacy keys so
    pre-chain readers keep working.

    Like the legacy single-clipper path, this **does not** recompute MD5s,
    thumbnails, or embeddings — the caller (``_apply_clipper`` in
    ``load_pipeline.py``) handles those in a single batched pass over the
    final clip list.
    """
    if not steps:
        return None

    # Each carrier tracks (media, trail_so_far, parent_origin, parent_name,
    # parent_index). parent_index identifies the original media this
    # carrier descends from so we can count outputs per parent at the
    # end and mark sub-items for recomputation.
    chain_changes_type = any(s["kind"] == "converter" for s in steps)
    carriers: list[tuple[dict[str, Any], list[ChainStep], dict[str, Any] | None, str, int]] = []
    for parent_idx, media in enumerate(clips_dict.values()):
        carriers.append(
            (
                media,
                [],
                media.get("origin") if isinstance(media.get("origin"), dict) else None,
                media.get("origin_name", "") or media.get("filename", ""),
                parent_idx,
            )
        )

    for step in steps:
        next_carriers: list[tuple[dict[str, Any], list[ChainStep], dict[str, Any] | None, str, int]] = []
        kind = step["kind"]
        n_in = len(carriers)
        # Phase name matches the legacy single-clipper progress contract
        # ("clipping" for clipper steps, "converting" for converter steps)
        # so the load-pipeline progress wrapper can keep its existing
        # phase-to-message mapping unchanged.
        phase = "clipping" if kind == "clipper" else "converting"
        for c_idx, (media, trail, parent_origin, parent_name, parent_idx) in enumerate(carriers):
            if on_progress is not None:
                on_progress(c_idx, n_in, phase)
            if kind == "clipper":
                outputs, trail_entries = _run_clipper_step(media, step)
            else:
                outputs, trail_entries = _run_converter_step(media, step)
            for out_media, entry in zip(outputs, trail_entries):
                next_carriers.append((out_media, trail + [entry], parent_origin, parent_name, parent_idx))
        carriers = next_carriers
        if not carriers:
            break

    # Determine final media type from the last step.
    final_type: str
    last_step = steps[-1]
    if last_step["kind"] == "clipper":
        from vtscore.media import get_clipper

        final_type = get_clipper(last_step["name"]).media_type
    else:
        from vtscore.converters import get_converter

        conv = get_converter(last_step["name"])
        # `conv` is guaranteed non-None — the chain ran successfully.
        assert conv is not None
        final_type = conv.target_type

    # Count outputs per parent to flag genuine sub-items.
    outputs_per_parent: dict[int, int] = {}
    for _media, _trail, _origin, _name, parent_idx in carriers:
        outputs_per_parent[parent_idx] = outputs_per_parent.get(parent_idx, 0) + 1

    # Replace clips_dict in place with the new final clips, and build the
    # per-clip recompute flag list parallel to clips_dict iteration order.
    clips_dict.clear()
    needs_recompute: list[bool] = []
    for new_id, (media, trail, parent_origin, parent_name, parent_idx) in enumerate(carriers, 1):
        # A clip is a genuine sub-item if its parent produced > 1 output
        # (the parent's MD5/embedding describes the whole, not this part)
        # or if the chain crosses media types (the parent's embedding is
        # in a different vector space).
        is_sub_item = chain_changes_type or outputs_per_parent[parent_idx] > 1
        media["id"] = new_id
        media.setdefault("type", final_type)
        _stamp_origin(media, parent_origin, parent_name, trail, is_sub_item=is_sub_item)
        clips_dict[new_id] = media
        needs_recompute.append(is_sub_item)

    return final_type, needs_recompute


# ---------------------------------------------------------------------------
# Replay (cross-dataset resolver path)
# ---------------------------------------------------------------------------


def _load_source_as_media(file_path: Path, source_media_type: str) -> dict[str, Any]:
    """Load *file_path* into a media dict matching *source_media_type*.

    Text reads as UTF-8 string; everything else reads as bytes. The dict
    contains the minimum keys that built-in clippers and converters look
    at: ``media_bytes`` / ``media_string``, ``filename``, and ``type``.
    """
    media: dict[str, Any] = {
        "media_type": source_media_type,
        "filename": file_path.name,
        "media_path": str(file_path),
    }
    if source_media_type == "text":
        media["media_string"] = file_path.read_text(encoding="utf-8", errors="replace")
    else:
        media["media_bytes"] = file_path.read_bytes()
    return media


def _output_matches_entry(out: dict[str, Any], entry: ChainStep) -> bool:
    """Return True iff every recorded disambiguator on *entry* matches *out*.

    Only fields actually recorded in *entry* are checked — missing fields
    don't constrain the match. Returns True when no disambiguators are
    recorded (the caller must decide what to do with an empty constraint
    set, since that case can't distinguish siblings).
    """
    cs = entry.get("clip_start")
    if cs is not None and str(out.get("clip_start")) != str(cs):
        return False
    ce = entry.get("clip_end")
    if ce is not None and str(out.get("clip_end")) != str(ce):
        return False
    cb = entry.get("clip_box")
    if cb is not None and ",".join(str(v) for v in (out.get("clip_box") or [])) != str(cb):
        return False
    ci = entry.get("clip_index")
    if ci is not None and str(out.get("clip_index")) != str(ci):
        return False
    ch = entry.get("content_hash")
    if ch is not None and _content_hash(out) != ch:
        return False
    return True


def _entry_has_disambiguators(entry: ChainStep) -> bool:
    """True iff *entry* carries any field that can distinguish sub-outputs."""
    return any(entry.get(k) is not None for k in ("clip_start", "clip_end", "clip_box", "clip_index", "content_hash"))


def _select_chain_output(  # noqa: C901
    outputs: list[dict[str, Any]],
    entry: ChainStep,
) -> dict[str, Any] | None:
    """Pick the sub-output recorded by *entry* from a replay's *outputs* list.

    Selection prefers content-level matching (boundary fields, ``clip_index``,
    ``content_hash``) over positional matching, because ``out_index`` is only
    meaningful when the clipper/converter produces the same outputs in the
    same order at replay time. Returns ``None`` rather than silently
    selecting an arbitrary output when nothing matches — the caller treats
    that as an embed failure, which is strictly better than training on
    the wrong sub-clip's embedding.
    """
    if not outputs:
        return None

    name = entry.get("name", "?")
    idx = entry.get("out_index")
    n_out_recorded = entry.get("n_out")
    n_out_now = len(outputs)
    has_disambiguators = _entry_has_disambiguators(entry)

    drift = isinstance(n_out_recorded, int) and n_out_recorded != n_out_now
    if drift:
        log.warning(
            "clipper_chain: replay output count drift for %r (recorded=%d, now=%d) — falling back to content matching",
            name,
            n_out_recorded,
            n_out_now,
        )

    # Prefer content matching when we have anything to match against AND
    # either the count drifted or the indexed pick wouldn't match.
    if has_disambiguators:
        in_range_pick = outputs[idx] if isinstance(idx, int) and 0 <= idx < n_out_now else None
        if in_range_pick is not None and not drift and _output_matches_entry(in_range_pick, entry):
            return in_range_pick
        # Either the index is out of range, the count drifted, or the indexed
        # pick disagrees with the recorded disambiguators. Search every output
        # for a full disambiguator match.
        candidates = [out for out in outputs if _output_matches_entry(out, entry)]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            log.warning(
                "clipper_chain: no replay output matches recorded disambiguators for %r (idx=%r, n_out=%r→%d)",
                name,
                idx,
                n_out_recorded,
                n_out_now,
            )
            return None
        log.warning(
            "clipper_chain: %d replay outputs match recorded disambiguators for %r — ambiguous, refusing to guess",
            len(candidates),
            name,
        )
        return None

    # No disambiguators recorded — the only handle we have is out_index.
    if isinstance(idx, int) and 0 <= idx < n_out_now and not drift:
        return outputs[idx]

    log.warning(
        "clipper_chain: cannot select replay output for %r — no disambiguators recorded and "
        "positional index unusable (idx=%r, n_out=%r→%d)",
        name,
        idx,
        n_out_recorded,
        n_out_now,
    )
    return None


_FINAL_EXT_BY_TYPE = {
    "text": ".txt",
    "audio": ".wav",
    "image": ".png",
    "video": ".mp4",
    "document": ".pdf",
}


def replay_chain_on_file(  # noqa: C901
    file_path: Path,
    steps: list[ChainStep],
    embedder_name: str = "",
) -> tuple[Any, bytes | None] | None:
    """Re-run a chain against a source file and embed the final clip.

    Used by the resolver replay path to reproduce the exact sub-clip the
    label originally referenced. Returns ``(embedding, content_bytes)``
    where ``content_bytes`` is the final clip's bytes (after the last
    chain step), or ``None`` if any step produced no output or the embed
    failed.

    ``content_bytes`` is ``None`` when the final chain step is a *video*
    clipper, because video clippers are metadata-only (they record
    ``clip_start`` / ``clip_end`` but do not slice the underlying bytes,
    so the clip's bytes equal the parent's bytes — caller should switch
    to a boundary-tag MD5 scheme for those clips).

    The chain must start with a step whose input type matches the source
    file's media type (the resolver infers this from the trail's first
    step).
    """
    from vtscore.converters import get_converter
    from vtscore.detectors.resolver import embed_file
    from vtscore.media import get_clipper

    if not steps:
        return None

    # Infer the source media type from the first step.
    first = steps[0]
    if first["kind"] == "clipper":
        source_type = get_clipper(first["name"]).media_type
    else:
        conv = get_converter(first["name"])
        if conv is None:
            return None
        source_type = conv.source_type

    media = _load_source_as_media(file_path, source_type)
    current_type = source_type

    for entry in steps:
        if entry["kind"] == "clipper":
            clipper = get_clipper(entry["name"])
            params = entry.get("params") or {}
            if params:
                clipper = clipper.with_params(params)
            outputs = clipper.clip(media)
            current_type = clipper.media_type
        else:
            conv = get_converter(entry["name"])
            if conv is None:
                return None
            outputs = conv.convert_normalized(media, entry.get("params") or {})
            current_type = conv.target_type
        picked = _select_chain_output(outputs, entry)
        if picked is None:
            return None
        media = picked

    # Write the final clip to a tempfile and embed it.
    ext = _FINAL_EXT_BY_TYPE.get(current_type, ".bin")
    fd, tmp = tempfile.mkstemp(suffix=ext)
    try:
        if current_type == "text":
            content = (media.get("media_string") or "").encode("utf-8")
        else:
            content = media.get("media_bytes") or b""
        os.write(fd, content)
        os.close(fd)
        embedding = embed_file(Path(tmp), current_type, embedder_name)
        if embedding is None:
            return None
        # Video clippers are metadata-only — the clip's bytes equal the
        # parent's, so a caller hashing ``content`` would collide across
        # distinct clips of the same parent.  Signal metadata-only by
        # returning ``None`` for content_bytes; the caller falls back to
        # parent-bytes + boundary-tag MD5.
        if current_type == "video":
            return embedding, None
        return embedding, content
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Trail parsing
# ---------------------------------------------------------------------------


def parse_trail(raw: Any) -> list[ChainStep] | None:
    """Decode a ``params['clipper_chain']`` value into a step list.

    Accepts a JSON string (the on-disk encoding) or a list (when callers
    construct one in memory). Returns ``None`` on malformed input rather
    than raising — the resolver falls back to the legacy single-clipper
    path on miss.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        data = raw
    elif isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return None
    else:
        return None
    if not isinstance(data, list):
        return None
    out: list[ChainStep] = []
    for entry in data:
        if not isinstance(entry, dict):
            return None
        if entry.get("kind") not in ("clipper", "converter"):
            return None
        if not entry.get("name"):
            return None
        out.append(entry)
    return out
