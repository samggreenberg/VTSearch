"""One parser for the ``origin.params`` clip dialects, shared by both replay paths.

An ``origin.params`` dict is the canonical, pickle-surviving record of how a
derived media was produced (see the origins-are-canonical rule in
``CLAUDE.md``).  Two independent code paths rederive content from it:

* :mod:`vtscore.media.lazy_clip` — serves a lazy clip's **display bytes** on
  demand, out of a process-scoped LRU, for the HTTP byte routes.
* :func:`vtscore.detectors.resolver._apply_clip_and_embed` — rederives a label's
  content on a resolved file and **embeds** it, for cross-dataset training.

Those two have genuinely different contracts (cached bytes vs. an embedding via
a tempfile; a media dict vs. an already-resolved ``Path``; fall through to
whole-file *serving* vs. whole-file *embedding*), so they stay separate.  What
they must **not** have separately is a reading of the params dialect: two
parsers of one wire format is exactly how a replay path drifts into rederiving
the wrong bytes, silently, on the load-bearing path of the invariant.

So this module owns the *parsing* only — params in, a validated recipe out —
and each caller keeps its own replay and fallback policy:

* :func:`parse_clip_box` — the ``clip_box`` pixel-region dialect.
* :func:`parse_converter_recipe` — the flat ``converter`` /
  ``converter_param_<key>`` / ``converter_out_index`` / ``converter_n_out`` /
  ``converter_content_hash`` dialect stamped by the importer-level converter
  path (``run_converters_on_folder``), returned as a
  :class:`FlatConverterRecipe` that can render itself into either caller's
  shape — a hashable cache key or a one-step chain entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "FlatConverterRecipe",
    "parse_clip_box",
    "parse_converter_recipe",
]


def parse_clip_box(raw: Any) -> tuple[int, int, int, int] | None:
    """Parse a ``clip_box`` into a 4-int pixel tuple, or ``None`` if malformed.

    Accepts the ``"x1,y1,x2,y2"`` string stored in ``origin.params`` as well
    as a native list/tuple (the in-memory form on a freshly clipped media).

    Returns ``None`` rather than raising on every malformed shape — wrong
    arity, a non-numeric component, a value that is neither string nor
    sequence — so a caller can fall through to whole-file handling instead of
    cropping to a half-parsed region.
    """
    if isinstance(raw, (list, tuple)):
        parts = list(raw)
    elif isinstance(raw, str):
        parts = [p for p in raw.split(",") if p != ""]
    else:
        return None
    if len(parts) != 4:
        return None
    try:
        return (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])), int(float(parts[3])))
    except (TypeError, ValueError):
        return None


def _as_int(raw: Any) -> int | None:
    """Coerce a params value to ``int``, or ``None`` when it isn't one.

    ``OverflowError`` is caught alongside the usual coercion failures: a
    non-finite float survives a pickle round-trip, and ``int(float("inf"))``
    raises it.  Everything in this module degrades to ``None`` rather than
    raising, so the replay paths can fall through to whole-file handling
    instead of propagating out of a byte route.
    """
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError, OverflowError):
        return None


@dataclass(frozen=True)
class FlatConverterRecipe:
    """A flat (non-chain) converter origin, parsed.

    ``out_index`` / ``n_out`` are ``None`` when absent *or* uncoercible — a
    garbled index is indistinguishable from a missing one for selection
    purposes, and both must leave the recorded output unpicked rather than
    picked wrongly.
    """

    name: str
    params: dict[str, Any]
    out_index: int | None
    n_out: int | None
    content_hash: str | None

    @property
    def is_replayable(self) -> bool:
        """True iff at least one sub-output disambiguator was recorded.

        Without an ``out_index`` or a ``content_hash``,
        :func:`~vtscore.datasets.clipper_chain._select_chain_output` has no
        handle to pick the recorded output with and refuses to guess, so the
        replay can only fail.  Callers check this to skip the converter run
        entirely and fall through to their whole-file path — the same outcome
        the selector would reach, reached without the work.
        """
        return self.out_index is not None or self.content_hash is not None

    def cache_key(self) -> tuple:
        """Render as a hashable tuple, for use as an LRU cache key."""
        return (
            "converter",
            self.name,
            tuple(sorted(self.params.items())),
            self.out_index,
            self.n_out,
            self.content_hash,
        )

    def chain_step(self) -> dict[str, Any]:
        """Render as a one-step ``clipper_chain`` entry.

        Lets the flat dialect reuse
        :func:`~vtscore.datasets.clipper_chain.replay_chain_on_file` and the
        shared sub-output selector, so a reference-converted media replays
        exactly like a chain-converted one.  Optional keys are omitted rather
        than set to ``None``, because the selector treats a present-but-``None``
        disambiguator as "not recorded" only by convention and a missing key
        says it unambiguously.
        """
        entry: dict[str, Any] = {"kind": "converter", "name": self.name, "params": dict(self.params)}
        if self.out_index is not None:
            entry["out_index"] = self.out_index
        if self.n_out is not None:
            entry["n_out"] = self.n_out
        if self.content_hash is not None:
            entry["content_hash"] = self.content_hash
        return entry


#: Prefix under which the flat converter dialect stores the converter's own params.
_PARAM_PREFIX = "converter_param_"


def parse_converter_recipe(params: dict[str, Any]) -> FlatConverterRecipe | None:
    """Parse a flat converter origin out of *params*, or ``None``.

    ``None`` means "no converter recorded here" — the params describe a plain
    clip, or nothing derived at all.  A recipe that parsed but carries no
    disambiguator is *not* ``None``; it comes back with
    :attr:`~FlatConverterRecipe.is_replayable` false, so a caller can tell
    "not a converter media" from "a converter media we cannot replay".
    """
    name = params.get("converter")
    if not name:
        return None
    return FlatConverterRecipe(
        name=str(name),
        params={k[len(_PARAM_PREFIX) :]: v for k, v in params.items() if k.startswith(_PARAM_PREFIX)},
        out_index=_as_int(params.get("converter_out_index")),
        n_out=_as_int(params.get("converter_n_out")),
        content_hash=params.get("converter_content_hash"),
    )
