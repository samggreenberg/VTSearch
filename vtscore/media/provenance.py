"""Human-readable provenance for media derived by converters and clippers.

A converter output or a clip records *how it was made* in ``origin.params``:
``converter`` / ``source_file`` / ``source_path`` / ``converter_param_<key>``
/ ``converter_out_index`` for converter output (see
:mod:`vtscore.converters.runner`), and ``clipper_chain`` /
``clipper`` / ``clipper_<key>`` / ``clip_start`` … for chain output (see
:func:`vtscore.datasets.clipper_chain._stamp_origin`).  Those keys are the
machine-readable recipe :mod:`vtscore.media.lazy_clip` replays to reproduce
the bytes on demand; they are not what a human wants to read in the labeling
UI's metadata grid.

This module turns the recipe into three curated display fields:

``Source``
    The original file the item was derived from - e.g. the video a frame was
    extracted from, or the recording an audio clip was cut out of.

``Derived Via``
    The converter / clipper chain that produced it, rendered with each step's
    display name and effective parameters
    (``"Video → Images (n_clips=2) → Object Crop"``).

``Imported Via``
    The importer that brought the corpus in, with its locator params
    (``"Manifest (paths_file=/data/list.txt)"``).  Present on every media,
    derived or not.

Each is **one row**, deliberately.  Flattening ``origin.params`` key-by-key
into a per-item grid reads as if every key were a property of *this item*,
which they mostly are not: a dataset-level import knob (``size=60``) is not a
fact about one image.  Folding them into a single "imported via" line keeps
the provenance visible without that false implication, and keeps a grid of
half a dozen fields from doubling in height.

The enriched *label export* still flattens the full ``origin.params``
key-by-key (see ``vtsearch.routes.labels.vote._build_entry_metadata``): an
export is a machine-facing artifact with opt-in columns, where the raw recipe
is the point.
"""

from __future__ import annotations

from typing import Any

#: Display label for the file a derived media came from.
SOURCE_LABEL = "Source"

#: Display label for the converter / clipper chain that produced a media.
DERIVED_VIA_LABEL = "Derived Via"

#: Display label for the importer that brought the corpus in.
IMPORTED_VIA_LABEL = "Imported Via"

#: ``origin.params`` keys left out of the ``Imported Via`` line.  These are
#: either represented by the curated fields above, duplicated by a dedicated
#: ``display_metadata`` entry (``clip_start`` → "Clip Start"), a machine-only
#: replay disambiguator (``converter_content_hash``), or redundant with a
#: field the media already carries (``media_type``).
HIDDEN_ORIGIN_PARAMS = frozenset(
    {
        "converter",
        "converter_content_hash",
        "converter_n_out",
        "converter_out_index",
        "clip_box",
        "clip_end",
        "clip_index",
        "clip_start",
        "clipper",
        "media_type",
        "source_file",
        "source_path",
        "source_specs",
    }
)

#: ``origin.params`` key prefixes dropped alongside :data:`HIDDEN_ORIGIN_PARAMS`.
#: ``clipper_`` covers both ``clipper_chain`` and every ``clipper_<param>``.
HIDDEN_ORIGIN_PARAM_PREFIXES = ("converter_param_", "clipper_")


def _plugin_display_name(kind: str, name: str) -> str:
    """Return the display name of the ``kind`` plugin called *name*.

    Falls back to *name* itself when the plugin isn't registered (a dataset
    imported by a plugin that has since been removed still renders something
    useful).
    """
    plugin: Any = None
    try:
        if kind == "converter":
            from vtscore.converters import get_converter  # noqa: PLC0415

            plugin = get_converter(name)
        elif kind == "importer":
            from vtscore.datasets.importers import get_importer  # noqa: PLC0415

            plugin = get_importer(name)
        else:
            from vtscore.media import get_clipper  # noqa: PLC0415

            plugin = get_clipper(name)
    except (KeyError, ImportError):
        return name
    if plugin is None:
        return name
    return getattr(plugin, "display_name", "") or name


def _describe_step(kind: str, name: str, params: dict[str, Any]) -> str:
    """Render one chain step as ``"Display Name (key=value, …)"``."""
    label = _plugin_display_name(kind, name)
    if not params:
        return label
    inner = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{label} ({inner})"


def _prefixed_params(params: dict[str, Any], prefix: str, *, skip: str = "") -> dict[str, Any]:
    """Collect ``{prefix}<key> → value`` entries back into a bare params dict."""
    return {k[len(prefix) :]: v for k, v in params.items() if k.startswith(prefix) and k != skip}


def _legacy_clipper_name(params: dict[str, Any]) -> str:
    """Return the stamped single-clipper name, ignoring no-op ``*_default`` stamps.

    A ``*_default`` clipper doesn't slice anything - it only records the
    default clipper's name and resolved parameters in every origin (see
    :func:`vtscore.datasets.stages.clipper._stamp_default_clipper`).  Treating
    it as a derivation would label every plainly-imported media as "Derived
    Via <default clipper>", which is noise rather than provenance.
    """
    name = str(params.get("clipper") or "")
    if name.endswith("_default"):
        return ""
    return name


def _describe_derivation(params: dict[str, Any]) -> str:
    """Render the full converter / clipper chain that produced a media."""
    from vtscore.datasets.clipper_chain import parse_trail  # noqa: PLC0415

    trail = parse_trail(params.get("clipper_chain"))
    if trail:
        return " → ".join(_describe_step(s["kind"], s["name"], dict(s.get("params") or {})) for s in trail)

    # No chain field: fall back to the flat single-step encodings.  A converter
    # output that was later stamped by a single clipper carries both, in that
    # order.
    steps: list[str] = []
    converter = str(params.get("converter") or "")
    if converter:
        steps.append(_describe_step("converter", converter, _prefixed_params(params, "converter_param_")))
    clipper = _legacy_clipper_name(params)
    if clipper:
        steps.append(_describe_step("clipper", clipper, _prefixed_params(params, "clipper_", skip="clipper_chain")))
    return " → ".join(steps)


def _describe_source(media: dict[str, Any], origin: dict[str, Any], params: dict[str, Any]) -> str:
    """Return the original file *media* was derived from, or ``""``.

    Converter output records its own source explicitly: ``source_path`` (the
    resolved absolute path) with ``source_file`` (the scan-relative name) as a
    fallback for datasets imported before ``source_path`` was stamped.  Chain
    output inherits the parent's ``origin_name`` instead - the clipper chain
    copies it onto every clip precisely so the trail back to the parent
    survives.
    """
    if origin.get("importer") == "converter":
        explicit = params.get("source_path") or params.get("source_file")
        if explicit:
            return str(explicit)
    inherited = media.get("origin_name") or media.get("media_path") or ""
    return str(inherited)


def _visible_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return *params* minus the machine-only recipe keys."""
    return {
        k: v
        for k, v in params.items()
        if k not in HIDDEN_ORIGIN_PARAMS and not k.startswith(HIDDEN_ORIGIN_PARAM_PREFIXES)
    }


def _describe_import(origin: dict[str, Any], params: dict[str, Any]) -> str:
    """Render the importer that brought the corpus in, with its locator params.

    Converter output carries the *converter's* recipe in its own params, not
    the import's, so it reports the parent importer and the ``parent_<key>``
    locators the runner copied over instead - otherwise every extracted frame
    would claim it was "imported via converter", which says nothing about
    where the corpus came from.
    """
    importer = str(origin.get("importer") or "")
    if importer == "converter":
        importer = str(params.get("parent_importer") or "")
        params = _prefixed_params(params, "parent_")
        params.pop("importer", None)
    else:
        params = _visible_params(params)
    if not importer:
        return ""
    return _describe_step("importer", importer, params)


def provenance_metadata(media: dict[str, Any]) -> dict[str, str]:
    """Return the curated provenance entries for *media*.

    ``Source`` and ``Derived Via`` are present only for media produced by a
    converter or a clipper chain - a plainly imported file is its own source,
    so labelling it as derived would be misleading.  ``Imported Via`` is
    present whenever the origin names an importer.
    """
    origin = media.get("origin")
    if not isinstance(origin, dict):
        return {}
    params = origin.get("params") or {}
    if not isinstance(params, dict):
        return {}

    result: dict[str, str] = {}
    derived_via = _describe_derivation(params)
    if derived_via:
        source = _describe_source(media, origin, params)
        if source:
            result[SOURCE_LABEL] = source
        result[DERIVED_VIA_LABEL] = derived_via
    imported_via = _describe_import(origin, params)
    if imported_via:
        result[IMPORTED_VIA_LABEL] = imported_via
    return result
