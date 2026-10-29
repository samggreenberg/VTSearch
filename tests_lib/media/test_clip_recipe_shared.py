"""The two clip-replay paths must read ``origin.params`` identically.

``vtscore.media.clipping.lazy_clip`` (display bytes, cached, for the HTTP byte routes)
and ``vtscore.detectors.resolver`` (rederive + embed, for cross-dataset
training) rebuild content from the same ``origin.params`` dialects but keep
their own replay and fallback policy.  These tests pin the half that must
*not* diverge: the parse.  A disagreement here means one path serves a
different region / sub-output than the other embeds - silently, on the
load-bearing path of the origins-are-canonical invariant.

See ``vtscore/media/clipping/clip_recipe.py``, and issue #3378.
"""

from __future__ import annotations

import json

import pytest

from vtscore.detectors.resolver import _converter_origin_to_chain
from vtscore.media.clipping.clip_recipe import converter_trail, parse_clip_box, parse_converter_recipe
from vtscore.media.clipping.lazy_clip import _converter_recipe, clip_recipe


def _converter_params(**overrides) -> dict:
    """A flat ``run_converters_on_folder`` origin's params, with overrides."""
    params = {
        "converter": "document2image",
        "converter_param_dpi": "150",
        "converter_param_fmt": "png",
        "converter_out_index": "3",
        "converter_n_out": "10",
        "converter_content_hash": "abc123def456",
    }
    params.update(overrides)
    return {k: v for k, v in params.items() if v is not None}


class TestParseClipBox:
    """The ``clip_box`` pixel-region dialect."""

    @pytest.mark.parametrize(
        "raw",
        ["1,2,30,40", "1.0,2.0,30.0,40.0", [1, 2, 30, 40], (1.7, 2.2, 30.9, 40.4)],
    )
    def test_accepts_every_recorded_spelling(self, raw):
        # Floats truncate toward zero, matching int(float(...)) on the wire form.
        assert parse_clip_box(raw) == (1, 2, 30, 40)

    def test_ignores_empty_segments_in_the_string_form(self):
        assert parse_clip_box("1,2,30,40,") == (1, 2, 30, 40)

    @pytest.mark.parametrize(
        "raw",
        [None, 42, "1,2,3", "1,2,3,4,5", "a,b,c,d", "", [1, 2, 3], {"x": 1}],
    )
    def test_returns_none_rather_than_raising_on_anything_malformed(self, raw):
        # A caller must be able to fall through to whole-file handling; a
        # half-parsed box would crop to the wrong region.
        assert parse_clip_box(raw) is None


class TestParseConverterRecipe:
    """The flat ``converter`` / ``converter_param_<key>`` dialect."""

    def test_parses_the_full_dialect(self):
        recipe = parse_converter_recipe(_converter_params())
        assert recipe is not None
        assert recipe.name == "document2image"
        assert recipe.params == {"dpi": "150", "fmt": "png"}
        assert (recipe.out_index, recipe.n_out) == (3, 10)
        assert recipe.content_hash == "abc123def456"
        assert recipe.is_replayable

    def test_no_converter_key_is_not_a_converter_origin(self):
        assert parse_converter_recipe({"clip_start": "0.0", "clip_end": "1.0"}) is None

    @pytest.mark.parametrize("garbled", ["not-an-int", "", 1.5e400])
    def test_uncoercible_index_reads_as_absent(self, garbled):
        # A garbled index is indistinguishable from a missing one for
        # selection: both must leave the output unpicked, never picked wrongly.
        recipe = parse_converter_recipe(_converter_params(converter_out_index=garbled))
        assert recipe is not None
        assert recipe.out_index is None

    def test_recipe_without_disambiguators_parses_but_is_not_replayable(self):
        # "Not a converter media" and "a converter media we cannot replay" are
        # different answers, and callers act on them differently.
        recipe = parse_converter_recipe(_converter_params(converter_out_index=None, converter_content_hash=None))
        assert recipe is not None
        assert not recipe.is_replayable

    @pytest.mark.parametrize("keep", ["converter_out_index", "converter_content_hash"])
    def test_either_disambiguator_alone_is_enough(self, keep):
        drop = {"converter_out_index": None, "converter_content_hash": None}
        drop.pop(keep)
        recipe = parse_converter_recipe(_converter_params(**drop))
        assert recipe is not None and recipe.is_replayable

    def test_chain_step_omits_absent_disambiguators(self):
        recipe = parse_converter_recipe(_converter_params(converter_n_out=None))
        assert recipe is not None
        step = recipe.chain_step()
        assert step == {
            "kind": "converter",
            "name": "document2image",
            "params": {"dpi": "150", "fmt": "png"},
            "out_index": 3,
            "content_hash": "abc123def456",
        }

    def test_cache_key_is_hashable_and_order_independent(self):
        a = parse_converter_recipe(_converter_params())
        b = parse_converter_recipe({k: v for k, v in reversed(list(_converter_params().items()))})
        assert a is not None and b is not None
        assert a.cache_key() == b.cache_key()
        assert len({a.cache_key(), b.cache_key()}) == 1


#: A tiling of the rendered page, recorded after the converter ran.
_TILE_TRAIL = [
    {
        "kind": "clipper",
        "name": "image_tiling",
        "params": {},
        "out_index": 1,
        "n_out": 3,
        "content_hash": "0123456789ab",
    }
]


class TestConverterTrail:
    """A flat converter composed with the chain that ran on its output (#5041)."""

    def test_the_recorded_chain_follows_the_converter_step(self):
        params = _converter_params(clipper_chain=json.dumps(_TILE_TRAIL))
        recipe = parse_converter_recipe(params)
        assert recipe is not None
        assert converter_trail(params) == [recipe.chain_step(), *_TILE_TRAIL]

    def test_no_chain_is_the_converter_alone(self):
        recipe = parse_converter_recipe(_converter_params())
        assert recipe is not None
        assert converter_trail(_converter_params()) == [recipe.chain_step()]

    def test_a_malformed_chain_cannot_be_replayed(self):
        # The converter alone would rebuild the whole page, not the tile.
        assert converter_trail(_converter_params(clipper_chain="{not a trail")) is None

    def test_an_unreplayable_converter_cannot_be_replayed(self):
        params = _converter_params(
            converter_out_index=None, converter_content_hash=None, clipper_chain=json.dumps(_TILE_TRAIL)
        )
        assert converter_trail(params) is None


class TestBothPathsAgree:
    """The regression guard: one origin, two readers, one reading."""

    def test_a_cut_converter_output_replays_the_same_trail_on_both_paths(self):
        params = _converter_params(clipper_chain=json.dumps(_TILE_TRAIL))
        recipe = clip_recipe({"media_type": "image", "origin": {"params": params}})
        assert recipe is not None
        kind, media_type, trail_json = recipe
        assert (kind, media_type) == ("chain", "image")
        assert json.loads(trail_json) == _converter_origin_to_chain(params)

    @pytest.mark.parametrize(
        "params",
        [
            _converter_params(),
            _converter_params(converter_n_out=None),
            _converter_params(converter_content_hash=None),
            _converter_params(converter_out_index=None),
            _converter_params(converter_out_index="bogus"),
            {"converter": "video2image", "converter_out_index": "0"},
        ],
    )
    def test_converter_dialect_reads_the_same_on_both_paths(self, params):
        lazy = _converter_recipe(params)
        chain = _converter_origin_to_chain(params)

        # Both gate on replayability, so they agree on *whether* to replay.
        assert (lazy is None) == (chain is None)
        if lazy is None:
            return
        assert chain is not None  # implied by the equality above; states it for the type checker

        # ...and on what was recorded.  The shapes differ by design (a hashable
        # cache key vs. a ChainStep); the content behind them must not.
        _, name, param_items, out_index, n_out, content_hash = lazy
        step = chain[0]
        assert step["name"] == name
        assert step["params"] == dict(param_items)
        assert step.get("out_index") == out_index
        assert step.get("n_out") == n_out
        assert step.get("content_hash") == content_hash

    @pytest.mark.parametrize(
        "raw, expected",
        [("1,2,30,40", (1, 2, 30, 40)), ([5, 6, 7, 8], (5, 6, 7, 8)), ("bad", None)],
    )
    def test_clip_box_reads_the_same_on_both_paths(self, raw, expected):
        # lazy_clip reads the box through clip_recipe; the resolver reads it
        # through _clip_image_to_bytes.  Both now go via parse_clip_box, so
        # pinning the parser pins both.
        media = {"media_type": "image", "origin": {"params": {"clip_box": raw}}}
        recipe = clip_recipe(media)
        assert recipe == (("image", expected) if expected else None)
        assert parse_clip_box(raw) == expected


class TestReplayableGateSkipsPointlessWork:
    """A recipe the selector could never resolve is refused before the run."""

    def test_resolver_declines_a_chain_with_no_disambiguator(self):
        # _select_chain_output refuses to guess without a handle, so building
        # the chain would only run the converter to reach the same None.
        params = _converter_params(converter_out_index=None, converter_content_hash=None)
        assert _converter_origin_to_chain(params) is None

    def test_resolver_still_declines_when_there_is_no_converter_at_all(self):
        assert _converter_origin_to_chain({"clipper": "audio_tile"}) is None
