"""Per-embedder-type combine conflict detection + keep-set resolution.

Exercises the pure helpers :func:`combine_type_state` and
:func:`resolve_keep_embedders` in ``vtscore.embedding.binding`` that back the
Combine-Datasets conflict-resolution UI.  Capabilities are stubbed so the tests
don't depend on which concrete embedders are registered.
"""

from __future__ import annotations

import pytest

from vtscore.embedding import binding as binding_mod
from vtscore.embedding.binding import (
    EMBEDDER_TYPE_PATCH_SEMANTIC,
    EMBEDDER_TYPE_SEMANTIC,
    EMBEDDER_TYPE_STRUCTURAL,
    combine_type_state,
    resolve_keep_embedders,
)

# embedder name -> (supports_text, supports_patch_regions, supports_geometric_verification)
_FAKE_CAPS = {
    "siglip": (True, False, False),
    "clip": (True, False, False),
    "single": (False, False, False),  # semantic, non-text single-vector (dinov2_single-like)
    "dino_patch": (False, True, False),
    "eupe_patch": (False, True, False),
    "sift_vlad": (False, False, True),
}


def _fake_type(name: str | None) -> str:
    caps = _FAKE_CAPS.get(name or "")
    if caps is None:
        return ""
    if caps[2]:
        return EMBEDDER_TYPE_STRUCTURAL
    if caps[1]:
        return EMBEDDER_TYPE_PATCH_SEMANTIC
    return EMBEDDER_TYPE_SEMANTIC


@pytest.fixture(autouse=True)
def fake_caps(monkeypatch):
    monkeypatch.setattr(binding_mod, "_capabilities", lambda name: _FAKE_CAPS.get(name or "", (False, False, False)))
    monkeypatch.setattr(binding_mod, "embedder_type", _fake_type)


class TestCombineTypeState:
    def test_identical_single_embedder_no_conflict(self):
        state = combine_type_state([["siglip"], ["siglip"]])
        assert set(state) == {EMBEDDER_TYPE_SEMANTIC}
        assert state[EMBEDDER_TYPE_SEMANTIC]["conflict"] is False
        assert state[EMBEDDER_TYPE_SEMANTIC]["options"] == ["siglip"]

    def test_name_clash_is_conflict(self):
        state = combine_type_state([["siglip"], ["clip"]])
        st = state[EMBEDDER_TYPE_SEMANTIC]
        assert st["conflict"] is True
        assert st["options"] == ["siglip", "clip"]
        assert st["reps"] == ["siglip", "clip"]

    def test_single_vector_vs_text_both_semantic_conflict(self):
        # dinov2_single-like vs siglip: both classify semantic → conflict on the
        # semantic type even though neither is a role "slot" clash.
        state = combine_type_state([["single"], ["siglip"]])
        st = state[EMBEDDER_TYPE_SEMANTIC]
        assert st["conflict"] is True
        assert st["options"] == ["single", "siglip"]

    def test_partial_coverage_is_conflict(self):
        # One dataset binds a patch embedder, the other doesn't.
        state = combine_type_state([["siglip", "dino_patch"], ["siglip"]])
        assert state[EMBEDDER_TYPE_SEMANTIC]["conflict"] is False
        patch = state[EMBEDDER_TYPE_PATCH_SEMANTIC]
        assert patch["conflict"] is True
        assert patch["options"] == ["dino_patch"]
        assert patch["n_present"] == 1
        assert patch["n_total"] == 2
        assert patch["reps"] == ["dino_patch", None]

    def test_type_absent_everywhere_omitted(self):
        state = combine_type_state([["siglip"], ["clip"]])
        assert EMBEDDER_TYPE_STRUCTURAL not in state
        assert EMBEDDER_TYPE_PATCH_SEMANTIC not in state

    def test_full_trio_all_agree(self):
        trio = ["siglip", "dino_patch", "sift_vlad"]
        state = combine_type_state([list(trio), list(trio)])
        assert set(state) == {
            EMBEDDER_TYPE_SEMANTIC,
            EMBEDDER_TYPE_PATCH_SEMANTIC,
            EMBEDDER_TYPE_STRUCTURAL,
        }
        assert all(not st["conflict"] for st in state.values())

    def test_semantic_text_wins_over_cobound_single_vector(self):
        # A dataset binding both a text embedder and a single-vector one reports
        # the text-capable embedder as its semantic representative.
        state = combine_type_state([["siglip", "single"], ["siglip"]])
        assert state[EMBEDDER_TYPE_SEMANTIC]["conflict"] is False
        assert state[EMBEDDER_TYPE_SEMANTIC]["options"] == ["siglip"]


class TestResolveKeepEmbedders:
    def test_no_conflicts_keeps_agreed(self):
        state = combine_type_state([["siglip"], ["siglip"]])
        keep, err = resolve_keep_embedders(state, None)
        assert err is None
        assert keep == ["siglip"]

    def test_unresolved_conflict_errors(self):
        state = combine_type_state([["siglip"], ["clip"]])
        keep, err = resolve_keep_embedders(state, None)
        assert keep == []
        assert err and "Semantic" in err

    def test_reembed_winner_kept(self):
        state = combine_type_state([["siglip"], ["clip"]])
        keep, err = resolve_keep_embedders(state, {EMBEDDER_TYPE_SEMANTIC: {"action": "reembed", "embedder": "clip"}})
        assert err is None
        assert keep == ["clip"]

    def test_drop_omits_type(self):
        # Drop the conflicted patch type; keep the agreed semantic one.
        state = combine_type_state([["siglip", "dino_patch"], ["siglip"]])
        keep, err = resolve_keep_embedders(state, {EMBEDDER_TYPE_PATCH_SEMANTIC: {"action": "drop"}})
        assert err is None
        assert keep == ["siglip"]

    def test_invalid_winner_errors(self):
        state = combine_type_state([["siglip"], ["clip"]])
        keep, err = resolve_keep_embedders(
            state, {EMBEDDER_TYPE_SEMANTIC: {"action": "reembed", "embedder": "dino_patch"}}
        )
        assert keep == []
        assert err and "dino_patch" in err

    def test_dropping_everything_errors(self):
        state = combine_type_state([["siglip"], ["clip"]])
        keep, err = resolve_keep_embedders(state, {EMBEDDER_TYPE_SEMANTIC: {"action": "drop"}})
        assert keep == []
        assert err and "At least one" in err

    def test_mixed_reembed_and_drop(self):
        # semantic conflict → re-embed to siglip; patch conflict → drop.
        state = combine_type_state([["siglip", "dino_patch"], ["clip"]])
        keep, err = resolve_keep_embedders(
            state,
            {
                EMBEDDER_TYPE_SEMANTIC: {"action": "reembed", "embedder": "siglip"},
                EMBEDDER_TYPE_PATCH_SEMANTIC: {"action": "drop"},
            },
        )
        assert err is None
        assert keep == ["siglip"]

    def test_invalid_action_errors(self):
        state = combine_type_state([["siglip"], ["clip"]])
        keep, err = resolve_keep_embedders(state, {EMBEDDER_TYPE_SEMANTIC: {"action": "bogus"}})
        assert keep == []
        assert err
