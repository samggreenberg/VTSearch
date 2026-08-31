"""Role-typed embedder binding on a dataset (v3 three-slot model).

Covers the pure derivation/validation helpers in
``vtscore.embedding.binding`` and the ``DatasetContext`` binding
properties that consume them.  Capability lookups are stubbed so the
tests don't depend on which concrete embedders happen to be registered;
one smoke class exercises the live registry for the wiring.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.embedding import binding as binding_mod
from vtscore.embedding.binding import (
    derive_binding,
    derive_binding_from_names,
    score_marker_embedder,
    score_marker_embedder_for_snap,
    validate_binding,
)
from vtscore.state.core import DatasetContext

# embedder name -> (supports_text, supports_patch_regions, supports_geometric_verification)
_FAKE_CAPS = {
    "faketext": (True, False, False),
    "fakepatch": (False, True, False),
    "fakestructural": (False, False, True),
    "fakesingle": (False, False, False),
    "fakeboth": (True, True, False),
    "fakeall": (True, True, True),
}


@pytest.fixture
def fake_caps(monkeypatch):
    monkeypatch.setattr(binding_mod, "_capabilities", lambda name: _FAKE_CAPS.get(name, (False, False, False)))


class TestDeriveBinding:
    def test_none_and_empty(self):
        assert derive_binding(None) == (None, None, None)
        assert derive_binding("") == (None, None, None)

    def test_text_embedder_fills_text_slot(self, fake_caps):
        assert derive_binding("faketext") == ("faketext", None, None)

    def test_patch_embedder_fills_patch_slot(self, fake_caps):
        assert derive_binding("fakepatch") == (None, "fakepatch", None)

    def test_structural_embedder_fills_structural_slot(self, fake_caps):
        assert derive_binding("fakestructural") == (None, None, "fakestructural")

    def test_single_vector_fills_neither(self, fake_caps):
        assert derive_binding("fakesingle") == (None, None, None)

    def test_dual_capability_fills_both(self, fake_caps):
        assert derive_binding("fakeboth") == ("fakeboth", "fakeboth", None)

    def test_triple_capability_fills_all(self, fake_caps):
        assert derive_binding("fakeall") == ("fakeall", "fakeall", "fakeall")

    def test_unknown_name_fills_neither(self, fake_caps):
        assert derive_binding("nope") == (None, None, None)


class TestDeriveBindingFromNames:
    def test_empty_iterable(self):
        assert derive_binding_from_names([]) == (None, None, None)
        assert derive_binding_from_names([None, ""]) == (None, None, None)

    def test_text_plus_patch_fills_both_slots(self, fake_caps):
        assert derive_binding_from_names(["faketext", "fakepatch"]) == ("faketext", "fakepatch", None)

    def test_full_trio_fills_all_slots(self, fake_caps):
        assert derive_binding_from_names(["faketext", "fakepatch", "fakestructural"]) == (
            "faketext",
            "fakepatch",
            "fakestructural",
        )

    def test_order_independent(self, fake_caps):
        assert derive_binding_from_names(["fakestructural", "fakepatch", "faketext"]) == (
            "faketext",
            "fakepatch",
            "fakestructural",
        )

    def test_first_of_each_role_wins(self, fake_caps):
        # A second text-capable name does not displace the first.
        assert derive_binding_from_names(["faketext", "fakeboth"]) == ("faketext", "fakeboth", None)

    def test_single_vector_among_others_is_skipped(self, fake_caps):
        assert derive_binding_from_names(["fakesingle", "faketext"]) == ("faketext", None, None)


class TestValidateBinding:
    def test_none_slots_ok(self, fake_caps):
        validate_binding(None, None, None)  # no raise

    def test_valid_roles_ok(self, fake_caps):
        validate_binding("faketext", "fakepatch", "fakestructural")  # no raise

    def test_text_slot_rejects_non_text(self, fake_caps):
        with pytest.raises(ValueError, match=r"text_embedder 'fakepatch' does not support text"):
            validate_binding("fakepatch", None)

    def test_patch_slot_rejects_non_patch(self, fake_caps):
        with pytest.raises(ValueError, match=r"patch_embedder 'faketext' does not produce patch"):
            validate_binding(None, "faketext")

    def test_structural_slot_rejects_non_structural(self, fake_caps):
        with pytest.raises(ValueError, match=r"structural_embedder 'faketext' does not support geometric"):
            validate_binding(None, None, "faketext")

    def test_unknown_name_rejected(self, fake_caps):
        with pytest.raises(ValueError):
            validate_binding("nope", None)


def _ctx_with_embedder(name: str) -> DatasetContext:
    ctx = DatasetContext("test_binding")
    ctx.medias[1] = {"id": 1, "embedder": name, "embedding": np.ones(4, dtype=np.float32)}
    return ctx


class TestDatasetContextDerivedBinding:
    def test_empty_dataset_binds_nothing(self):
        ctx = DatasetContext("empty")
        assert ctx.text_embedder is None
        assert ctx.patch_embedder is None
        assert ctx.supports_text is False
        assert ctx.supports_patch_regions is False

    def test_text_dataset(self, fake_caps):
        ctx = _ctx_with_embedder("faketext")
        assert ctx.text_embedder == "faketext"
        assert ctx.patch_embedder is None
        assert ctx.supports_text is True
        assert ctx.supports_patch_regions is False

    def test_patch_dataset(self, fake_caps):
        ctx = _ctx_with_embedder("fakepatch")
        assert ctx.text_embedder is None
        assert ctx.patch_embedder == "fakepatch"
        assert ctx.supports_text is False
        assert ctx.supports_patch_regions is True

    def test_single_vector_dataset_binds_neither(self, fake_caps):
        ctx = _ctx_with_embedder("fakesingle")
        assert ctx.text_embedder is None
        assert ctx.patch_embedder is None

    def test_two_embedder_dataset_derives_both_slots(self, fake_caps):
        # A v3 media carries one vector per bound embedder under "embeddings";
        # the binding is recovered by role-typing both keys, not just the
        # recorded primary.
        ctx = DatasetContext("two")
        ctx.medias[1] = {
            "id": 1,
            "embedder": "fakepatch",
            "embedding": np.ones(4, dtype=np.float32),
            "embeddings": {
                "faketext": np.ones(4, dtype=np.float32),
                "fakepatch": np.ones(4, dtype=np.float32),
            },
        }
        assert ctx.text_embedder == "faketext"
        assert ctx.patch_embedder == "fakepatch"
        assert ctx.supports_text is True
        assert ctx.supports_patch_regions is True


class TestDatasetContextExplicitBinding:
    def test_explicit_overrides_derivation(self, fake_caps):
        # Derived binding would be text-only; explicit binding adds the patch slot.
        ctx = _ctx_with_embedder("faketext")
        ctx.bind_embedders(text_embedder="faketext", patch_embedder="fakepatch")
        assert ctx.text_embedder == "faketext"
        assert ctx.patch_embedder == "fakepatch"
        assert ctx.supports_text is True
        assert ctx.supports_patch_regions is True

    def test_explicit_binds_full_trio(self, fake_caps):
        ctx = _ctx_with_embedder("faketext")
        ctx.bind_embedders(
            text_embedder="faketext",
            patch_embedder="fakepatch",
            structural_embedder="fakestructural",
        )
        assert ctx.text_embedder == "faketext"
        assert ctx.patch_embedder == "fakepatch"
        assert ctx.structural_embedder == "fakestructural"
        assert ctx.supports_text is True
        assert ctx.supports_patch_regions is True
        assert ctx.supports_geometric_verification is True

    def test_explicit_validates_roles(self, fake_caps):
        ctx = DatasetContext("bad")
        with pytest.raises(ValueError, match="does not support text"):
            ctx.bind_embedders(text_embedder="fakepatch")
        # A rejected binding must not be stored - derivation still governs.
        assert ctx._binding_explicit is False

    def test_explicit_validates_structural_role(self, fake_caps):
        ctx = DatasetContext("bad")
        with pytest.raises(ValueError, match="does not support geometric"):
            ctx.bind_embedders(structural_embedder="faketext")
        assert ctx._binding_explicit is False

    def test_explicit_empty_triple_overrides_derivation(self, fake_caps):
        ctx = _ctx_with_embedder("faketext")
        ctx.bind_embedders()  # all None, explicit
        assert ctx.text_embedder is None
        assert ctx.patch_embedder is None
        assert ctx.structural_embedder is None


class TestRoutedEmbedder:
    """The v3 routing table: which bound embedder serves each operation role."""

    def test_text_dataset_roles(self, fake_caps):
        ctx = _ctx_with_embedder("faketext")
        assert ctx.routed_embedder("text") == "faketext"
        assert ctx.routed_embedder("patch") is None
        assert ctx.routed_embedder("score") == "faketext"

    def test_patch_dataset_roles(self, fake_caps):
        ctx = _ctx_with_embedder("fakepatch")
        assert ctx.routed_embedder("text") is None
        assert ctx.routed_embedder("patch") == "fakepatch"
        assert ctx.routed_embedder("score") == "fakepatch"

    def test_single_vector_dataset_routes_nothing(self, fake_caps):
        # A slot-less single-vector dataset (e.g. dinov2_single) binds neither
        # role; score falls through to None so the matrix layer reads the
        # primary vector rather than 400-ing.
        ctx = _ctx_with_embedder("fakesingle")
        assert ctx.routed_embedder("text") is None
        assert ctx.routed_embedder("patch") is None
        assert ctx.routed_embedder("score") is None

    def test_structural_dataset_roles(self, fake_caps):
        ctx = _ctx_with_embedder("fakestructural")
        assert ctx.routed_embedder("text") is None
        assert ctx.routed_embedder("patch") is None
        assert ctx.routed_embedder("structural") == "fakestructural"
        assert ctx.routed_embedder("score") == "fakestructural"

    def test_score_prefers_patch_over_text(self, fake_caps):
        ctx = DatasetContext("dual")
        ctx.bind_embedders(text_embedder="faketext", patch_embedder="fakepatch")
        assert ctx.routed_embedder("text") == "faketext"
        assert ctx.routed_embedder("patch") == "fakepatch"
        # score = patch-if-set-else-text
        assert ctx.routed_embedder("score") == "fakepatch"

    def test_score_prefers_structural_over_patch_and_text(self, fake_caps):
        ctx = DatasetContext("trio")
        ctx.bind_embedders(
            text_embedder="faketext",
            patch_embedder="fakepatch",
            structural_embedder="fakestructural",
        )
        assert ctx.routed_embedder("text") == "faketext"
        assert ctx.routed_embedder("patch") == "fakepatch"
        assert ctx.routed_embedder("structural") == "fakestructural"
        # score precedence: structural ▸ patch ▸ text
        assert ctx.routed_embedder("score") == "fakestructural"

    def test_score_falls_back_to_text_when_no_patch(self, fake_caps):
        ctx = DatasetContext("texty")
        ctx.bind_embedders(text_embedder="faketext", patch_embedder=None)
        assert ctx.routed_embedder("score") == "faketext"

    def test_unknown_role_raises(self, fake_caps):
        ctx = _ctx_with_embedder("faketext")
        with pytest.raises(ValueError, match="unknown embedder routing role"):
            ctx.routed_embedder("bogus")


class TestScoreMarkerEmbedder:
    """The v3 model-keying marker - the embedder component of the
    ``(detector, dataset, embedder)`` MLP key (patch-embedder.md Phase 2b.5).

    The marker must equal the **score** embedder (structural ▸ patch ▸ text)
    the MLP is trained/scored against, falling back to the primary *name* for a
    slot-less single-vector dataset so it is always a concrete string to compare
    against ``DetectorContext.embedder``.
    """

    def _media(self, primary: str, names: list[str]) -> dict:
        return {
            "embedder": primary,
            "embedding": np.ones(4, dtype=np.float32),
            "embeddings": {n: np.ones(4, dtype=np.float32) for n in names},
        }

    def test_text_only_marks_text(self, fake_caps):
        assert score_marker_embedder(self._media("faketext", ["faketext"])) == "faketext"

    def test_patch_only_marks_patch(self, fake_caps):
        assert score_marker_embedder(self._media("fakepatch", ["fakepatch"])) == "fakepatch"

    def test_dual_marks_patch_not_primary(self, fake_caps):
        # The crux of Phase 2b.5: a dual dataset's primary mirror is the text
        # embedder, but the MLP scores against the patch embedder.  The marker
        # must follow the scored space (patch), not media["embedder"] (text),
        # or a stale cross-space MLP survives a dataset switch.
        media = self._media("faketext", ["faketext", "fakepatch"])
        assert media["embedder"] == "faketext"
        assert score_marker_embedder(media) == "fakepatch"

    def test_structural_wins_over_patch_and_text(self, fake_caps):
        # On a full trio the score space is structural; the marker must follow it.
        media = self._media("faketext", ["faketext", "fakepatch", "fakestructural"])
        assert score_marker_embedder(media) == "fakestructural"

    def test_single_vector_falls_back_to_primary_name(self, fake_caps):
        # routed_embedder("score") is None here, but the marker is a concrete
        # name (the primary) so the str-compare invalidation still works.
        assert score_marker_embedder(self._media("fakesingle", ["fakesingle"])) == "fakesingle"

    def test_no_embedder_marks_empty(self, fake_caps):
        assert score_marker_embedder({}) == ""

    def test_for_snap_empty(self):
        assert score_marker_embedder_for_snap(None) == ""
        assert score_marker_embedder_for_snap({}) == ""

    def test_for_snap_uses_first_media(self, fake_caps):
        snap = {7: self._media("faketext", ["faketext", "fakepatch"])}
        assert score_marker_embedder_for_snap(snap) == "fakepatch"


class TestRealRegisteredEmbedder:
    """Smoke test against the live registry.  Reading the capability flags
    does not load model weights, so this is cheap and CPU-only."""

    def test_real_text_embedder_binds_text_slot(self):
        from vtscore.media import all_embedders

        text_emb = next((e for e in all_embedders() if e.supports_text), None)
        if text_emb is None:
            pytest.skip("no text-capable embedder registered")
        ctx = _ctx_with_embedder(text_emb.name)
        assert ctx.text_embedder == text_emb.name
        assert ctx.supports_text is True

    def test_real_patch_embedder_binds_patch_slot(self):
        from vtscore.media import all_embedders

        patch_emb = next((e for e in all_embedders() if e.supports_patch_regions), None)
        if patch_emb is None:
            pytest.skip("no patch-capable embedder registered")
        ctx = _ctx_with_embedder(patch_emb.name)
        assert ctx.patch_embedder == patch_emb.name
        assert ctx.supports_patch_regions is True

    def test_real_structural_embedder_binds_structural_slot(self):
        from vtscore.media import all_embedders

        struct_emb = next((e for e in all_embedders() if e.supports_geometric_verification), None)
        if struct_emb is None:
            pytest.skip("no structural embedder registered")
        ctx = _ctx_with_embedder(struct_emb.name)
        assert ctx.structural_embedder == struct_emb.name
        assert ctx.supports_geometric_verification is True


class TestEmbedderSupportsPatchRegions:
    """The public single-capability read used to gate patch-embedder callers.

    Added for the #3329 fit-quality finding: the coverage atlas's typicality
    guard is uninformative in a patch space, so the domain-shift route refuses
    those references.  The gate reads the declared capability rather than a
    hard-coded name list, so a newly registered patch embedder is covered the
    day it lands.
    """

    def test_agrees_with_the_registry_for_every_embedder(self):
        from vtscore.embedding.binding import embedder_supports_patch_regions
        from vtscore.media import all_embedders

        for emb in all_embedders():
            assert embedder_supports_patch_regions(emb.name) is bool(emb.supports_patch_regions), emb.name

    def test_finds_at_least_one_of_each(self):
        """Guard against the agreement test passing vacuously on an empty side."""
        from vtscore.embedding.binding import embedder_supports_patch_regions
        from vtscore.media import all_embedders

        names = [e.name for e in all_embedders()]
        assert any(embedder_supports_patch_regions(n) for n in names)
        assert any(not embedder_supports_patch_regions(n) for n in names)

    def test_blank_and_unknown_names_are_false(self):
        from vtscore.embedding.binding import embedder_supports_patch_regions

        assert embedder_supports_patch_regions(None) is False
        assert embedder_supports_patch_regions("") is False
        assert embedder_supports_patch_regions("no_such_embedder") is False
