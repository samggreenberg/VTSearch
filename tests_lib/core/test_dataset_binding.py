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
from vtscore.embedding.binding import derive_binding, validate_binding
from vtscore.state.core import DatasetContext

# embedder name -> (supports_text, supports_patch_regions)
_FAKE_CAPS = {
    "faketext": (True, False),
    "fakepatch": (False, True),
    "fakesingle": (False, False),
    "fakeboth": (True, True),
}


@pytest.fixture
def fake_caps(monkeypatch):
    monkeypatch.setattr(binding_mod, "_capabilities", lambda name: _FAKE_CAPS.get(name, (False, False)))


class TestDeriveBinding:
    def test_none_and_empty(self):
        assert derive_binding(None) == (None, None)
        assert derive_binding("") == (None, None)

    def test_text_embedder_fills_text_slot(self, fake_caps):
        assert derive_binding("faketext") == ("faketext", None)

    def test_patch_embedder_fills_patch_slot(self, fake_caps):
        assert derive_binding("fakepatch") == (None, "fakepatch")

    def test_single_vector_fills_neither(self, fake_caps):
        assert derive_binding("fakesingle") == (None, None)

    def test_dual_capability_fills_both(self, fake_caps):
        assert derive_binding("fakeboth") == ("fakeboth", "fakeboth")

    def test_unknown_name_fills_neither(self, fake_caps):
        assert derive_binding("nope") == (None, None)


class TestValidateBinding:
    def test_none_slots_ok(self, fake_caps):
        validate_binding(None, None)  # no raise

    def test_valid_roles_ok(self, fake_caps):
        validate_binding("faketext", "fakepatch")  # no raise

    def test_text_slot_rejects_non_text(self, fake_caps):
        with pytest.raises(ValueError, match=r"text_embedder 'fakepatch' does not support text"):
            validate_binding("fakepatch", None)

    def test_patch_slot_rejects_non_patch(self, fake_caps):
        with pytest.raises(ValueError, match=r"patch_embedder 'faketext' does not produce patch"):
            validate_binding(None, "faketext")

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


class TestDatasetContextExplicitBinding:
    def test_explicit_overrides_derivation(self, fake_caps):
        # Derived binding would be text-only; explicit binding adds the patch slot.
        ctx = _ctx_with_embedder("faketext")
        ctx.bind_embedders(text_embedder="faketext", patch_embedder="fakepatch")
        assert ctx.text_embedder == "faketext"
        assert ctx.patch_embedder == "fakepatch"
        assert ctx.supports_text is True
        assert ctx.supports_patch_regions is True

    def test_explicit_validates_roles(self, fake_caps):
        ctx = DatasetContext("bad")
        with pytest.raises(ValueError, match="does not support text"):
            ctx.bind_embedders(text_embedder="fakepatch")
        # A rejected binding must not be stored - derivation still governs.
        assert ctx._binding_explicit is False

    def test_explicit_empty_pair_overrides_derivation(self, fake_caps):
        ctx = _ctx_with_embedder("faketext")
        ctx.bind_embedders()  # both None, explicit
        assert ctx.text_embedder is None
        assert ctx.patch_embedder is None


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
