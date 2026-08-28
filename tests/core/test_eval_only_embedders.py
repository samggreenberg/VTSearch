"""Tests for ``MediaEmbedder.eval_only`` — research arms kept out of the app.

An eval-only embedder is registered and resolvable by name, so the eval harness
and the pre-embedded pile can use it, but it is withheld from the two
enumerations the app builds its pickers and defaults from.  The distinction
matters because a study arm is chosen for *differing* from the shipped embedder
in one controlled way; nothing in that choice says it is good, supported, or
licensed for users.

The concrete arm is ``clip_l`` (#3292), added to test whether #3287's
``calibration_fraction`` result follows single-vector geometry or the SigLIP
family.  These tests pin the contract rather than that one embedder, so a future
eval arm inherits the guarantee.
"""

from __future__ import annotations

import pytest

import app as app_module  # noqa: F401  (triggers conftest media init)
from vtscore.media import (
    all_embedders,
    all_embedders_dict,
    embedders_for_type,
    get_embedder,
)


class TestEvalOnlyContract:
    def test_default_is_false(self):
        """An embedder that says nothing is a normal, user-selectable one."""
        assert get_embedder("siglip").eval_only is False

    def test_clip_l_is_eval_only(self):
        assert get_embedder("clip_l").eval_only is True

    def test_shipped_clip_is_not_eval_only(self):
        """`clip` predates #3292 and is app-selectable; #3292 must not change that.

        Withdrawing a choice users already have is a production decision, not
        something an experiment gets to do on its way past.
        """
        assert get_embedder("clip").eval_only is False


class TestWithheldFromTheApp:
    def test_not_in_embedders_for_type(self):
        """The picker/default path — this is the one that keeps it out of the UI."""
        names = {e.name for e in embedders_for_type("image")}
        assert "clip_l" not in names
        assert "siglip" in names, "sanity: the filter must not empty the listing"

    def test_not_in_all_embedders_dict(self):
        """What ``GET /api/embedders`` serialises."""
        names = {d["name"] for d in all_embedders_dict()}
        assert "clip_l" not in names
        assert "clip" in names

    def test_never_becomes_the_default(self):
        """``embedders_for_type(t)[0]`` is how callers ask for the default."""
        assert embedders_for_type("image")[0].name == "siglip"

    def test_api_embedders_endpoint_omits_it(self, client):
        resp = client.get("/api/embedders")
        assert resp.status_code == 200
        payload = resp.get_json()
        embs = payload["embedders"] if isinstance(payload, dict) else payload
        assert "clip_l" not in {e["name"] for e in embs}


class TestStillResolvable:
    def test_registry_still_holds_it(self):
        """``all_embedders`` is the registry, not a user-facing listing.

        Docs inventories and name validation read it, and both should see the
        arm; only the app-facing enumerations filter.
        """
        assert "clip_l" in {e.name for e in all_embedders()}

    def test_get_embedder_resolves_it(self):
        """A pile cell embedded by an eval arm still has to load."""
        emb = get_embedder("clip_l")
        assert emb.name == "clip_l"
        assert emb.media_type_id == "image"

    def test_to_dict_carries_the_flag(self):
        """So a listing that does obtain one can tell what it is looking at."""
        assert get_embedder("clip_l").to_dict()["eval_only"] is True


class TestClipLIdentity:
    """The properties #3292 depends on, pinned so a checkpoint swap is loud."""

    def test_dimension_matches_siglip(self):
        """768-d is the whole reason this checkpoint and not ``clip``'s base/32.

        It takes output dimensionality off the table as an explanation for any
        difference between the CLIP and SigLIP arms.
        """
        assert get_embedder("clip_l").embedding_dim == get_embedder("siglip").embedding_dim == 768

    def test_is_a_different_checkpoint_from_the_shipped_clip(self):
        assert get_embedder("clip_l").model_id == "openai/clip-vit-large-patch14"
        assert get_embedder("clip").model_id == "openai/clip-vit-base-patch32"

    def test_declares_a_text_tower(self):
        """Without one, a study cell opens on the known-good fallback (#3278)."""
        assert get_embedder("clip_l").supports_text is True

    def test_is_not_patch_capable(self):
        """Single-vector is the property the study is testing; assert it."""
        assert get_embedder("clip_l").supports_patch_regions is False


@pytest.mark.parametrize("name", ["clip_l"])
def test_eval_only_arms_are_not_the_default_for_any_media_type(name):
    emb = get_embedder(name)
    assert emb.is_default is False, "an eval-only embedder must never be a default"
