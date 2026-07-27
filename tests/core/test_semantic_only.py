"""Tests for the server-tier ``semantic_only`` embedder-type lock.

Covers the persisted server-tier value, the process-level override set by
``--semantic-only`` / ``VTSEARCH_SEMANTIC_ONLY`` (``set_cli_semantic_only``),
the resolver (``get_effective_semantic_only``), the API contract (read-only:
exposed in ``GET /api/settings`` but not settable via ``PUT``), the
``GET /api/embedders`` filter, and the route-level rejection of a
patch/structural embedder or detector type (issue #2696).
"""

from __future__ import annotations

import pytest

import app as app_module  # noqa: F401  (triggers conftest media init)
from vtsearch import settings as settings_mod
from vtsearch.schemas.settings import AppSettingsSchema, SettingsUpdateSchema


@pytest.fixture(autouse=True)
def _reset_cli_semantic_only():
    """The CLI/env override is process-global; clear it around each test."""
    settings_mod.set_cli_semantic_only(None)
    yield
    settings_mod.set_cli_semantic_only(None)


class TestEffectiveResolution:
    def test_default_is_off(self):
        assert settings_mod.get_effective_semantic_only() is False

    def test_cli_override_wins_over_persisted(self, isolated_settings):
        settings_mod.set_semantic_only(False)
        settings_mod.set_cli_semantic_only(True)
        assert settings_mod.get_effective_semantic_only() is True

    def test_falls_back_to_persisted_when_no_override(self, isolated_settings):
        settings_mod.set_semantic_only(True)
        assert settings_mod.get_cli_semantic_only() is None
        assert settings_mod.get_effective_semantic_only() is True

    def test_override_clears_back_to_persisted(self, isolated_settings):
        settings_mod.set_semantic_only(True)
        settings_mod.set_cli_semantic_only(False)
        assert settings_mod.get_effective_semantic_only() is False
        settings_mod.set_cli_semantic_only(None)
        assert settings_mod.get_effective_semantic_only() is True


class TestApiContract:
    def test_not_settable_via_put(self, client, isolated_settings):
        """A PUT body carrying semantic_only is silently ignored: the update
        schema excludes it, so it never reaches a setter."""
        resp = client.put("/api/settings", json={"semantic_only": True})
        assert resp.status_code == 200
        assert settings_mod.get_semantic_only() is False

    def test_get_reflects_effective_value(self, client):
        settings_mod.set_cli_semantic_only(True)
        assert client.get("/api/settings").get_json()["semantic_only"] is True

    def test_get_reflects_default(self, client):
        assert client.get("/api/settings").get_json()["semantic_only"] is False

    def test_schema_marks_it_read_only(self):
        # Dumpable (so the New-detector modal can drop its type picker) but not
        # loadable (set via CLI/env/file, never PUT).
        assert AppSettingsSchema().fields["semantic_only"].dump_only is True
        assert "semantic_only" not in SettingsUpdateSchema().fields


class TestEmbeddersListing:
    """``GET /api/embedders`` is the chokepoint every picker reads."""

    @staticmethod
    def _names(client, **params) -> list[str]:
        resp = client.get("/api/embedders", query_string=params)
        assert resp.status_code == 200
        return [e["name"] for e in resp.get_json()["embedders"]]

    def test_prototypes_listed_when_unlocked(self, client):
        names = self._names(client, media_type="image")
        assert "dinov3_patch" in names
        assert "sift_vlad" in names

    def test_prototypes_withheld_when_locked(self, client):
        settings_mod.set_cli_semantic_only(True)
        names = self._names(client, media_type="image")
        assert "dinov3_patch" not in names
        assert "sift_vlad" not in names
        # ...but the Semantic image embedders are all still on offer.
        assert "siglip" in names

    def test_lock_applies_to_the_unfiltered_listing_too(self, client):
        settings_mod.set_cli_semantic_only(True)
        names = self._names(client)
        assert not {"dinov2_patch", "dinov3_patch", "eupe_patch", "sift_vlad"} & set(names)
        assert "clap" in names

    def test_semantic_only_media_types_are_untouched(self, client):
        """Audio/text/video bind no prototype embedder, so the lock is a no-op
        for them - the pickers there must not shrink."""
        before = self._names(client, media_type="audio")
        settings_mod.set_cli_semantic_only(True)
        assert self._names(client, media_type="audio") == before


class TestDetectorCreateRejection:
    def test_structural_detector_rejected_when_locked(self, client):
        settings_mod.set_cli_semantic_only(True)
        resp = client.post(
            "/api/detectors",
            json={"name": "locked-structural", "media_type": "image", "text_query": "a dog", "embedder_type": "structural"},
        )
        assert resp.status_code == 400
        assert "Semantic" in resp.get_json()["message"]

    def test_patch_detector_rejected_when_locked(self, client):
        settings_mod.set_cli_semantic_only(True)
        resp = client.post(
            "/api/detectors",
            json={"name": "locked-patch", "media_type": "image", "text_query": "a dog", "embedder_type": "patch_semantic"},
        )
        assert resp.status_code == 400

    def test_semantic_detector_still_allowed_when_locked(self, client):
        settings_mod.set_cli_semantic_only(True)
        resp = client.post(
            "/api/detectors",
            json={"name": "locked-semantic", "media_type": "image", "text_query": "a dog", "embedder_type": "semantic"},
        )
        assert resp.status_code == 201

    def test_structural_detector_allowed_when_unlocked(self, client):
        resp = client.post(
            "/api/detectors",
            json={"name": "unlocked-structural", "media_type": "image", "text_query": "a dog", "embedder_type": "structural"},
        )
        assert resp.status_code == 201


class TestImportRejection:
    """A hand-rolled import must not bind a type the pickers hide."""

    def test_patch_embedder_rejected_when_locked(self, client):
        settings_mod.set_cli_semantic_only(True)
        resp = client.post(
            "/api/dataset/import/server_folder",
            json={
                "path": "/tmp/whatever",
                "media_type": "image",
                "embedder": "siglip",
                "embedders": ["siglip", "dinov3_patch"],
            },
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert "dinov3_patch" in body["message"]

    def test_structural_primary_embedder_rejected_when_locked(self, client):
        settings_mod.set_cli_semantic_only(True)
        resp = client.post(
            "/api/dataset/import/server_folder",
            json={"path": "/tmp/whatever", "media_type": "image", "embedder": "sift_vlad"},
        )
        assert resp.status_code == 400
        assert "sift_vlad" in resp.get_json()["message"]

    def test_semantic_embedder_passes_the_gate_when_locked(self):
        """The Semantic path is untouched by the lock. Asserted against the
        guard directly rather than through the route, so the request doesn't
        start a real background import just to prove it wasn't rejected."""
        from vtsearch.routes._shared import abort_if_semantic_only_embedders

        settings_mod.set_cli_semantic_only(True)
        # No raise == not rejected. Unknown names are left to their own
        # downstream validation, so they pass the gate too.
        abort_if_semantic_only_embedders(["siglip", "clap", ""])
        abort_if_semantic_only_embedders([])
        abort_if_semantic_only_embedders(["not_a_registered_embedder"])

    def test_guard_is_a_no_op_when_unlocked(self):
        from vtsearch.routes._shared import abort_if_semantic_only_embedders

        abort_if_semantic_only_embedders(["sift_vlad", "dinov3_patch"])
