"""API tests for the seed-importer routes (list / run / field options).

No seed importer ships in-tree — the family exists so a third-party package
can add one — so these tests install stubs onto the route module's registry
accessors, which is exactly the surface an entry-point plugin lands on.
"""

from __future__ import annotations

import io

import pytest

import vtscore.security.path_validation as paths_mod
from vtscore.plugins import PluginField
from vtscore.seed_importers.base import SeedImporter, SeedMediaItem


@pytest.fixture
def example_media_dir(tmp_path, monkeypatch):
    """Redirect ``example_media/`` to a per-test directory and return it.

    Same reasoning as the datasource-importer suite: ``data/example_media/``
    is a real shared directory, and patching the one definition both writers
    and readers go through keeps the suite off it without a cross-worker
    cleanup race.
    """
    media_dir = tmp_path / "example_media"
    monkeypatch.setattr(paths_mod, "example_media_dir", lambda: media_dir)
    return media_dir


class _StubSeedImporter(SeedImporter):
    """Return one seed per requested count."""

    name = "stub_seeds"
    display_name = "Stub Seeds"
    description = "Fetch some near-miss media."
    max_items = 3
    fields = [
        PluginField(key="count", label="How many", field_type="number", default="2"),
    ]

    def run(self, field_values):
        n = int(field_values.get("count") or 0)
        return [SeedMediaItem(data=f"seed-{i}".encode(), filename=f"near-{i}.wav") for i in range(n)]


@pytest.fixture
def stub_importer(monkeypatch):
    """Register :class:`_StubSeedImporter` as the only seed importer."""
    importer = _StubSeedImporter()
    monkeypatch.setattr("vtsearch.routes.media.seed.list_seed_importers", lambda: [importer])
    monkeypatch.setattr(
        "vtsearch.routes.media.seed.get_seed_importer",
        lambda name: importer if name == importer.name else None,
    )
    return importer


class TestSeedImportersList:
    def test_empty_on_a_vanilla_install(self, client):
        """Nothing in-tree registers a seed importer, so the Blank flow grows no tabs."""
        resp = client.get("/api/seed-importers")
        assert resp.status_code == 200
        assert resp.get_json() == {"importers": []}

    def test_lists_a_registered_importer_with_its_fields_and_cap(self, client, stub_importer):
        resp = client.get("/api/seed-importers")
        assert resp.status_code == 200
        importers = resp.get_json()["importers"]
        assert [imp["name"] for imp in importers] == ["stub_seeds"]
        assert importers[0]["max_items"] == 3
        assert [f["key"] for f in importers[0]["fields"]] == ["count"]

    def test_hidden_plugins_filtered(self, client, stub_importer, monkeypatch):
        monkeypatch.setattr(
            "vtsearch.settings.get_effective_hidden_plugins",
            lambda: {"seed_importers": {"stub_seeds"}},
        )
        resp = client.get("/api/seed-importers")
        assert resp.get_json()["importers"] == []


class TestSeedImportRun:
    def test_saves_every_seed_into_example_media(self, client, stub_importer, example_media_dir):
        resp = client.post("/api/seed-import/stub_seeds", json={"count": "2"})
        assert resp.status_code == 201, resp.get_json()
        body = resp.get_json()

        assert body["count"] == 2
        assert body["truncated"] is False
        assert [item["original_name"] for item in body["items"]] == ["near-0.wav", "near-1.wav"]
        for item in body["items"]:
            saved = example_media_dir / item["filename"]
            assert saved.is_file()
            # The storage name is generated, never the plugin's filename, but
            # the suffix is kept because it drives media-type inference.
            assert saved.suffix == ".wav"
            assert item["filename"] != item["original_name"]
        assert {p.read_bytes() for p in example_media_dir.iterdir()} == {b"seed-0", b"seed-1"}

    def test_reports_origin_when_the_importer_supplies_one(self, client, stub_importer, example_media_dir, monkeypatch):
        origin = {"importer": "url_download", "params": {"url": "https://example.com/a.wav"}}
        monkeypatch.setattr(
            stub_importer,
            "run",
            lambda values: [SeedMediaItem(data=b"x", filename="a.wav", origin=origin)],
        )

        resp = client.post("/api/seed-import/stub_seeds", json={"count": "1"})
        assert resp.get_json()["items"][0]["origin"] == origin

    def test_origin_is_null_when_the_importer_reports_none(self, client, stub_importer, example_media_dir):
        resp = client.post("/api/seed-import/stub_seeds", json={"count": "1"})
        assert resp.get_json()["items"][0]["origin"] is None

    def test_empty_batch_is_a_valid_answer(self, client, stub_importer, example_media_dir):
        resp = client.post("/api/seed-import/stub_seeds", json={"count": "0"})
        assert resp.status_code == 201
        assert resp.get_json() == {"items": [], "count": 0, "truncated": False}

    def test_batch_is_truncated_at_the_importers_cap(self, client, stub_importer, example_media_dir):
        resp = client.post("/api/seed-import/stub_seeds", json={"count": "9"})
        body = resp.get_json()

        assert body["count"] == stub_importer.max_items
        assert body["truncated"] is True
        assert len(list(example_media_dir.iterdir())) == stub_importer.max_items

    def test_dataless_items_are_skipped_without_sinking_the_batch(
        self, client, stub_importer, example_media_dir, monkeypatch
    ):
        monkeypatch.setattr(
            stub_importer,
            "run",
            lambda values: [
                SeedMediaItem(data=b"", filename="empty.wav"),
                SeedMediaItem(data=b"real", filename="real.wav"),
            ],
        )

        body = client.post("/api/seed-import/stub_seeds", json={"count": "2"}).get_json()

        assert body["count"] == 1
        assert body["items"][0]["original_name"] == "real.wav"

    def test_bad_user_input_maps_to_400(self, client, stub_importer, monkeypatch):
        def _bad(values):
            raise ValueError("Unknown cluster id")

        monkeypatch.setattr(stub_importer, "run", _bad)

        resp = client.post("/api/seed-import/stub_seeds", json={"count": "1"})
        assert resp.status_code == 400
        assert "Unknown cluster id" in resp.get_json()["message"]

    def test_upstream_error_maps_to_502(self, client, stub_importer, monkeypatch):
        def _boom(values):
            raise RuntimeError("service unreachable")

        monkeypatch.setattr(stub_importer, "run", _boom)

        resp = client.post("/api/seed-import/stub_seeds", json={"count": "1"})
        assert resp.status_code == 502
        assert "service unreachable" in resp.get_json()["message"]

    def test_non_list_return_maps_to_502(self, client, stub_importer, monkeypatch):
        monkeypatch.setattr(stub_importer, "run", lambda values: "not a list")

        resp = client.post("/api/seed-import/stub_seeds", json={"count": "1"})
        assert resp.status_code == 502

    def test_unknown_importer_is_404(self, client, stub_importer):
        resp = client.post("/api/seed-import/nope", json={})
        assert resp.status_code == 404

    def test_file_field_importer_accepts_multipart(self, client, monkeypatch, example_media_dir):
        class _UploadSeedImporter(SeedImporter):
            name = "upload_seeds"
            fields = [PluginField(key="listing", label="Listing", field_type="file")]

            def run(self, field_values):
                names = field_values["listing"].read().decode().split()
                return [SeedMediaItem(data=n.encode(), filename=f"{n}.wav") for n in names]

        importer = _UploadSeedImporter()
        monkeypatch.setattr("vtsearch.routes.media.seed.list_seed_importers", lambda: [importer])
        monkeypatch.setattr(
            "vtsearch.routes.media.seed.get_seed_importer",
            lambda name: importer if name == importer.name else None,
        )

        resp = client.post(
            "/api/seed-import/upload_seeds",
            data={"listing": (io.BytesIO(b"one two"), "list.txt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 201, resp.get_json()
        assert [item["original_name"] for item in resp.get_json()["items"]] == ["one.wav", "two.wav"]


class TestSeedImporterFieldOptions:
    def test_non_dynamic_field_is_400(self, client, stub_importer):
        resp = client.post(
            "/api/seed-import/stub_seeds/options",
            json={"field_key": "count", "values": {}},
        )
        assert resp.status_code == 400

    def test_unknown_field_is_400(self, client, stub_importer):
        resp = client.post(
            "/api/seed-import/stub_seeds/options",
            json={"field_key": "nope", "values": {}},
        )
        assert resp.status_code == 400

    def test_dynamic_options_are_normalised(self, client, monkeypatch):
        class _DynamicSeedImporter(SeedImporter):
            name = "dynamic_seeds"
            fields = [PluginField(key="cluster", label="Cluster", field_type="select", dynamic_options=True)]

            def get_field_options(self, field_key, current_values):
                return ["c1", ("c2", "Cluster two")]

        importer = _DynamicSeedImporter()
        monkeypatch.setattr("vtsearch.routes.media.seed.list_seed_importers", lambda: [importer])
        monkeypatch.setattr(
            "vtsearch.routes.media.seed.get_seed_importer",
            lambda name: importer if name == importer.name else None,
        )

        resp = client.post(
            "/api/seed-import/dynamic_seeds/options",
            json={"field_key": "cluster", "values": {}},
        )
        assert resp.status_code == 200
        assert resp.get_json()["options"] == [
            {"value": "c1", "label": "c1"},
            {"value": "c2", "label": "Cluster two"},
        ]

    def test_importer_without_dynamic_support_is_501(self, client, monkeypatch):
        class _NoOptionsSeedImporter(SeedImporter):
            name = "no_options"
            fields = [PluginField(key="cluster", label="Cluster", field_type="select", dynamic_options=True)]

        importer = _NoOptionsSeedImporter()
        monkeypatch.setattr("vtsearch.routes.media.seed.list_seed_importers", lambda: [importer])
        monkeypatch.setattr(
            "vtsearch.routes.media.seed.get_seed_importer",
            lambda name: importer if name == importer.name else None,
        )

        resp = client.post(
            "/api/seed-import/no_options/options",
            json={"field_key": "cluster", "values": {}},
        )
        assert resp.status_code == 501


class TestSeededDetectorCreate:
    """A detector created from seeds has examples but no training labels."""

    def test_seed_examples_do_not_become_training_labels(self, client, stub_importer, example_media_dir):
        seeds = client.post("/api/seed-import/stub_seeds", json={"count": "2"}).get_json()

        resp = client.post(
            "/api/detectors/registry",
            json={
                "name": "Near misses",
                "media_type": "audio",
                "text_query": "",
                "media_example": "",
                "examples": [{"type": "media", "value": item["filename"], "labeled": False} for item in seeds["items"]],
            },
        )
        assert resp.status_code == 201, resp.get_json()
        entry = resp.get_json()["detector"]

        # Both seeds are remembered as examples (they steer the first sort)…
        assert len(entry["examples"]) == 2
        assert all(ex["labeled"] is False for ex in entry["examples"])
        # …but neither is a label, so the detector starts untrained.
        assert entry["num_training"] == 0

    def test_hand_picked_examples_still_count_as_training_labels(self, client, stub_importer, example_media_dir):
        seeds = client.post("/api/seed-import/stub_seeds", json={"count": "1"}).get_json()

        resp = client.post(
            "/api/detectors/registry",
            json={
                "name": "Hand picked",
                "media_type": "audio",
                "text_query": "",
                "media_example": "",
                "examples": [{"type": "media", "value": seeds["items"][0]["filename"]}],
            },
        )
        assert resp.status_code == 201, resp.get_json()
        assert resp.get_json()["detector"]["num_training"] == 1
