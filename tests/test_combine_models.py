"""Tests for combining trainable models via labelset merge.

Covers both the underlying ``LabelSet.merge`` helper and the
``POST /api/trainable-models/combine`` endpoint.
"""

from __future__ import annotations

import shutil

import pytest

from vtsearch.datasets.labelset import LabeledElement, LabelSet
from vtsearch.settings import get_trainable_models_dir


@pytest.fixture(autouse=True)
def clean_trainable_models_dir():
    tm_dir = get_trainable_models_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)
    yield
    tm_dir = get_trainable_models_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)


def _el(md5, label, *, importer="server_folder", path="/data", name="", metadata=None):
    """Build a LabeledElement with a fully-populated origin."""
    origin = {"importer": importer, "params": {"path": path}}
    return LabeledElement(
        md5=md5,
        label=label,
        origin=origin,
        origin_name=name or md5,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# LabelSet.merge — direct unit tests
# ---------------------------------------------------------------------------


class TestLabelSetMerge:
    def test_merge_disjoint_origins(self):
        a = LabelSet([_el("aa", "good", path="/A", name="a1")])
        b = LabelSet([_el("bb", "bad", path="/B", name="b1")])
        merged = a.merge(b)
        assert len(merged) == 2
        labels = {(e.md5, e.label) for e in merged}
        assert labels == {("aa", "good"), ("bb", "bad")}

    def test_merge_dedupes_same_origin_same_label(self):
        e1 = _el("xx", "good", path="/A", name="same")
        e2 = _el("xx", "good", path="/A", name="same")
        merged = LabelSet([e1]).merge(LabelSet([e2]))
        assert len(merged) == 1
        assert merged.elements[0].label == "good"

    def test_merge_drops_conflicting_origins(self):
        e_good = _el("xx", "good", path="/A", name="same")
        e_bad = _el("xx", "bad", path="/A", name="same")
        e_other = _el("yy", "good", path="/A", name="other")
        merged = LabelSet([e_good, e_other]).merge(LabelSet([e_bad]))
        keys = {(e.md5, e.label) for e in merged}
        assert keys == {("yy", "good")}

    def test_merge_falls_back_to_md5_for_legacy_no_origin(self):
        # Two legacy entries with no origin but same md5 + label dedupe;
        # if labels disagree they drop.
        legacy_good = LabeledElement(md5="aa", label="good")
        legacy_good_2 = LabeledElement(md5="aa", label="good")
        legacy_bad = LabeledElement(md5="bb", label="bad")
        legacy_bad_conflict = LabeledElement(md5="bb", label="good")

        merged = LabelSet([legacy_good, legacy_bad]).merge(LabelSet([legacy_good_2, legacy_bad_conflict]))
        keys = [(e.md5, e.label) for e in merged]
        assert keys == [("aa", "good")]  # bb dropped on conflict

    def test_merge_drops_elements_with_neither_origin_nor_md5(self):
        ghost = LabeledElement(md5="", label="good", origin=None)
        real = _el("aa", "good")
        merged = LabelSet([ghost]).merge(LabelSet([real]))
        assert len(merged) == 1
        assert merged.elements[0].md5 == "aa"

    def test_merge_shallow_merges_metadata_later_wins(self):
        e1 = _el("xx", "good", metadata={"src": "A", "shared": 1})
        e2 = _el("xx", "good", metadata={"shared": 2, "extra": "B"})
        merged = LabelSet([e1]).merge(LabelSet([e2]))
        assert len(merged) == 1
        assert merged.elements[0].metadata == {"src": "A", "shared": 2, "extra": "B"}

    def test_merge_distinguishes_by_origin_name_within_same_origin(self):
        # Same importer+params but different origin_name → distinct elements.
        a = _el("a1", "good", path="/A", name="file_1.wav")
        b = _el("a2", "bad", path="/A", name="file_2.wav")
        merged = LabelSet([a]).merge(LabelSet([b]))
        assert len(merged) == 2

    def test_merge_preserves_first_seen_order(self):
        a = _el("aa", "good", name="a")
        b = _el("bb", "good", name="b")
        c = _el("cc", "good", name="c")
        merged = LabelSet([a, b]).merge(LabelSet([c, a]))  # a duplicated
        assert [e.origin_name for e in merged] == ["a", "b", "c"]

    def test_merge_rejects_unknown_conflict_policy(self):
        with pytest.raises(ValueError):
            LabelSet([]).merge(LabelSet([]), conflict_policy="majority")

    def test_merge_three_way_conflict_drops_all_disputed(self):
        a = _el("xx", "good")
        b = _el("xx", "good")
        c = _el("xx", "bad")
        merged = LabelSet([a]).merge(LabelSet([b]), LabelSet([c]))
        assert len(merged) == 0


# ---------------------------------------------------------------------------
# POST /api/trainable-models/combine endpoint
# ---------------------------------------------------------------------------


def _create_model(client, name, *, media_type="audio", text_query="q"):
    res = client.post(
        "/api/trainable-models",
        json={"name": name, "media_type": media_type, "text_query": text_query},
    )
    assert res.status_code == 201, res.get_json()


def _save_labelset(client, name, labelset_dict):
    """Directly write a labelset onto an existing trainable model on disk."""
    from vtsearch.models.trainable_model_store import _model_path, _read_model, _write_model

    p = _model_path(name)
    data = _read_model(p)
    assert data is not None
    data["labelset"] = labelset_dict
    _write_model(p, data)


class TestCombineEndpointValidation:
    def test_missing_names(self, client):
        res = client.post("/api/trainable-models/combine", json={"new_name": "X"})
        assert res.status_code == 400
        assert "names" in res.get_json()["error"]

    def test_only_one_name(self, client):
        res = client.post(
            "/api/trainable-models/combine",
            json={"names": ["A"], "new_name": "X"},
        )
        assert res.status_code == 400

    def test_missing_new_name(self, client):
        res = client.post(
            "/api/trainable-models/combine",
            json={"names": ["A", "B"]},
        )
        assert res.status_code == 400
        assert "new_name" in res.get_json()["error"]

    def test_unknown_source_model(self, client):
        _create_model(client, "A")
        res = client.post(
            "/api/trainable-models/combine",
            json={"names": ["A", "Nope"], "new_name": "X"},
        )
        assert res.status_code == 404

    def test_media_type_mismatch(self, client):
        _create_model(client, "A", media_type="audio")
        _create_model(client, "B", media_type="image")
        res = client.post(
            "/api/trainable-models/combine",
            json={"names": ["A", "B"], "new_name": "X"},
        )
        assert res.status_code == 400
        assert "media_type" in res.get_json()["error"]

    def test_new_name_collision(self, client):
        _create_model(client, "A")
        _create_model(client, "B")
        _create_model(client, "X")  # already exists
        # Give A and B at least one matching label so the merge isn't empty.
        ls = LabelSet([_el("aa", "good")]).to_dict()
        _save_labelset(client, "A", ls)
        _save_labelset(client, "B", ls)
        res = client.post(
            "/api/trainable-models/combine",
            json={"names": ["A", "B"], "new_name": "X"},
        )
        assert res.status_code == 409

    def test_unsupported_conflict_policy(self, client):
        _create_model(client, "A")
        _create_model(client, "B")
        res = client.post(
            "/api/trainable-models/combine",
            json={"names": ["A", "B"], "new_name": "X", "conflict_policy": "majority"},
        )
        assert res.status_code == 400

    def test_empty_after_merge_rejected(self, client):
        _create_model(client, "A")
        _create_model(client, "B")
        _save_labelset(client, "A", LabelSet([_el("aa", "good")]).to_dict())
        _save_labelset(client, "B", LabelSet([_el("aa", "bad")]).to_dict())
        res = client.post(
            "/api/trainable-models/combine",
            json={"names": ["A", "B"], "new_name": "X"},
        )
        assert res.status_code == 422
        assert "empty" in res.get_json()["error"].lower()


class TestCombineEndpointSuccess:
    def test_combine_disjoint_labelsets(self, client):
        _create_model(client, "A", text_query="alpha")
        _create_model(client, "B", text_query="beta")
        _save_labelset(
            client,
            "A",
            LabelSet([_el("aa", "good", name="a1"), _el("bb", "bad", name="b1")]).to_dict(),
        )
        _save_labelset(
            client,
            "B",
            LabelSet([_el("cc", "good", name="c1")]).to_dict(),
        )

        res = client.post(
            "/api/trainable-models/combine",
            json={"names": ["A", "B"], "new_name": "Combined"},
        )
        assert res.status_code == 201
        body = res.get_json()
        assert body["success"] is True
        assert body["num_labels"] == 3
        assert body["combined_from"] == ["A", "B"]
        assert body["source_label_counts"] == [2, 1]
        assert body["media_type"] == "audio"

    def test_combine_persists_combined_model(self, client):
        _create_model(client, "A", text_query="alpha")
        _create_model(client, "B", text_query="beta")
        _save_labelset(client, "A", LabelSet([_el("aa", "good")]).to_dict())
        _save_labelset(client, "B", LabelSet([_el("bb", "good")]).to_dict())

        client.post(
            "/api/trainable-models/combine",
            json={"names": ["A", "B"], "new_name": "Combined"},
        )

        got = client.get("/api/trainable-models/Combined").get_json()
        assert got["name"] == "Combined"
        assert got["combined_from"] == ["A", "B"]
        assert len(got["labelset"]["labels"]) == 2
        # Should NOT inherit any labelset_source from sources.
        assert "labelset_source" not in got

    def test_combine_drops_conflicts_keeps_agreements(self, client):
        _create_model(client, "A")
        _create_model(client, "B")
        _save_labelset(
            client,
            "A",
            LabelSet([_el("agree", "good"), _el("conflict", "good")]).to_dict(),
        )
        _save_labelset(
            client,
            "B",
            LabelSet([_el("agree", "good"), _el("conflict", "bad")]).to_dict(),
        )
        res = client.post(
            "/api/trainable-models/combine",
            json={"names": ["A", "B"], "new_name": "Combined"},
        )
        assert res.status_code == 201
        labels = res.get_json()
        assert labels["num_labels"] == 1

    def test_combine_dedupes_examples(self, client):
        _create_model(client, "A", text_query="dog barking")
        _create_model(client, "B", text_query="dog barking")  # same text_query
        _save_labelset(client, "A", LabelSet([_el("aa", "good")]).to_dict())
        _save_labelset(client, "B", LabelSet([_el("bb", "good")]).to_dict())

        res = client.post(
            "/api/trainable-models/combine",
            json={"names": ["A", "B"], "new_name": "Combined"},
        )
        body = res.get_json()
        # Both source models auto-created an example for their text_query.
        # Dedup should collapse the duplicates.
        assert body["examples"] == [{"type": "text", "value": "dog barking"}]

    def test_combine_three_models(self, client):
        _create_model(client, "A")
        _create_model(client, "B")
        _create_model(client, "C")
        _save_labelset(client, "A", LabelSet([_el("aa", "good")]).to_dict())
        _save_labelset(client, "B", LabelSet([_el("bb", "good")]).to_dict())
        _save_labelset(client, "C", LabelSet([_el("cc", "bad")]).to_dict())

        res = client.post(
            "/api/trainable-models/combine",
            json={"names": ["A", "B", "C"], "new_name": "Combined"},
        )
        assert res.status_code == 201
        assert res.get_json()["num_labels"] == 3
        assert res.get_json()["source_label_counts"] == [1, 1, 1]
