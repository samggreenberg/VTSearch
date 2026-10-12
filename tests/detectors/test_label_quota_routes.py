"""Test, AutoFind, Find and the CLI give the Goods' centroid under the label quota (#4643).

The rule lives in the library (``tests_lib/detectors/test_label_quota.py``);
these pin that every path that hands a detector out goes through it and says so:
``/api/find-label`` and ``/api/auto-detect`` report ``label_quota``, a cold
Find asks for the quota, a labelset with no Good is refused with a reason, and
the CLI names a detector it ran as the centroid.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vtsearch.settings import get_detectors_dir

DIM = 4
EMB = "test_embedder"


@pytest.fixture(autouse=True)
def clean_detectors_dir():
    import shutil

    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)
    yield
    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)


def _unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


def _media(cid: int, vec: np.ndarray) -> dict:
    return {
        "id": cid,
        "media_type": "image",
        "embedder": EMB,
        "embeddings": {EMB: vec},
        "filename": f"m{cid}.png",
        "md5": f"m{cid}",
        "origin_name": f"m{cid}.png",
        "origin": {"importer": "test", "params": {}},
    }


def _corpus() -> dict[int, dict]:
    """Twelve media: ids 1-6 lean toward one direction, ids 7-12 toward another."""
    rng = np.random.default_rng(4643)
    corpus: dict[int, dict] = {}
    for cid in range(1, 13):
        base = np.array([1.0, 0.2, 0.0, 0.0]) if cid <= 6 else np.array([0.0, 0.2, 1.0, 0.0])
        corpus[cid] = _media(cid, _unit(base + 0.1 * rng.standard_normal(DIM)))
    return corpus


def _activate(corpus: dict[int, dict]):
    from vtscore.state.core import DatasetContext, register_context, set_thread_dataset_context

    ctx = DatasetContext("ds-label-quota")
    ctx.medias.update(corpus)
    register_context(ctx)
    set_thread_dataset_context(ctx)
    return ctx


def _labels(corpus: dict[int, dict], goods: list[int], bads: list[int]) -> list[dict]:
    return [
        {
            "md5": corpus[cid]["md5"],
            "label": label,
            "origin": corpus[cid]["origin"],
            "origin_name": corpus[cid]["origin_name"],
            "filename": corpus[cid]["filename"],
        }
        for label, ids in (("good", goods), ("bad", bads))
        for cid in ids
    ]


def _write_detector(name: str, corpus: dict[int, dict], goods: list[int], bads: list[int]) -> str:
    from vtscore.detectors.registry import register_detector
    from vtscore.detectors.store import _detector_path
    from vtscore.detectors.store import _write_detector as write

    labels = _labels(corpus, goods, bads)
    write(_detector_path(name), {"name": name, "media_type": "image", "labelset": {"labels": labels}})
    return register_detector(name=name, media_type="image", num_training=len(labels))["id"]


class TestFindLabel:
    def test_one_good_is_enough_and_gives_the_goods_centroid(self, client):
        corpus = _corpus()
        _activate(corpus)
        detector_id = _write_detector("one-good", corpus, goods=[1], bads=[])

        resp = client.post("/api/find-label", json={"detector_id": detector_id})
        assert resp.status_code == 200, resp.get_json()
        body = resp.get_json()
        assert body["label_quota"] == {
            "tier": "centroid",
            "n_good": 1,
            "n_bad": 0,
            "goods_owed": 2,
            "bads_owed": 4,
            "good_quota": 3,
            "bad_quota": 4,
            "dry_bad_quota": 16,
        }
        assert body["threshold"] == 0.5
        # The centroid of one Good is that Good: the media leaning its way rank first.
        assert {r["id"] for r in body["results"][:6]} == set(range(1, 7))

    def test_under_the_bad_quota_it_is_still_the_centroid(self, client):
        corpus = _corpus()
        _activate(corpus)
        detector_id = _write_detector("few-bads", corpus, goods=[1, 2, 3], bads=[7, 8, 9])

        quota = client.post("/api/find-label", json={"detector_id": detector_id}).get_json()["label_quota"]
        assert (quota["tier"], quota["goods_owed"], quota["bads_owed"]) == ("centroid", 0, 1)

    def test_at_the_quota_it_is_the_trained_detector(self, client):
        corpus = _corpus()
        _activate(corpus)
        detector_id = _write_detector("at-quota", corpus, goods=[1, 2, 3], bads=[7, 8, 9, 10])

        quota = client.post("/api/find-label", json={"detector_id": detector_id}).get_json()["label_quota"]
        assert (quota["tier"], quota["goods_owed"], quota["bads_owed"]) == ("trained", 0, 0)

    def test_no_good_is_refused_with_the_reason(self, client):
        corpus = _corpus()
        _activate(corpus)
        detector_id = _write_detector("no-good", corpus, goods=[], bads=[7, 8])

        resp = client.post("/api/find-label", json={"detector_id": detector_id})
        assert resp.status_code == 400
        assert "at least one Good" in resp.get_json()["message"]


class TestAutoFind:
    def test_each_detector_reports_its_tier(self, client):
        from vtsearch.settings import add_autofind_detector

        corpus = _corpus()
        _activate(corpus)
        _write_detector("af-centroid", corpus, goods=[1, 2], bads=[7])
        _write_detector("af-trained", corpus, goods=[1, 2, 3], bads=[7, 8, 9, 10])
        add_autofind_detector("af-centroid")
        add_autofind_detector("af-trained")

        resp = client.post("/api/auto-detect", json={})
        assert resp.status_code == 200, resp.get_json()
        results = resp.get_json()["results"]
        assert results["af-centroid"]["label_quota"]["tier"] == "centroid"
        assert results["af-centroid"]["label_quota"]["goods_owed"] == 1
        assert results["af-trained"]["label_quota"]["tier"] == "trained"


class TestColdFind:
    def test_a_cold_find_asks_for_the_quota_and_scores_the_centroid(self, monkeypatch):
        import app as app_module
        import vtscore.detectors.labelset_training as lt
        import vtsearch.routes.detectors.find as find_mod
        from vtscore.detectors.centroid_head import is_centroid_head

        corpus = _corpus()
        seen: list = []
        real = lt.labelset_train_and_score

        def _spy(*args, **kwargs):
            out = real(*args, **kwargs)
            seen.append((kwargs.get("label_quota"), out[2]))
            return out

        monkeypatch.setattr(lt, "labelset_train_and_score", _spy)
        monkeypatch.setattr(find_mod, "_load_find_dataset_medias", lambda ds: corpus)
        dc = {
            "name": "cold-centroid",
            "detector_id": "cold-centroid",
            "embedder_type": "",
            "detector_data": {"media_type": "image", "labelset": {"labels": _labels(corpus, [1], [7])}},
        }
        with app_module.app.test_request_context("/api/find"):
            positives, _negatives, *_ = find_mod._score_dataset({"name": "corpus", "pkl_path": "x"}, [dc], 0, 0)

        assert len(seen) == 1 and seen[0][0] is True and is_centroid_head(seen[0][1])
        assert positives and {m["id"] for m in positives} <= set(range(1, 7))


class TestCli:
    def test_the_cli_names_a_detector_it_ran_as_the_centroid(self, capsys):
        from vtscore import cli, cli_progress
        from vtscore.datasets.labelset import LabeledElement, LabelSet
        from vtscore.detectors.centroid_head import centroid_head
        from vtscore.state.core import DetectorContext

        ctx = DetectorContext("cli", name="cli", media_type="image")
        ctx.model = centroid_head(_unit([1, 0, 0, 0]), 0.5)
        ls = LabelSet([LabeledElement(md5="a", label="good"), LabeledElement(md5="b", label="bad")])
        previous = cli_progress.get_format()
        cli_progress.set_format("json")
        try:
            cli._report_centroid("cli", ctx, ls)
        finally:
            cli_progress.set_format(previous)
        event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert event["event"] == "detector_centroid"
        assert (event["tier"], event["goods_owed"], event["bads_owed"]) == ("centroid", 2, 3)

    def test_a_trained_detector_is_not_named(self, capsys):
        from vtscore import cli
        from vtscore.datasets.labelset import LabelSet
        from vtscore.state.core import DetectorContext
        from vtscore.training.mlp import LINEAR_SVM_HEAD, build_model

        ctx = DetectorContext("cli", name="cli", media_type="image")
        ctx.model = build_model(DIM, hidden_dim=LINEAR_SVM_HEAD)
        cli._report_centroid("cli", ctx, LabelSet([]))
        assert "centroid" not in capsys.readouterr().out
