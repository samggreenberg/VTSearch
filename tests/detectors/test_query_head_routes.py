"""Test, AutoFind, the export and the CLI give a typed-query detector its query's own set under the quota (#4758).

The head lives in the library (``tests_lib/detectors/test_query_head.py``);
these pin that every path that hands a detector out gives it and says so.  A
detector made from a typed query reports ``label_quota.tier == "query"`` and
returns the text sort's set, from before its first label until its labels give
a trained head.  A detector with no query keeps the Goods' centroid and the
refusal with no Good (``test_label_quota_routes.py``), and so does a dataset
with no text slot to embed the query in.

The corpus is dual-embedder, as SigLIP text + DINOv3 patch is: the detector's
labels live in one space and the typed query in the dataset's text slot, of a
different width.  Its one Good leans one way in the label space and the query
points elsewhere in the text space, so the Goods' centroid and the query rank
different media first.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

import vtscore.detectors.query_head as qh
from vtsearch.settings import get_detectors_dir

EMB = "test_embedder"
TEXT_EMB = "test_text_embedder"
DIM = 4
TEXT_DIM = 8
QUERY = "a planted thing"
N_MEDIA = 60
#: The media the typed query points at in the text space.
QUERY_CLUSTER = set(range(7, 13))
#: The media the Good (id 1) leans toward in the label space.
GOOD_CLUSTER = set(range(1, 7))


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


def _corpus() -> tuple[dict[int, dict], np.ndarray]:
    """Sixty media in two spaces, and the typed query's vector in the text one.

    Label space: ids 1-6 lean one way, the rest another.  Text space: ids 7-12
    sit tight around the query, the rest point anywhere.
    """
    rng = np.random.default_rng(4758)
    query = _unit(rng.standard_normal(TEXT_DIM))
    corpus: dict[int, dict] = {}
    for cid in range(1, N_MEDIA + 1):
        base = np.array([1.0, 0.2, 0.0, 0.0]) if cid in GOOD_CLUSTER else np.array([0.0, 0.2, 1.0, 0.0])
        text = query + 0.1 * rng.standard_normal(TEXT_DIM) if cid in QUERY_CLUSTER else rng.standard_normal(TEXT_DIM)
        corpus[cid] = {
            "id": cid,
            "media_type": "image",
            "embedder": EMB,
            "embeddings": {
                EMB: _unit(base + 0.1 * rng.standard_normal(DIM)),
                TEXT_EMB: _unit(text),
            },
            "filename": f"m{cid}.png",
            "md5": f"m{cid}",
            "origin_name": f"m{cid}.png",
            "origin": {"importer": "test", "params": {}},
        }
    return corpus, query


@pytest.fixture
def world(monkeypatch):
    """The corpus, active, with a text slot that embeds the typed query; ``(corpus, query_vec)``."""
    from vtscore.state.core import DatasetContext, register_context, set_thread_dataset_context

    corpus, query = _corpus()
    ctx = DatasetContext("ds-query-head")
    ctx.medias.update(corpus)
    register_context(ctx)
    set_thread_dataset_context(ctx)
    monkeypatch.setattr(qh, "text_embedder_for_snap", lambda snap: TEXT_EMB if snap else None)
    monkeypatch.setattr(qh, "embed_detector_query", lambda text, media_type, embedder: query if text == QUERY else None)
    return corpus, query


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


def _write_detector(
    name: str,
    corpus: dict[int, dict],
    goods: list[int],
    bads: list[int],
    *,
    text_query: str = "",
    examples: list[dict] | None = None,
) -> str:
    """A detector file and registry entry; *text_query* / *examples* as the New Detector modal writes them."""
    from vtscore.detectors.registry import register_detector
    from vtscore.detectors.store import _detector_path
    from vtscore.detectors.store import _write_detector as write

    labels = _labels(corpus, goods, bads)
    data = {"name": name, "media_type": "image", "labelset": {"labels": labels}, "text_query": text_query}
    if examples is not None:
        data["examples"] = examples
    write(_detector_path(name), data)
    return register_detector(
        name=name, media_type="image", num_training=len(labels), text_query=text_query, examples=examples
    )["id"]


def _text_sort_set(corpus: dict[int, dict], query: np.ndarray, beta: float, *, nudge: float) -> set[int]:
    """The media the text sort keeps for *query* at *beta*: cosine at or above its display line (+ *nudge*)."""
    from vtscore.training.thresholds import text_sort_cuts

    ids = sorted(corpus)
    cosines = [float(np.dot(corpus[c]["embeddings"][TEXT_EMB], query)) for c in ids]
    line = text_sort_cuts(cosines, beta=beta).threshold + nudge
    return {c for c, cos in zip(ids, cosines, strict=True) if cos >= line}


def _assert_text_sorts_set(kept: set[int], corpus: dict[int, dict], query: np.ndarray, beta: float) -> None:
    """*kept* is the set the text sort draws at *beta* (to float32 rounding at its line), and not empty."""
    assert kept in (
        _text_sort_set(corpus, query, beta, nudge=-1e-6),
        _text_sort_set(corpus, query, beta, nudge=1e-6),
    ), kept
    assert kept


def _find_label(client, detector_id: str) -> dict:
    resp = client.post("/api/find-label", json={"detector_id": detector_id})
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


class TestFindLabel:
    def test_before_any_label_it_is_the_typed_querys_own_set(self, client, world):
        """Before #4758 a detector with no label was refused (400); now it is its query."""
        corpus, query = world
        detector_id = _write_detector("query-empty", corpus, goods=[], bads=[], text_query=QUERY)

        body = _find_label(client, detector_id)
        assert body["label_quota"] == {
            "tier": "query",
            "n_good": 0,
            "n_bad": 0,
            "goods_owed": 3,
            "bads_owed": 4,
            "good_quota": 3,
            "bad_quota": 4,
            "dry_bad_quota": 16,
            "hold_labels": 30,
            "hold_owed": 0,
        }
        assert body["threshold"] == 0.5
        # Ranked by cosine to the query, in the text space.
        cosine = {c: float(np.dot(m["embeddings"][TEXT_EMB], query)) for c, m in corpus.items()}
        assert [r["id"] for r in body["results"]] == sorted(cosine, key=lambda c: -cosine[c])
        kept = {r["id"] for r in body["results"] if r["score"] >= body["threshold"]}
        _assert_text_sorts_set(kept, corpus, query, body["balance"]["beta"])

    def test_under_the_quota_it_is_the_query_not_the_goods_centroid(self, client, world):
        corpus, query = world
        detector_id = _write_detector("query-one-good", corpus, goods=[1], bads=[40, 41], text_query=QUERY)

        body = _find_label(client, detector_id)
        quota = body["label_quota"]
        assert (quota["tier"], quota["n_good"], quota["n_bad"]) == ("query", 1, 2)
        assert (quota["goods_owed"], quota["bads_owed"]) == (2, 2)
        # The Goods' centroid would rank the Good's cluster first; the query ranks its own.
        assert {r["id"] for r in body["results"][:6]} == QUERY_CLUSTER
        kept = {r["id"] for r in body["results"] if r["score"] >= body["threshold"]}
        _assert_text_sorts_set(kept, corpus, query, body["balance"]["beta"])

    def test_a_first_text_example_is_the_query_too(self, client, world):
        """A detector file with no ``text_query`` but a text seed example: that example is its query."""
        corpus, _query = world
        detector_id = _write_detector(
            "query-example",
            corpus,
            goods=[],
            bads=[],
            examples=[{"type": "media", "value": "x.png"}, {"type": "text", "value": QUERY}],
        )
        assert _find_label(client, detector_id)["label_quota"]["tier"] == "query"

    def test_at_the_quota_it_is_the_trained_detector(self, client, world):
        corpus, _query = world
        detector_id = _write_detector(
            "query-at-quota", corpus, goods=[1, 2, 3], bads=[40, 41, 42, 43], text_query=QUERY
        )
        quota = _find_label(client, detector_id)["label_quota"]
        assert (quota["tier"], quota["goods_owed"], quota["bads_owed"]) == ("trained", 0, 0)

    def test_without_a_query_it_is_the_goods_centroid_as_before(self, client, world):
        corpus, _query = world
        detector_id = _write_detector("no-query-one-good", corpus, goods=[1], bads=[])

        body = _find_label(client, detector_id)
        assert body["label_quota"]["tier"] == "centroid"
        assert {r["id"] for r in body["results"][:6]} == GOOD_CLUSTER

    def test_without_a_query_or_a_label_it_is_still_refused(self, client, world):
        corpus, _query = world
        detector_id = _write_detector("no-query-empty", corpus, goods=[], bads=[])

        resp = client.post("/api/find-label", json={"detector_id": detector_id})
        assert resp.status_code == 400
        assert "no labels" in resp.get_json()["message"]

    def test_without_a_query_or_a_good_it_is_still_refused_with_the_reason(self, client, world):
        corpus, _query = world
        detector_id = _write_detector("no-query-no-good", corpus, goods=[], bads=[40, 41])

        resp = client.post("/api/find-label", json={"detector_id": detector_id})
        assert resp.status_code == 400
        assert "at least one Good" in resp.get_json()["message"]

    def test_a_dataset_with_no_text_slot_gives_the_centroid_and_nothing(self, client, world, monkeypatch):
        """No text slot, no way to embed the query: the detector falls back to what its labels give."""
        corpus, _query = world
        monkeypatch.setattr(qh, "text_embedder_for_snap", lambda snap: None)
        one_good = _write_detector("query-no-slot", corpus, goods=[1], bads=[], text_query=QUERY)
        empty = _write_detector("query-no-slot-empty", corpus, goods=[], bads=[], text_query=QUERY)

        assert _find_label(client, one_good)["label_quota"]["tier"] == "centroid"
        assert client.post("/api/find-label", json={"detector_id": empty}).status_code == 400


class TestAutoFind:
    def test_a_typed_query_detector_runs_as_its_query_and_says_so(self, client, world):
        """Before #4758 AutoFind dropped a detector with no Good; now a typed-query one runs as its query."""
        from vtsearch.settings import add_autofind_detector

        corpus, _query = world
        _write_detector("af-query", corpus, goods=[], bads=[], text_query=QUERY)
        _write_detector("af-centroid", corpus, goods=[1, 2], bads=[40])
        add_autofind_detector("af-query")
        add_autofind_detector("af-centroid")

        resp = client.post("/api/auto-detect", json={})
        assert resp.status_code == 200, resp.get_json()
        results = resp.get_json()["results"]
        assert results["af-query"]["label_quota"]["tier"] == "query"
        assert results["af-query"]["label_quota"]["goods_owed"] == 3
        hits = {h["id"] for h in results["af-query"]["hits"]}
        assert hits and hits <= QUERY_CLUSTER, hits
        assert results["af-centroid"]["label_quota"]["tier"] == "centroid"


class TestPortableExport:
    def test_the_manifest_names_the_text_embedder_the_query_scores_in(self, client, world):
        """The bundle's head is the query's, so its embedder and width are the text slot's, not the labels'."""
        corpus, _query = world
        detector_id = _write_detector("query-export", corpus, goods=[1], bads=[], text_query=QUERY)

        resp = client.post(f"/api/detectors/{detector_id}/portable-bundle")
        assert resp.status_code == 200, resp.get_json()
        with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
            manifest = json.loads(zf.read("manifest.json"))
        assert manifest["embedder"]["name"] == TEXT_EMB
        assert manifest["embedder"]["embedding_dim"] == TEXT_DIM
        assert manifest["scoring"]["threshold"] == 0.5
        assert manifest["training_labels"] == {"good": 1, "bad": 0}

    def test_without_a_query_it_names_the_detectors_own_embedder(self, client, world):
        corpus, _query = world
        detector_id = _write_detector("centroid-export", corpus, goods=[1], bads=[])

        resp = client.post(f"/api/detectors/{detector_id}/portable-bundle")
        assert resp.status_code == 200, resp.get_json()
        with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
            manifest = json.loads(zf.read("manifest.json"))
        assert (manifest["embedder"]["name"], manifest["embedder"]["embedding_dim"]) == (EMB, DIM)


class TestCli:
    def test_a_label_less_typed_query_detector_runs_and_says_so(self, world, capsys):
        """Before #4758 the CLI raised "has no labels"; now it scores the query, in the text space."""
        from vtscore import cli, cli_progress
        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.query_head import is_query_head
        from vtscore.state.core import DetectorContext

        corpus, _query = world
        _write_detector("cli-query", corpus, goods=[], bads=[], text_query=QUERY)

        previous = cli_progress.get_format()
        cli_progress.set_format("json")
        try:
            out = cli._load_and_train_detectors(["cli-query"], "image", corpus)
        finally:
            cli_progress.set_format(previous)

        assert is_query_head(out["cli-query"]["mlp"])
        assert out["cli-query"]["embedder"] == TEXT_EMB
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
        (event,) = [e for e in events if e.get("event") in ("detector_query", "detector_centroid")]
        assert event["event"] == "detector_query"
        assert event["detector"] == "cli-query"
        assert (event["tier"], event["n_good"], event["goods_owed"], event["bads_owed"]) == ("query", 0, 3, 4)

        # The text-mode line names the query it scores as.
        ctx = DetectorContext("cli-query", name="cli-query", media_type="image")
        ctx.model = out["cli-query"]["mlp"]
        cli._report_centroid("cli-query", ctx, LabelSet([]))
        line = capsys.readouterr().out.strip()
        assert f"its typed query's own set ('{QUERY}')" in line and "3 more Goods and 4 more Bads" in line

    def test_without_a_query_a_label_less_detector_still_raises(self, world):
        from vtscore import cli

        corpus, _query = world
        _write_detector("cli-empty", corpus, goods=[], bads=[])
        with pytest.raises(ValueError, match="has no labels"):
            cli._load_and_train_detectors(["cli-empty"], "image", corpus)


#: The typed query a re-sort swaps in (#4801), and the media whose text vector it embeds as.
RESORT_QUERY = "another planted thing"
RESORT_TARGET = 30


@pytest.fixture
def resort_world(world, monkeypatch):
    """``world`` with a second typed query, a re-sort's; ``(corpus, query_vec, resort_vec)``."""
    corpus, query = world
    resort = corpus[RESORT_TARGET]["embeddings"][TEXT_EMB]
    vectors = {QUERY: query, RESORT_QUERY: resort}
    monkeypatch.setattr(qh, "embed_detector_query", lambda text, media_type, embedder: vectors.get(text))
    return corpus, query, resort


def _set_examples(client, name: str, examples: list[dict]) -> None:
    resp = client.put(f"/api/detectors/{name}/examples", json={"examples": examples})
    assert resp.status_code == 200, resp.get_json()


class TestReSort:
    """The Train view's re-sort prompt saves its example (#4801), so Find gives what the opening sorts by."""

    def test_a_re_sorted_typed_query_detectors_test_is_the_new_querys_set(self, client, resort_world):
        corpus, _query, resort = resort_world
        detector_id = _write_detector("resorted", corpus, goods=[1], bads=[40], text_query=QUERY)
        assert {r["id"] for r in _find_label(client, detector_id)["results"][:6]} == QUERY_CLUSTER

        _set_examples(client, "resorted", [{"type": "text", "value": RESORT_QUERY}])

        body = _find_label(client, detector_id)
        assert body["label_quota"]["tier"] == "query"
        cosine = {c: float(np.dot(m["embeddings"][TEXT_EMB], resort)) for c, m in corpus.items()}
        assert [r["id"] for r in body["results"]] == sorted(cosine, key=lambda c: -cosine[c])
        assert body["results"][0]["id"] == RESORT_TARGET
        kept = {r["id"] for r in body["results"] if r["score"] >= body["threshold"]}
        _assert_text_sorts_set(kept, corpus, resort, body["balance"]["beta"])

    def test_the_registry_entry_takes_the_new_query_too(self, client, resort_world):
        """The next Train session opens on the registry entry's query, so it must not keep the old one."""
        from vtscore.detectors.registry import get_detector
        from vtscore.detectors.store import _detector_path, _read_detector

        corpus, _query, _resort = resort_world
        detector_id = _write_detector(
            "resorted-registry",
            corpus,
            goods=[],
            bads=[],
            text_query=QUERY,
            examples=[{"type": "text", "value": QUERY}],
        )

        _set_examples(client, "resorted-registry", [{"type": "text", "value": RESORT_QUERY}])

        entry = get_detector(detector_id)
        data = _read_detector(_detector_path("resorted-registry"))
        assert entry is not None and data is not None
        for record in (entry, data):
            assert record["text_query"] == RESORT_QUERY
            assert record["examples"] == [{"type": "text", "value": RESORT_QUERY}]

    def test_a_re_sort_to_a_media_seed_drops_the_query_and_votes_nothing(self, client, resort_world):
        """A media re-sort steers the sort, as a seed does: no query, and no Good the user did not vote."""
        from vtscore.detectors.registry import get_detector
        from vtscore.detectors.store import _detector_path, _read_detector

        corpus, _query, _resort = resort_world
        detector_id = _write_detector("resorted-media", corpus, goods=[1], bads=[40], text_query=QUERY)
        seed = {"type": "media", "value": "seed.png", "labeled": False}

        _set_examples(client, "resorted-media", [seed])

        entry = get_detector(detector_id)
        data = _read_detector(_detector_path("resorted-media"))
        assert entry is not None and data is not None
        for record in (entry, data):
            assert (record["text_query"], record["media_example"]) == ("", "seed.png")
        assert len(data["labelset"]["labels"]) == 2
        # With no query, the detector is its Goods' centroid under the quota, as before #4758.
        body = _find_label(client, detector_id)
        assert body["label_quota"]["tier"] == "centroid"
        assert {r["id"] for r in body["results"][:6]} == GOOD_CLUSTER

    def test_a_loaded_detector_drops_the_old_querys_head_and_keeps_its_session(self, client, resort_world):
        """The write re-points the live labelset cache (no rehydrate) and drops the stale query head."""
        from vtscore.detectors.dataset_sync import _detector_file_mtime
        from vtscore.detectors.query_head import query_head_text
        from vtscore.detectors.store import _detector_path
        from vtscore.state.core import DetectorContext, register_detector_context

        corpus, _query, _resort = resort_world
        detector_id = _write_detector("resorted-live", corpus, goods=[1], bads=[40], text_query=QUERY)
        det_ctx = DetectorContext(detector_id, name="resorted-live", media_type="image")
        register_detector_context(det_ctx)
        _find_label(client, detector_id)
        assert query_head_text(det_ctx.model) == QUERY

        _set_examples(client, "resorted-live", [{"type": "text", "value": RESORT_QUERY}])

        assert det_ctx.model is None
        assert det_ctx.cached_labelset is not None
        assert det_ctx.cached_labelset_mtime == _detector_file_mtime(_detector_path("resorted-live"))
        assert _find_label(client, detector_id)["results"][0]["id"] == RESORT_TARGET
        assert query_head_text(det_ctx.model) == RESORT_QUERY

    def test_re_saving_the_same_query_keeps_its_head(self, client, resort_world):
        from vtscore.detectors.query_head import query_head_text
        from vtscore.state.core import DetectorContext, register_detector_context

        corpus, _query, _resort = resort_world
        detector_id = _write_detector("resaved", corpus, goods=[1], bads=[40], text_query=QUERY)
        det_ctx = DetectorContext(detector_id, name="resaved", media_type="image")
        register_detector_context(det_ctx)
        _find_label(client, detector_id)
        head = det_ctx.model

        _set_examples(client, "resaved", [{"type": "text", "value": QUERY}])

        assert det_ctx.model is head and query_head_text(head) == QUERY
