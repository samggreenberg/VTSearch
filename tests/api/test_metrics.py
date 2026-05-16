"""Tests for the Prometheus ``/metrics`` endpoint and the metrics module.

Covers the four signals called out in the 12.16 roadmap entry — vote
count, embedding latency, training time, RAM by dataset — plus the
exposition format contract and the no-throw invariants of the recording
helpers.
"""

from __future__ import annotations

import numpy as np

from vtsearch import metrics
from vtsearch.state.core import DatasetContext, _contexts, _state_lock


def _scrape(client) -> str:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.mimetype == "text/plain"
    assert "version=0.0.4" in response.headers["Content-Type"]
    return response.get_data(as_text=True)


class TestMetricsEndpoint:
    def test_metrics_endpoint_exposes_known_series(self, client):
        body = _scrape(client)

        # Every metric this module owns must be present in the output —
        # even before any work has happened — so dashboards can rely on
        # them appearing in the registry the moment the app is up.
        assert "vtsearch_votes_total" in body
        assert "vtsearch_embedding_seconds" in body
        assert "vtsearch_training_seconds" in body
        assert "vtsearch_dataset_memory_bytes" in body
        assert "vtsearch_datasets_loaded" in body
        assert "vtsearch_detectors_loaded" in body

    def test_help_lines_are_present(self, client):
        body = _scrape(client)
        assert "# HELP vtsearch_votes_total" in body
        assert "# TYPE vtsearch_votes_total counter" in body
        assert "# HELP vtsearch_embedding_seconds" in body
        assert "# TYPE vtsearch_embedding_seconds histogram" in body
        assert "# TYPE vtsearch_training_seconds histogram" in body
        assert "# TYPE vtsearch_dataset_memory_bytes gauge" in body


class TestVoteCounter:
    def test_record_vote_event_increments_counter(self, client):
        before = _scrape(client)
        metrics.record_vote_event("good", media_type="audio")
        metrics.record_vote_event("good", media_type="audio")
        metrics.record_vote_event("bad", media_type="audio")
        after = _scrape(client)

        good_line = 'vtsearch_votes_total{media_type="audio",vote="good"}'
        bad_line = 'vtsearch_votes_total{media_type="audio",vote="bad"}'

        before_good = _sample_value(before, good_line)
        after_good = _sample_value(after, good_line)
        after_bad = _sample_value(after, bad_line)

        assert after_good - before_good == 2.0
        assert after_bad >= 1.0

    def test_toggle_vote_increments_counter(self, client):
        from vtsearch.state.votes import toggle_vote
        from vtsearch.state import medias

        # Pick any media id from the pre-generated test dataset.
        media_id = next(iter(medias))

        before = _sample_value(_scrape(client), "vtsearch_votes_total")
        toggle_vote(media_id, "good")
        toggle_vote(media_id, "good")  # toggle off → records "unlabel"
        after_body = _scrape(client)

        # Both events must show up: one good + one unlabel.
        assert _sample_value(after_body, "vtsearch_votes_total") - before >= 2.0
        assert 'vote="unlabel"' in after_body

    def test_missing_labels_are_replaced_with_unknown(self, client):
        metrics.record_vote_event("", media_type="")
        body = _scrape(client)
        assert 'vote="unknown"' in body
        assert 'media_type="unknown"' in body


class TestEmbeddingLatency:
    def test_time_embedding_records_observation(self, client):
        with metrics.time_embedding("test_embedder", "audio"):
            pass
        body = _scrape(client)
        count_line = 'vtsearch_embedding_seconds_count{embedder="test_embedder",media_type="audio"}'
        assert _sample_value(body, count_line) >= 1.0

    def test_embedder_embed_media_records_latency(self, client):
        """MediaEmbedder.embed_media wraps each call with time_embedding."""
        from vtsearch.media import embedders_for_type
        from vtsearch.state import medias

        audio_emb = embedders_for_type("audio")[0]
        media = next(iter(medias.values()))

        before = _sample_value(
            _scrape(client),
            f'vtsearch_embedding_seconds_count{{embedder="{audio_emb.name}",media_type="audio"}}',
        )
        audio_emb.embed_media(media)
        after = _sample_value(
            _scrape(client),
            f'vtsearch_embedding_seconds_count{{embedder="{audio_emb.name}",media_type="audio"}}',
        )
        assert after - before == 1.0


class TestTrainingTime:
    def test_time_training_records_observation(self, client):
        with metrics.time_training("in_memory"):
            pass
        body = _scrape(client)
        assert _sample_value(body, 'vtsearch_training_seconds_count{kind="in_memory"}') >= 1.0

    def test_train_and_score_records_observation(self, client):
        """The training entry point must increment the histogram on a real fit."""
        from vtsearch.detectors.training import train_and_score
        from vtsearch.state import medias

        media_ids = list(medias.keys())
        assert len(media_ids) >= 4, "Test dataset must have enough medias to train"
        good = {media_ids[0]: None, media_ids[1]: None}
        bad = {media_ids[2]: None, media_ids[3]: None}

        before = _sample_value(_scrape(client), 'vtsearch_training_seconds_count{kind="in_memory"}')
        results, threshold, model = train_and_score(dict(medias), good, bad)
        assert model is not None  # training must have actually happened
        after = _sample_value(_scrape(client), 'vtsearch_training_seconds_count{kind="in_memory"}')
        assert after - before == 1.0


class TestDatasetMemoryGauge:
    def test_loaded_dataset_appears_with_bytes(self, client):
        ctx = DatasetContext("metrics_test_ds")
        ctx.dataset_display_name = "Metrics Test"
        # Seed with a tiny medias dict carrying a real ndarray so the
        # collector has an embedding-size signal to read.
        emb = np.zeros(8, dtype=np.float32)
        ctx.medias[1] = {"embedding": emb}

        with _state_lock:
            _contexts["metrics_test_ds"] = ctx
        try:
            body = _scrape(client)
            assert "metrics_test_ds" in body
            # Either the matrix or the per-media estimate must produce a
            # positive byte count for the seeded context.
            assert (
                _sample_value(
                    body,
                    'vtsearch_dataset_memory_bytes{dataset_id="metrics_test_ds",name="Metrics Test"}',
                )
                > 0.0
            )
            assert _sample_value(body, "vtsearch_datasets_loaded") >= 1.0
        finally:
            with _state_lock:
                _contexts.pop("metrics_test_ds", None)

    def test_unregistered_dataset_disappears_from_output(self, client):
        body = _scrape(client)
        assert "metrics_test_ds" not in body


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_value(body: str, line_prefix: str) -> float:
    """Return the numeric sample for the first metric line starting with *prefix*.

    Returns 0.0 if no matching line is found — useful for "before" reads
    where the counter may not yet have been initialised. Comment lines
    (``# HELP`` / ``# TYPE``) are ignored.
    """
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(line_prefix):
            # The exposition format is "<series> <value> [<timestamp>]"
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                try:
                    return float(parts[1])
                except ValueError:
                    return 0.0
    return 0.0
