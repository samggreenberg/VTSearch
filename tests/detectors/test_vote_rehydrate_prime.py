"""A vote's own labelset rewrite must not make the next requests re-read it (#3853).

Every vote rewrites the detector file and re-stamps ``cached_labelset_mtime``.
The rehydrate pre-check in ``before_request`` compares that stamp against a
TTL-cached stat, and until the writer also primed that cache every request in
the following second (the SPA fires about ten) parsed the whole detector JSON
to find, under the lock, that it was already fresh - about twelve spurious
parses per vote on a live session.
"""

from __future__ import annotations

import logging

from tests import load_detector_and_wait
from tests.helpers import setup_trainable_model_in_registry
from vtscore.detectors.dataset_sync import _detector_file_mtime_cached, reset_mtime_cache_for_tests
from vtscore.detectors.store import _detector_path
from vtscore.state.core import get_active_detector_context
from vtsearch.state import snapshot_medias

DETECTOR = "prime-after-vote"


def test_fanout_after_a_vote_does_not_rehydrate(client, caplog):
    detector_id = setup_trainable_model_in_registry(
        DETECTOR, good_ids=[1, 2, 3], bad_ids=[18, 19, 20], snap=snapshot_medias()
    )
    load_detector_and_wait(client, detector_id)
    reset_mtime_cache_for_tests()
    # One request settles the post-load state before the vote under test.
    assert client.get("/api/votes").status_code == 200

    assert client.post("/api/medias/4/vote", json={"target": "good"}).status_code == 200

    with caplog.at_level(logging.INFO, logger="vtscore.detectors.dataset_sync"):
        for _ in range(3):
            assert client.get("/api/votes").status_code == 200
            assert client.get("/api/labeling-status").status_code == 200
    rehydrates = [r.getMessage() for r in caplog.records if "rehydrating votes" in r.getMessage()]
    assert rehydrates == [], rehydrates

    ctx = get_active_detector_context()
    path = _detector_path(DETECTOR)
    assert ctx.cached_labelset_mtime == path.stat().st_mtime
    assert _detector_file_mtime_cached(path) == ctx.cached_labelset_mtime
