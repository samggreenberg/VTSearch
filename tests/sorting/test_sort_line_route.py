"""``GET /api/sort/line``: a text or example sort's line, redrawn at the balance now set (#4760).

A typed query's display line moves with the balance (#4603), and an example
sort's does too (#4732), but neither ranking does.  So the client hands back the
``sort_token`` of the sort on screen after a balance change and the server
redraws the line over the ranking it already holds.  That the redraw is the
sort's own rule, bit for bit, is pinned in ``tests_lib/sorting/test_sort_line.py``;
these pin the route: it reads the balance now set, counts over the whole
ranking, answers only for a sort with a line to redraw, and every example route
registers one.
"""

from __future__ import annotations

import pytest

from tests.fixtures.medias import NUM_MEDIAS
from vtscore.state.sort_results_cache import sort_results_cache
from vtscore.training.query_sort import SortLine
from vtsearch.state import medias, set_beta


@pytest.fixture
def tenth_of_the_balance(monkeypatch):
    """Every sort line sits at a tenth of the balance, so a redraw shows which balance it read."""
    monkeypatch.setattr(SortLine, "threshold_at", lambda self, beta: round(beta / 10, 4))


def _above(results: list[dict], threshold: float) -> int:
    return sum(1 for r in results if r["similarity"] >= threshold)


class TestRedrawingTheLine:
    def test_a_text_sort_is_redrawn_at_the_balance_now_set(self, client, tenth_of_the_balance):
        sort = client.post("/api/sort", json={"text": "high pitched beep"}).get_json()
        set_beta(4.0)
        resp = client.get(f"/api/sort/line?token={sort['sort_token']}")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body == {"threshold": 0.4, "above_threshold": _above(sort["results"], 0.4), "total": NUM_MEDIAS}

    def test_the_redraw_is_the_sort_rerun_at_the_new_balance(self, client):
        before = client.post("/api/sort", json={"text": "high pitched beep"}).get_json()
        set_beta(0.25)
        line = client.get(f"/api/sort/line?token={before['sort_token']}").get_json()
        after = client.post("/api/sort", json={"text": "high pitched beep"}).get_json()
        assert line["threshold"] == after["threshold"]
        assert line["above_threshold"] == after["above_threshold"]
        # Only the line is redrawn: the acquisition cut is not the route's to move.
        assert "acq_threshold" not in line
        assert after["acq_threshold"] == before["acq_threshold"]

    def test_a_page_reports_the_redrawn_line(self, client, tenth_of_the_balance):
        token = client.post("/api/sort", json={"text": "tone"}).get_json()["sort_token"]
        set_beta(2.0)
        client.get(f"/api/sort/line?token={token}")
        assert client.get(f"/api/sort/page?token={token}").get_json()["threshold"] == 0.2


class TestEveryExampleSortRegistersItsLine:
    """The three server-media example routes send the whole ranking, but register it like the rest."""

    def test_example_sort_by_id(self, client, tenth_of_the_balance):
        media_id = next(iter(medias.keys()))
        sort = client.post("/api/example-sort-by-id", json={"media_id": media_id}).get_json()
        assert len(sort["results"]) == sort["total"] == NUM_MEDIAS
        assert sort["has_more_below"] is False
        assert sort["above_threshold"] == _above(sort["results"], sort["threshold"])
        set_beta(0.5)
        body = client.get(f"/api/sort/line?token={sort['sort_token']}").get_json()
        assert body["threshold"] == 0.05

    def test_the_whole_ranking_is_sent_at_any_size(self, client, monkeypatch):
        import vtscore.state.sort_results_cache as sort_cache

        monkeypatch.setattr(sort_cache, "SORT_WINDOW_THRESHOLD", 3)
        media_id = next(iter(medias.keys()))
        sort = client.post("/api/example-sort-by-id", json={"media_id": media_id}).get_json()
        assert len(sort["results"]) == NUM_MEDIAS
        assert sort["has_more_below"] is False


class TestNoLineToRedraw:
    def test_unknown_token_404s(self, client):
        assert client.get("/api/sort/line?token=nope-not-a-real-token").status_code == 404

    def test_missing_token_422s(self, client):
        assert client.get("/api/sort/line").status_code == 422

    def test_a_sort_stored_without_a_line_404s(self, client):
        # A learned or label-file sort: registered for paging, with no line here to redraw.
        from vtscore.state.core import get_active_context

        dataset_id = getattr(get_active_context(), "dataset_id", "") or ""
        token = sort_results_cache.store([{"id": 1, "score": 0.9}], 0.5, dataset_id=dataset_id)
        assert client.get(f"/api/sort/line?token={token}").status_code == 404
