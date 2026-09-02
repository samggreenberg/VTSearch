"""API tests for the windowed sort response + ``/api/sort/page`` (S3/S17/S19).

These assert the *additive* backend contract: every sort route still returns the
full ``results`` list, but now also carries ``sort_token`` / ``total`` /
``above_threshold``, and the token pages a cached window back out.  The frontend
still consumes the full ``results`` (windowed model lands separately).
"""

from __future__ import annotations

import vtscore.state.sort_results_cache as sort_cache
from tests.fixtures.medias import NUM_MEDIAS


class TestSortResponseWindowMeta:
    def test_text_sort_carries_window_meta(self, client):
        resp = client.post("/api/sort", json={"text": "high pitched beep"})
        assert resp.status_code == 200
        data = resp.get_json()
        # Full list still returned (additive change).
        assert len(data["results"]) == NUM_MEDIAS
        assert data["total"] == NUM_MEDIAS
        assert isinstance(data["sort_token"], str) and data["sort_token"]
        # above_threshold matches how many rows are >= the returned threshold.
        expected = sum(1 for r in data["results"] if r["similarity"] >= data["threshold"])
        assert data["above_threshold"] == expected

    def test_each_sort_mints_a_fresh_token(self, client):
        t1 = client.post("/api/sort", json={"text": "beep"}).get_json()["sort_token"]
        t2 = client.post("/api/sort", json={"text": "beep"}).get_json()["sort_token"]
        assert t1 != t2

    def test_below_threshold_transmits_full_list(self, client, monkeypatch):
        # NUM_MEDIAS (20) < a high threshold → full list, no windowing.
        monkeypatch.setattr(sort_cache, "SORT_WINDOW_THRESHOLD", 1000)
        data = client.post("/api/sort", json={"text": "beep"}).get_json()
        assert len(data["results"]) == NUM_MEDIAS
        assert data["has_more_below"] is False


class TestWindowedTransmission:
    """At/above the (test-lowered) threshold, only a head window is transmitted."""

    def _shrink_window(self, monkeypatch):
        monkeypatch.setattr(sort_cache, "SORT_WINDOW_THRESHOLD", 3)
        monkeypatch.setattr(sort_cache, "SORT_WINDOW_HEAD", 2)
        monkeypatch.setattr(sort_cache, "SORT_WINDOW_TAIL", 1)

    def test_windows_the_transmitted_results(self, client, monkeypatch):
        self._shrink_window(monkeypatch)
        data = client.post("/api/sort", json={"text": "beep"}).get_json()
        above = data["above_threshold"]
        expected = min(NUM_MEDIAS, min(above, 2) + 1)
        assert data["total"] == NUM_MEDIAS
        assert len(data["results"]) == expected
        assert data["has_more_below"] is (expected < NUM_MEDIAS)

    def test_paging_reconstructs_the_full_ranking(self, client, monkeypatch):
        self._shrink_window(monkeypatch)
        data = client.post("/api/sort", json={"text": "sine wave"}).get_json()
        token = data["sort_token"]
        collected = [r["id"] for r in data["results"]]
        assert data["has_more_below"] is True  # 20 items, tiny window

        offset = len(collected)
        while True:
            body = client.get(f"/api/sort/page?token={token}&offset={offset}&limit=5").get_json()
            collected.extend(r["id"] for r in body["results"])
            if not body["has_more"]:
                break
            offset += 5
        assert len(collected) == NUM_MEDIAS
        assert len(set(collected)) == NUM_MEDIAS


class TestSortPage:
    def test_page_returns_a_window(self, client):
        sort = client.post("/api/sort", json={"text": "a beeping sound"}).get_json()
        token = sort["sort_token"]
        full_ids = [r["id"] for r in sort["results"]]

        page = client.get(f"/api/sort/page?token={token}&offset=1&limit=2")
        assert page.status_code == 200
        body = page.get_json()
        assert [r["id"] for r in body["results"]] == full_ids[1:3]
        assert body["offset"] == 1
        assert body["limit"] == 2
        assert body["total"] == NUM_MEDIAS
        assert body["has_more"] is (3 < NUM_MEDIAS)

    def test_page_default_offset_and_limit(self, client):
        token = client.post("/api/sort", json={"text": "tone"}).get_json()["sort_token"]
        body = client.get(f"/api/sort/page?token={token}").get_json()
        assert body["offset"] == 0
        assert body["limit"] == 200

    def test_unknown_token_404s(self, client):
        resp = client.get("/api/sort/page?token=nope-not-a-real-token")
        assert resp.status_code == 404

    def test_missing_token_422s(self, client):
        resp = client.get("/api/sort/page?offset=0&limit=10")
        assert resp.status_code == 422

    def test_limit_out_of_range_422s(self, client):
        token = client.post("/api/sort", json={"text": "tone"}).get_json()["sort_token"]
        assert client.get(f"/api/sort/page?token={token}&limit=0").status_code == 422
        assert client.get(f"/api/sort/page?token={token}&limit=99999").status_code == 422

    def test_full_pages_reconstruct_the_ranking(self, client):
        sort = client.post("/api/sort", json={"text": "sine wave"}).get_json()
        token = sort["sort_token"]
        expected = [r["id"] for r in sort["results"]]

        collected: list[int] = []
        offset = 0
        limit = 2
        while True:
            body = client.get(f"/api/sort/page?token={token}&offset={offset}&limit={limit}").get_json()
            collected.extend(r["id"] for r in body["results"])
            if not body["has_more"]:
                break
            offset += limit
        assert collected == expected
