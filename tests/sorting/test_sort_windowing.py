"""API tests for the windowed sort response + ``/api/sort/page`` (S3/S17/S19).

These assert the *additive* backend contract: every sort route still returns the
full ``results`` list, but now also carries ``sort_token`` / ``total`` /
``above_threshold``, and the token pages a cached window back out.  The frontend
still consumes the full ``results`` (windowed model lands separately).
"""

from __future__ import annotations

import app as app_module


class TestSortResponseWindowMeta:
    def test_text_sort_carries_window_meta(self, client):
        resp = client.post("/api/sort", json={"text": "high pitched beep"})
        assert resp.status_code == 200
        data = resp.get_json()
        # Full list still returned (additive change).
        assert len(data["results"]) == app_module.NUM_MEDIAS
        assert data["total"] == app_module.NUM_MEDIAS
        assert isinstance(data["sort_token"], str) and data["sort_token"]
        # above_threshold matches how many rows are >= the returned threshold.
        expected = sum(1 for r in data["results"] if r["similarity"] >= data["threshold"])
        assert data["above_threshold"] == expected

    def test_each_sort_mints_a_fresh_token(self, client):
        t1 = client.post("/api/sort", json={"text": "beep"}).get_json()["sort_token"]
        t2 = client.post("/api/sort", json={"text": "beep"}).get_json()["sort_token"]
        assert t1 != t2


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
        assert body["total"] == app_module.NUM_MEDIAS
        assert body["has_more"] is (3 < app_module.NUM_MEDIAS)

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
