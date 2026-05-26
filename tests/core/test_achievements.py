"""Tests for the Achievement system.

Covers:
- record_* helpers (vote, dataset load, detector import, find)
- counter → tier index mapping
- pending_announcements computation
- acknowledge tracking
- importer exclusion for datasets_loaded
- /api/achievements and /api/achievements/<id>/acknowledge routes
"""

from __future__ import annotations

import json
import pathlib

import app as app_module  # noqa: F401 - triggers conftest media init
from vtsearch import achievements
from vtsearch import settings as settings_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _by_id(state: dict, category: str) -> dict:
    for a in state["achievements"]:
        if a["id"] == category:
            return a
    raise KeyError(category)


def _pending_for(state: dict, category: str) -> list[dict]:
    return [p for p in state["pending_announcements"] if p["id"] == category]


# ---------------------------------------------------------------------------
# Empty / defaults
# ---------------------------------------------------------------------------


class TestEmptyState:
    def test_empty_state_when_no_history(self):
        state = achievements.get_full_state()
        assert state["tier_names"] == ["Bronze", "Silver", "Gold", "Platinum"]
        assert state["pending_announcements"] == []
        assert len(state["achievements"]) == 10
        for a in state["achievements"]:
            assert a["counter"] == 0
            assert a["tier_idx"] == -1
            assert a["next_threshold"] == a["tiers"][0]
        # docs field is always present, with read=False for every doc.
        assert {d["id"] for d in state["docs"]} == {
            "readme",
            "user_guide",
            "cli",
            "api",
        }
        assert all(d["read"] is False for d in state["docs"])

    def test_known_categories_present(self):
        ids = {a["id"] for a in achievements.get_full_state()["achievements"]}
        assert ids == {
            "datasets_loaded",
            "votes_cast",
            "detectors_trained",
            "detectors_imported",
            "find_media",
            "days_active",
            "media_types_touched",
            "vote_streak",
            "hours_voted",
            "docs_read",
        }


# ---------------------------------------------------------------------------
# record_vote
# ---------------------------------------------------------------------------


class TestRecordVote:
    def test_vote_increments_votes_cast(self):
        for _ in range(3):
            achievements.record_vote("det-1")
        state = achievements.get_full_state()
        votes = _by_id(state, "votes_cast")
        assert votes["counter"] == 3
        assert votes["tier_idx"] == -1  # 3 < 100

    def test_first_vote_credits_detector_trained(self):
        achievements.record_vote("det-1")
        state = achievements.get_full_state()
        assert _by_id(state, "detectors_trained")["counter"] == 1

    def test_subsequent_votes_same_detector_dont_double_count(self):
        for _ in range(5):
            achievements.record_vote("det-1")
        state = achievements.get_full_state()
        assert _by_id(state, "detectors_trained")["counter"] == 1
        assert _by_id(state, "votes_cast")["counter"] == 5

    def test_distinct_detectors_each_count_once(self):
        for did in ("a", "b", "c"):
            for _ in range(2):
                achievements.record_vote(did)
        state = achievements.get_full_state()
        assert _by_id(state, "detectors_trained")["counter"] == 3
        assert _by_id(state, "votes_cast")["counter"] == 6

    def test_empty_detector_id_does_not_credit(self):
        achievements.record_vote("")
        state = achievements.get_full_state()
        assert _by_id(state, "votes_cast")["counter"] == 1
        assert _by_id(state, "detectors_trained")["counter"] == 0


# ---------------------------------------------------------------------------
# record_dataset_load
# ---------------------------------------------------------------------------


class TestRecordDatasetLoad:
    def test_real_importer_counts(self):
        achievements.record_dataset_load("server_folder")
        state = achievements.get_full_state()
        ds = _by_id(state, "datasets_loaded")
        assert ds["counter"] == 1
        assert ds["tier_idx"] == 0  # Bronze threshold is 1

    def test_demo_does_not_count(self):
        achievements.record_dataset_load("demo")
        assert _by_id(achievements.get_full_state(), "datasets_loaded")["counter"] == 0

    def test_synthetic_does_not_count(self):
        achievements.record_dataset_load("synthetic")
        assert _by_id(achievements.get_full_state(), "datasets_loaded")["counter"] == 0

    def test_mixed_runs(self):
        for name in ["server_folder", "demo", "pickle", "synthetic", "http_archive", "server_files"]:
            achievements.record_dataset_load(name)
        # server_folder + pickle + http_archive + server_files = 4
        assert _by_id(achievements.get_full_state(), "datasets_loaded")["counter"] == 4


# ---------------------------------------------------------------------------
# record_detector_import / record_find
# ---------------------------------------------------------------------------


class TestImportAndFind:
    def test_import_dedupes_by_detector_id(self):
        for _ in range(3):
            achievements.record_detector_import("det-X")
        assert _by_id(achievements.get_full_state(), "detectors_imported")["counter"] == 1

    def test_import_distinct_detectors_count_separately(self):
        for did in ("a", "b", "c"):
            achievements.record_detector_import(did)
        assert _by_id(achievements.get_full_state(), "detectors_imported")["counter"] == 3

    def test_import_empty_id_is_noop(self):
        achievements.record_detector_import("")
        assert _by_id(achievements.get_full_state(), "detectors_imported")["counter"] == 0

    def test_find_accumulates(self):
        achievements.record_find(50)
        achievements.record_find(150)
        assert _by_id(achievements.get_full_state(), "find_media")["counter"] == 200

    def test_find_zero_or_negative_is_noop(self):
        achievements.record_find(0)
        achievements.record_find(-5)
        assert _by_id(achievements.get_full_state(), "find_media")["counter"] == 0


# ---------------------------------------------------------------------------
# Vote-driven achievements: days_active / hours_voted / media_types / streak
# ---------------------------------------------------------------------------


def _ts(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> float:
    """Build a UTC unix timestamp without pulling in test-only dependencies."""
    from datetime import datetime, timezone

    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc).timestamp()


class TestDaysActive:
    def test_first_vote_credits_first_day(self):
        achievements.record_vote("det", "audio", now=_ts(2026, 5, 13, 12))
        assert _by_id(achievements.get_full_state(), "days_active")["counter"] == 1

    def test_same_day_does_not_double_count(self):
        achievements.record_vote("det", "audio", now=_ts(2026, 5, 13, 8))
        achievements.record_vote("det", "audio", now=_ts(2026, 5, 13, 23, 59, 59))
        assert _by_id(achievements.get_full_state(), "days_active")["counter"] == 1

    def test_different_days_each_count_once(self):
        for day in (10, 11, 12):
            achievements.record_vote("det", "audio", now=_ts(2026, 5, day, 9))
        assert _by_id(achievements.get_full_state(), "days_active")["counter"] == 3

    def test_bronze_threshold_at_two_days(self):
        achievements.record_vote("det", "audio", now=_ts(2026, 5, 10))
        achievements.record_vote("det", "audio", now=_ts(2026, 5, 11))
        day_state = _by_id(achievements.get_full_state(), "days_active")
        assert day_state["tier_idx"] == 0
        assert day_state["next_threshold"] == 20


class TestMediaTypesTouched:
    def test_each_distinct_type_counts_once(self):
        for mt in ("audio", "image", "text", "audio"):
            achievements.record_vote("det", mt)
        assert _by_id(achievements.get_full_state(), "media_types_touched")["counter"] == 3

    def test_empty_media_type_does_not_credit(self):
        achievements.record_vote("det", "")
        assert _by_id(achievements.get_full_state(), "media_types_touched")["counter"] == 0

    def test_bronze_at_two_types(self):
        achievements.record_vote("det", "audio")
        achievements.record_vote("det", "image")
        mt_state = _by_id(achievements.get_full_state(), "media_types_touched")
        assert mt_state["tier_idx"] == 0
        assert mt_state["next_threshold"] == 3

    def test_platinum_at_all_five_types(self):
        for mt in ("audio", "image", "text", "video", "document"):
            achievements.record_vote("det", mt)
        mt_state = _by_id(achievements.get_full_state(), "media_types_touched")
        assert mt_state["counter"] == 5
        assert mt_state["tier_idx"] == 3  # Platinum
        assert mt_state["next_threshold"] is None


class TestHoursVoted:
    def test_first_vote_credits_one_hour(self):
        achievements.record_vote("det", "audio", now=_ts(2026, 5, 13, 14, 30))
        assert _by_id(achievements.get_full_state(), "hours_voted")["counter"] == 1

    def test_same_hour_across_days_dedupes(self):
        achievements.record_vote("det", "audio", now=_ts(2026, 5, 13, 14, 0))
        achievements.record_vote("det", "audio", now=_ts(2026, 5, 14, 14, 59))
        assert _by_id(achievements.get_full_state(), "hours_voted")["counter"] == 1

    def test_distinct_hours_count(self):
        for hour in (0, 5, 10, 15, 20):
            achievements.record_vote("det", "audio", now=_ts(2026, 5, 13, hour))
        assert _by_id(achievements.get_full_state(), "hours_voted")["counter"] == 5

    def test_around_the_clock_platinum_at_24_hours(self):
        for hour in range(24):
            achievements.record_vote("det", "audio", now=_ts(2026, 5, 13, hour))
        hv = _by_id(achievements.get_full_state(), "hours_voted")
        assert hv["counter"] == 24
        assert hv["tier_idx"] == 3  # Platinum at 24
        assert hv["next_threshold"] is None


class TestVoteStreak:
    def test_single_vote_streak_is_one(self):
        achievements.record_vote("det", "audio", now=_ts(2026, 5, 13, 12))
        assert _by_id(achievements.get_full_state(), "vote_streak")["counter"] == 1

    def test_gap_within_10min_extends_streak(self):
        base = _ts(2026, 5, 13, 12, 0, 0)
        achievements.record_vote("det", "audio", now=base)
        achievements.record_vote("det", "audio", now=base + 60)  # 1 min
        achievements.record_vote("det", "audio", now=base + 120)  # 2 min
        assert _by_id(achievements.get_full_state(), "vote_streak")["counter"] == 3

    def test_exactly_10min_gap_still_extends(self):
        # The spec is "at most a 10min gap", so gap == 600s continues the streak.
        base = _ts(2026, 5, 13, 12, 0, 0)
        achievements.record_vote("det", "audio", now=base)
        achievements.record_vote("det", "audio", now=base + 600)
        assert _by_id(achievements.get_full_state(), "vote_streak")["counter"] == 2

    def test_gap_over_10min_resets_current_but_preserves_max(self):
        base = _ts(2026, 5, 13, 12, 0, 0)
        # Run of 3 within 10-min gaps.
        achievements.record_vote("det", "audio", now=base)
        achievements.record_vote("det", "audio", now=base + 300)
        achievements.record_vote("det", "audio", now=base + 600)
        # 11-minute gap → resets the running streak to 1.
        achievements.record_vote("det", "audio", now=base + 600 + 660)
        achievements.record_vote("det", "audio", now=base + 600 + 660 + 60)
        streak = _by_id(achievements.get_full_state(), "vote_streak")
        assert streak["counter"] == 3  # max watermark, not current

    def test_streak_watermark_climbs_during_run(self):
        # Cross the Bronze threshold of 200 votes within tight gaps.
        ts = _ts(2026, 5, 13, 12, 0, 0)
        for _ in range(200):
            achievements.record_vote("det", "audio", now=ts)
            ts += 60  # 1-min gaps - well under 10 min
        streak = _by_id(achievements.get_full_state(), "vote_streak")
        assert streak["counter"] == 200
        assert streak["tier_idx"] == 0  # Bronze at 200

    def test_long_pause_then_new_streak_does_not_combine(self):
        # 50 in a row, hour-long break, 30 more - watermark stays at 50.
        ts = _ts(2026, 5, 13, 8, 0, 0)
        for _ in range(50):
            achievements.record_vote("det", "audio", now=ts)
            ts += 30
        ts += 60 * 60  # 1-hour gap
        for _ in range(30):
            achievements.record_vote("det", "audio", now=ts)
            ts += 30
        streak = _by_id(achievements.get_full_state(), "vote_streak")
        assert streak["counter"] == 50


class TestVoteStateRoundTrip:
    def test_new_state_keys_persisted(self, isolated_settings):
        achievements.record_vote("det", "audio", now=_ts(2026, 5, 13, 9))
        raw = json.loads(isolated_settings.read_text())
        state = raw["achievement_state"]
        assert state["days_seen"] == ["2026-05-13"]
        assert state["hours_seen"] == [9]
        assert state["media_types_seen"] == ["audio"]
        assert state["current_streak"] == 1
        assert state["counters"]["vote_streak"] == 1
        assert state["last_vote_ts"] > 0.0


# ---------------------------------------------------------------------------
# Readme Reader: record_doc_phrase + docs in get_full_state
# ---------------------------------------------------------------------------


class TestReadmeReader:
    """Behavioural tests for the docs_read achievement."""

    def _phrase_for(self, doc_id: str) -> str:
        """Lookup the canonical phrase for *doc_id* from the private list."""
        for d in achievements._DOCS_RAW:
            if d["id"] == doc_id:
                return d["phrase"]
        raise KeyError(doc_id)

    def test_correct_phrase_credits_doc(self):
        result = achievements.record_doc_phrase(self._phrase_for("readme"))
        assert result["matched"] is True
        assert result["doc_id"] == "readme"
        assert result["already_read"] is False
        state = achievements.get_full_state()
        assert _by_id(state, "docs_read")["counter"] == 1
        assert [d for d in state["docs"] if d["id"] == "readme"][0]["read"] is True

    def test_wrong_phrase_is_no_op(self):
        result = achievements.record_doc_phrase("definitely not a real phrase")
        assert result["matched"] is False
        assert result["doc_id"] is None
        state = achievements.get_full_state()
        assert _by_id(state, "docs_read")["counter"] == 0
        assert all(d["read"] is False for d in state["docs"])

    def test_case_insensitive_match(self):
        phrase = self._phrase_for("cli").upper()
        result = achievements.record_doc_phrase(phrase)
        assert result["matched"] is True
        assert result["doc_id"] == "cli"

    def test_whitespace_tolerant_match(self):
        phrase = "  " + self._phrase_for("api").replace(" ", "   ") + "\n"
        result = achievements.record_doc_phrase(phrase)
        assert result["matched"] is True
        assert result["doc_id"] == "api"

    def test_duplicate_credit_reports_already_read(self):
        phrase = self._phrase_for("user_guide")
        first = achievements.record_doc_phrase(phrase)
        assert first["already_read"] is False
        second = achievements.record_doc_phrase(phrase)
        assert second["matched"] is True
        assert second["already_read"] is True
        # Counter stays at 1, not 2.
        assert _by_id(achievements.get_full_state(), "docs_read")["counter"] == 1

    def test_all_four_docs_unlock_platinum(self):
        for doc_id in ("readme", "user_guide", "cli", "api"):
            achievements.record_doc_phrase(self._phrase_for(doc_id))
        state = achievements.get_full_state()
        docs_read = _by_id(state, "docs_read")
        assert docs_read["counter"] == 4
        assert docs_read["tier_idx"] == 3  # Platinum
        assert docs_read["next_threshold"] is None

    def test_state_round_trips_via_settings(self, isolated_settings):
        achievements.record_doc_phrase(self._phrase_for("readme"))
        achievements.record_doc_phrase(self._phrase_for("api"))
        raw = json.loads(isolated_settings.read_text())
        state = raw["achievement_state"]
        assert sorted(state["docs_read_ids"]) == ["api", "readme"]
        assert state["counters"]["docs_read"] == 2


class TestReadmeReaderDocsDriftGuard:
    """Each doc file must actually contain the phrase listed in the registry.

    This catches silent drift where someone edits one side (the doc or the
    DOCS_RAW table) without updating the other.
    """

    def test_each_doc_contains_its_phrase(self):
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        for doc in achievements._DOCS_RAW:
            text = (repo_root / doc["path"]).read_text(encoding="utf-8")
            assert achievements._normalize_phrase(doc["phrase"]) in achievements._normalize_phrase(text), (
                f"Doc {doc['path']} is missing its Readme Reader phrase "
                f"{doc['phrase']!r}; update either the doc footer or "
                f"_DOCS_RAW in vtsearch/achievements.py."
            )

    def test_no_two_docs_share_a_phrase(self):
        phrases = [achievements._normalize_phrase(d["phrase"]) for d in achievements._DOCS_RAW]
        assert len(phrases) == len(set(phrases))


# ---------------------------------------------------------------------------
# Readme Reader: API routes
# ---------------------------------------------------------------------------


class TestReadmeReaderApi:
    def _phrase_for(self, doc_id: str) -> str:
        for d in achievements._DOCS_RAW:
            if d["id"] == doc_id:
                return d["phrase"]
        raise KeyError(doc_id)

    def test_check_phrase_correct(self, client):
        resp = client.post(
            "/api/achievements/check-phrase",
            json={"phrase": self._phrase_for("readme")},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["matched"] is True
        assert body["doc_id"] == "readme"
        assert body["already_read"] is False

    def test_check_phrase_wrong(self, client):
        resp = client.post("/api/achievements/check-phrase", json={"phrase": "nope"})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["matched"] is False
        assert body["doc_id"] is None

    def test_check_phrase_missing_body(self, client):
        resp = client.post("/api/achievements/check-phrase", json={})
        # Schema validation: missing required "phrase" → 422.
        assert resp.status_code == 422

    def test_check_phrase_non_string(self, client):
        resp = client.post("/api/achievements/check-phrase", json={"phrase": 42})
        # Schema validation: "phrase" must be a string → 422.
        assert resp.status_code == 422

    def test_docs_raw_returns_markdown(self, client):
        resp = client.get("/api/achievements/docs/readme/raw")
        assert resp.status_code == 200
        assert resp.mimetype == "text/plain"
        body = resp.get_data(as_text=True)
        # Should contain the README's phrase footer.
        assert "all aboard the embedding express" in body.lower()

    def test_docs_raw_unknown_id(self, client):
        resp = client.get("/api/achievements/docs/not-a-doc/raw")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tier progression and announcements
# ---------------------------------------------------------------------------


class TestTiers:
    def test_bronze_at_first_threshold(self):
        # votes_cast: Bronze=100
        for _ in range(100):
            achievements.record_vote("d")
        state = achievements.get_full_state()
        votes = _by_id(state, "votes_cast")
        assert votes["tier_idx"] == 0
        assert votes["next_threshold"] == 1000

    def test_pending_announcements_for_unannounced_tiers(self):
        achievements.record_dataset_load("server_folder")  # Bronze
        state = achievements.get_full_state()
        pending = _pending_for(state, "datasets_loaded")
        assert len(pending) == 1
        assert pending[0]["tier_idx"] == 0
        assert pending[0]["tier_name"] == "Bronze"
        assert pending[0]["threshold"] == 1

    def test_acknowledge_clears_pending(self):
        achievements.record_dataset_load("server_folder")
        achievements.acknowledge("datasets_loaded", 0)
        state = achievements.get_full_state()
        assert _pending_for(state, "datasets_loaded") == []

    def test_acknowledge_lower_tier_is_noop(self):
        achievements.record_dataset_load("server_folder")
        achievements.acknowledge("datasets_loaded", 0)
        # Acknowledging the same tier again should not change anything.
        assert achievements.acknowledge("datasets_loaded", 0) is False

    def test_jumping_multiple_tiers_pends_each(self):
        # find_media: 200 / 2000 / 20000 / 200000
        achievements.record_find(2500)  # Bronze + Silver in one go
        state = achievements.get_full_state()
        pending = _pending_for(state, "find_media")
        assert [p["tier_idx"] for p in pending] == [0, 1]

    def test_invalid_category_acknowledge_returns_false(self):
        assert achievements.acknowledge("not_a_real_category", 0) is False

    def test_invalid_tier_idx_acknowledge_returns_false(self):
        assert achievements.acknowledge("datasets_loaded", 99) is False
        assert achievements.acknowledge("datasets_loaded", -1) is False


# ---------------------------------------------------------------------------
# Persistence in settings.json
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_state_round_trips_via_settings(self, isolated_settings):
        achievements.record_vote("det-1")
        achievements.record_find(50)
        raw = json.loads(isolated_settings.read_text())
        state = raw["achievement_state"]
        assert state["counters"]["votes_cast"] == 1
        assert state["counters"]["detectors_trained"] == 1
        assert state["counters"]["find_media"] == 50
        assert state["trained_detector_ids"] == ["det-1"]

    def test_load_reads_existing_state(self, isolated_settings):
        # Write directly and reload.
        isolated_settings.write_text(
            json.dumps(
                {
                    "achievement_state": {
                        "counters": {"votes_cast": 500},
                        "announced": {"votes_cast": 0},
                        "trained_detector_ids": [],
                        "imported_detector_ids": [],
                    }
                }
            )
        )
        settings_mod.reset()
        state = achievements.get_full_state()
        votes = _by_id(state, "votes_cast")
        assert votes["counter"] == 500
        # Bronze already announced - no pending for tier 0.
        pending = _pending_for(state, "votes_cast")
        assert [p["tier_idx"] for p in pending] == []


# ---------------------------------------------------------------------------
# Flask API routes
# ---------------------------------------------------------------------------


class TestAchievementsApi:
    def test_get_endpoint_returns_state(self, client):
        achievements.record_vote("d")
        resp = client.get("/api/achievements")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["tier_names"] == ["Bronze", "Silver", "Gold", "Platinum"]
        ids = {a["id"] for a in body["achievements"]}
        assert "votes_cast" in ids

    def test_acknowledge_endpoint(self, client):
        achievements.record_dataset_load("server_folder")
        resp = client.post(
            "/api/achievements/datasets_loaded/acknowledge",
            json={"tier_idx": 0},
        )
        assert resp.status_code == 200
        assert resp.get_json()["changed"] is True

        # Second call returns changed=False.
        resp = client.post(
            "/api/achievements/datasets_loaded/acknowledge",
            json={"tier_idx": 0},
        )
        assert resp.get_json()["changed"] is False

    def test_acknowledge_requires_tier_idx(self, client):
        resp = client.post("/api/achievements/votes_cast/acknowledge", json={})
        # Schema validation: missing required "tier_idx" → 422.
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Integration hooks via real action endpoints
# ---------------------------------------------------------------------------


class TestActionHooks:
    def test_vote_endpoint_increments_counter(self, client):
        # Find any media id from the test fixture.
        listing = client.get("/api/medias/ids").get_json()
        media_id = listing[0]["id"]
        resp = client.post(f"/api/medias/{media_id}/vote", json={"target": "good"})
        assert resp.status_code == 200
        state = achievements.get_full_state()
        assert _by_id(state, "votes_cast")["counter"] == 1

    def test_unvote_does_not_decrement(self, client):
        listing = client.get("/api/medias/ids").get_json()
        media_id = listing[0]["id"]
        client.post(f"/api/medias/{media_id}/vote", json={"target": "good"})
        client.post(f"/api/medias/{media_id}/vote", json={"target": "none"})  # un-vote
        # Counter should be 1: only the add counts, not the remove.
        assert _by_id(achievements.get_full_state(), "votes_cast")["counter"] == 1

    def test_idempotent_re_vote_does_not_increment(self, client):
        """H1 fix: sending target=good on an already-good media is idempotent
        and must not credit a second vote.  This is the achievement-counter
        side of the inflation race - two stale-view tabs each POSTing the
        same target collapse into one increment on the server."""
        listing = client.get("/api/medias/ids").get_json()
        media_id = listing[0]["id"]
        client.post(f"/api/medias/{media_id}/vote", json={"target": "good"})
        client.post(f"/api/medias/{media_id}/vote", json={"target": "good"})
        client.post(f"/api/medias/{media_id}/vote", json={"target": "good"})
        assert _by_id(achievements.get_full_state(), "votes_cast")["counter"] == 1

    def test_find_label_does_not_credit_votes_cast(self, client):
        """Find-mode auto-labels must not count toward votes_cast or vote_streak.

        The app applies labels to every media item in Find mode, but those are
        system-generated scores — the user did not cast them — so they must not
        inflate the Votes Cast or Marathoner achievements.
        """
        from helpers import setup_trainable_model_in_registry
        from vtsearch.state import snapshot_medias

        detector_id = setup_trainable_model_in_registry(
            "find-label-achievement-test",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )
        resp = client.post("/api/find-label", json={"detector_id": detector_id})
        assert resp.status_code == 200, resp.get_json()
        state = achievements.get_full_state()
        assert _by_id(state, "votes_cast")["counter"] == 0
        assert _by_id(state, "vote_streak")["counter"] == 0


# ---------------------------------------------------------------------------
# disable_achievements opt-out
# ---------------------------------------------------------------------------


class TestDisableAchievements:
    def test_record_hooks_are_noops_when_disabled(self):
        settings_mod.set_disable_achievements(True)
        achievements.record_vote("det-1", media_type="audio")
        achievements.record_dataset_load("server_folder")
        achievements.record_detector_import("det-X")
        achievements.record_find(500)
        result = achievements.record_doc_phrase("all aboard the embedding express")
        assert result["matched"] is False  # disabled blocks credit
        state = achievements.get_full_state()
        for a in state["achievements"]:
            assert a["counter"] == 0
            assert a["tier_idx"] == -1
        assert state["pending_announcements"] == []
        assert all(d["read"] is False for d in state["docs"])

    def test_get_full_state_zeroes_existing_counters_when_disabled(self):
        # Accrue real progress first.
        achievements.record_vote("det-1", media_type="audio")
        achievements.record_dataset_load("server_folder")
        achievements.record_find(500)
        # Flip the toggle through the same code path the route uses.
        settings_mod.set_disable_achievements(True)
        state = achievements.get_full_state()
        for a in state["achievements"]:
            assert a["counter"] == 0
            assert a["tier_idx"] == -1
        assert state["pending_announcements"] == []

    def test_wipe_state_clears_stored_counters(self):
        achievements.record_vote("det-1", media_type="audio")
        achievements.record_find(123)
        # Sanity check that something was stored.
        from vtsearch.auth import get_current_user
        from vtsearch.settings import _user_caches

        username = get_current_user()
        assert "achievement_state" in _user_caches.get(username, {})

        achievements.wipe_state()
        assert "achievement_state" not in _user_caches.get(username, {})

    def test_settings_route_wipes_on_false_to_true_transition(self, client):
        # Build up some real progress.
        achievements.record_vote("det-1", media_type="audio")
        achievements.record_find(50)
        # Flip via the public PUT endpoint.
        resp = client.put("/api/settings", json={"disable_achievements": True})
        assert resp.status_code == 200
        assert resp.get_json()["disable_achievements"] is True

        # State on disk is gone; the public read returns a zeroed shell.
        from vtsearch.auth import get_current_user
        from vtsearch.settings import _user_caches

        username = get_current_user()
        assert "achievement_state" not in _user_caches.get(username, {})

        state = achievements.get_full_state()
        for a in state["achievements"]:
            assert a["counter"] == 0

    def test_re_enabling_does_not_restore_old_counters(self, client):
        achievements.record_vote("det-1", media_type="audio")
        client.put("/api/settings", json={"disable_achievements": True})
        client.put("/api/settings", json={"disable_achievements": False})
        # Counters should still be zero - the wipe is permanent.
        state = achievements.get_full_state()
        for a in state["achievements"]:
            assert a["counter"] == 0
