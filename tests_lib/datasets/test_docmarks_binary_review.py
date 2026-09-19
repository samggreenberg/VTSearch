"""DocMarks review as VTSearch Good/Bad queues: votes map back to each audit's verdicts."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_DOCMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "docmarks"


@pytest.fixture(scope="module")
def br():
    sys.path.insert(0, str(_DOCMARKS))
    try:
        yield importlib.import_module("binary_review")
    finally:
        sys.path.remove(str(_DOCMARKS))


def _q(task, key, fn):
    return fn, {"task": task, "key": key}


class TestBankOnlyReadsDocmarks:
    def test_vg_scale_detectors_are_ignored(self, br):
        dets = [
            {"name": "docmarks pm_crest -- right logo same as left?"},
            {"name": "bottle incl jars -- seat check"},
            {"name": "docmarksish not ours"},
        ]
        assert [d["name"] for d in br.docmarks_detectors(dets)] == ["docmarks pm_crest -- right logo same as left?"]

    def test_queue_names_must_carry_the_prefix(self, br, tmp_path):
        with pytest.raises(ValueError):
            br.emit([], tmp_path, "pm_crest -- same?", pages={}, corpus=tmp_path)

    def test_a_file_voted_both_ways_is_refused(self, br):
        with pytest.raises(ValueError):
            br.votes_from_labels({"good": [{"filename": "a.jpg"}], "bad": [{"filename": "a.jpg"}]})


class TestUcsf:
    ROWS = [
        {
            "task": "ucsf_classes",
            "proposal": "p",
            "sheet": "p__a_00.png",
            "cells": [{"index": 0}, {"index": 1}, {"index": 2}],
            "verdict": "",
        },
        {
            "task": "ucsf_classes",
            "proposal": "p",
            "sheet": "p__b_00.png",
            "cells": [{"index": 0}, {"index": 1}],
            "verdict": "",
        },
        {"task": "ucsf_classes_relation", "proposal": "p", "relation": ""},
    ]
    QS = dict(
        [
            _q("ucsf_classes", {"sheet": "p__a_00.png", "cell": 0}, "a0"),
            _q("ucsf_classes", {"sheet": "p__a_00.png", "cell": 1}, "a1"),
            _q("ucsf_classes", {"sheet": "p__a_00.png", "cell": 2}, "a2"),
            _q("ucsf_classes", {"sheet": "p__b_00.png", "cell": 0}, "b0"),
            _q("ucsf_classes", {"sheet": "p__b_00.png", "cell": 1}, "b1"),
            _q("ucsf_classes_relation", {"proposal": "p", "target": "tobacco800/logo_x"}, "rel"),
        ]
    )

    def test_answered_sheets_become_cell_lists_and_partial_ones_stay_blank(self, br):
        votes = {"a0": "good", "a1": "bad", "a2": "good", "b0": "good", "rel": "good"}
        rows, unanswered = br.translate_ucsf(self.ROWS, self.QS, votes)
        assert rows[0]["verdict"] == "0,2"
        assert rows[1]["verdict"] == "" and any("p__b_00" in u for u in unanswered)
        assert rows[2]["relation"] == "extends tobacco800/logo_x"
        assert self.ROWS[0]["verdict"] == ""  # originals untouched

    def test_all_none_and_new(self, br):
        votes = {"a0": "good", "a1": "good", "a2": "good", "b0": "bad", "b1": "bad", "rel": "bad"}
        rows, unanswered = br.translate_ucsf(self.ROWS, self.QS, votes)
        assert [r.get("verdict") for r in rows[:2]] == ["all", "none"] and rows[2]["relation"] == "new"
        assert unanswered == []


class TestQueryCropsAndBoxes:
    def test_query_crops_keep_rank_order_capped(self, br):
        qs = dict(_q("query_crops", {"class_id": "c", "index": i}, f"q{i}") for i in range(6))
        votes = {f"q{i}": ("good" if i != 1 else "bad") for i in range(6)}
        rows, unanswered = br.translate_query_crops([{"class_id": "c", "verdict": ""}], qs, votes, cap=3)
        assert rows[0]["verdict"] == "0,2,3" and unanswered == []

    def test_query_crops_unanswered_is_reported_not_guessed(self, br):
        qs = dict(_q("query_crops", {"class_id": "c", "index": i}, f"q{i}") for i in range(2))
        rows, unanswered = br.translate_query_crops([{"class_id": "c", "verdict": ""}], qs, {"q0": "good"})
        assert rows[0]["verdict"] == "" and unanswered

    def test_boxes_only_ask_about_changed_members(self, br):
        assert br.box_questionable({"flags": [], "new_box": [1, 2, 3, 4]})
        assert not br.box_questionable({"flags": ["unchanged"], "new_box": [1, 2, 3, 4]})
        assert not br.box_questionable({"flags": ["no fit"], "new_box": None})
        qs = dict(
            [
                _q("box_tighten", {"class_id": "c", "index": 5}, "m5"),
                _q("box_tighten", {"class_id": "c", "index": 9}, "m9"),
            ]
        )
        rows, _ = br.translate_box_tighten([{"class_id": "c", "verdict": ""}], qs, {"m5": "good", "m9": "bad"})
        assert rows[0]["verdict"] == "5"


class TestRendering:
    def test_region_and_scale(self, br):
        assert br.expand((100, 100, 40, 20), None, 1000, 1000) == (90, 90, 150, 130)
        assert br.expand((0, 0, 40, 20), (0, 0, 80, 20), 60, 1000) == (0, 0, 60, 40)
        # a 50 px mark is enlarged toward 400 px but never past the panel
        assert br.panel_scale((0, 0, 60, 60), 50, (840, 580)) == 8.0
        assert br.panel_scale((0, 0, 600, 600), 500, (840, 580)) == pytest.approx(580 / 600)

    def test_class_name_leads_the_queue_name(self, br):
        assert br.short_class("tobacco800/logo_ajj10e00_1") == "t800 logo_ajj10e00_1"


class TestCompleteness2:
    def test_tile_box_goods_are_held_back_for_a_tight_box(self, br):
        qs = {
            "c0": {"task": "completeness2", "key": {"class_id": "c", "index": 0, "tile_box": False}},
            "c1": {"task": "completeness2", "key": {"class_id": "c", "index": 1, "tile_box": True}},
            "c2": {"task": "completeness2", "key": {"class_id": "c", "index": 2, "tile_box": False}},
        }
        rows, notes = br.translate_completeness2(
            [{"class_id": "c", "verdict": "none"}], qs, {"c0": "good", "c1": "good", "c2": "bad"}
        )
        assert rows[0]["verdict"] == "0" and rows[0]["needs_tight_box"] == [1]
        assert any("tile box" in n for n in notes)

    def test_sig_only_candidates_are_tiles(self, br):
        assert br.tile_box({"methods": "SIG"})
        assert not br.tile_box({"methods": "OCR+SIG"})


def _archive(root: Path, queue: str, name: str, votes: dict) -> None:
    """A ``votes_<date>.json`` beside a queue, in the shape ``bank`` writes."""
    import json

    d = root / queue
    d.mkdir(parents=True, exist_ok=True)
    labels = {"good": [], "bad": []}
    for fn, label in votes.items():
        labels[label].append({"filename": fn, "label": label})
    (d / "votes_2026-09-19.json").write_text(
        json.dumps({"detector": {"name": name}, "labels": labels}), encoding="utf-8"
    )


def _cleared(root: Path, name: str, votes: dict) -> None:
    """A ``.cleared`` detector backup, in the shape the app exports."""
    import json

    d = root / "detectors-cleared-20260919"
    d.mkdir(parents=True, exist_ok=True)
    labels = [{"filename": fn, "label": label} for fn, label in votes.items()]
    (d / f"{name.replace(' ', '_')}.json.cleared").write_text(
        json.dumps({"name": name, "labelset": {"labels": labels}}), encoding="utf-8"
    )


class TestBankMergesEveryVoteSource:
    """#3964: banking from the app alone emptied the verdicts of every CLEARED queue."""

    def test_a_cleared_queue_survives_a_later_bank(self, br, tmp_path):
        root, keep = tmp_path / "binary", tmp_path / "keep"
        name = "docmarks pm_crest -- right logo same as left?"
        # banked, then cleared off the dashboard: the app no longer holds it
        _archive(root, "docmarks_pm_crest", name, {"a.jpg": "good", "b.jpg": "bad"})
        _cleared(keep, name, {"a.jpg": "good", "b.jpg": "bad"})
        votes, sources = br.collect_votes(None, root, keep)
        assert votes[name] == {"a.jpg": "good", "b.jpg": "bad"}
        assert sources == {"archive": 1, "cleared backup": 1}

    def test_the_live_app_wins_where_sources_disagree(self, br, tmp_path, monkeypatch):
        root, keep = tmp_path / "binary", tmp_path / "keep"
        name = "docmarks rjr_block -- right logo same as left?"
        _cleared(keep, name, {"a.jpg": "bad"})
        monkeypatch.setattr(br, "docmarks_detectors", lambda _d: [{"name": name}])
        monkeypatch.setattr(
            br,
            "api",
            lambda base, path, *a, **k: (
                {"good": [{"filename": "a.jpg"}], "bad": []} if "labels-detail" in path else {"detectors": []}
            ),
        )
        votes, _ = br.collect_votes("http://app", root, keep)
        assert votes[name]["a.jpg"] == "good"  # the later vote, not the backup's

    def test_a_backup_without_labels_is_not_a_queue(self, br, tmp_path):
        keep = tmp_path / "keep"
        _cleared(keep, "docmarks empty -- same?", {})
        votes, sources = br.collect_votes(None, tmp_path / "binary", keep)
        assert votes == {} and sources == {}


class TestGuardWrite:
    """The same shape as ``dropped_rows``: the thing overwritten is human work."""

    ANSWERED = [{"verdict_source": "vtsearch", "verdict": "all"}, {"verdict_source": "vtsearch", "verdict": "none"}]

    def _dest(self, tmp_path, rows):
        import json

        d = tmp_path / "verdicts.from_vtsearch.jsonl"
        d.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return d

    def test_a_shrinking_write_is_refused_and_names_the_backups(self, br, tmp_path):
        dest = self._dest(tmp_path, self.ANSWERED)
        with pytest.raises(SystemExit) as exc:
            br.guard_write(dest, [{"verdict": ""}], tmp_path / "keep", allow_loss=False)
        assert "answers 2 item(s) and this run answers 0" in str(exc.value)
        assert ".cleared" in str(exc.value)

    def test_allow_loss_is_the_way_past_it(self, br, tmp_path):
        dest = self._dest(tmp_path, self.ANSWERED)
        br.guard_write(dest, [{"verdict": ""}], tmp_path / "keep", allow_loss=True)

    def test_growing_and_equal_writes_pass(self, br, tmp_path):
        dest = self._dest(tmp_path, self.ANSWERED)
        br.guard_write(dest, self.ANSWERED + [{"verdict_source": "vtsearch"}], tmp_path / "keep", allow_loss=False)
        br.guard_write(dest, self.ANSWERED, tmp_path / "keep", allow_loss=False)

    def test_a_first_write_has_nothing_to_lose(self, br, tmp_path):
        br.guard_write(tmp_path / "missing.jsonl", [], tmp_path / "keep", allow_loss=False)


class TestArchiveVotes:
    def test_an_archive_is_never_shrunk(self, br, tmp_path):
        import json

        q = tmp_path / "docmarks_pm_crest"
        q.mkdir()
        big = {"detector": {"name": "x"}, "labels": {"good": [{"filename": f"{i}.jpg"} for i in range(5)], "bad": []}}
        first = br.archive_votes(q, "2026-09-19", big)
        second = br.archive_votes(q, "2026-09-19", {"detector": {"name": "x"}, "labels": {"good": [], "bad": []}})
        assert first.name == "votes_2026-09-19.json"
        assert second != first  # the fuller archive is still there
        assert len(json.loads(first.read_text())["labels"]["good"]) == 5
