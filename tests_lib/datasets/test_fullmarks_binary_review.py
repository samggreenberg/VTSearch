"""FullMarks review as VTSearch Good/Bad queues: votes map back to each audit's verdicts."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

_FULLMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "fullmarks"


@pytest.fixture(scope="module")
def br():
    sys.path.insert(0, str(_FULLMARKS))
    try:
        yield importlib.import_module("binary_review")
    finally:
        sys.path.remove(str(_FULLMARKS))


def _q(task, key, fn):
    return fn, {"task": task, "key": key}


class TestBankOnlyReadsFullmarks:
    def test_vg_scale_detectors_are_ignored(self, br):
        dets = [
            {"name": "fullmarks pm_crest -- right logo same as left?"},
            {"name": "bottle incl jars -- seat check"},
            {"name": "fullmarksish not ours"},
        ]
        assert [d["name"] for d in br.fullmarks_detectors(dets)] == ["fullmarks pm_crest -- right logo same as left?"]

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


class TestSheetPresentation:
    """A review sheet is fetched whole for every vote, so how it is drawn is not cosmetic.

    Two failures this covers, both found the hard way on 2026-09-20:

    * a portrait page letterboxed into the landscape default rendered ~448 px
      wide, where a letterhead crest is ~35 px and cannot be ruled in or out;
    * RGB at quality 90 made the sheets 198 KB median / 450 KB p90 against the
      124 KB the reviewer labels at a page a second, and he got six-second
      stalls waiting on the centre panel.
    """

    def _q(self, br, **kw):
        base = {
            "filename": "x.jpg",
            "task": "contamination",
            "question": "is the mark on this page?",
            "refs": [],
            "page_id": "ucsf/a#0",
            "box": [0, 0, 100, 100],
        }
        return br.Question(**{**base, **kw})

    def test_canvas_defaults_to_the_shared_landscape_sheet(self, br):
        assert br.sheet_size(self._q(br)) == br.CANVAS

    def test_a_portrait_question_overrides_the_canvas(self, br):
        assert br.sheet_size(self._q(br, canvas=[1530, 1350])) == (1530, 1350)

    def test_the_footer_names_the_page_by_default(self, br):
        q = self._q(br, item="7/50", detail="ranked")
        assert br.footer_text(q) == "7/50   ·   ucsf/a#0   ·   ranked"

    def test_an_anonymous_question_gives_the_reviewer_no_tell(self, br):
        """A planted control must not be identifiable from its footer."""
        q = self._q(br, item="7/50", detail="", anonymous=True)
        assert br.footer_text(q) == ""
        assert "ucsf" not in br.footer_text(q)

    def test_anonymous_still_shows_detail_when_one_is_set(self, br):
        assert br.footer_text(self._q(br, detail="note", anonymous=True)) == "note"


class TestRenderedSheetWeight:
    """The knobs that decide what a sheet costs, exercised through a real render."""

    @staticmethod
    def _page(tmp_path, w=600, h=800):
        from PIL import Image

        path = tmp_path / "page.png"
        img = Image.new("RGB", (w, h), "white")
        # a dark block in the middle, white margin all round: something to find,
        # and something for the border trim to shave.
        for x in range(w // 4, 3 * w // 4):
            for y in range(h // 3, 2 * h // 3):
                img.putpixel((x, y), (10, 10, 10) if (x + y) % 3 else (200, 30, 30))
        img.save(path)

        return SimpleNamespace(page_id="ucsf/a#0", path=str(path), width=w, height=h, marks=[])

    def _q(self, br, **kw):
        base = {
            "filename": "sheet.jpg",
            "task": "contamination",
            "question": "is the mark on this page?",
            "refs": [],
            "page_id": "ucsf/a#0",
            "box": [0, 0, 600, 800],
            "outline": False,
        }
        return br.Question(**{**base, **kw})

    def test_greyscale_writes_a_single_channel_sheet(self, br, tmp_path):
        from PIL import Image

        page = self._page(tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        br.render(self._q(br, greyscale=True, quality=80), {"ucsf/a#0": page}, tmp_path, out)
        with Image.open(out / "sheet.jpg") as im:
            assert im.mode == "L", "a scanned page is greyscale; three channels are paid for and unused"

    def test_greyscale_at_q80_is_smaller_than_rgb_at_q90(self, br, tmp_path):
        page = self._page(tmp_path)
        pages = {"ucsf/a#0": page}
        heavy, light = tmp_path / "heavy", tmp_path / "light"
        heavy.mkdir()
        light.mkdir()
        br.render(self._q(br), pages, tmp_path, heavy)
        br.render(self._q(br, greyscale=True, quality=80), pages, tmp_path, light)
        assert (light / "sheet.jpg").stat().st_size < (heavy / "sheet.jpg").stat().st_size

    def test_the_canvas_override_reaches_the_written_sheet(self, br, tmp_path):
        from PIL import Image

        out = tmp_path / "out"
        out.mkdir()
        br.render(self._q(br, canvas=[900, 1100]), {"ucsf/a#0": self._page(tmp_path)}, tmp_path, out)
        with Image.open(out / "sheet.jpg") as im:
            assert im.size == (900, 1100)


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

    def test_a_tile_over_a_boxed_mark_is_an_ordinary_accept(self, br):
        """#4040: the applier reassigns the mark under the tile and never reads the tile box.

        Holding these back did not merely delay them: the un-accepted indices were
        written to ``adjudications.json`` as cannot-links, recording the opposite
        of the vote that was cast.
        """
        qs = {
            "c0": {"task": "completeness2", "key": {"class_id": "c", "index": 0, "tile_box": True}},
            "c1": {"task": "completeness2", "key": {"class_id": "c", "index": 1, "tile_box": True}},
        }
        row = {
            "class_id": "c",
            "verdict": "none",
            "candidates": [
                {"index": 0, "page_id": "p0", "box": [0, 0, 216, 206], "mark_index": 5},
                {"index": 1, "page_id": "p1", "box": [0, 0, 216, 206], "mark_index": None},
            ],
        }
        rows, notes = br.translate_completeness2([row], qs, {"c0": "good", "c1": "good"})
        assert rows[0]["verdict"] == "0", "a tile over a boxed mark is a reassignment"
        assert rows[0]["needs_tight_box"] == [1], "only a tile over bare page needs drawing"
        assert any("tile box" in n for n in notes)

    def test_a_tile_box_without_candidates_still_needs_a_box(self, br):
        """A queue built before #4040 carries no candidate list to consult."""
        qs = {"c0": {"task": "completeness2", "key": {"class_id": "c", "index": 0, "tile_box": True}}}
        rows, _ = br.translate_completeness2([{"class_id": "c", "verdict": "none"}], qs, {"c0": "good"})
        assert rows[0]["verdict"] == "none" and rows[0]["needs_tight_box"] == [0]

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
        name = "fullmarks pm_crest -- right logo same as left?"
        # banked, then cleared off the dashboard: the app no longer holds it
        _archive(root, "fullmarks_pm_crest", name, {"a.jpg": "good", "b.jpg": "bad"})
        _cleared(keep, name, {"a.jpg": "good", "b.jpg": "bad"})
        votes, sources = br.collect_votes(None, root, keep)
        assert votes[name] == {"a.jpg": "good", "b.jpg": "bad"}
        assert sources == {"archive": 1, "cleared backup": 1}

    def test_the_live_app_wins_where_sources_disagree(self, br, tmp_path, monkeypatch):
        root, keep = tmp_path / "binary", tmp_path / "keep"
        name = "fullmarks rjr_block -- right logo same as left?"
        _cleared(keep, name, {"a.jpg": "bad"})
        monkeypatch.setattr(br, "fullmarks_detectors", lambda _d: [{"name": name}])
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
        _cleared(keep, "fullmarks empty -- same?", {})
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

        q = tmp_path / "fullmarks_pm_crest"
        q.mkdir()
        big = {"detector": {"name": "x"}, "labels": {"good": [{"filename": f"{i}.jpg"} for i in range(5)], "bad": []}}
        first = br.archive_votes(q, "2026-09-19", big)
        second = br.archive_votes(q, "2026-09-19", {"detector": {"name": "x"}, "labels": {"good": [], "bad": []}})
        assert first.name == "votes_2026-09-19.json"
        assert second != first  # the fuller archive is still there
        assert len(json.loads(first.read_text())["labels"]["good"]) == 5


def _queue(root: Path, dirname: str, name: str, questions: dict) -> Path:
    """A per-pass queue directory: a manifest and one rendered image per question."""
    import json

    d = root / dirname
    (d / "images").mkdir(parents=True, exist_ok=True)
    for fn in questions:
        (d / "images" / fn).write_bytes(b"jpeg")
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": name,
                "tasks": sorted({q["task"] for q in questions.values()}),
                "questions": questions,
            }
        ),
        encoding="utf-8",
    )
    return d


def _crop(i, cid="t/logo_a"):
    return {"task": "query_crops", "key": {"class_id": cid, "index": i, "n": 3}}


def _compl(i, cid="t/logo_a"):
    return {"task": "completeness2", "key": {"class_id": cid, "index": i, "tile_box": False}}


class TestRegroupOnePairPerClass:
    """Sam, 2026-09-19: one pair per class when the passes are this small; the image carries the question."""

    def test_both_passes_merge_and_answered_questions_are_dropped(self, br, tmp_path):
        root, keep = tmp_path / "binary", tmp_path / "keep"
        _queue(root, "q", "fullmarks t logo_a -- good extra query crop?", {"c0.jpg": _crop(0), "c1.jpg": _crop(1)})
        _queue(
            root,
            "c",
            "fullmarks t logo_a -- right mark same as left? (completeness 2)",
            {"m0.jpg": _compl(0), "m1.jpg": _compl(1)},
        )
        _archive(root, "q", "fullmarks t logo_a -- good extra query crop?", {"c0.jpg": "good"})
        (out,) = br.regroup(root, keep, base=None)
        name, qdir, n = out
        assert name == "fullmarks t logo_a -- read the question on each image"
        assert n == 3  # c0 is answered, so it is not asked again
        assert sorted(x.name for x in (qdir / "images").glob("*.jpg")) == ["c1.jpg", "m0.jpg", "m1.jpg"]

    def test_the_merged_manifest_keeps_each_question_task_so_bank_still_routes_it(self, br, tmp_path):
        import json

        root, keep = tmp_path / "binary", tmp_path / "keep"
        _queue(root, "q", "fullmarks t logo_a -- good extra query crop?", {"c0.jpg": _crop(0)})
        _queue(root, "c", "fullmarks t logo_a -- right mark same as left? (completeness 2)", {"m0.jpg": _compl(0)})
        (_name, qdir, _n) = br.regroup(root, keep, base=None)[0]
        m = json.loads((qdir / "manifest.json").read_text())
        assert m["tasks"] == ["completeness2", "query_crops"]
        assert {fn: q["task"] for fn, q in m["questions"].items()} == {
            "c0.jpg": "query_crops",
            "m0.jpg": "completeness2",
        }
        # bank groups by the question's own task, so one queue reaches two slates
        assert {br.TRANSLATORS[q["task"]][1] for q in m["questions"].values()} == {
            br.translate_query_crops,
            br.translate_completeness2,
        }

    def test_box_tighten_is_left_as_its_own_pair(self, br, tmp_path):
        root, keep = tmp_path / "binary", tmp_path / "keep"
        _queue(
            root,
            "b",
            "fullmarks staver boxes -- red box right?",
            {"b0.jpg": {"task": "box_tighten", "key": {"class_id": "t/logo_a", "index": 0}}},
        )
        assert br.regroup(root, keep, base=None) == []

    def test_a_question_answered_since_the_last_regroup_is_removed_from_the_merged_queue(self, br, tmp_path):
        root, keep = tmp_path / "binary", tmp_path / "keep"
        _queue(root, "q", "fullmarks t logo_a -- good extra query crop?", {"c0.jpg": _crop(0), "c1.jpg": _crop(1)})
        (_n1, qdir, first) = br.regroup(root, keep, base=None)[0]
        assert first == 2
        _archive(root, "q", "fullmarks t logo_a -- good extra query crop?", {"c0.jpg": "bad"})
        (_n2, qdir2, second) = br.regroup(root, keep, base=None)[0]
        assert (qdir2, second) == (qdir, 1)
        # the stale image is gone: load_queue imports the folder, so leaving it would re-ask it
        assert [x.name for x in (qdir / "images").glob("*.jpg")] == ["c1.jpg"]

    def test_a_question_without_a_class_is_refused_rather_than_grouped_wrongly(self, br, tmp_path):
        root, keep = tmp_path / "binary", tmp_path / "keep"
        _queue(
            root,
            "u",
            "fullmarks pm_crest -- right logo same as left?",
            {"s0.jpg": {"task": "ucsf_classes", "key": {"sheet": "p__a_00.png", "cell": 0}}},
        )
        with pytest.raises(SystemExit, match="key.class_id"):
            br.regroup(root, keep, base=None)


class TestVotesRouteByQuestionNotByFolder:
    """Sam, 2026-09-19: he loaded a second dataset into one detector while voting.

    The 90 labels that detector held were answers to two classes' questions, and
    keying votes by the folder they sat in dropped all 47 belonging to the other
    class.  A vote answers the QUESTION its filename names.
    """

    def _two_queues(self, br, tmp_path):
        root = tmp_path / "binary"
        _queue(root, "a", "fullmarks t logo_a -- read the question on each image", {"qcrop__a__c0.jpg": _crop(0)})
        _queue(
            root,
            "b",
            "fullmarks t logo_b -- read the question on each image",
            {"compl2__b__c0.jpg": _compl(0, "t/logo_b")},
        )
        manifests = {}
        for mf in sorted(root.glob("*/manifest.json")):
            import json

            m = json.loads(mf.read_text())
            manifests[m["dataset_name"]] = (mf.parent, m)
        return root, manifests

    def test_a_detector_holding_two_queues_routes_each_vote_to_its_own_slate(self, br, tmp_path):
        root, manifests = self._two_queues(br, tmp_path)
        # both votes cast in queue a's detector, as the merged dataset produced
        by_queue = {
            "fullmarks t logo_a -- read the question on each image": {
                "qcrop__a__c0.jpg": "good",
                "compl2__b__c0.jpg": "bad",
            }
        }
        questions = br.question_index(manifests)
        index, conflicts, unplaceable = br.vote_index(by_queue, questions)
        assert (conflicts, unplaceable) == ([], [])
        assert index == {"qcrop__a__c0.jpg": "good", "compl2__b__c0.jpg": "bad"}
        # and they land on different slates
        assert questions["qcrop__a__c0.jpg"]["task"] == "query_crops"
        assert questions["compl2__b__c0.jpg"]["task"] == "completeness2"

    def test_the_same_question_in_several_manifests_is_not_a_conflict(self, br, tmp_path):
        """A regrouped queue hardlinks its predecessor's images: 770 of 2,738 filenames are claimed twice."""
        root = tmp_path / "binary"
        _queue(root, "old", "fullmarks t logo_a -- good extra query crop?", {"c0.jpg": _crop(0)})
        _queue(root, "new", "fullmarks t logo_a -- read the question on each image", {"c0.jpg": _crop(0)})
        import json

        manifests = {
            json.loads(mf.read_text())["dataset_name"]: (mf.parent, json.loads(mf.read_text()))
            for mf in sorted(root.glob("*/manifest.json"))
        }
        assert br.question_index(manifests)["c0.jpg"] == _crop(0)

    def test_a_filename_claimed_twice_with_different_questions_raises(self, br, tmp_path):
        root = tmp_path / "binary"
        _queue(root, "one", "fullmarks t logo_a -- read the question on each image", {"c0.jpg": _crop(0)})
        _queue(root, "two", "fullmarks t logo_b -- read the question on each image", {"c0.jpg": _compl(0, "t/logo_b")})
        import json

        manifests = {
            json.loads(mf.read_text())["dataset_name"]: (mf.parent, json.loads(mf.read_text()))
            for mf in sorted(root.glob("*/manifest.json"))
        }
        with pytest.raises(SystemExit, match="claimed by"):
            br.question_index(manifests)

    def test_a_vote_naming_no_question_is_reported_not_dropped(self, br, tmp_path):
        _root, manifests = self._two_queues(br, tmp_path)
        questions = br.question_index(manifests)
        index, _conflicts, unplaceable = br.vote_index(
            {"fullmarks t logo_a -- read the question on each image": {"ghost.jpg": "good"}}, questions
        )
        assert index == {}
        assert len(unplaceable) == 1 and "ghost.jpg" in unplaceable[0]

    def test_one_question_voted_both_ways_in_two_queues_is_reported_not_guessed(self, br, tmp_path):
        _root, manifests = self._two_queues(br, tmp_path)
        questions = br.question_index(manifests)
        index, conflicts, _unplaceable = br.vote_index(
            {
                "fullmarks t logo_a -- read the question on each image": {"qcrop__a__c0.jpg": "good"},
                "fullmarks t logo_b -- read the question on each image": {"qcrop__a__c0.jpg": "bad"},
            },
            questions,
        )
        assert index == {}  # no rule can say which click was later
        assert len(conflicts) == 1 and "qcrop__a__c0.jpg" in conflicts[0]

    def test_regroup_does_not_re_ask_a_question_answered_in_a_foreign_detector(self, br, tmp_path):
        root, keep = tmp_path / "binary", tmp_path / "keep"
        _queue(root, "q", "fullmarks t logo_a -- good extra query crop?", {"c0.jpg": _crop(0), "c1.jpg": _crop(1)})
        # c0 was answered while the queue was loaded into some OTHER detector
        _archive(root, "q", "fullmarks t logo_z -- read the question on each image", {"c0.jpg": "good"})
        (out,) = br.regroup(root, keep, base=None)
        _name, qdir, n = out
        assert n == 1
        assert [x.name for x in (qdir / "images").glob("*.jpg")] == ["c1.jpg"]


@pytest.fixture(scope="module")
def cs():
    sys.path.insert(0, str(_FULLMARKS))
    try:
        yield importlib.import_module("contamination_sample")
    finally:
        sys.path.remove(str(_FULLMARKS))


class TestUcsfUnbandedFrame:
    """The #3922 draw: UCSF pages the letterhead pull never looked at, Food-weighted."""

    @staticmethod
    def _page(source, author=None):
        return SimpleNamespace(source=source, meta={"letterhead_author": author} if author else {})

    def test_a_banded_page_or_another_source_is_never_in_the_frame(self, cs):
        pages = {
            "ucsf/a": self._page("ucsf"),
            "ucsf/b": self._page("ucsf", author="Philip Morris"),
            "ucsf/c": self._page("ucsf"),
            "tobacco800/d": self._page("tobacco800"),
        }
        food, other = cs.unbanded_ucsf(pages, {"ucsf/a": "Food", "ucsf/c": "Opioids", "ucsf/b": "Tobacco"})
        assert (food, other) == (["ucsf/a"], ["ucsf/c"])

    def test_the_draw_takes_the_food_share_and_fills_the_rest_elsewhere(self, cs):
        import random

        food = [f"f{i}" for i in range(100)]
        other = [f"o{i}" for i in range(100)]
        drawn = cs.food_weighted(food, other, random.Random(0), 40)
        assert len(drawn) == len(set(drawn)) == 40
        assert sum(p.startswith("f") for p in drawn) == round(40 * cs.FOOD_SHARE)

    def test_a_short_stratum_is_taken_whole_rather_than_padded(self, cs):
        import random

        drawn = cs.food_weighted(["f0", "f1"], [f"o{i}" for i in range(100)], random.Random(0), 40)
        assert {"f0", "f1"} <= set(drawn)
        assert len(drawn) == len(set(drawn))


class TestInkCrop:
    """Whole-page questions crop to the inked region; a speck does not hold the crop open."""

    @staticmethod
    def _page(specks=True):
        from PIL import Image, ImageDraw

        im = Image.new("L", (1000, 1400), 255)
        d = ImageDraw.Draw(im)
        d.rectangle([300, 400, 700, 900], fill=0)  # the content
        if specks:
            for x, y in [(20, 20), (980, 1380), (15, 700), (990, 60)]:
                d.point((x, y), fill=0)
        return im

    def test_specks_at_the_border_are_ignored(self, br):
        left, top, right, bottom = br.ink_box(self._page())
        assert 280 <= left <= 300 and 380 <= top <= 400 and 700 <= right <= 720 and 900 <= bottom <= 920

    def test_a_faint_stamp_made_of_fragments_is_kept(self, br):
        from PIL import ImageDraw

        im = self._page(specks=False)
        d = ImageDraw.Draw(im)
        for i in range(12):  # broken strokes 4 px apart: each tiny, together a mark
            d.rectangle([850 + 8 * (i % 4), 1200 + 8 * (i // 4), 853 + 8 * (i % 4), 1203 + 8 * (i // 4)], fill=0)
        assert br.ink_box(im)[2] >= 880 and br.ink_box(im)[3] >= 1220

    def test_a_blank_page_is_left_alone(self, br):
        from PIL import Image

        assert br.ink_box(Image.new("L", (100, 100), 255)) is None


class TestClassRefs:
    def test_a_band_located_or_page_sized_instance_is_never_a_reference(self, br):
        Mark = SimpleNamespace
        pages = {
            "u/band": SimpleNamespace(
                width=1000,
                height=1000,
                marks=[Mark(class_id="c", box=[0, 0, 900, 200], provenance="ucsf_classes_band")],
            ),
            "u/wide": SimpleNamespace(
                width=1000, height=1000, marks=[Mark(class_id="c", box=[0, 0, 800, 300], provenance="ucsf_classes")]
            ),
            "u/tight": SimpleNamespace(
                width=1000, height=1000, marks=[Mark(class_id="c", box=[5, 5, 60, 60], provenance="ucsf_classes")]
            ),
        }
        classes = {"c": {"query_crop": "/q.png", "page_ids": ["u/band", "u/wide", "u/tight"]}}
        refs = br.class_refs("c", classes, pages)
        assert [r.page_id for r in refs] == [None, "u/tight"]


class TestRealPixelsFirst:
    """#4073: widen the window to real pixels before enlarging a small mark."""

    def test_a_small_mark_is_shown_at_most_at_the_cap_with_real_context(self, br):
        panel = (958, 1232)
        region = br.expand([600, 200, 24, 20], None, 1240, 1680, 0.25)
        wide = br.widen_to_real_pixels(region, 24, panel, 1240, 1680)
        w, h = wide[2] - wide[0], wide[3] - wide[1]
        scale = min(panel[0] / w, panel[1] / h)
        assert scale <= br.MAX_UPSCALE + 1e-6
        assert wide[0] <= 600 and wide[1] <= 200 and wide[2] >= 624 and wide[3] >= 220  # the mark stays in view
        assert 0 <= wide[0] and 0 <= wide[1] and wide[2] <= 1240 and wide[3] <= 1680

    def test_the_window_shifts_rather_than_leaving_the_page(self, br):
        wide = br.widen_to_real_pixels((0, 0, 40, 30), 30, (958, 1232), 1240, 1680)
        assert wide[0] == 0 and wide[1] == 0 and wide[2] > 40 and wide[3] > 30

    def test_a_large_mark_is_not_widened_past_what_fills_the_panel(self, br):
        region = (100, 100, 1100, 1400)
        assert br.widen_to_real_pixels(region, 1000, (958, 1232), 1240, 1680) == region

    def test_the_band_box_queue_routes_to_its_own_slate(self, br):
        src, _translator = br.TRANSLATORS["box_tighten_band"]
        assert src(Path("/c")) == Path("/c/audit/box_tighten_band/verdicts.jsonl")
        rows = [{"class_id": "c", "members": [{"index": 0}, {"index": 1}], "verdict": ""}]
        questions = {"a.jpg": {"task": "box_tighten_band", "key": {"class_id": "c", "index": 1}}}
        filled, unanswered = br.TRANSLATORS["box_tighten_band"][1](rows, questions, {"a.jpg": "good"})
        assert filled[0]["verdict"] == "1" and unanswered == []


class TestDrawnBoxes:
    """#4109: a box the reviewer draws with a Good vote replaces the proposal, mapped back to the page."""

    def _q(self, br):
        return br.Question(
            filename="a.jpg",
            task="box_tighten_band",
            question="?",
            refs=[],
            page_id="u/p",
            box=[263, 388, 84, 27],
            key={"class_id": "c", "index": 0},
        )

    def test_a_box_drawn_on_the_sheet_maps_back_to_the_page_pixels_it_covered(self, br):
        q, page = self._q(br), SimpleNamespace(width=1240, height=1680)
        region, s, ox, oy = br.outlined_placement(q, page)
        W, H = br.sheet_size(q)
        x, y, w, h = q.box
        norm = [
            (ox + (x - region[0]) * s) / W,
            (oy + (y - region[1]) * s) / H,
            (ox + (x + w - region[0]) * s) / W,
            (oy + (y + h - region[1]) * s) / H,
        ]
        assert br.sheet_box_to_page(q, page, norm) == [263, 388, 84, 27]

    def test_a_drawn_box_is_clipped_to_what_the_sheet_showed(self, br):
        q, page = self._q(br), SimpleNamespace(width=1240, height=1680)
        region, *_ = br.outlined_placement(q, page)
        x, y, w, h = br.sheet_box_to_page(q, page, [0.0, 0.0, 1.0, 1.0])
        assert x >= region[0] and y >= region[1] and x + w <= region[2] and y + h <= region[3]

    def test_the_drawn_box_replaces_the_proposal_and_keeps_it_for_the_record(self, br):
        rows = [{"class_id": "c", "members": [{"index": 0, "new_box": [1, 1, 5, 5]}], "verdict": ""}]
        questions = {"a.jpg": {"task": "box_tighten_band", "key": {"class_id": "c", "index": 0}}}
        filled, _ = br.TRANSLATORS["box_tighten_band"][1](
            rows, questions, {"a.jpg": "good"}, drawn={"a.jpg": [9, 9, 30, 20]}
        )
        m = filled[0]["members"][0]
        assert filled[0]["verdict"] == "0" and m["new_box"] == [9, 9, 30, 20]
        assert m["proposed_box"] == [1, 1, 5, 5] and m["drawn_by_reviewer"] is True


class TestBoxDraw:
    """#4125: only a drawn box changes a mark; Bad is reported, never applied."""

    def test_drawn_confirmed_and_bad_votes(self, br):
        rows = [
            {
                "class_id": "c",
                "members": [
                    {"index": 0, "page_id": "u/a", "new_box": None},
                    {"index": 1, "page_id": "u/b", "new_box": None},
                    {"index": 2, "page_id": "u/c", "new_box": None},
                ],
                "verdict": "",
            }
        ]
        questions = {f"{i}.jpg": {"task": "box_draw", "key": {"class_id": "c", "index": i}} for i in range(3)}
        votes = {"0.jpg": "good", "1.jpg": "good", "2.jpg": "bad"}
        out, notes = br.TRANSLATORS["box_draw"][1](rows, questions, votes, drawn={"0.jpg": [5, 6, 7, 8]})
        m = out[0]["members"]
        assert out[0]["verdict"] == "0" and m[0]["new_box"] == [5, 6, 7, 8]
        assert m[1].get("confirmed_as_is") and m[1]["new_box"] is None
        assert m[2].get("reviewer_says_not_on_page") and any("u/c" in n for n in notes)
        assert "box_draw" in br.DRAWN_BOX_TASKS
