"""One media, one label per detector (issue #3174).

A detector's labelset used to grow a *second* ``LabeledElement`` for a media
whose existing entry named the same file by a different origin - an exemplar
carrying the ``example_media`` sentinel, a label imported from a plain md5
list, a label saved under another dataset's importer.  The write path decided
"this entry belongs to the active dataset" by comparing
:func:`~vtscore.datasets.labelset.element_key` (origin-preferred), while the
*read* path (``restore_labels_from_detector``) turned entries into votes by
``resolve_media_ids`` (origin **or** md5).  Anything that matched only on md5
was therefore restored as a vote, re-emitted as a fresh element, and kept
alongside its original - so ``num_training`` counted 356 for a 300-image pass.

These tests pin the symmetry: an entry that becomes a vote is an entry the
vote-derived labelset supersedes.
"""

from __future__ import annotations

import shutil

import pytest

from tests import load_detector_and_wait
from vtscore.detectors.store import _detector_path, _read_detector, _write_detector
from vtsearch.settings import get_detectors_dir


@pytest.fixture(autouse=True)
def clean_detectors_dir():
    tm_dir = get_detectors_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)
    yield
    tm_dir = get_detectors_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)


def _register(client, name: str) -> str:
    res = client.post(
        "/api/detectors/registry",
        json={"name": name, "media_type": "audio", "text_query": "t"},
    )
    assert res.status_code in (200, 201), res.get_data(as_text=True)
    return res.get_json()["detector"]["id"]


def _seed_labelset(name: str, labels: list[dict]) -> None:
    path = _detector_path(name)
    data = _read_detector(path)
    assert data is not None
    data["labelset"] = {"labels": labels}
    _write_detector(path, data)


def _labels(name: str) -> list[dict]:
    data = _read_detector(_detector_path(name))
    assert data is not None
    return data["labelset"]["labels"]


class TestVoteSyncCollapsesMd5MatchedEntries:
    """The post-vote sync path (``sync_labels_to_loaded_detector``)."""

    def test_md5_matched_entry_is_superseded_not_duplicated(self, client):
        """The reported shape: two ``bad`` elements for one media.

        The pair was indistinguishable in ``labels-detail`` (which does not
        render ``origin``) - same md5, same label, different element id.
        """
        from vtsearch.state import medias

        if not medias:
            pytest.skip("No medias loaded")

        detector_id = _register(client, "DupBad")
        first_id = next(iter(medias))
        media = medias[first_id]

        # An entry that names the same bytes but not the same origin: exactly
        # what a plain md5 label list, or an exemplar's example_media
        # sentinel, leaves behind.
        _seed_labelset(
            "DupBad",
            [{"md5": media["md5"], "label": "bad", "origin_name": "elsewhere.wav"}],
        )
        load_detector_and_wait(client, detector_id)

        assert client.post(f"/api/medias/{first_id}/vote", json={"target": "bad"}).status_code == 200

        labels = _labels("DupBad")
        assert len(labels) == 1, f"labelset stored the same media twice: {labels}"
        assert labels[0]["md5"] == media["md5"]
        assert labels[0]["label"] == "bad"
        # The surviving entry is the vote-derived one, carrying the active
        # dataset's provenance rather than the stale name.
        assert labels[0]["origin_name"] == media.get("origin_name")

    def test_num_training_counts_distinct_media(self, client):
        """The counter a reviewer reads as progress must not exceed the slate."""
        from vtscore.detectors.registry import get_detector
        from vtsearch.state import medias

        ids = list(medias)[:3]
        if len(ids) < 3:
            pytest.skip("Need at least 3 medias")

        detector_id = _register(client, "CountTruth")
        _seed_labelset(
            "CountTruth",
            [{"md5": medias[cid]["md5"], "label": "bad", "origin_name": f"stale{cid}.wav"} for cid in ids],
        )
        load_detector_and_wait(client, detector_id)

        for cid in ids:
            client.post(f"/api/medias/{cid}/vote", json={"target": "bad"})

        assert len(_labels("CountTruth")) == 3
        entry = get_detector(detector_id)
        assert entry is not None
        assert entry["num_training"] == 3

    def test_cross_dataset_entry_still_survives(self, client):
        """The merge must stay non-destructive for entries that resolve nowhere."""
        from vtsearch.state import medias

        if not medias:
            pytest.skip("No medias loaded")

        detector_id = _register(client, "KeepForeign")
        _seed_labelset(
            "KeepForeign",
            [
                {
                    "md5": "ff" * 16,
                    "label": "good",
                    "origin": {"importer": "ds_a", "params": {"size": "100"}},
                    "origin_name": "from_other_dataset.wav",
                }
            ],
        )
        load_detector_and_wait(client, detector_id)

        first_id = next(iter(medias))
        client.post(f"/api/medias/{first_id}/vote", json={"target": "good"})

        labels = _labels("KeepForeign")
        assert {lbl["md5"] for lbl in labels} == {"ff" * 16, medias[first_id]["md5"]}

    def test_preexisting_duplicate_pair_heals_on_the_next_vote(self, client):
        """A labelset that already carries the duplicate collapses on write."""
        from vtsearch.state import medias

        if not medias:
            pytest.skip("No medias loaded")

        detector_id = _register(client, "HealDupes")
        first_id = next(iter(medias))
        media = medias[first_id]
        _seed_labelset(
            "HealDupes",
            [
                {"md5": media["md5"], "label": "bad", "origin_name": "stale.wav"},
                {
                    "md5": media["md5"],
                    "label": "bad",
                    "origin": media.get("origin"),
                    "origin_name": media.get("origin_name"),
                },
            ],
        )
        load_detector_and_wait(client, detector_id)

        client.post(f"/api/medias/{first_id}/vote", json={"target": "bad"})
        assert len(_labels("HealDupes")) == 1


class TestExplicitSaveCollapsesMd5MatchedEntries:
    """``POST /api/detectors/<name>/labels`` shares the merge helper."""

    def test_save_supersedes_an_md5_matched_entry(self, client):
        from vtsearch.state import medias

        if not medias:
            pytest.skip("No medias loaded")

        _register(client, "SaveDedupe")
        first_id = next(iter(medias))
        media = medias[first_id]
        _seed_labelset(
            "SaveDedupe",
            [{"md5": media["md5"], "label": "good", "origin_name": "elsewhere.wav"}],
        )

        client.post(f"/api/medias/{first_id}/vote", json={"target": "good"})
        res = client.post("/api/detectors/SaveDedupe/labels")
        assert res.status_code == 200
        assert res.get_json()["num_labels"] == 1
        assert len(_labels("SaveDedupe")) == 1


class TestRestoredRegionBoxSurvivesResync:
    """A drawn region must not be erased by the labelset round-trip."""

    def test_region_box_rides_through_restore_and_back(self, client):
        from vtsearch.state import medias

        if not medias:
            pytest.skip("No medias loaded")

        detector_id = _register(client, "KeepRegion")
        first_id = next(iter(medias))
        media = medias[first_id]
        box = [0.1, 0.2, 0.6, 0.7]
        _seed_labelset(
            "KeepRegion",
            [
                {
                    "md5": media["md5"],
                    "label": "good",
                    "origin": media.get("origin"),
                    "origin_name": media.get("origin_name"),
                    "region_box": box,
                }
            ],
        )
        load_detector_and_wait(client, detector_id)

        # Voting a *different* media resyncs the whole labelset to disk; the
        # region-voted element must be re-emitted with its box intact.
        others = [cid for cid in medias if cid != first_id]
        if not others:
            pytest.skip("Need a second media")
        client.post(f"/api/medias/{others[0]}/vote", json={"target": "bad"})

        labels = _labels("KeepRegion")
        kept = [lbl for lbl in labels if lbl["md5"] == media["md5"]]
        assert len(kept) == 1
        assert kept[0].get("region_box") == box


class TestImportMergeKeepsOneElementPerMedia:
    """``_merge_entries_into_labelset`` - last write wins, no contradicting pair."""

    def test_conflicting_label_replaces_rather_than_appends(self):
        from vtscore.datasets.labelset import LabelSet
        from vtsearch.routes.detectors.labels import _merge_entries_into_labelset

        ls = LabelSet.from_dict({"labels": [{"md5": "aa" * 16, "label": "good", "origin_name": "x.wav"}]})
        applied, skipped, new_entries = _merge_entries_into_labelset(
            ls, [{"md5": "aa" * 16, "label": "bad", "origin_name": "x.wav"}]
        )

        assert (applied, skipped) == (1, 0)
        assert len(ls.elements) == 1
        assert ls.elements[0].label == "bad"
        assert len(new_entries) == 1

    def test_restating_the_same_label_is_skipped(self):
        from vtscore.datasets.labelset import LabelSet
        from vtsearch.routes.detectors.labels import _merge_entries_into_labelset

        ls = LabelSet.from_dict({"labels": [{"md5": "aa" * 16, "label": "good", "origin_name": "x.wav"}]})
        applied, skipped, _ = _merge_entries_into_labelset(
            ls, [{"md5": "aa" * 16, "label": "good", "origin_name": "x.wav"}]
        )

        assert (applied, skipped) == (0, 1)
        assert len(ls.elements) == 1

    def test_same_md5_under_a_different_origin_collapses(self):
        """Origin-keyed and md5-keyed records of one file are one media."""
        from vtscore.datasets.labelset import LabelSet
        from vtsearch.routes.detectors.labels import _merge_entries_into_labelset

        ls = LabelSet.from_dict(
            {
                "labels": [
                    {
                        "md5": "aa" * 16,
                        "label": "good",
                        "origin": {"importer": "ds_a", "params": {}},
                        "origin_name": "x.wav",
                    }
                ]
            }
        )
        applied, skipped, _ = _merge_entries_into_labelset(
            ls,
            [
                {
                    "md5": "aa" * 16,
                    "label": "bad",
                    "origin": {"importer": "ds_b", "params": {}},
                    "origin_name": "copy_of_x.wav",
                }
            ],
        )

        assert (applied, skipped) == (1, 0)
        assert len(ls.elements) == 1
        assert ls.elements[0].label == "bad"

    def test_distinct_media_still_accumulate(self):
        from vtscore.datasets.labelset import LabelSet
        from vtsearch.routes.detectors.labels import _merge_entries_into_labelset

        ls = LabelSet.from_dict({"labels": [{"md5": "aa" * 16, "label": "good"}]})
        applied, skipped, _ = _merge_entries_into_labelset(
            ls,
            [
                {"md5": "bb" * 16, "label": "good"},
                {"label": "not-a-label", "md5": "cc" * 16},
            ],
        )

        assert (applied, skipped) == (1, 1)
        assert len(ls.elements) == 2
