"""Vote surfacing provenance: the vocabulary, the state threading, the round-trip.

Recording-only feature (issue #2850): nothing here asserts a behaviour change,
because there is none.  What these tests protect is that the record is
*written* and *survives* — the two ways a provenance feature dies silently.

The load-bearing cases:

* an idempotent re-vote must not overwrite an existing record (a stale tab
  rewriting the context of a click it did not make), and
* a detector-load restore must carry the record back into vote state, because
  the labelset is rebuilt from live votes on every single vote — so a restore
  that dropped it would erase every recorded vote's provenance on the user's
  next click.  That is exactly the bug ``region_box`` had.
"""

from __future__ import annotations

import pytest

from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.datasets.vote_provenance import (
    FLOWS,
    METADATA_KEY,
    SCHEMA_VERSION,
    attach_provenance,
    coerce_provenance,
    normalize_provenance,
    read_provenance,
)
from vtscore.state import get_active_detector_context
from vtscore.state.core import get_active_context
from vtscore.state.votes import apply_label, set_vote

FULL = {
    "flow": "list_review",
    "select_mode": "hard",
    "sort_kind": "learned",
    "rank_at_vote": 37,
    "score_at_vote": 0.512,
}


def _media(cid: int) -> dict:
    return {"md5": f"md5-{cid}", "filename": f"m{cid}.wav", "origin_name": f"m{cid}.wav"}


@pytest.fixture
def one_media():
    """Register a single media in the active dataset and return its id."""
    ctx = get_active_context()
    cid = (max(ctx.medias) if ctx.medias else 0) + 1
    ctx.medias[cid] = _media(cid)
    try:
        yield cid
    finally:
        ctx.medias.pop(cid, None)


class TestVocabulary:
    """:mod:`vtscore.datasets.vote_provenance` normalisation."""

    def test_full_payload_round_trips_with_a_schema_version(self):
        assert normalize_provenance(FULL) == {"v": SCHEMA_VERSION, **FULL}

    def test_axes_are_independent(self):
        """The whole point of the shape: a hand-picked ``hard`` select mode
        outside autopilot is recordable, and so is autopilot's top-of-list
        ``good`` phase.  A fused ``autopilot:hard`` enum could express
        neither."""
        manual_hard = normalize_provenance({"flow": "list_review", "select_mode": "hard"})
        assert manual_hard == {"v": 1, "flow": "list_review", "select_mode": "hard"}

        autopilot_top = normalize_provenance({"flow": "autopilot", "phase": "good", "select_mode": "top"})
        assert autopilot_top["phase"] == "good"
        assert autopilot_top["select_mode"] == "top"

    def test_empty_and_contentless_payloads_are_dropped(self):
        for raw in (None, {}, {"flow": "unknown"}):
            assert normalize_provenance(raw) is None

    def test_phase_is_dropped_off_a_non_autopilot_flow(self):
        out = normalize_provenance({"flow": "list_review", "phase": "hard"})
        assert "phase" not in out

    def test_absent_fields_are_omitted_not_stored_as_null(self):
        out = normalize_provenance({"flow": "bulk"})
        assert out == {"v": SCHEMA_VERSION, "flow": "bulk"}

    @pytest.mark.parametrize(
        "raw",
        [
            {"flow": "nope"},
            {"phase": "idle"},
            {"select_mode": "toplist"},
            {"sort_kind": "cosine"},
            {"surfaced_by": "autopilot:hard"},
            {"rank_at_vote": -1},
            {"rank_at_vote": 1.5},
            {"rank_at_vote": True},
            {"score_at_vote": float("nan")},
            {"score_at_vote": float("inf")},
            "not a dict",
        ],
    )
    def test_malformed_payloads_are_rejected(self, raw):
        with pytest.raises(ValueError):
            normalize_provenance(raw)

    def test_coerce_never_raises(self):
        """Payloads off disk get dropped, not repaired: a half-understood
        record is worse than none, because a calibration partition would
        trust it."""
        assert coerce_provenance({"flow": "nope"}) is None
        assert coerce_provenance("garbage") is None
        assert coerce_provenance(FULL) == {"v": SCHEMA_VERSION, **FULL}

    def test_attach_preserves_importer_metadata(self):
        merged = attach_provenance({"contentID": "x"}, {"flow": "autopilot", "phase": "new"})
        assert merged["contentID"] == "x"
        assert merged[METADATA_KEY]["phase"] == "new"

    def test_attach_is_a_no_op_without_provenance(self):
        """A build that records nothing must emit a byte-identical labelset."""
        assert attach_provenance({"contentID": "x"}, None) == {"contentID": "x"}
        assert attach_provenance(None, None) is None

    def test_attach_does_not_mutate_the_callers_dict(self):
        """``hit_custom_metadata`` output is shared across every element built
        from one media, so mutating it would leak one vote's context onto
        another's element."""
        original = {"contentID": "x"}
        attach_provenance(original, {"flow": "bulk"})
        assert original == {"contentID": "x"}

    def test_read_provenance_tolerates_missing_metadata(self):
        assert read_provenance(None) is None
        assert read_provenance({}) is None
        assert read_provenance({METADATA_KEY: {"flow": "bogus"}}) is None

    def test_flow_vocabulary_is_closed(self):
        """A new flow value is a wire-format change; adding one should be a
        deliberate edit here as well as in the module."""
        assert FLOWS == {
            "autopilot",
            "list_review",
            "find_verify",
            "labelset_review",
            "seed_example",
            "import",
            "bulk",
            "undo",
            "unknown",
        }


class TestVoteStateThreading:
    """``set_vote`` / ``apply_label`` record into ``ctx.vote_provenance``."""

    def test_set_vote_records_provenance(self, one_media):
        set_vote(one_media, "good", provenance=FULL)
        assert get_active_detector_context().vote_provenance[one_media] == {
            "v": SCHEMA_VERSION,
            **FULL,
        }

    def test_bad_votes_carry_provenance_too(self, one_media):
        """The surfacing context is just as real for a no-vote, and the
        calibration set the recording exists for is built from both sides."""
        set_vote(one_media, "bad", provenance={"flow": "autopilot", "phase": "hard"})
        assert get_active_detector_context().vote_provenance[one_media]["phase"] == "hard"

    def test_idempotent_revote_does_not_overwrite(self, one_media):
        """The audit-H1 rule extended to provenance: a stale tab re-sending the
        target a media already holds made no surfacing event, so it must not
        rewrite the record the original click left."""
        set_vote(one_media, "good", provenance={"flow": "autopilot", "phase": "hard"})
        set_vote(one_media, "good", provenance={"flow": "list_review"})
        recorded = get_active_detector_context().vote_provenance[one_media]
        assert recorded["flow"] == "autopilot"
        assert recorded["phase"] == "hard"

    def test_a_real_flip_does_overwrite(self, one_media):
        """A good→bad flip *is* a new surfacing event."""
        set_vote(one_media, "good", provenance={"flow": "autopilot", "phase": "hard"})
        set_vote(one_media, "bad", provenance={"flow": "list_review"})
        assert get_active_detector_context().vote_provenance[one_media]["flow"] == "list_review"

    def test_unvote_clears_provenance(self, one_media):
        set_vote(one_media, "good", provenance=FULL)
        set_vote(one_media, "none")
        assert one_media not in get_active_detector_context().vote_provenance

    def test_a_vote_without_provenance_clears_a_stale_record(self, one_media):
        """The new vote was surfaced somehow; keeping the previous vote's
        context would attribute it to a flow that did not produce it."""
        set_vote(one_media, "good", provenance={"flow": "autopilot", "phase": "hard"})
        set_vote(one_media, "bad")
        assert one_media not in get_active_detector_context().vote_provenance

    def test_malformed_provenance_is_dropped_not_raised(self, one_media):
        """The state layer coerces; rejection belongs at the request boundary."""
        set_vote(one_media, "good", provenance={"flow": "nope"})
        assert one_media not in get_active_detector_context().vote_provenance

    def test_apply_label_records_provenance(self, one_media):
        apply_label(one_media, "good", provenance={"flow": "seed_example"})
        assert get_active_detector_context().vote_provenance[one_media]["flow"] == "seed_example"


class TestLabelsetRoundTrip:
    """Provenance survives composition, serialisation, and re-parse."""

    def test_from_clips_and_votes_writes_the_namespaced_key(self):
        medias = {1: _media(1), 2: _media(2)}
        ls = LabelSet.from_clips_and_votes(
            medias,
            {1: None},
            {2: None},
            vote_provenance={1: {"v": 1, "flow": "autopilot", "phase": "hard"}, 2: {"v": 1, "flow": "bulk"}},
        )
        by_label = {el.label: el for el in ls.elements}
        assert by_label["good"].metadata[METADATA_KEY]["phase"] == "hard"
        assert by_label["bad"].metadata[METADATA_KEY]["flow"] == "bulk"

    def test_votes_without_provenance_get_no_metadata_key(self):
        ls = LabelSet.from_clips_and_votes({1: _media(1)}, {1: None}, {})
        assert ls.elements[0].metadata is None

    def test_survives_json_round_trip(self):
        ls = LabelSet.from_clips_and_votes({1: _media(1)}, {1: None}, {}, vote_provenance={1: {"v": 1, **FULL}})
        restored = LabelSet.from_dict(ls.to_dict())
        assert read_provenance(restored.elements[0].metadata) == {"v": SCHEMA_VERSION, **FULL}

    def test_dupe_set_members_share_the_representatives_record(self):
        """Same rule as ``region_box``: the representative is what the user was
        actually shown, so it is what was surfaced."""
        rep = {
            "md5": "rep",
            "filename": "rep.wav",
            "origin": {
                "importer": "dupe_set",
                "members": [
                    {"md5": "a", "origin_name": "a.wav", "filename": "a.wav"},
                    {"md5": "b", "origin_name": "b.wav", "filename": "b.wav"},
                ],
            },
        }
        ls = LabelSet.from_clips_and_votes(
            {1: rep}, {1: None}, {}, vote_provenance={1: {"v": 1, "flow": "autopilot", "phase": "new"}}
        )
        assert len(ls.elements) == 2
        assert all(read_provenance(el.metadata)["phase"] == "new" for el in ls.elements)

    def test_provenance_does_not_displace_importer_custom_metadata(self):
        media = {**_media(1), "custom_metadata": {"contentID": "abc"}}
        ls = LabelSet.from_clips_and_votes({1: media}, {1: None}, {}, vote_provenance={1: {"v": 1, "flow": "bulk"}})
        meta = ls.elements[0].metadata
        assert meta["contentID"] == "abc"
        assert meta[METADATA_KEY]["flow"] == "bulk"


class TestElementVoteFlip:
    """``apply_element_vote_in_data`` (the dashboard's labelset review)."""

    def _data(self, label: str = "good") -> dict:
        from vtscore.detectors.labelset_elements import stable_element_id

        el = LabeledElement(md5="abc", label=label, origin_name="a.wav", filename="a.wav")
        return {"labelset": LabelSet([el]).to_dict()}, stable_element_id(el)

    def test_a_flip_stamps_the_new_provenance(self):
        from vtscore.detectors.labelset_elements import apply_element_vote_in_data

        data, eid = self._data("good")
        changed, updated, action = apply_element_vote_in_data(
            data, eid, "bad", provenance={"v": 1, "flow": "labelset_review"}
        )
        assert (changed, action) == (True, "flipped")
        assert read_provenance(updated.metadata)["flow"] == "labelset_review"

    def test_an_idempotent_reassert_leaves_the_record_alone(self):
        from vtscore.detectors.labelset_elements import apply_element_vote_in_data

        data, eid = self._data("good")
        changed, _updated, action = apply_element_vote_in_data(
            data, eid, "good", provenance={"v": 1, "flow": "labelset_review"}
        )
        assert (changed, action) == (False, "unchanged")
        stored = LabelSet.from_dict(data["labelset"]).elements[0]
        assert read_provenance(stored.metadata) is None


class TestRestorationCycle:
    """The erasure hazard: vote -> compose -> restore -> recompose.

    ``sync_labels_to_loaded_detector`` rebuilds the entire labelset from live
    vote state on *every* vote.  So if a detector load restores votes without
    their recorded provenance, the user's very next click rewrites the whole
    labelset with the context stripped out.  ``region_box`` shipped with
    exactly this bug; this is the test that says provenance did not.
    """

    def test_provenance_survives_a_detector_load_and_the_next_resync(self):
        from vtscore.detectors.label_restoration import restore_labels_from_detector

        det = get_active_detector_context()
        snap = get_active_context().medias
        set_vote(1, "good", provenance={"flow": "autopilot", "phase": "hard"})
        set_vote(2, "bad", provenance={"flow": "list_review", "rank_at_vote": 4})

        # What the detector JSON would hold after the vote-triggered sync.
        on_disk = LabelSet.from_clips_and_votes(
            snap,
            dict(det.good_votes),
            dict(det.bad_votes),
            expand_dupes=False,
            vote_provenance=dict(det.vote_provenance),
        ).to_dict()

        # Loading the detector afresh (as a dataset switch would).
        det.good_votes.clear()
        det.bad_votes.clear()
        det.vote_provenance.clear()
        restore_labels_from_detector({"labelset": on_disk})

        assert det.vote_provenance[1]["phase"] == "hard"
        assert det.vote_provenance[2]["rank_at_vote"] == 4

        # The next vote resyncs the whole labelset from live state; the
        # restored records must still be in what it writes.
        resynced = LabelSet.from_clips_and_votes(
            snap,
            dict(det.good_votes),
            dict(det.bad_votes),
            expand_dupes=False,
            vote_provenance=dict(det.vote_provenance),
        )
        by_md5 = {el.md5: el for el in resynced.elements}
        assert read_provenance(by_md5[snap[1]["md5"]].metadata)["phase"] == "hard"
        assert read_provenance(by_md5[snap[2]["md5"]].metadata)["flow"] == "list_review"

    def test_a_legacy_labelset_restores_without_provenance(self):
        """Elements written before this feature carry none, and must restore
        cleanly rather than raising or inventing a record."""
        from vtscore.detectors.label_restoration import restore_labels_from_detector

        snap = get_active_context().medias
        legacy = LabelSet.from_clips_and_votes(snap, {1: None}, {}, expand_dupes=False)
        restored = restore_labels_from_detector({"labelset": legacy.to_dict()})

        det = get_active_detector_context()
        assert restored == 1
        assert 1 in det.good_votes
        assert det.vote_provenance == {}
