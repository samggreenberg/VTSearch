"""Tests for cross-dataset label restoration (:mod:`vtscore.detectors.label_restoration`).

This is the code path behind the "detectors are reusable across compatible
datasets" product promise: when a detector trained on Dataset A is loaded
against Dataset B, its saved labelset entries are re-attached to the loaded
medias.  Matching happens in two passes:

1. **Direct** — by origin+origin_name, MD5, or (last resort) origin_name,
   via :func:`vtscore.state.resolve_media_ids`.
2. **Fallback** (``_resolve_unmatched``) — for entries that matched nothing
   in pass 1, resolve the original file from its origin trail, hash it, and
   match by content MD5.  This is what lets labels re-attach when the same
   underlying file lives in both datasets under different provenance.

Restored labels are applied ``silent=True`` so they seed autopilot's good/bad
gates without contaminating the per-session Smart/Stable trends.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager

from vtscore.datasets.labelset import LabelSet, LabeledElement
from vtscore.detectors.label_restoration import restore_labels_from_detector
from vtscore.state import get_active_context, get_active_detector_context


def _det_data(elements: list[LabeledElement]) -> dict:
    """Wrap a list of :class:`LabeledElement` in a detector-data dict."""
    return {"labelset": LabelSet(elements).to_dict()}


def _fake_resolver(path):
    """Build a stand-in for ``resolve_file_context`` that yields *path*.

    ``path`` may be a real :class:`~pathlib.Path`, a bogus path (to exercise
    the OSError branch), or ``None`` (unresolvable origin).
    """

    @contextmanager
    def _cm(origin, origin_name="", filename=""):
        yield path

    return _cm


class TestDirectMatchPass:
    """First-pass matching by origin / MD5 against the loaded dataset."""

    def test_labels_restored_by_origin_and_md5(self):
        snap = get_active_context().medias
        good_ids = [1, 2]
        bad_ids = [3]
        labelset = LabelSet.from_clips_and_votes(
            snap, {i: None for i in good_ids}, {i: None for i in bad_ids}
        )

        restored = restore_labels_from_detector({"labelset": labelset.to_dict()})

        assert restored == 3
        det = get_active_detector_context()
        assert set(det.good_votes) == set(good_ids)
        assert set(det.bad_votes) == set(bad_ids)

    def test_restored_labels_are_silent(self):
        """Restored labels must not append to ``label_history`` (silent seed)."""
        snap = get_active_context().medias
        labelset = LabelSet.from_clips_and_votes(snap, {1: None, 2: None}, {})

        restore_labels_from_detector({"labelset": labelset.to_dict()})

        det = get_active_detector_context()
        assert set(det.good_votes) == {1, 2}
        assert list(det.label_history) == []

    def test_invalid_labels_are_skipped(self):
        """Entries whose label is neither good nor bad are ignored."""
        snap = get_active_context().medias
        m1 = snap[1]
        elements = [
            LabeledElement(
                md5=m1["md5"],
                label="maybe",  # not good/bad
                origin=m1["origin"],
                origin_name=m1["origin_name"],
            ),
        ]

        restored = restore_labels_from_detector(_det_data(elements))

        assert restored == 0
        det = get_active_detector_context()
        assert not det.good_votes
        assert not det.bad_votes

    def test_one_entry_matching_multiple_dupe_medias_counts_once(self):
        """An entry matching several medias (same MD5) restores all but counts once."""
        ctx = get_active_context()
        m1 = ctx.medias[1]
        # Add a second media that shares media 1's MD5 (a duplicate file).
        dupe_id = max(ctx.medias) + 1
        dupe = dict(m1)
        dupe["id"] = dupe_id
        dupe["origin_name"] = "dupe_of_1.wav"
        dupe["origin"] = {"importer": "other", "params": {}}
        ctx.medias[dupe_id] = dupe

        elements = [
            LabeledElement(md5=m1["md5"], label="good", origin=None, origin_name="")
        ]
        restored = restore_labels_from_detector(_det_data(elements))

        # Both medias get labeled, but the entry counts as one restoration.
        assert restored == 1
        det = get_active_detector_context()
        assert set(det.good_votes) == {1, dupe_id}


class TestEmptyInputs:
    """Guard clauses that short-circuit to zero restorations."""

    def test_no_labelset_key(self):
        assert restore_labels_from_detector({}) == 0

    def test_empty_labelset(self):
        assert restore_labels_from_detector({"labelset": {"labels": []}}) == 0

    def test_no_medias_loaded(self):
        snap = get_active_context().medias
        labelset = LabelSet.from_clips_and_votes(snap, {1: None}, {})
        det_data = {"labelset": labelset.to_dict()}

        ctx = get_active_context()
        saved = dict(ctx.medias)
        ctx.medias.clear()
        try:
            assert restore_labels_from_detector(det_data) == 0
        finally:
            ctx.medias.update(saved)


class TestUnmatchedFallbackPass:
    """Second pass: resolve the origin file and match by content MD5."""

    @staticmethod
    def _foreign_entry(md5="0" * 32, label="good"):
        """A labelset entry that cannot match the loaded dataset in pass 1.

        It carries an origin+origin_name pair absent from the dataset and an
        MD5 that isn't present either, so :func:`resolve_media_ids` returns
        nothing (and the origin_name fallback is suppressed because the entry
        *has* an origin key).  Only the second pass can match it.
        """
        return LabeledElement(
            md5=md5,
            label=label,
            origin={"importer": "foreign", "params": {"tag": "x"}},
            origin_name="foreign_clip.wav",
            filename="foreign_clip.wav",
        )

    def test_resolved_file_md5_matches_loaded_media(self, tmp_path, monkeypatch):
        """Origin resolves to a file whose bytes hash to a loaded media's MD5."""
        m1 = get_active_context().medias[1]
        resolved = tmp_path / "resolved.wav"
        resolved.write_bytes(m1["media_bytes"])
        assert hashlib.md5(resolved.read_bytes()).hexdigest() == m1["md5"]

        monkeypatch.setattr(
            "vtscore.detectors.resolver.resolve_file_context",
            _fake_resolver(resolved),
        )

        restored = restore_labels_from_detector(_det_data([self._foreign_entry()]))

        assert restored == 1
        det = get_active_detector_context()
        assert set(det.good_votes) == {1}
        # Fallback path also applies silently.
        assert list(det.label_history) == []

    def test_bad_label_restored_via_fallback(self, tmp_path, monkeypatch):
        m1 = get_active_context().medias[1]
        resolved = tmp_path / "resolved.wav"
        resolved.write_bytes(m1["media_bytes"])
        monkeypatch.setattr(
            "vtscore.detectors.resolver.resolve_file_context",
            _fake_resolver(resolved),
        )

        restored = restore_labels_from_detector(
            _det_data([self._foreign_entry(label="bad")])
        )

        assert restored == 1
        det = get_active_detector_context()
        assert set(det.bad_votes) == {1}

    def test_resolved_file_md5_absent_from_dataset(self, tmp_path, monkeypatch):
        """Origin resolves, but the file's content isn't in the loaded dataset."""
        resolved = tmp_path / "unrelated.bin"
        resolved.write_bytes(b"content that hashes to nothing in the dataset")
        monkeypatch.setattr(
            "vtscore.detectors.resolver.resolve_file_context",
            _fake_resolver(resolved),
        )

        restored = restore_labels_from_detector(_det_data([self._foreign_entry()]))

        assert restored == 0
        det = get_active_detector_context()
        assert not det.good_votes
        assert not det.bad_votes

    def test_origin_unresolvable(self, monkeypatch):
        """No resolver can locate the file -> the entry is left unrestored."""
        monkeypatch.setattr(
            "vtscore.detectors.resolver.resolve_file_context",
            _fake_resolver(None),
        )

        restored = restore_labels_from_detector(_det_data([self._foreign_entry()]))

        assert restored == 0
        assert not get_active_detector_context().good_votes

    def test_resolved_path_unreadable(self, tmp_path, monkeypatch):
        """Resolver yields a path that doesn't exist -> OSError is swallowed."""
        missing = tmp_path / "does_not_exist.wav"
        monkeypatch.setattr(
            "vtscore.detectors.resolver.resolve_file_context",
            _fake_resolver(missing),
        )

        restored = restore_labels_from_detector(_det_data([self._foreign_entry()]))

        assert restored == 0
        assert not get_active_detector_context().good_votes

    def test_direct_and_fallback_combined(self, tmp_path, monkeypatch):
        """One entry matches directly; a second only via the fallback pass."""
        ctx = get_active_context()
        m1 = ctx.medias[1]
        m2 = ctx.medias[2]

        direct = LabeledElement(
            md5=m1["md5"],
            label="good",
            origin=m1["origin"],
            origin_name=m1["origin_name"],
        )
        foreign = self._foreign_entry(label="good")

        resolved = tmp_path / "resolved.wav"
        resolved.write_bytes(m2["media_bytes"])
        monkeypatch.setattr(
            "vtscore.detectors.resolver.resolve_file_context",
            _fake_resolver(resolved),
        )

        restored = restore_labels_from_detector(_det_data([direct, foreign]))

        assert restored == 2
        assert set(get_active_detector_context().good_votes) == {1, 2}
