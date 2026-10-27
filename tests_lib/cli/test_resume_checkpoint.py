"""The ``--resume`` checkpoint: identity, fingerprints, file round-trip and threshold pinning (#5280).

The end-to-end resume runs are in ``tests/cli/test_cli_resume.py``; these pin
:mod:`vtscore.cli_resume` on its own - what counts as the same run, the same
chunk and the same cut, and that the file never holds an importer's field
values in the clear.
"""

from __future__ import annotations

import json

import pytest

from vtscore.cli_resume import (
    THRESHOLD_DRIFT_LIMIT,
    ResumeRefusedError,
    StreamCheckpoint,
    checkpoint_path_for,
    chunk_digest,
    detector_identity,
    key_digest,
    run_identity,
)


def _identity(**overrides):
    sections = {
        "source": {"kind": "importer", "importer": "server_folder", "params": {"path": "/data", "password": "hunter2"}},
        "chunk_size": 500,
        "detectors": {"dogs": {"labelset": {"labels": []}}},
        "settings": {"beta": 1.0},
        "scoring": {"keep_negatives": False, "override_clipper": None},
        "output": {"exporter": "server_json_file", "fields": {"filepath": "/out/hits.ndjson"}},
    }
    sections.update(overrides)
    return run_identity(**sections)


def _media(media_id: int, md5: str) -> dict:
    return {"id": media_id, "filename": f"{media_id}.jpg", "origin_name": f"{media_id}.jpg", "md5": md5}


class TestIdentity:
    def test_the_same_run_has_the_same_identity(self):
        assert _identity() == _identity()

    def test_field_values_are_digested_not_stored(self):
        assert "hunter2" not in json.dumps(_identity())

    def test_a_changed_section_is_named_in_the_refusal(self, tmp_path):
        checkpoint = StreamCheckpoint(path=tmp_path / "c.json", identity=_identity())

        with pytest.raises(ResumeRefusedError, match="Changed: the chunk size; the settings") as exc:
            checkpoint.check_identity(_identity(chunk_size=250, settings={"beta": 2.0}))

        assert "Delete the checkpoint" in str(exc.value)

    def test_a_matching_identity_passes(self, tmp_path):
        StreamCheckpoint(path=tmp_path / "c.json", identity=_identity()).check_identity(_identity())

    def test_a_detectors_derived_caches_do_not_change_its_identity(self):
        det = {"media_type": "image", "labelset": {"labels": [{"md5": "a"}]}, "beta": 1.0}
        cached = {**det, "evidence_baseline": {"n": 3}, "line_verdicts": [1]}

        assert detector_identity(cached) == detector_identity(det)

    def test_a_detectors_labels_change_its_identity(self):
        det = {"media_type": "image", "labelset": {"labels": [{"md5": "a"}]}}
        relabelled = {**det, "labelset": {"labels": [{"md5": "b"}]}}

        assert detector_identity(relabelled) != detector_identity(det)


class TestChunkDigest:
    def test_is_stable(self):
        chunk = {1: _media(1, "a"), 2: _media(2, "b")}
        assert chunk_digest(chunk) == chunk_digest(dict(chunk))
        assert len(chunk_digest(chunk)) == 16

    @pytest.mark.parametrize(
        "changed",
        [
            {1: _media(1, "a"), 2: _media(2, "rewritten")},
            {1: _media(2, "b"), 2: _media(1, "a")},
            {1: _media(1, "a")},
            {2: _media(2, "a"), 3: _media(3, "b")},
        ],
        ids=["content", "order", "removed", "renumbered"],
    )
    def test_tells_a_changed_chunk_apart(self, changed):
        assert chunk_digest(changed) != chunk_digest({1: _media(1, "a"), 2: _media(2, "b")})

    def test_ignores_what_embedding_adds(self):
        chunk = {1: _media(1, "a")}
        embedded = {1: {**_media(1, "a"), "embeddings": {"clip": [0.1, 0.2]}}}
        assert chunk_digest(embedded) == chunk_digest(chunk)


class TestKeyDigest:
    """The fingerprint of a chunk passed unbuilt: its size and its importer's key (#5314)."""

    KEY = [["a.jpg", 10, 1_700_000_000_000_000_000], ["b.jpg", 20, 1_700_000_000_000_000_000]]

    def test_is_stable(self):
        assert key_digest(2, self.KEY) == key_digest(2, [list(row) for row in self.KEY])
        assert len(key_digest(2, self.KEY)) == 16

    @pytest.mark.parametrize(
        ("count", "key"),
        [
            (2, [KEY[1], KEY[0]]),
            (2, [KEY[0], ["b.jpg", 21, KEY[1][2]]]),
            (2, [KEY[0], ["b.jpg", 20, KEY[1][2] + 1]]),
            (1, KEY[:1]),
            (3, KEY),
        ],
        ids=["order", "size", "mtime", "removed", "count"],
    )
    def test_tells_a_changed_chunk_apart(self, count, key):
        assert key_digest(count, key) != key_digest(2, self.KEY)


class TestCheckpointFile:
    def test_round_trips(self, tmp_path):
        path = checkpoint_path_for(tmp_path / "hits.ndjson")
        assert path.name == "hits.ndjson.resume.json"
        StreamCheckpoint(
            path=path,
            identity=_identity(),
            chunks_done=2,
            chunk_digests=["aa", "bb"],
            thresholds={"dogs": 0.25},
            position=1234,
        ).save()

        loaded = StreamCheckpoint.load(path)

        assert loaded is not None
        assert (loaded.chunks_done, loaded.chunk_digests, loaded.thresholds, loaded.position) == (
            2,
            ["aa", "bb"],
            {"dogs": 0.25},
            1234,
        )
        loaded.check_identity(_identity())

    def test_a_v1_checkpoint_is_refused(self, tmp_path):
        """v1 fingerprinted every chunk by its media; v2 by its key where it has one."""
        path = tmp_path / "c.json"
        StreamCheckpoint(path=path, identity=_identity()).save()
        data = json.loads(path.read_text())
        assert data["format"] == "vtsearch-stream-checkpoint/v2"
        path.write_text(json.dumps({**data, "format": "vtsearch-stream-checkpoint/v1"}))

        with pytest.raises(ResumeRefusedError, match="cannot be read"):
            StreamCheckpoint.load(path)

    def test_no_file_is_no_checkpoint(self, tmp_path):
        assert StreamCheckpoint.load(tmp_path / "none.json") is None

    @pytest.mark.parametrize(
        "text", ["{not json", '{"format": "other"}', '{"format": "vtsearch-stream-checkpoint/v1"}']
    )
    def test_an_unreadable_file_is_refused(self, tmp_path, text):
        path = tmp_path / "c.json"
        path.write_text(text)
        with pytest.raises(ResumeRefusedError, match="cannot be read"):
            StreamCheckpoint.load(path)

    def test_clear_removes_it_and_tolerates_none(self, tmp_path):
        checkpoint = StreamCheckpoint(path=tmp_path / "c.json", identity=_identity())
        checkpoint.save()
        checkpoint.clear()
        checkpoint.clear()
        assert not checkpoint.path.exists()


class TestProgressChecks:
    def _checkpoint(self, tmp_path, **kwargs):
        return StreamCheckpoint(path=tmp_path / "c.json", identity=_identity(), **kwargs)

    def test_a_completed_chunk_must_match(self, tmp_path):
        checkpoint = self._checkpoint(tmp_path, chunks_done=2, chunk_digests=["aa", "bb"])
        checkpoint.check_chunk(2, "bb")
        with pytest.raises(ResumeRefusedError, match="chunk 2 no longer holds the same media"):
            checkpoint.check_chunk(2, "cc")

    def test_the_source_must_reach_the_completed_chunks(self, tmp_path):
        checkpoint = self._checkpoint(tmp_path, chunks_done=3, chunk_digests=["a", "b", "c"])
        checkpoint.check_reached(3)
        with pytest.raises(ResumeRefusedError, match="holds 2 chunk"):
            checkpoint.check_reached(2)


class TestPinThresholds:
    def _checkpoint(self, tmp_path):
        return StreamCheckpoint(path=tmp_path / "c.json", identity=_identity(), thresholds={"dogs": 0.4})

    def test_an_identical_threshold_is_kept_quietly(self, tmp_path):
        mlps = {"dogs": {"threshold": 0.4}}
        assert self._checkpoint(tmp_path).pin_thresholds(mlps) == {}
        assert mlps["dogs"]["threshold"] == 0.4

    def test_noise_is_pinned_to_the_recorded_cut(self, tmp_path):
        mlps = {"dogs": {"threshold": 0.4 + THRESHOLD_DRIFT_LIMIT / 10}}
        moved = self._checkpoint(tmp_path).pin_thresholds(mlps)
        assert mlps["dogs"]["threshold"] == 0.4
        assert set(moved) == {"dogs"}

    def test_a_real_move_is_refused(self, tmp_path):
        with pytest.raises(ResumeRefusedError, match="retrained to threshold 0.5000"):
            self._checkpoint(tmp_path).pin_thresholds({"dogs": {"threshold": 0.5}})

    def test_a_different_set_of_detectors_is_refused(self, tmp_path):
        with pytest.raises(ResumeRefusedError, match="the detectors that run are"):
            self._checkpoint(tmp_path).pin_thresholds({"cats": {"threshold": 0.4}})
