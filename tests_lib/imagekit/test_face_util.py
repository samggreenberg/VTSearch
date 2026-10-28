"""imagekit.face.face_util: CSV and NPZ round trips."""

from __future__ import annotations

import numpy as np
import pytest

from imagekit.face.compat import CompatibilityError
from imagekit.face.face_util import (
    assign_face_indices,
    embedding_key,
    load_detections_csv,
    load_embeddings_npz,
    load_face_records,
    load_face_records_csv,
    save_detections_csv,
    save_embeddings_npz,
    save_face_records_csv,
    split_embedding_key,
)
from imagekit.face.records import FaceDetection, FaceRecord

_LM = np.array([[10, 20], [30, 20], [20, 30], [12, 40], [28, 40]], np.float32)


def _records(compat="oss-test:1") -> list[FaceRecord]:
    rng = np.random.default_rng(42)
    out = []
    for media_id, n in (("a.jpg", 2), ("dir/b, c.png", 1)):
        for i in range(n):
            det = FaceDetection((i, i, 10 + i, 12 + i), _LM if i == 0 else None, 0.9 - i / 10)
            out.append(FaceRecord.from_detection(det, rng.standard_normal(512), compat, media_id))
    return out


def test_embedding_keys():
    assert embedding_key("x::y", 3) == "x::y::3"
    assert split_embedding_key("x::y::3") == ("x::y", 3)
    with pytest.raises(ValueError):
        split_embedding_key("plain")


def test_face_indices_count_per_media_in_order():
    indexed = assign_face_indices(_records())
    assert [i for i, _ in indexed] == [0, 1, 0]


def test_detections_csv_round_trip(tmp_path):
    rows = [("a.jpg", 0, FaceDetection((1, 2, 3, 4), _LM, 0.9)), ("a.jpg", 1, FaceDetection((5, 6, 7, 8), None, 0.6))]
    assert save_detections_csv(tmp_path / "d.csv", rows) == 2
    back = load_detections_csv(tmp_path / "d.csv")
    assert [(m, i) for m, i, _ in back] == [("a.jpg", 0), ("a.jpg", 1)]
    assert back[0][2].bbox == pytest.approx((1, 2, 3, 4)) and back[1][2].landmarks is None
    (tmp_path / "bad.csv").write_text("x,y\n1,2\n")
    with pytest.raises(ValueError, match="not a detections CSV"):
        load_detections_csv(tmp_path / "bad.csv")


def test_records_csv_and_npz_round_trip(tmp_path):
    records = _records()
    assert save_face_records_csv(tmp_path / "r.csv", records) == 3
    assert save_embeddings_npz(tmp_path / "r.npz", records) == 3
    vectors, compat = load_embeddings_npz(tmp_path / "r.npz")
    assert compat == "oss-test:1" and set(vectors) == {"a.jpg::0", "a.jpg::1", "dir/b, c.png::0"}
    csv_only = load_face_records_csv(tmp_path / "r.csv")
    assert all(r.embedding.shape == (0,) for r in csv_only)
    joined = load_face_records(tmp_path / "r.csv", tmp_path / "r.npz")
    assert [r.media_id for r in joined] == ["a.jpg", "a.jpg", "dir/b, c.png"]
    for original, back in zip(records, joined):
        np.testing.assert_array_equal(back.embedding, original.embedding)
        assert back.quality == pytest.approx(original.quality, abs=1e-5)


def test_empty_run_still_writes_both_files(tmp_path):
    assert save_face_records_csv(tmp_path / "e.csv", []) == 0
    assert save_embeddings_npz(tmp_path / "e.npz", []) == 0
    assert load_face_records(tmp_path / "e.csv", tmp_path / "e.npz") == []


def test_mixed_spaces_cannot_share_an_archive(tmp_path):
    records = _records("a:1") + _records("b:1")
    with pytest.raises(CompatibilityError):
        save_embeddings_npz(tmp_path / "m.npz", records)


def test_join_refuses_a_missing_vector_or_foreign_archive(tmp_path):
    records = _records()
    save_face_records_csv(tmp_path / "r.csv", records)
    save_embeddings_npz(tmp_path / "r.npz", records[:1])
    with pytest.raises(ValueError, match="no embedding for"):
        load_face_records(tmp_path / "r.csv", tmp_path / "r.npz")
    save_embeddings_npz(tmp_path / "other.npz", _records("other:9"))
    with pytest.raises(ValueError, match="compat"):
        load_face_records(tmp_path / "r.csv", tmp_path / "other.npz")
    np.savez(tmp_path / "foreign.npz", x=np.zeros(3))
    with pytest.raises(ValueError, match="not an imagekit"):
        load_embeddings_npz(tmp_path / "foreign.npz")
