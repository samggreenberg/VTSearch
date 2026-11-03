"""The two CLIs (imagekit.cli behind scripts/data/detect_faces.py and scripts/data/process_faces.py)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from imagekit.cli import detect_faces_main, process_faces_main
from imagekit.face import load_detections_csv, load_face_records, register_provider
from tests_lib.imagekit._synthetic_models import build_model_dir, scrfd_face

KPS = ((-5, -5), (5, -5), (0, 0), (-4, 6), (4, 6))
FACES = [scrfd_face(0, 20, 30, 0.9, (16, 16, 16, 16), kps=KPS), scrfd_face(2, 10, 5, 0.7, (40, 40, 40, 40), kps=KPS)]


@pytest.fixture
def models(tmp_path: Path) -> dict[str, Path]:
    return build_model_dir(tmp_path / "models", FACES)


@pytest.fixture
def images(tmp_path: Path) -> Path:
    """Two readable images (the synthetic detector finds two faces in anything) and one broken file."""
    root = tmp_path / "images"
    (root / "sub").mkdir(parents=True)
    for name in ("a.png", "sub/b.jpg"):
        cv2.imwrite(str(root / name), np.zeros((640, 640, 3), np.uint8))
    (root / "bad.png").write_bytes(b"not an image")
    return root


def _config_file(tmp_path: Path, models: dict[str, Path], **extra) -> Path:
    """A pipeline config with paths *relative* to its own directory."""
    cfg = {
        "detector": {"model_bundle": str(models["detector"].relative_to(tmp_path))},
        "embedder": {
            "model": str(models["embedder"].relative_to(tmp_path)),
            "config": str(models["embedder_config"].relative_to(tmp_path)),
        },
    }
    cfg.update(extra)
    path = tmp_path / "faces.json"
    path.write_text(json.dumps(cfg))
    return path


class TestDetectFaces:
    def test_dir_run_warns_and_continues_then_counts(self, models, images, tmp_path, capsys):
        out = tmp_path / "dets.csv"
        assert detect_faces_main([str(models["detector"]), str(out), "--dir", str(images)]) == 0
        captured = capsys.readouterr()
        assert "Total Detects: 4" in captured.out
        assert "WARNING" in captured.err and "bad.png" in captured.err
        rows = load_detections_csv(out)
        assert [(m, i) for m, i, _ in rows] == [("a.png", 0), ("a.png", 1), ("sub/b.jpg", 0), ("sub/b.jpg", 1)]
        assert rows[0][2].landmarks is not None

    def test_list_and_img_inputs(self, models, images, tmp_path, capsys):
        listing = tmp_path / "list.txt"
        listing.write_text(f"{images / 'a.png'}\n# skip\n{images / 'missing.png'}\n")
        assert detect_faces_main([str(models["detector"]), str(tmp_path / "l.csv"), "--list", str(listing)]) == 0
        assert "Total Detects: 2" in capsys.readouterr().out
        assert (
            detect_faces_main(
                [
                    str(models["detector"]),
                    str(tmp_path / "i.csv"),
                    "--img",
                    str(images / "a.png"),
                    "--confidence-threshold",
                    "0.8",
                ]
            )
            == 0
        )
        assert "Total Detects: 1" in capsys.readouterr().out

    def test_exactly_one_input_selector(self, models, images, tmp_path):
        with pytest.raises(SystemExit):
            detect_faces_main([str(models["detector"]), str(tmp_path / "x.csv")])
        with pytest.raises(SystemExit):
            detect_faces_main([str(models["detector"]), str(tmp_path / "x.csv"), "--dir", str(images), "--img", "a"])

    def test_a_bad_model_is_fatal(self, images, tmp_path):
        (tmp_path / "bad.onnx").write_bytes(b"nope")
        with pytest.raises(Exception):
            detect_faces_main([str(tmp_path / "bad.onnx"), str(tmp_path / "x.csv"), "--dir", str(images)])


class TestProcessFaces:
    def test_writes_csv_and_npz_that_join_back(self, models, images, tmp_path, capsys):
        cfg = _config_file(tmp_path, models)
        prefix = tmp_path / "out" / "faces"
        prefix.parent.mkdir()
        assert process_faces_main([str(cfg), str(prefix), "--dir", str(images), "--num-threads", "1"]) == 0
        captured = capsys.readouterr()
        assert "Total Faces: 4" in captured.out and "bad.png" in captured.err
        records = load_face_records(prefix.with_suffix(".csv"), prefix.with_suffix(".npz"))
        assert [(r.media_id, r.face_index) for r in records] == [
            ("a.png", 0),
            ("a.png", 1),
            ("sub/b.jpg", 0),
            ("sub/b.jpg", 1),
        ]
        assert all(r.compat == "oss-synthetic:1" and r.embedding.shape == (512,) for r in records)
        assert records[0].quality == pytest.approx(np.linalg.norm(records[0].embedding), abs=1e-4)

    def test_provider_flag_and_config_provider(self, models, images, tmp_path, capsys):
        calls: list[str] = []

        class Recording:
            def initialize(self, opts, config):
                calls.append(f"init threads={opts.num_threads}")

            def process(self, image, *, confidence_threshold=None, media_id=""):
                calls.append(f"process {media_id}")
                return []

            def process_buffer(self, buffer, *, confidence_threshold=None, media_id=""):
                return []

            def compatibility(self):
                return "rec:1"

        register_provider("recording", Recording)
        cfg = _config_file(tmp_path, models, provider="recording", runtime={"num_threads": 3})
        assert process_faces_main([str(cfg), str(tmp_path / "p"), "--img", str(images / "a.png")]) == 0
        assert calls == ["init threads=3", f"process {images / 'a.png'}"]
        assert "Total Faces: 0" in capsys.readouterr().out
        calls.clear()
        oss_cfg = _config_file(tmp_path, models)
        assert (
            process_faces_main(
                [
                    str(oss_cfg),
                    str(tmp_path / "q"),
                    "--img",
                    str(images / "a.png"),
                    "--provider",
                    "recording",
                    "--num-threads",
                    "5",
                ]
            )
            == 0
        )
        assert calls[0] == "init threads=5"

    def test_an_unknown_provider_is_fatal(self, models, images, tmp_path):
        cfg = _config_file(tmp_path, models)
        with pytest.raises(KeyError):
            process_faces_main([str(cfg), str(tmp_path / "p"), "--img", str(images / "a.png"), "--provider", "nope"])
