"""End-to-end tests for ``--stream-results --resume`` (#5280).

A resumable streamed run checkpoints after every chunk it writes, and the same
command run again after an interruption continues after the last one.  These
tests pin the guarantees ``docs/CLI.md`` makes for it:

- an interrupted run, resumed, leaves the file an uninterrupted run writes:
  no chunk repeated, none dropped, the header written once - even when the
  process died partway through writing a chunk;
- the completed chunks are checked against the source, but not embedded or
  scored again, and a folder's are not even read (#5314);
- a checkpoint from a different run (chunk size, detectors, what is scored),
  a source that changed in the part already done, a retrained threshold that
  moved, or an output that lost what was written is refused;
- the checkpoint is removed once the run completes.

The interruption is a ``KeyboardInterrupt`` from the scorer, which is what a
Ctrl-C does and which the CLI's ``except Exception`` does not swallow.
"""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import pytest

from vtsearch.settings import get_detectors_dir
from vtscore.datasets.importers.base import DatasetImporter, PluginField
from vtscore.utils.hashing import content_md5


# ---------------------------------------------------------------------------
# Fixtures: a 2-dim audio dataset and a stubbed head that splits it by id
# ---------------------------------------------------------------------------


def _make_audio_media(media_id: int, *, salt: bytes = b"") -> dict:
    raw = media_id.to_bytes(4, "little") + salt + b"\x00" * 96
    # Unit-norm 2-dim embedding whose dim 0 decreases with id (wrapping every
    # 9 ids so long sources stay in range); the stubbed head thresholds dim 0.
    e0 = 1.0 - 0.1 * (((media_id - 1) % 9) + 1)
    e1 = (1.0 - e0 * e0) ** 0.5
    return {
        "id": media_id,
        "media_type": "audio",
        "duration": 1.0,
        "file_size": len(raw),
        "md5": content_md5(raw),
        "embedding": [e0, e1],
        "embedder": "clap",
        "media_bytes": None,
        "media_string": None,
        "media_path": None,
        "filename": f"clip_{media_id:03d}.wav",
        "category": "test",
        "origin": {"importer": "stub_ds", "params": {}},
        "origin_name": f"clip_{media_id:03d}.wav",
    }


def _write_pickle_dataset(path: Path, medias: dict) -> None:
    from vtscore.datasets.container import write_container

    write_container(path, pickle.dumps({"medias": medias}), {"format_version": 1})


def _write_detector(name: str, *, extra_label: bool = False) -> None:
    from vtscore.detectors.store import _detector_path, _write_detector as write

    labels = [
        {"md5": "a" * 32, "label": "good", "origin": {"importer": "s", "params": {}}, "origin_name": "a"},
        {"md5": "b" * 32, "label": "bad", "origin": {"importer": "s", "params": {}}, "origin_name": "b"},
    ]
    if extra_label:
        labels.append({"md5": "c" * 32, "label": "good", "origin": {"importer": "s", "params": {}}, "origin_name": "c"})
    write(_detector_path(name), {"name": name, "media_type": "audio", "labelset": {"labels": labels}})


def _settings_file(tmp_path: Path, detector_name: str) -> Path:
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"autofind_detectors": [detector_name], "detectors_dir": str(get_detectors_dir())}))
    return p


@pytest.fixture(autouse=True)
def _clean_detectors_dir():
    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)
    yield
    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)


class _Head:
    """The stubbed training: a fixed head and a threshold a test can move."""

    def __init__(self) -> None:
        self.threshold = 0.5
        self.trained = 0


@pytest.fixture
def head(monkeypatch) -> _Head:
    """Train a deterministic head whose logit is ``100 * (embedding[0] - 0.65)``.

    So ids whose dim 0 is 0.9 / 0.8 / 0.7 are above the 0.5 threshold and the
    rest below, a split to assert against.
    """
    import torch
    from torch import nn

    import vtscore.cli as cli_mod

    state = _Head()

    def _fake_load_and_train(detector_names, media_type, first_chunk_medias, routed=None, override_clipper=None):
        state.trained += 1
        linear = nn.Linear(2, 1)
        with torch.no_grad():
            linear.weight.data = torch.tensor([[100.0, 0.0]])
            linear.bias.data = torch.tensor([-65.0])
        mlp = nn.Sequential(linear)
        mlp.eval()
        return {name: {"mlp": mlp, "threshold": state.threshold} for name in detector_names}

    monkeypatch.setattr(cli_mod, "_load_and_train_detectors", _fake_load_and_train)
    return state


class _Interrupt:
    """Make the scorer raise ``KeyboardInterrupt`` on its *n*-th call (1-based), like a Ctrl-C."""

    def __init__(self, monkeypatch) -> None:
        import vtscore.cli as cli_mod

        self.calls = 0
        self.at: int | None = None
        real = cli_mod._score_medias_with_detectors

        def _score(*args: Any, **kwargs: Any):
            self.calls += 1
            if self.at is not None and self.calls == self.at:
                raise KeyboardInterrupt
            return real(*args, **kwargs)

        monkeypatch.setattr(cli_mod, "_score_medias_with_detectors", _score)


@pytest.fixture
def scorer(monkeypatch) -> _Interrupt:
    return _Interrupt(monkeypatch)


def _run(
    tmp_path: Path,
    out: Path,
    *,
    chunk_size: int = 2,
    dataset: Path | None = None,
    exporter: str = "server_json_file",
    keep_negatives: bool = True,
    resume: bool = True,
) -> None:
    from vtscore.cli import autodetect_main_chunked

    autodetect_main_chunked(
        dataset_path=str(dataset or tmp_path / "ds.pkl"),
        chunk_size=chunk_size,
        settings_path=str(tmp_path / "settings.json"),
        exporter_name=exporter,
        exporter_field_values={"filepath": str(out)},
        stream_results=True,
        keep_negatives=keep_negatives,
        resume=resume,
    )


def _checkpoint(out: Path) -> Path:
    return out.with_name(out.name + ".resume.json")


@pytest.fixture
def run_setup(client, tmp_path):
    """A detector, a settings file naming it, and a 9-media pickle (5 chunks of 2)."""
    _write_detector("resume-det")
    _settings_file(tmp_path, "resume-det")
    _write_pickle_dataset(tmp_path / "ds.pkl", {i: _make_audio_media(i) for i in range(1, 10)})
    return tmp_path


def _reference(tmp_path: Path, name: str = "reference.ndjson", **kwargs: Any) -> bytes:
    """The bytes an uninterrupted run of the same command writes."""
    out = tmp_path / name
    _run(tmp_path, out, **kwargs)
    return out.read_bytes()


def _interrupted(tmp_path: Path, out: Path, scorer: _Interrupt, at: int, **kwargs: Any) -> None:
    """Run until the scorer is interrupted on its *at*-th call."""
    scorer.calls, scorer.at = 0, at
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, out, **kwargs)
    scorer.calls, scorer.at = 0, None


def _error(capsys) -> str:
    return capsys.readouterr().err


# ---------------------------------------------------------------------------
# Resuming
# ---------------------------------------------------------------------------


class TestResumeContinuesTheRun:
    def test_resumed_run_writes_the_file_an_uninterrupted_run_writes(self, run_setup, head, scorer):
        tmp_path = run_setup
        expected = _reference(tmp_path)
        out = tmp_path / "hits.ndjson"

        _interrupted(tmp_path, out, scorer, at=3)
        checkpoint = json.loads(_checkpoint(out).read_text())
        assert checkpoint["chunks_done"] == 2
        assert checkpoint["position"] == out.stat().st_size

        _run(tmp_path, out)

        assert out.read_bytes() == expected
        lines = out.read_text().splitlines()
        assert sum(1 for line in lines if "_meta" in json.loads(line)) == 1
        ids = [json.loads(line)["id"] for line in lines[1:]]
        assert sorted(ids) == list(range(1, 10))

    def test_completed_chunks_are_not_scored_again(self, run_setup, head, scorer):
        tmp_path = run_setup
        out = tmp_path / "hits.ndjson"
        _interrupted(tmp_path, out, scorer, at=4)

        _run(tmp_path, out)

        # Chunks 4 and 5 only; 1-3 were written before the interruption.
        assert scorer.calls == 2

    def test_a_chunk_torn_by_a_kill_is_cut_back_and_written_once(self, run_setup, head, scorer):
        """A kill mid-chunk leaves bytes past the last checkpoint; the resume drops them."""
        tmp_path = run_setup
        expected = _reference(tmp_path)
        out = tmp_path / "hits.ndjson"
        _interrupted(tmp_path, out, scorer, at=3)
        with out.open("a") as f:
            f.write('{"detector": "resume-det", "id": 5, "score": 0.7')

        _run(tmp_path, out)

        assert out.read_bytes() == expected

    def test_interrupted_before_any_chunk_resumes_after_the_header(self, run_setup, head, scorer):
        tmp_path = run_setup
        expected = _reference(tmp_path)
        out = tmp_path / "hits.ndjson"
        _interrupted(tmp_path, out, scorer, at=1)
        assert json.loads(_checkpoint(out).read_text())["chunks_done"] == 0

        _run(tmp_path, out)

        assert out.read_bytes() == expected

    def test_the_detectors_are_retrained_on_the_first_chunk(self, run_setup, head, scorer):
        tmp_path = run_setup
        out = tmp_path / "hits.ndjson"
        _interrupted(tmp_path, out, scorer, at=3)
        head.trained = 0

        _run(tmp_path, out)

        assert head.trained == 1

    def test_the_checkpoint_and_lock_are_removed_when_the_run_completes(self, run_setup, head, scorer):
        tmp_path = run_setup
        out = tmp_path / "hits.ndjson"
        _interrupted(tmp_path, out, scorer, at=3)
        assert _checkpoint(out).exists()

        _run(tmp_path, out)

        assert sorted(p.name for p in tmp_path.glob("hits.ndjson*")) == ["hits.ndjson"]

    def test_a_fresh_resumable_run_matches_a_plain_streamed_run(self, run_setup, head):
        tmp_path = run_setup
        plain = _reference(tmp_path, "plain.ndjson", resume=False)
        resumable = _reference(tmp_path, "resumable.ndjson")

        assert resumable == plain
        assert not _checkpoint(tmp_path / "resumable.ndjson").exists()

    def test_rerunning_a_completed_run_starts_it_over(self, run_setup, head):
        tmp_path = run_setup
        out = tmp_path / "hits.ndjson"
        expected = _reference(tmp_path, "hits.ndjson")

        _run(tmp_path, out)

        assert out.read_bytes() == expected

    def test_a_source_that_grew_past_the_completed_chunks_resumes(self, run_setup, head, scorer):
        """Files added after the part already done are picked up."""
        tmp_path = run_setup
        out = tmp_path / "hits.ndjson"
        _interrupted(tmp_path, out, scorer, at=3)
        _write_pickle_dataset(tmp_path / "ds.pkl", {i: _make_audio_media(i) for i in range(1, 12)})
        expected = _reference(tmp_path, "grown.ndjson")

        _run(tmp_path, out)

        assert out.read_bytes() == expected

    def test_an_exporter_that_commits_before_any_chunk_keeps_the_checkpoint_whole(
        self, run_setup, head, scorer, monkeypatch
    ):
        """A commit made before the completed chunks are re-read still records all their fingerprints."""
        from vtscore.exporters import get_exporter

        real = get_exporter("server_json_file")

        class _Eager(type(real)):
            def export_cli_streaming_resumable(self, header, chunks, field_values, resume):
                if resume.position is not None:
                    resume.commit(resume.position)
                return super().export_cli_streaming_resumable(header, chunks, field_values, resume)

        eager = _Eager()
        monkeypatch.setattr("vtscore.exporters.get_exporter", lambda name: eager if name == real.name else None)
        tmp_path = run_setup
        expected = _reference(tmp_path)
        out = tmp_path / "hits.ndjson"
        _interrupted(tmp_path, out, scorer, at=4)
        _interrupted(tmp_path, out, scorer, at=1)

        checkpoint = json.loads(_checkpoint(out).read_text())
        assert checkpoint["chunks_done"] == len(checkpoint["chunk_digests"]) == 3

        _run(tmp_path, out)

        assert out.read_bytes() == expected

    def test_csv_output_resumes(self, run_setup, head, scorer):
        tmp_path = run_setup
        expected = _reference(tmp_path, "reference.csv", exporter="server_csv_file")
        out = tmp_path / "hits.csv"
        _interrupted(tmp_path, out, scorer, at=3, exporter="server_csv_file")

        _run(tmp_path, out, exporter="server_csv_file")

        assert out.read_bytes() == expected
        assert out.read_text().count("detector,threshold") == 1

    def test_a_threshold_within_noise_is_pinned_to_the_checkpoints(self, run_setup, head, scorer):
        tmp_path = run_setup
        expected = _reference(tmp_path)
        out = tmp_path / "hits.ndjson"
        _interrupted(tmp_path, out, scorer, at=3)
        head.threshold = 0.5 + 1e-7

        _run(tmp_path, out)

        assert out.read_bytes() == expected


# ---------------------------------------------------------------------------
# Refusing
# ---------------------------------------------------------------------------


class TestResumeRefuses:
    def test_a_different_chunk_size(self, run_setup, head, scorer, capsys):
        tmp_path = run_setup
        out = tmp_path / "hits.ndjson"
        _interrupted(tmp_path, out, scorer, at=3)
        before = out.read_bytes()

        with pytest.raises(SystemExit):
            _run(tmp_path, out, chunk_size=3)

        err = _error(capsys)
        assert "the chunk size" in err
        assert "Delete the checkpoint" in err
        assert out.read_bytes() == before

    def test_changed_detector_labels(self, run_setup, head, scorer, capsys):
        tmp_path = run_setup
        out = tmp_path / "hits.ndjson"
        _interrupted(tmp_path, out, scorer, at=3)
        _write_detector("resume-det", extra_label=True)

        with pytest.raises(SystemExit):
            _run(tmp_path, out)

        assert "the detectors or their labels" in _error(capsys)

    def test_a_change_to_what_is_scored(self, run_setup, head, scorer, capsys):
        tmp_path = run_setup
        out = tmp_path / "hits.ndjson"
        _interrupted(tmp_path, out, scorer, at=3)

        with pytest.raises(SystemExit):
            _run(tmp_path, out, keep_negatives=False)

        assert "what is scored" in _error(capsys)

    def test_a_source_changed_in_the_part_already_done(self, run_setup, head, scorer, capsys):
        tmp_path = run_setup
        out = tmp_path / "hits.ndjson"
        _interrupted(tmp_path, out, scorer, at=4)
        medias = {i: _make_audio_media(i) for i in range(1, 10)}
        medias[3] = _make_audio_media(3, salt=b"rewritten")
        _write_pickle_dataset(tmp_path / "ds.pkl", medias)
        before = out.read_bytes()
        scorer.calls = 0

        with pytest.raises(SystemExit):
            _run(tmp_path, out)

        assert "chunk 2 no longer holds the same media" in _error(capsys)
        assert scorer.calls == 0
        assert out.read_bytes() == before

    def test_a_source_that_shrank_below_the_completed_chunks(self, run_setup, head, scorer, capsys):
        tmp_path = run_setup
        out = tmp_path / "hits.ndjson"
        _interrupted(tmp_path, out, scorer, at=4)
        _write_pickle_dataset(tmp_path / "ds.pkl", {i: _make_audio_media(i) for i in range(1, 5)})

        with pytest.raises(SystemExit):
            _run(tmp_path, out)

        assert "the source now holds 2 chunk(s), but the checkpoint records 3" in _error(capsys)

    def test_a_threshold_that_moved(self, run_setup, head, scorer, capsys):
        tmp_path = run_setup
        out = tmp_path / "hits.ndjson"
        _interrupted(tmp_path, out, scorer, at=3)
        head.threshold = 0.6

        with pytest.raises(SystemExit):
            _run(tmp_path, out)

        assert "retrained to threshold 0.6000" in _error(capsys)

    def test_a_missing_output(self, run_setup, head, scorer, capsys):
        tmp_path = run_setup
        out = tmp_path / "hits.ndjson"
        _interrupted(tmp_path, out, scorer, at=3)
        out.unlink()

        head.trained = 0

        with pytest.raises(SystemExit):
            _run(tmp_path, out)

        assert f"its output {out} is missing" in _error(capsys)
        assert head.trained == 0

    def test_an_output_shorter_than_the_checkpoint(self, run_setup, head, scorer, capsys):
        tmp_path = run_setup
        out = tmp_path / "hits.ndjson"
        _interrupted(tmp_path, out, scorer, at=3)
        out.write_bytes(out.read_bytes()[:10])

        with pytest.raises(SystemExit):
            _run(tmp_path, out)

        assert "fewer than the" in _error(capsys)

    def test_a_second_run_on_the_same_output_at_once(self, run_setup, head, capsys):
        from vtscore.cli_resume import exclusive_run

        out = run_setup / "hits.ndjson"
        with exclusive_run(out), pytest.raises(SystemExit):
            _run(run_setup, out)

        assert "Another --resume run is writing" in _error(capsys)
        assert not out.exists()

    def test_an_unreadable_checkpoint(self, run_setup, head, capsys):
        tmp_path = run_setup
        out = tmp_path / "hits.ndjson"
        _checkpoint(out).write_text("{not json")

        with pytest.raises(SystemExit):
            _run(tmp_path, out)

        assert "cannot be read" in _error(capsys)

    def test_an_exporter_that_cannot_resume(self, run_setup, head, capsys):
        with pytest.raises(SystemExit):
            _run(run_setup, run_setup / "hits.ndjson", exporter="gui")

        err = _error(capsys)
        assert "cannot resume a streamed run" in err
        assert "server_csv_file, server_json_file" in err

    def test_a_dated_output_path(self, run_setup, head, capsys):
        with pytest.raises(SystemExit):
            _run(run_setup, run_setup / "hits_{YYYYMMDD-HHMMSS}.ndjson")

        assert "uses {YYYYMMDD-HHMMSS}" in _error(capsys)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"stream_results": False}, "requires --stream-results"),
            ({"chunk_size": None}, "requires --chunk-size"),
            ({"as_labelset": True}, "cannot resume a labelset stream"),
        ],
    )
    def test_a_run_it_cannot_make_resumable(self, tmp_path, kwargs, message):
        from vtscore.cli import _run_source, _SourceSpec

        spec = _SourceSpec(kind="pickle", dataset_path=str(tmp_path / "ds.pkl"), chunk_size=kwargs.pop("chunk_size", 2))
        options = {"stream_results": True, **kwargs}
        with pytest.raises(ValueError, match=message):
            _run_source(spec, save_dataset=False, dry_run=True, resume=True, **options)

    def test_a_multi_dataset_run(self, tmp_path):
        from vtscore.cli import _run_source, _SourceSpec
        from vtscore.datasets.importers.base import OutputSpec

        spec = _SourceSpec(
            kind="importer",
            importer_name="server_folder",
            chunk_size=2,
            outputs=(OutputSpec(media_type="audio"), OutputSpec(media_type="image")),
        )
        with pytest.raises(ValueError, match="cannot resume a multi-dataset run"):
            _run_source(spec, save_dataset=False, dry_run=True, stream_results=True, resume=True)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


class TestResumeDryRun:
    def test_reports_a_fresh_start(self, run_setup, capsys):
        from vtscore.cli import autodetect_main_chunked

        out = run_setup / "hits.ndjson"
        autodetect_main_chunked(
            str(run_setup / "ds.pkl"),
            2,
            settings_path=str(run_setup / "settings.json"),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(out)},
            dry_run=True,
            stream_results=True,
            resume=True,
        )

        assert f"Resume: yes (checkpoint {_checkpoint(out)}: none yet" in capsys.readouterr().out

    def test_reports_how_far_a_checkpoint_got(self, run_setup, head, scorer, capsys):
        from vtscore.cli import autodetect_main_chunked

        out = run_setup / "hits.ndjson"
        _interrupted(run_setup, out, scorer, at=3)
        capsys.readouterr()
        autodetect_main_chunked(
            str(run_setup / "ds.pkl"),
            2,
            settings_path=str(run_setup / "settings.json"),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(out)},
            dry_run=True,
            stream_results=True,
            resume=True,
        )

        assert "2 chunk(s) already written" in capsys.readouterr().out

    def test_refuses_an_exporter_that_cannot_resume(self, run_setup, capsys):
        from vtscore.cli import autodetect_main_chunked

        with pytest.raises(SystemExit):
            autodetect_main_chunked(
                str(run_setup / "ds.pkl"),
                2,
                settings_path=str(run_setup / "settings.json"),
                exporter_name="webhook",
                exporter_field_values={"url": "https://example.com/hook"},
                dry_run=True,
                stream_results=True,
                resume=True,
            )

        assert "cannot resume a streamed run" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# An importer source: completed chunks are read raw, not embedded
# ---------------------------------------------------------------------------


class _StubImporter(DatasetImporter):
    """Yields fixed chunks of already-embedded medias, as a chunked importer would.

    It overrides only ``run_chunked_cli``, so a resumed run reaches its chunks
    through the base class's ``run_chunked_cli_deferred``: built, with no key.
    """

    name = "stub"
    display_name = "Stub"
    description = "Test"
    fields: list[PluginField] = []

    def __init__(self, medias: list[dict]) -> None:
        super().__init__()
        self.medias = medias

    def run(self, field_values: dict, medias: dict, thin: bool = False) -> None:
        raise NotImplementedError

    def validate_cli_field_values(self, field_values: dict) -> None:
        pass

    def run_chunked_cli(self, field_values: dict, chunk_size: int, thin: bool = False):
        import numpy as np

        from vtscore.embedding.media_vectors import set_media_embedding

        for start in range(0, len(self.medias), chunk_size):
            chunk = {}
            for i, media in enumerate(self.medias[start : start + chunk_size], 1):
                fresh = {k: v for k, v in media.items() if k not in ("embedding", "embedder")}
                set_media_embedding(fresh, "clap", np.asarray(media["embedding"], dtype=np.float32))
                chunk[i] = dict(fresh, id=i)
            yield chunk


class TestResumeFromAnImporter:
    def test_completed_chunks_are_not_embedded_again(self, client, tmp_path, monkeypatch, head, scorer):
        import vtscore.cli as cli_mod

        _write_detector("resume-det")
        _settings_file(tmp_path, "resume-det")
        importer = _StubImporter([_make_audio_media(i) for i in range(1, 10)])
        monkeypatch.setattr("vtscore.datasets.importers.get_importer", lambda _name: importer)
        embedded: list[int] = []
        monkeypatch.setattr(
            cli_mod,
            "_embed_loaded_medias",
            lambda chunk: embedded.append(min(m["id"] for m in chunk.values())) or chunk,
        )

        def run(out: Path) -> None:
            cli_mod.autodetect_importer_main_chunked(
                "stub",
                {"path": "/data"},
                2,
                settings_path=str(tmp_path / "settings.json"),
                exporter_name="server_json_file",
                exporter_field_values={"filepath": str(out)},
                stream_results=True,
                keep_negatives=True,
                resume=True,
            )

        run(tmp_path / "reference.ndjson")
        assert len((tmp_path / "reference.ndjson").read_text().splitlines()) == 10
        out = tmp_path / "hits.ndjson"
        scorer.calls, scorer.at = 0, 3
        with pytest.raises(KeyboardInterrupt):
            run(out)
        scorer.at = None
        embedded.clear()

        run(out)

        # The first chunk (ids 1-2) trains the detectors; 3-4 were written; 5+ are new.
        assert embedded == [1, 5, 7, 9]
        assert out.read_bytes() == (tmp_path / "reference.ndjson").read_bytes()


# ---------------------------------------------------------------------------
# A folder source: completed chunks are walked, not read (#5314)
# ---------------------------------------------------------------------------


class _FolderRun:
    """A ``server_folder`` resumable run over a folder of nine wavs, two per chunk."""

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        import numpy as np

        import vtscore.cli as cli_mod
        import vtscore.datasets.loader_folder as loader_folder
        from tests_lib.helpers import make_wav_file
        from vtscore.embedding.media_vectors import set_media_embedding

        self.tmp_path = tmp_path
        self.folder = tmp_path / "clips"
        self.folder.mkdir()
        for i in range(1, 10):
            make_wav_file(self.folder, f"clip_{i:03d}.wav", frequency=400.0 + 10 * i)
        _write_detector("resume-det")
        _settings_file(tmp_path, "resume-det")

        #: Every file the folder loader builds (reads, hashes, thumbnails), in order.
        self.built: list[str] = []
        real_build = loader_folder._build_per_file_media

        def build(**kwargs: Any) -> dict:
            self.built.append(kwargs["rel_path"])
            return real_build(**kwargs)

        def embed(chunk: dict) -> dict:
            # The pickle fixture's 2-dim embedding, keyed by the file's number.
            for media in chunk.values():
                number = int(media["filename"][5:8])
                set_media_embedding(media, "clap", np.asarray(_make_audio_media(number)["embedding"], np.float32))
            return chunk

        monkeypatch.setattr(loader_folder, "_build_per_file_media", build)
        monkeypatch.setattr(cli_mod, "_embed_loaded_medias", embed)

    def run(self, out: Path) -> None:
        from vtscore.cli import autodetect_importer_main_chunked

        autodetect_importer_main_chunked(
            "server_folder",
            {"path": str(self.folder), "media_type": "audio"},
            2,
            settings_path=str(self.tmp_path / "settings.json"),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(out)},
            stream_results=True,
            keep_negatives=True,
            resume=True,
        )

    def walk(self) -> list[str]:
        """The files in walk order: what an uninterrupted run builds."""
        self.built.clear()
        self.run(self.tmp_path / "reference.ndjson")
        walk = list(self.built)
        self.built.clear()
        return walk


@pytest.fixture
def folder_run(client, tmp_path, monkeypatch, head) -> _FolderRun:
    return _FolderRun(tmp_path, monkeypatch)


class TestResumeFromAFolder:
    def test_completed_chunks_are_walked_without_reading_their_files(self, folder_run, scorer, capsys):
        walk = folder_run.walk()
        out = folder_run.tmp_path / "hits.ndjson"
        scorer.calls, scorer.at = 0, 3
        with pytest.raises(KeyboardInterrupt):
            folder_run.run(out)
        scorer.at = None
        folder_run.built.clear()
        capsys.readouterr()

        folder_run.run(out)

        # Chunk 1 is built to retrain the detectors; chunk 2 was written and is
        # only walked; chunks 3-5 are new.
        assert folder_run.built == walk[:2] + walk[4:]
        assert "Skipping chunk 2 (2 medias): already written; checked without reading its files." in (
            capsys.readouterr().out
        )
        assert out.read_bytes() == (folder_run.tmp_path / "reference.ndjson").read_bytes()

    def test_a_file_removed_from_the_part_already_done_is_refused(self, folder_run, scorer, capsys):
        walk = folder_run.walk()
        out = folder_run.tmp_path / "hits.ndjson"
        scorer.calls, scorer.at = 0, 4
        with pytest.raises(KeyboardInterrupt):
            folder_run.run(out)
        scorer.calls, scorer.at = 0, None
        (folder_run.folder / walk[3]).unlink()
        before = out.read_bytes()

        with pytest.raises(SystemExit):
            folder_run.run(out)

        assert "chunk 2 no longer holds the same media" in _error(capsys)
        assert scorer.calls == 0
        assert out.read_bytes() == before

    def test_a_file_rewritten_in_the_part_already_done_is_refused(self, folder_run, scorer, capsys):
        walk = folder_run.walk()
        out = folder_run.tmp_path / "hits.ndjson"
        scorer.calls, scorer.at = 0, 4
        with pytest.raises(KeyboardInterrupt):
            folder_run.run(out)
        scorer.calls, scorer.at = 0, None
        rewritten = folder_run.folder / walk[5]
        rewritten.write_bytes(rewritten.read_bytes() + b"\x00")

        with pytest.raises(SystemExit):
            folder_run.run(out)

        assert "chunk 3 no longer holds the same media" in _error(capsys)
        assert scorer.calls == 0
