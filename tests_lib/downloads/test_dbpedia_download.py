"""Tests for DBpedia-14 (Wikipedia ontology) download and load_demo_source integration."""

import io
import tarfile
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dbpedia_tgz(tmp_path: Path, articles_per_class: int = 3) -> Path:
    """Build a minimal dbpedia_csv.tgz containing train.csv with 14 classes."""
    train_lines: list[str] = []
    test_lines: list[str] = []
    for class_idx in range(1, 15):
        for i in range(1, articles_per_class + 1):
            title = f"Title {class_idx}-{i}"
            abstract = f"Class {class_idx} abstract number {i}."
            train_lines.append(f'"{class_idx}","{title}","{abstract}"')
        # Half a doc in the test split so train+test both flow through.
        test_lines.append(f'"{class_idx}","Test {class_idx}","Test abstract for class {class_idx}."')

    archive_path = tmp_path / "dbpedia_csv.tgz"
    with tarfile.open(archive_path, "w:gz") as tar:

        def _add_text(name: str, body: str) -> None:
            data = body.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        _add_text("dbpedia_csv/train.csv", "\n".join(train_lines) + "\n")
        _add_text("dbpedia_csv/test.csv", "\n".join(test_lines) + "\n")
    return archive_path


# ---------------------------------------------------------------------------
# download_dbpedia
# ---------------------------------------------------------------------------


class TestDownloadDbpedia:
    def test_returns_articles_by_ontology_class(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        archive_path = _make_dbpedia_tgz(tmp_path, articles_per_class=3)

        def fake_download(url, dest, size, cb):
            # The downloader writes to a temp path; copy our fixture there.
            archive_path.replace(dest)

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_with_progress", fake_download),
        ):
            result = dl_module.download_dbpedia(on_progress=lambda *a: None)

        assert "Company" in result
        assert "WrittenWork" in result
        # 3 train + 1 test per class.
        assert len(result["Company"]) == 4
        # All values are strings combining title + abstract.
        sample = result["Company"][0]
        assert isinstance(sample, str) and sample

    def test_cached_extraction_skips_download(self, tmp_path):
        """If the extract dir exists, no download is triggered."""
        from vtscore.datasets import downloader as dl_module

        extract_dir = tmp_path / "dbpedia_csv"
        extract_dir.mkdir()
        (extract_dir / "train.csv").write_text('"1","Hi","An animal lives."\n', encoding="utf-8")

        download_called = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = dl_module.download_dbpedia(on_progress=lambda *a: None)

        assert not download_called
        assert "Company" in result

    def test_uses_classes_txt_when_present(self, tmp_path):
        """A classes.txt sidecar overrides the hardcoded ontology names."""
        from vtscore.datasets import downloader as dl_module

        extract_dir = tmp_path / "dbpedia_csv"
        extract_dir.mkdir()
        (extract_dir / "classes.txt").write_text("Alpha\nBeta\n", encoding="utf-8")
        (extract_dir / "train.csv").write_text('"1","First","Hello."\n"2","Second","World."\n', encoding="utf-8")

        with patch.object(dl_module.core, "DATA_DIR", tmp_path):
            result = dl_module.download_dbpedia(on_progress=lambda *a: None)

        assert set(result.keys()) == {"Alpha", "Beta"}

    def test_skips_malformed_rows(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        extract_dir = tmp_path / "dbpedia_csv"
        extract_dir.mkdir()
        (extract_dir / "train.csv").write_text(
            '"1","Good","An abstract."\n"not-a-number","X","Y"\n"99","X","Y"\n',
            encoding="utf-8",
        )

        with patch.object(dl_module.core, "DATA_DIR", tmp_path):
            result = dl_module.download_dbpedia(on_progress=lambda *a: None)

        # Only the valid row survives.
        assert sum(len(v) for v in result.values()) == 1


# ---------------------------------------------------------------------------
# load_demo_source - dbpedia branch
# ---------------------------------------------------------------------------


class TestLoadDemoSourceDbpedia:
    def test_dbpedia_source_populates_clips(self):
        from vtscore.datasets import downloader as dl_module
        from tests_lib.downloads._helpers import make_text_embedder_stub, make_text_media_type_stub

        fake_articles = {
            "Company": ["Apple article.", "Google article."],
            "Animal": ["Dog article.", "Cat article."],
        }

        mt = make_text_media_type_stub()
        emb = make_text_embedder_stub()
        clips: dict = {}

        with patch.object(dl_module, "download_dbpedia", return_value=fake_articles):
            mt.load_demo_source(
                source="dbpedia",
                categories=["Company", "Animal"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=emb,
            )

        assert len(clips) == 4
        assert {c["category"] for c in clips.values()} == {"Company", "Animal"}

    def test_dbpedia_slice_is_applied(self):
        from vtscore.datasets import downloader as dl_module
        from tests_lib.downloads._helpers import make_text_embedder_stub, make_text_media_type_stub

        fake_articles = {"Plant": [f"Plant abstract {i}." for i in range(10)]}

        mt = make_text_media_type_stub()
        emb = make_text_embedder_stub()
        clips: dict = {}

        with patch.object(dl_module, "download_dbpedia", return_value=fake_articles):
            mt.load_demo_source(
                source="dbpedia",
                categories=["Plant"],
                slice_start=2,
                slice_end=5,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=emb,
            )

        assert len(clips) == 3
