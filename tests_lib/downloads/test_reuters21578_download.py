"""Tests for Reuters-21578 download and load_demo_source integration."""

from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_REUTERS_BLOCK = """<REUTERS NEWID="{newid}">
<TOPICS>{topics}</TOPICS>
<TITLE>{title}</TITLE>
<BODY>{body}</BODY>
</REUTERS>"""


def _topic_tags(topics: list[str]) -> str:
    return "".join(f"<D>{t}</D>" for t in topics)


def _make_sgm(path: Path, entries: list[tuple[list[str], str, str]]) -> None:
    blocks = [
        _REUTERS_BLOCK.format(newid=i, topics=_topic_tags(t), title=title, body=body)
        for i, (t, title, body) in enumerate(entries, start=1)
    ]
    path.write_text("\n".join(blocks), encoding="latin-1")


# ---------------------------------------------------------------------------
# download_reuters21578
# ---------------------------------------------------------------------------


class TestDownloadReuters21578:
    def test_returns_articles_by_topic(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        extract_dir = tmp_path / "reuters21578"
        extract_dir.mkdir()
        _make_sgm(
            extract_dir / "reut2-000.sgm",
            [
                (["earn"], "Q1 PROFIT UP", "Company reported strong earnings. Reuter"),
                (["acq"], "BIGCO BUYS SMALLCO", "Acquisition for $1bn. Reuter"),
                (["grain", "wheat"], "WHEAT PRICES RISE", "Grain markets rallied. Reuter"),
            ],
        )

        with patch.object(dl_module.core, "DATA_DIR", tmp_path):
            result = dl_module.download_reuters21578(on_progress=lambda *a: None)

        assert "earn" in result
        assert "acq" in result
        # Multi-topic story counted under both topics.
        assert "grain" in result and "wheat" in result
        assert any("WHEAT PRICES RISE" in s for s in result["wheat"])
        # Reuter trailer is stripped.
        assert not any(s.endswith("Reuter") for s in result["earn"])

    def test_cached_extraction_skips_download(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        extract_dir = tmp_path / "reuters21578"
        extract_dir.mkdir()
        _make_sgm(
            extract_dir / "reut2-000.sgm",
            [(["earn"], "TITLE", "A short story. Reuter")],
        )

        download_called = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = dl_module.download_reuters21578(on_progress=lambda *a: None)

        assert not download_called
        assert "earn" in result

    def test_skips_blocks_without_topics_or_body(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        extract_dir = tmp_path / "reuters21578"
        extract_dir.mkdir()
        sgm_text = """<REUTERS NEWID="1">
<TOPICS></TOPICS>
<BODY>No topics here. Reuter</BODY>
</REUTERS>
<REUTERS NEWID="2">
<TOPICS><D>earn</D></TOPICS>
</REUTERS>
<REUTERS NEWID="3">
<TOPICS><D>earn</D></TOPICS>
<BODY>Real story body. Reuter</BODY>
</REUTERS>
"""
        (extract_dir / "reut2-000.sgm").write_text(sgm_text, encoding="latin-1")

        with patch.object(dl_module.core, "DATA_DIR", tmp_path):
            result = dl_module.download_reuters21578(on_progress=lambda *a: None)

        # Only the third block survives.
        assert sum(len(v) for v in result.values()) == 1
        assert "Real story body" in result["earn"][0]


# ---------------------------------------------------------------------------
# load_demo_source: reuters21578 branch
# ---------------------------------------------------------------------------


class TestLoadDemoSourceReuters:
    def test_reuters_source_populates_clips(self):
        from vtscore.datasets import downloader as dl_module
        from tests_lib.downloads._helpers import make_text_embedder_stub, make_text_media_type_stub

        fake_articles = {
            "earn": ["Earn story one.", "Earn story two."],
            "acq": ["Acq story one.", "Acq story two."],
        }

        mt = make_text_media_type_stub()
        emb = make_text_embedder_stub()
        clips: dict = {}

        with patch.object(dl_module, "download_reuters21578", return_value=fake_articles):
            mt.load_demo_source(
                source="reuters21578",
                categories=["earn", "acq"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=emb,
            )

        assert len(clips) == 4
        assert {c["category"] for c in clips.values()} == {"earn", "acq"}

    def test_reuters_slice_is_applied(self):
        from vtscore.datasets import downloader as dl_module
        from tests_lib.downloads._helpers import make_text_embedder_stub, make_text_media_type_stub

        fake_articles = {"grain": [f"Grain story {i}." for i in range(10)]}

        mt = make_text_media_type_stub()
        emb = make_text_embedder_stub()
        clips: dict = {}

        with patch.object(dl_module, "download_reuters21578", return_value=fake_articles):
            mt.load_demo_source(
                source="reuters21578",
                categories=["grain"],
                slice_start=2,
                slice_end=5,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=emb,
            )

        assert len(clips) == 3
