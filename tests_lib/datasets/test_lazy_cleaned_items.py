"""A reference (thin) import keeps only a recipe for the items a cleaner rewrote.

A cleaned unit's recipe is its whole chain trail, replayed from the source and
served only while it still reproduces the content hash the trail recorded (see
:mod:`vtscore.media.clipping.lazy_clip`).  These tests import the same folder thin and
full and require the two to agree on every observable: the bytes, the stored
MD5, the embedding, and the Original.  They also pin the failure mode: a source
edited after import serves *nothing*, never the edited file.

The same holds for a converter output (a PDF's text, a rendered page) that a
cleaner rewrote or a clipper cut (#5041): its trail is led by the converter, so
the replay starts from the PDF.
"""

from __future__ import annotations

import hashlib
import io
import os
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vtscore.media as media_registry
from vtscore.converters.runner import run_converters_on_folder
from vtscore.datasets.clipper_chain import has_original_payload, original_hash, parse_trail
from vtscore.datasets.loader_folder import load_dataset_from_folder
from vtscore.datasets.stages import clipper as clipper_stage
from vtscore.datasets.stages.embedding import embed_missing
from vtscore.embedding.media_vectors import media_embedding
from vtscore.media.clipping.cleaner import MediaCleaner
from vtscore.media.embedder import MediaEmbedder
from vtscore.media.clipping.lazy_clip import clip_recipe, embed_view, lazy_original_bytes, recipe_is_exclusive
from vtscore.state import DatasetContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _payload_like_an_embedder(media: dict[str, Any]) -> bytes | None:
    """Read a media's content the way every shipped embedder does: payload, then path."""
    if media.get("media_type") == "text":
        text = media.get("media_string")
        if isinstance(text, str):
            return text.encode("utf-8")
    else:
        blob = media.get("media_bytes")
        if isinstance(blob, (bytes, bytearray)):
            return bytes(blob)
    path = media.get("media_path")
    return Path(path).read_bytes() if path else None


class _ContentEmbedder(MediaEmbedder):
    """A vector that is a pure function of the content it is handed."""

    NAME = "content_test_embedder"

    def __init__(self, media_type: str, name: str = NAME) -> None:
        super().__init__()
        self._media_type = media_type
        self._name = name
        self._model = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def media_type_id(self) -> str:
        return self._media_type

    def _load_models_impl(self) -> None:  # pragma: no cover - model is pre-set
        self._model = True

    def _embed_media_impl(self, media: dict) -> np.ndarray | None:
        payload = _payload_like_an_embedder(media)
        if payload is None:
            return None
        return np.frombuffer(hashlib.md5(payload).digest(), dtype=np.uint8).astype(np.float32)  # noqa: S324

    def _embed_media_bulk_impl(self, medias: list[dict]) -> list[np.ndarray | None]:
        return [self._embed_media_impl(m) for m in medias]


def _wav(samples: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.astype(np.int16).tobytes())
    return buf.getvalue()


def _padded_tone(freq: float = 440.0, sr: int = 16000) -> bytes:
    """One second of silence, one of tone, one of silence: the silence trim fires."""
    silence = np.zeros(sr)
    tone = np.sin(2 * np.pi * freq * np.arange(sr) / sr) * 10000
    return _wav(np.concatenate([silence, tone, silence]), sr)


def _bordered_png(seed: int = 42) -> bytes:
    """A patterned square inside a flat white border: the edge trim fires."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    pixels = np.full((64, 64, 3), 255, dtype=np.uint8)
    pixels[16:48, 16:48] = rng.integers(0, 200, size=(32, 32, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")
    return buf.getvalue()


#: media type -> (file name, file bytes, cleaner that fires on them)
_CASES: dict[str, tuple[str, bytes, str]] = {
    "text": ("doc.txt", "hello   world\n\n\n\nsecond   para".encode("utf-8"), "text_whitespace"),
    "audio": ("tone.wav", _padded_tone(), "audio_silence_trim"),
    "image": ("photo.png", _bordered_png(), "image_edge_trim"),
}

#: The same files after a user edited them: different content the cleaner still fires on.
_EDITED: dict[str, bytes] = {
    "text": "hello   there\n\n\n\nsecond   para".encode("utf-8"),
    "audio": _padded_tone(880.0),
    "image": _bordered_png(seed=7),
}


def _import(
    folder: Path,
    media_type: str,
    cleaner: str,
    *,
    thin: bool,
    clipper: str | None = None,
    clipper_params: dict | None = None,
) -> dict[int, dict[str, Any]]:
    """Run the load stages that matter here: import, chain + re-embed, re-lazify."""
    medias: dict[int, dict[str, Any]] = {}
    load_dataset_from_folder(folder, media_type, medias, thin=thin)
    clipper_stage._apply_clipper(
        medias,
        clipper or f"{media_type}_default",
        clipper_params,
        chain_steps=[{"kind": "cleaner", "name": cleaner, "params": {}}],
        embedder=_ContentEmbedder(media_type),
    )
    ctx = DatasetContext("_test_lazy_cleaned")
    ctx.medias = medias
    clipper_stage._relazify_reference_clips_stage(ctx)
    return medias


def _served(media: dict[str, Any]) -> bytes | None:
    """What the byte / text routes serve for *media*."""
    mt = media_registry.get(media["media_type"])
    if media["media_type"] == "text":
        text = mt._resolve_media_string(media)
        return text.encode("utf-8") if text else None
    return mt._resolve_media_bytes(media)


def _stored_original(media: dict[str, Any]) -> bytes | None:
    if media["media_type"] == "text":
        text = media.get("original_media_string")
        return text.encode("utf-8") if text is not None else None
    return media.get("original_media_bytes")


@pytest.fixture
def folder_for(tmp_path):
    """Write one case's file into its own folder and return ``(folder, path)``."""

    def make(media_type: str) -> tuple[Path, Path]:
        name, data, _cleaner = _CASES[media_type]
        folder = tmp_path / media_type
        folder.mkdir()
        path = folder / name
        path.write_bytes(data)
        return folder, path

    return make


# ---------------------------------------------------------------------------
# A thin import matches a full one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("media_type", sorted(_CASES))
class TestThinMatchesFull:
    def test_cleaned_item_stores_only_a_recipe(self, media_type, folder_for):
        folder, path = folder_for(media_type)
        (item,) = _import(folder, media_type, _CASES[media_type][2], thin=True).values()

        assert has_original_payload(item), "the cleaner must actually fire for this test to mean anything"
        assert item["media_bytes"] is None
        assert item["media_string"] is None
        assert not [k for k in item if k.startswith("original_media_")]
        assert item["media_path"] == str(path)
        assert recipe_is_exclusive(item)

    def test_bytes_md5_embedding_and_original_match_a_full_import(self, media_type, folder_for):
        folder, _path = folder_for(media_type)
        cleaner = _CASES[media_type][2]
        (full,) = _import(folder, media_type, cleaner, thin=False).values()
        (thin,) = _import(folder, media_type, cleaner, thin=True).values()

        assert _served(thin) == _served(full)
        assert thin["md5"] == full["md5"]
        name = _ContentEmbedder.NAME
        np.testing.assert_array_equal(media_embedding(thin, name), media_embedding(full, name))
        assert lazy_original_bytes(thin) == _stored_original(full)
        assert lazy_original_bytes(thin) != _served(thin)

    def test_an_edited_source_serves_nothing(self, media_type, folder_for):
        """Never the edited file in place of the cleaned item, nor its Original."""
        folder, path = folder_for(media_type)
        (thin,) = _import(folder, media_type, _CASES[media_type][2], thin=True).values()
        assert _served(thin) is not None  # warms the cache: the edit must still be noticed

        before = path.stat()
        path.write_bytes(_EDITED[media_type])
        # Some edits keep the size, and a coarse filesystem clock can keep the
        # mtime too; push the mtime on so the test never depends on either.
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))

        assert _served(thin) is None
        assert lazy_original_bytes(thin) is None

    def test_a_later_reembed_sees_the_cleaned_payload(self, media_type, folder_for, monkeypatch):
        """A detector with another embedder re-embeds the dataset later
        (``embed_missing``); ``media_path`` names the uncleaned source, so the
        embedder must be handed the replay, not the file."""
        folder, _path = folder_for(media_type)
        (full,) = _import(folder, media_type, _CASES[media_type][2], thin=False).values()
        (thin,) = _import(folder, media_type, _CASES[media_type][2], thin=True).values()
        later = _ContentEmbedder(media_type, name="content_test_embedder_later")
        monkeypatch.setitem(media_registry._embedder_registry, later.name, later)

        embed_missing({1: full, 2: thin}, later.name)

        np.testing.assert_array_equal(media_embedding(thin, later.name), media_embedding(full, later.name))


class TestLoudnessNormalizedThinImport:
    def test_a_normalized_clip_keeps_only_a_recipe_that_replays_it_exactly(self, tmp_path):
        """A gain rewrites every sample, so the recipe holds only if the cleaner is
        deterministic: the dry run must rebuild the normalized bytes exactly."""
        sr = 16000
        quiet = np.sin(2 * np.pi * 440.0 * np.arange(sr) / sr) * 300  # about -41 dBFS
        folder = tmp_path / "audio"
        folder.mkdir()
        path = folder / "quiet.wav"
        path.write_bytes(_wav(quiet, sr))

        (full,) = _import(folder, "audio", "audio_loudness_normalize", thin=False).values()
        (thin,) = _import(folder, "audio", "audio_loudness_normalize", thin=True).values()

        assert has_original_payload(thin)
        assert thin["media_bytes"] is None and recipe_is_exclusive(thin)
        assert _served(thin) == _served(full) != path.read_bytes()
        assert thin["md5"] == full["md5"]
        assert lazy_original_bytes(thin) == _stored_original(full) == path.read_bytes()


class TestEmbedView:
    def test_inline_and_whole_file_media_are_handed_over_as_is(self, tmp_path):
        src = tmp_path / "a.txt"
        src.write_text("hello", encoding="utf-8")
        inline = {"media_type": "text", "media_string": "hello", "origin": {"params": {}}}
        whole_file = {"media_type": "text", "media_string": None, "media_path": str(src), "origin": {"params": {}}}
        assert embed_view(inline) is inline
        assert embed_view(whole_file) is whole_file

    def test_a_view_never_exposes_the_source_path(self, folder_for):
        folder, _path = folder_for("image")
        (thin,) = _import(folder, "image", _CASES["image"][2], thin=True).values()
        view = embed_view(thin)
        assert view is not thin
        assert view.get("media_path") is None
        assert view.get("media_bytes") == _served(thin)
        # A plugin that copies the view loses the payload, and with it any way
        # to reach the source: it embeds nothing rather than the wrong file.
        copied = dict(view)
        assert _payload_like_an_embedder(copied) is None


# ---------------------------------------------------------------------------
# Sub-clips, text clips, and the dry run that gates re-lazification
# ---------------------------------------------------------------------------


class TestCleanedSubClips:
    def test_a_cleaned_tile_replays_its_tile_and_its_original_is_the_tile(self, folder_for):
        """The Original of a cleaned tile is the pre-clean tile, not the recording."""
        folder, path = folder_for("audio")
        tiling = {"duration": 1.5, "min_overlap": 0.0}
        full = _import(folder, "audio", "audio_silence_trim", thin=False, clipper="sound_tiling", clipper_params=tiling)
        thin = _import(folder, "audio", "audio_silence_trim", thin=True, clipper="sound_tiling", clipper_params=tiling)
        assert len(thin) == len(full) > 1

        cleaned_tiles = 0
        for f, t in zip(full.values(), thin.values()):
            assert _served(t) == _served(f)
            assert t["md5"] == f["md5"]
            if has_original_payload(f):
                cleaned_tiles += 1
                assert t["media_bytes"] is None
                original = lazy_original_bytes(t)
                assert original == _stored_original(f)
                assert original != path.read_bytes()
        assert cleaned_tiles >= 2, "the trim must fire on more than one tile to exercise the sibling memo"

    def test_siblings_share_one_tiling_of_their_source(self, folder_for, monkeypatch):
        """The dry run replays every cleaned tile; without the memo each replay
        would re-tile the whole recording, quadratic in the tile count."""
        from vtscore.datasets import clipper_chain  # noqa: PLC0415

        runs: list[str] = []
        real = clipper_chain._run_recorded_step

        def counting(index, entry, media):
            runs.append(entry["name"])
            return real(index, entry, media)

        monkeypatch.setattr(clipper_chain, "_run_recorded_step", counting)
        folder, _path = folder_for("audio")
        thin = _import(
            folder,
            "audio",
            "audio_silence_trim",
            thin=True,
            clipper="sound_tiling",
            clipper_params={"duration": 0.75, "min_overlap": 0.0},
        )
        lazy = [m for m in thin.values() if m["media_bytes"] is None and recipe_is_exclusive(m)]
        assert len(lazy) >= 2
        assert runs == ["sound_tiling"]

    def test_the_trail_records_the_original_hash_once(self, folder_for):
        folder, _path = folder_for("text")
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(folder, "text", medias, thin=False)
        clipper_stage._apply_clipper(
            medias,
            "text_default",
            None,
            chain_steps=[
                {"kind": "cleaner", "name": "text_whitespace", "params": {}},
                {"kind": "cleaner", "name": "text_markup_strip", "params": {}},
            ],
            embedder=_ContentEmbedder("text"),
        )
        (item,) = medias.values()
        trail = parse_trail(item["origin"]["params"]["clipper_chain"])
        assert trail is not None
        from vtscore.datasets.clipper_chain import _content_hash  # noqa: PLC0415

        assert [e.get("original_hash") is not None for e in trail] == [True, False]
        assert original_hash(trail) == _content_hash({"media_string": item["original_media_string"]})


class TestThinTextChains:
    def test_a_thin_text_clipper_cuts_like_a_full_one(self, tmp_path):
        """Thin text parents used to reach the chain with no payload, so the
        clipper silently no-opped while the trail said it ran."""
        (tmp_path / "doc.txt").write_text("First one.   Second one here.", encoding="utf-8")
        full = _import(tmp_path, "text", "text_whitespace", thin=False, clipper="text_sentence")
        thin = _import(tmp_path, "text", "text_whitespace", thin=True, clipper="text_sentence")

        assert len(thin) == len(full) == 2
        for f, t in zip(full.values(), thin.values()):
            assert _served(t) == _served(f)
            assert t["md5"] == f["md5"]

    def test_uncut_uncleaned_text_goes_back_to_its_file(self, tmp_path):
        (tmp_path / "doc.txt").write_text("already clean", encoding="utf-8")
        (item,) = _import(tmp_path, "text", "text_whitespace", thin=True).values()
        assert item["media_string"] is None
        assert item["media_path"] == str(tmp_path / "doc.txt")
        assert _served(item) == b"already clean"

    def test_an_uncleaned_text_clip_stays_materialized(self, tmp_path):
        """It has no slice recipe, and its ``media_path`` is the whole document."""
        (tmp_path / "doc.txt").write_text("First one. Second one.", encoding="utf-8")
        thin = _import(tmp_path, "text", "text_whitespace", thin=True, clipper="text_sentence")
        assert sorted(m["media_string"] for m in thin.values()) == ["First one.", "Second one."]


class _CountingTextCleaner(MediaCleaner):
    """A gate whose output differs on every run: no replay can ever reproduce it."""

    def __init__(self) -> None:
        self.runs = 0

    @property
    def name(self) -> str:
        return "text_test_counting"

    @property
    def media_type(self) -> str:
        return "text"

    @property
    def description(self) -> str:
        return "Append a run counter."

    def clean(self, media: dict[str, Any]) -> dict[str, Any]:
        self.runs += 1
        return dict(media, media_string=f"{media.get('media_string')} #{self.runs}")


class TestDryRunGate:
    def test_an_irreproducible_clean_keeps_both_payloads(self, tmp_path, monkeypatch):
        """If the replay cannot rebuild the cleaned payload, dropping it would
        lose the only copy: the item stays as every cleaned item used to."""
        counting = _CountingTextCleaner()
        monkeypatch.setitem(media_registry._cleaner_registry, counting.name, counting)
        (tmp_path / "doc.txt").write_text("hello", encoding="utf-8")

        (item,) = _import(tmp_path, "text", counting.name, thin=True).values()

        assert item["media_string"] == "hello #1"
        assert item["original_media_string"] == "hello"
        assert counting.runs == 2  # the import, then the dry run that failed


# ---------------------------------------------------------------------------
# Converter outputs a cleaner rewrote or a clipper cut (#5041)
# ---------------------------------------------------------------------------


def _pdf(*lines: str, height: int = 300) -> bytes:
    """A one-page PDF with one line of text per 200pt band, so the tiles of its render differ."""
    fitz = pytest.importorskip("fitz")  # PyMuPDF, from the ``agpl`` extra
    doc = fitz.open()
    page = doc.new_page(width=200, height=height)
    for i, line in enumerate(lines):
        page.insert_text((20, 100 + 200 * i), line)
    data = doc.tobytes()
    doc.close()
    return data


def _step(kind: str, name: str) -> dict[str, Any]:
    return {"kind": kind, "name": name, "params": {}}


#: case -> (target type, converter, chain steps, PDF lines, page height)
_CONVERTED: dict[str, tuple[str, str, list[dict[str, Any]], tuple[str, ...], int]] = {
    "cleaned text": ("text", "document2text", [_step("cleaner", "text_whitespace")], ("hello      world",), 300),
    "sentences": ("text", "document2text", [_step("clipper", "text_sentence")], ("One here. Two there.",), 300),
    "cleaned sentences": (
        "text",
        "document2text",
        [_step("clipper", "text_sentence"), _step("cleaner", "text_whitespace")],
        ("One   here. Two   there.",),
        300,
    ),
    "page tiles": ("image", "document2image", [_step("clipper", "image_tiling")], ("top", "middle", "bottom"), 600),
}

#: The cases whose units carry an Original.
_CLEANED_CASES = {"cleaned text", "cleaned sentences"}


def _import_converted(folder: Path, case: str, *, thin: bool) -> dict[int, dict[str, Any]]:
    """Convert every PDF in *folder* the way *case* says, run its chain, re-lazify."""
    target, converter, steps, _lines, _height = _CONVERTED[case]
    medias: dict[int, dict[str, Any]] = {}
    run_converters_on_folder(
        folder, target, medias, thin=thin, converter_specs=[{"converter": converter, "params": {}}]
    )
    clipper_stage._apply_clipper(
        medias, f"{target}_default", None, chain_steps=steps, embedder=_ContentEmbedder(target)
    )
    ctx = DatasetContext("_test_lazy_converted")
    ctx.medias = medias
    clipper_stage._relazify_reference_clips_stage(ctx)
    return medias


@pytest.fixture
def pdf_for(tmp_path):
    """Write one case's PDF (or *lines* in its place) and return ``(folder, path)``."""

    def make(case: str, lines: tuple[str, ...] | None = None) -> tuple[Path, Path]:
        _target, _converter, _steps, default_lines, height = _CONVERTED[case]
        path = tmp_path / "doc.pdf"
        path.write_bytes(_pdf(*(lines or default_lines), height=height))
        return tmp_path, path

    return make


@pytest.mark.parametrize("case", sorted(_CONVERTED))
class TestConvertedThinMatchesFull:
    def test_each_unit_stores_only_a_recipe(self, case, pdf_for):
        folder, path = pdf_for(case)
        thin = _import_converted(folder, case, thin=True)

        assert len(thin) > (1 if _CONVERTED[case][2][0]["kind"] == "clipper" else 0)
        for item in thin.values():
            assert item["media_bytes"] is None
            assert item["media_string"] is None
            assert not [k for k in item if k.startswith("original_media_")]
            assert item["media_path"] == str(path.resolve())
            recipe = clip_recipe(item)
            assert recipe is not None and recipe[0] == "chain"
            assert recipe_is_exclusive(item)

    def test_bytes_md5_embedding_and_original_match_a_full_import(self, case, pdf_for):
        folder, _path = pdf_for(case)
        full = _import_converted(folder, case, thin=False)
        thin = _import_converted(folder, case, thin=True)
        assert len(thin) == len(full)

        name = _ContentEmbedder.NAME
        originals = 0
        for f, t in zip(full.values(), thin.values()):
            assert _served(t) == _served(f)
            assert t["md5"] == f["md5"]
            np.testing.assert_array_equal(media_embedding(t, name), media_embedding(f, name))
            assert has_original_payload(t) == has_original_payload(f)
            if has_original_payload(f):
                originals += 1
                assert lazy_original_bytes(t) == _stored_original(f)
                assert lazy_original_bytes(t) != _served(t)
        assert (originals > 0) == (case in _CLEANED_CASES), "the cleaner must fire exactly where the case says"

    def test_an_edited_pdf_serves_nothing_rather_than_other_content(self, case, pdf_for):
        folder, path = pdf_for(case)
        thin = _import_converted(folder, case, thin=True)
        before = {i: _served(m) for i, m in thin.items()}
        assert all(before.values())  # warms the caches: the edit must still be noticed

        stat = path.stat()
        path.write_bytes(_pdf(*(f"edited {line}" for line in _CONVERTED[case][3]), height=_CONVERTED[case][4]))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

        after = {i: _served(m) for i, m in thin.items()}
        assert any(served is None for served in after.values())
        # A unit the edit happened to leave byte-identical may still serve; nothing serves other bytes.
        assert all(after[i] in (None, before[i]) for i in thin)
        assert all(lazy_original_bytes(m) is None for m in thin.values())

    def test_a_label_on_a_unit_embeds_what_the_dataset_holds(self, case, pdf_for, monkeypatch):
        """Cross-dataset training resolves a label to the PDF and replays the
        same converter-led trail; replaying the chain alone fed it the PDF."""
        from vtscore.detectors import resolver  # noqa: PLC0415

        embedded: list[bytes] = []

        def capture(path, media_type, embedder_name="", **_kwargs):
            embedded.append(Path(path).read_bytes())
            return np.ones(4, dtype=np.float32)

        monkeypatch.setattr(resolver, "embed_file", capture)
        folder, path = pdf_for(case)
        for item in _import_converted(folder, case, thin=False).values():
            embedded.clear()
            assert resolver._embed_resolved_label(path, item["media_type"], item["origin"]) is not None
            assert embedded == [_served(item)]


class TestConvertedOutputsNoStepChanged:
    def test_an_unchanged_output_keeps_the_plain_converter_recipe(self, pdf_for):
        folder, _path = pdf_for("cleaned text", lines=("already clean",))
        (item,) = _import_converted(folder, "cleaned text", thin=True).values()

        assert not has_original_payload(item)
        assert item["media_string"] is None
        recipe = clip_recipe(item)
        assert recipe is not None and recipe[0] == "converter"
        assert _served(item) == b"already clean"

    def test_a_later_reembed_sees_the_cleaned_text(self, pdf_for, monkeypatch):
        folder, _path = pdf_for("cleaned text")
        (full,) = _import_converted(folder, "cleaned text", thin=False).values()
        (thin,) = _import_converted(folder, "cleaned text", thin=True).values()
        later = _ContentEmbedder("text", name="content_test_embedder_later")
        monkeypatch.setitem(media_registry._embedder_registry, later.name, later)

        embed_missing({1: full, 2: thin}, later.name)

        np.testing.assert_array_equal(media_embedding(thin, later.name), media_embedding(full, later.name))
