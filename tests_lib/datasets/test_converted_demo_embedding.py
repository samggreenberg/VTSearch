"""Library-tier tests: a converted demo load does not embed its source media.

A converter replaces every loaded media with its outputs, which leave
unembedded for the framework embed stage. Embedding the sources first was a
whole wasted pass: an image→face demo ran FaceNet over every full photo, and
that pass filled most of the progress bar before any conversion began, so the
real work crawled along its last few percent (#5176).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from vtscore.datasets import loader_demo as _loader_demo
from vtscore.datasets.loader import load_demo_dataset
from vtscore.media import get as media_get

#: A real image-typed demo id; its source is never touched (load_demo_source is
#: stubbed), only its registry entry is read.
DEMO_ID = "caltech101_s"


def _load(tmp_path, monkeypatch, **kwargs) -> dict:
    """Run ``load_demo_dataset`` with the source loader stubbed; return its kwargs."""
    monkeypatch.setattr(_loader_demo, "EMBEDDINGS_DIR", tmp_path)
    seen: dict = {}

    def fake_source(*, clips, **source_kwargs):
        seen.update(source_kwargs)
        clips[1] = {"id": 1, "media_type": "image", "embeddings": {}, "filename": "a.png", "category": "c"}

    with (
        patch.object(media_get("image"), "load_demo_source", side_effect=fake_source),
        patch("vtscore.converters.runner.apply_converter_to_demo") as convert,
    ):
        load_demo_dataset(DEMO_ID, {}, on_progress=lambda *a, **k: None, use_cache=False, **kwargs)
    seen["converted"] = convert.called
    return seen


def test_converted_demo_skips_source_embedding(tmp_path, monkeypatch):
    seen = _load(tmp_path, monkeypatch, converter_name="image2face")
    assert seen["converted"]
    assert seen["skip_embedding"] is True


@pytest.mark.parametrize("clipper", ["", "image_default"])
def test_plain_demo_still_embeds_its_media(tmp_path, monkeypatch, clipper):
    seen = _load(tmp_path, monkeypatch, clipper_name=clipper)
    assert not seen["converted"]
    assert seen["skip_embedding"] is False
