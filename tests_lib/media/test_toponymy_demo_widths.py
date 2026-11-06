"""The Synthetic World Map demos import under every embedder the picker offers (#5508).

Both demos ship synthetic vectors filed under the load's embedder name, and the
load pipeline refuses a named vector whose width differs from the one that
embedder declares.  They used to be baked at a fixed width (768 for image, 512
for audio), so picking any other embedder made the import fail once it was
running.  They are now baked at the chosen embedder's declared width.
"""

from __future__ import annotations

import pytest

import vtscore.media._toponymy_demo as toponymy_demo
from vtscore.datasets.stages.embedding import _check_named_vector_widths
from vtscore.embedding.binding import expected_dim_for_embedder
from vtscore.media import embedders_for_type
from vtscore.media import get as media_get
from vtscore.media._toponymy_demo import SOURCE_ID, TAXONOMY, baked_dim


@pytest.fixture
def one_item_per_city(monkeypatch):
    """Generate one item per city (108 in all) rather than 12, to keep the loads quick."""
    real = toponymy_demo.generate_items
    monkeypatch.setattr(toponymy_demo, "generate_items", lambda dim, items_per_city=1: real(dim, 1))


def _load(media_type: str, embedder) -> dict[int, dict]:
    clips: dict[int, dict] = {}
    media_get(media_type).load_demo_source(
        source=SOURCE_ID,
        categories=list(TAXONOMY),
        slice_start=0,
        slice_end=None,
        clips=clips,
        on_progress=lambda *_args: None,
        embedder=embedder,
    )
    return clips


@pytest.mark.parametrize("media_type", ["image", "audio"])
def test_every_embedder_gets_vectors_of_its_declared_width(media_type, one_item_per_city):
    embedders = embedders_for_type(media_type)
    # The bug needs at least two declared widths to show; both types have them.
    assert len({expected_dim_for_embedder(e.name) for e in embedders}) > 1
    for embedder in embedders:
        clips = _load(media_type, embedder)
        assert len(clips) == 108
        # The check the import failed on, run over this demo's own output.
        _check_named_vector_widths(clips)
        expected = expected_dim_for_embedder(embedder.name)
        if expected is not None:
            assert clips[1]["embeddings"][embedder.name].shape == (expected,), embedder.name


def test_image_demo_bakes_at_sift_vlad_width(one_item_per_city):
    # The issue's repro: an 8192-D embedder, against the old fixed 768.
    sift_vlad = next(e for e in embedders_for_type("image") if e.name == "sift_vlad")
    clips = _load("image", sift_vlad)
    assert clips[1]["embeddings"]["sift_vlad"].shape == (8192,)


@pytest.mark.parametrize(("name", "fallback"), [("", 768), ("no_such_embedder", 512)])
def test_baked_dim_falls_back_when_no_width_is_declared(name, fallback):
    assert baked_dim(name, fallback=fallback) == fallback


def test_baked_dim_reads_the_declared_width():
    assert baked_dim("siglip_l", fallback=768) == expected_dim_for_embedder("siglip_l") == 1152
