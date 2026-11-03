"""The Synthetic World Map demos keep every item, not one per city (#5515).

Every item in a leaf city renders from that city's colour (image) or pitch
(audio).  The renderers used to read nothing else, so a city's items carried
byte-identical media, and the load's exact-duplicate stage
(:func:`~vtscore.state.media_lookup.collapse_duplicates`) folded each city into a
single media: 108 in all, rather than 108 x ``DEFAULT_ITEMS_PER_CITY``.  Each
renderer now folds the item's serial into one level of a few pixels or samples,
which gives every item bytes of its own without changing how it looks or sounds.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from PIL import Image

from vtscore.media import get as media_get
from vtscore.media._toponymy_demo import (
    DEFAULT_ITEMS_PER_CITY,
    SERIAL_MARK_BITS,
    SOURCE_ID,
    TAXONOMY,
    serial_mark,
    total_cities,
)
from vtscore.state.media_lookup import collapse_duplicates


@pytest.fixture(scope="module", params=["image", "audio"])
def demo(request) -> tuple[str, dict[int, dict]]:
    """``(media_type, clips)`` for one full-size demo load, shared by this module's tests."""
    clips: dict[int, dict] = {}
    media_get(request.param).load_demo_source(
        source=SOURCE_ID,
        categories=list(TAXONOMY),
        slice_start=0,
        slice_end=None,
        clips=clips,
        on_progress=lambda *_args: None,
        embedder=None,
    )
    return request.param, clips


def _decode(media_type: str, data: bytes) -> np.ndarray:
    """A tile's RGB pixels or a tone's PCM samples, widened so differences cannot wrap."""
    if media_type == "image":
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.int32)
    with wave.open(io.BytesIO(data)) as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.int32)


def test_dedup_keeps_every_item(demo):
    _media_type, clips = demo
    assert total_cities() == 108
    assert len(clips) == 108 * DEFAULT_ITEMS_PER_CITY
    # A copy: collapsing rewrites the origins of the medias it keeps.
    medias = {cid: dict(media) for cid, media in clips.items()}
    assert collapse_duplicates(medias) == 0, "items of one city still share bytes"
    assert len(medias) == 108 * DEFAULT_ITEMS_PER_CITY


def test_items_of_a_city_differ_by_one_level_in_a_few_places(demo):
    media_type, clips = demo
    by_city: dict[str, list[np.ndarray]] = {}
    for media in clips.values():
        by_city.setdefault(media["category"], []).append(_decode(media_type, media["media_bytes"]))
    assert len(by_city) == total_cities()
    for city, decoded in by_city.items():
        assert len(decoded) == DEFAULT_ITEMS_PER_CITY, city
        stack = np.stack(decoded)
        spread = stack.max(axis=0) - stack.min(axis=0)
        # They differ, but by one level at most, and only where the mark is written.
        assert spread.max() == 1, city
        assert np.count_nonzero(spread) <= SERIAL_MARK_BITS, city
        if media_type == "image":
            # Still a solid tile of the city's colour, give or take the mark.
            for pixels in decoded:
                assert np.abs(pixels - pixels[0, 0]).max() <= 1, city


def test_serial_mark_spells_the_serial_in_binary():
    mark = serial_mark(0b1011)
    assert mark.shape == (SERIAL_MARK_BITS,)
    assert mark[:5].tolist() == [1, 1, 0, 1, 0]
    assert int((mark.astype(np.int64) << np.arange(SERIAL_MARK_BITS)).sum()) == 0b1011


@pytest.mark.parametrize("serial", [-1, 2**SERIAL_MARK_BITS])
def test_serial_mark_refuses_a_serial_it_cannot_spell(serial):
    with pytest.raises(ValueError, match="does not fit"):
        serial_mark(serial)
