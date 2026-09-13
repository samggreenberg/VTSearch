"""The VG dims cache has to degrade, not collapse (#3822).

`vg_image_dims.json` is the one artifact that stands between every VG-derived
build and 108k JPEG header reads, and it was accepted only when
``len(raw) >= len(paths)``. That is an all-or-nothing guard on a *shared file
that outlives the code*: the copy on scratch held 108,075 entries against 108,245
JPEGs — exactly the 170 corrupt images an older writer dropped instead of
recording — so the guard had been failing since the day the cache was written,
and every build silently paid the full scan with one log line to say so.

The same shape as #3297, two directories over: a shared artifact on scratch stayed
in an old shape while the code moved, and nothing caught it for eleven days.

So these pin the properties that make a cache un-collapsible rather than the
repair:

* **a gap costs the gap, not the cache** — an unknown id is read, a known one is
  not, and one new file does not disable the other 108,244;
* **a ``null`` entry is an answer** — "read, and would not parse" is a fact, and
  re-reading it every run is what made the misses get dropped in the first place;
* **membership does not move**: the dims a caller gets are the same whether the
  cache is complete, partial or absent. That is the property that makes the fix
  safe to land under a built dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def vgs():
    """``pilebuild.vgsource``, which touches no pile path at import."""
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    from pilebuild import vgsource

    return vgsource


def _jpeg(path: Path, size: tuple[int, int]) -> None:
    """A real JPEG, because the read under test is a real header parse."""
    from PIL import Image

    Image.new("RGB", size, (7, 7, 7)).save(path, "JPEG")


@pytest.fixture
def source(tmp_path):
    """``(paths, cache_path)`` — three readable JPEGs and one file that is not."""
    images = tmp_path / "img"
    images.mkdir()
    paths = {}
    for iid, size in ((1, (10, 20)), (2, (30, 40)), (3, (50, 60))):
        p = images / f"{iid}.jpg"
        _jpeg(p, size)
        paths[iid] = p
    broken = images / "4.jpg"
    broken.write_bytes(b"not a jpeg")
    paths[4] = broken
    return paths, tmp_path / "dims.json"


def _write_cache(path: Path, entries: dict) -> None:
    path.write_text(json.dumps({str(k): v for k, v in entries.items()}))


class TestAGapCostsTheGapNotTheCache:
    def test_one_unknown_id_does_not_disable_the_rest(self, vgs, source, monkeypatch):
        """The bug, in miniature. Under the old length guard a cache one entry
        short was thrown away whole; here image 3 is read and 1 and 2 are not."""
        paths, cache = source
        _write_cache(cache, {1: [10, 20], 2: [30, 40], 4: None})
        read = []
        real = vgs.read_jpeg_dims

        def spy(subset, **kw):
            read.append(sorted(subset))
            return real(subset, **kw)

        monkeypatch.setattr(vgs, "read_jpeg_dims", spy)

        dims = vgs.image_dims(paths, cache)

        assert read == [[3]]
        assert dims == {1: (10, 20), 2: (30, 40), 3: (50, 60)}

    def test_a_complete_cache_reads_nothing(self, vgs, source, monkeypatch):
        paths, cache = source
        _write_cache(cache, {1: [10, 20], 2: [30, 40], 3: [50, 60], 4: None})
        monkeypatch.setattr(vgs, "read_jpeg_dims", lambda *_a, **_k: pytest.fail("read a header it had cached"))
        assert vgs.image_dims(paths, cache) == {1: (10, 20), 2: (30, 40), 3: (50, 60)}

    def test_a_missing_cache_reads_everything(self, vgs, source):
        paths, cache = source
        assert not cache.exists()
        assert vgs.image_dims(paths, cache) == {1: (10, 20), 2: (30, 40), 3: (50, 60)}


class TestANullEntryIsAnAnswer:
    def test_a_cached_miss_is_not_reread(self, vgs, source, monkeypatch):
        """Image 4 is corrupt and the cache says so. Treating that as a gap is
        what makes a cache of 108,245 files permanently 170 short."""
        paths, cache = source
        _write_cache(cache, {1: [10, 20], 2: [30, 40], 3: [50, 60], 4: None})
        monkeypatch.setattr(vgs, "read_jpeg_dims", lambda *_a, **_k: pytest.fail("re-read a known-corrupt file"))
        assert 4 not in vgs.image_dims(paths, cache)

    def test_the_owner_records_the_miss_rather_than_dropping_it(self, vgs, source):
        paths, cache = source
        vgs.image_dims(paths, cache, write=True)
        raw = json.loads(cache.read_text())
        # One entry per FILE, misses included -- which is what lets a completeness
        # check of any kind pass at all.
        assert set(raw) == {"1", "2", "3", "4"}
        assert raw["4"] is None

    def test_the_written_cache_satisfies_the_next_run_completely(self, vgs, source, monkeypatch):
        paths, cache = source
        vgs.image_dims(paths, cache, write=True)
        monkeypatch.setattr(
            vgs, "read_jpeg_dims", lambda *_a, **_k: pytest.fail("the cache it just wrote was not enough")
        )
        assert vgs.image_dims(paths, cache) == {1: (10, 20), 2: (30, 40), 3: (50, 60)}


class TestMembershipDoesNotMove:
    def test_complete_partial_and_absent_caches_agree(self, vgs, source):
        """The property that makes this safe to land under a built dataset: the
        dims a build sees do not depend on the state of the cache, so no image
        enters or leaves a cell because the cache was repaired."""
        paths, cache = source
        absent = vgs.image_dims(paths, cache)
        _write_cache(cache, {1: [10, 20], 4: None})
        partial = vgs.image_dims(paths, cache)
        vgs.image_dims(paths, cache, write=True)
        complete = vgs.image_dims(paths, cache)
        assert absent == partial == complete

    def test_ids_the_cache_knows_but_the_source_does_not_are_dropped(self, vgs, source):
        """A cache outliving its source must not widen a build. Image 99 is not
        on disk; no amount of caching makes it a candidate."""
        paths, cache = source
        _write_cache(cache, {1: [10, 20], 99: [1, 1]})
        assert 99 not in vgs.image_dims(paths, cache)


class TestTheWrite:
    def test_a_reader_never_writes(self, vgs, source):
        """A build is not the right thing to be rewriting a shared artifact, and
        several run at once."""
        paths, cache = source
        vgs.image_dims(paths, cache)
        assert not cache.exists()

    def test_nothing_is_rewritten_when_nothing_was_missing(self, vgs, source):
        paths, cache = source
        vgs.image_dims(paths, cache, write=True)
        before = cache.stat().st_mtime_ns
        vgs.image_dims(paths, cache, write=True)
        assert cache.stat().st_mtime_ns == before

    def test_the_write_leaves_no_temporary_behind(self, vgs, source):
        """It is atomic -- written beside the cache and renamed -- so a loser in a
        race leaves a whole file rather than half of one."""
        paths, cache = source
        vgs.image_dims(paths, cache, write=True)
        assert [p.name for p in cache.parent.iterdir() if p.name.startswith(cache.name)] == [cache.name]


class TestOnePathForOneArtifact:
    def test_the_scan_and_the_loader_spell_the_cache_the_same_way(self, vgs):
        """Two spellings of a shared artifact is how a caller ends up reporting on
        a file nobody else is using (#3299)."""
        if str(_PILE_DIR) not in sys.path:
            sys.path.insert(0, str(_PILE_DIR))
        import pile_config as pc

        assert vgs.vg_dims_cache() == pc.PILE / "vg_image_dims.json"
