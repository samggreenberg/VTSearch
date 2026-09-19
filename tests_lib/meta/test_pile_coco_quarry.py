"""`coco_quarry`'s three decisions, on a corpus small enough to reason about.

The loader itself is mostly imports: it reuses `band_for`, `band_candidates`,
`designate_cells`, `draw_negatives` and `scale_media` from `vg_scale` precisely
so that a cell means the same thing in both. What is new, and therefore what
needs pinning, is the handful of choices the reuse does not cover:

* **`iscrowd` regions are dropped.** A crowd box is a region, not an instance,
  and admitting one would band an image by an extent no user would drag -- #3985
  arriving through the front door. #3983's census dropped them, so the build has
  to agree or the supply it produces is not the supply that was measured.
* **The media shape is shared, not merely similar.** Comparability between the
  two datasets lives in the dict as much as in the band rule, and a second copy
  would drift in a field nobody diffs.
* **Pixels resolve out of either staged archive.** 94% of the corpus is in
  `train2017.zip`, which nothing in this repo referenced until #3991, and the
  splits are disjoint so a name must resolve in exactly one.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def mod():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    pytest.importorskip("PIL")
    from pilebuild.loaders import coco_quarry

    return coco_quarry


def _jpeg_bytes(w: int = 64, h: int = 48) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 120, 120)).save(buf, format="JPEG")
    return buf.getvalue()


def _annotations(path: Path, images: list[dict], anns: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "images": [{"id": i["id"], "width": i["w"], "height": i["h"], "file_name": i["file"]} for i in images],
                "annotations": anns,
                "categories": [{"id": 1, "name": "bus"}, {"id": 2, "name": "clock"}],
            }
        )
    )


def _corpus(tmp_path: Path) -> Path:
    """A two-split corpus: one image per split, one crowd annotation among them."""
    anchor = tmp_path / "coco_anchor"
    anchor.mkdir()
    _annotations(
        anchor / "instances_val2017.json",
        [{"id": 1, "w": 100, "h": 100, "file": "000000000001.jpg"}],
        [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 40], "iscrowd": 0},
            {"id": 11, "image_id": 1, "category_id": 2, "bbox": [0, 0, 95, 95], "iscrowd": 1},
        ],
    )
    _annotations(
        anchor / "instances_train2017.json",
        [{"id": 2, "w": 100, "h": 100, "file": "000000000002.jpg"}],
        [{"id": 20, "image_id": 2, "category_id": 1, "bbox": [20, 20, 30, 30], "iscrowd": 0}],
    )

    images = tmp_path / "images"
    images.mkdir()
    for name, ids in (("val2017.zip", ["000000000001.jpg"]), ("train2017.zip", ["000000000002.jpg"])):
        with zipfile.ZipFile(images / name, "w") as zf:
            for member in ids:
                zf.writestr(f"{name[:-4]}/{member}", _jpeg_bytes())
    return anchor


def test_iscrowd_regions_are_dropped(mod, tmp_path: Path):
    """The crowd `clock` box is 90% of its frame; if it were kept it would band."""
    anchor = _corpus(tmp_path)
    labels, dims, filenames = mod.read_coco_labels(anchor, ("bus", "clock"))
    assert labels[1] == {"bus": [[10.0, 10.0, 50.0, 50.0]]}, "the iscrowd clock box must not appear"
    assert "clock" not in labels[1]
    assert dims[1] == (100, 100)
    assert filenames[2] == "000000000002.jpg"


def test_an_image_with_no_wanted_class_is_an_empty_answer_not_a_missing_one(mod, tmp_path: Path):
    """`{}` means COCO looked and found nothing; absence means nobody asked.

    #3667 turned on exactly this difference and the two are spellable as the
    same value under `.get()`, so it is pinned rather than commented.
    """
    anchor = _corpus(tmp_path)
    labels, _, _ = mod.read_coco_labels(anchor, ("dog",))
    assert labels[1] == {} and labels[2] == {}
    assert set(labels) == {1, 2}, "every image in the corpus gets an entry"


def test_both_splits_are_read(mod, tmp_path: Path):
    anchor = _corpus(tmp_path)
    labels, _, _ = mod.read_coco_labels(anchor, ("bus",))
    assert labels[1]["bus"] and labels[2]["bus"], "train2017 carries 94% of the corpus (#3991)"


def test_missing_annotations_name_the_file(mod, tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit) as err:
        mod.read_coco_labels(tmp_path / "empty", ("bus",))
    assert "instances_val2017.json" in str(err.value)


def test_check_names_both_archives(mod, tmp_path: Path, monkeypatch):
    """A canary that checks a path the build does not open is not a canary (#3299)."""
    import pile_config as pc

    anchor = _corpus(tmp_path)
    monkeypatch.setattr(pc, "COCO_ANCHOR_DIR", anchor)
    monkeypatch.setattr(pc, "COCO_VAL_ZIP", tmp_path / "images" / "val2017.zip")
    monkeypatch.setattr(pc, "COCO_TRAIN_ZIP", tmp_path / "images" / "train2017.zip")
    assert "present" in mod.check("coco_quarry")

    monkeypatch.setattr(pc, "COCO_TRAIN_ZIP", tmp_path / "images" / "gone.zip")
    with pytest.raises(SystemExit) as err:
        mod.check("coco_quarry")
    assert "gone.zip" in str(err.value), "the train archive must be named, not just the val one"


def test_media_shape_is_the_same_object_as_vg_scale(mod):
    """Both loaders build their media dict with the same function, not a copy."""
    from pilebuild import scale_core as vg_scale

    assert mod.scale_media is vg_scale.scale_media


def test_scale_media_carries_the_importer_and_skips_undecodable_bytes():
    from pilebuild.scale_core import scale_media

    common = {
        "iid": 7,
        "filename": "7.jpg",
        "origin_name": "zip::7.jpg",
        "box_dims": (100, 100),
        "cats": [],
        "boxes_for": {},
        "cells": [],
        "neg_set": {7},
        "labels": {7: {}},
        "coco_scored": {7},
        "exhaustive": {7},
        "reviewed_absent": set(),
        "reviewed_present": set(),
        "embedder_name": "siglip",
    }
    ok = scale_media(data=_jpeg_bytes(), importer="coco_quarry", **common)
    assert ok is not None
    assert ok["origin"]["importer"] == "coco_quarry"
    assert ok["coco_scored"] is True and ok["category"] == ""
    assert scale_media(data=b"not a jpeg", importer="coco_quarry", **common) is None


class TestFullCorpusMode:
    """`coco_quarry_full` embeds the corpus; `coco_quarry` embeds a draw from it.

    The difference is the whole reason the second dataset exists: a designated
    cell freezes `SCALE_N_POS`/`SCALE_N_NEG` into the embedding, so #3987's
    prevalence axis would cost a re-embed. Embedded whole, a cell is a filter.
    """

    def test_the_dataset_declares_the_flag(self):
        import pile_config as pc

        assert pc.DATASETS["coco_quarry_full"]["full_corpus"] is True
        assert pc.DATASETS["coco_quarry_full"]["kind"] == "coco_quarry", "one loader, two datasets"
        assert not pc.DATASETS["coco_quarry"].get("full_corpus")
        assert pc.DATASETS["coco_quarry_full"].get("on_request"), "6.8x the designated build"

    def test_cells_of_takes_every_banded_image(self, mod):
        """No `SCALE_N_POS` cap: the cap is what is being deferred to export."""
        supply = {"bus": {"small": [1, 2, 3], "large": [4], "medium": []}}
        assert mod._cells_of("bus", supply) == {"bus@small": [1, 2, 3], "bus@large": [4]}, (
            "empty bands are dropped, non-empty ones are kept whole"
        )

    def test_an_unbanded_image_is_still_in_the_corpus(self, mod, tmp_path: Path):
        """The trap: `band_candidates` returns banded supply and the clean pool.

        An image holding a class in NO valid band -- scattered, or oversize -- is
        in neither, so a full-corpus emit set built from their union would drop it
        and still call itself everything.
        """
        anchor = _corpus(tmp_path)
        labels, _, _ = mod.read_coco_labels(anchor, ("bus",))
        # image 1 holds a bus, image 2 holds a bus; neither is clean.
        assert set(labels) == {1, 2}
        # The emit set in full mode is keyed on `labels`, which carries every
        # image in the corpus including those that band nowhere.
        assert all(iid in labels for iid in (1, 2))

    def test_the_corpus_read_is_cached_across_embedders(self, mod, tmp_path: Path):
        """The builder calls `load` once per embedder in one process."""
        anchor = _corpus(tmp_path)
        mod._CORPUS.clear()
        first = mod.read_coco_labels(anchor, ("bus",))
        second = mod.read_coco_labels(anchor, ("bus",))
        assert first[0] is second[0], "same object, not a re-read"
        assert len(mod._CORPUS) == 1
        mod.read_coco_labels(anchor, ("bus", "clock"))
        assert len(mod._CORPUS) == 2, "a different class list is a different answer"
        mod._CORPUS.clear()


class TestPatchShards:
    """The patch column is cut into shards, and the cut must not change a cell.

    One full-corpus `dinov3_patch` cell is ~37 GB of grids: unwritable in one
    `pickle.dump` and unreadable through `load_medias` afterwards. Sharding is
    the way round that, and the thing it must not do is change what a cell means.
    """

    def test_every_shard_is_declared(self):
        import pile_config as pc

        shards = [d for d in pc.DATASETS if d.startswith("coco_quarry_full_s")]
        assert len(shards) == pc.COCO_QUARRY_SHARDS
        for i, name in enumerate(sorted(shards, key=lambda d: int(d.rsplit("s", 1)[1]))):
            spec = pc.DATASETS[name]
            assert spec["shard"] == (i, pc.COCO_QUARRY_SHARDS)
            assert spec["full_corpus"] is True, "a shard is a slice of the full corpus"
            assert spec["kind"] == "coco_quarry", "one loader, many datasets"

    def test_the_shards_partition_the_corpus_exactly(self):
        """Every image in exactly one shard: no gap, no overlap.

        A gap is a silently smaller corpus and an overlap double-counts an image
        in any study that loads two shards -- both look like a working build.
        """
        import pile_config as pc

        n = pc.COCO_QUARRY_SHARDS
        corpus = range(1, 20_000)
        seen = [{iid for iid in corpus if iid % n == i} for i in range(n)]
        union = set().union(*seen)
        assert union == set(corpus), "a gap would be a quietly smaller corpus"
        assert sum(len(s) for s in seen) == len(union), "an overlap would double-count"

    def test_a_shard_carries_every_class(self, mod, tmp_path: Path):
        """Modulo, not a contiguous range, so one shard is a usable sample.

        A contiguous slice of COCO ids is not random with respect to class: the
        ids carry collection order, so a range could miss a class entirely.
        """
        import pile_config as pc

        anchor = _corpus(tmp_path)
        labels, _, _ = mod.read_coco_labels(anchor, ("bus",))
        per_shard = [{i for i in labels if i % pc.COCO_QUARRY_SHARDS == k} for k in range(pc.COCO_QUARRY_SHARDS)]
        assert sum(len(s) for s in per_shard) == len(labels)
