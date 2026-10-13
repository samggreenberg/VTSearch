"""`o365_better`'s decisions, on a corpus small enough to reason about (#4815).

Like `coco_better`, the loader is mostly the shared scale core: banding,
designation, the pool draw and the media dict are imported, not restated. What
is new, and so what is pinned here:

* **The class map goes through the rules, and a misspelt name is refused.**
  Objects365's own spellings carry typos, so a "corrected" name would map
  nothing and say nothing.
* **A box is a positive only if it is ONE real object.** A box the release flags
  ``iscrowd``/``isfake``/``isreflected``, one with no area, or one under an
  ambiguous name holds the class -- so the image is no negative for it -- but
  is never a positive. `coco_better` drops crowd regions outright; at 14% of
  Objects365's boxes that would put crowds of a class into its negatives.
* **Its own dropped cells, its own labels source,** through two optional
  arguments on the shared core whose defaults leave `coco_better` unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"

#: A handful of real Objects365 names, spelt as the release spells them.
_CATS = ["Bus", "Clock", "Desk", "Dinning Table", "Person"]


@pytest.fixture(scope="module")
def pc():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import pile_config

    return pile_config


@pytest.fixture(scope="module")
def mod(pc):
    from pilebuild.loaders import o365_better

    return o365_better


@pytest.fixture
def small_c(pc, monkeypatch):
    """*C* cut down to three classes, mapped and flagged the way the real tables are."""
    monkeypatch.setattr(pc, "SCALE_CLASSES", ("bus", "clock", "dining table"))
    monkeypatch.setattr(pc, "O365_NAMES", {"bus": ("Bus",), "clock": ("Clock",), "dining table": ("Dinning Table",)})
    monkeypatch.setattr(pc, "O365_AMBIGUOUS", {"dining table": ("Desk",)})
    monkeypatch.setattr(pc, "O365_DROPPED_CELLS", frozenset())
    return pc.SCALE_CLASSES


def _ann(aid: int, iid: int, cat: str, bbox: list[float], **flags: int) -> dict:
    return {
        "id": aid,
        "image_id": iid,
        "category_id": _CATS.index(cat) + 1,
        "bbox": bbox,
        "iscrowd": flags.get("iscrowd", 0),
        "isfake": flags.get("isfake", 0),
        "isreflected": flags.get("isreflected", 0),
    }


def _write(path: Path, n_images: int, anns: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": i,
                        "width": 100,
                        "height": 100,
                        "file_name": f"images/v1/patch0/objects365_v1_{i:08d}.jpg",
                    }
                    for i in range(1, n_images + 1)
                ],
                "annotations": anns,
                "categories": [{"id": i + 1, "name": n} for i, n in enumerate(_CATS)],
            }
        )
    )
    return path


class TestTheReader:
    def test_an_image_with_nothing_of_c_is_an_empty_answer_not_a_missing_one(self, mod, small_c, tmp_path):
        """`{}` is "the source looked and found nothing", which #3667 turned on."""
        src = mod.read_o365_labels(_write(tmp_path / "a.json", 2, [_ann(1, 1, "Person", [0, 0, 50, 50])]), small_c)
        assert src.labels == {1: {}, 2: {}}, "a class outside C is not read, and every image gets an entry"

    @pytest.mark.parametrize("flag", ["iscrowd", "isfake", "isreflected"])
    def test_a_flagged_box_holds_the_class_but_is_never_a_positive(self, mod, small_c, tmp_path, flag):
        path = _write(tmp_path / "a.json", 1, [_ann(1, 1, "Bus", [10, 10, 40, 40], **{flag: 1})])
        src = mod.read_o365_labels(path, small_c)
        assert "bus" in src.labels[1], "the image holds a bus, so it may not be a negative for one"
        assert (1, "bus") in src.doubtful
        assert src.reasons[(1, "bus")] == {flag}

    def test_a_box_with_no_area_is_doubtful(self, mod, small_c, tmp_path):
        src = mod.read_o365_labels(_write(tmp_path / "a.json", 1, [_ann(1, 1, "Clock", [10, 10, 0, 5])]), small_c)
        assert (1, "clock") in src.doubtful and src.reasons[(1, "clock")] == {"no_area"}

    def test_an_unflagged_box_wins_and_the_flagged_ones_are_dropped(self, mod, small_c, tmp_path):
        """The crowd box is 90% of the frame; kept, it would be the largest instance and band `large`."""
        anns = [_ann(1, 1, "Bus", [10, 10, 20, 20]), _ann(2, 1, "Bus", [0, 0, 95, 95], iscrowd=1)]
        src = mod.read_o365_labels(_write(tmp_path / "a.json", 1, anns), small_c)
        assert src.labels[1] == {"bus": [[10, 10, 30, 30]]}
        assert not src.doubtful and not src.reasons

    def test_an_ambiguous_name_holds_the_class_but_is_never_a_positive(self, mod, small_c, tmp_path):
        anns = [_ann(1, 1, "Desk", [0, 0, 90, 60]), _ann(2, 2, "Desk", [0, 0, 90, 60])]
        anns.append(_ann(3, 2, "Dinning Table", [5, 5, 30, 30]))
        src = mod.read_o365_labels(_write(tmp_path / "a.json", 2, anns), small_c)
        assert (1, "dining table") in src.doubtful and src.reasons[(1, "dining table")] == {"name:Desk"}
        assert src.labels[2] == {"dining table": [[5, 5, 35, 35]]}, "a real dining table bands; the desk does not"
        assert (2, "dining table") not in src.doubtful

    def test_a_misspelt_name_is_refused_not_mapped_to_nothing(self, mod, pc, small_c, tmp_path, monkeypatch):
        monkeypatch.setattr(pc, "O365_NAMES", {**pc.O365_NAMES, "dining table": ("Dining Table",)})
        with pytest.raises(SystemExit) as err:
            mod.read_o365_labels(_write(tmp_path / "a.json", 1, []), small_c)
        assert "Dining Table" in str(err.value)

    def test_missing_annotations_name_the_file(self, mod, small_c, tmp_path):
        with pytest.raises(SystemExit) as err:
            mod.read_o365_labels(tmp_path / "gone.json", small_c)
        assert "gone.json" in str(err.value)


class TestTheBuild:
    def test_a_doubtful_pair_is_neither_a_positive_nor_clean(self, mod, small_c, tmp_path):
        anns = [
            _ann(1, 1, "Bus", [10, 10, 40, 40]),  # a positive
            _ann(2, 2, "Bus", [10, 10, 40, 40], iscrowd=1),  # a crowd of buses: neither
            _ann(3, 3, "Desk", [0, 0, 90, 60]),  # maybe a dining table: neither
        ]
        src = mod.read_o365_labels(_write(tmp_path / "a.json", 4, anns), small_c)
        supply, _, clean = mod.banded(src)
        banded = {iid for bands in supply.values() for ids in bands.values() for iid in ids}
        assert banded == {1}
        assert clean == [4], "only the image holding nothing of C, not even doubtfully, is clean"

    def test_load_reads_the_patch_tree_and_records_the_source(self, mod, pc, small_c, tmp_path, monkeypatch):
        pytest.importorskip("PIL")
        import io

        from PIL import Image

        val = tmp_path / "val"
        anns = [_ann(1, 1, "Bus", [10, 10, 40, 40]), _ann(2, 2, "Desk", [0, 0, 90, 60])]
        _write(val / "zhiyuan_objv2_val.json", 3, anns)
        for i in (1, 2, 3):
            path = val / f"images/v1/patch0/objects365_v1_{i:08d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            buf = io.BytesIO()
            Image.new("RGB", (100, 100), (90, 90, 90)).save(buf, format="JPEG")
            path.write_bytes(buf.getvalue())
        monkeypatch.setattr(pc, "O365_VAL_DIR", val)
        monkeypatch.setattr(pc, "O365_VAL_ANNOTATIONS", val / "zhiyuan_objv2_val.json")
        monkeypatch.setattr(pc, "SCALE_N_POS", 1)
        monkeypatch.setattr(pc, "SCALE_N_NEG", 1)
        monkeypatch.setattr(pc, "SCALE_N_NEG_SPARE", 0)

        medias: dict[int, dict] = {}
        mod.load("o365_better", medias, "siglip")

        assert set(medias) == {1, 3}, "the positive and the one clean negative; the desk image is neither"
        assert medias[1]["categories"] == ["bus@large"]
        assert medias[3]["categories"] == [] and "bus@large" in medias[3]["evaluable_categories"]
        for m in medias.values():
            assert m["origin"]["importer"] == "o365_better"
            assert m["origin"]["params"]["labels"] == "objects365"
            assert m["coco_scored"] is True, "every image is exhaustively annotated, so every negative is provable"

    def test_check_names_every_patch_directory(self, mod, pc, tmp_path, monkeypatch):
        """A canary that checks a path the build does not open is not a canary (#3299)."""
        val = tmp_path / "val"
        (val / "zhiyuan_objv2_val.json").parent.mkdir(parents=True)
        (val / "zhiyuan_objv2_val.json").write_text("{}")
        for d in pc.O365_VAL_PATCH_DIRS:
            (val / d).mkdir(parents=True)
        monkeypatch.setattr(pc, "O365_VAL_DIR", val)
        monkeypatch.setattr(pc, "O365_VAL_ANNOTATIONS", val / "zhiyuan_objv2_val.json")
        assert "44 patch directories" in mod.check("o365_better")

        (val / "images/v2/patch43").rmdir()
        with pytest.raises(SystemExit) as err:
            mod.check("o365_better")
        assert "patch43" in str(err.value), "the last patch must be checked, not a sample"


class TestTheCensus:
    def test_it_prices_an_ambiguous_name_by_what_it_alone_withholds(self, pc, small_c, tmp_path, monkeypatch):
        """Image 2 is held only through Desk, image 3 through Desk AND a crowd of buses.

        Ruling Desk out frees image 2 into the pool, and frees image 3's dining-table
        pair into the negatives -- but not image 3 itself, which still holds a bus.
        """
        import o365_supply

        monkeypatch.setattr(pc, "SCALE_N_POS", 1)
        monkeypatch.setattr(pc, "SCALE_N_NEG", 1)
        monkeypatch.setattr(pc, "SCALE_N_NEG_SPARE", 0)
        anns = [
            _ann(1, 1, "Bus", [10, 10, 40, 40]),
            _ann(2, 2, "Desk", [0, 0, 90, 60]),
            _ann(3, 3, "Desk", [0, 0, 90, 60]),
            _ann(4, 3, "Bus", [0, 0, 20, 20], iscrowd=1),
        ]
        res = o365_supply.census(_write(tmp_path / "a.json", 4, anns), floor=1)

        assert res["cells"]["bus@large"] == 1 and res["clean_pool"] == 1
        assert res["clean_if_no_ambiguous"] == 2, "image 2 only; image 3 still holds a crowd of buses"
        assert res["ambiguous_name_price"]["Desk"] == {"pairs_alone": 2, "clean_alone": 1}
        assert res["doubtful_by_reason"] == {"name:Desk": 2, "iscrowd": 1}
        assert res["haystack"]["bus@large"] == 1, "the one clean image is the bus cell's only negative"


class TestTheTables:
    def test_every_class_in_c_is_mapped(self, pc):
        assert set(pc.O365_NAMES) == set(pc.SCALE_CLASSES)
        assert all(pc.O365_NAMES[c] for c in pc.SCALE_CLASSES)

    def test_no_name_is_two_classes(self, pc):
        owner: dict[str, str] = {}
        for cls, names in pc.O365_NAMES.items():
            for name in names:
                assert name not in owner, f"{name!r} is both {owner[name]!r} and {cls!r}"
                owner[name] = cls

    def test_no_name_is_both_definite_and_ambiguous_for_one_class(self, pc):
        assert set(pc.O365_AMBIGUOUS) <= set(pc.SCALE_CLASSES)
        for cls, names in pc.O365_AMBIGUOUS.items():
            both = set(names) & set(pc.O365_NAMES.get(cls, ()))
            assert not both, f"{cls!r}: {sorted(both)} cannot be the class and maybe the class"

    def test_every_dropped_cell_is_a_cell(self, pc):
        cells = {pc.scale_cell(c, b) for c in pc.SCALE_CLASSES for b in pc.BOX_BANDS}
        assert pc.O365_DROPPED_CELLS <= cells, sorted(pc.O365_DROPPED_CELLS - cells)

    def test_the_patch_directories_are_the_releases_two_rounds(self, pc):
        dirs = pc.O365_VAL_PATCH_DIRS
        assert len(dirs) == 44
        assert dirs[0] == "images/v1/patch0" and dirs[15] == "images/v1/patch15"
        assert dirs[16] == "images/v2/patch16" and dirs[-1] == "images/v2/patch43"

    def test_the_dataset_is_on_request_and_boxed(self, pc):
        spec = pc.DATASETS["o365_better"]
        assert spec["kind"] == "o365_better" and spec["boxed"] is True
        assert spec.get("on_request"), "its images are staged by hand, not in every checkout's pile"


class TestTheSharedCore:
    def test_the_loader_imports_the_core_rather_than_copying_it(self, mod):
        from pilebuild import scale_core

        for name in ("band_candidates", "designate_cells", "draw_negatives", "scale_media"):
            assert getattr(mod, name) is getattr(scale_core, name), name

    def test_designate_cells_drops_coco_betters_cells_by_default(self, pc, monkeypatch):
        from pilebuild.scale_core import designate_cells

        monkeypatch.setattr(pc, "SCALE_CLASSES", ("bus",))
        monkeypatch.setattr(pc, "SCALE_N_POS", 1)
        monkeypatch.setattr(pc, "SCALE_DROPPED_CELLS", frozenset({"bus@small"}))
        supply = {"bus": {"small": [1], "medium": [2], "large": [3]}}
        assert set(designate_cells(supply, {}, {})) == {"bus@medium", "bus@large"}
        assert set(designate_cells(supply, {}, {}, dropped=frozenset({"bus@large"}))) == {"bus@small", "bus@medium"}
