"""The SOD sweep's ``vg_s``/``vg_m``/``vg_l``/``vg_a`` must match the GUI demo slices.

``scripts/sod/sweep.py --datasets vg_m`` is only comparable to the
``vtscore.eval --datasets visual_genome_m`` experiments under ``docs/experiments/`` if
both resolve to the *same images*. The demo loader
(``_collect_visual_genome_files``) keeps images with >=1 object in the 100-category
demo vocabulary, drops those whose JPEG is missing, sorts by ``image_id``, and only
then takes a fractional slice of that flat list. These tests pin the two parts of that
rule the sweep re-derives: the vocabulary matcher and the presence-before-slice order.

No staged corpus needed: ``SodDataset.__new__`` sidesteps the ``__init__`` that would
demand ``/exp/scale26/...``, and a stub reader stands in for the zip index.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sod"))

from datasets import _CONFIG, _VG_DEMO_SLICES, SodDataset, _vg_demo_matcher  # noqa: E402


class _StubReader:
    """Minimal ``_VgZipReader`` stand-in: only membership matters to the slicer."""

    def __init__(self, present: set[int]) -> None:
        self._present = present

    def has(self, image_id: int) -> bool:
        return image_id in self._present


def _slicer(demo_slice, present):
    ds = SodDataset.__new__(SodDataset)
    ds._demo_slice = demo_slice
    ds._reader = _StubReader(present)
    return ds


class TestConfig:
    def test_slices_share_the_vg_corpus_and_carry_demo_fractions(self):
        for name, frac in _VG_DEMO_SLICES.items():
            cfg = _CONFIG[name]
            assert cfg["kind"] == "vg"
            assert cfg["extract"] == _CONFIG["vg"]["extract"]
            assert cfg["images"] == _CONFIG["vg"]["images"]
            assert cfg["negatives_exhaustive"] is False
            assert cfg["demo_slice"] == frac

    def test_fractions_match_the_demo_dataset_definitions(self):
        # vtscore/media/image/_demo_sources.py, the visual_genome_* DemoDataset entries.
        assert _VG_DEMO_SLICES == {
            "vg_s": (0.0, 1 / 50),
            "vg_m": (1 / 50, 3 / 50),
            "vg_l": (3 / 50, 7 / 50),
            "vg_a": (0.0, None),
        }

    def test_whole_corpus_vg_is_not_sliced(self):
        assert "demo_slice" not in _CONFIG["vg"]

    def test_s_m_l_are_disjoint_and_a_covers_everything(self):
        present = set(range(1000))
        invocab = set(range(1000))
        got = {n: _slicer(_VG_DEMO_SLICES[n], present)._demo_universe(invocab) for n in ("vg_s", "vg_m", "vg_l")}
        assert got["vg_s"] & got["vg_m"] == set()
        assert got["vg_m"] & got["vg_l"] == set()
        assert got["vg_s"] & got["vg_l"] == set()
        assert _slicer(_VG_DEMO_SLICES["vg_a"], present)._demo_universe(invocab) == invocab


class TestDemoUniverse:
    def test_slice_is_taken_over_the_sorted_in_vocab_list(self):
        invocab = {900, 100, 500, 300, 700}  # deliberately unsorted
        universe = _slicer((0.0, 0.4), invocab)._demo_universe(invocab)
        assert universe == {100, 300}  # sorted -> [100,300,500,700,900], first int(5*0.4)=2

    def test_out_of_vocab_ids_are_excluded(self):
        present = set(range(100))
        universe = _slicer((0.0, None), {1, 2, 3})._demo_universe(present & {1, 2, 3})
        assert universe == {1, 2, 3}

    def test_missing_jpegs_are_pruned_before_slicing_not_after(self):
        """The 4 real in-vocab images with no JPEG must not shift later slice indices."""
        invocab = set(range(10))
        present = set(range(10)) - {0, 1}  # ids 0,1 annotated but absent from the zips
        # Present+sorted = [2..9] (8 ids); the second quarter is [4, 5].
        assert _slicer((0.25, 0.5), present)._demo_universe(invocab) == {4, 5}
        # Slicing first and pruning after would have given [2,3] minus nothing = {2,3}.

    def test_end_fraction_none_runs_to_the_end(self):
        invocab = set(range(10))
        assert _slicer((0.5, None), invocab)._demo_universe(invocab) == set(range(5, 10))


class TestVocabMatcher:
    @pytest.mark.parametrize("name", ["man", "tree", "building", "sign"])
    def test_exact_vocabulary_hits(self, name):
        assert _vg_demo_matcher()(name) is True

    @pytest.mark.parametrize("name", ["trees", "buildings", "signs"])
    def test_naive_plurals_fold_onto_the_singular(self, name):
        assert _vg_demo_matcher()(name) is True

    @pytest.mark.parametrize("name", ["  Tree  ", "TREE"])
    def test_case_and_whitespace_are_normalized(self, name):
        assert _vg_demo_matcher()(name) is True

    @pytest.mark.parametrize("name", ["quasar", "", None])
    def test_out_of_vocabulary_names_miss(self, name):
        assert _vg_demo_matcher()(name) is False

    def test_list_valued_names_match_on_any_member(self):
        assert _vg_demo_matcher()(["quasar", "tree"]) is True
        assert _vg_demo_matcher()(["quasar", "nebula"]) is False
