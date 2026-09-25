"""A pile dataset the calibration config half-knows runs the wrong arm silently.

`experiment_config` carries three per-dataset tables and `pile_config.DATASETS`
carries the truth. Nothing held them together, and the failure is asymmetric:

* A dataset missing from `BOXED_BY_DATASET` reads as boxless, so `styles_for`
  falls a patch embedder back to `whole_image` **without a word** -- a region arm
  that binary-votes while every output says region voting. That cost a full
  108-cell arm once already (see the note on `vg_scale` in that table).
* A dataset missing from `EXPERIMENT_QUERIES` opens on three random known-goods
  instead of a text sort, which is the seeding confound #3278 added the region
  pair to remove, entering from the other side.

Both are invisible in the rows. So the registry is pinned here rather than in a
comment asking the next person to remember.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CALIB = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "calibration"
_PILE = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def cfg():
    for p in (_CALIB, _PILE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import experiment_config

    return experiment_config


@pytest.fixture(scope="module")
def pc():
    if str(_PILE) not in sys.path:
        sys.path.insert(0, str(_PILE))
    import pile_config

    return pile_config


def _runnable(cfg, pc) -> list[str]:
    """Pile datasets the calibration harness can actually enumerate cells for.

    A dataset it has never heard of is not a gap -- `coco_better_full` and its
    shards are export sources, not study environments, and nothing should run a
    123,287-image cell. The claim here is narrower and is the one that bites:
    a dataset the config *does* know must be described consistently by all three
    tables.
    """
    return [ds for ds in pc.DATASETS if ds in cfg.DATASET_EMBEDDERS]


class TestTheTablesAgreeWithThePile:
    def test_a_roster_for_a_retired_dataset_is_not_an_error(self, cfg, pc):
        """The reverse direction is deliberately NOT asserted, and this records why.

        `vg_scale*` and `vg_box_*` still carry rosters while `pile_config` no
        longer builds them: #4038 retired Visual Genome by making those cells
        readable-by-path and unrebuildable, precisely so the studies conditioned
        on them keep running. A roster naming a dataset the pile cannot build is
        therefore a *supported* state, and a test demanding the two sets match
        would fail on the design rather than on a defect.

        Pinned as a test rather than a comment because the natural thing to write
        here is the symmetric check, and it would look correct.
        """
        retired = sorted(set(cfg.DATASET_EMBEDDERS) - set(pc.DATASETS))
        assert "vg_scale" in retired, (
            "vg_scale is back in pile_config.DATASETS -- if Visual Genome was un-retired, "
            "this test's premise needs re-reading, not deleting"
        )

    def test_a_boxed_pile_dataset_is_marked_boxed(self, cfg, pc):
        """The silent one. Missing here means a patch arm binary-votes."""
        missing = [ds for ds in _runnable(cfg, pc) if pc.DATASETS[ds].get("boxed") and not cfg.BOXED_BY_DATASET.get(ds)]
        assert not missing, (
            f"{missing} are boxed in pile_config and not in BOXED_BY_DATASET; "
            "a patch embedder there falls back to whole_image with no warning"
        )

    def test_nothing_claims_boxes_the_pile_does_not_build(self, cfg, pc):
        wrong = [
            ds
            for ds, boxed in cfg.BOXED_BY_DATASET.items()
            if boxed and ds in pc.DATASETS and not pc.DATASETS[ds].get("boxed")
        ]
        assert not wrong, f"{wrong} are marked boxed but the pile does not build boxes for them"


class TestCocoBetterIsRunnable:
    """#4051 needs it to be; #4044's plumbing is useless on a dataset the grid
    cannot enumerate."""

    def test_it_has_an_embedder_roster(self, cfg):
        assert cfg.DATASET_EMBEDDERS["coco_better"]

    def test_the_region_arm_is_the_pair_not_the_bare_patch_embedder(self, cfg):
        """Bare `dinov3_patch` has no text tower, so it opens on known-goods while
        the whole-image arms open on a sort -- a seeding difference inside the
        voting-mode axis (#3276, #3278)."""
        roster = cfg.DATASET_EMBEDDERS["coco_better"]
        patch = [e for e in roster if cfg.is_patch_embedder(e)]
        assert patch, "no region arm at all"
        for e in patch:
            assert cfg.text_embedder(e), f"{e} cannot open on a text sort"

    def test_region_voting_is_really_on_for_the_patch_arm(self, cfg):
        for e in cfg.DATASET_EMBEDDERS["coco_better"]:
            if cfg.is_patch_embedder(e):
                assert cfg.region_voting_for("coco_better", e)

    def test_every_cell_has_a_typed_query(self, cfg, pc):
        """`CALIB_CATEGORY_MODE=all` designates all 75; a cell with no query would
        silently open the other way."""
        queries = cfg.EXPERIMENT_QUERIES["coco_better"]
        expected = {pc.scale_cell(cls, band) for cls in pc.SCALE_CLASSES for band in pc.BOX_BANDS}
        assert expected <= set(queries), f"no query for {sorted(expected - set(queries))}"

    def test_it_shares_vg_scales_queries_exactly(self, cfg):
        """The two sets exist to be read against each other, so a query that
        drifted between them would put a seeding axis inside the source axis."""
        assert cfg.EXPERIMENT_QUERIES["coco_better"] == cfg.EXPERIMENT_QUERIES["vg_scale"]
