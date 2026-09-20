"""Cross-band testing and train-side size mixes (#4044).

A cell is ``class@band`` and ``media_is_evaluable`` drops an image holding the
class at another size, so until now a cell could only be tested at the size it
was trained at.  These pin the three properties that make the new breakdown
trustworthy: the cohorts are the ones the diagonal arm holds out (so a 3x3 table
pairs), a mix is a retag that leaves every downstream consumer alone, and the
headline columns do not move.
"""

from __future__ import annotations

import pytest

from vtscore.eval import scale_bands as sb
from vtscore.eval.labels import (
    evaluable_pool,
    media_is_evaluable,
    media_is_positive,
    region_box_for_category,
)
from vtscore.eval.voting_iterations import _split_media_ids

BANDS = ("small", "medium", "large")


def _medias(per_band=(10, 14, 6), n_neg=40):
    """One class at three sizes against a pool of negatives all three share."""
    medias: dict[int, dict] = {}
    cells = [f"car@{b}" for b in BANDS]
    nxt = 1
    for band, n in zip(BANDS, per_band):
        cell = f"car@{band}"
        for _ in range(n):
            medias[nxt] = {
                "id": nxt,
                "category": cell,
                "categories": [cell],
                # A positive is scorable only in its own cell -- the same class
                # at another size is neither a positive nor a negative there.
                "evaluable_categories": [cell],
                "regions": [{"box": [0.1, 0.1, 0.2, 0.2], "label": cell}],
            }
            nxt += 1
    for _ in range(n_neg):
        medias[nxt] = {"id": nxt, "category": "", "categories": [], "evaluable_categories": list(cells)}
        nxt += 1
    return medias


class TestParsing:
    def test_a_banded_cell_splits_into_class_and_band(self):
        assert sb.parse_cell("car@small") == ("car", "small")
        assert sb.parse_cell("stop sign@large") == ("stop sign", "large")

    def test_an_unbanded_category_has_no_band(self):
        """Every non-scale dataset takes this path; cross-band testing is off."""
        assert sb.parse_cell("dog") == ("dog", None)

    def test_cells_are_read_off_the_data_not_spelled(self):
        assert sb.cells_of_class(_medias(), "car") == ["car@large", "car@medium", "car@small"]

    def test_a_band_with_no_reported_column_is_named(self):
        medias = _medias()
        medias[999] = {"id": 999, "categories": ["car@enormous"], "evaluable_categories": ["car@enormous"]}
        assert sb.unreportable_bands(medias, "car") == ["enormous"]


class TestCohorts:
    def test_the_own_band_cohort_is_the_harness_holdout(self):
        """The load-bearing one: a replayed split that drifts would leak training
        images into an off-diagonal cohort with nothing failing."""
        import numpy as np

        medias = _medias()
        for cell in ("car@small", "car@medium", "car@large"):
            pool = evaluable_pool(medias, cell)
            _, test_ids = _split_media_ids(pool, 0.5, np.random.RandomState(7))
            expected = sorted(cid for cid in test_ids if media_is_positive(pool[cid], cell))
            band = sb.parse_cell(cell)[1]
            assert band is not None
            got = sb.band_cohorts(medias, cell, sim_fraction=0.5, seed=7)
            assert sorted(got[band]) == expected

    def test_a_bands_cohort_does_not_depend_on_which_arm_asks(self):
        """What makes the 3x3 a matrix: one column, one set of images."""
        medias = _medias()
        from_small = sb.band_cohorts(medias, "car@small", sim_fraction=0.5, seed=3)
        from_large = sb.band_cohorts(medias, "car@large", sim_fraction=0.5, seed=3)
        assert sorted(from_small["medium"]) == sorted(from_large["medium"])

    def test_every_cohort_member_is_a_positive_of_its_own_band(self):
        medias = _medias()
        cohorts = sb.band_cohorts(medias, "car@small", sim_fraction=0.5, seed=1)
        for band, ids in cohorts.items():
            assert ids
            assert all(media_is_positive(medias[cid], f"car@{band}") for cid in ids)

    def test_negatives_never_enter_a_cohort(self):
        """They are shared across the bands, so they belong to the run's one FPR."""
        medias = _medias()
        cohorts = sb.band_cohorts(medias, "car@small", sim_fraction=0.5, seed=1)
        every = {cid for ids in cohorts.values() for cid in ids}
        assert all(medias[cid]["categories"] for cid in every)

    def test_an_unbanded_target_has_no_cohorts(self):
        medias = {1: {"categories": ["dog"], "evaluable_categories": ["dog"]}}
        assert sb.band_cohorts(medias, "dog", sim_fraction=0.5, seed=1) == {}


class TestNaturalMix:
    def test_the_shares_are_the_corpus_band_counts(self):
        mix = sb.natural_mix(_medias(per_band=(10, 14, 6)), "car")
        assert mix == pytest.approx({"small": 1 / 3, "medium": 14 / 30, "large": 0.2})

    def test_shares_sum_to_one(self):
        assert sum(sb.natural_mix(_medias(), "car").values()) == pytest.approx(1.0)

    def test_weights_are_normalised_however_they_are_written(self):
        medias = _medias()
        assert sb.resolve_mix(medias, "car", {"small": 1, "large": 3}) == pytest.approx(
            sb.resolve_mix(medias, "car", {"small": 0.25, "large": 0.75})
        )


class TestProjectMix:
    def test_the_quota_matches_the_requested_shares(self):
        medias = _medias(per_band=(40, 40, 40))
        out, report = sb.project_mix(medias, "car", {"small": 1, "medium": 2, "large": 1}, n_positives=20)
        assert report["positives_by_band"] == {"small": 5, "medium": 10, "large": 5}
        assert report["n_positives"] == 20
        assert sum(1 for m in out.values() if media_is_positive(m, "car@mix")) == 20

    def test_the_realised_size_is_exactly_what_was_asked(self):
        """Largest-remainder, so a mixed arm is comparable with the pure ones."""
        medias = _medias(per_band=(40, 40, 40))
        _, report = sb.project_mix(medias, "car", {"small": 1, "medium": 1, "large": 1}, n_positives=100)
        assert report["n_positives"] == 100

    def test_a_band_that_cannot_fill_its_share_is_reported_not_papered_over(self):
        medias = _medias(per_band=(2, 40, 40))
        _, report = sb.project_mix(medias, "car", {"small": 1, "medium": 1, "large": 1}, n_positives=30)
        assert report["positives_by_band"]["small"] == 2
        assert report["realised_mix"]["small"] < report["requested_mix"]["small"]

    def test_negatives_carry_over_and_stay_negatives(self):
        medias = _medias()
        out, report = sb.project_mix(medias, "car", "natural", n_positives=12)
        negatives = [cid for cid, m in out.items() if not media_is_positive(m, "car@mix")]
        assert len(negatives) == report["n_negatives"] == 40
        assert all(media_is_evaluable(out[cid], "car@mix") for cid in negatives)

    def test_an_undrawn_positive_is_excluded_rather_than_demoted(self):
        """It holds a car; calling it a negative would penalise finding one."""
        medias = _medias(per_band=(40, 40, 40))
        out, _ = sb.project_mix(medias, "car", {"small": 1}, n_positives=5)
        smalls = [cid for cid, m in medias.items() if sb.band_of(m, "car") == "small"]
        left_out = [cid for cid in smalls if cid not in out]
        assert len(left_out) == 35

    def test_region_labels_are_retagged_so_region_voting_still_finds_a_box(self):
        """An unrelabelled mix silently falls back to whole-image votes (#2877)."""
        out, _ = sb.project_mix(_medias(), "car", "natural", n_positives=9)
        positive = next(m for m in out.values() if media_is_positive(m, "car@mix"))
        assert region_box_for_category(positive, "car@mix") is not None

    def test_the_band_survives_the_retag(self):
        """Which is what lets a mixed arm still report per-band columns."""
        out, _ = sb.project_mix(_medias(), "car", "natural", n_positives=9)
        bands = {sb.band_of(m, "car") for m in out.values() if media_is_positive(m, "car@mix")}
        assert bands <= set(BANDS) and bands

    def test_the_source_pool_is_not_mutated(self):
        medias = _medias()
        before = {cid: list(m["categories"]) for cid, m in medias.items()}
        sb.project_mix(medias, "car", "natural", n_positives=9)
        assert {cid: list(m["categories"]) for cid, m in medias.items()} == before

    def test_the_draw_is_the_pure_cells_own_first_n(self):
        """A mix's small slice is a subset of `car@small`, not a second draw."""
        medias = _medias(per_band=(40, 40, 40))
        small = {cid for cid, m in medias.items() if sb.band_of(m, "car") == "small"}
        a, _ = sb.project_mix(medias, "car", {"small": 1}, n_positives=5)
        b, _ = sb.project_mix(medias, "car", {"small": 1}, n_positives=10)
        drawn_a = {cid for cid in a if media_is_positive(a[cid], "car@mix")}
        drawn_b = {cid for cid in b if media_is_positive(b[cid], "car@mix")}
        assert drawn_a < drawn_b <= small


class TestThroughTheHarness:
    """The knob end to end, and the guarantee that it moves nothing else."""

    @staticmethod
    def _banded_clips(dim=16, n_per_band=14, n_neg=45, seed=0):
        """Three bands of one class against a pool all three share.

        Each band sits a little further from the negatives, so a model trained
        on one band is genuinely better at it than at the others -- which is the
        effect the per-band columns exist to show.
        """
        import numpy as np

        rng = np.random.RandomState(seed)
        medias: dict[int, dict] = {}
        cells = [f"car@{b}" for b in BANDS]
        nxt = 1
        for offset, band in zip((1.4, 1.0, 0.6), BANDS):
            cell = f"car@{band}"
            for _ in range(n_per_band):
                emb = rng.normal(offset, 0.25, dim).astype("float32")
                medias[nxt] = {
                    "id": nxt,
                    "embeddings": {"emb": emb},
                    "category": cell,
                    "categories": [cell],
                    "evaluable_categories": [cell],
                }
                nxt += 1
        for _ in range(n_neg):
            emb = rng.normal(-1.0, 0.25, dim).astype("float32")
            medias[nxt] = {
                "id": nxt,
                "embeddings": {"emb": emb},
                "category": "",
                "categories": [],
                "evaluable_categories": list(cells),
            }
            nxt += 1
        return medias

    def _run(self, **kw):
        from vtscore.eval.voting_iterations import simulate_voting_iterations

        return simulate_voting_iterations(
            self._banded_clips(),
            "car@small",
            seed=11,
            dataset_name="banded",
            calibrate_count=1,
            **kw,
        )

    def test_off_still_emits_the_columns_as_nan(self):
        """A fixed schema, so a frame with the breakdown concatenates with one
        without it -- the shape SKYLINE_COLUMNS already takes.  NaN rather than
        0 for the counts: nobody looked is not the same as none were found."""
        import math

        from vtscore.eval.voting_columns import BAND_COLUMNS

        rows = self._run()
        assert rows
        assert set(BAND_COLUMNS) <= set(rows[0])
        assert all(math.isnan(rows[0][col]) for col in BAND_COLUMNS)

    def test_the_headline_columns_do_not_move(self):
        """The whole safety property: an arm's shipped numbers are bit-identical
        with the breakdown on and off, so no published result shifts under it."""
        plain = self._run()
        banded = self._run(test_bands="auto")
        keys = ("t", "cost", "fpr", "fnr", "auroc", "average_precision", "n_test_pos", "n_test_neg")
        assert [{k: r[k] for k in keys} for r in plain] == [{k: r[k] for k in keys} for r in banded]

    def test_every_reported_band_gets_a_column(self):
        from vtscore.eval.voting_columns import BAND_COLUMNS

        rows = self._run(test_bands="auto")
        assert rows
        assert set(BAND_COLUMNS) <= set(rows[0])
        for band in BANDS:
            assert rows[0][f"n_test_pos_{band}"] > 0

    def test_the_own_band_column_agrees_with_the_headline_fnr(self):
        """`fnr_small` on a `car@small` arm is the headline `fnr`; anything else
        means the cohort was built off the wrong pool."""
        for row in self._run(test_bands="auto"):
            assert row["fnr_small"] == pytest.approx(row["fnr"], abs=1e-6)

    def test_recall_is_the_complement_of_the_band_fnr(self):
        row = self._run(test_bands="auto")[-1]
        for band in BANDS:
            assert row[f"recall_{band}"] == pytest.approx(1.0 - row[f"fnr_{band}"], abs=1e-6)

    def test_a_named_subset_reports_only_those_bands(self):
        row = self._run(test_bands=["small", "large"])[0]
        assert row["n_test_pos_medium"] == 0
        import math

        assert math.isnan(row["fnr_medium"])

    def test_an_unbanded_target_is_refused_rather_than_silently_empty(self):
        from vtscore.eval.voting_iterations import simulate_voting_iterations

        medias = {
            cid: {**m, "categories": ["car"], "evaluable_categories": ["car"]}
            for cid, m in self._banded_clips().items()
        }
        with pytest.raises(ValueError, match="carries no band suffix"):
            simulate_voting_iterations(medias, "car", seed=1, calibrate_count=1, test_bands="auto")

    def test_prevalence_thinning_is_refused_rather_than_mis_paired(self):
        from vtscore.eval.voting_iterations import simulate_voting_iterations

        with pytest.raises(ValueError, match="cannot both be set"):
            simulate_voting_iterations(
                self._banded_clips(),
                "car@small",
                seed=1,
                calibrate_count=1,
                test_bands="auto",
                target_prevalence=0.05,
            )
