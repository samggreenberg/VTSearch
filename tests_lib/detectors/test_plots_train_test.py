"""Confusion-matrix metrics + AUROC for the train/test sweep's plots.

``accuracy``/``balanced_accuracy``/``precision``/``recall`` are *derived* from fields
already on every row (``fpr``, ``fnr``, ``n_test``, ``n_test_pos``) rather than
recomputed from scores, so they appear on results.jsonl files written before this
existed. That only holds if the integer confusion counts come back exactly from rates
stored at 6 decimals - the first test class pins precisely that.

``auroc`` cannot be derived (it needs the score vector), so it is computed in
``region_curve._oracle_extra`` via :func:`vtscore.eval.error_metrics.roc_auc`, whose
tie handling is the part worth pinning: image scores are max-pools over regions, so
ties are routine and a degenerate all-equal head must read 0.5, not 0 or 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from vtscore.eval.error_metrics import f1_at, roc_auc, weighted_error

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sod"))

import plots_train_test as ptt  # noqa: E402


def _row_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    """A results row exactly as the sweep would write it for this confusion matrix."""
    p, n = tp + fn, fp + tn
    return {
        "dataset": "d",
        "class": "c",
        "embedder": "e",
        "proposal": "whole",
        "head": "mlp",
        "seed": 0,
        "k": 1,
        "fpr": round(fp / n, 6),
        "fnr": round(fn / p, 6),
        "n_test": p + n,
        "n_test_pos": p,
    }


class TestExactRecovery:
    """The derivation is only legitimate if the 6-dp rates rebuild the counts exactly."""

    @pytest.mark.parametrize(
        ("tp", "fp", "tn", "fn"),
        [(274, 0, 770, 0), (1, 1, 1, 1), (137, 385, 385, 137), (3, 997, 2, 998), (999, 1, 1, 999)],
    )
    def test_counts_round_trip(self, tp, fp, tn, fn):
        got = ptt._confusion(_row_from_counts(tp, fp, tn, fn))
        assert got == (tp, fp, tn, fn)

    def test_round_trips_at_a_large_test_set(self):
        """5e-7 * N must stay under 0.5, i.e. exact for any N below a million."""
        tp, fp, tn, fn = 123_456, 7_891, 300_000, 54_321
        assert ptt._confusion(_row_from_counts(tp, fp, tn, fn)) == (tp, fp, tn, fn)


class TestDerivedValues:
    def test_matches_textbook_definitions(self):
        tp, fp, tn, fn = 60, 20, 100, 40
        m = ptt.derived_metrics(_row_from_counts(tp, fp, tn, fn))
        assert m["recall"] == pytest.approx(tp / (tp + fn))
        assert m["precision"] == pytest.approx(tp / (tp + fp))
        assert m["accuracy"] == pytest.approx((tp + tn) / (tp + fp + tn + fn))
        tpr, tnr = tp / (tp + fn), tn / (tn + fp)
        assert m["balanced_accuracy"] == pytest.approx((tpr + tnr) / 2)

    def test_derived_precision_recall_reproduce_the_stored_f1(self):
        """Cross-check against an independently computed f1_at on real scores."""
        rng = np.random.default_rng(0)
        scores = np.concatenate([rng.normal(1.0, 1.0, 200), rng.normal(0.0, 1.0, 800)])
        labels = np.array([1.0] * 200 + [0.0] * 800)
        thr = 0.5
        err = weighted_error(scores, labels, thr)
        row = {
            "fpr": err["fpr"], "fnr": err["fnr"],
            "n_test": len(labels), "n_test_pos": int(labels.sum()),
        }  # fmt: skip
        m = ptt.derived_metrics(row)
        prec, rec = m["precision"], m["recall"]
        assert 2 * prec * rec / (prec + rec) == pytest.approx(f1_at(scores, labels, thr), abs=1e-6)

    def test_all_positive_collapse_reads_as_expected(self):
        """The degenerate case seen on vg_s: FNR 0, FPR 1 -> recall 1, accuracy = prevalence."""
        m = ptt.derived_metrics(_row_from_counts(tp=274, fp=770, tn=0, fn=0))
        assert m["recall"] == pytest.approx(1.0)
        assert m["balanced_accuracy"] == pytest.approx(0.5)
        assert m["accuracy"] == pytest.approx(274 / 1044)

    def test_precision_is_nan_when_nothing_is_predicted_positive(self):
        m = ptt.derived_metrics(_row_from_counts(tp=0, fp=0, tn=770, fn=274))
        assert m["precision"] != m["precision"]  # NaN, matching f1_at's convention
        assert m["recall"] == pytest.approx(0.0)

    def test_unreconstructable_row_yields_all_nan(self):
        m = ptt.derived_metrics({"fpr": 0.1})  # no fnr / counts
        assert all(v != v for v in m.values())


class TestEnrichRows:
    def test_adds_fields_without_mutating_the_input(self):
        row = _row_from_counts(60, 20, 100, 40)
        original = dict(row)
        out = ptt.enrich_rows([row])
        assert row == original
        assert set(ptt.DERIVED_METRICS) <= set(out[0])

    def test_is_idempotent(self):
        rows = ptt.enrich_rows([_row_from_counts(60, 20, 100, 40)])
        assert ptt.enrich_rows(rows) == rows

    def test_never_overwrites_a_field_the_row_already_has(self):
        row = {**_row_from_counts(60, 20, 100, 40), "precision": 0.123}
        assert ptt.enrich_rows([row])[0]["precision"] == 0.123


class TestRocAuc:
    def test_perfect_and_inverted_separation(self):
        y = [1.0, 1.0, 0.0, 0.0]
        assert roc_auc([3.0, 2.0, 1.0, 0.0], y) == pytest.approx(1.0)
        assert roc_auc([0.0, 1.0, 2.0, 3.0], y) == pytest.approx(0.0)

    def test_all_scores_tied_reads_as_chance(self):
        """A degenerate head that scores everything the same must be 0.5, not 0 or 1."""
        assert roc_auc([1.0] * 6, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.5)

    def test_partial_ties_use_midranks(self):
        # One positive above, one tied with the negative: 1.0 (win) + 0.5 (tie) over 2 pairs.
        assert roc_auc([2.0, 1.0, 1.0], [1.0, 1.0, 0.0]) == pytest.approx(0.75)

    def test_matches_the_mann_whitney_definition_on_random_data(self):
        rng = np.random.default_rng(7)
        s = rng.normal(size=300)
        y = (rng.random(300) < 0.3).astype(float)
        pos, neg = s[y == 1.0], s[y == 0.0]
        brute = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
        assert roc_auc(s, y) == pytest.approx(brute, abs=1e-6)

    def test_is_invariant_to_monotone_rescaling(self):
        rng = np.random.default_rng(1)
        s = rng.normal(size=200)
        y = (rng.random(200) < 0.4).astype(float)
        assert roc_auc(s, y) == pytest.approx(roc_auc(3.0 * s + 10.0, y))

    @pytest.mark.parametrize("labels", [[1.0, 1.0], [0.0, 0.0]])
    def test_single_class_is_nan(self, labels):
        assert roc_auc([1.0, 2.0], labels) != roc_auc([1.0, 2.0], labels)

    def test_empty_is_nan(self):
        assert roc_auc([], []) != roc_auc([], [])

    def test_non_finite_scores_are_dropped(self):
        clean = roc_auc([3.0, 2.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])
        withnan = roc_auc([3.0, 2.0, 1.0, 0.0, float("nan")], [1.0, 1.0, 0.0, 0.0, 1.0])
        assert withnan == pytest.approx(clean)


class TestRenderAll:
    @staticmethod
    def _rows():
        out = []
        for seed in range(3):
            for k in (1, 2, 3):
                r = _row_from_counts(60 + seed, 20, 100, 40 - seed)
                r.update(seed=seed, k=k, cost=0.3, f1=0.5, auroc=0.8)
                out.append(r)
        return out

    def test_all_plus_std_emits_two_files_per_metric(self, tmp_path):
        ptt.render_all(self._rows(), tmp_path, metrics=("cost", "recall"), band="all+std")
        names = sorted(p.name for p in tmp_path.glob("*.png"))
        assert names == [
            "d_c_cost_vs_t.png",
            "d_c_cost_vs_t_summary.png",
            "d_c_recall_vs_t.png",
            "d_c_recall_vs_t_summary.png",
        ]

    def test_single_band_emits_one_file_per_metric(self, tmp_path):
        ptt.render_all(self._rows(), tmp_path, metrics=("cost", "recall"), band="std")
        assert sorted(p.name for p in tmp_path.glob("*.png")) == ["d_c_cost_vs_t.png", "d_c_recall_vs_t.png"]

    def test_derived_metrics_are_plottable_without_being_in_the_rows(self, tmp_path):
        rows = self._rows()
        assert "precision" not in rows[0]
        ptt.render_all(rows, tmp_path, metrics=("precision",), band="std")
        assert (tmp_path / "d_c_precision_vs_t.png").exists()

    def test_unknown_band_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="unknown band"):
            ptt.render_all(self._rows(), tmp_path, band="nope")

    def test_default_metric_list_covers_the_request(self):
        for m in ("accuracy", "balanced_accuracy", "precision", "recall", "auroc"):
            assert m in ptt.TRAIN_TEST_METRICS

    def test_reference_lines_reach_both_files_of_an_all_plus_std_pair(self, tmp_path):
        ref = {"cost": 0.25, "recall": 0.9}
        ptt.render_all(self._rows(), tmp_path, metrics=("cost", "recall"), band="all+std", reference=ref)
        assert len(list(tmp_path.glob("*.png"))) == 4  # rendering with a reference must not fail

    def test_reference_for_an_unplotted_metric_is_ignored(self, tmp_path):
        ptt.render_all(self._rows(), tmp_path, metrics=("cost",), band="std", reference={"auroc": 0.7})
        assert (tmp_path / "d_c_cost_vs_t.png").exists()


class TestReferenceCsv:
    """``metric,value`` with no header, any subset, any order (per the example file)."""

    @staticmethod
    def _write(tmp_path, text):
        p = tmp_path / "reference.csv"
        p.write_text(text, encoding="utf-8")
        return p

    def test_reads_the_documented_example_shape(self, tmp_path):
        p = self._write(
            tmp_path,
            "fpr,.1\nfnr,.2\nf1,.3\naccuracy,.4\nbalanced_accuracy,.5\nprecision,.6\nrecall,.7\nauroc,.8\n",
        )
        assert ptt.load_reference_csv(p) == {
            "fpr": 0.1, "fnr": 0.2, "f1": 0.3, "accuracy": 0.4,
            "balanced_accuracy": 0.5, "precision": 0.6, "recall": 0.7, "auroc": 0.8,
        }  # fmt: skip

    def test_the_real_example_file_parses(self):
        p = Path(__file__).resolve().parents[2] / "data" / "cats" / "reference" / "reference.csv"
        if not p.exists():
            pytest.skip("example reference.csv not present in this checkout")
        ref = ptt.load_reference_csv(p)
        assert ref and all(isinstance(v, float) for v in ref.values())
        assert set(ref) <= set(ptt.TRAIN_TEST_METRICS)

    def test_subset_and_arbitrary_order(self, tmp_path):
        assert ptt.load_reference_csv(self._write(tmp_path, "auroc,0.8\nfpr,0.1\n")) == {"auroc": 0.8, "fpr": 0.1}

    def test_names_match_case_space_and_hyphen_insensitively(self, tmp_path):
        p = self._write(tmp_path, "Balanced Accuracy,0.5\nBALANCED-accuracy,0.6\n")
        assert ptt.load_reference_csv(p) == {"balanced_accuracy": 0.6}  # later wins

    def test_blank_lines_and_comments_are_skipped(self, tmp_path):
        assert ptt.load_reference_csv(self._write(tmp_path, "\n# a note\nfpr,0.1\n\n")) == {"fpr": 0.1}

    def test_header_row_is_tolerated(self, tmp_path):
        assert ptt.load_reference_csv(self._write(tmp_path, "metric,value\nfpr,0.1\n")) == {"fpr": 0.1}

    def test_whitespace_around_values_is_fine(self, tmp_path):
        assert ptt.load_reference_csv(self._write(tmp_path, " recall , 0.7 \n")) == {"recall": 0.7}

    def test_non_float_value_is_an_error_not_a_silent_drop(self, tmp_path):
        """The caller guarantees a float, so a bad one means a malformed file."""
        with pytest.raises(ValueError, match="is not a float"):
            ptt.load_reference_csv(self._write(tmp_path, "fpr,0.1\nrecall,high\n"))

    def test_missing_value_column_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="expected 'metric,value'"):
            ptt.load_reference_csv(self._write(tmp_path, "fpr\n"))

    def test_unknown_metric_is_kept_and_warned(self, tmp_path, capsys):
        assert ptt.load_reference_csv(self._write(tmp_path, "not_a_metric,0.5\n")) == {"not_a_metric": 0.5}
        assert "unknown metric" in capsys.readouterr().out

    def test_every_default_metric_name_is_accepted(self, tmp_path, capsys):
        p = self._write(tmp_path, "".join(f"{m},0.5\n" for m in ptt.TRAIN_TEST_METRICS))
        assert set(ptt.load_reference_csv(p)) == set(ptt.TRAIN_TEST_METRICS)
        assert "unknown metric" not in capsys.readouterr().out
