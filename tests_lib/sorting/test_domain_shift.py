"""Tests for the split-conformal kNN domain-shift report (#3348)."""

import numpy as np
import pytest

from vtscore.coverage import domain_shift_report as exported
from vtscore.coverage.domain_shift import (
    DOMAIN_SHIFT_MAX_QUERY,
    ReferenceTooSmall,
    domain_shift_report,
    typicality_pvalues,
)


def _two_block_vectors(n=200, dim=32, seed=7):
    """Unit vectors confined to the first dim//2 coordinates, two clusters."""
    rng = np.random.default_rng(seed)
    half = dim // 2
    base = rng.standard_normal((n, half)).astype(np.float32)
    base[: n // 2, 0] += 4.0
    base[n // 2 :, 1] += 4.0
    full = np.concatenate([base, np.zeros((n, half), dtype=np.float32)], axis=1)
    full /= np.linalg.norm(full, axis=1, keepdims=True)
    return full


def _foreign(n, dim=32, seed=11):
    """Unit vectors confined to the *other* half of the coordinates."""
    rng = np.random.default_rng(seed)
    half = dim // 2
    out = rng.standard_normal((n, half)).astype(np.float32)
    out[:, 0] += 4.0
    full = np.concatenate([np.zeros((n, half), dtype=np.float32), out], axis=1)
    full /= np.linalg.norm(full, axis=1, keepdims=True)
    return full


def _diffuse(n, dim=64, seed=3):
    """A spread-out, many-cluster space: the shape a patch embedder's pooled vectors have,
    where a single mean direction per region describes the points poorly (#3329: r-bar 0.61)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((40, dim))
    x = centers[rng.integers(0, 40, n)] + 1.2 * rng.standard_normal((n, dim))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


class TestTypicalityPvalues:
    def test_in_domain_pvalues_are_uniform(self):
        """The property the atlas guard lacked: about alpha of the reference's own items fall below alpha."""
        data = _diffuse(6000)
        ref, fresh = data[:4000], data[4000:]
        p = typicality_pvalues(ref, fresh)
        assert p.shape == (2000,)
        # Binomial sd at n=2000, p=0.05 is 0.005.
        assert abs(float(np.mean(p < 0.05)) - 0.05) < 0.02
        assert abs(float(np.median(p)) - 0.5) < 0.05

    def test_shifted_items_look_atypical(self):
        p = typicality_pvalues(_two_block_vectors(300), _foreign(100))
        assert float(np.median(p)) < 0.05
        assert float(np.mean(p < 0.05)) > 0.9

    def test_smallest_p_is_one_over_calibration_size(self):
        ref = _two_block_vectors(100)
        p = typicality_pvalues(ref, _foreign(5))
        assert float(p.min()) == pytest.approx(1.0 / (50 + 1))

    def test_deterministic(self):
        ref, q = _two_block_vectors(200), _two_block_vectors(50, seed=5)
        assert np.array_equal(typicality_pvalues(ref, q), typicality_pvalues(ref, q))

    def test_scale_invariant(self):
        """Inputs are normalised; a vector's norm carries nothing (#4128's un-normalised cell)."""
        ref, q = _two_block_vectors(200), _two_block_vectors(50, seed=5)
        assert np.allclose(typicality_pvalues(ref * 15.0, q * 3.0), typicality_pvalues(ref, q))

    def test_too_small_reference_refused(self):
        with pytest.raises(ReferenceTooSmall):
            typicality_pvalues(_two_block_vectors(39), _two_block_vectors(5))

    def test_width_mismatch_refused(self):
        with pytest.raises(ValueError, match="width"):
            typicality_pvalues(_two_block_vectors(100, dim=32), _two_block_vectors(5, dim=16))


class TestDomainShiftReport:
    def test_exported_from_the_package(self):
        assert exported is domain_shift_report

    def test_no_shift_not_flagged(self):
        report = domain_shift_report(_two_block_vectors(300), _two_block_vectors(200, seed=23))
        assert report["n_items"] == 200
        assert not report["shifted"]
        assert report["frac_atypical"] < 2 * report["alpha"]
        assert report["expected_atypical"] == report["alpha"] == 0.05

    def test_diffuse_space_not_flagged_on_its_own_data(self):
        """#3329's failure: the atlas guard fired on 80 % of dinov3_patch's own held-out splits."""
        data = _diffuse(5000, seed=8)
        assert not domain_shift_report(data[:4000], data[4000:])["shifted"]

    def test_full_shift_flagged(self):
        report = domain_shift_report(_two_block_vectors(300), _foreign(100, seed=29))
        assert report["shifted"]
        assert report["frac_atypical"] > 0.9
        assert report["median_pvalue"] < 0.05

    def test_partial_injection_flagged(self):
        """A foreign cluster hidden in mostly in-domain data still flags, at about its share."""
        mixed = np.concatenate([_two_block_vectors(160, seed=31), _foreign(40, seed=37)])
        report = domain_shift_report(_two_block_vectors(300), mixed)
        assert report["shifted"]
        assert 0.15 < report["frac_atypical"] < 0.35

    def test_queries_capped(self):
        data = _diffuse(DOMAIN_SHIFT_MAX_QUERY + 500, dim=16, seed=4)
        report = domain_shift_report(_diffuse(200, dim=16, seed=5), data)
        assert report["n_items"] == DOMAIN_SHIFT_MAX_QUERY

    def test_empty_matrix(self):
        report = domain_shift_report(_two_block_vectors(100), np.zeros((0, 32), dtype=np.float32))
        assert report["n_items"] == 0
        assert not report["shifted"]
