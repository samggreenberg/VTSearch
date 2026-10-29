"""The ``vtscore.training.thresholds`` package contract (issue #3381).

The module was one 2680-line file until it was split along its five seams.
Three things had to survive that split and are pinned here:

1. **Every name still resolves off the package.**  Callers, docs and the eval
   harness all say ``vtscore.training.thresholds.X``; the ``__init__``
   re-export is what keeps that true, and it is easy to forget a name when a
   submodule grows one.
2. **The layering stays a DAG.**  ``knobs -> gmm -> {anchored, conformal} ->
   blend`` is the whole point of the split: the mixture layer knows nothing of
   folds, and the two rival estimators know nothing of each other.  A back edge
   would quietly restore the tangle.
3. **``calculate_gmm_threshold`` is ``fit_gmm_threshold`` with the fit
   discarded.**  The two carried byte-identical bodies before the split; the
   delegation is what stops them drifting apart again.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from vtscore.training import thresholds
from vtscore.training.thresholds import (
    anchored,
    blend,
    calculate_gmm_threshold,
    conformal,
    fit_gmm_threshold,
    gmm,
    knobs,
)

#: The layer each submodule sits in.  A submodule may import from a *lower*
#: number only; ``anchored`` and ``conformal`` share a rank because they are
#: rival estimators rather than layers, and neither may import the other.
LAYER = {"knobs": 0, "gmm": 1, "anchored": 2, "conformal": 2, "blend": 3}

SUBMODULES = {"knobs": knobs, "gmm": gmm, "anchored": anchored, "conformal": conformal, "blend": blend}

PKG_DIR = Path(thresholds.__file__).parent


class TestPackageReExport:
    def test_every_exported_name_resolves(self):
        for name in thresholds.__all__:
            assert hasattr(thresholds, name), f"{name} is in __all__ but not importable off the package"

    def test_every_submodule_top_level_name_is_exported(self):
        """No submodule may grow a top-level name the package does not re-export.

        Anything importable from the old single module stayed importable; this
        keeps that true for names added later, private ones included - the eval
        harness and the tests reach for ``_rate_cut`` and ``_GMM_MAX_SAMPLES``.
        """
        exported = set(thresholds.__all__)
        for mod_name, mod in SUBMODULES.items():
            tree = ast.parse((PKG_DIR / f"{mod_name}.py").read_text(encoding="utf-8"))
            for node in tree.body:
                names: list[str] = []
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names = [node.name]
                elif isinstance(node, ast.Assign):
                    names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    names = [node.target.id]
                for name in names:
                    assert name in exported, f"{mod_name}.{name} is not re-exported from the package __init__"
                    assert getattr(thresholds, name) is getattr(mod, name)


class TestLayering:
    def test_no_back_edges(self):
        for mod_name, layer in LAYER.items():
            tree = ast.parse((PKG_DIR / f"{mod_name}.py").read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                if not node.module.startswith("vtscore.training.thresholds"):
                    continue
                target = node.module.rsplit(".", 1)[-1]
                assert target in LAYER, f"{mod_name} imports unknown sibling {node.module}"
                assert LAYER[target] < layer, f"{mod_name} imports {target}, which is not below it"

    def test_gmm_layer_is_self_contained(self):
        """The mixture layer imports no sibling at all - not even the knobs."""
        tree = ast.parse((PKG_DIR / "gmm.py").read_text(encoding="utf-8"))
        siblings = [
            n.module
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("vtscore.training.thresholds")
        ]
        assert siblings == []


def _bimodal(n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    lo = rng.normal(0.2, 0.05, size=n // 2)
    hi = rng.normal(0.8, 0.05, size=n - n // 2)
    return np.clip(np.concatenate([lo, hi]), 0.0, 1.0).tolist()


class TestGmmThresholdDelegation:
    @pytest.mark.parametrize(
        "scores",
        [
            [],
            [0.4],
            [0.1, 0.9],
            _bimodal(400, 11),
            [0.5] * 50,  # degenerate: the fit collapses and both fall back
            [float("nan")] * 4,
        ],
        ids=["empty", "single", "pair", "bimodal", "degenerate", "nan"],
    )
    def test_matches_the_fitting_variant_bit_for_bit(self, scores):
        cut, _fit = fit_gmm_threshold(scores)
        got = calculate_gmm_threshold(scores)
        assert (np.isnan(got) and np.isnan(cut)) or got == cut
