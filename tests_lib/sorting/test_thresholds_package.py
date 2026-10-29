"""The ``vtscore.training.thresholds`` package contract (issue #3381).

The module was one 2680-line file until it was split along its five seams.
Three things had to survive that split and are pinned here:

1. **Every name still resolves off the package.**  Callers, docs and the eval
   harness all say ``vtscore.training.thresholds.X``; the ``__init__``
   re-export is what keeps that true, and it is easy to forget a name when a
   submodule grows one.
2. **The layering stays a DAG.**  ``knobs -> {gmm, fold_orderings} ->
   {anchored, conformal} -> blend`` is the whole point of the split: the
   mixture layer knows nothing of folds, the fold fitting knows nothing of the
   conformal rule that reads it (#5428), and the two rival estimators know
   nothing of each other.  A back edge would quietly restore the tangle.  The
   mixture layer is itself four layers since #5424 (``gmm_core -> gmm_em ->
   gmm_threshold -> text_sort``), with ``gmm`` a re-export facade over them
   that must keep resolving every name.
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
    fold_orderings,
    gmm,
    gmm_core,
    gmm_em,
    gmm_threshold,
    knobs,
    text_sort,
)

#: The layer each submodule sits in.  A submodule may import from a *lower*
#: number only; ``anchored`` and ``conformal`` share a rank because they are
#: rival estimators rather than layers, and neither may import the other.
#: ``fold_orderings`` shares ``gmm_core``'s: it reads only the knobs.
LAYER = {
    "knobs": 0,
    "gmm_core": 1,
    "fold_orderings": 1,
    "gmm_em": 2,
    "gmm_threshold": 3,
    "text_sort": 4,
    "gmm": 5,
    "anchored": 6,
    "conformal": 6,
    "blend": 7,
}

#: The four modules the mixture was split into (#5424), in layer order; ``gmm``
#: re-exports them all.
MIXTURE_LAYERS = {"gmm_core": gmm_core, "gmm_em": gmm_em, "gmm_threshold": gmm_threshold, "text_sort": text_sort}

#: The submodules that define names.  The ``gmm`` facade defines none of its
#: own, so its contract is pinned separately (:class:`TestGmmFacade`).
SUBMODULES = {
    "knobs": knobs,
    **MIXTURE_LAYERS,
    "fold_orderings": fold_orderings,
    "anchored": anchored,
    "conformal": conformal,
    "blend": blend,
}

PKG_DIR = Path(thresholds.__file__).parent


def _top_level_names(mod_name: str) -> list[str]:
    """Every name *mod_name*'s source binds at module level (defs, classes, assignments)."""
    tree = ast.parse((PKG_DIR / f"{mod_name}.py").read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names.extend(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return names


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
            for name in _top_level_names(mod_name):
                assert name in exported, f"{mod_name}.{name} is not re-exported from the package __init__"
                assert getattr(thresholds, name) is getattr(mod, name)


class TestGmmFacade:
    """``thresholds.gmm.X`` outlived the #5424 split, for every name the four mixture layers define."""

    def test_every_mixture_name_resolves_off_the_facade(self):
        defined: set[str] = set()
        for mod_name, mod in MIXTURE_LAYERS.items():
            for name in _top_level_names(mod_name):
                defined.add(name)
                assert getattr(gmm, name) is getattr(mod, name), f"gmm.{name} is not {mod_name}.{name}"
        assert set(gmm.__all__) == defined

    def test_the_facade_defines_nothing_of_its_own(self):
        """A body here would be a second home for a name, and patching it would reach no caller."""
        assert _top_level_names("gmm") == ["__all__"]


class TestConformalReExportsTheFolds:
    def test_every_public_fold_name_imports_from_conformal(self):
        """The fold code left ``conformal`` (#5428); its public names still import from there."""
        tree = ast.parse((PKG_DIR / "fold_orderings.py").read_text(encoding="utf-8"))
        public = [
            name
            for node in tree.body
            for name in (
                [node.name]
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                else [t.id for t in node.targets if isinstance(t, ast.Name)]
                if isinstance(node, ast.Assign)
                else []
            )
            if not name.startswith("_")
        ]
        assert "compute_fold_orderings" in public
        for name in public:
            assert getattr(conformal, name, None) is getattr(fold_orderings, name), name


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

    def test_gmm_layer_reads_only_itself_and_the_knobs(self):
        """The mixture layers import no sibling outside themselves but the knobs.

        At the split (#3381) the mixture imported no sibling at all, and this
        test pinned that.  It always read ``scored_mask``, though, which lived
        outside the package in ``vtscore.utils.scores``; #5399 folded that into
        the knobs, beside the other sentinels.  The knobs import nothing, so the
        edge cannot carry folds or an estimator into the mixture layers.
        """
        allowed = {f"vtscore.training.thresholds.{name}" for name in (*MIXTURE_LAYERS, "knobs")}
        for mod_name in (*MIXTURE_LAYERS, "gmm"):
            tree = ast.parse((PKG_DIR / f"{mod_name}.py").read_text(encoding="utf-8"))
            siblings = {
                n.module
                for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("vtscore.training.thresholds")
            }
            assert siblings <= allowed, f"{mod_name} imports {sorted(siblings - allowed)}"


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
