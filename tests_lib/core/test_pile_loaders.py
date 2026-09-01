"""Every pile dataset kind has one module owning both halves of it.

``build_pile.py`` used to carry two switches on ``DATASETS[ds]["kind"]``, a
thousand lines apart: one deciding how to build a cell, the other deciding what
a rebuild of it would read. They drifted, and the drift is #3299 -- the canary
checked ``COCO_IMAGES`` while the builder opened ``val2017.zip`` inline, and
reported ``coco_val`` REBUILD-BROKEN against a staging area that was entirely
intact. A false alarm costs what a true alarm costs and is spent on nothing;
a canary that cries wolf on its first real run is one people learn to skip.

So the two live in one module per kind now, and these tests pin that: a kind in
``pile_config.DATASETS`` with no module fails here rather than at 3am on the
GRID, and a module missing either half fails too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def loaders():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import pilebuild.loaders as mod

    return mod


@pytest.fixture(scope="module")
def pc():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import pile_config

    return pile_config


def test_every_declared_kind_has_a_loader(loaders, pc):
    declared = {spec.get("kind") for spec in pc.DATASETS.values()}
    assert declared <= set(loaders.LOADERS), (
        f"pile_config.DATASETS declares kinds with no pilebuild/loaders/ module: "
        f"{sorted(declared - set(loaders.LOADERS))}"
    )


def test_every_loader_owns_both_halves(loaders):
    for kind, module in loaders.LOADERS.items():
        assert callable(getattr(module, "load", None)), f"{kind}: no load()"
        assert callable(getattr(module, "check", None)), f"{kind}: no check()"


def test_every_dataset_resolves_to_its_module(loaders, pc):
    for ds, spec in pc.DATASETS.items():
        assert loaders.loader_for(ds, spec.get("kind")) is loaders.LOADERS[spec["kind"]]


def test_an_unknown_kind_fails_loudly(loaders):
    """It must not fall through to the demo loader.

    The old dispatch ended in ``else: _load_demo(...)``, so a typo in
    ``DATASETS`` built a plausible-looking cell out of the wrong source.
    """
    with pytest.raises(SystemExit) as exc:
        loaders.loader_for("made_up", "not_a_kind")
    assert "not_a_kind" in str(exc.value)
    assert "pilebuild/loaders/" in str(exc.value)


def test_build_pile_still_exports_what_its_callers_import():
    """Sibling scripts do ``from build_pile import _vg_image_paths`` and friends.

    Five ``pile/make_*.py`` sheet builders plus ``precision/`` and ``fastproc/``
    ``build_arm.py`` import off this module, and none of them are covered by a
    test that would run here -- so the surface is pinned by reading it.
    """
    source = (_PILE_DIR / "build_pile.py").read_text()
    for name in ("_vg_image_paths", "build_cell", "assert_vtscore_is_this_checkout", "_band_categories"):
        assert name in source, f"build_pile no longer provides {name}"
