"""A boxed pile cell that lost its labels must not be built, and must not verify (#4117).

A second ``build_cell`` of ``visual_genome_m`` in the same pile root went
through the app's demo cache, which reloaded every media with no ``categories``
and no ``regions``. The cell had the right media count, vectors and patch grids,
so ``--verify`` passed it: the cross-cell count check compares cells with each
other, and every one of them agreed. These tests pin the three pieces of the
fix -- the check itself, the build refusing before it writes, and the demo
loader not reading the cache at all -- plus the provenance backfill that says a
pre-#3683 cell's batch size is unknown instead of leaving the key out.

The ``build_pile`` wiring is checked by reading the source, as in
``test_pile_provenance_batch_size``: importing it runs ``setup_env()``.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

_PILE = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"
_BUILD = _PILE / "build_pile.py"
_AUDIT = _PILE / "pilebuild" / "audit.py"
_DEMO = _PILE / "pilebuild" / "loaders" / "demo.py"


@pytest.fixture(scope="module")
def audit():
    if str(_PILE) not in sys.path:
        sys.path.insert(0, str(_PILE))
    import pilebuild.audit as mod

    return mod


def _vg(n: int = 3) -> dict[int, dict]:
    """``visual_genome_m``-shaped medias: one positive with a box, the rest negatives."""
    medias = {i: {"category": "dog", "categories": [], "regions": []} for i in range(n)}
    medias[0] = {"category": "dog", "categories": ["dog"], "regions": [{"box": [0.1, 0.1, 0.5, 0.5], "label": "dog"}]}
    return medias


def test_a_labelled_boxed_cell_passes(audit):
    assert audit.label_problems("visual_genome_m", _vg()) == []


def test_the_4117_shape_is_named(audit):
    """Every media present, every label gone: what the demo-cache rebuild wrote."""
    bare = {i: {"category": m["category"]} for i, m in _vg().items()}

    problems = audit.label_problems("visual_genome_m", bare)

    assert problems == ["3/3 medias carry no `categories`", "3/3 medias carry no `regions`"]


def test_one_bare_media_is_enough(audit):
    medias = _vg()
    del medias[2]["regions"]

    assert audit.label_problems("visual_genome_m", medias) == ["1/3 medias carry no `regions`"]


def test_empty_lists_are_labels_but_a_cell_needs_one_box(audit):
    """Negatives carry ``[]``; a cell where every media does has no positives to drag."""
    medias = {i: {"categories": [], "regions": []} for i in range(3)}

    assert audit.label_problems("visual_genome_m", medias) == ["none of 3 medias carries a region box"]


def test_an_unboxed_dataset_is_not_asked_for_boxes(audit):
    assert audit.label_problems("caltech101_m", {0: {"category": "ant"}}) == []


def _func(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(f"{path.name} no longer defines {name}()")


def _call_lines(fn: ast.FunctionDef, name: str) -> list[int]:
    """Line numbers of calls to *name*, as a bare name or as an attribute."""
    out = []
    for n in ast.walk(fn):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if (isinstance(f, ast.Name) and f.id == name) or (isinstance(f, ast.Attribute) and f.attr == name):
            out.append(n.lineno)
    return out


def test_build_cell_checks_labels_before_it_embeds_or_writes():
    """Refused before ``dump_medias`` truncates the previous cell, and before the embed pass."""
    build = _func(_BUILD, "build_cell")
    checks = _call_lines(build, "label_problems")
    assert checks, "build_cell() no longer checks the loaded medias' labels"
    for later in ("embed_missing", "dump_medias"):
        lines = _call_lines(build, later)
        assert lines, f"build_cell() no longer calls {later}()"
        assert min(checks) < min(lines), f"the label check must run before {later}()"


def test_verify_checks_labels():
    assert _call_lines(_func(_AUDIT, "verify"), "label_problems"), "--verify no longer checks cell labels"


def test_the_demo_loader_bypasses_the_app_cache():
    calls = [
        n
        for n in ast.walk(_func(_DEMO, "load"))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "load_demo_dataset"
    ]
    assert calls, "the pile's demo loader no longer calls load_demo_dataset"
    for call in calls:
        use_cache = {kw.arg: kw.value for kw in call.keywords}.get("use_cache")
        assert isinstance(use_cache, ast.Constant) and use_cache.value is False, (
            "the pile must build demo cells from the source (use_cache=False): a cache hit "
            "hands back another build's vectors under this build's provenance"
        )


def test_backfill_says_a_pre_3683_batch_size_is_unknown(tmp_path: Path, monkeypatch):
    if str(_PILE) not in sys.path:
        sys.path.insert(0, str(_PILE))
    import pile_config as pc
    import pilebuild.provenance_report as report

    cell = tmp_path / "visual_genome_m__siglip.pkl"
    cell.write_bytes(b"cell")
    side = tmp_path / "visual_genome_m__siglip.provenance.json"
    side.write_text(json.dumps({"device": {"hostname": "rack4n01", "gpu_name": "V100"}}))
    monkeypatch.setattr(pc, "cells", lambda: [("visual_genome_m", "siglip")])
    monkeypatch.setattr(pc, "cell_path", lambda ds, emb: cell)
    monkeypatch.setattr(pc, "provenance_path", lambda ds, emb: side)
    monkeypatch.setattr(report, "_sacct_build_nodes", lambda: {})

    report.provenance_report(backfill=True)

    dev = json.loads(side.read_text())["device"]
    assert dev["embed_batch_size"] is None
    assert "unknown" in dev["embed_batch_size_note"]
    assert dev["hostname"] == "rack4n01", "the backfill must not disturb what the sidecar already recorded"


def test_backfill_leaves_a_recorded_batch_size_alone(tmp_path: Path, monkeypatch):
    if str(_PILE) not in sys.path:
        sys.path.insert(0, str(_PILE))
    import pile_config as pc
    import pilebuild.provenance_report as report

    cell = tmp_path / "c.pkl"
    cell.write_bytes(b"cell")
    side = tmp_path / "c.provenance.json"
    side.write_text(json.dumps({"device": {"hostname": "rack4n01", "embed_batch_size": 64}}))
    monkeypatch.setattr(pc, "cells", lambda: [("visual_genome_m", "siglip")])
    monkeypatch.setattr(pc, "cell_path", lambda ds, emb: cell)
    monkeypatch.setattr(pc, "provenance_path", lambda ds, emb: side)
    monkeypatch.setattr(report, "_sacct_build_nodes", lambda: {})

    report.provenance_report(backfill=True)

    dev = json.loads(side.read_text())["device"]
    assert dev["embed_batch_size"] == 64
    assert "embed_batch_size_note" not in dev
