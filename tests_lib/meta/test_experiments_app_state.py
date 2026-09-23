"""No experiment script may reach into the app's data directory by hand (#4066).

`scripts/experiments/` talks to a running VTSearch, and there are two ways to do
it. One is the HTTP API, which every script here already uses for reads; the
other is opening `data/` and editing the files under it, which
`regroup_negative_pass.py` did:

    for f in (D / "detectors").glob("*.json"):
        f.unlink()
    ...
    json.dump([keep], (D / "dataset_registry.json").open("w"), indent=1)

Both halves of that are losses waiting to happen, and neither is hypothetical:

* **A detector file is frequently the only copy of a human review.** Banking
  after a clear cost this project 1,725 answered questions once already, and a
  glob-and-unlink over `detectors/` does exactly that with no confirmation and
  no undo. `load_slates.py` states the rule at the top of its own docstring --
  "Existing detectors are reused, never recreated" -- because it learned it.
* **A registry write behind the app's back desynchronises it from the app.**
  Registration is a route (`/api/detectors/registry`), not a JSON file: writing
  the manifest directly leaves a live process holding the old one, and skipping
  the manifest leaves a detector the UI never lists. `load_slates.py` documents
  that trap too.

The script this gate was written for survived the Visual Genome retirement
(#4038) only because its hardcoded data dir had stopped being the deployment
(#3877 moved it), so it would have operated on an absent directory rather than a
live one. That is luck, not safety, and luck is what a gate is for.

**What this checks, and why it is slightly wider than "writes".** It bans naming
these paths in code at all, reads included, rather than trying to decide which
occurrences are writes. Two reasons. Deciding needs dataflow -- the unlink above
is spelled `f.unlink()`, with `f` bound by a `for`, so no sink-shaped check sees
the word `detectors` anywhere near it. And a read is the step before a write: it
is how you find the file you are about to edit, which is precisely what the loop
above is doing on line 33. The API serves both directions, so nothing here loses
a capability by going through it.

Prose is exempt: the scan is over string constants that are *code*, so
docstrings and comments may name these paths freely -- a retired script's
docstring explaining what it used to do is worth keeping.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "scripts" / "experiments"

#: The app's two JSON manifests. `vtscore.datasets.registry` and
#: `vtscore.detectors.registry` own these; a script's route to them is the API.
REGISTRY_FILES = ("dataset_registry.json", "detector_registry.json")

#: The directory holding one JSON per detector, i.e. per human review.
DETECTOR_DIR = "detectors"

#: How a script is *supposed* to say "detectors": as a route, not a directory.
#: Every occurrence in this tree today is one of these, which is what makes the
#: ban below cost nothing.
API_PREFIX = "/api/"


def _scripts() -> list[Path]:
    return sorted(p for p in EXPERIMENTS.rglob("*.py") if p.is_file())


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """`id()` of every string constant that is a docstring, not code."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def _code_strings(path: Path) -> list[tuple[int, str]]:
    """Every `(lineno, value)` string constant in *path* that is not a docstring.

    f-strings are included via their literal segments, which is how a route like
    ``f"/api/detectors/{name}/labels-detail"`` is seen -- and correctly cleared.
    """
    tree = ast.parse(path.read_text())
    skip = _docstring_nodes(tree)
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            out.append((node.lineno, node.value))
    return out


def _path_join_segments(path: Path) -> list[tuple[int, str]]:
    """String constants used as a path segment, i.e. either side of ``a / "b"``.

    This is the spelling the retired script used (``D / "detectors"``), and the
    one a bare substring scan misses when the directory is built rather than
    written out whole.
    """
    tree = ast.parse(path.read_text())
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        for side in (node.left, node.right):
            if isinstance(side, ast.Constant) and isinstance(side.value, str):
                out.append((side.lineno, side.value))
    return out


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: str(p.relative_to(EXPERIMENTS)))
def test_no_experiment_script_names_a_registry_manifest(script: Path) -> None:
    """The registries are written by the app, through `/api/*/registry`."""
    hits = [(n, s) for n, s in _code_strings(script) if any(f in s for f in REGISTRY_FILES)]
    assert not hits, (
        f"{script.relative_to(REPO_ROOT)} names a registry manifest in code: {hits}. "
        "Registration goes through the API (`/api/datasets`, `/api/detectors/registry`) "
        "so the running app stays in step with the file; see `load_slates.py` for the "
        "shape. Naming it in a docstring or comment is fine -- this only sees code."
    )


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: str(p.relative_to(EXPERIMENTS)))
def test_no_experiment_script_builds_a_path_into_the_detectors_directory(script: Path) -> None:
    """A detector JSON is usually the only copy of a human review."""
    hits = [(n, s) for n, s in _path_join_segments(script) if s.strip("/") == DETECTOR_DIR]
    assert not hits, (
        f"{script.relative_to(REPO_ROOT)} builds a filesystem path into the app's "
        f"detectors directory: {hits}. A detector file is often the only copy of a "
        "human review, so read and write them over the API "
        "(`/api/detectors/registry`, `/api/detectors/<name>/labels-detail`) rather "
        "than globbing `data/detectors/`."
    )


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: str(p.relative_to(EXPERIMENTS)))
def test_no_experiment_script_names_the_detectors_directory_as_a_literal_path(script: Path) -> None:
    """The same ban for a path written out whole, minus the API routes."""
    hits = [
        (n, s)
        for n, s in _code_strings(script)
        if f"{DETECTOR_DIR}/" in s and API_PREFIX not in s and not s.startswith(f"{API_PREFIX.strip('/')}/")
    ]
    assert not hits, (
        f"{script.relative_to(REPO_ROOT)} names the detectors directory as a path: "
        f"{hits}. Every mention of `detectors/` in this tree is an `/api/` route, "
        "which is the supported way to reach them; a filesystem path is not."
    )


def test_the_api_routes_this_gate_exempts_are_really_in_use() -> None:
    """A gate nobody can trip is as useless as one that fires on everything.

    The two tests above clear `/api/detectors/...` unconditionally. That is only
    safe while the API is what scripts actually use -- if this ever finds none,
    the exemption has stopped describing the tree and the gate is passing for
    the wrong reason.
    """
    users = [
        s.relative_to(EXPERIMENTS)
        for s in _scripts()
        if any(f"{API_PREFIX}{DETECTOR_DIR}" in v for _, v in _code_strings(s))
    ]
    assert len(users) >= 5, f"expected the detector API to be in wide use; found {users}"
