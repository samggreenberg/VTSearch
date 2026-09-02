"""The call surface of ``simulate_voting_iterations``.

Forty-odd knobs is what a pre-registered experiment harness looks like, and
that is fine - but only while they are *named*.  As plain positional-or-keyword
parameters they were an ordering hazard with no error attached to it: inserting
one mid-list rebinds every caller that passed the neighbours positionally, and
the run does not crash, it silently measures a different arm.  Study code is the
worst possible place for that failure mode, because a wrong-but-plausible
trajectory is indistinguishable from a real result.

So the keyword-only boundary is pinned here rather than left to review.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "vtscore" / "eval" / "voting_iterations.py"

#: The only three arguments a caller may pass positionally: the pool, the
#: category it is detecting, and the seed that makes the cell reproducible.
POSITIONAL = ("clips_dict", "target_category", "seed")


def _signature() -> ast.arguments:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "simulate_voting_iterations":
            return node.args
    raise AssertionError("simulate_voting_iterations not found in vtscore/eval/voting_iterations.py")


class TestKeywordOnlyBoundary:
    def test_only_the_three_core_arguments_are_positional(self):
        args = _signature()
        assert tuple(a.arg for a in args.args) == POSITIONAL
        assert not args.posonlyargs

    def test_every_experiment_knob_is_keyword_only(self):
        """The knobs live after ``*``, so their order cannot be load-bearing."""
        args = _signature()
        assert len(args.kwonlyargs) > 30, (
            "the experiment knobs moved back in front of the `*` marker; a new "
            "parameter inserted mid-list would silently rebind positional callers"
        )

    def test_no_caller_relies_on_more_than_the_core_positionals(self):
        """The boundary is only safe while nothing passes past it positionally.

        Scanned over the tree rather than trusted: an experiment script that
        spread a fourth positional argument would fail at import time in a
        SLURM job rather than here.
        """
        offenders = []
        for path in REPO_ROOT.rglob("*.py"):
            if "node_modules" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name != "simulate_voting_iterations":
                    continue
                if len(node.args) > len(POSITIONAL) or any(isinstance(a, ast.Starred) for a in node.args):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        assert not offenders, "callers passing past the keyword-only boundary: " + ", ".join(offenders)
