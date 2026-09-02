"""The eval/app drift gate itself: `scripts/check-eval-app-sync.py`.

The gate's whole job is to notice when app code that `vtscore.eval` reproduces
has moved.  If it stops noticing - or starts crying wolf over a reworded
comment - it gets ignored, and then the eval default arm drifts from the app
exactly the way it did before the gate existed.  So pin both halves: it fires on
a logic change, and it stays quiet for everything else.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check-eval-app-sync.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("_eval_app_sync_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves annotations through sys.modules, so register first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


class TestManifestIsLive:
    """Every mirror resolves on both sides, and its pin is current."""

    def test_no_drift(self):
        drifts = gate.check()
        assert not drifts, "\n".join(f"{d.mirror.id}: {d.reason} ({d.detail})" for d in drifts)

    def test_every_mirror_names_both_sides(self):
        for mirror in gate.MIRRORS:
            assert mirror.app.split(":", 1)[0] in ("py", "ts"), mirror.id
            assert "::" in mirror.harness, mirror.id
            assert mirror.kind in ("ported", "default"), mirror.id
            assert mirror.note.strip(), mirror.id

    def test_the_manifest_is_not_empty(self):
        """A gate with nothing pinned passes vacuously and fools everyone."""
        assert len(gate.MIRRORS) >= 5
        assert {m.kind for m in gate.MIRRORS} == {"ported", "default"}


class TestDigestSensitivity:
    """What trips the gate, and what must not."""

    BASE = '''
def f(a, b):
    """A docstring."""
    # A comment.
    if a < 5:
        return "red"
    return b
'''

    def test_logic_change_trips(self):
        changed = self.BASE.replace("a < 5", "a < 7")
        assert gate._normalize_python(changed) != gate._normalize_python(self.BASE)

    def test_comment_change_does_not_trip(self):
        changed = self.BASE.replace("# A comment.", "# A completely different comment.")
        assert gate._normalize_python(changed) == gate._normalize_python(self.BASE)

    def test_docstring_change_does_not_trip(self):
        changed = self.BASE.replace('"""A docstring."""', '"""Rewritten, at length.\n\n    More prose.\n    """')
        assert gate._normalize_python(changed) == gate._normalize_python(self.BASE)

    def test_reformatting_does_not_trip(self):
        """Including the magic trailing comma `ruff format` adds when it wraps."""
        changed = self.BASE.replace("def f(a, b):", "def f(\n    a,\n    b,\n):")
        assert gate._normalize_python(changed) == gate._normalize_python(self.BASE)

    def test_a_meaningful_trailing_comma_still_trips(self):
        """Only the comma immediately before a closer is dropped, not tuple-ness."""
        base = "x = (1, 2)\n"
        assert gate._normalize_python(base.replace("(1, 2)", "(1, 2, 3)")) != gate._normalize_python(base)

    def test_a_returned_string_is_not_mistaken_for_a_docstring(self):
        """Only a string in *statement* position is a docstring."""
        changed = self.BASE.replace('return "red"', 'return "green"')
        assert gate._normalize_python(changed) != gate._normalize_python(self.BASE)

    def test_typescript_comment_change_does_not_trip(self):
        base = "checkPhaseTransition() {\n  // why\n  if (a < 5) { return 'good'; }\n}"
        changed = base.replace("// why", "// a different why")
        assert gate._normalize_typescript(changed) == gate._normalize_typescript(base)

    def test_typescript_logic_change_trips(self):
        base = "checkPhaseTransition() {\n  if (a < 5) { return 'good'; }\n}"
        changed = base.replace("a < 5", "a < 7")
        assert gate._normalize_typescript(changed) != gate._normalize_typescript(base)


class TestFstringsDigestTheSameOnEveryInterpreter:
    """A pin must travel between the Python versions the repo supports (>=3.10).

    Python 3.12 (PEP 701) stopped emitting an f-string as one STRING token and
    started splitting it into FSTRING_START / FSTRING_MIDDLE / FSTRING_END plus
    the real tokens of each replacement field.  The normalizer is token-based
    precisely so it would be version-stable, so this went unnoticed: the three
    mirrored `labeling_progress._compute_*_status` functions all contain an
    f-string, so the gate was red on 3.12+ and green on 3.10/3.11 for the same
    tree, and `--update` only moved the failure to the other half of the range
    (issue #3117).  These assert the collapsed, <=3.11-shaped form, so they fail
    on an unfixed 3.12+ and pass everywhere once it is normalized away.
    """

    def test_a_simple_fstring_stays_one_token(self):
        assert 'f"a{b}c"' in gate._normalize_python('x = f"a{b}c"\n').splitlines()

    def test_a_nested_fstring_stays_one_token(self):
        """Legal on 3.12+, so the run has to be matched by depth, not first END."""
        source = "x = f\"{f'{y}'}\"\n"
        assert "f\"{f'{y}'}\"" in gate._normalize_python(source).splitlines()

    def test_a_multiline_fstring_stays_one_token(self):
        """Exercises the multi-row branch of the source slice."""
        source = 'x = f"""\nhello {name}\n"""\n'
        assert 'f"""\nhello {name}\n"""' in gate._normalize_python(source)

    def test_an_fstring_is_still_part_of_the_digest(self):
        """Collapsing must not amount to dropping it - rewording is a real change."""
        base = 'def f(good, bad):\n    return f"Currently {good}g, {bad}b."\n'
        changed = base.replace("{good}g, {bad}b.", "{good}g and {bad}b.")
        assert gate._normalize_python(changed) != gate._normalize_python(base)

    def test_an_fstring_expression_change_trips(self):
        base = 'def f(good, bad):\n    return f"{good}g, {bad}b"\n'
        changed = base.replace("{good}g", "{bad}g")
        assert gate._normalize_python(changed) != gate._normalize_python(base)

    def test_logic_beside_an_fstring_still_trips(self):
        """The real shape of the mirrored functions: a threshold, then a message."""
        base = 'def f(good, bad):\n    if good < 5 or bad < 5:\n        return f"need {good}"\n    return None\n'
        changed = base.replace("good < 5 or bad < 5", "good < 7 or bad < 7")
        assert gate._normalize_python(changed) != gate._normalize_python(base)

    def test_reformatting_around_an_fstring_does_not_trip(self):
        base = 'def f(good):\n    return dict(reason=f"need {good}", status="red")\n'
        changed = base.replace(
            'dict(reason=f"need {good}", status="red")',
            'dict(\n        reason=f"need {good}",\n        status="red",\n    )',
        )
        assert gate._normalize_python(changed) == gate._normalize_python(base)


class TestResolutionFailures:
    """A moved or renamed side is drift too, not a silent pass."""

    def _mirror(self, **kwargs):
        base = dict(
            id="probe",
            app="py:vtscore.detectors.training.train_and_threshold",
            harness="vtscore/eval/voting_iterations.py::simulate_voting_iterations",
            kind="default",
            note="probe",
        )
        base.update(kwargs)
        return gate.Mirror(**base)

    def test_missing_app_symbol_is_an_error(self):
        with pytest.raises(gate.MirrorError, match="no top-level"):
            gate._app_source(self._mirror(app="py:vtscore.detectors.training.no_such_function"))

    def test_missing_app_module_is_an_error(self):
        with pytest.raises(gate.MirrorError, match="does not exist"):
            gate._app_source(self._mirror(app="py:vtscore.detectors.no_such_module.thing"))

    def test_missing_typescript_anchor_is_an_error(self):
        with pytest.raises(gate.MirrorError, match="not found"):
            gate._app_source(self._mirror(app=f"ts:{gate.AUTOPILOT_TS}::noSuchMethod("))

    def test_renamed_harness_symbol_is_an_error(self):
        with pytest.raises(gate.MirrorError, match="not found"):
            gate._check_harness_side(self._mirror(harness="vtscore/eval/voting_iterations.py::no_such_symbol"))

    def test_deleted_harness_file_is_an_error(self):
        with pytest.raises(gate.MirrorError, match="does not exist"):
            gate._check_harness_side(self._mirror(harness="vtscore/eval/no_such_file.py::thing"))


class TestHarnessAnchorsAreSpecific:
    """A mirror's harness side must name the code that does the mirroring.

    ``_check_harness_side`` is a substring test on the file, so *any* name that
    survives in the text satisfies it.  That makes a coarse anchor two kinds of
    useless at once: the gate's failure message points the reconciler at the
    whole enclosing function instead of at the reproduction, and the existence
    check passes even when the reproduction itself has been deleted, because the
    function around it is still there.

    Both ``training.*_default`` mirrors used to name the thousand-line
    ``simulate_voting_iterations`` for exactly that reason (#3403).  They now
    name the small helper that resolves the defaults, so deleting a resolution
    trips the gate.  Pin the property rather than the helper's current name.
    """

    #: Mirror id -> the app-side call its harness anchor must contain.
    RESOLUTION_CALLS = {
        "training.blend_schedule_default": "production_schedule_for",
        "training.split_fraction_default": "production_split_for",
    }

    def _harness_function_source(self, mirror) -> str:
        rel, _, symbol = mirror.harness.partition("::")
        tree = ast.parse((gate.REPO_ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == symbol:
                return ast.get_source_segment((gate.REPO_ROOT / rel).read_text(encoding="utf-8"), node) or ""
        raise AssertionError(f"{symbol!r} is not a function in {rel}")

    @pytest.mark.parametrize("mirror_id", sorted(RESOLUTION_CALLS))
    def test_anchor_contains_the_resolution_it_claims_to_pin(self, mirror_id):
        mirror = next(m for m in gate.MIRRORS if m.id == mirror_id)
        source = self._harness_function_source(mirror)
        assert self.RESOLUTION_CALLS[mirror_id] in source, (
            f"{mirror_id} names {mirror.harness}, which no longer resolves the default it pins"
        )

    @pytest.mark.parametrize("mirror_id", sorted(RESOLUTION_CALLS))
    def test_anchor_is_small_enough_to_read(self, mirror_id):
        """The reconciler reads this function when the gate trips."""
        mirror = next(m for m in gate.MIRRORS if m.id == mirror_id)
        lines = self._harness_function_source(mirror).count("\n") + 1
        assert lines < 120, f"{mirror_id} points at {lines} lines; a tripped gate should not need a tour"
