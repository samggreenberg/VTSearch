"""The eval/app drift gate itself: `scripts/check-eval-app-sync.py`.

The gate's whole job is to notice when app code that `vtscore.eval` reproduces
has moved.  If it stops noticing - or starts crying wolf over a reworded
comment - it gets ignored, and then the eval default arm drifts from the app
exactly the way it did before the gate existed.  So pin both halves: it fires on
a logic change, and it stays quiet for everything else.
"""

from __future__ import annotations

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
