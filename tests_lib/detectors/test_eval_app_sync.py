"""The eval/app drift gate itself: `scripts/check-eval-app-sync.py`.

The gate's whole job is to notice when app code that `vtscore.eval` reproduces
has moved.  If it stops noticing - or starts crying wolf over a reworded
comment - it gets ignored, and then the eval default arm drifts from the app
exactly the way it did before the gate existed.  So pin both halves: it fires on
a logic change, and it stays quiet for everything else.

"Both halves" is now literal in a second sense too - the gate digests the
harness side as well as the app side (#3406), because a hand copy drifts just as
easily by being edited as by being left behind.
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

    def test_every_ported_mirror_pins_its_harness_side(self):
        """A hand copy is exactly the thing that can drift by being edited.

        `no_harness_pin` exists for a harness symbol too coarse to digest - one
        function serving several mirrors and much else besides.  That is never
        true of a `ported` mirror: the harness side of a port is the port, so
        an opt-out there would re-open the hole #3406 closed.  The escape hatch
        for a coarse anchor is to extract the reproduction (as #3403 did), not
        to stop watching it.
        """
        for mirror in gate.MIRRORS:
            if mirror.kind == "ported":
                assert mirror.no_harness_pin is None, f"{mirror.id} is a hand copy but is not harness-pinned"

    def test_an_opt_out_says_why(self):
        """The field is a written-down decision, not a flag to flip."""
        for mirror in gate.MIRRORS:
            if mirror.no_harness_pin is not None:
                assert len(mirror.no_harness_pin.strip()) > 40, mirror.id

    def test_most_mirrors_are_pinned_on_both_sides(self):
        """An opt-out is the exception; a gate mostly opted out is not a gate."""
        opted_out = [m.id for m in gate.MIRRORS if m.no_harness_pin is not None]
        assert len(opted_out) < len(gate.MIRRORS) / 2, opted_out

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
            gate._harness_source(self._mirror(harness="vtscore/eval/voting_iterations.py::no_such_symbol"))

    def test_deleted_harness_file_is_an_error(self):
        with pytest.raises(gate.MirrorError, match="does not exist"):
            gate._harness_source(self._mirror(harness="vtscore/eval/no_such_file.py::thing"))

    def test_a_harness_side_naming_no_symbol_is_an_error(self):
        with pytest.raises(gate.MirrorError, match="names no symbol"):
            gate._harness_source(self._mirror(harness="vtscore/eval/voting_iterations.py::"))

    def test_an_unresolvable_harness_side_is_drift_even_when_it_is_not_pinned(self):
        """Resolution is not conditional on pinning - a deleted copy is the loudest case."""
        with pytest.raises(gate.MirrorError, match="not found"):
            gate._digests(
                self._mirror(
                    harness="vtscore/eval/voting_iterations.py::no_such_symbol",
                    no_harness_pin="declared: the anchor is too coarse to digest, at length.",
                )
            )


class TestHarnessAnchorsAreSpecific:
    """A mirror's harness side must name the code that does the mirroring.

    A coarse anchor is two kinds of useless at once: the gate's failure message
    points the reconciler at the whole enclosing function instead of at the
    reproduction, and - now that the harness side is digested (#3406) - its pin
    trips on every unrelated edit to that function, which trains people to run
    ``--update`` without reading.  That is why the coarse anchors are the ones
    that opt out of a harness pin, and why the fix is to extract rather than to
    widen.

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


class TestTheHarnessSideIsWatchedToo:
    """#3406: an edit to the *copy* is drift, and used to be invisible.

    Before this, only the app side carried a digest and the harness side was a
    substring existence check.  So the gate could see the original move but not
    the copy, which is precisely the direction the Smart-indicator plumbing
    actually drifted in (#2923): the app stood still, the harness's own
    implementation stopped meaning what its mirror claimed, and the gate stayed
    green throughout.

    ``check()`` reads the real manifest and the committed pins, so these drive
    it through a one-mirror manifest and a fabricated pins file instead - that
    way a reason string can be asserted without editing tracked source.
    """

    #: A real, resolvable mirror, so both sides digest for real.
    def _probe(self, **kwargs):
        base = dict(
            id="probe",
            app="py:vtscore.detectors.training.train_and_threshold",
            harness="vtscore/eval/autopilot_flow.py::next_phase",
            kind="ported",
            note="probe",
        )
        base.update(kwargs)
        return gate.Mirror(**base)

    def _check_with(self, monkeypatch, mirror, pins):
        monkeypatch.setattr(gate, "MIRRORS", [mirror])
        monkeypatch.setattr(gate, "_load_pins", lambda: pins)
        return gate.check()

    def test_a_moved_harness_side_reports_harness_changed(self, monkeypatch):
        mirror = self._probe()
        pins = {"probe": {**gate._digests(mirror), "harness": "0" * 64}}
        (drift,) = self._check_with(monkeypatch, mirror, pins)
        assert drift.reason == "harness-changed"
        assert "harness: pinned" in drift.detail

    def test_a_moved_app_side_reports_app_changed(self, monkeypatch):
        mirror = self._probe()
        pins = {"probe": {**gate._digests(mirror), "app": "0" * 64}}
        (drift,) = self._check_with(monkeypatch, mirror, pins)
        assert drift.reason == "app-changed"

    def test_both_sides_moving_is_one_drift_naming_both(self, monkeypatch):
        """The ordinary shape of a faithful port - reported once, not twice."""
        mirror = self._probe()
        pins = {"probe": {"app": "0" * 64, "harness": "1" * 64}}
        (drift,) = self._check_with(monkeypatch, mirror, pins)
        assert drift.reason == "app-changed, harness-changed"

    def test_an_unpinned_harness_side_is_reported_on_its_own(self, monkeypatch):
        """A mirror added without `--update` must not pass on its app half alone."""
        mirror = self._probe()
        pins = {"probe": {"app": gate._digests(mirror)["app"]}}
        (drift,) = self._check_with(monkeypatch, mirror, pins)
        assert drift.reason == "harness-unpinned"

    def test_an_opted_out_mirror_records_no_harness_digest(self, monkeypatch):
        mirror = self._probe(kind="default", no_harness_pin="declared: the anchor is far too coarse to digest.")
        assert set(gate._digests(mirror)) == {"app"}
        assert not self._check_with(monkeypatch, mirror, {"probe": gate._digests(mirror)})

    def test_a_leftover_digest_from_a_dropped_pin_is_reported(self, monkeypatch):
        """Opting a mirror out must not leave its old harness pin asserting things."""
        mirror = self._probe(kind="default", no_harness_pin="declared: the anchor is far too coarse to digest.")
        pins = {"probe": {**gate._digests(mirror), "harness": "0" * 64}}
        (drift,) = self._check_with(monkeypatch, mirror, pins)
        assert drift.reason == "side-not-pinned-anymore"
        assert "harness" in drift.detail

    def test_a_multi_symbol_harness_ref_covers_every_symbol(self):
        """`GOOD_TARGET,BAD_TARGET`: watching one of a pair is half a mirror."""
        mirror = next(m for m in gate.MIRRORS if m.id == "autopilot.vote_targets")
        source = gate._harness_source(mirror)
        assert "GOOD_TARGET" in source and "BAD_TARGET" in source

    def test_a_name_surviving_only_in_a_comment_is_not_a_symbol(self, tmp_path):
        """The substring check the August 2026 audit flagged: `symbol in text`.

        A reproduction deleted with a note saying so left its own name behind in
        that note, and the old existence check was happy.
        """
        module = tmp_path / "harness.py"
        module.write_text("# next_phase was inlined into the caller\ndef other():\n    return 1\n")
        with pytest.raises(gate.MirrorError, match="no top-level"):
            gate._py_symbol_source(module, "next_phase")


class TestThePinsFile:
    """Its shape is part of the contract - `--update` and `check()` must agree."""

    def test_it_records_exactly_the_sides_each_mirror_pins(self):
        pins = gate._load_pins()
        assert set(pins) == {m.id for m in gate.MIRRORS}
        for mirror in gate.MIRRORS:
            expected = {"app"} if mirror.no_harness_pin else {"app", "harness"}
            assert set(pins[mirror.id]) == expected, mirror.id

    def test_the_one_sided_format_says_what_to_run(self, monkeypatch, tmp_path):
        """Only reachable mid-rebase, and a bare crash there is a puzzle."""
        stale = tmp_path / "pins.json"
        stale.write_text('{"autopilot.phase_machine": "deadbeef"}')
        monkeypatch.setattr(gate, "PINS_PATH", stale)
        with pytest.raises(SystemExit, match="--update"):
            gate._load_pins()
