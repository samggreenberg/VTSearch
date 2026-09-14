"""A verdict records the rule it was answered under, and can be asked about (#3814).

A correction row used to say only *what* a reviewer answered. The class rule is
the question -- it is the detector name they read while voting (#3612) -- so a
ruling that moves a class boundary silently changed what every row cast under the
old wording meant. It cost the 80-image planter recheck (#3778): nothing on the
31 `vase` rows or the 49 `bowl` rows said which wording they answered, so all 80
had to go back in front of a human to find the 21 that had actually moved.

Two fields fix it and one law protects them. `rule` is the name the reviewer saw;
`rule_digest` covers the wording behind that name, which can be rewritten while
the name stands still (#3756 did that to `bench`). The law is that **absence
means unknown**: the 872 rows that predate the fields can never be back-filled,
and a reader that treats them as current has re-created the bug.

These pin both halves -- the stamp each writer produces (and, more importantly,
the stamp it refuses to invent) and the four states a reader can distinguish.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"

_CLASS = "vase"
_RULE = "vase not planters"


@pytest.fixture(scope="module")
def pc():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import pile_config

    return pile_config


@pytest.fixture(scope="module")
def corr(pc):
    from pilebuild import corrections

    return corrections


@pytest.fixture(scope="module")
def ingest(pc):
    """``ingest_slate``, imported without letting it build a pile directory.

    The module calls ``pc.setup_env()`` at import, which mkdirs under
    ``VTS_PILE`` -- ``/expscratch/$USER/vts-cache`` by default, which exists on
    the cluster and nowhere else. The sibling script tests pay for that with a
    subprocess and a ``VTS_PILE`` override; there is nothing to isolate here,
    because :func:`ingest_slate.rule_stamp_for` reads the rule table and touches
    no path at all.
    """
    real, pc.setup_env = pc.setup_env, lambda *a, **k: None
    try:
        import ingest_slate
    finally:
        pc.setup_env = real
    return ingest_slate


def _stamp(rule: str = _RULE, digest: str | None = None) -> dict[str, str]:
    return {"rule": rule, **({"rule_digest": digest} if digest else {})}


def _row(cls: str = _CLASS, **extra) -> dict:
    return {"image_id": 7, "class": cls, "present": True, "boxes": [], "source": "human_review", **extra}


class TestTheStamp:
    """What a writer records about the rule, and what it declines to record."""

    def test_the_name_is_what_the_reviewer_would_have_seen(self, pc):
        assert pc.rule_stamp(_CLASS)["rule"] == pc.review_name(_CLASS) == _RULE

    def test_an_unruled_class_stamps_its_bare_name(self, pc):
        """`review_name`'s fallback, so every class is stampable, not just the ruled ones."""
        assert "cat" not in pc.SCALE_CLASS_RULES
        assert pc.rule_stamp("cat")["rule"] == "cat"

    def test_the_digest_moves_when_the_TEST_is_rewritten_under_an_unchanged_name(self, pc, monkeypatch):
        """The whole reason a name is not enough: #3756 edited `bench`'s Bad list in place."""
        before = pc.rule_digest("bench")
        rule = pc.SCALE_CLASS_RULES["bench"]
        monkeypatch.setitem(pc.SCALE_CLASS_RULES, "bench", rule._replace(test=rule.test + " Bad: a church pew."))

        assert pc.rule_stamp("bench")["rule"] == rule.name, "the name has not moved"
        assert pc.rule_digest("bench") != before, "but the wording behind it has"

    def test_the_digest_moves_when_the_name_is_rewritten(self, pc, monkeypatch):
        before = pc.rule_digest(_CLASS)
        rule = pc.SCALE_CLASS_RULES[_CLASS]
        monkeypatch.setitem(pc.SCALE_CLASS_RULES, _CLASS, rule._replace(name="vase incl planters"))

        assert pc.rule_digest(_CLASS) != before

    def test_the_digest_is_stable_across_calls(self, pc):
        """A stamp written twice must compare equal, or every row reads as edited."""
        assert pc.rule_digest(_CLASS) == pc.rule_digest(_CLASS)

    def test_two_classes_do_not_share_a_digest(self, pc):
        digests = {cls: pc.rule_digest(cls) for cls in pc.SCALE_CLASSES}
        assert len(set(digests.values())) == len(digests)


class TestReadingADetectorName:
    """The slate's detector is the only surviving record of a PAST review's rule."""

    def test_it_inverts_review_name_for_every_class_and_every_pass(self, pc):
        classes = set(pc.SCALE_CLASSES) | set(pc.SCALE_CLASS_RULES) | {"cat"}
        for cls in sorted(classes):
            for suffix in ("", *pc.REVIEW_SUFFIXES):
                detector = pc.review_name(cls, suffix)
                assert pc.rule_of_review_name(detector) == pc.review_name(cls), f"{cls}/{suffix!r}"

    def test_no_rule_name_ends_IN_a_pass_suffix(self, pc):
        """What makes the inverse well defined rather than a lucky parse."""
        for cls, rule in pc.SCALE_CLASS_RULES.items():
            for suffix in pc.REVIEW_SUFFIXES:
                assert not rule.name.endswith(f" {suffix}"), f"{cls}: {rule.name!r} ends in a pass suffix"

    def test_it_drops_the_recheck_tail(self, pc):
        """`make_class_recheck.py` appends the question; `bank_verdicts.py` strips it the same way."""
        assert pc.rule_of_review_name(f"{_RULE} -- recheck: is there one in this image?") == _RULE

    def test_a_rule_that_has_been_ruled_away_still_reads_back(self, pc):
        """The case that matters: a detector naming a rule the table no longer holds."""
        assert pc.rule_of_review_name("vase incl pots and planters positives") == "vase incl pots and planters"


class TestWhatIngestStamps:
    """`ingest_slate.py` takes the rule from the SLATE, never from today's table."""

    def test_a_current_detector_gets_the_name_and_the_digest(self, pc, ingest):
        assert ingest.rule_stamp_for(_RULE, _CLASS) == pc.rule_stamp(_CLASS)

    def test_a_suffixed_detector_is_still_recognised(self, pc, ingest):
        assert ingest.rule_stamp_for(f"{_RULE} positives", _CLASS) == pc.rule_stamp(_CLASS)

    def test_a_superseded_detector_records_what_it_ASKED(self, ingest):
        stamp = ingest.rule_stamp_for("vase incl pots and planters", _CLASS)

        assert stamp["rule"] == "vase incl pots and planters"

    def test_a_superseded_detector_records_NO_digest(self, ingest):
        """A digest covers a `test` body, and no detector name can reconstruct a withdrawn one.

        Taking it from the table anyway would stamp today's wording onto a vote
        cast under yesterday's -- the exact substitution the field exists to
        prevent, made harder to see by looking precise.
        """
        assert "rule_digest" not in ingest.rule_stamp_for("vase incl pots and planters", _CLASS)


class TestRuleState:
    """The four states a reader can tell apart."""

    def test_a_matching_name_and_digest_is_current(self, corr, pc):
        row = _row(**pc.rule_stamp(_CLASS))

        assert corr.rule_state(row, pc.rule_stamp(_CLASS)) == corr.RULE_CURRENT

    def test_a_moved_name_is_superseded(self, corr, pc):
        row = _row(**_stamp("vase incl pots and planters"))

        assert corr.rule_state(row, pc.rule_stamp(_CLASS)) == corr.RULE_SUPERSEDED

    def test_a_matching_name_with_a_moved_digest_is_edited(self, corr, pc):
        row = _row(**_stamp(_RULE, "deadbeef0000"))

        assert corr.rule_state(row, pc.rule_stamp(_CLASS)) == corr.RULE_EDITED

    def test_a_matching_name_with_no_digest_is_current(self, corr, pc):
        """The name is what the reviewer saw, so the row's claim is met.

        The digest is the stricter, optional check; its absence means the
        `edited` question cannot be put to this row, not that the answer is no.
        """
        row = _row(**_stamp(_RULE))

        assert corr.rule_state(row, pc.rule_stamp(_CLASS)) == corr.RULE_CURRENT

    def test_an_unstamped_row_is_UNKNOWN_and_never_current(self, corr, pc):
        """The law. Reading these as current is #3814 itself, not a rounding of it."""
        assert corr.rule_state(_row(), pc.rule_stamp(_CLASS)) == corr.RULE_UNKNOWN

    def test_an_empty_rule_string_is_unknown_too(self, corr, pc):
        assert corr.rule_state(_row(rule=""), pc.rule_stamp(_CLASS)) == corr.RULE_UNKNOWN

    def test_the_872_committed_rows_are_all_unknown(self, corr):
        """The live record, read as it stands: nothing in it can be back-filled.

        A regression here would mean some reader had started treating the legacy
        rows as answered, which is the one outcome this issue forbids.
        """
        import json

        rows = json.loads((_PILE_DIR / "human_record" / "PILE__corrections.json").read_text())
        states = {state for _, state in corr.rule_states(rows)}

        assert states == {corr.RULE_UNKNOWN}, f"expected every legacy row to be unknown, got {states}"


class TestTheQuery:
    """What a build and a tool each get to ask."""

    def test_superseded_rows_are_the_actionable_set(self, corr, pc):
        rows = [
            _row(**pc.rule_stamp(_CLASS)),
            _row(**_stamp("vase incl pots and planters")) | {"image_id": 8},
            _row() | {"image_id": 9},
        ]

        assert [r["image_id"] for r in corr.superseded_rows(rows)] == [8]

    def test_unknown_rows_are_NOT_reported_as_superseded(self, corr):
        """What makes the build's line self-suppressing instead of a constant.

        872 unstamped rows is a number nobody can act on, and a build that prints
        one every run teaches everyone to skip the line that matters.
        """
        assert corr.superseded_rows([_row(), _row() | {"image_id": 8}]) == []

    def test_states_are_resolved_per_class(self, corr, pc):
        rows = [_row(**pc.rule_stamp(_CLASS)), _row("bowl", **pc.rule_stamp("bowl"))]

        assert [s for _, s in corr.rule_states(rows)] == [corr.RULE_CURRENT, corr.RULE_CURRENT]

    def test_a_stamp_from_ANOTHER_class_reads_as_superseded(self, corr, pc):
        """A row is checked against its own class's rule, not against any rule."""
        rows = [_row("bowl", **pc.rule_stamp(_CLASS))]

        assert [s for _, s in corr.rule_states(rows)] == [corr.RULE_SUPERSEDED]


class TestWhatTheBuildSays:
    """Every build reads `corrections.json`, so what it says there is a standing cost.

    The ruling on #3814's second open question: the build names the one state
    somebody can act on, and stays silent about the one nobody can.
    """

    def _load(self, corr, pc, tmp_path, monkeypatch, rows):
        path = tmp_path / "corrections.json"
        path.write_text(json.dumps(rows))
        monkeypatch.setenv("VTS_CORRECTIONS", str(path))
        return corr.load_corrections()

    def test_it_names_a_row_whose_rule_was_ruled_away(self, corr, pc, tmp_path, monkeypatch, capsys):
        self._load(corr, pc, tmp_path, monkeypatch, [_row(**_stamp("vase incl pots and planters"))])

        out = capsys.readouterr().out
        assert "ruled away" in out and _CLASS in out and "rule_drift.py" in out

    def test_it_says_NOTHING_about_the_unstamped_rows(self, corr, pc, tmp_path, monkeypatch, capsys):
        """872 of them, un-actionable and un-repairable: a line every run is a line ignored."""
        self._load(corr, pc, tmp_path, monkeypatch, [_row(), _row() | {"image_id": 8}])

        assert capsys.readouterr().out == ""

    def test_it_says_nothing_when_every_row_is_current(self, corr, pc, tmp_path, monkeypatch, capsys):
        self._load(corr, pc, tmp_path, monkeypatch, [_row(**pc.rule_stamp(_CLASS))])

        assert capsys.readouterr().out == ""

    def test_a_superseded_row_is_still_LOADED(self, corr, pc, tmp_path, monkeypatch):
        """Named, not refused: the reviewer was right and the definition moved after them."""
        got = self._load(corr, pc, tmp_path, monkeypatch, [_row(**_stamp("vase incl pots and planters"))])

        assert (7, _CLASS) in got


_COLS = ["image_id", "class", "stratum", "cell", "text_score", "reference", "exhaustive", "n_boxes", "detector"]


def _run_corrections(tmp_path: Path, verdicts: list[dict]) -> dict[int, dict]:
    """Drive `verdicts_to_corrections.py` over *verdicts* and return its rows by image id.

    A subprocess, like its sibling tests: ``pile_config.setup_env()`` edits
    ``os.environ`` and ``sys.meta_path`` at import, and ``VTS_PILE`` has to be
    somewhere that exists.
    """
    slates = tmp_path / "slates"
    (slates / _CLASS).mkdir(parents=True)
    with (slates / _CLASS / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLS)
        w.writeheader()
        for v in verdicts:
            w.writerow(
                {
                    "image_id": v["image_id"],
                    "class": _CLASS,
                    "stratum": v["stratum"],
                    "cell": f"{_CLASS}@medium",
                    "text_score": "0.0",
                    "reference": "absent",
                    "exhaustive": "no",
                    "n_boxes": "0",
                    "detector": _RULE,
                }
            )
    vpath = tmp_path / "verdicts.json"
    vpath.write_text(json.dumps(verdicts))
    out = tmp_path / "corrections.json"
    proc = subprocess.run(  # noqa: S603  # interpreter + test-controlled args
        [
            sys.executable,
            "verdicts_to_corrections.py",
            "--verdicts",
            str(vpath),
            "--adjudication",
            str(tmp_path / "no_adjudication.json"),
            "--slates",
            str(slates),
            "--triage",
            str(tmp_path / "no_triage.json"),
            "--sheets",
            str(tmp_path / "no_sheets"),
            "--merge",
            "",
            "--out",
            str(out),
        ],
        cwd=_PILE_DIR,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "VTS_PILE": str(tmp_path / "pile")},
    )
    assert proc.returncode == 0, proc.stderr
    return {int(r["image_id"]): r for r in json.loads(out.read_text())}


def _verdict(iid: int, **extra) -> dict:
    return {
        "image_id": iid,
        "class": _CLASS,
        "stratum": "boundary",
        "human": "present",
        "reference": "absent",
        "exhaustive": "no",
        "box": None,
        "text_score": 0.0,
        "export": "test",
        **extra,
    }


class TestCarryThrough:
    """`verdicts_to_corrections.py` carries the verdict's stamp and never re-derives one.

    This is the half that would be easiest to get subtly wrong. The script reads
    August verdict files and runs today; taking the rule from
    ``SCALE_CLASS_RULES`` at that moment would stamp every one of them with a
    wording their reviewers never saw -- which looks like a fix and is the bug.
    """

    def test_a_stamped_verdict_produces_a_stamped_row(self, tmp_path):
        got = _run_corrections(tmp_path, [_verdict(1, rule=_RULE, rule_digest="0123456789ab")])

        assert got[1]["rule"] == _RULE
        assert got[1]["rule_digest"] == "0123456789ab"

    def test_a_verdict_cast_under_a_WITHDRAWN_rule_keeps_that_rule(self, tmp_path):
        got = _run_corrections(tmp_path, [_verdict(1, rule="vase incl pots and planters")])

        assert got[1]["rule"] == "vase incl pots and planters"

    def test_an_unstamped_verdict_produces_an_unstamped_row(self, tmp_path):
        """The anti-regression. Every verdict file banked before #3814 is this case."""
        got = _run_corrections(tmp_path, [_verdict(1)])

        assert "rule" not in got[1] and "rule_digest" not in got[1]

    def test_a_rebox_carries_the_stamp_too(self, tmp_path):
        got = _run_corrections(
            tmp_path,
            [_verdict(1, stratum="positive_boxed", box=[0.1, 0.1, 0.6, 0.6], reference="present", rule=_RULE)],
        )

        assert got[1]["source"] == "human_rebox" and got[1]["rule"] == _RULE

    def test_a_confirmation_carries_the_stamp_too(self, tmp_path):
        """The modal outcome of the exhaustive pass, and the one that buys the most."""
        got = _run_corrections(tmp_path, [_verdict(1, stratum="positive_boxed", reference="present", rule=_RULE)])

        assert got[1]["source"] == "human_confirmed" and got[1]["rule"] == _RULE
