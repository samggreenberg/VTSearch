"""The regeneration that would have caught #4003's scare, run every suite (#4007).

`corrections.json` is the one artefact on this pile that no rebuild recreates,
and for weeks nothing exercised the chain that reproduces it. When somebody
finally did, they ran `verdicts_to_corrections.py` — step 2 of 3, named after the
output — got 872 of 4,709 rows, and reasonably concluded the file was no longer
reproducible. It was; the composition lived in another file's usage block.

So the test is not "does the chain run". A partial chain runs fine and writes a
plausible file. It is a **row-by-row diff against the committed copy**, because
the last time this mattered 81 of the shared rows differed while the count looked
right, and reporting only-new, only-reference and differing separately is what
turns "it drifted" into a sentence someone can act on.

The whole chain takes ~3 s, so it runs here rather than behind a marker: a check
that has to be remembered is a check that stops being true quietly, which is the
failure this file exists to prevent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"
_RECORD = _PILE_DIR / "human_record"

#: What each step contributes, from #4007's attribution. A total that still looks
#: about right is exactly how a missing step hides, so the parts are named.
ATTRIBUTION = {"pass": 3837, "campaigns": 791, "recheck_revisited": 81, "total": 4709}


def _load(name: str):
    """Import one pile script by path, without importing the package."""
    spec = importlib.util.spec_from_file_location(name, _PILE_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(_PILE_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_PILE_DIR))
    return module


@pytest.fixture(scope="module")
def regen():
    return _load("regenerate_corrections")


class TestTheChainIsRunnableFromTheRepository:
    """Every input the chain names has to be in the record, or a checkout cannot run it."""

    def test_every_default_verdict_file_is_committed(self, regen):
        from verdicts_to_corrections import DEFAULT_VERDICTS  # noqa: PLC0415

        for path in DEFAULT_VERDICTS:
            assert path.is_file(), f"{path.name} is not in the record"
            assert _RECORD in path.parents, f"{path} resolves outside human_record/"

    def test_the_pass_reads_the_committed_slates_not_scratch(self):
        """#4007's near miss: slates.json was an input to 3,837 rows and lived only on scratch."""
        pass_verdicts = _load("pass_verdicts")
        parser_default = pass_verdicts.HR / "VLM3720__slates.json"
        assert parser_default.is_file()
        source = (_PILE_DIR / "pass_verdicts.py").read_text()
        assert "/expscratch/" not in source.split("def main")[-1], "the pass still defaults at a scratch path"

    def test_the_committed_copy_is_what_the_diff_is_judged_against(self, regen):
        rows = json.loads(regen.COMMITTED.read_text())
        assert len(rows) == ATTRIBUTION["total"]


@pytest.fixture(scope="module")
def result(regen, tmp_path_factory):
    """One run of the whole chain (~3 s), shared by the rows-level tests."""
    workdir = tmp_path_factory.mktemp("chain")
    rows = regen.regenerate(workdir / "corrections.json", workdir, quiet=True)
    return rows, json.loads(regen.COMMITTED.read_text())


class TestRegenerationMatchesTheCommittedCopy:
    """The diff, not the exit status."""

    def test_no_row_is_only_in_one_of_them_and_none_differ(self, regen, result):
        new, reference = result
        diff = regen.compare(new, reference)
        assert diff == {"only_new": [], "only_reference": [], "differing": []}

    def test_the_row_count_reconciles_to_the_attribution(self, result):
        """#4007's three writers sum to the file: a missing step shows up here by name."""
        new, _ = result
        parts = ATTRIBUTION["pass"] + ATTRIBUTION["campaigns"] + ATTRIBUTION["recheck_revisited"]
        assert parts == ATTRIBUTION["total"] == len(new)

    def test_step_two_alone_is_short_and_that_is_not_a_defect(self, regen, result):
        """The trap, pinned: a partial chain produces a plausible file 3,837 rows short."""
        new, _ = result
        by_source = {}
        for row in new:
            by_source[row.get("source")] = by_source.get(row.get("source"), 0) + 1
        assert by_source.get("human_review", 0) >= ATTRIBUTION["pass"], (
            "the per-class pass is missing: its rows are the ones a step-2-only run drops"
        )


class TestWriteLiveRefusesToOverwriteUnbankedWork:
    """The #4016 review's hazard: the guard checked the file it MATCHED, not the one it OVERWROTE.

    Matching the regeneration against the committed copy says the repository is
    self-consistent. The live file is a different file, and the two agree only
    until somebody banks. #4002's 93 judgements are queued to do exactly that.
    """

    def test_live_drift_sees_a_banked_row_the_committed_copy_lacks(self, regen, tmp_path):
        reference = [{"image_id": 1, "class": "bowl", "present": True}]
        live = tmp_path / "corrections.json"
        live.write_text(json.dumps([*reference, {"image_id": 2, "class": "cup", "present": True}]))
        assert regen.live_drift(live, reference)["only_new"] == [(2, "cup")]

    def test_live_drift_is_empty_when_they_agree(self, regen, tmp_path):
        reference = [{"image_id": 1, "class": "bowl", "present": True}]
        live = tmp_path / "corrections.json"
        live.write_text(json.dumps(reference))
        assert not any(regen.live_drift(live, reference).values())

    def test_a_missing_live_file_is_not_drift(self, regen, tmp_path):
        """Restoring live from the repository after a purge is the capability this keeps."""
        assert not any(regen.live_drift(tmp_path / "gone.json", []).values())

    def test_write_live_refuses_and_writes_nothing_when_live_is_ahead(self, regen, tmp_path, monkeypatch):
        """The 93 rows, in miniature: banked-but-not-exported work must survive a regeneration."""
        committed = json.loads(regen.COMMITTED.read_text())
        banked = [*committed, {"image_id": 999_000_001, "class": "bowl", "present": True, "source": "human_review"}]
        live = tmp_path / "corrections.json"
        live.write_text(json.dumps(banked))
        before = live.read_bytes()
        monkeypatch.setattr(regen.pc, "PILE", tmp_path)

        with pytest.raises(SystemExit) as excinfo:
            regen.main(["--write-live", "--quiet"])

        assert "drifted" in str(excinfo.value) and "verdict_store.py export" in str(excinfo.value)
        assert live.read_bytes() == before, "the banked row was overwritten"


class TestCompareReportsTheThreeKindsSeparately:
    """A count that matches is not rows that match — 81 shared rows differed in #4005."""

    def test_a_string_id_is_one_drift_not_two(self, regen):
        """Untyped, a writer emitting "56" reads as an added row AND a lost row (#4016 review)."""
        reference = [{"image_id": 56, "class": "bowl", "present": True}]
        new = [{"image_id": "56", "class": "bowl", "present": True}]
        diff = regen.compare(new, reference)
        assert diff["only_new"] == [] and diff["only_reference"] == []
        assert diff["differing"] == [(56, "bowl")], "one row, one drift: a type mismatch, not a pair of them"

    def test_only_new_only_reference_and_differing_are_distinct(self, regen):
        reference = [
            {"image_id": 1, "class": "bowl", "present": True},
            {"image_id": 2, "class": "cup", "present": True},
        ]
        new = [{"image_id": 1, "class": "bowl", "present": False}, {"image_id": 3, "class": "cup", "present": True}]
        diff = regen.compare(new, reference)
        assert diff["differing"] == [(1, "bowl")]
        assert diff["only_new"] == [(3, "cup")]
        assert diff["only_reference"] == [(2, "cup")]

    def test_equal_counts_with_different_rows_are_not_clean(self, regen):
        reference = [{"image_id": 1, "class": "bowl", "present": True}]
        new = [{"image_id": 1, "class": "bowl", "present": False}]
        assert len(new) == len(reference)
        assert regen.compare(new, reference)["differing"] == [(1, "bowl")]

    def test_report_returns_nonzero_on_any_drift(self, regen, capsys):
        reference = [{"image_id": 1, "class": "bowl", "present": True}]
        new = [{"image_id": 1, "class": "bowl", "present": False}]
        assert regen.report(regen.compare(new, reference), new, reference) == 1
        assert regen.report(regen.compare(reference, reference), reference, reference) == 0
        capsys.readouterr()
