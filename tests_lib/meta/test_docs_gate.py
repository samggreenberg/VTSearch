"""The documentation drift gate itself: `scripts/check-docs.py`.

The gate's job is to notice when a doc points at something that is no longer
there. A gate that stops noticing is worse than no gate, because the repo now
*believes* its cross-references are live; a gate that cries wolf gets an
allowlist entry pasted in without reading it. So pin both halves: every
invariant fires on a real defect, and stays quiet on the legitimate shapes the
doc set actually uses.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check-docs.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("_docs_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()
FILES, DIRS = gate.build_inventory(gate.tracked_files())
TOP_LEVEL = frozenset({d for d in DIRS if "/" not in d} | {"data", "static"})
# A document location that exists, so relative-link resolution behaves the way
# it does for a real doc. The file itself is never read: text is passed in.
FAKE_DOC = REPO_ROOT / "docs" / "__synthetic__.md"


def run(text: str, doc: Path = FAKE_DOC) -> list[str]:
    """Check `text` as if it were `doc`; return failures as `CHECK:message`."""
    failures = gate.check_markdown(doc, text, FILES, DIRS, TOP_LEVEL, {})
    return [f"{f.check}:{f.message}" for f in failures]


def checks(text: str, doc: Path = FAKE_DOC) -> list[str]:
    return [f.split(":", 1)[0] for f in run(text, doc)]


class TestRepoIsClean:
    """The gate passes on the repo as committed."""

    def test_no_drift(self):
        assert gate.main([]) == 0

    def test_runs_fast_enough_to_gate(self):
        """The gate must stay cheap enough to run on every invocation.

        It times **CPU**, not the wall clock. The suite runs under
        ``pytest -n auto`` on whatever machine is free, so the wall clock
        measures the node's load: a call that takes 6.9 s alone took 20.7 s
        with a second suite on the same GRID node (#4906), and a cold page cache
        alone stretches a 1 s run to 12 s. CPU time is the work the gate does,
        which is what this test guards. The children's CPU is counted too, so
        work moved into a subprocess still shows up.

        The bound is deliberately loose. It catches what the test is for — a
        gate that grew super-linear in repo size, or reintroduced a per-file
        subprocess — not tenths of a second.

        **User** CPU only, not system. A 1.3 s-user run of this gate took 2.9 s
        of system time on a healthy cloud container and 30 s of it on the same
        container once its memory and file paths had degraded late in a
        session (every page fault and read slowed, the gate's own work did
        not). System time is the machine's cost of serving the gate; the
        gate's algorithm is all in user time, and that is what the bound is
        about.
        """
        import resource

        def cpu_seconds() -> float:
            own = resource.getrusage(resource.RUSAGE_SELF)
            children = resource.getrusage(resource.RUSAGE_CHILDREN)
            return own.ru_utime + children.ru_utime

        start = cpu_seconds()
        gate.main([])
        elapsed = cpu_seconds() - start
        assert elapsed < 20.0, f"check-docs used {elapsed:.1f}s of user CPU; it runs on every ./run-tests.sh invocation"


class TestSlugify:
    """GitHub's heading-slug algorithm, including the trap that started this."""

    def test_spaced_hyphen_produces_three_hyphens(self):
        # The bug that killed the whole USER_GUIDE table of contents: a heading
        # with a spaced hyphen slugs to THREE hyphens, and every hand-written
        # link used two.
        assert gate.slugify("Autopilot - the guided workflow") == "autopilot---the-guided-workflow"

    @pytest.mark.parametrize(
        ("heading", "slug"),
        [
            ("The resolution chain", "the-resolution-chain"),
            ("`PluginField`", "pluginfield"),
            ("Resolver hooks (app integration)", "resolver-hooks-app-integration"),
            ("Votes are `dict[int, None]`, not sets", "votes-are-dictint-none-not-sets"),
            ("Region-aware training on patch datasets", "region-aware-training-on-patch-datasets"),
            ("[Linked heading](target.md)", "linked-heading"),
        ],
    )
    def test_slugs(self, heading, slug):
        assert gate.slugify(heading) == slug

    def test_duplicate_headings_get_numeric_suffixes(self):
        anchors = gate.anchors_of("## Notes\n\n## Notes\n\n## Notes\n")
        assert anchors == frozenset({"notes", "notes-1", "notes-2"})

    def test_html_anchors_count(self):
        assert "manual" in gate.anchors_of('<a name="manual"></a>\n\ntext\n')


class TestLinkCheck:
    def test_dead_relative_link_fires(self):
        assert checks("See [base](../vtscore/datasets/importers/base.py).") == ["LINK"]

    def test_live_relative_link_is_quiet(self):
        assert run("See [ML](ML.md) and [core](../vtscore/state/core.py).") == []

    def test_external_links_are_ignored(self):
        assert run("[a](https://example.com/x.py) [b](mailto:x@y.z)") == []

    def test_absolute_link_fires(self):
        assert checks("[docs](/docs/ML.md)") == ["LINK"]

    def test_reference_definition_is_checked(self):
        assert checks("text\n\n[ml]: nope-not-here.md\n") == ["LINK"]

    def test_slide_fragment_links_resolve_against_the_deck_root(self):
        """Fragments author figure paths relative to `slides/`, not to themselves.

        `slides/build.py` repoints them when it assembles a deck, so resolving
        them against `slides/fragments/` would flag every working slide.
        """
        fragment = REPO_ROOT / "slides" / "fragments" / "__synthetic__.md"
        assert run("![bg fit](figs/cost-traces.webp)", fragment) == []

    def test_slide_fragment_links_are_still_checked(self):
        """The override changes the base directory, it does not disable the check."""
        fragment = REPO_ROOT / "slides" / "fragments" / "__synthetic__.md"
        assert checks("![bg fit](figs/no-such-figure.png)", fragment) == ["LINK"]

    def test_link_base_override_does_not_leak_to_other_docs(self):
        """A doc outside the override prefix still resolves against its own dir."""
        assert checks("![bg fit](figs/cost-traces.webp)") == ["LINK"]


class TestAnchorCheck:
    def test_dead_in_page_anchor_fires(self):
        assert checks("[jump](#no-such-heading)\n\n## Real heading\n") == ["ANCHOR"]

    def test_live_in_page_anchor_is_quiet(self):
        assert run("[jump](#real-heading)\n\n## Real heading\n") == []

    def test_dead_cross_file_anchor_fires(self):
        assert checks("[x](ML.md#definitely-not-a-heading-in-ml)") == ["ANCHOR"]

    def test_live_cross_file_anchor_is_quiet(self):
        assert run("[x](ML.md#threshold-calibration)") == []


class TestPathCheck:
    def test_missing_backticked_path_fires(self):
        assert checks("The ABC lives in `vtscore/datasets/importers/base.py`.") == ["PATH"]

    def test_live_backticked_path_is_quiet(self):
        assert run("The ABC lives in `vtscore/datasets/importers/base/core.py`.") == []

    def test_directory_reference_is_quiet(self):
        assert run("Sources live under `vtscore/datasets/sources/`.") == []

    def test_line_number_suffix_is_stripped(self):
        # Only an experiment report may carry a line suffix at all (LINENO
        # refuses one everywhere else), so that is where the PATH check still
        # has to see through it to the file.
        doc = REPO_ROOT / "docs" / "experiments" / "__synthetic__.md"
        assert run("See `vtscore/state/core.py:42`.", doc) == []
        assert checks("See `vtscore/state/gone.py:42`.", doc) == ["PATH"]

    def test_glob_reference_resolves(self):
        assert run("Embedders are `vtscore/media/*/embedder_*.py`.") == []
        assert checks("Embedders are `vtscore/media/*/embedder.py`.") == ["PATH"]

    def test_allowlisted_runtime_and_fictional_paths_are_quiet(self):
        assert run("Settings persist to `data/settings.json`.") == []
        assert run("Drop it at `vtsearch/auth/my_provider.py`.") == []

    def test_non_repo_tokens_are_not_path_claims(self):
        # None of these name anything in this repo, and none of them claim to.
        text = (
            "Send `application/json`. Results land in `results/summary.json` "
            "and `agg/rate_*.csv`. Use `and/or` freely. Module `vtscore.state/`."
        )
        assert run(text) == []

    def test_elided_paths_are_ignored(self):
        assert run("See `frontend/.../browse-canvas/browse-canvas.component.ts`.") == []

    def test_paths_inside_fenced_blocks_are_ignored(self):
        text = "```bash\npython vtsearch/does/not/exist.py\ncat `vtscore/gone.py`\n```\n"
        assert run(text) == []

    def test_code_span_wrapping_a_line_break_still_resolves(self):
        # A span that wraps mis-pairs every backtick after it when the scan is
        # per-line, which silently skipped the references on those lines.
        text = "Override `resolve_file(origin, origin_name, filename) ->\nPath | None` (`vtscore/state/gone.py`).\n"
        assert checks(text) == ["PATH"]

    def test_plans_are_exempt(self):
        # A plan names files its work has not created yet; that is not a claim
        # about this checkout.
        text = "Write the report to `docs/experiments/not-yet/REPORT.md`."
        assert run(text, REPO_ROOT / "docs" / "plans" / "__synthetic__.md") == []
        assert checks(text) == ["PATH"]

    def test_an_experiment_report_may_not_cite_a_script_that_is_not_in_the_tree(self):
        # Experiment reports are NOT exempt. A report that cites its analysis
        # script is claiming the run can be reproduced; when the script was
        # never committed (overview-bench, `label_noise.py`) the claim is false
        # and nothing caught it.
        doc = REPO_ROOT / "docs" / "experiments" / "__synthetic__.md"
        assert checks("Scripted in `scripts/experiments/calibration/gone.py`.", doc) == ["PATH"]

    def test_an_experiment_reports_scratch_paths_are_still_fine(self):
        # The reason the blanket exemption existed: a run record cites the
        # cluster dir it lived in and the artifacts it wrote there. Those are
        # not repo paths, and the top-level-directory rule already skips them.
        doc = REPO_ROOT / "docs" / "experiments" / "__synthetic__.md"
        text = "Aggregates in `agg/cut_contrasts.csv`, cells in `cells/task_*.csv`, run in `results-ab/`."
        assert checks(text, doc) == []


class TestLeakCheck:
    def test_home_directory_path_fires(self):
        assert checks("**See also:** [`/home/user/VTSearch/docs/CLI.md`](CLI.md)") == ["LEAK"]

    def test_macos_home_directory_path_fires(self):
        assert checks("Run it from `/Users/someone/VTSearch`.") == ["LEAK"]

    def test_leak_inside_a_fenced_block_is_ignored(self):
        assert run('```json\n{"filepath": "/home/user/results.json"}\n```\n') == []

    def test_allowlisted_documents_are_exempt(self):
        doc = REPO_ROOT / next(iter(gate.ALLOWED_LEAKS))
        assert "LEAK" not in checks("ran in `/home/someone/experiments/x`", doc)


class TestFenceCheck:
    def test_fence_preceded_by_text_fires(self):
        assert checks("Response -> ```json\n{}\n```\n") == ["FENCE"]

    def test_blockquoted_fence_is_valid(self):
        assert run("> ```bash\n> ./run-tests.sh\n> ```\n") == []

    def test_normal_fence_is_quiet(self):
        assert run("```python\nx = 1\n```\n") == []


class TestLineAnchorCheck:
    """A line number is wrong by the next edit to the file it points into (#5290)."""

    @pytest.mark.parametrize(
        "text",
        [
            "Scored in `vtscore/state/core.py:42`.",
            "Scored in vtscore/state/core.py:42, unquoted.",
            "The guard lives in `browse-canvas.component.ts:120`.",
            "A range: `vtscore/state/core.py:42-60`.",
            "See the guard (line 42).",
            "See the guard (lines 42-60).",
            "See the guard (Line 42 of `vtscore/state/core.py`).",
            "pyproject.toml (~line 127) pins it.",
        ],
    )
    def test_both_spellings_fire(self, text):
        assert checks(text) == ["LINENO"]

    def test_message_names_the_anchor(self):
        assert run("See `vtscore/state/core.py:42`.") == [
            "LINENO:line-number anchor 'vtscore/state/core.py:42'; cite the module and symbol instead, "
            "since a line number rots on the next edit"
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "Cite `vtscore/state/core.py::DatasetContext` by symbol.",
            "Run `pytest tests_lib/meta/test_docs_gate.py::TestLineAnchorCheck`.",
            "The app listens on `localhost:5000`; the job starts at 10:30.",
            "Write `file.py:NNN` and `(line NNN)` when describing the rule itself.",
            "A one-line 2x speedup, on line two (of three).",
        ],
    )
    def test_legitimate_shapes_are_quiet(self, text):
        assert run(text) == []

    def test_fenced_tool_output_is_ignored(self):
        # Quoted output carries line numbers of its own; it is not a pointer.
        text = '```text\ntests/test_x.py:42: AssertionError\n  File "x.py", line 199 in f (line 199)\n```\n'
        assert run(text) == []

    def test_plans_and_readmes_are_held_to_it(self):
        # The old check only looked under vtscore/docs/; these are the docs it missed.
        for rel in ("docs/plans/__synthetic__.md", "README.md", "scripts/screenshots/__synthetic__.md"):
            assert checks("See `vtscore/state/core.py:42`.", REPO_ROOT / rel) == ["LINENO"], rel

    def test_experiment_reports_are_exempt(self):
        # A study record describes the code at the commit it ran on.
        doc = REPO_ROOT / "docs" / "experiments" / "2026-01-01-x" / "REPORT.md"
        assert run("Scored in `vtscore/state/core.py:42` (line 42).", doc) == []


class TestPlanRefCheck:
    def test_repo_has_no_dangling_plan_citations(self):
        assert gate.check_plan_refs(FILES) == []

    def test_deleted_plan_citation_fires(self):
        text = '"""See docs/plans/long-since-shipped.md for the rationale."""\n'
        failures = gate.check_plan_refs_in(REPO_ROOT / "vtscore" / "__synthetic__.py", text, FILES)
        assert [f.check for f in failures] == ["PLAN"]

    def test_live_plan_citation_is_quiet(self):
        text = "See docs/plans/codebase-audit-2026-08.md for the audit.\n"
        assert gate.check_plan_refs_in(REPO_ROOT / "vtscore" / "__synthetic__.py", text, FILES) == []

    def test_non_markdown_sources_are_scanned(self):
        # The point of scanning the whole tree: module docstrings cite plans far
        # more often than other plans do.
        assert any(
            not rel.endswith(".md") and "docs/plans/" in (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
            for rel in FILES
            if rel.endswith(".py")
        )


class TestStudyDirCheck:
    """Study directories are dated, or `ls` stops sorting the archive."""

    def test_repo_studies_are_all_dated(self):
        assert gate.check_study_dirs(DIRS) == []

    def test_undated_study_directory_fires(self):
        dirs = DIRS | {"docs/experiments/brand-new-study"}
        failures = gate.check_study_dirs(frozenset(dirs))
        assert [f.check for f in failures] == ["STUDY"]
        assert "YYYY-MM-DD" in failures[0].message

    def test_dated_study_directory_passes_without_any_index(self):
        # No index file to keep in step (#4138): a dated directory is enough.
        dirs = DIRS | {"docs/experiments/2026-01-01-brand-new-study"}
        assert gate.check_study_dirs(frozenset(dirs)) == []

    def test_nested_directories_inside_a_study_are_not_studies(self):
        # `figures/`, `agg/` and friends live one level down and are not named
        # for a date; only the study directory itself is.
        dirs = DIRS | {
            "docs/experiments/2026-01-01-brand-new-study",
            "docs/experiments/2026-01-01-brand-new-study/figures",
        }
        # Undated `figures/` would fire if it were taken for a study.
        assert gate.check_study_dirs(frozenset(dirs)) == []


# A run-tests.sh and a docs/TESTING.md cut down to the shapes the GATES check
# reads: a commented stage-1 block, lanes of all three kinds (one started from
# two places, one with a shell expansion in its name), a row whose case differs
# from the lane's, and a table after the section that must not be read.
GATE_SCRIPT = """\
# Stage 1: the stale-tree check, alone.
# See scripts/check-phantom-base.py for both signals.
if ! python scripts/check-phantom-base.py ; then
    _blocked "stale tree"
fi
# Stage 2: the cheap gates.
_start_lane gate "ruff check" _gate_ruff_check
_start_lane gate "documentation" _gate_check_docs
# Stage 3: the heavy gates.
_start_lane gating "pyright ($_pyright_threads thread(s))" _lane_pyright
_start_lane free "pip-audit" _lane_pip_audit
if $_late; then
    _start_lane free "pip-audit" _lane_pip_audit
fi
"""

GATE_DOC = """\
# Testing

### The stages

**Stage 1 — the stale-tree check, alone:**

| Gate | Command it runs | Notes |
|------|-----------------|-------|
| Stale tree | `scripts/check-phantom-base.py` | |

**Stage 2 — cheap gates, concurrently:**

| Gate | Command it runs | Notes |
|------|-----------------|-------|
| ruff check | `ruff check .` | |
| Documentation | `scripts/check-docs.py` | |

**Stage 3 — heavy gates:**

| Gate | Command it runs | Lane | When | Notes |
|------|-----------------|------|------|-------|
| pyright (N thread(s)) | `scripts/pyright-incremental.py --threads N` | gating | Full run | |
| pip-audit | `pip-audit` | free, collected after pytest | As pyright | |

**Stage 4 — pytest, alone.**

## Test Groups

| Group | Tier | Description |
|-------|------|-------------|
| `core` | both | A table outside the section, so not a gate. |
"""


def gate_failures(script: str = GATE_SCRIPT, doc: str = GATE_DOC) -> list[str]:
    """GATES failures as `<file>:<message>`, so a test can say which side fired."""
    return [f"{f.path.name}:{f.message}" for f in gate.check_gate_tables(script, doc)]


class TestGateTableCheck:
    """docs/TESTING.md's stage tables list the gates run-tests.sh runs (#5289)."""

    def test_repo_tables_match_run_tests(self):
        assert gate.check_gate_tables_in_repo() == []

    def test_matching_fixture_is_quiet(self):
        # Case aside, `N` for a shell expansion, and a lane started twice is
        # one gate.
        assert gate_failures() == []

    def test_gate_without_a_row_fires_on_the_script(self):
        script = GATE_SCRIPT + '_start_lane gate "deptry" _gate_deptry\n'
        [failure] = gate_failures(script=script)
        assert failure.startswith("run-tests.sh:") and "'deptry'" in failure

    def test_row_without_a_gate_fires_on_the_doc(self):
        doc = GATE_DOC.replace("| ruff check |", "| Spelling | `codespell` | |\n| ruff check |")
        [failure] = gate_failures(doc=doc)
        assert failure.startswith("TESTING.md:") and "'Spelling'" in failure

    def test_a_row_that_labels_the_gate_differently_fires_both_ways(self):
        # The shape of the drift this check found on its first run: a table
        # that said "Lint" for the gate the run prints as "ruff check".
        doc = GATE_DOC.replace("| ruff check |", "| Lint |")
        failures = gate_failures(doc=doc)
        assert len(failures) == 2
        assert any(f.startswith("TESTING.md:") and "'Lint'" in f for f in failures)
        assert any(f.startswith("run-tests.sh:") and "'ruff check'" in f for f in failures)

    def test_row_in_the_wrong_stage_fires(self):
        script = GATE_SCRIPT.replace('gate "documentation"', 'free "documentation"')
        [failure] = gate_failures(script=script)
        assert "'Documentation'" in failure and "stage 3" in failure

    def test_row_with_the_wrong_lane_kind_fires(self):
        doc = GATE_DOC.replace("| free, collected after pytest |", "| gating |")
        [failure] = gate_failures(doc=doc)
        assert "'pip-audit'" in failure and "'gating'" in failure and "free" in failure

    def test_repeated_row_fires(self):
        doc = GATE_DOC.replace(
            "| ruff check | `ruff check .` | |", "| ruff check | `ruff check .` | |\n| Ruff check | x | |"
        )
        [failure] = gate_failures(doc=doc)
        assert "repeats" in failure

    def test_unknown_lane_kind_fires(self):
        script = GATE_SCRIPT.replace('gate "documentation"', 'eager "documentation"')
        failures = gate_failures(script=script)
        assert any("'eager'" in f for f in failures)

    def test_commented_out_lane_is_not_a_gate(self):
        script = GATE_SCRIPT + '# _start_lane gate "retired" _gate_retired\n'
        assert gate_failures(script=script) == []

    def test_stage1_script_without_a_row_fires(self):
        script = GATE_SCRIPT.replace(
            "fi\n# Stage 2", "fi\nif ! python scripts/check-something-new.py ; then exit 1; fi\n# Stage 2"
        )
        [failure] = gate_failures(script=script)
        assert failure.startswith("run-tests.sh:") and "scripts/check-something-new.py" in failure

    def test_stage1_row_the_script_does_not_run_fires(self):
        doc = GATE_DOC.replace("`scripts/check-phantom-base.py`", "`scripts/check-retired.py`")
        failures = gate_failures(doc=doc)
        assert any(f.startswith("TESTING.md:") and "scripts/check-retired.py" in f for f in failures)
        assert any(f.startswith("run-tests.sh:") and "scripts/check-phantom-base.py" in f for f in failures)

    def test_missing_stages_section_fails_loudly(self):
        # A renamed heading must not read as "no rows, so nothing to compare".
        doc = GATE_DOC.replace("### The stages", "### How the run is staged")
        [failure] = gate_failures(doc=doc)
        assert "The stages" in failure

    def test_script_without_lanes_fails_loudly(self):
        script = "\n".join(line for line in GATE_SCRIPT.splitlines() if "_start_lane" not in line)
        failures = gate_failures(script=script)
        assert any("_start_lane" in f for f in failures)

    def test_escaped_pipe_stays_inside_its_cell(self):
        assert gate.table_cells(r"| a | `x \| y` | c |") == ["a", r"`x \| y`", "c"]


class TestAllowlistsStayHonest:
    """An allowlist entry that no longer describes reality is dead weight."""

    def test_fictional_example_paths_do_not_exist(self):
        for token, reason in gate.ALLOWED_PATHS.items():
            if "fictional" not in reason:
                continue
            assert not (REPO_ROOT / token).exists(), (
                f"{token} is allowlisted as a fictional example but now exists; drop the entry"
            )

    def test_leak_allowlist_entries_point_at_real_files(self):
        for rel in gate.ALLOWED_LEAKS:
            assert (REPO_ROOT / rel).exists(), f"{rel} is allowlisted for path leaks but no longer exists"

    def test_line_anchor_allowlist_entries_point_at_real_paths(self):
        for rel in gate.ALLOWED_LINE_ANCHORS:
            assert (REPO_ROOT / rel).exists(), f"{rel} may cite line numbers but no longer exists"

    def test_plan_self_reference_entries_point_at_real_files(self):
        for rel in gate.PLAN_SELF_REFERENCE:
            assert (REPO_ROOT / rel).exists(), f"{rel} is exempt from the PLAN check but no longer exists"

    def test_link_base_overrides_point_at_real_directories(self):
        for prefix, base in gate.LINK_BASE_OVERRIDES.items():
            assert (REPO_ROOT / prefix).is_dir(), f"{prefix} has a link-base override but no longer exists"
            assert (REPO_ROOT / base).is_dir(), f"{prefix} resolves links against {base}, which no longer exists"
