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
        import time

        start = time.perf_counter()
        gate.main([])
        assert time.perf_counter() - start < 5.0


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
        assert run("See `vtscore/state/core.py:42`.") == []
        assert checks("See `vtscore/state/gone.py:42`.") == ["PATH"]

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

    def test_plans_and_experiments_are_exempt(self):
        # Plans name files their work has not created yet; experiment reports
        # name a cluster scratch dir. Neither is a claim about this checkout.
        text = "Write the report to `docs/experiments/not-yet/REPORT.md`."
        assert run(text, REPO_ROOT / "docs" / "plans" / "__synthetic__.md") == []
        assert run(text, REPO_ROOT / "docs" / "experiments" / "__synthetic__.md") == []
        assert checks(text) == ["PATH"]


class TestLeakCheck:
    def test_home_directory_path_fires(self):
        assert checks("**See also:** [`/home/user/VTSearch/docs/CLI.md`](../CLI.md)") == ["LEAK"]

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


class TestPlanRefCheck:
    def test_repo_has_no_dangling_plan_citations(self):
        assert gate.check_plan_refs(FILES) == []

    def test_deleted_plan_citation_fires(self):
        text = '"""See docs/plans/long-since-shipped.md for the rationale."""\n'
        failures = gate.check_plan_refs_in(REPO_ROOT / "vtscore" / "__synthetic__.py", text, FILES)
        assert [f.check for f in failures] == ["PLAN"]

    def test_live_plan_citation_is_quiet(self):
        text = "See docs/plans/documentation-accuracy.md for the audit.\n"
        assert gate.check_plan_refs_in(REPO_ROOT / "vtscore" / "__synthetic__.py", text, FILES) == []

    def test_non_markdown_sources_are_scanned(self):
        # The point of scanning the whole tree: module docstrings cite plans far
        # more often than other plans do.
        assert any(
            not rel.endswith(".md") and "docs/plans/" in (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
            for rel in FILES
            if rel.endswith(".py")
        )


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
