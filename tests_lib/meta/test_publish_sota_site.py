"""Tests for the State of the App links on GitHub Pages.

`.github/workflows/publish-sota.yml` builds a small site with
`scripts/publish-sota-site.py` and deploys it on every release to ``main`` that
touches a report. Like the slide publisher, nothing in ``./run-tests.sh`` ever
runs the deploy, so its failures would show only as a link that quietly stops
moving. The checks below cover the ways it can rot: the builder picking the
wrong report, a review whose title no longer parses falling off the links, and
the workflow drifting off ``main`` or losing the Pages permissions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/publish-sota.yml"
BUILDER = REPO_ROOT / "scripts/publish-sota-site.py"

# Write-ups whose directory says State of the App but whose title names no
# review kind, on purpose. A new entry is a decision: a review that lands here
# by a mistyped title would never reach its stable link.
EXPECTED_SKIPS = {"2026-10-01-state-of-the-app-per-floor"}


def _load_builder():
    spec = importlib.util.spec_from_file_location("_publish_sota_site", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


site = _load_builder()


def _report(experiments: Path, directory: str, title: str, *, viewer: str | None = None) -> None:
    d = experiments / directory
    d.mkdir(parents=True)
    (d / "REPORT.md").write_text(f"# {title}\n\nbody\n", encoding="utf-8")
    if viewer is not None:
        (d / "viewer.html").write_text(viewer, encoding="utf-8")


def _build(tmp_path: Path, experiments: Path):
    out = tmp_path / "site"
    series, skipped = site.build(out, experiments=experiments, repo="o/r", ref="main", sha="abc1234")
    return out, {s.slug: s for s in series}, skipped


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("# State of the App: Binary Photo — 2026-10-05", "Binary Photo"),
        ("# State of the App: Region Photo - 2026-09-24", "Region Photo"),
        ("# State of the App: Document Logo, round 3", "Document Logo"),
        ("# State of the App: Document Logo (#4457)", "Document Logo"),
        ("# State of the App: Structural Document", "Document Logo"),
        ("# State of the App per floor: sessions at P = 10 / 50 / 90%", None),
        ("# Some other experiment", None),
    ],
)
def test_parse_kind(line: str, kind: str | None) -> None:
    assert site.parse_kind(line) == kind


def test_newest_report_of_each_kind_wins(tmp_path: Path) -> None:
    exp = tmp_path / "experiments"
    _report(exp, "2026-09-24-state-of-the-app-binary-photo", "State of the App: Binary Photo — 2026-09-24")
    _report(exp, "2026-10-05-state-of-the-app-binary-photo", "State of the App: Binary Photo — 2026-10-05")
    _report(exp, "2026-09-30-state-of-the-app-binary-photo", "State of the App: Binary Photo — 2026-09-30")
    _report(exp, "2026-10-01-state-of-the-app-structural-document", "State of the App: Structural Document")
    _report(exp, "2026-10-03-state-of-the-app-document-logo-4457", "State of the App: Document Logo, round 3")
    _report(exp, "2026-10-07-unrelated-study", "Unrelated study")

    out, by_slug, skipped = _build(tmp_path, exp)

    assert sorted(by_slug) == ["binary-photo", "document-logo"]
    assert [r.date for r in by_slug["binary-photo"].reports] == ["2026-10-05", "2026-09-30", "2026-09-24"]
    assert by_slug["document-logo"].latest.directory == "2026-10-03-state-of-the-app-document-logo-4457"
    assert skipped == []
    redirect = (out / "sota/binary-photo/index.html").read_text()
    assert (
        'url=https://github.com/o/r/blob/main/docs/experiments/2026-10-05-state-of-the-app-binary-photo/REPORT.md"'
        in redirect
    )


def test_viewer_is_copied_byte_for_byte(tmp_path: Path) -> None:
    exp = tmp_path / "experiments"
    viewer = "<!doctype html><title>viewer</title>é"
    _report(exp, "2026-10-05-state-of-the-app-binary-photo", "State of the App: Binary Photo", viewer=viewer)

    out, _, _ = _build(tmp_path, exp)

    assert (out / "sota/binary-photo/viewer.html").read_text(encoding="utf-8") == viewer
    assert 'href="binary-photo/viewer.html"' in (out / "sota/index.html").read_text()


def test_missing_viewer_is_said_not_borrowed(tmp_path: Path) -> None:
    """An older report's viewer must never answer for the newest one."""
    exp = tmp_path / "experiments"
    _report(exp, "2026-09-24-state-of-the-app-binary-photo", "State of the App: Binary Photo", viewer="OLD VIEWER")
    _report(exp, "2026-10-05-state-of-the-app-binary-photo", "State of the App: Binary Photo")

    out, _, _ = _build(tmp_path, exp)

    page = (out / "sota/binary-photo/viewer.html").read_text()
    assert "OLD VIEWER" not in page
    assert "No viewer for this report" in page
    assert "2026-10-05-state-of-the-app-binary-photo/REPORT.md" in page


def test_untitled_state_of_the_app_dir_is_reported(tmp_path: Path) -> None:
    exp = tmp_path / "experiments"
    _report(exp, "2026-10-05-state-of-the-app-binary-photo", "State of the App: Binary Photo")
    _report(exp, "2026-10-06-state-of-the-app-region-photo", "Region Photo review")  # title lost its prefix

    _, by_slug, skipped = _build(tmp_path, exp)

    assert sorted(by_slug) == ["binary-photo"]
    assert skipped == ["2026-10-06-state-of-the-app-region-photo"]


def test_rebuild_replaces_the_site(tmp_path: Path) -> None:
    exp = tmp_path / "experiments"
    _report(exp, "2026-10-05-state-of-the-app-binary-photo", "State of the App: Binary Photo")
    out = tmp_path / "site"
    (out / "sota/stale-kind").mkdir(parents=True)

    site.build(out, experiments=exp, repo="o/r", ref="main", sha="abc1234")

    assert not (out / "sota/stale-kind").exists()


def test_the_real_reports_all_reach_a_link(tmp_path: Path) -> None:
    """Every State of the App review in the tree is published, under the kinds we expect.

    Asserting kinds rather than dates keeps this from rotting at the next review;
    what it catches is a report whose title stopped parsing.
    """
    out, by_slug, skipped = _build(tmp_path, REPO_ROOT / "docs/experiments")

    assert {"binary-photo", "region-photo", "document-logo"} <= set(by_slug)
    unexpected = set(skipped) - EXPECTED_SKIPS
    assert not unexpected, (
        f"{sorted(unexpected)} look like State of the App reports but their REPORT.md titles name no kind, "
        "so they would never reach a stable link. Title them '# State of the App: <Kind> — <date>', or add "
        "them to EXPECTED_SKIPS if they are deliberately not a review."
    )
    for slug, series in by_slug.items():
        assert (REPO_ROOT / "docs/experiments" / series.latest.directory / "REPORT.md").is_file()
        assert (out / "sota" / slug / "index.html").is_file()
        assert (out / "sota" / slug / "viewer.html").is_file()


def _workflow() -> dict:
    parsed = yaml.safe_load(WORKFLOW.read_text())
    if True in parsed:  # YAML 1.1 reads a bare `on:` as True
        parsed["on"] = parsed.pop(True)
    return parsed


def test_workflow_publishes_from_main_when_a_report_changes() -> None:
    """The links follow releases (owner, 2026-10-06), not every merge to dev.

    `main` is also the only branch the github-pages environment deploys from
    by default, so a trigger on any other branch would fail at the deploy step.
    """
    triggers = _workflow()["on"]
    push = triggers["push"]
    assert push["branches"] == ["main"]
    for required in (
        "docs/experiments/*state-of-the-app*/**",
        "scripts/publish-sota-site.py",
        ".github/workflows/publish-sota.yml",
    ):
        assert required in push["paths"], f"{required} would not trigger a republish"
    assert "workflow_dispatch" in triggers, "no way to republish by hand"


def test_workflow_can_deploy_pages() -> None:
    perms = _workflow()["permissions"]
    assert perms["pages"] == "write"
    assert perms["id-token"] == "write", "deploy-pages authenticates with an OIDC token"
    job = _workflow()["jobs"]["publish"]
    assert job["environment"]["name"] == "github-pages"


def test_workflow_invokes_the_committed_builder() -> None:
    steps = _workflow()["jobs"]["publish"]["steps"]
    runs = " ".join(step.get("run", "") for step in steps)
    assert "scripts/publish-sota-site.py" in runs
    uses = [step.get("uses", "") for step in steps]
    assert any(u.startswith("actions/upload-pages-artifact@") for u in uses)
    assert any(u.startswith("actions/deploy-pages@") for u in uses)
