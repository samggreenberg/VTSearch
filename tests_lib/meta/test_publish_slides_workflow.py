"""Tests for the slide-deck publishing workflow.

`.github/workflows/publish-slides.yml` is the repository's only GitHub Actions
workflow, and it is the one piece of tooling here whose failures are invisible
locally: it renders the decks and pushes the PDFs to the rolling
``slides-latest`` release, so nothing in ``./run-tests.sh`` ever executes it and
a typo only shows up as a stale release nobody is watching.

So the checks below cover exactly the ways it can rot silently -- a renamed
script, a `paths:` filter that stops matching the slide sources, a missing
`contents: write` (the upload 403s), or a trigger drifting onto ``main`` against
the repo's branch policy.

Deliberately nothing here reads ``slides/`` itself. ``./run-tests.sh slides``
rests on a change confined to that directory being unobservable from the test
suite, so reaching in for a deck name to assert against would narrow that fast
path's soundness for no gain -- everything worth pinning about the publish
pipeline is in the workflow and the script, both of which sit outside it.
"""

from __future__ import annotations

import stat
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/publish-slides.yml"
PUBLISH_SCRIPT = REPO_ROOT / "scripts/publish-slides.sh"


def _workflow() -> dict:
    parsed = yaml.safe_load(WORKFLOW.read_text())
    # YAML 1.1 reads a bare `on:` key as the boolean True. That is how every
    # GitHub workflow in the world parses, so normalise rather than fight it.
    if True in parsed:
        parsed["on"] = parsed.pop(True)
    return parsed


def test_workflow_and_script_both_exist() -> None:
    assert WORKFLOW.is_file(), "the publish workflow is gone"
    assert PUBLISH_SCRIPT.is_file(), "the workflow's script is gone"
    mode = PUBLISH_SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "scripts/publish-slides.sh is not executable"


def test_workflow_invokes_the_committed_script() -> None:
    """The logic lives in a script so it can be run from a laptop.

    A workflow that inlined the render and upload would be debuggable only by
    pushing to `dev`, which is the failure mode this split exists to avoid.
    """
    steps = _workflow()["jobs"]["publish"]["steps"]
    runs = " ".join(step.get("run", "") for step in steps)
    assert "scripts/publish-slides.sh" in runs


def test_workflow_triggers_on_dev_when_slides_change() -> None:
    triggers = _workflow()["on"]
    push = triggers["push"]
    assert push["branches"] == ["dev"], (
        "publishing must follow `dev`; CLAUDE.md's branch policy makes `main` a "
        "human-only release branch that slide work never lands on directly"
    )
    # The script is as much a source of the published PDF as the decks are, so a
    # change to it has to republish too.
    for required in ("slides/**", "scripts/publish-slides.sh"):
        assert required in push["paths"], f"{required} would not trigger a republish"
    assert "workflow_dispatch" in triggers, "no way to republish by hand"


def test_workflow_can_write_releases() -> None:
    """Without `contents: write` the asset upload 403s at the last step."""
    assert _workflow()["permissions"]["contents"] == "write"


def test_publish_script_refuses_to_publish_a_dirty_tree() -> None:
    """The release body names a commit, so the PDFs must match it.

    Rendering from a dirty checkout would attach a deck that exists at no commit
    at all, which is exactly the provenance the release is supposed to carry.
    """
    body = PUBLISH_SCRIPT.read_text()
    assert "git status --porcelain -- slides/" in body
    assert "--allow-dirty" in body


def test_publish_script_moves_the_rolling_tag() -> None:
    """GitHub will not re-target a release whose tag already exists.

    So the tag itself has to move on each publish; otherwise `slides-latest`
    keeps pointing at whatever commit first created it while serving assets
    built from a much later one.
    """
    body = PUBLISH_SCRIPT.read_text()
    assert "git push --force origin" in body
    assert "refs/tags/" in body


def test_repository_has_no_other_workflows() -> None:
    """A guard on the claim CLAUDE.md makes about this repo.

    `./run-tests.sh` is the only gate here, and that is only true while no
    workflow runs tests. A second workflow appearing is not necessarily wrong,
    but it has to be a decision someone made on purpose -- and it has to update
    the docs that promise there is no CI backstop.
    """
    workflows = sorted(p.name for p in (REPO_ROOT / ".github/workflows").iterdir())
    assert workflows == ["publish-slides.yml"], (
        f"unexpected workflow(s): {workflows}. If this is deliberate, update the "
        f"'no CI backstop' claims in CLAUDE.md and run-tests.sh in the same commit."
    )
