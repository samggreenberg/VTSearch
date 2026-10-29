"""Tests for `scripts/experiments/study_data.py`, which keeps a study's bulk data out of the tree.

A study's cell CSVs and unlinked figures leave ``docs/experiments/`` for a
tarball on the ``experiments-data`` release (#5404). Two failures would be
silent: archiving a file something still reads (a report link breaks, or a
slide script stops finding its CSV), and an archive whose bytes cannot be put
back. The first is covered by the keep rule's tests, the second by a full
archive -> publish -> fetch round trip in a throwaway repository.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/experiments/study_data.py"
WORKFLOW = REPO_ROOT / ".github/workflows/publish-experiment-data.yml"
STUDY = "2026-01-01-demo"


def _load():
    spec = importlib.util.spec_from_file_location("_study_data", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sd = _load()


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", *args],  # noqa: S607 - git resolved from PATH
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _png(path: Path, colour: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (40, 30), colour)
    image.putpixel((3, 4), (1, 2, 3))
    image.save(path)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository with one study holding one file of each kind the keep rule sorts."""
    study = tmp_path / "docs/experiments" / STUDY
    _write(
        study / "REPORT.md",
        "# Demo\n\n![fig](figures/linked.png)\n\n[viewer](viewer.html)\n\n[`linked.png`](figures/linked.png)\n\n"
        f"The cut is in `docs/experiments/{STUDY}/cited.csv`. Cells are in `cells.csv`.\n",
    )
    _png(study / "figures/linked.png", (200, 10, 10))
    _png(study / "figures/from_page.png", (10, 200, 10))
    _png(study / "figures/unlinked.png", (10, 10, 200))
    _write(study / "viewer.html", '<html><img src="figures/from_page.png"></html>\n')
    _write(study / "cells.csv", "cell,cost\na,0.1\n")
    _write(study / "cited.csv", "k,v\n")
    _write(study / "tables/read_by_test.csv", "x\n1\n")
    _write(study / "tables/rep1-steps.csv", "x\n1\n")
    _write(study / "analyze.py", "print('rebuilds the figures')\n")
    # Live code: names the study and the files it opens, one through an f-string.
    _write(
        tmp_path / "tests_lib/test_demo.py",
        f'STUDY = "{STUDY}"\nA = "read_by_test.csv"\nB = [f"rep{{r}}-steps.csv" for r in (1,)]\n',
    )
    # Experiment code reading a finished study keeps nothing in the tree.
    _write(tmp_path / "scripts/experiments/rebuild.py", f'P = "{STUDY}/cells.csv"\n')
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "study")
    monkeypatch.setattr(sd, "ROOT", tmp_path)
    return tmp_path


def _status(repo: Path) -> str:
    return _git(repo, "status", "--porcelain", "--untracked-files=all")


def test_keep_rule_moves_only_what_nothing_reads(repo: Path) -> None:
    tree = sd.Tree.load()
    prefix = f"docs/experiments/{STUDY}/"
    archivable = {p.removeprefix(prefix) for p in tree.archivable(STUDY)}
    # Linked, embedded by a kept page, cited by full path, read by live code,
    # a page, a script or prose: all stay. The unread cell table and figure go.
    assert archivable == {"cells.csv", "figures/unlinked.png"}


def test_archive_publish_fetch_round_trip(repo: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    study = repo / "docs/experiments" / STUDY
    before = {name: (study / name).read_bytes() for name in ("cells.csv", "figures/unlinked.png")}
    head = _git(repo, "rev-parse", "HEAD")

    assert sd.main(["archive", STUDY]) == 0
    manifest = json.loads((study / "archived-data.json").read_text())
    (batch,) = manifest["batches"]
    assert batch["source_commit"] == head
    assert batch["asset"] == f"{STUDY}.{head[:12]}.tar.gz"
    assert set(batch["files"]) == set(before)
    assert not any((study / name).exists() for name in before)
    _git(repo, "commit", "-qm", "archive")
    assert _status(repo) == ""

    assets = tmp_path_factory.mktemp("assets")
    assert sd.main(["publish", "--out", str(assets)]) == 0
    tarball = (assets / batch["asset"]).read_bytes()
    assert sd.main(["publish", "--out", str(assets)]) == 0
    assert (assets / batch["asset"]).read_bytes() == tarball, "the tarball must be reproducible from git"

    assert sd.main(["fetch", STUDY, "--asset-dir", str(assets)]) == 0
    assert {name: (study / name).read_bytes() for name in before} == before
    assert _status(repo) == "", "fetched files must stay out of git status"

    assert sd.main(["clean", "--all"]) == 0
    assert not any((study / name).exists() for name in before)
    assert _status(repo) == ""

    assert sd.main(["fetch", "--all", "--from-git"]) == 0
    assert {name: (study / name).read_bytes() for name in before} == before


def test_archive_refuses_uncommitted_data(repo: Path) -> None:
    (repo / "docs/experiments" / STUDY / "cells.csv").write_text("edited\n")
    with pytest.raises(SystemExit, match="commit these first"):
        sd.main(["archive", STUDY])


def test_fetch_rejects_bytes_the_manifest_does_not_record(repo: Path, tmp_path_factory) -> None:
    sd.main(["archive", STUDY])
    _git(repo, "commit", "-qm", "archive")
    assets = tmp_path_factory.mktemp("assets")
    sd.main(["publish", "--out", str(assets)])
    (asset,) = assets.iterdir()
    batch = json.loads((repo / "docs/experiments" / STUDY / "archived-data.json").read_text())["batches"][0]
    batch["files"]["cells.csv"]["oid"] = "0" * 40
    (repo / "docs/experiments" / STUDY / "archived-data.json").write_text(json.dumps({"batches": [batch]}))
    with pytest.raises(SystemExit, match="does not match"):
        sd.main(["fetch", STUDY, "--asset-dir", str(asset.parent)])


def test_webp_converts_linked_figures_losslessly_and_repoints_links(repo: Path) -> None:
    study = repo / "docs/experiments" / STUDY
    pixels = {
        name: Image.open(study / f"figures/{name}.png").convert("RGB").tobytes() for name in ("linked", "from_page")
    }

    assert sd.main(["webp", STUDY]) == 0

    for name, original in pixels.items():
        assert not (study / f"figures/{name}.png").exists()
        assert Image.open(study / f"figures/{name}.webp").convert("RGB").tobytes() == original
    assert (study / "figures/unlinked.png").exists(), "an unlinked figure is archived as it is, not converted"
    report = (study / "REPORT.md").read_text()
    assert "![fig](figures/linked.webp)" in report
    assert "[`linked.webp`](figures/linked.webp)" in report, "link text naming the old file follows it"
    assert 'src="figures/from_page.webp"' in (study / "viewer.html").read_text()
    staged = _git(repo, "status", "--porcelain")
    assert "D  docs/experiments/2026-01-01-demo/figures/linked.png" in staged
    assert "A  docs/experiments/2026-01-01-demo/figures/linked.webp" in staged


def test_workflow_publishes_from_dev_with_write_access() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    on = workflow.get("on", workflow.get(True))  # PyYAML reads a bare `on:` key as True
    assert on["push"]["branches"] == ["dev"]
    assert "docs/experiments/*/archived-data.json" in on["push"]["paths"]
    assert workflow["permissions"]["contents"] == "write"
    steps = workflow["jobs"]["publish"]["steps"]
    assert any("study_data.py publish" in step.get("run", "") for step in steps)


def test_archived_files_stay_out_of_the_tree() -> None:
    """A fetched file committed back (`git add -f`) would undo the archive."""
    tracked = set(
        subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(REPO_ROOT), "-c", "core.quotepath=off", "ls-files", "docs/experiments"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\n")
    )
    back = [
        f"docs/experiments/{study}/{name}"
        for path in sorted((REPO_ROOT / "docs/experiments").glob("*/archived-data.json"))
        for study in [path.parent.name]
        for batch in json.loads(path.read_text())["batches"]
        for name in batch["files"]
        if f"docs/experiments/{study}/{name}" in tracked
    ]
    assert back == []
