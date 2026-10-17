"""The incremental pyright gate: it re-checks exactly the files whose inputs changed.

``scripts/pyright-incremental.py`` lets a session's second and later full
runs pay for the files they changed (plus everything importing them) instead
of the 239s whole-tree check. That is sound only if its import resolution
never misses a repository module and its baseline is only ever written by a
green run -- a miss in either direction is a type error that silently goes
unreported. So the resolution rules, the closure, and the baseline semantics
are each pinned here against throwaway git repositories, and one test walks
the real tree to prove the resolver never calls a repo module external.

The real pyright is never invoked: the runner is injected, and records what
it would have been asked to run.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "pyright-incremental.py"


def _load():
    spec = importlib.util.spec_from_file_location("_pyright_incremental", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MOD = _load()

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def git(repo: Path, *args: str) -> str:
    env = {**os.environ, **_GIT_ENV}
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(k, None)
    out = subprocess.run(  # noqa: S603  # fixed argv, no shell
        ["git", *args],  # noqa: S607  # git resolves on PATH
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


CONFIG = """{
  // comments and a trailing comma, as the real file allows
  "include": ["pkg", "tests",],
  "exclude": ["**/__pycache__", "scripts"],
  "extraPaths": ["tests", "scripts/experiments"],
}
"""

#: The fixture tree. Leaves that import nothing (a, f, g) stay out of every
#: closure but their own; b -> a, and c/d/e -> base, give the closure shapes.
FILES = {
    "pyrightconfig.json": CONFIG,
    "pyproject.toml": "[tool.ruff]\nline-length = 120\n",
    "pkg/__init__.py": "",
    "pkg/base.py": "BASE = 1\n",
    "pkg/a.py": "A = 1\n",
    "pkg/b.py": "from pkg.a import A\n",
    "pkg/c.py": "from pkg import base\n",
    "pkg/d.py": "from pkg import base\n",
    "pkg/e.py": "from pkg import base\n",
    "pkg/f.py": "",
    "pkg/g.py": "",
    "tests/test_b.py": "import pkg.b\n",
    "tests/test_exp.py": "def test():\n    import expmod\n",
    "scripts/experiments/expmod.py": "X = 1\n",
}
CHECKABLE = 11  # pkg/* (9) + tests/* (2); scripts/ is excluded


def write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "dev")
    for rel, text in FILES.items():
        write(r, rel, text)
    git(r, "add", "-A")
    git(r, "commit", "-qm", "base")
    return r


@pytest.fixture(autouse=True)
def _stable_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real pyright, no real distribution scan: the environment key is a constant."""
    monkeypatch.setenv("PYRIGHT_PYTHON_FORCE_VERSION", "1.1.414")
    monkeypatch.delenv("VTSEARCH_FRESH", raising=False)
    monkeypatch.setattr(MOD, "environment_key", lambda: "env-A")


class Recorder:
    """Stands in for pyright: records the argv it was handed, answers a fixed status."""

    def __init__(self, status: int = 0) -> None:
        self.status = status
        self.calls: list[tuple[list[str], Path]] = []

    def __call__(self, argv: list[str], cwd: Path) -> int:
        self.calls.append((list(argv), Path(cwd)))
        return self.status


def baseline_of(repo: Path) -> Path:
    return repo / ".gates-cache" / "pyright-baseline.json"


def run(repo: Path, *extra: str, status: int = 0) -> tuple[int, Recorder]:
    rec = Recorder(status)
    rc = MOD.main(["--repo", str(repo), *extra], runner=rec)
    return rc, rec


def green_full_run(repo: Path) -> None:
    """Establish a baseline the way a session's first full run does."""
    rc, rec = run(repo)
    assert rc == 0
    assert rec.calls == [(["pyright"], repo)]
    assert baseline_of(repo).is_file()


def plan(repo: Path):
    return MOD.make_plan(repo, baseline_of(repo))


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


class TestResolution:
    ROOT = ("",)

    def resolve(self, spec, importer, files, roots=ROOT):
        return MOD.resolve(spec, importer, set(files), roots)

    def test_absolute_import_depends_on_the_package_chain_and_the_module(self) -> None:
        files = ["a/__init__.py", "a/b/__init__.py", "a/b/c.py", "a/other.py"]
        assert self.resolve(("a.b.c", 0, ()), "t.py", files) == {"a/__init__.py", "a/b/__init__.py", "a/b/c.py"}

    def test_from_import_of_a_submodule_adds_the_submodule(self) -> None:
        files = ["a/__init__.py", "a/b/__init__.py", "a/b/c.py", "a/b/z.py"]
        # `d` is an attribute of a.b, not a module: nothing extra for it.
        assert self.resolve(("a.b", 0, ("c", "d")), "t.py", files) == {
            "a/__init__.py",
            "a/b/__init__.py",
            "a/b/c.py",
        }

    def test_from_import_of_a_module_file(self) -> None:
        files = ["a/__init__.py", "a/b.py"]
        assert self.resolve(("a.b", 0, ("c",)), "t.py", files) == {"a/__init__.py", "a/b.py"}

    def test_relative_import_one_level(self) -> None:
        files = ["pkg/sub/__init__.py", "pkg/sub/x.py", "pkg/sub/m.py"]
        assert self.resolve(("", 1, ("x",)), "pkg/sub/m.py", files) == {"pkg/sub/__init__.py", "pkg/sub/x.py"}

    def test_relative_import_two_levels_with_a_module(self) -> None:
        files = ["pkg/__init__.py", "pkg/m2.py", "pkg/sub/m.py", "pkg/sub/__init__.py"]
        assert self.resolve(("m2", 2, ("y",)), "pkg/sub/m.py", files) == {"pkg/__init__.py", "pkg/m2.py"}

    def test_relative_import_from_a_root_level_file(self) -> None:
        assert self.resolve(("", 1, ("x",)), "conftest.py", ["x.py", "y.py"]) == {"x.py"}

    def test_relative_import_above_the_root_is_external(self) -> None:
        assert self.resolve(("", 3, ("z",)), "pkg/m.py", ["z.py", "pkg/__init__.py"]) == set()

    def test_star_import_depends_on_the_module_alone(self) -> None:
        assert self.resolve(("a", 0, ("*",)), "t.py", ["a.py", "a/x.py"]) == {"a.py"}

    def test_a_name_that_is_both_a_file_and_a_package_depends_on_both(self) -> None:
        files = ["x.py", "x/__init__.py", "x/y.py"]
        assert self.resolve(("x.y", 0, ()), "t.py", files) == {"x.py", "x/__init__.py", "x/y.py"}

    def test_stub_files_resolve_like_modules(self) -> None:
        assert self.resolve(("a", 0, ()), "t.py", ["a.pyi"]) == {"a.pyi"}

    def test_extra_paths_root(self) -> None:
        files = ["scripts/experiments/expmod.py"]
        assert self.resolve(("expmod", 0, ()), "tests/t.py", files) == set()
        assert self.resolve(("expmod", 0, ()), "tests/t.py", files, ("", "scripts/experiments")) == set(files)

    def test_an_external_import_resolves_to_nothing(self) -> None:
        assert self.resolve(("numpy", 0, ("ndarray",)), "t.py", ["a.py", "pkg/__init__.py"]) == set()

    def test_imports_are_found_at_any_nesting_depth(self) -> None:
        text = (
            "import a\n"
            "from b import c, d\n"
            "from . import e\n"
            "from ..f import *\n"
            "def g():\n"
            "    import h.i\n"
            "    try:\n"
            "        from j import k\n"
            "    except ImportError:\n"
            "        pass\n"
            "if TYPE_CHECKING:\n"
            "    from l import m\n"
        )
        assert set(MOD.parse_imports(text)) == {
            ("a", 0, ()),
            ("b", 0, ("c", "d")),
            ("", 1, ("e",)),
            ("f", 2, ("*",)),
            ("h.i", 0, ()),
            ("j", 0, ("k",)),
            ("l", 0, ("m",)),
        }

    def test_a_file_that_does_not_parse_has_no_imports(self) -> None:
        assert MOD.parse_imports("def (:\n") == []


class TestConfig:
    def test_reads_jsonc_with_comments_and_trailing_commas(self, repo: Path) -> None:
        cfg = MOD.load_config(repo)
        assert cfg.include == ("pkg", "tests")
        assert cfg.exclude == ("**/__pycache__", "scripts")
        assert cfg.roots == ("", "tests", "scripts/experiments")

    def test_checkable_mirrors_include_minus_exclude(self, repo: Path) -> None:
        cfg = MOD.load_config(repo)
        assert MOD.is_checkable("pkg/a.py", cfg)
        assert MOD.is_checkable("tests/test_b.py", cfg)
        assert not MOD.is_checkable("scripts/experiments/expmod.py", cfg)
        assert not MOD.is_checkable("pkg/__pycache__/a.py", cfg)
        assert not MOD.is_checkable("setup.py", cfg)
        assert not MOD.is_checkable("pkg/data.json", cfg)

    def test_the_real_config_loads(self) -> None:
        cfg = MOD.load_config(REPO_ROOT)
        assert "vtscore" in cfg.include and "tests_lib" in cfg.include
        assert "scripts/experiments/pile" in cfg.roots
        assert MOD.is_checkable("vtscore/config/paths.py", cfg)
        assert not MOD.is_checkable("scripts/check-docs.py", cfg)
        assert not MOD.is_checkable("app.py", cfg)


# ---------------------------------------------------------------------------
# the closure
# ---------------------------------------------------------------------------


class TestClosure:
    def test_changing_a_module_rechecks_its_importers_transitively(self, repo: Path) -> None:
        green_full_run(repo)
        write(repo, "pkg/a.py", "A = 2\n")
        p = plan(repo)
        assert p.mode == "scoped"
        assert p.changed == ["pkg/a.py"]
        assert p.targets == ["pkg/a.py", "pkg/b.py", "tests/test_b.py"]
        assert p.checkable == CHECKABLE

    def test_changing_a_mid_module_rechecks_only_what_is_above_it(self, repo: Path) -> None:
        green_full_run(repo)
        write(repo, "pkg/b.py", "from pkg.a import A\nB = A\n")
        assert plan(repo).targets == ["pkg/b.py", "tests/test_b.py"]

    def test_changing_a_leaf_rechecks_the_leaf_alone(self, repo: Path) -> None:
        green_full_run(repo)
        write(repo, "tests/test_b.py", "import pkg.b\nassert pkg.b\n")
        assert plan(repo).targets == ["tests/test_b.py"]

    def test_deleting_a_module_rechecks_its_former_importers(self, repo: Path) -> None:
        green_full_run(repo)
        (repo / "pkg" / "a.py").unlink()
        p = plan(repo)
        assert p.changed == ["pkg/a.py"]
        assert p.targets == ["pkg/b.py", "tests/test_b.py"]

    def test_a_new_file_is_its_own_target(self, repo: Path) -> None:
        green_full_run(repo)
        write(repo, "tests/test_c.py", "from pkg.a import A\n")
        assert plan(repo).targets == ["tests/test_c.py"]

    def test_a_package_init_is_an_input_of_every_import_through_it(self, repo: Path) -> None:
        green_full_run(repo)
        write(repo, "pkg/__init__.py", "VERSION = 1\n")
        p = plan(repo)
        # b, c, d, e import through `pkg`; a, f, g import nothing; test_b through b.
        assert p.targets == ["pkg/__init__.py", "pkg/b.py", "pkg/c.py", "pkg/d.py", "pkg/e.py", "tests/test_b.py"]

    def test_a_change_outside_the_include_set_rechecks_its_importers_but_not_itself(self, repo: Path) -> None:
        green_full_run(repo)
        write(repo, "scripts/experiments/expmod.py", "X = 2\n")
        p = plan(repo)
        assert p.changed == ["scripts/experiments/expmod.py"]
        assert p.targets == ["tests/test_exp.py"]

    def test_unchanged_tree_has_nothing_to_check(self, repo: Path) -> None:
        green_full_run(repo)
        p = plan(repo)
        assert p.mode == "scoped" and p.changed == [] and p.targets == []

    def test_only_changed_blobs_are_reparsed(self, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        green_full_run(repo)
        write(repo, "pkg/f.py", "F = 1\n")
        parsed: list[str] = []
        real = MOD.parse_imports

        def counting(text: str):
            parsed.append(text)
            return real(text)

        monkeypatch.setattr(MOD, "parse_imports", counting)
        assert plan(repo).targets == ["pkg/f.py"]
        assert parsed == ["F = 1\n"]


# ---------------------------------------------------------------------------
# baseline semantics
# ---------------------------------------------------------------------------


class TestBaseline:
    def test_no_baseline_means_a_full_run_that_then_writes_one(self, repo: Path) -> None:
        assert plan(repo).mode == "full"
        assert plan(repo).reason == "no baseline"
        rc, rec = run(repo, "--threads", "2")
        assert rc == 0
        assert rec.calls == [(["pyright", "--threads", "2"], repo)]
        data = json.loads(baseline_of(repo).read_text())
        assert data["format"] == MOD.BASELINE_FORMAT
        assert data["env"] == "env-A"
        assert set(data["manifest"]) == set(FILES)
        # Every source blob's imports are cached; the config files have none.
        source_hashes = {h for rel, h in data["manifest"].items() if rel.endswith((".py", ".pyi"))}
        assert set(data["imports"]) == source_hashes
        assert data["imports"][data["manifest"]["pkg/b.py"]] == [["pkg.a", 0, ["A"]]]

    def test_a_red_run_writes_no_baseline(self, repo: Path) -> None:
        rc, _ = run(repo, status=1)
        assert rc == 1
        assert not baseline_of(repo).exists()

    def test_a_red_scoped_run_leaves_the_baseline_alone(self, repo: Path) -> None:
        green_full_run(repo)
        before = baseline_of(repo).read_bytes()
        write(repo, "pkg/a.py", "A = 'oops'\n")
        rc, rec = run(repo, status=1)
        assert rc == 1
        assert rec.calls == [(["pyright", "pkg/a.py", "pkg/b.py", "tests/test_b.py"], repo)]
        assert baseline_of(repo).read_bytes() == before

    def test_a_green_scoped_run_advances_the_baseline(self, repo: Path) -> None:
        green_full_run(repo)
        write(repo, "pkg/a.py", "A = 2\n")
        rc, rec = run(repo)
        assert rc == 0
        assert rec.calls[0][0] == ["pyright", "pkg/a.py", "pkg/b.py", "tests/test_b.py"]
        p = plan(repo)
        assert p.changed == [] and p.targets == []

    def test_nothing_to_check_runs_nothing_and_stays_green(self, repo: Path) -> None:
        green_full_run(repo)
        rc, rec = run(repo)
        assert rc == 0
        assert rec.calls == []
        assert baseline_of(repo).is_file()

    def test_an_environment_change_forces_a_full_run(self, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        green_full_run(repo)
        monkeypatch.setattr(MOD, "environment_key", lambda: "env-B")
        p = plan(repo)
        assert p.mode == "full" and "environment" in p.reason

    def test_a_config_change_forces_a_full_run(self, repo: Path) -> None:
        green_full_run(repo)
        write(repo, "pyrightconfig.json", CONFIG + "\n// edited\n")
        p = plan(repo)
        assert p.mode == "full" and p.reason.startswith("pyrightconfig.json changed")

    def test_a_pyproject_change_forces_a_full_run(self, repo: Path) -> None:
        green_full_run(repo)
        write(repo, "pyproject.toml", FILES["pyproject.toml"] + "# edited\n")
        p = plan(repo)
        assert p.mode == "full" and p.reason.startswith("pyproject.toml changed")

    def test_a_closure_over_the_threshold_forces_a_full_run(self, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        assert MOD.FULL_THRESHOLD == 0.6
        green_full_run(repo)
        write(repo, "pkg/a.py", "A = 2\n")
        assert plan(repo).mode == "scoped"  # 3 of 11
        monkeypatch.setattr(MOD, "FULL_THRESHOLD", 0.2)
        p = plan(repo)
        assert p.mode == "full" and p.reason.startswith("3 of 11 checkable files affected")

    def test_fresh_env_var_and_full_flag_force_a_full_run(self, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        green_full_run(repo)
        rc, rec = run(repo, "--full")
        assert rc == 0 and rec.calls == [(["pyright"], repo)]
        monkeypatch.setenv("VTSEARCH_FRESH", "1")
        rc, rec = run(repo)
        assert rc == 0 and rec.calls == [(["pyright"], repo)]

    def test_dry_run_prints_the_plan_and_touches_nothing(self, repo: Path, capsys: pytest.CaptureFixture) -> None:
        green_full_run(repo)
        capsys.readouterr()  # the full run's own summary line
        before = baseline_of(repo).read_bytes()
        write(repo, "pkg/a.py", "A = 2\n")
        rc, rec = run(repo, "--dry-run", status=1)
        assert rc == 0 and rec.calls == []
        out = capsys.readouterr().out
        assert out.startswith("pyright (incremental): 1 changed, 3 to check of 11 checkable; baseline ")
        assert out.splitlines()[1:] == ["  pkg/a.py", "  pkg/b.py", "  tests/test_b.py"]
        assert baseline_of(repo).read_bytes() == before

    def test_full_run_summary_names_its_reason(self, repo: Path, capsys: pytest.CaptureFixture) -> None:
        run(repo, "--dry-run")
        assert capsys.readouterr().out.strip() == "pyright (full): no baseline"

    def test_a_malformed_baseline_falls_back_to_full(self, repo: Path, capsys: pytest.CaptureFixture) -> None:
        baseline_of(repo).parent.mkdir()
        baseline_of(repo).write_text("{not json")
        rc, rec = run(repo)
        assert rc == 0 and rec.calls == [(["pyright"], repo)]
        assert "running pyright in full" in capsys.readouterr().err
        # ...and the green full run replaced it with a sound one.
        assert json.loads(baseline_of(repo).read_text())["format"] == MOD.BASELINE_FORMAT

    def test_a_directory_that_is_not_a_repo_falls_back_to_full(self, tmp_path: Path, capsys) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        rc, rec = run(plain)
        assert rc == 0 and rec.calls == [(["pyright"], plain)]
        assert "running pyright in full" in capsys.readouterr().err
        assert not baseline_of(plain).exists()

    def test_an_explicit_baseline_path_is_honoured(self, repo: Path, tmp_path: Path) -> None:
        elsewhere = tmp_path / "b.json"
        rc, _ = run(repo, "--baseline", str(elsewhere))
        assert rc == 0 and elsewhere.is_file() and not baseline_of(repo).exists()

    def test_untracked_and_deleted_files_are_part_of_the_manifest(self, repo: Path) -> None:
        write(repo, "pkg/new_untracked.py", "N = 1\n")
        (repo / "pkg" / "g.py").unlink()
        paths = MOD.source_paths(repo)
        assert "pkg/new_untracked.py" in paths
        assert "pkg/g.py" not in paths


# ---------------------------------------------------------------------------
# the real tree
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tree():
    """The real repository's import graph, built once per worker (parsing ~2,000 files)."""
    cfg = MOD.load_config(REPO_ROOT)
    paths = [p for p in MOD.source_paths(REPO_ROOT) if p.endswith((".py", ".pyi"))]
    files = set(paths)
    specs = {rel: MOD.parse_imports((REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")) for rel in paths}
    graph = MOD.build_graph(files, lambda rel: specs[rel], cfg.roots)
    return cfg, files, specs, graph


@pytest.mark.xdist_group("pyright-incremental-real-tree")
class TestTheRealTree:
    def test_no_repo_module_is_ever_called_external(self, tree) -> None:
        """For every absolute import the resolver leaves external, no file for it exists under any root."""
        cfg, files, specs, _ = tree
        misses: list[str] = []
        for rel, file_specs in specs.items():
            for spec in file_specs:
                module, level, names = spec
                if level or not module:
                    continue
                if MOD.resolve(spec, rel, files, cfg.roots):
                    continue
                parts = module.split(".")
                for root in cfg.roots:
                    base = REPO_ROOT / root if root else REPO_ROOT
                    for cand in (
                        base.joinpath(*parts).with_suffix(".py"),
                        base.joinpath(*parts).with_suffix(".pyi"),
                        base.joinpath(*parts, "__init__.py"),
                        base.joinpath(*parts, "__init__.pyi"),
                    ):
                        if cand.is_file():
                            misses.append(f"{rel}: import {module} -> {cand.relative_to(REPO_ROOT)}")
        assert not misses, "\n".join(misses)

    def test_known_edges_are_present(self, tree) -> None:
        _, _, _, graph = tree
        assert "app.py" in graph["tests/conftest.py"]
        reg = graph["tests_lib/meta/test_calibration_dataset_registry.py"]
        assert "scripts/experiments/pile/pile_config.py" in reg
        assert "scripts/experiments/calibration/experiment_config.py" in reg
        assert "tests_shared/state_reset.py" in graph["tests_lib/conftest.py"]
        # stubPath: a change to the local transformers stub re-checks its importers.
        assert "typings/transformers/__init__.pyi" in graph["vtscore/config/processor_backend.py"]

    def test_a_core_module_reaches_most_of_the_tree(self, tree) -> None:
        cfg, files, _, graph = tree
        closure = MOD.reverse_closure(["vtscore/state/core.py"], (graph,))
        checkable = {rel for rel in files if MOD.is_checkable(rel, cfg)}
        assert len(closure & checkable) > 0.6 * len(checkable)
