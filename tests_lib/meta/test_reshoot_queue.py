"""The reshoot queue's half of `scripts/screenshots/wiring-check.py`.

A GUI-changing session queues the screenshots it moved as one file under
`docs/reshoot-queue/` instead of re-rendering them, and the release run drains
the queue. The gate is what stops an entry from naming a shot that no longer
exists, so pin both halves: it reports the malformed shapes, and it passes the
ones the directory's README tells a session to write.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "screenshots" / "wiring-check.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("_wiring_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()
SHOTS = set(gate.manifest_ids())
GROUPS = gate.slide_groups()


def errors_for(tmp_path: Path, **files: str) -> list[str]:
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return gate.queue_errors(gate.queue_entries(tmp_path), SHOTS, GROUPS)


def test_slide_groups_are_the_shooters_command_line_groups():
    assert {"steps", "make-detector", "train-loop", "find", "region-voting"} <= GROUPS


def test_a_well_formed_entry_passes(tmp_path):
    entry = "# #1: a change\n\n- `new-detector` — moved\n- `slides:steps` — moved too\n"
    assert errors_for(tmp_path, **{"1-a-change.md": entry}) == []


def test_the_readme_is_not_an_entry(tmp_path):
    readme = "# Queue\n\n```markdown\n- `no-such-shot` — an example\n```\n"
    assert errors_for(tmp_path, **{"README.md": readme}) == []


def test_the_real_readme_example_would_pass():
    readme = (REPO_ROOT / "docs" / "reshoot-queue" / "README.md").read_text(encoding="utf-8")
    fence = readme.split("```markdown\n", 1)[1].split("```", 1)[0]
    example = gate.QUEUE_ITEM_RE.findall(fence)
    assert example, "the README should show an example entry"
    assert gate.queue_errors({Path("example.md"): example}, SHOTS, GROUPS) == []


def test_an_unknown_shot_is_reported(tmp_path):
    (error,) = errors_for(tmp_path, **{"2-x.md": "- `no-such-shot` — moved\n"})
    assert "'no-such-shot'" in error and "manifest" in error


def test_an_unknown_slide_group_is_reported(tmp_path):
    (error,) = errors_for(tmp_path, **{"3-x.md": "- `slides:nope` — moved\n"})
    assert "'slides:nope'" in error and "shoot-ui-figs.mjs" in error


def test_an_entry_naming_nothing_is_reported(tmp_path):
    (error,) = errors_for(tmp_path, **{"4-x.md": "# a change\n\nthe dashboard moved\n"})
    assert "names no shot" in error


def test_a_missing_queue_directory_queues_nothing(tmp_path):
    assert gate.queue_entries(tmp_path / "absent") == {}
