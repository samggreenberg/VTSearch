"""The State of the App's two paths side by side (#4655).

``scripts/experiments/state_of_app/compare_paths.py`` puts the DINOv3 region
path beside the SigLIP binary path, from the two reports' committed CSVs.
Pinned here:

* **The lead is region minus binary,** at every point and in every column,
  with the reports' minus sign.
* **Two reports at different presets are refused**, rather than drawn as if
  they matched.
* **A committed report's table is what its CSVs give:** every Region Photo
  report with an "Against the binary path" section carries
  ``compare_paths.table`` of its own CSVs and those of the binary report it
  links, so a re-scored CSV cannot leave the table behind.

Meta-group: the subject is repo tooling under ``scripts/experiments/``.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SOTA = _ROOT / "scripts" / "experiments" / "state_of_app"
_REPORTS = _ROOT / "docs" / "experiments"
#: The heading a Region Photo report's comparison sits under.
SECTION = "### Against the binary path"


@pytest.fixture(scope="module")
def cp():
    pytest.importorskip("matplotlib")
    # The script imports its sibling `figures` by bare name; leave sys.path and
    # sys.modules as they were, so another test's by-path load is not shadowed.
    saved_path, saved_figures = list(sys.path), sys.modules.get("figures")
    try:
        spec = importlib.util.spec_from_file_location("_compare_paths", _SOTA / "compare_paths.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.path[:] = saved_path
        if saved_figures is None:
            sys.modules.pop("figures", None)
        else:
            sys.modules["figures"] = saved_figures


#: Planted returned sets: (precision, recall, F-beta, returned) per point, chosen so no two leads coincide.
BINARY = {"typed query": (0.50, 0.30, 0.45, 20), "25": (0.52, 0.31, 0.46, 21), "50": (0.55, 0.32, 0.48, 22),
          "100": (0.60, 0.34, 0.52, 23), "150": (0.62, 0.36, 0.55, 24), "after the check": (0.66, 0.35, 0.58, 22)}  # fmt: skip
REGION = {"typed query": (0.50, 0.30, 0.45, 20), "25": (0.49, 0.33, 0.43, 25), "50": (0.58, 0.37, 0.53, 27),
          "100": (0.68, 0.41, 0.61, 29), "150": (0.71, 0.44, 0.65, 30), "after the check": (0.77, 0.43, 0.69, 31)}  # fmt: skip


def _report(d: Path, points: dict, betas=(0.25, 1.0)) -> Path:
    d.mkdir()
    rows = [
        {"beta": b, "point": p, "precision": v[0], "recall": v[1], "fbeta": v[2], "returned, median": v[3], "runs": 9}
        for b in betas
        for p, v in points.items()
    ]
    pd.DataFrame(rows).to_csv(d / "precision_recall_path.csv", index=False)
    clicks = [
        {"beta": b, "t": t, "fbeta": points["150"][2], "precision": 0.5, "recall": 0.5, "runs": 9, "with_detector": 0}
        for b in betas
        for t in range(0, 151, 50)
    ]
    pd.DataFrame(clicks).to_csv(d / "objective_by_click.csv", index=False)
    return d


def test_the_lead_is_region_minus_binary_at_every_point(cp, tmp_path) -> None:
    _clicks, path = cp.load(_report(tmp_path / "b", BINARY), _report(tmp_path / "r", REGION))
    rows = [line.split(" | ") for line in cp.table(path).splitlines() if "region − binary" in line]
    assert len(rows) == 2, "one lead row per preset"
    lead = [cell.strip(" |") for cell in rows[0][2:]]
    want = [f"{REGION[p][2] - BINARY[p][2]:+.2f}" for p in cp.POINTS]
    end_b, end_r = BINARY["after the check"], REGION["after the check"]
    want += [f"{end_r[0] - end_b[0]:+.2f}", f"{end_r[1] - end_b[1]:+.2f}", f"{end_r[3] - end_b[3]:+.0f}"]
    assert lead == [w.replace("-", "−") for w in want]
    assert "−0.03" in lead, "a negative lead prints the reports' minus sign"


def test_the_figures_are_written(cp, tmp_path) -> None:
    clicks, path = cp.load(_report(tmp_path / "b", BINARY), _report(tmp_path / "r", REGION))
    assert cp.figure_objective(clicks, path, tmp_path).stat().st_size > 0
    assert cp.figure_path(path, tmp_path).stat().st_size > 0


def test_reports_at_different_presets_are_refused(cp, tmp_path) -> None:
    binary = _report(tmp_path / "b", BINARY, betas=(0.25, 1.0))
    region = _report(tmp_path / "r", REGION, betas=(0.25, 4.0))
    with pytest.raises(SystemExit, match="different presets"):
        cp.load(binary, region)


def _compared_reports() -> list[tuple[Path, Path]]:
    pairs = []
    for report in sorted(_REPORTS.glob("*-state-of-the-app-region-photo/REPORT.md")):
        text = report.read_text(encoding="utf-8")
        if SECTION not in text:
            continue
        section = text.split(SECTION, 1)[1].split("\n## ", 1)[0].split("\n### ", 1)[0]
        link = re.search(r"\]\(\.\./([^)/]*-state-of-the-app-binary-photo)/REPORT\.md\)", section)
        assert link, f"{report}: the section names no Binary Photo report to compare against"
        pairs.append((report.parent, _REPORTS / link.group(1)))
    return pairs


def test_a_committed_reports_table_is_what_its_csvs_give(cp) -> None:
    pairs = _compared_reports()
    assert pairs, f"no Region Photo report carries {SECTION!r}"
    for region, binary in pairs:
        _clicks, path = cp.load(binary, region)
        assert cp.table(path) in (region / "REPORT.md").read_text(encoding="utf-8"), (
            f"{region.name}: the table under {SECTION!r} is not compare_paths.py's over the committed CSVs; "
            f"re-run it (--binary {binary.relative_to(_ROOT)} --region {region.relative_to(_ROOT)})"
        )
