"""The `npm audit` gate's waiver table: `scripts/npm-audit-gate.py` (#4447).

A waiver is a hole in a security gate, so what needs pinning is that the hole
stays exactly one advisory wide and closes on its own: an unwaived advisory
still fails, a waiver stops applying once upstream narrows the vulnerable range,
a waiver that suppresses nothing fails as stale, and a report that never ran
fails rather than reading as clean.

Reports are synthesized in the `npm audit --json` v2 shape, copied from a real
run, so the tests never touch the network or `frontend/node_modules`.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "npm-audit-gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("_npm_audit_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()

WAIVER = gate.Waiver(ghsa="GHSA-aaaa-bbbb-cccc", package="leafpkg", range="<=4.2.0", reason="test")


def advisory(package: str, ghsa: str, rng: str, severity: str = "high") -> dict[str, Any]:
    return {
        "source": 1,
        "name": package,
        "dependency": package,
        "title": f"{package} is vulnerable",
        "url": f"https://github.com/advisories/{ghsa}",
        "severity": severity,
        "cwe": [],
        "cvss": {},
        "range": rng,
    }


def report(*leaves: dict[str, Any]) -> dict[str, Any]:
    """A report with each advisory on a leaf package, plus a parent reached only by name."""
    vulns: dict[str, Any] = {}
    for leaf in leaves:
        vulns[leaf["name"]] = {"name": leaf["name"], "severity": leaf["severity"], "via": [leaf], "range": "*"}
        vulns[f"{leaf['name']}-parent"] = {
            "name": f"{leaf['name']}-parent",
            "severity": leaf["severity"],
            "via": [leaf["name"]],
            "range": "*",
        }
    return {"auditReportVersion": 2, "vulnerabilities": vulns, "metadata": {}}


def verdict(rep: dict[str, Any], waivers: tuple[Any, ...] = (WAIVER,)) -> Any:
    return gate.evaluate(gate.advisories(rep), waivers)


class TestWaivers:
    def test_a_clean_report_passes(self) -> None:
        v = verdict(report(), waivers=())
        assert v.ok

    def test_an_unwaived_advisory_fails(self) -> None:
        v = verdict(report(advisory("piscina", "GHSA-67c8-pqhq-4rmx", "5.0.0 - 5.3.1", "critical")), waivers=())
        assert not v.ok
        assert [a.package for a in v.unwaived] == ["piscina"]

    def test_parents_reached_by_name_are_not_extra_advisories(self) -> None:
        found = gate.advisories(report(advisory("leafpkg", "GHSA-aaaa-bbbb-cccc", "<=4.2.0")))
        assert [(a.ghsa, a.package) for a in found] == [("GHSA-aaaa-bbbb-cccc", "leafpkg")]

    def test_a_waived_advisory_passes(self) -> None:
        v = verdict(report(advisory("leafpkg", "GHSA-aaaa-bbbb-cccc", "<=4.2.0")))
        assert v.ok
        assert [a.ghsa for a in v.waived] == ["GHSA-aaaa-bbbb-cccc"]

    def test_a_waiver_covers_only_its_own_advisory(self) -> None:
        v = verdict(
            report(
                advisory("leafpkg", "GHSA-aaaa-bbbb-cccc", "<=4.2.0"),
                advisory("piscina", "GHSA-67c8-pqhq-4rmx", "5.0.0 - 5.3.1", "critical"),
            )
        )
        assert not v.ok
        assert [a.package for a in v.unwaived] == ["piscina"]

    def test_a_waiver_lapses_when_upstream_narrows_the_range(self) -> None:
        v = verdict(report(advisory("leafpkg", "GHSA-aaaa-bbbb-cccc", "<4.2.1")))
        assert not v.ok
        assert [(w.ghsa, a.range) for w, a in v.moved] == [("GHSA-aaaa-bbbb-cccc", "<4.2.1")]
        assert "patched release likely exists" in gate.render(v)

    def test_a_waiver_that_suppresses_nothing_is_stale(self) -> None:
        v = verdict(report())
        assert not v.ok
        assert v.stale == [WAIVER]
        assert "Stale waiver" in gate.render(v)


class TestTheShippedTable:
    def test_every_waiver_names_a_ghsa_and_a_reason(self) -> None:
        for w in gate.WAIVERS:
            assert w.ghsa.startswith("GHSA-"), w
            assert w.package and w.range and w.reason.strip(), w


class TestTheCommandLine:
    def run(self, tmp_path: Path, rep: dict[str, Any]) -> subprocess.CompletedProcess[str]:
        path = tmp_path / "audit.json"
        path.write_text(json.dumps(rep))
        return subprocess.run(  # noqa: S603  # this interpreter + a repo-local gate path
            [sys.executable, str(SCRIPT), "--report", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_an_unwaived_advisory_exits_nonzero(self, tmp_path: Path) -> None:
        out = self.run(tmp_path, report(advisory("piscina", "GHSA-67c8-pqhq-4rmx", "5.0.0 - 5.3.1", "critical")))
        assert out.returncode == 1
        assert "piscina" in out.stderr

    def test_an_npm_error_is_not_a_clean_audit(self, tmp_path: Path) -> None:
        out = self.run(tmp_path, {"message": "request to registry failed", "error": {"code": "ENOTFOUND"}})
        assert out.returncode == 1
        assert "nothing was audited" in out.stderr

    def test_a_report_of_the_wrong_shape_is_not_a_clean_audit(self, tmp_path: Path) -> None:
        out = self.run(tmp_path, {"advisories": {}, "metadata": {}})
        assert out.returncode == 1
        assert "nothing was audited" in out.stderr
