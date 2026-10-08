#!/usr/bin/env python3
"""`npm audit` for the frontend tree, with an explicit table of waived advisories.

The gate audits the whole tree, dev dependencies included (see the comment on
``_lane_npm_audit`` in `run-tests.sh` for why it stays that wide). `npm audit`
itself has no way to set one advisory aside, so an advisory with **no patched
release anywhere** turns the gate red on every branch and keeps it red, with
nothing a contributor can do about it (#4447).

``WAIVERS`` is that escape hatch, the counterpart of ``PIP_AUDIT_IGNORE`` in
`run-tests.sh`. Prefer a lockfile refresh or an ``overrides`` pin in
`frontend/package.json` whenever a fixed version exists; a waiver is only for an
advisory npm reports against *every* published version. Each waiver is pinned
to the advisory's package and vulnerable range as npm reports them, so it
cannot outlive its reason:

* Upstream ships a fix -> the advisory's range narrows (``<=4.2.0`` becomes
  ``<4.2.1``) -> the waiver stops matching and the gate fails, saying to refresh
  the lockfile.
* The vulnerable package leaves the tree -> the advisory stops being reported
  -> the waiver is stale and the gate fails, saying to delete it. A waiver that
  suppresses nothing would otherwise be unfalsifiable, which is how the vulture
  whitelist rotted.

Every other advisory fails the gate exactly as bare `npm audit` would, and so
does a report that cannot be read (offline, registry error): an audit that did
not run must not read as an audit that found nothing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


@dataclass(frozen=True)
class Waiver:
    ghsa: str
    package: str
    #: The vulnerable range exactly as `npm audit --json` reports it.
    range: str
    reason: str


WAIVERS: tuple[Waiver, ...] = (
    Waiver(
        ghsa="GHSA-ch52-4w7c-c8xp",
        package="http-cache-semantics",
        range="<=4.2.0",
        reason=(
            "No patched release (4.2.0 is the latest). The exploit needs a shared, "
            "multi-user HTTP cache; here the package sits under @angular/cli -> pacote "
            "-> make-fetch-happen, the CLI's private on-disk package cache for "
            "`ng add` / `ng update`. Nothing in the built app or the dev server uses it."
        ),
    ),
)


@dataclass(frozen=True)
class Advisory:
    ghsa: str
    package: str
    range: str
    severity: str
    title: str
    url: str


@dataclass
class Verdict:
    unwaived: list[Advisory]
    waived: list[Advisory]
    #: Waivers whose advisory is reported, but no longer over the pinned range.
    moved: list[tuple[Waiver, Advisory]]
    #: Waivers whose advisory is not reported at all.
    stale: list[Waiver]

    @property
    def ok(self) -> bool:
        return not (self.unwaived or self.moved or self.stale)


class UnreadableReport(Exception):
    pass


def advisories(report: dict[str, Any]) -> list[Advisory]:
    """Every distinct advisory in an `npm audit --json` (v2) report.

    A vulnerable package's ``via`` mixes advisory objects with bare package
    names; a name means "vulnerable because it depends on that package", whose
    own entry carries the advisory. So the advisories are exactly the objects,
    and the names add nothing that a waiver could need to see.
    """
    if "error" in report:
        raise UnreadableReport(json.dumps(report["error"], indent=2))
    if report.get("auditReportVersion") != 2 or not isinstance(report.get("vulnerabilities"), dict):
        raise UnreadableReport("not an `npm audit --json` v2 report (no auditReportVersion 2 / vulnerabilities)")
    seen: dict[tuple[str, str, str], Advisory] = {}
    for vuln in report["vulnerabilities"].values():
        for via in vuln.get("via", []):
            if not isinstance(via, dict):
                continue
            url = str(via.get("url", ""))
            adv = Advisory(
                ghsa=url.rstrip("/").rsplit("/", 1)[-1] or str(via.get("source", "?")),
                package=str(via.get("name", "?")),
                range=str(via.get("range", "?")),
                severity=str(via.get("severity", "?")),
                title=str(via.get("title", "")),
                url=url,
            )
            seen.setdefault((adv.ghsa, adv.package, adv.range), adv)
    return list(seen.values())


def evaluate(found: list[Advisory], waivers: tuple[Waiver, ...] = WAIVERS) -> Verdict:
    by_key = {(w.ghsa, w.package): w for w in waivers}
    verdict = Verdict(unwaived=[], waived=[], moved=[], stale=[])
    matched: set[tuple[str, str]] = set()
    for adv in found:
        waiver = by_key.get((adv.ghsa, adv.package))
        if waiver is None:
            verdict.unwaived.append(adv)
            continue
        matched.add((waiver.ghsa, waiver.package))
        if adv.range == waiver.range:
            verdict.waived.append(adv)
        else:
            verdict.moved.append((waiver, adv))
    verdict.stale = [w for w in waivers if (w.ghsa, w.package) not in matched]
    return verdict


def render(verdict: Verdict) -> str:
    lines: list[str] = []
    for adv in verdict.unwaived:
        lines.append(f"  {adv.severity:<8} {adv.package} {adv.range}  {adv.title}")
        lines.append(f"           {adv.url}")
    if verdict.unwaived:
        lines[:0] = [f"{len(verdict.unwaived)} unwaived advisor{'y' if len(verdict.unwaived) == 1 else 'ies'}:"]
        lines.append(
            "Run `cd frontend && npm audit` for the dependency chains. Fix with a lockfile refresh or an\n"
            "`overrides` pin in frontend/package.json; waive in scripts/npm-audit-gate.py only when no\n"
            "published version is patched."
        )
    for waiver, adv in verdict.moved:
        lines.append(
            f"Waiver for {waiver.ghsa} ({waiver.package}) pins range {waiver.range!r}, but npm now reports "
            f"{adv.range!r}: a patched release likely exists. Refresh frontend/package-lock.json (or pin it "
            f"in `overrides`), then delete the waiver."
        )
    for waiver in verdict.stale:
        lines.append(
            f"Stale waiver: {waiver.ghsa} ({waiver.package} {waiver.range}) is no longer reported. "
            f"Delete it from WAIVERS in scripts/npm-audit-gate.py."
        )
    if verdict.ok:
        lines.append("npm audit: no unwaived advisories.")
    for adv in verdict.waived:
        lines.append(f"  waived: {adv.ghsa} {adv.package} {adv.range} ({adv.severity})")
    return "\n".join(lines)


def run_npm_audit(frontend: Path) -> dict[str, Any]:
    out = subprocess.run(  # noqa: S603  # fixed argv, no shell
        ["npm", "audit", "--json"],  # noqa: S607  # npm resolves on PATH, as for the build gate
        cwd=frontend,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError as exc:
        raise UnreadableReport(f"npm audit exited {out.returncode} without JSON:\n{out.stderr.strip()}") from exc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frontend", type=Path, default=FRONTEND)
    ap.add_argument("--report", type=Path, help="read a saved `npm audit --json` report instead of running npm")
    args = ap.parse_args(argv)

    try:
        report = json.loads(args.report.read_text()) if args.report else run_npm_audit(args.frontend)
        verdict = evaluate(advisories(report))
    except UnreadableReport as exc:
        print(f"npm audit report unreadable, so nothing was audited:\n{exc}", file=sys.stderr)
        return 1
    print(render(verdict), file=sys.stdout if verdict.ok else sys.stderr)
    return 0 if verdict.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
