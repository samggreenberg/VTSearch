"""Run the vulture dead-code audit, and gate the whitelist against rot.

This script is the **single source of truth** for the vulture invocation.
It used to be a copy-pasted command line living in two places at once
(``.vulture-whitelist.py``'s docstring and ``docs/RELEASE.md``), which is
how it silently drifted: both copies scanned ``vtsearch/ app.py tests/``
and nothing else, so ``vtscore/`` -- 100k lines and the tier that external
extensions actually import -- was never looked at.

Two modes:

``python scripts/vulture-audit.py``
    The pre-release audit (``docs/RELEASE.md`` step 1). Prints every
    finding and exits 3 if there are any. Not a ``run-tests.sh`` gate: a
    vulture hit on a public ``vtscore`` name is not evidence of anything
    (see the module note in ``.vulture-whitelist.py``), so a human
    triages the list rather than a script rejecting the push.

``python scripts/vulture-audit.py --check-whitelist``
    The gate that *is* wired into ``run-tests.sh``. It asserts only what
    can be asserted mechanically: that every entry in
    ``.vulture-whitelist.py`` still suppresses a real finding. An entry
    that suppresses nothing is unfalsifiable -- it makes a claim about the
    codebase that nothing can ever check, and it will quietly outlive the
    symbol it was written for.

Both modes share one vulture pass. The whitelist is applied *by this
script* rather than by handing the file to vulture as an extra source
path, because vulture's own mechanism (a bare ``name`` expression counts
as a use) is one-way: it can suppress a finding but can never tell you
that it suppressed nothing.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WHITELIST = REPO_ROOT / ".vulture-whitelist.py"

# Every tier that defines or *consumes* first-party Python. Consumers matter
# as much as definitions: `scripts/` is scanned because the experiment
# drivers import `vtscore`, and scanning them resolves ten `vtscore` findings
# that would otherwise read as dead.
SCAN_PATHS = [
    "vtsearch/",
    "app.py",
    "tests/",
    "vtscore/",
    "tests_lib/",
    "scripts/",
]

# Framework-managed declaration files: every field assignment in a marshmallow
# Schema or a pydantic BaseModel looks unused because both frameworks collect
# fields via metaclass at class-creation time.
EXCLUDE = [
    "*/vtsearch/schemas/*",
    "*/vtsearch/settings_models.py",
]

# Scanned (so their imports count as uses) but not *reported on*. One-off
# experiment drivers are archival records of runs that already happened; an
# unused constant in one is not a maintenance liability, and reporting them
# buries the findings that are.
REPORT_EXCLUDE_PREFIXES = ("scripts/experiments/",)

IGNORE_DECORATORS = [
    "@*.route",
    "@*.before_request",
    "@*.after_request",
    "@*.errorhandler",
    "@*.teardown_request",
    "@*.context_processor",
    "@bp.*",
    "@app.*",
    "@pytest.fixture",
    "@pytest.mark.*",
    "@fixture",
    "@*.fixture",
]

IGNORE_NAMES = [
    "Meta",
    "model_config",
    "_keys_to_ignore_on_load_unexpected",
    "test_*",
    "Test*",
    "setup_method",
    "teardown_method",
    "setup_class",
    "teardown_class",
    "pytest_*",
    "pytestmark",
    "__enter__",
    "__exit__",
    "__package__",
]

MIN_CONFIDENCE = 60


def whitelist_names() -> list[str]:
    """Parse ``.vulture-whitelist.py`` into the list of names it whitelists.

    Entries are bare name expressions at module level (``EXPORTER  # noqa``).
    Parsing with ``ast`` rather than a regex means the module docstring and
    the section comments can never be mistaken for entries.
    """
    tree = ast.parse(WHITELIST.read_text(), filename=str(WHITELIST))
    return [
        node.value.id
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Name)
    ]


def scavenge():
    """Run one vulture pass over the whole tree, whitelist not applied."""
    from vulture.core import Vulture

    vult = Vulture(ignore_names=IGNORE_NAMES, ignore_decorators=IGNORE_DECORATORS)
    vult.scavenge([str(REPO_ROOT / p) for p in SCAN_PATHS], exclude=EXCLUDE)
    return vult.get_unused_code(min_confidence=MIN_CONFIDENCE)


def relpath(item) -> str:
    try:
        return str(Path(item.filename).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(item.filename)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-whitelist",
        action="store_true",
        help="Gate mode: fail if any whitelist entry suppresses nothing.",
    )
    args = parser.parse_args()

    try:
        items = scavenge()
    except ImportError:
        print("vulture is not installed; skipping (pip install -e '.[dev]').")
        return 0

    entries = whitelist_names()
    flagged = {item.name for item in items}

    if args.check_whitelist:
        # Measured against every flagged name, including those inside
        # REPORT_EXCLUDE_PREFIXES: an entry covering a finding we choose not
        # to print is still doing work, and must not be reported as rot.
        inert = [name for name in entries if name not in flagged]
        if inert:
            print(f"{WHITELIST.name}: {len(inert)} of {len(entries)} entries suppress nothing:")
            for name in inert:
                print(f"  {name}")
            print()
            print("Each of these makes a claim about the codebase that nothing can check.")
            print("Either the symbol is gone, or it acquired a real caller that vulture can")
            print("now see. Delete the entry (and its comment) from .vulture-whitelist.py.")
            return 1
        print(f"vulture whitelist: {len(entries)} entries, all load-bearing")
        return 0

    findings = [
        item
        for item in items
        if item.name not in entries and not relpath(item).startswith(REPORT_EXCLUDE_PREFIXES)
    ]
    for item in sorted(findings, key=lambda i: (relpath(i), i.first_lineno)):
        print(
            f"{relpath(item)}:{item.first_lineno}: unused {item.typ} "
            f"'{item.name}' ({item.confidence}% confidence)"
        )
    if findings:
        print()
        print(f"{len(findings)} finding(s). Triage per docs/RELEASE.md step 1.")
        return 3
    print("vulture: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
