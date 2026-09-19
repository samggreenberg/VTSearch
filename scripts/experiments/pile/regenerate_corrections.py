#!/usr/bin/env python3
"""Regenerate `corrections.json` by running the whole chain, not one step of it (#4007).

**This exists because "run the recipe" meant step 2 for weeks.**
`verdicts_to_corrections.py` is named after the output and its docstring opens on
the #3732 near-loss, so it reads as the recipe; it is one of three steps and
covers 872 of the live file's 4,709 rows. A regeneration that looked authoritative
therefore came up 3,837 rows short, and the only thing standing between that
result and the loss of most of the file was
:func:`~pilebuild.corrections.dropped_rows` refusing to write it (#4003, #4005).
The composition lived in a *different* file's usage block, where nobody running
the obvious script would look. Now it is a command:

    python regenerate_corrections.py                 # to a scratch path, then diff
    python regenerate_corrections.py --check         # diff only, exit 1 on any drift

The three steps, in order, each keeping its own CLI for the times one is wanted
alone:

1. ``pass_verdicts.py`` turns the #3926/#3940 per-class pass into verdicts —
   **3,837 rows**, the bulk of the file, and the step whose absence is invisible
   because the other two still produce a plausible file;
2. ``verdicts_to_corrections.py`` merges the #3156/#3588 campaigns with that pass
   — **791 rows** of its own;
3. ``apply_recheck.py`` re-answers what a later ruling re-asked — **81 rows**
   revisited, 21 retired and 60 stamped with the rule in force.

**The default output is a scratch path and never the live file.** Writing the
file the build reads is `--write-live`, spelled out, because this script's whole
subject is a regeneration that silently meant less than it looked like. Even then
the live write is `verdicts_to_corrections.py`'s own, with its `dropped_rows`
guard intact: **never reach for ``--allow-loss``**, which from a partial chain
would destroy 3,837 rows of human work.

**Judge a run by the diff, never by whether it ran.** `--check` compares
row-by-row against the committed `human_record/PILE__corrections.json` and reports
only-live, only-new and differing-shared separately, because a count that matches
is not the same as rows that match: 81 of the shared rows differed the last time
this mattered, and a count would have called that clean.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pile_config as pc  # noqa: E402
from pilebuild.corrections import write_json_locked  # noqa: E402
from verdicts_to_corrections import DEFAULT_VERDICTS  # noqa: E402

#: The committed copy of what the live file holds, and what a regeneration is
#: judged against: it is content-identical to `{PILE}/corrections.json` and it is
#: in the repository, so the comparison needs no scratch filesystem.
COMMITTED = HERE / "human_record" / "PILE__corrections.json"


def log(msg: str) -> None:
    print(f"[regen] {msg}", flush=True)


def _run(args: list[str], quiet: bool) -> None:
    """One step, with its own CLI, failing loudly rather than leaving a partial file."""
    log(f"$ {' '.join(Path(a).name if a.endswith('.py') else a for a in args)}")
    proc = subprocess.run(  # noqa: S603
        [sys.executable, *args],
        cwd=HERE,
        capture_output=quiet,
        text=True,
    )
    if proc.returncode != 0:
        if quiet and proc.stdout:
            print(proc.stdout)
        if quiet and proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"step failed ({Path(args[0]).name}, exit {proc.returncode})")


def regenerate(out: Path, workdir: Path, quiet: bool = False) -> list[dict]:
    """Run all three steps into *workdir* and return the rows written to *out*."""
    pass_json = workdir / "pass.json"
    _run(["pass_verdicts.py", "--out", str(pass_json)], quiet)
    _run(
        [
            "verdicts_to_corrections.py",
            "--verdicts",
            ",".join([*(str(p) for p in DEFAULT_VERDICTS), str(pass_json)]),
            "--out",
            str(out),
        ],
        quiet,
    )
    _run(["apply_recheck.py", "--corrections", str(out)], quiet)
    return json.loads(out.read_text())


def compare(new: list[dict], reference: list[dict]) -> dict[str, list]:
    """Row-by-row, keyed on the pair a row is about — never a count (#4007)."""
    kn = {(r["image_id"], r["class"]): r for r in new}
    kr = {(r["image_id"], r["class"]): r for r in reference}
    return {
        "only_new": sorted(kn.keys() - kr.keys()),
        "only_reference": sorted(kr.keys() - kn.keys()),
        "differing": sorted(k for k in kn.keys() & kr.keys() if kn[k] != kr[k]),
    }


def report(diff: dict[str, list], new: list[dict], reference: list[dict]) -> int:
    log(f"regenerated {len(new)} rows, committed copy has {len(reference)}")
    for name, rows in diff.items():
        log(f"  {name:<16}{len(rows):>6}")
    if not any(diff.values()) and len(new) == len(reference):
        log("identical: every row matches the committed copy")
        return 0
    for name, rows in diff.items():
        for key in rows[:10]:
            log(f"    {name}: {key}")
        if len(rows) > 10:
            log(f"    {name}: ... and {len(rows) - 10} more")
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="where to write (default: a temporary directory, discarded)")
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare against the committed copy and exit 1 on any drift",
    )
    ap.add_argument(
        "--write-live",
        action="store_true",
        help=f"also copy the result over {pc.PILE / 'corrections.json'} (spelled out on purpose)",
    )
    ap.add_argument("--quiet", action="store_true", help="hide each step's own output")
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="regen-corrections-") as tmp:
        workdir = Path(tmp)
        out = args.out or workdir / "corrections.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        new = regenerate(out, workdir, quiet=args.quiet)
        reference = json.loads(COMMITTED.read_text())
        status = report(compare(new, reference), new, reference)
        if args.check:
            return status
        if args.write_live:
            live = pc.PILE / "corrections.json"
            if status:
                raise SystemExit(f"refusing --write-live: the regeneration does not match {COMMITTED.name}")
            # Locked and atomic, like every other writer of this file: it is
            # shared by every session on this pile and has no history (#3729).
            write_json_locked(live, new)
            log(f"wrote {len(new)} rows to {live}")
        elif not args.out:
            log("scratch run: nothing kept (pass --out to keep it, --write-live to install it)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
