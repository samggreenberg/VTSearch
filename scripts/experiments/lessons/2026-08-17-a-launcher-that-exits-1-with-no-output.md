# 2026-08-17 — a launcher that exits 1 with no output submitted nothing (#3143)

**What broke.** `launch_precision.sh size fp32_l40s` printed *nothing* and exited
1. No preflight verdict, no `SUBMIT FAILED`, no job id — and no job. Two runs in
a row looked identical to a launch that had merely been quiet.

The cause was one line in the launcher's own preflight:

```sh
set -euo pipefail
...
zero=$(find "$STUDY/piles" -name '*.pkl' -size 0 2>/dev/null | wc -l)
```

On the very first run `$STUDY/piles` does not exist yet, so `find` exits 1.
`pipefail` propagates that to the pipeline, the pipeline is the whole command
substitution, so the *assignment* fails — and `set -e` takes the script down at
that point. `2>/dev/null` hid the only clue. The check that existed to stop a
silent skip became the thing that silently skipped everything.

**Cost.** ~15 minutes and two blind re-runs, plus one `bash -x` to find it. Cheap
this time only because it happened at `size` — the deliberate "time one arm before
committing to all six" step — rather than at `arms`, where the same launcher would
have submitted zero of six arms and been believed for as long as nobody checked
`squeue`.

**The general form.** `set -e` + `pipefail` + `count=$(find … | wc -l)` over a
path that does not exist yet is an abort, not a zero. The pattern is invisible
because every part of it is good practice on its own, and because the failing
path is the *first* run — the one where there is nothing to compare against. It
sits in the same family as "an empty job id is not a launch" (#2897) and "a
`--gres=none` refusal is not a launch" (#2905), one step earlier: there, a
submission failed and said nothing; here, the submission was never reached.

The rule that generalises: **a launcher's exit status is not its report.** If a
launcher can exit non-zero without naming what it refused, that is a bug in the
launcher regardless of what caused it.

**Prevented, twice over.** The specific bug is fixed — `mkdir -p` the dir first
and `|| true` on the counting pipeline, in `launch_precision.sh` (three sites) and
`launch_bench.sh` (one). But fixing one line would not stop the next `set -e`
abort anywhere else in a launcher, so both launchers also carry:

```sh
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR
```

Verified to fire on exactly the original construct. That is the part that
generalises: whatever aborts a launcher next, it now names the line and says that
nothing was submitted, so silence can never again be mistaken for a quiet pass.

**Still only advice** for the rest of the tree, though nothing else currently has
it. Every other counting script here (`preflight.sh`, `status_folds_2897.sh`) uses
`set -uo pipefail` *without* `-e`, which is why the same lines are harmless there
— worth knowing before someone "tightens" one of them by adding `-e`. Re-run the
scan before adding a launcher:

```sh
for f in $(grep -rl pipefail scripts/experiments --include='*.sh'); do
  awk -v F="$f" '/=\$\(/ && /wc -l|grep -c/ && !/\|\| true/ && !/\|\| echo/ {print F":"FNR": "$0}' "$f"
done
```

**Second, smaller lesson from the same hour.** Do not point `suite.sbatch` at the
worktree you launch from: it does `git checkout --detach`, and `preflight.sh`
check 4 then compares `HEAD` against `origin/HEAD` (i.e. `main`) and fails with
"worktree is not at origin/HEAD". Re-attach with `git checkout -B <branch>
origin/<branch>`, or give the test suite its own worktree.
