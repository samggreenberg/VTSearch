# 2026-08-29 — a fresh worktree ran 49 fewer tests and still said ALL TESTS PASSED (#2877)

**Study:** the #2877 ship PR (`ACQUISITION_INCLUSION_OFFSET = -3`). **Cost:**
~25 min of re-runs. Caught by comparing pass counts against the previous run,
not by anything the suite said.

## What happened

`./run-tests.sh` on the ship branch, in a **fresh worktree**, reported:

```
ALL 9260 TESTS PASSED (56 skipped, 2 xfailed, total: 9316)
RUN PASSED (all gates green)
```

The study branch, an hour earlier, had reported `9310 passed, 6 skipped` — the
same 9316 total. **50 tests had stopped running, and the suite's headline was
still "ALL TESTS PASSED".**

The skip reasons, once enumerated with `pytest -rs`:

```
49  cd frontend && npm install && npm run build:prod
 3  only meaningful when VTSEARCH_BLOCK_FLASK is set
 1  Demo source directory not present on disk
 1  Angular build not present (run 'npm run build:prod' in frontend/)
```

A fresh `git worktree add` has no `frontend/node_modules` and no built Angular
output, so every test that needs the built frontend skips itself. `run-tests.sh`
still builds and typechecks the frontend (it reported `Frontend build OK`), which
is why nothing looked wrong.

Nothing was broken and the change was clean — but the run that "verified" a
change to a shipped constant had silently excluded the entire frontend-dependent
surface, and would have excluded it just as silently for a change that did break
something there.

## The wrong diagnosis, and why it was tempting

The two runs had also landed on **different nodes** (`rack2n10` 187G vs
`rack4n03` 564G), and `tests_lib/gpu/` holds 34 test functions gated on
`torch.cuda.is_available()`. That is the right order of magnitude, it explains a
node-dependent skip count, and it is wrong.

It was settled by **re-running the study branch pinned to the ship run's node**
(`sbatch --nodelist=rack2n10`): 9310 passed, 6 skipped — identical to the other
node. The node explained nothing.

**A plausible mechanism of the right size is not evidence.** The cheap control
here was one `--nodelist` flag.

## Still only advice

**A skip count is part of a test result, and only means anything against a
baseline.** `N passed` alone cannot distinguish "everything ran and passed" from
"a chunk stopped being collected". Before trusting a green suite on a branch,
compare its *passed* and *skipped* counts against the last known-good run of the
same tree; if they moved, enumerate with `pytest -rs` before reading the
headline.

**Verify a ship in a worktree that has the frontend built** — the same one the
previous green run used — or link `node_modules` and build first. The existing
`vts-*-tests` worktrees exist for this; a fresh worktree is the wrong place to
verify anything that could touch the app surface.

Not made a preflight check because `preflight.sh` gates *experiment launches*,
not test runs, and the natural home for this — comparing against a stored
baseline count — needs somewhere to keep the baseline.
