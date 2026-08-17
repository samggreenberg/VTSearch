# 2026-08-07 — a fresh worktree ran the *other* worktree's code (#2846)

**What happened.** The #2846 Grid re-measure got a fresh worktree off `dev`, as
every study should. `source gridenv.sh` succeeded, `PYTHONPATH` pointed at it,
and the first command — `selftest_analyze_cut.py` — died on
`cannot import name '_CUT_DIAGNOSTIC_COLUMNS' from
'/exp/sgreenberg/projects/vts-calib/vtscore/eval/voting_iterations.py'`. A
worktree created ten minutes earlier was importing a *different* checkout.

**Two independent hijacks, and each one alone is silent.**

1. `gridenv.sh` prepends `$WT/.shadow` to `PYTHONPATH` — a directory holding a
   no-op `__editable___vtsearch_0_1_0_finder.py` that shadows the venv's
   editable-install finder. That directory is **untracked**, so it does not
   exist in a new worktree, and nothing created it. The finder then wins.
2. `common.setup_env()` does `sys.path.insert(0, VTS_REPO)` with VTS_REPO
   **defaulting to `/exp/$USER/projects/vts-calib`** — a shared worktree from an
   older study. `sys.path[0]` beats `PYTHONPATH`, so even a correct `.shadow`
   would not have saved it.

The traceback only appeared because the branch had *added* a symbol. Had this
re-measure only *changed* behaviour — which is the usual case, and is exactly
what a cut-rule study does — every job would have run the wrong `cut_rules.py`
and produced a clean, plausible, wrong table.

**Cost.** ~15 minutes, caught before launch. The counterfactual is a full study.

**Now prevented (code, not advice):** `gridenv.sh` creates the `.shadow` shim if
it is missing and exports `VTS_REPO="${VTS_REPO:-$_VTS_WT}"`, so sourcing it from
a worktree pins *that* worktree at `sys.path[0]`. `preflight.sh` already checked
that VTS_REPO was set and clean — it just could not check the one thing that was
wrong, which was that nobody had set it at all.

**Also prevented (code):** `preflight.sh` now resolves `import vtscore` the way a
job does — through `common.setup_env()` — and fails if the file it lands on is
not inside `VTS_REPO`. That is the direct evidence the run measures your branch,
and it is checked rather than remembered. Setting `VTS_REPO` correctly is not the
same thing: this run had it right and still imported the other checkout.
