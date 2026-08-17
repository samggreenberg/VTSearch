# 2026-08-13 — the report died on the laptop's RAM, not on the GRID's (#3131)

**What happened.** Every expensive thing in this study ran on the cluster, and
then the write-up ran `./run-tests.sh` on the laptop — a 3 GB box. `pyright`
runs under node with a ~1.8 GB default heap, so it aborted with *"Ineffective
mark-compacts near heap limit"*, which reads like a type-check failure and is
not one. Bumping `NODE_OPTIONS=--max-old-space-size=8192` on a machine with 3 GB
total made it worse rather than better, and the retry was killed outright: the
session ended on `Exit code 137`, the OOM killer, with the report committed but
the gate never green.

Two smaller traps rode along. The wall-clock wrapper in `run-tests.sh` treats
137 as *its* kill signal, so an OOM death is announced as **`TESTS TIMED OUT`** —
the banner names the wrong cause. And a standalone `pyright` had already
returned `0 errors` minutes earlier; the same command inside the suite died,
because the suite had the frontend build's memory alongside it.

**Cost.** The session, and the three hours of work in its context.

**Prevented?** *Partly.* `run-tests.sh` now reads `MemAvailable` up front and
prints a LOW MEMORY banner with the `sbatch` line to run it on the GRID instead.
It warns rather than blocks — 6 GB is the observed want, not a measured cliff.

**The general form, and the one worth keeping:** *the machine you analyse on is
part of the experiment's resource plan.* The GRID was sitting there with 502 GB
on the login node and idle `cpu` nodes. A gate that needs 6 GB belongs on a
node that has it, and moving it there costs one `sbatch`. Nothing about "the run
is finished" means the laptop can now take the load — the write-up's gates are
the heaviest thing a study asks of a *local* machine, and they arrive exactly
when there is the most uncommitted context to lose.
