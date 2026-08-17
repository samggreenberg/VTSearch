# 2026-08-12 — a launcher default outlives the directory it names (#2881)

**What happened.** `launch_tail_2881.sh` defaulted `VTSEARCH_DATA_DIR` to
`/exp/$USER/max-patch/datadir` and its reused prepare to
`/exp/$USER/calibration-safe-linear/results` — two study dirs that had been
archived to `/expscratch` and deleted that morning. The launcher did not notice
either: nothing reads the data dir until a cell runs, so the failure would have
surfaced as individual cells dying inside a 552-cell array, which reads as "a few
flaky cells" rather than "the whole run is pointed at nothing."

The second half was worse, because it fails *silently* rather than loudly. The
archived prepare's `crops/` entries are symlinks into the deleted `max-patch`
dir, and the launcher copies them with `ln -s "$(readlink -f "$f")"`. **`readlink
-f` resolves a dangling link happily** — it returns the path the link points at
whether or not anything is there — so the reuse step "succeeds" and recreates the
link, still dangling, with no error anywhere.

**Cost.** None this time: the run was launched by hand with both paths
overridden, and the archive still held the real files. But that is luck, not
process — a launcher run as documented would have burned the array.

**Prevented.** Two new preflight checks:

- **check 10** — `VTSEARCH_DATA_DIR` exists, has `embeddings/`, and that directory
  holds at least one `.pkl`. Points at `pile_env.sh` when it does not.
- **check 11** (`--reuse-prepare DIR`) — `prepare_info.json` exists and every
  entry under `crops/` resolves, tested with `[[ -e ]]`, which follows the link.
  That is the one-line check `readlink -f` is not.

`launch_tail_2881.sh` now sources `pile_env.sh` for its data/models dirs and
passes `--reuse-prepare` when the dir exists. **The general form: a launcher
default that names a *study* dir is a time bomb — studies get archived. Name the
pile, or fail loudly at preflight.**
