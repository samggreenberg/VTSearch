# 2026-08-12 — an idempotency guard matched the wrong occurrence (#3129)

**What happened.** Twice in one session, a patch script guarded "have I already
applied this?" on a substring that appears elsewhere in the target file, so it
skipped the edit **and reported success**:

* `if "vg_box_small" in s` matched the `BOXED_BY_DATASET` entry added by an
  earlier patch, so the `DATASET_EMBEDDERS` registration never happened. The
  verifier then reported PASS over an empty grid.
* `if "--job-name" in s` matched the `sbatch --job-name=bench-prep` lines already
  in the launcher, so the preflight flags were never wired.

Both were caught only by inspecting the file afterwards. A patch that skips
silently is worse than one that fails: it leaves the tree looking done.

**Prevented?** *Advice only.* Guard on a string **unique to the insertion** — a
new symbol name, or a literal marker from the inserted block — never on a token
that could plausibly appear in the surrounding file. And verify the postcondition
(grep for what you inserted) rather than trusting the script's own report.
