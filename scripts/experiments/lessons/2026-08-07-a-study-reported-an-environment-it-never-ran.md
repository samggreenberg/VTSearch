# 2026-08-07 — a study reported an environment it never ran (#2877)

**What happened.** #2877 pre-registered `visual_genome_m × siglip` as the
**region-voting** generalisation check for the acquisition cut, and justified
the whole exercise on region voting's scoring geometry: *"a media's score is a
max over ~24 region-node scores, so the Bad mode is an extreme-value
statistic."* The harness takes `region_voting=True` for that arm, the run
drained clean, the analysis was careful, and a report went out describing a
region-voting result. **That arm does not region-vote.**

`region_voting` is a *request*, not a guarantee. `_good_training_vec` pools the
dragged ground-truth box only when the media carries a stored `patch_grid`, and
silently falls back to the whole-image embedding otherwise. `siglip` is a
single-vector embedder: no `patch_grid`, no `patch_regions`. So the run trained
on whole-image vectors (verified: the region-voting vector is **byte-identical**
to the whole-image vector on 200/200 medias carrying a box), scored whole-image
(`region_aware=False`), and blended under **`cap50`** — the *binary* schedule.
It was a second **binary**-voting environment throughout.

**Why it survived every check.** Nothing was broken, so nothing complained. The
dataset really is boxed; the flag really was set; `REGION_VOTING_BY_DATASET` is
keyed by **dataset** while whether region voting happens is a property of the
**embedder**, and the two only coincide for patch embedders. The harness's own
config docstring called the whole VG block "region voting", so the mislabel was
inherited rather than invented, and it had been sitting there across several
studies.

**Cost.** A published report, a PR and an issue comment all had to be corrected,
including a headline recommendation ("gate the offset by voting mode") that the
run could not support — both environments were binary. The measurement itself
survived intact; only what it was evidence *about* changed. It would have been
far worse merged: a voting-mode conditional shipped on evidence from two
binary-voting runs.

**Now prevented (code, not advice):** `simulate_voting_iterations` warns when
`region_voting=True` is requested and no media carries a `patch_grid`, naming
the consequence ("this run is BINARY voting"). `experiment_config.py`'s
docstring now says which VG arms actually region-vote and which silently do not.

**Still advice — a flag you passed is not a property you got.** When an
experiment's *rationale* rests on a mechanism ("max over region nodes",
"grouped calibration", "patch geometry"), verify the mechanism is present in the
data before running, not the flag that requests it. One line is enough:
`any(m.get("patch_grid") is not None for m in medias.values())`. The general
form: **assert the premise, not the parameter.** A silent fallback is designed
to keep a run working, which is exactly why it will not tell you the run changed
meaning. This is a sibling of the #2846 lesson — there a worktree silently ran
another checkout's code; here a config silently ran another environment's
geometry.
