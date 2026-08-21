# 2026-08-21 — a launcher pinned a head that stopped being production (#2865)

**Study:** #2865 cut-rule × Inclusion sweep. **Cost:** near-miss — caught while
reading the launcher before submitting, roughly three hours after the app-side
change landed. Had it not been caught, 336 cells and ~250 CPU-hours would have
produced a clean, plausible table about a detector nobody has.

`launch_incl_2865.sh` was written on 2026-08-12 and carried:

```sh
export CALIB_HEAD=linear
```

which was correct that day: the logistic head was production. On the morning of
the run, PR #3198 made the **linear SVM** the shipped head and moved
`PRODUCTION_HEAD` with it, so an *unset* `CALIB_HEAD` now resolves to
`linear_svm` and the pin had quietly become a named legacy arm. The same launcher
also still scored `max_patch_pca_hac`, a geometry #2886 removed from ingest.

**The general shape is the opposite of how it feels.** The worry with a launcher
is that *it* will rot. What actually happens is that the **app moves** and the
launcher keeps faithfully reproducing a configuration that used to be production.
Nothing errors, no arm is missing, the analysis runs, the report is internally
consistent — and it is about the wrong detector. It is the same failure family as
[a study default is not a shipped
default](2026-08-12-a-study-default-is-not-a-shipped-default.md), one step
earlier: there, a *study* default was mistaken for a shipped one; here, a pin
that *was* shipped stopped being.

**Prevented.** `preflight.sh` check 12 compares every knob that has a named
shipped constant — `CALIB_HEAD` vs `PRODUCTION_HEAD`, `CALIB_PATCH_STYLES` vs
`PRODUCTION_PATCH_STYLE`, `CALIB_ANCHORED_{WEIGHTS,RULES,FOLD_COMBINES}` vs the
`FOLD_ANCHOR_*` constants, `CALIB_ACQ_INCLUSION_OFFSET` vs
`ACQUISITION_INCLUSION_OFFSET`, plus `CALIB_SAFE_THRESHOLDS` and
`CALIB_BLEND_SCHEDULE` — and **fails** on any divergence the study has not
*declared* by name:

```sh
bash scripts/experiments/preflight.sh --exp "$CALIB_EXP" --diverges head,anchor_weight
```

That is the whole design: a study is always allowed to pin the axis it sweeps,
and never allowed to pin one silently. Verified against the exact stale
configuration this incident is about — `CALIB_HEAD=linear`,
`CALIB_PATCH_STYLES=max_patch_pca_hac`, a rule set without `mid_tilt` — which the
check names, all three, and refuses.

To give the patch geometry something to be checked *against*,
`vtscore/eval/voting_iterations.py` now exports `PRODUCTION_PATCH_STYLE` rather
than inlining `"max_patch"` at the resolution site. A default that is not named
cannot be compared with, which is why it was the one knob with no constant.

**Still only advice:** the check can only compare knobs that *have* a named
constant. A default expressed as a literal inside a function body is invisible to
it — so when you add one, give it a name, the way `PRODUCTION_HEAD` and
`PRODUCTION_PATCH_STYLE` have one.
