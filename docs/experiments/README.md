# `docs/experiments/` — index

One directory per study: a `REPORT.md` written on top of numbers generated
deterministically from the run's own CSVs, plus the generated tables and figures.
These are the **record of what a run produced** — they are archives, not plans,
and they are not pruned when the work they justified ships.

**Every study directory is named `YYYY-MM-DD-<slug>`**, dated when its report
first landed, so a bare `ls` sorts the archive chronologically and the newest
work is the last thing printed. The date is the *study's*, not its last edit: a
report that is later corrected, re-skinned or extended keeps the name it was
filed under, because that name is what every link in the tree points at. The
index below runs the other way — **newest first**. `scripts/check-docs.py`
enforces both halves: the name shape, and that every study has a row here.

**What a report owes its reader** (the long form is in the `grid-experiments`
skill): two significant digits unless a decision needs a third, arm differences
**paired** and carrying a standard error; **figures** emitted by a committed
script from the same CSVs as the tables; and **literal examples** of the errors
behind every error rate, so a reader can tell a model error from an annotation
error. Analysis scripts must be committed: the doc gate checks that paths cited
here resolve.

**Every study that simulates a user clicking owes the quality-over-clicks pair**
— how good that user's detector is as they click more, **averaged** (one panel
per dataset, one line per arm) and again **per run** (one panel per arm, every
seed as its own line). Click 0 is the free text sort, so the far left is what
typing was worth and the far right is what clicking was worth. One
implementation, [`scripts/experiments/calibration/curves.py`](../../scripts/experiments/calibration/curves.py);
do not write it again.

**…and it owes an interactive `viewer.html`, linked from the report.** The PNGs
answer the questions the analyzer asked; the viewer is what lets a reader ask
their own — any dataset (one / all / each), any category (one / all / each), any
subset of arms, any subset of embedders, seeds averaged or every seed, and any
metric the run emitted (cost, precision, recall, F1, FPR, FNR, average
precision, AUROC). Two draw toggles decide the shape: **overlay** puts every
varying dimension on one chart in distinct hues (off, each gets its own chart
with the ±1 SD shadow, which is the only place that shadow is readable), and
**oracle threshold** adds the cut the test labels say the model should have
used, dotted beside the solid line it achieved. Two more reference marks are
notched into the margins: the free text sort at the left and, for a run
launched with `CALIB_SKYLINE_ARMS`, the supervised skyline at the right. Built
by [`scripts/experiments/calibration/viewer.py`](../../scripts/experiments/calibration/viewer.py)
from the same CSVs as everything else, self-contained, no network.

**A page can be re-skinned without its results.** The template carries all of
the viewer's behaviour and the payload only the study's numbers, so
`python viewer.py --reskin docs/experiments/*/viewer.html` pushes a template
improvement onto every committed report — even one whose results directory is
long gone. It cannot add data the payload never carried, so a *series* still
needs a rebuild from the cell CSVs. Both of the ones added in #3325 were
backfilled that way in #3326: the oracle cut cost nothing to recover (its inputs
are on every base row ever emitted), and the skyline was measured by a second,
cheaper pass over the same cells and merged in with `--skyline-results`, because
it is vote-independent and re-running the loop for it would have replaced the
performance rows the reports' tables were read off.

**The other archives.** [`docs/reports/`](../reports/) holds standalone HTML study
pages (narrative write-ups with inlined charts — see [its
index](../reports/README.md)); [`docs/plans/`](../plans/) holds work still owed.

Add a row here whenever you add a study — at the top, since the table runs
newest first.

| Study | Question | Verdict |
|-------|----------|---------|
| [`2026-08-30-fit-quality-3329/`](2026-08-30-fit-quality-3329/PREREG.md) | Is the 2-component Gaussian mixture a **good** fit, not just the best one (#3329)? Absolute goodness-of-fit for the mixture that sets every threshold the app computes: tail calibration at the cut, class-conditional shape on the logit axis, component-to-class identification, and whether the shipped k=0.3 anchoring moves the fit at all. 192 cells on `vg_scale_any`, three geometries. | **Pre-registered, not yet run** - the plan is linked because no report exists yet. Every fit diagnostic in the tree is *relative* (one family against another, one cut against another), so a misspecification both families share is invisible to all of them; this is the first absolute measurement. H4 is pre-registered so that "the fit is wrong and it costs nothing" is a reportable finding rather than a null. |
| [`2026-08-29-inclusion-knob-3196/`](2026-08-29-inclusion-knob-3196/REPORT.md) | Does the Inclusion knob still have authority under the shipped linear SVM head (#3196)? 21 stops of the slider x 5 cut rules, two full head arms, 576 cells on `vg_scale`. | **The premise does not reproduce on real data, and the sign is the other way round** - the knob is at least as live under the SVM and significantly livelier in both DINOv3 environments. Nothing ships. The flat band the issue found is real but its axis is the **target**: dead-step rate rises monotonically small -> large under both heads, and 7 of the 8 individually dead region cells are `@large` (`clock@large` in all four seeds). |
| [`2026-08-28-voted-exclusion-3308/`](2026-08-28-voted-exclusion-3308/PLAN.md) | Did the #3308 voted-media exclusion buy anything on real data, and is `EXCLUSION_MIN_REMAINDER = 60` the right floor (#3312)? Two stages: production scale, where the floor is inert, and deep voting on a modest collection, where the remainder crosses every arm's floor. | **Pre-registered, not yet run** — the plan is linked because no report exists yet. Both numbers behind the shipped floor today are synthetic; the live hypothesis is that the change is a rigor improvement with no measurable benefit at production scale, and the design makes that a reportable finding rather than a null. |
| [`2026-08-28-calibration-fold-count-3310/`](2026-08-28-calibration-fold-count-3310/REPORT.md) | Does more cross-calibration ever pay, once its wall-clock price is charged (#3314)? Executes [`PLAN.md`](2026-08-28-calibration-fold-count-3310/PLAN.md), pre-registered before the run. | **Keep `calibrate_count = 2`** — the gate closes on **cost, not benefit**. More folds do help, early and only on the DINOv3 geometries (−0.0057 ± 0.0012 at K=6, 1–25 votes, region voting), but every fold count that clears the margin costs ≥ 2.3× the user's per-step retrain against a 1.5× ceiling; K=3 is affordable and half the margin. 86% of a fold's price is the anchored EM — make that cheaper and K=6 lands inside the ceiling. |
| [`2026-08-27-good-mining-3267/`](2026-08-27-good-mining-3267/REPORT.md) | Does a different Autopilot **opening** mine better Goods (#3267)? Eight openings x 24 environments x 42 seeds, 200 clicks each, every cell seeded from a typed query. | **`top_long` (`g8@top,b4@mid`) — just take more off the top**: +5.8 positives/100 clicks and −0.018 cost against `prod`, and +11 against the length-matched control, so it wins on *where* it clicks. Yield falls monotonically with sampling depth and the gain is concentrated in **scarce** categories (+9.5 vs +2.5). The shippable Inclusion lever gets ~65% of it, but only because `k-10` **saturates to `@top`** — it is a lossy way of asking for the top of the sort. |
| [`2026-08-27-calibration-fraction-3287/`](2026-08-27-calibration-fraction-3287/REPORT.md) | What Train/Calibrate split should a detector use (#3287)? `calibration_fraction` has been `0.5` since it was introduced — the obvious default, never measured — and it sits on the shipped threshold path. | **0.5 is not optimal on the shipped default**, and the axis is the **embedder, not the voting mode**: `siglip` wants 0.4 (−0.013 ± 0.003, winning in every vote band), `siglip2_l` and both CLIP capacities want 0.3, and `dinov3_patch` wants 0.5 in *either* voting mode. So the per-mode default the issue proposed would average across a disagreement. Evidence only; no production change proposed. |
| [`2026-08-25-calibration-fold-combine/`](2026-08-25-calibration-fold-combine/REPORT.md) | Pooling fold scores or averaging per-fold cuts (#3115) — two docstrings in the repo assert contradictory premises about whether fold scores are comparable. | **Both are wrong, in different places.** Averaging beats pooling in both voting modes; the comparability premise is **mode-dependent** (quantile space is worth −0.032 on region and costs +0.036 on binary). The contamination argument is **prevented by the stratified splitter** and its cited evidence conflates a degenerate *cut* with a degenerate *holdout*. Production's anchored rule is untouched. |
| [`2026-08-25-vg-scale/`](2026-08-25-vg-scale/REPORT.md) | Where does VTSearch stand on same-class target size (#3156 / #3276)? 12 classes x {small, medium, large} x 5 embedder columns x 20 seeds, shipped defaults, 3600 cells. | A map, not a decision: cost is ~4x higher on sub-patch targets than on large ones and four fifths of it is the RANKING, not the cut; region voting is lowest in every band without removing the size effect; and the column with the best free text sort (`siglip2_l`) takes a median 29 clicks to beat it. Carries an interactive [`viewer.html`](2026-08-25-vg-scale/viewer.html) over every slice. |
| [`2026-08-24-transfer-2883/`](2026-08-24-transfer-2883/REPORT.md) | Is the #2836 decomposition's dominant `transfer` term a real cost (#2883)? 552 cells, [pre-registered](2026-08-24-transfer-2883/PREREG.md) before submission. | **`transfer` is not a cost — it is a variance measured against an optimistic reference.** All four pre-registered hypotheses land on both arms independently; one sub-prediction inside H4 was wrong and is called out. |
| [`2026-08-21-inclusion-cut-rule/`](2026-08-21-inclusion-cut-rule/REPORT.md) | Which cut rule should answer the Inclusion knob (#2865)? Five candidate rules × thirteen stops of the knob, scored at their own inclusion, on the two region-voting environments plus their binary controls. | Keep the shipped `mid_tilt`: it delivers 95% of the slider and nothing beat it without losing materially somewhere. The `mid` cut it replaced is **inert** — one admitted set for the whole knob in every measured step. |
| [`2026-08-19-linhead-convergence-2808/`](2026-08-19-linhead-convergence-2808/REPORT.md) | Is the linear head's spike reduction limited by early stopping (#2808)? 450 cells over two datasets × two embedders. | **No — and the question was built on a misread number** (#2847's conformal-threshold pair, not the threshold we ship). Convergence buys −0.016 ± 0.005 in worst-step regret, vanishes on the default embedder and costs ~5.3× training: **do not raise `TRAIN_EPOCHS`.** What is worth acting on is +1.2 ± 0.5 positives — positives, not spikes, are the binding constraint. |
| [`2026-08-18-fastproc-3146/`](2026-08-18-fastproc-3146/REPORT.md) | Should the image embedders switch to the fast (torchvision) processor (#3146)? | **They already had.** `transformers` 5 removed the `Fast` suffix, so the flip happened silently inside our own `>=4.49` pin. GPU preprocessing — the issue's other proposed fix — is **not** cleared: a 1013-cell power run leaves `cost` and `fnr` grazing the 0.005 margin for 1.09× per cell. What outlives the issue is a **second unrecorded axis on the pile** — which backend built a cell. Narrative version: [the HTML report](../reports/2026-08-18-image-processor-3146.html). |
| [`2026-08-18-gpu-node-3160/`](2026-08-18-gpu-node-3160/REPORT.md) | Why do two nodes both answering to `gres/gpu:v100` produce `siglip2_l` vectors 1.5e-04 apart (#3160), 30× what the calibration studies resolve? | **A type is not a device — and the device was never the problem.** Two V100 parts fed the same input compute bit-identically at every one of 27 blocks; the divergence enters on the **host**, where the 384 px resize rounds differently under AVX-512 than under AVX2. The nine outlier nodes are exactly the nine whose CPUs have no AVX-512. |
| [`2026-08-17-embed-precision-3143/`](2026-08-17-embed-precision-3143/REPORT.md) | Can the image encoders run in half precision (#3143)? Ten arms, precision crossed deliberately with the card. | fp16 is a **very small perturbation** — 2.9e-6 cosine on `siglip2_l`, retrieval order intact, no detectable benchmark effect at 1013 paired cells — so adopt it selectively for the heavy encoders, but **do not flip the default**: `siglip` gains nothing and a global flip would strand the fp32 pile. `bf16` is disqualified. §5 is where the cross-node 1.5e-04 was found, chased in [`2026-08-18-gpu-node-3160/`](2026-08-18-gpu-node-3160/REPORT.md). |
| [`2026-08-12-overview-bench/`](2026-08-12-overview-bench/REPORT.md) | What does a user actually get from each shipped configuration (#3129)? 3 representations × 6 haystacks × typed-vs-clicked, at production defaults. | A characterization, not a ship decision: boxes are worth more than the encoder (strip them and DINOv3 is worse than the shipped default), target scale is the strongest axis, positives are the binding constraint, and VG's own labels bound how good any of these numbers can look. |
| [`2026-08-12-calibration-fold-count/`](2026-08-12-calibration-fold-count/REPORT.md) | Is 2 still the right number of cross-calibration folds (#2897)? A counterfactual screen plus a live A/B, decision rules fixed as module constants before the run. | **Keep `calibrate_count = 2`; the study ships nothing.** Binary voting says so outright — no K beats 2 by the margin and more folds monotonically hurt in the deep regime. Region voting mechanically recommends K=6, but at 2.68× the interactive retrain, with its live effect below the ship margin once acquisition feedback moves, and the mechanism check that was meant to validate it **cannot be read** (a defect in the instrument, not a result). Follow-ups #3115, #3116. |
| [`2026-08-07-acquisition-inclusion/`](2026-08-07-acquisition-inclusion/REPORT.md) | Should the acquisition threshold be decoupled from the reporting threshold? (follow-up to #2847 / PR #2873) | Shipped; the design now lives in [`docs/ML.md`](../ML.md#threshold-calibration). |
| [`2026-08-07-spike-check-2847/`](2026-08-07-spike-check-2847/REPORT.md) | Do the MLP-era cost spikes survive today's stack (#2847)? | Better — and it is mostly the *threshold*, not the head. |
| [`2026-08-05-population-anchored-calibration/`](2026-08-05-population-anchored-calibration/REPORT.md) | Fuse the haystack distribution into the trained threshold instead of scheduling it out (#2852 / #2853). Two runs: deep-regime, then anchor-mass sweep. | Fold-anchored mixture at κ=0.3 with the `mid` cut; fusion does **not** beat the blend on binary voting — a live regression tracked in [`population-anchored-calibration.md`](../plans/population-anchored-calibration.md). |
| [`2026-08-04-gmm-cut/`](2026-08-04-gmm-cut/REPORT.md) | Where should the GMM path place its cut (#2836)? Autopilot simulation plus a synthetic bench with known ground truth. | See BLUF; the #2846 remeasurement and the #2881 tail-α run ([pre-registration](2026-08-04-gmm-cut/PREREG-2881.md), [result](2026-08-04-gmm-cut/REPORT-2881.md)) close the EVT cut line — beating production now needs a better *fit*, not a better cut. |
| [`2026-08-04-mixin-schedule/`](2026-08-04-mixin-schedule/REPORT.md) | How safe should safe-thresholds be (#2841) — how much GMM, for how long? | Neither voting mode should ever hand over: `slow_cap50` for region, `cap50` for binary. |
| [`2026-08-03-safe-thresholds/`](2026-08-03-safe-thresholds/REPORT.md) | Should `safe_thresholds` be forced on for everyone (#2799)? | Yes — and the setting was later deleted rather than left as a way to opt into a worse threshold. |
| [`2026-07-31-calibration/`](2026-07-31-calibration/REPORT.md) | Trained vs oracle thresholds for Autopilot (#2781) — run twice, before and after PR #2784. | The runaway threshold was the conformal walk's `k=0` anchor, not the `NO_GOOD_THRESHOLD` sentinel; the raw-patch tree is calibration-bottlenecked and no pre-registered remedy recovers it. |
| [`2026-07-29-max-patch/`](2026-07-29-max-patch/REPORT.md) | Does the HAC region tree earn its keep against raw-patch max-pooling? | Ship tree-free MaxPatch; drop the HAC tree from ingest (#2886). |
| [`2026-07-27-inclusion-knob/`](2026-07-27-inclusion-knob/REPORT.md) | Why doesn't the Inclusion slider move the included set (#2693), and what mechanism would? | Plus [`SELECTION-BIAS.md`](2026-07-27-inclusion-knob/SELECTION-BIAS.md), the study of what the detector's own sort costs the calibration set. |
| [`2026-07-22-mlp-vs-svm/`](2026-07-22-mlp-vs-svm/REPORT.md) | Would a linear or RBF SVM rank better than the detector MLP? | Keep the MLP — no variant met the pre-registered switch criterion. **Superseded as a product decision:** it measured the auto-sized MLP head, and the shipped head has since moved to a linear SVM; see [`docs/ML.md`](../ML.md#the-three-heads-which-one-is-shipped-and-why). Narrative version: [the HTML report](../reports/2026-07-22-mlp-vs-svm-ranker.html). |
