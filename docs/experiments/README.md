# `docs/experiments/` — index

One directory per study: a `REPORT.md` written on top of numbers generated
deterministically from the run's own CSVs, plus the generated tables and figures.
These are the **record of what a run produced** — they are archives, not plans,
and they are not pruned when the work they justified ships.

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
long gone. It cannot add data the payload never carried: a study that measured
no oracle cut and no skyline still shows neither.

**The other archives.** [`docs/reports/`](../reports/) holds standalone HTML study
pages (narrative write-ups with inlined charts — see [its
index](../reports/README.md)); [`docs/plans/`](../plans/) holds work still owed.

Add a row here whenever you add a study.

| Study | Question | Verdict |
|-------|----------|---------|
| [`acquisition-inclusion/`](acquisition-inclusion/REPORT.md) | Should the acquisition threshold be decoupled from the reporting threshold? (follow-up to #2847 / PR #2873) | Shipped; the design now lives in [`docs/ML.md`](../ML.md#threshold-calibration). |
| [`calibration/`](calibration/REPORT.md) | Trained vs oracle thresholds for Autopilot (#2781) — run twice, before and after PR #2784. | The runaway threshold was the conformal walk's `k=0` anchor, not the `NO_GOOD_THRESHOLD` sentinel; the raw-patch tree is calibration-bottlenecked and no pre-registered remedy recovers it. |
| [`calibration-fold-combine/`](calibration-fold-combine/REPORT.md) | Pooling fold scores or averaging per-fold cuts (#3115) — two docstrings in the repo assert contradictory premises about whether fold scores are comparable. | **Both are wrong, in different places.** Averaging beats pooling in both voting modes; the comparability premise is **mode-dependent** (quantile space is worth −0.032 on region and costs +0.036 on binary). The contamination argument is **prevented by the stratified splitter** and its cited evidence conflates a degenerate *cut* with a degenerate *holdout*. Production's anchored rule is untouched. |
| [`good-mining-3267/`](good-mining-3267/REPORT.md) | Does a different Autopilot **opening** mine better Goods (#3267)? Eight openings x 24 environments x 42 seeds, 200 clicks each, every cell seeded from a typed query. | **`top_long` (`g8@top,b4@mid`) — just take more off the top**: +5.8 positives/100 clicks and −0.018 cost against `prod`, and +11 against the length-matched control, so it wins on *where* it clicks. Yield falls monotonically with sampling depth and the gain is concentrated in **scarce** categories (+9.5 vs +2.5). The shippable Inclusion lever gets ~65% of it, but only because `k-10` **saturates to `@top`** — it is a lossy way of asking for the top of the sort. |
| [`gmm-cut/`](gmm-cut/REPORT.md) | Where should the GMM path place its cut (#2836)? Autopilot simulation plus a synthetic bench with known ground truth. | See BLUF; the #2846 remeasurement and the #2881 tail-α run ([pre-registration](gmm-cut/PREREG-2881.md), [result](gmm-cut/REPORT-2881.md)) close the EVT cut line — beating production now needs a better *fit*, not a better cut. |
| [`inclusion-knob/`](inclusion-knob/REPORT.md) | Why doesn't the Inclusion slider move the included set (#2693), and what mechanism would? | Plus [`SELECTION-BIAS.md`](inclusion-knob/SELECTION-BIAS.md), the study of what the detector's own sort costs the calibration set. |
| [`inclusion-cut-rule/`](inclusion-cut-rule/REPORT.md) | Which cut rule should answer the Inclusion knob (#2865)? Five candidate rules × thirteen stops of the knob, scored at their own inclusion, on the two region-voting environments plus their binary controls. | Keep the shipped `mid_tilt`: it delivers 95% of the slider and nothing beat it without losing materially somewhere. The `mid` cut it replaced is **inert** — one admitted set for the whole knob in every measured step. |
| [`inclusion-knob-3196/`](inclusion-knob-3196/REPORT.md) | Does the Inclusion knob still have authority under the shipped linear SVM head (#3196)? 21 stops of the slider x 5 cut rules, two full head arms, 576 cells on `vg_scale`. | **The premise does not reproduce on real data, and the sign is the other way round** - the knob is at least as live under the SVM and significantly livelier in both DINOv3 environments. Nothing ships. The flat band the issue found is real but its axis is the **target**: dead-step rate rises monotonically small -> large under both heads, and 7 of the 8 individually dead region cells are `@large` (`clock@large` in all four seeds). |
| [`max-patch/`](max-patch/REPORT.md) | Does the HAC region tree earn its keep against raw-patch max-pooling? | Ship tree-free MaxPatch; drop the HAC tree from ingest (#2886). |
| [`mixin-schedule/`](mixin-schedule/REPORT.md) | How safe should safe-thresholds be (#2841) — how much GMM, for how long? | Neither voting mode should ever hand over: `slow_cap50` for region, `cap50` for binary. |
| [`mlp-vs-svm/`](mlp-vs-svm/REPORT.md) | Would a linear or RBF SVM rank better than the detector MLP? | Keep the MLP — no variant met the pre-registered switch criterion. **Superseded as a product decision:** it measured the auto-sized MLP head, and the shipped head has since moved to a linear SVM; see [`docs/ML.md`](../ML.md#the-three-heads-which-one-is-shipped-and-why). Narrative version: [the HTML report](../reports/2026-07-22-mlp-vs-svm-ranker.html). |
| [`overview-bench/`](overview-bench/REPORT.md) | What does a user actually get from each shipped configuration (#3129)? 3 representations × 6 haystacks × typed-vs-clicked, at production defaults. | A characterization, not a ship decision: boxes are worth more than the encoder (strip them and DINOv3 is worse than the shipped default), target scale is the strongest axis, positives are the binding constraint, and VG's own labels bound how good any of these numbers can look. |
| [`population-anchored-calibration/`](population-anchored-calibration/REPORT.md) | Fuse the haystack distribution into the trained threshold instead of scheduling it out (#2852 / #2853). Two runs: deep-regime, then anchor-mass sweep. | Fold-anchored mixture at κ=0.3 with the `mid` cut; fusion does **not** beat the blend on binary voting — a live regression tracked in [`population-anchored-calibration.md`](../plans/population-anchored-calibration.md). |
| [`safe-thresholds/`](safe-thresholds/REPORT.md) | Should `safe_thresholds` be forced on for everyone (#2799)? | Yes — and the setting was later deleted rather than left as a way to opt into a worse threshold. |
| [`spike-check-2847/`](spike-check-2847/REPORT.md) | Do the MLP-era cost spikes survive today's stack (#2847)? | Better — and it is mostly the *threshold*, not the head. |
| [`vg-scale/`](vg-scale/REPORT.md) | Where does VTSearch stand on same-class target size (#3156 / #3276)? 12 classes x {small, medium, large} x 5 embedder columns x 20 seeds, shipped defaults, 3600 cells. | A map, not a decision: cost is ~4x higher on sub-patch targets than on large ones and four fifths of it is the RANKING, not the cut; region voting is lowest in every band without removing the size effect; and the column with the best free text sort (`siglip2_l`) takes a median 29 clicks to beat it. Carries an interactive [`viewer.html`](vg-scale/viewer.html) over every slice. |
| [`voted-exclusion-3308/`](voted-exclusion-3308/PLAN.md) | Did the #3308 voted-media exclusion buy anything on real data, and is `EXCLUSION_MIN_REMAINDER = 60` the right floor (#3312)? Two stages: production scale, where the floor is inert, and deep voting on a modest collection, where the remainder crosses every arm's floor. | **Pre-registered, not yet run** — the plan is linked because no report exists yet. Both numbers behind the shipped floor today are synthetic; the live hypothesis is that the change is a rigor improvement with no measurable benefit at production scale, and the design makes that a reportable finding rather than a null. |
