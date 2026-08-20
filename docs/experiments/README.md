# `docs/experiments/` — index

One directory per study: a `REPORT.md` written on top of numbers generated
deterministically from the run's own CSVs, plus the generated tables and figures.
These are the **record of what a run produced** — they are archives, not plans,
and they are not pruned when the work they justified ships.

**What a report owes its reader** (the long form is in the `grid-experiments`
skill): two significant digits unless a decision needs a third, arm differences
**paired** and carrying a standard error; **figures** — the headline metric over
the axis the user spends, averaged over seeds *and* broken out per run — emitted
by a committed script from the same CSVs as the tables; and **literal examples**
of the errors behind every error rate, so a reader can tell a model error from an
annotation error. Analysis scripts must be committed: the doc gate checks that
paths cited here resolve.

**The other archives.** [`docs/reports/`](../reports/) holds standalone HTML study
pages (narrative write-ups with inlined charts — see [its
index](../reports/README.md)); [`docs/plans/`](../plans/) holds work still owed.

Add a row here whenever you add a study.

| Study | Question | Verdict |
|-------|----------|---------|
| [`acquisition-inclusion/`](acquisition-inclusion/REPORT.md) | Should the acquisition threshold be decoupled from the reporting threshold? (follow-up to #2847 / PR #2873) | Shipped; the design now lives in [`docs/ML.md`](../ML.md#threshold-calibration). |
| [`calibration/`](calibration/REPORT.md) | Trained vs oracle thresholds for Autopilot (#2781) — run twice, before and after PR #2784. | The runaway threshold was the conformal walk's `k=0` anchor, not the `NO_GOOD_THRESHOLD` sentinel; the raw-patch tree is calibration-bottlenecked and no pre-registered remedy recovers it. |
| [`gmm-cut/`](gmm-cut/REPORT.md) | Where should the GMM path place its cut (#2836)? Autopilot simulation plus a synthetic bench with known ground truth. | See BLUF; the #2846 remeasurement and the #2881 tail-α run ([pre-registration](gmm-cut/PREREG-2881.md), [result](gmm-cut/REPORT-2881.md)) close the EVT cut line — beating production now needs a better *fit*, not a better cut. |
| [`inclusion-knob/`](inclusion-knob/REPORT.md) | Why doesn't the Inclusion slider move the included set (#2693), and what mechanism would? | Plus [`SELECTION-BIAS.md`](inclusion-knob/SELECTION-BIAS.md), the study of what the detector's own sort costs the calibration set. |
| [`max-patch/`](max-patch/REPORT.md) | Does the HAC region tree earn its keep against raw-patch max-pooling? | Ship tree-free MaxPatch; drop the HAC tree from ingest (#2886). |
| [`mixin-schedule/`](mixin-schedule/REPORT.md) | How safe should safe-thresholds be (#2841) — how much GMM, for how long? | Neither voting mode should ever hand over: `slow_cap50` for region, `cap50` for binary. |
| [`mlp-vs-svm/`](mlp-vs-svm/REPORT.md) | Would a linear or RBF SVM rank better than the detector MLP? | Keep the MLP — no variant met the pre-registered switch criterion. **Superseded as a product decision:** it measured the auto-sized MLP head, and the shipped head has since moved to a linear SVM; see [`docs/ML.md`](../ML.md#the-three-heads-which-one-is-shipped-and-why). Narrative version: [the HTML report](../reports/2026-07-22-mlp-vs-svm-ranker.html). |
| [`overview-bench/`](overview-bench/REPORT.md) | What does a user actually get from each shipped configuration (#3129)? 3 representations × 6 haystacks × typed-vs-clicked, at production defaults. | A characterization, not a ship decision: boxes are worth more than the encoder (strip them and DINOv3 is worse than the shipped default), target scale is the strongest axis, positives are the binding constraint, and VG's own labels bound how good any of these numbers can look. |
| [`population-anchored-calibration/`](population-anchored-calibration/REPORT.md) | Fuse the haystack distribution into the trained threshold instead of scheduling it out (#2852 / #2853). Two runs: deep-regime, then anchor-mass sweep. | Fold-anchored mixture at κ=0.3 with the `mid` cut; fusion does **not** beat the blend on binary voting — a live regression tracked in [`population-anchored-calibration.md`](../plans/population-anchored-calibration.md). |
| [`safe-thresholds/`](safe-thresholds/REPORT.md) | Should `safe_thresholds` be forced on for everyone (#2799)? | Yes — and the setting was later deleted rather than left as a way to opt into a worse threshold. |
| [`spike-check-2847/`](spike-check-2847/REPORT.md) | Do the MLP-era cost spikes survive today's stack (#2847)? | Better — and it is mostly the *threshold*, not the head. |
