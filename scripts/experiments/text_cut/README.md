# `text_cut/` — what should draw the line on a typed-query sort (#3826)

The question: every text sort draws a green/red line with
`calculate_gmm_threshold`, the midpoint of a two-Gaussian fit. #3585 found that
fit barely identifiable on a text sort (re-initialising it moved 6.2% of the
verdicts). This harness measures the shipped line and its candidate replacements
on **stability** and **admitted-set quality**, over four environments and four
text towers.

The report is
[`docs/experiments/2026-09-22-text-cut-3826/REPORT.md`](../../../docs/experiments/2026-09-22-text-cut-3826/REPORT.md).

| Stage | Script | What it produces |
|---|---|---|
| capture | `capture_text_sorts_3826.py` | one `.npz` per (dataset, text embedder): every category's whole score array **and its labels**, plus a fidelity check against the app's own scoring call |
| the rules | `rules_3826.py` | every candidate line as `scores -> cut`, and each fitted rule's optimiser perturbations |
| gate | `gate_3826.py` | one CSV per capture: (sort, rule, frame) rows for the `full` sort, `opt:*` optimiser perturbations and `boot:*` bootstrap resamples |
| tables | `analyze_3826.py` | stability, quality, paired, per-dataset, leave-one-dataset-out, null-model check, Autopilot, literal examples |
| figures | `figures_3826.py` | the report's figures, from the same CSVs |

`launch_3826.sh` drives it on SLURM (`capture`, `gate`, `analyse`, `status`).
Results root: `/expscratch/sgreenberg/textcut-3826`.

## Three things to know before reusing this

**Every admitted set of one sort is nested.** All lines are thresholds on the
same array, so two lines disagree on exactly `|n_adm_a - n_adm_b|` medias. The
analyzer reads every comparison off admitted counts; there is no per-media join
to get wrong.

**The gate asserts `gmm_shipped` is `calculate_gmm_threshold` to the bit** on
every sort before it records anything. The rule is re-implemented (to expose the
EM's tolerance and start), and a re-implementation that drifted from the shipped
rule would make every comparison against it meaningless.

**The #3585 corpus rides along without labels.** `sorts_vgscale.npz` and
`sorts_overview.npz` go through the gate as a stability-only frame with the
pre-#3585 sklearn fit as a reference rule, which is where the issue's own
numbers are reproduced before anything new is claimed. The `vg_scale` pile has
been rebuilt since (13.4k → 12-16k medias per cell), so the new `vg_scale`
captures are not the same arrays.
