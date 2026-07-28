# MLP vs SVM ranker — extending the study

**Status: the core study shipped.** The image + SigLIP experiment that asked
"should VTSearch's MLP ranker be replaced by an SVM?" is complete — the report,
figures, and raw CSVs live in
[`docs/experiments/mlp-vs-svm/`](../experiments/mlp-vs-svm/REPORT.md).

**Verdict: keep the MLP.** The SVMs are more label-efficient in the first ~50
votes (they beat the MLP's cost@50, p < 0.05) but the MLP overtakes them
decisively by 200 votes (p < 0.001) and misses fewer rare-event matches, so no
SVM variant met the pre-registered switch criterion.

The trainer-pluggable evaluation machinery built for this study is now permanent
and reusable: `vtscore/eval/trainers.py` (registry + parameterised SVM specs),
the trainer/prevalence knobs on `vtscore/eval/voting_iterations.py`,
`vtscore/eval/seed_scores.py`, `vtscore/eval/timing_benchmark.py`, the wider SVM
grid + cuML backend in `vtscore/training/svm.py`, and the runner in
`scripts/experiments/mlp_vs_svm/`.

## Open work (only if a result motivates it)

<!-- item-sep -->

- **Hybrid ranker (SVM early → MLP later).** The image result shows a crossover:
  the SVMs have lower cost than the MLP through roughly vote 50 but the MLP is
  already ahead by vote 100 (cost@50 MLP 0.387 vs SVM 0.330; cost@100 MLP 0.359
  vs SVM 0.378). A hybrid that runs an SVM for the first ~40–60 votes and then
  switches to the MLP could in principle capture the SVM's early label-efficiency
  and the MLP's late dominance. The harness supports testing this directly: add a
  `svm_until@N` trainer (SVM while `n_votes < N`, MLP after) to
  `vtscore/eval/trainers.py` and run it through the same `stage_b_autopilot.py`
  grid. Measure, don't assume — the open risks are (a) a **calibration
  discontinuity** at the switch (the score distribution changes families, so the
  cross-calibrated threshold jumps and results reshuffle mid-session); (b)
  **closed-loop divergence** — the SVM chose the early votes, so the MLP inherits
  an SVM-shaped vote history rather than its own; and (c) **rare-event FNR** — the
  MLP's biggest edge is missing fewer rare matches, and that matters from the
  first votes, so an SVM-early phase could cost recall exactly where it's most
  expensive. The prior is that the gain is small (a ~0.05 cost edge over a short
  window) and may not survive the switch cost.

<!-- item-sep -->

- **Other media types.** The harness is single-embedder by construction but not
  by limitation. Audio (CLAP), video (X-CLIP), text (E5), and patch/region
  embedders could each be run through the same `stage_b_autopilot.py` grid by
  adding their datasets to `experiment_config.py` and pointing `seed_scores` at
  the right embedder — no new machinery, just configuration + report faceting.
  The image result (MLP wins as votes accumulate; SVM only leads at very low
  vote counts) is the prior to test against; a media type where concepts are
  tighter single clusters (identity-like) is where an RBF-SVM would have its best
  shot.

<!-- item-sep -->

- **GPU SVM backend.** cuML's SVM is wired in (`train_svm(backend="auto"|"cuml")`)
  but is currently broken on the HLTCOE Grid (its kernels fail to compile — an
  nvrtc CUDA-13/CUDA-12 toolchain mismatch), so the study ran SVMs on sklearn-CPU
  and the Stage C timing compares MLP-GPU vs SVM-CPU. If the cluster's RAPIDS/CUDA
  stack is fixed, re-run Stage C with cuML enabled (drop `VTSEARCH_DISABLE_CUML=1`)
  for an apples-to-apples GPU timing comparison; the sklearn-parity cross-check
  (`svm_backend_parity`) is already in place to validate the backends agree.
