# Inclusion-knob experiment (issue #2693)

Measures why the Inclusion slider fails to move the decision threshold and
compares candidate knob designs. Findings + recommendation:
[`docs/experiments/2026-07-27-inclusion-knob/REPORT.md`](../../../docs/experiments/2026-07-27-inclusion-knob/REPORT.md).

A follow-on study asks whether the conformal rule's `alpha(k)` budget — a
split-conformal guarantee, so it assumes exchangeable calibration votes —
survives the fact that VTSearch's votes are chosen by the detector's own sort:
[`docs/experiments/2026-07-27-inclusion-knob/SELECTION-BIAS.md`](../../../docs/experiments/2026-07-27-inclusion-knob/SELECTION-BIAS.md).

Runs on a single CPU box (no GPU, no SLURM). Scratch data (AG News CSV, E5
embedding cache) lives under `INCKNOB_EXP` (default `~/.cache/incknob-exp`);
committed outputs go to `docs/experiments/2026-07-27-inclusion-knob/`.

## Stages

```bash
cd scripts/experiments/inclusion_knob
python prepare_agnews.py      # download AG News + embed 2400 items with E5 (~3 min CPU)
python run_sweep.py           # full grid (~15 min CPU); --quick for a smoke test
python summarize.py           # figures + summary_tables.md from sweep.csv
```

Selection-bias study (independent of the stages above, same cached embeddings):

```bash
python run_autopilot_sweep.py   # canonical Autopilot vote order (~45 min CPU); --quick to smoke-test
python summarize_autopilot.py   # autopilot_prod_tables.md from the two autopilot_prod_*.csv frames
```

`run_autopilot_sweep.py` is a **driver** over
`vtscore.eval.voting_iterations.simulate_voting_iterations`, not a simulation of
its own (issue #3408). It therefore measures the shipped detector by
construction — the `linear_svm` head, the app's Autopilot phase machine, the
acquisition-inclusion offset, the production Train/Calibrate split — and it does
**not** reproduce the committed `autopilot_sweep.csv`, which is the 2026-07-30
run's record from the earlier hand-rolled loop and stays as it is. New runs write
`autopilot_prod_steps.csv` / `autopilot_prod_budget.csv`.

## Files

| file | role |
|---|---|
| `common.py` | env setup, paths, timing helpers |
| `prepare_agnews.py` | Stage 0: cache real E5 passage embeddings for AG News |
| `synthetic.py` | controlled two-cluster arm with tuned overlap levels |
| `knobs.py` | treatments (raw / label-smoothed), temperature fit, the four knob designs, metrics |
| `run_sweep.py` | Stage 1: the grid; writes `sweep.csv` |
| `summarize.py` | Stage 2: figures + markdown tables |
| `run_autopilot_sweep.py` | Selection-bias study: thin driver over `simulate_voting_iterations`; writes `autopilot_prod_steps.csv` + `autopilot_prod_budget.csv` |
| `summarize_autopilot.py` | Tables for the selection-bias study |
