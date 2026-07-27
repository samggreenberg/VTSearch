# Inclusion-knob experiment (issue #2693)

Measures why the Inclusion slider fails to move the decision threshold and
compares candidate knob designs. Findings + recommendation:
[`docs/experiments/inclusion-knob/REPORT.md`](../../../docs/experiments/inclusion-knob/REPORT.md).

Runs on a single CPU box (no GPU, no SLURM). Scratch data (AG News CSV, E5
embedding cache) lives under `INCKNOB_EXP` (default `~/.cache/incknob-exp`);
committed outputs go to `docs/experiments/inclusion-knob/`.

## Stages

```bash
cd scripts/experiments/inclusion_knob
python prepare_agnews.py      # download AG News + embed 2400 items with E5 (~3 min CPU)
python run_sweep.py           # full grid (~15 min CPU); --quick for a smoke test
python summarize.py           # figures + summary_tables.md from sweep.csv
```

## Files

| file | role |
|---|---|
| `common.py` | env setup, paths, timing helpers |
| `prepare_agnews.py` | Stage 0: cache real E5 passage embeddings for AG News |
| `synthetic.py` | controlled two-cluster arm with tuned overlap levels |
| `knobs.py` | treatments (raw / label-smoothed), temperature fit, the four knob designs, metrics |
| `run_sweep.py` | Stage 1: the grid; writes `sweep.csv` |
| `summarize.py` | Stage 2: figures + markdown tables |
