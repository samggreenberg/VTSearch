# MLP vs SVM ranker study — results

**[→ Read the report: `REPORT.md`](REPORT.md)**

Verdict: **keep the MLP.** The SVMs are more label-efficient in the first ~50
votes but the MLP overtakes them decisively by 200 votes, especially on
rare-event false-negatives — so no SVM variant met the pre-registered switch
criterion. See `REPORT.md` for the full write-up, curves, and take-aways.

Run on the HLTCOE Grid, image + SigLIP only, 3 datasets × 5 categories ×
{natural, 1%-rare} × 5 seeds. The runner lives in
[`scripts/experiments/mlp_vs_svm/`](../../../scripts/experiments/mlp_vs_svm).

## Artifacts

| File | What |
|---|---|
| `REPORT.md` | The report (figures embedded). |
| `fig_*.png` | Cost / FPR / FNR curves, Stage A screen, Stage C timing. |
| `stage_b.csv.gz` | Definitive per-vote run: every (dataset, category, arm, seed, trainer, t) row. |
| `stage_a.csv.gz` | Kernel/hyperparameter screen (label-count sweep). |
| `stage_c.csv` | GPU/CPU train + inference timing. |
| `stage_c_parity.json`, `prepare_info.json` | Provenance (backend parity, dataset counts). |

## Regenerate the report from the CSVs

```bash
cd docs/experiments/mlp-vs-svm && mkdir -p stage_b && \
  gunzip -c stage_b.csv.gz | ...  # split back per-cell, or point summarize at a dir of CSVs
gunzip -k stage_a.csv.gz
python ../../../scripts/experiments/mlp_vs_svm/summarize.py --results .
```

`summarize.py` is deterministic from the CSVs, so the write-up can't drift from
the data.
