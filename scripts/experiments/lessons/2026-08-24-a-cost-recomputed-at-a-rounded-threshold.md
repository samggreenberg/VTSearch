# 2026-08-24 — a cost re-evaluated at a rounded threshold is a different cost (#2883)

**Study:** #2883 transfer characterisation. **Cost:** none — it was caught by a
sanity check written in the same hour, on three smoke cells, before the array was
submitted. It is recorded because of *how large it was*, not how long it lived.

The study needed the test-set oracle's cost sitting in the `__cutdiag` frame next
to a new cross-fitted one, so the two references could be differenced within a
row. The obvious way to get it, given a row that already carries the oracle
threshold:

```python
naive_cost, _fpr, _fnr = operating_cost(base_scores, base_labels, rows[0]["oracle_threshold"], wf, wn)
```

`oracle_threshold` goes through `_r()` on the way into the row — it is **rounded
to six decimals**. Re-scoring at a rounded threshold moves any test item whose
score lies between the true cut and the rounded one across the boundary. On this
grid the test split carries ~55 positives, so **one FNR step is 1/55 = 0.018** —
and the term the whole study exists to measure is **+0.037**. The recomputed
reference disagreed with the row's own `oracle_cost` by up to **0.0183**: half
the quantity under study, injected by a rounding that looks like a rounding.

The fix is to take the number from the function that computes it, unrounded,
rather than reconstructing it from a rounded artefact of itself:

```python
_naive_tau, naive_cost, _nfpr, _nfnr = oracle_cut(base_scores, base_labels, wf, wn)
```

**The general form.** *A rounded threshold is a display value, not an input.*
Anywhere a decision boundary is persisted and later re-applied to data, the
persisted copy is lossy in a way that is invisible in the column and
discontinuous in the metric — because a threshold's effect on a rate is a step
function of the items near it, and the step size is `1/n` of the *rarer* class,
not of the sample. The fewer positives, the bigger the artefact: this is worst
exactly where these studies live.

Worth checking for the same shape: any analysis that reads a `tau_*` column out
of `__cutdiag` and re-scores with it rather than differencing it against another
`tau_*`. Differences of rounded thresholds are fine; *re-application* of one is
not.

**Status: prevented, for this analyzer.** `analyze_transfer.reference_sanity`
joins the new `cost_test_oracle_naive` against the row's own `oracle_cost` and
fails the run's summary when they disagree by more than 2e-6, and
`selftest_analyze_transfer` plants a misaligned frame to assert the check
actually fires rather than merely existing. That is a check on *this* pair of
columns; the general habit above is still advice.
