# 2026-08-21 — a pooled number over an axis that rescales the metric is that axis's endpoints (#2865)

**Study:** #2865 cut-rule × Inclusion sweep. **Cost:** ~1 hour, and only that
because the sizing step existed: four timed cells were run through the analyzer
before the 336-cell array was submitted, and the verdict block came back with
`d_regret_pooled: 26.6` for an arm whose inclusion-0 regret is a few hundredths.

The sweep scores each arm at each stop `k` of the Inclusion knob under **that
stop's own cost weights**, which is right — scoring every row at the run's
reporting inclusion would flatten every arm's regret curve by construction. What
it missed is that those weights are not a reweighting, they are a **rescaling**:

```python
def inclusion_cost_weights(inclusion_value):
    if inclusion_value >= 0:
        return 1.0, 2.0**inclusion_value
    return 2.0 ** (-inclusion_value), 1.0
```

`cost = fpr_weight*FPR + fnr_weight*FNR`, so a cost at `k=10` is denominated in
units **1024×** the ones at `k=0`. Two consequences, both silent:

- Every pooled statistic over the knob is the `k=±10` pair with a rounding error
  attached. The mean "over the knob" never saw the middle of it.
- The pre-registered **harm tolerance of 0.01 cost units** means "a thousandth
  of an error rate" at one end of the slider and "a whole error rate" at the
  other. A fixed tolerance on a rescaling axis is not one bar, it is thirteen.

The fix is a change of *units*, not of the decision rule: divide by `2**|k|`, the
larger of the two weights. It is exactly 1 at inclusion 0, so nothing the two
prior calibration runs measured moves; elsewhere it recovers a weighted mean of
FPR and FNR that is bounded like a rate. The raw difference rides along as
`d_regret_cost` for anyone who wants it.

**The general form.** *When a sweep's axis changes the metric's scale, no
statistic may be pooled across it until the metric is re-expressed in a unit the
axis does not move.* This is a sibling of ["band the axis the mechanism runs
on"](../../../.claude/skills/grid-experiments/SKILL.md) rather than the same
thing: banding protects you from averaging across a *crossover*, where the mean
hides a real reversal; this protects you from averaging across a *rescaling*,
where the mean is arithmetically dominated by the axis's endpoints and is not a
summary of anything. Both look like an ordinary column of numbers.

Places worth checking for the same shape: anything pooled over `inclusion_k`
(`INCLUSION_SWEEP_KS` rows in `analyze.py`'s budget sweep are ratios against
`alpha(k)`, so they are already scale-free — but the raw `sweep_threshold` /
`excess_fnr` columns beside them are not), and any future sweep over a knob that
multiplies a loss term rather than reallocating it.

**Status: advice, not prevented.** There is no mechanical check that says "this
axis rescales the metric" — it is a property of what the numbers mean, not of the
frame's shape. What *is* now in the tree is the smoke test that caught it:
`launch_incl_2865.sh size` runs one cell per arm and the analyzer is run on those
cells before the array is submitted. A verdict block from four real cells is
cheap, and it is where an absurd number has a chance to look absurd.
