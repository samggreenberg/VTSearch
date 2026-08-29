# 2026-08-29 — a per-mode split that still pooled two environments (#2877)

**Study:** #2877's acquisition-offset re-run on the pile. **Cost:** near-miss —
caught while reading the finished report, before anything was written up. The
verdict would have been wrong in a way that read as a clean per-mode result.

## What happened

The analyzer was rebuilt for this study specifically so that a mean over two
voting modes could not hide a disagreement between them: the ship rule is
evaluated per mode, the pooled table is stamped descriptive, and a planted-answer
self-test asserts that the pooled and region verdicts *differ* on a grid sized so
pooling would hide the split.

It worked. And then the `binary` group turned out to hold **two environments**:

| environment | mode | adopted |
|---|---|---|
| `siglip × whole_image` | binary | −1, −2, −3, −4 (cost *falls* 0.02–0.04) |
| `siglip+dinov3_patch × whole_image` | binary | **−1 only** (spikes 4.5% → 35.6%) |
| `siglip+dinov3_patch × max_patch` | region | −1, −2, −3, −4 |

Pooled into one `binary` verdict those two give **`-1` only** — the second
environment's spike rejections dominate. Read that way the study says "binary
wants −1, region wants anything", which is a clean, plausible, publishable
voting-mode split, and it is not what the data says. The largest disagreement in
the whole grid is *between the two binary environments*, which is precisely the
evidence that a mode gate is the wrong shape.

The same hazard the split was built to prevent, one level down.

## Why it was invisible

The grouping variable was chosen from the study's *question* (does the voting
mode matter?) rather than from the grid's *structure*. Three environments were
run; two of them share a mode; the code grouped by mode and so silently averaged
a pair that disagrees. Nothing was broken, and the per-mode tables looked exactly
as designed.

It surfaced only because the binary half finished hours before the region half
and was analysed alone — that run adopted all four arms, the pooled-binary run
adopted one, and the two could not both be right.

## Now prevented

- `per_env_acq_2877.py` computes the ship rule per `(embedder, style)` —
  environment — reusing `analyze_acq._core_summary` so it is the same statistic
  by the same code path, not a second implementation that can drift.
- It also **emits the report's tables** (`--markdown`) and figures
  (`--figures`). A hand-transcribed draft of those tables had already put a
  `genuine_blip_rate` in a column headed "deep spikes"; the two differ by an
  order of magnitude in the one environment where the verdict turns on them.

## Still only advice

**Group by the grid's structure, not by the study's question.** A per-X split is
only safe when X is the finest axis along which cells can disagree. Before
reading any grouped verdict, count the distinct `(dataset, embedder, style)`
cells inside each group: more than one means the group is itself a pooled
result, and pooling is what the split was supposed to stop.

The diagnostic that caught it generalises: **analyse a subset early and check the
subset's verdict against the whole's.** They disagreed here, and the disagreement
was the finding.

## A related near-miss in the same run

The plan pre-registered that the absolute deep-spike thresholds
(`SPIKE_DEEP_COST = 0.25`, calibrated to COCO's ~0.137 cost scale) would be
near-saturated at this study's 0.24–0.45 costs, and would therefore be
uninformative. **Wrong**, and it decided the outcome: a deep spike needs a high
absolute cost *and* a ≥0.20 excess over the oracle, and here cost is high because
the *rankings* are hard (oracle 0.22–0.38), so the excess is ~0.03 and base rates
land at 0.5–4.5%. That left the guardrail live — and every rejection in the study
is a spike rejection, not a cost one. Had the prediction held, the run would have
adopted `-4` in two environments on cost alone.

**An absolute threshold does not transfer, and *which direction* it fails in
depends on whether the environment's cost is driven by the cut or by the
ranking.** #2877's original environment was expensive at the cut; two of these
three are expensive at the ranking.
