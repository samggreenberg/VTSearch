# A cheaper stop-list from Bad votes (#4432): affordable, but not shipped

**Question.** #4170's stop-list did not ship because of cost. Each Bad vote
prunes the Good templates (drops the descriptors that Bad's page matches), and
each pruned template was then re-verified against all 2,000 pages in the
shortlist, so retrain p90 was 11 to 13 s. Pruning only removes template
descriptors, and matching runs from template to page. So a pruned template
cannot fit a page its unpruned self fails on. Does re-verifying it only where the
unpruned template passes make the stop-list cheap enough, and does it then earn
its place?

**Answer: it is cheap now, but it does not clear its bar, so `STOPLIST_POLICY`
stays `"off"`.** The cheaper verification is merged, so a later stop-list design
starts from it.

| `gated` stop-list vs the shipped path (H1, #4440), FullMarks v5.0 tier `m`, 36 classes, 95% intervals | 10 clicks | 25 clicks | 50 clicks |
|---|---|---|---|
| ranking AP, replicate 1 | +0.009 [+0.000, +0.021] | +0.011 [+0.000, +0.029] | +0.014 [+0.000, +0.034] |
| ranking AP, replicate 2 (halves swapped) | +0.005 [−0.006, +0.025] | +0.013 [−0.004, +0.035] | −0.019 [−0.069, +0.012] |
| **ranking AP, both pooled** | +0.007 [−0.002, +0.022] | **+0.012 [−0.001, +0.032]** | −0.003 [−0.031, +0.019] |
| returned-set F1, both pooled | −0.004 [−0.021, +0.012] | **+0.004 [−0.016, +0.025]** | +0.003 [−0.041, +0.040] |
| retrain p90 (replicate 1, same GPU type) | | 3.2 s vs 3.0 s shipped (11.2 s before) | |

**The pre-registered bar** was AP at 25 clicks above the shipped path with the
interval clear of zero, and retrain p90 ≤ 5 s. Replicate 1 met it, just
(lower bound +0.0004). The gain came from two classes, and the returned-set F1
was unresolved, so the owner asked for a second replicate before shipping. The
rule for it was posted on #4432 before any results:

- ship if, pooled over both replicates, AP at 25 clicks still clears zero
  **and** the returned-set F1 difference at 25 clicks has a lower bound above
  −0.02;
- otherwise, merge the cheaper verification with the policy off.

F1 passes (lower bound −0.016). **AP does not (lower bound −0.001).**

## What the stop-list does, class by class

The gain is real but narrow: the same two classes win in both replicates.

| class | AP at 25 clicks, replicate 1 | replicate 2 |
|---|---|---|
| `tobacco800/logo_asg54f00_1` | 0.40 → 0.67 (+0.27) | +0.30 |
| `staver/stamp_stampds-00213_1` | 0.76 → 0.87 (+0.11) | +0.21 |
| the other 34 | within ±0.01, except `tobacco800/logo_cgr96c00_1` in replicate 2 (−0.07 AP, −0.23 F1) | |

**The loss that matters is a session that never recovers.** In replicate 2,
`ucsf/logo_p_lorillard_crest` falls from AP 0.81 to 0.004 at 50 clicks. Both
arms take the same harmful first click: a Good whose box sends test AP from
0.67 to 0.002. Then 40 Bad votes follow:

| click | 0 | 1 (Good) | 2–38 (Bads) | 40 | 42–50 |
|---|---|---|---|---|---|
| shipped: AP / found | 0.67 / 0 | 0.002 / 1 | 0.001 / 1 | 0.39 / 2 | **0.81 / 4** |
| stop-list: AP / found | 0.67 / 0 | 0.002 / 1 | 0.004 / 1 | 0.004 / 1 | **0.004 / 1** |

On the shipped path, Stage 1 keeps walking down the same queries and reaches
positives by click 40. With the stop-list, every Bad prunes the queries
further, and the walk never gets back to the mark. Replicate 1 has no such
session, which is why it read cleaner.

## What changed

- **`VerificationCache.best_many(..., parents=)`.** A pruned template (keyed
  by its prune tag) is verified only on pages where its unpruned parent clears
  the 8-inlier gate. Every other page gets no fit. `maybe_structural_rerank`
  passes the parents whenever the stop-list prunes. With the shipped policy
  `"off"` nothing is pruned, so the app is unchanged.
- **`sota_documents.py --swap-halves`.** Clicks in the test half and scores on
  the click half: a second, independent replicate per class.

## Reproduce

```bash
source scripts/experiments/pile/pile_env.sh
for flags in "" "--stoplist gated" "--swap-halves" "--stoplist gated --swap-halves"; do
  python scripts/experiments/fullmarks/sota_documents.py --tier m --max-v 50 $flags \
      --matrix /expscratch/$USER/fullmarks/votes-4162/matrix-m \
      --feature-cache /expscratch/$USER/fullmarks/features --out <dir>/<arm>
done
```

Run directory: `/expscratch/sgreenberg/stoplist-4432/`. Replicate 1's
shipped-path arm is #4440's validation run (`/expscratch/sgreenberg/hybrid-4440/validate`).
Replicate 2's gated arm ran in two parts: 9 classes on a V100 node that ran out
of memory (cancelled), then the remaining 27 on an L40S. Its timings are not
comparable; its AP and F1 do not depend on the GPU. `measurements/` holds all
four arms' `steps.csv`.
