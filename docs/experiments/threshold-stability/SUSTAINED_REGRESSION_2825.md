# Sustained wrong-way runs (#2825) — when more labels make the detector worse, and it stays worse

**Question.** Normally more good/bad votes → lower held-out cost. Find the runs where the
cost curve legitimately goes the *other* way and **stays** there, and work out what is going
wrong in each. Explicitly not the #2790 deep spike, which is transient and recovers.

**Metric.** `cost = FNR + FPR` on the held-out set, throughout — the same headline the sweeps
were run on. Detection scans `cost`. The ranking-vs-calibration split compares it with
`oracle_cost`, which is that same cost at the best achievable cut, so the whole analysis stays
in one unit. `average_precision`/`auroc` are printed where a source carries them but never
classify anything.

**Data.** Existing per-step rows only — **no new runs**. 4,484 trajectories:

| source | dataset | mode | embedder | runs |
|---|---|---|---|---|
| `vg_bool/` | VG | boolean (`whole`) × 4 heads | SigLIP 1 + 2 | 1,800 |
| `broad_hs2/` | COCO | boolean × 5 heads | SigLIP 2 | 1,500 |
| `results/cells/` | COCO | boolean, 6 calibration arms | SigLIP 2 | 300 |
| `acq/` | COCO | boolean, acquisition arms | SigLIP 2 | 300 |
| `hac_newgmm/`, `hac_confirm/` | COCO | **region voting** (`hac`) | DINOv3 | 200 |
| `hac_dinov2/` | COCO | **region voting** | DINOv2 | 40 |
| `max-patch/results/cells.csv.gz` | VG | **region voting** (max_patch / max_patch_hac / max_hac) + whole_image, t→150 | DINOv3-patch, SigLIP | 344 |

Tool: `scripts/experiments/threshold_stability/sustained_regression.py` (new; sibling of
`deep_spikes.py`). 29 unit tests in `tests_lib/sorting/test_sustained_regression.py`.

---

## How a run has to earn the label

Two criteria, both on a median-smoothed `cost` series so a single-step spike cannot trigger
either:

1. **Sustained rise** — cost climbs from the run's best-so-far over ≥ K=5 steps by ≥ δ=0.10,
   mostly upward, and then *persists* at that level rather than snapping back.
2. **Late gap** — the median of the last 10 steps sits ≥ δ above an earlier best that never
   came back.

### The threshold-only version of this question is meaningless

At K=5 / δ=0.10, magnitude alone flags **2,185 of 4,484 runs (49%)**. The same thresholds
applied to *surrogate* runs — built by reshuffling each run's own step-to-step cost moves —
flag **74–90%**. A magnitude rule here is mostly measuring how jumpy a run is, and these runs
are very jumpy.

So every detection additionally has to clear a **per-run permutation test**: 99 surrogates of
that run's own volatility with the net drift removed, keep it only if its severity lands in
the top 5%.

Two details that turned out to matter, both now pinned by tests:

- **Reshuffle in blocks of 5, not single moves.** A #2790 deep spike is a `+0.9` step
  immediately followed by a `−0.9` recovery. Shuffling single moves separates the pair, so the
  surrogate manufactures exactly the sustained level shifts we are testing for, and the null
  becomes self-defeating. Blocks keep the pairing.
- **Anchor the segment floor at the *last* best-so-far, not the first.** With the first, a long
  flat stretch gets swallowed into the segment and drags the "mostly upward" fraction to the
  rejection boundary, losing real climbs off a plateau.

### Calibration of the test itself

Running the full pipeline on trend-free surrogates (type-I error, target ≤ 0.05):

| source | type-I |
|---|---|
| VG boolean SigLIP 1 / 2 | 0.028 / 0.031 |
| COCO region voting DINOv3 | 0.056 |
| VG max_patch / max_patch_hac / max_hac | 0.033 / 0.000 / 0.111 |

**Power is the honest weakness.** Planting a sustained `+0.30` rise into a trendless surrogate
of each run's own volatility, the detector recovers it only **9–11%** of the time on boolean
runs and **7–28%** on region-voting runs. These cost curves are heavy-tailed enough that a
trend has to be extreme to separate from the spikes.

> **Everything below is a lower bound.** 5.0% is the incidence of wrong-way runs *severe
> enough to prove themselves against their own noise*, not the incidence of the phenomenon.

---

## What was found

**223 of 4,484 runs (5.0%)** are sustained wrong-way runs.

They are not marginal. Median rise in cost **+0.309**; median cost **0.486 at onset → 0.742 at
the last step of the run**; and **98% are still worse at the final step than where the segment
started**. This is not a dip that recovers — the user finishes the session with a worse
detector than they had 40 votes earlier.

### 1. It is an FNR collapse, not an FPR one

| | median Δ over the segment |
|---|---|
| Δcost | **+0.309** |
| ΔFNR | **+0.530** |
| ΔFPR | **−0.164** |

**93% of detections are FNR-driven.** The detector gets *more* precise and dramatically less
complete: the cut marches up through the test positives until it is finding almost nothing.
Median Δthreshold +0.091 — but note the #2790 methodology caveat: the model is retrained every
vote, so raw threshold values are not comparable across steps. The FNR/FPR movement is.

### 2. The mechanism: a run of bad votes with no positives to hold the cut down

This is the finding. Across the wrong-way segment:

| | median |
|---|---|
| good votes added | **1** |
| bad votes added | **9** |
| share of votes that were GOOD, inside the segment | **0.09** |
| share of votes that were GOOD, over the whole run | 0.22 |
| segments in which **not one** new positive arrived | **34%** |
| longest stretch with no new positive | **7 steps** |

`n_good` at onset: median **3**. `n_bad` at onset: median **7**. Onset phase: `hard` in
**219 of 223**. Median onset step **t=10** — this is an *early-loop* failure that then never
recovers.

A worked example — COCO `carrot`, seed 1, SVM head, straight from the per-step rows:

```
  t   cost   oracle   fnr     fpr    threshold  n_good  n_bad  phase  calib
  7  0.2065  0.1570  0.0000  0.2065   -0.1941      3      4    hard   blend   <- best
  9  0.1935  0.1623  0.0789  0.1145   -0.2571      3      6    hard   blend
 11  0.2787  0.2367  0.2368  0.0419   -0.2082      3      8    hard   blend
 13  0.3803  0.1643  0.3684  0.0119   -0.0935      3     10    hard   blend
 15  0.7681  0.2358  0.7632  0.0049   -0.1228      3     12    hard   blend
 17  0.8951  0.2580  0.8947  0.0004    0.1205      3     14    hard   blend   <- peak
 ...  n_good finally moves at t=18; cost claws back only to 0.50 by t=60
```

Nine consecutive votes, every one of them a **bad**, `n_good` frozen at 3. FNR climbs
0.08 → 0.89 while FPR collapses to 0.0004. The run ends at cost 0.502 against a best of 0.194.

This is the **sustained sibling of the #2790 deep spike**: same root cause (positive
starvation — the cut is under-determined because almost nothing is pinning it from below), but
where a spike is one bad boundary vote that snaps back, this is a *prolonged* run of bad votes
that walks the cut all the way up and leaves it there. #2790 found the transient form. This is
the form that does not recover.

### 3. Mostly a calibration failure — but the ranking genuinely degrades too

Splitting the cost rise into the part an oracle threshold could undo and the part it could not:

| | share of detections |
|---|---|
| calibration (oracle flat, gap to it opens) | **56%** |
| mixed | 35% |
| ranking (oracle_cost itself carries ≥ half the rise) | **10%** |

Median Δ`oracle_cost` **+0.058** against median Δ`cost` +0.316 — a ranking share of 0.17. But
**`oracle_cost` itself rose by > 0.05 in 56% of detections**. So the issue's "scariest case"
is real and common, just not dominant: in most wrong-way runs more labels are genuinely making
the *ranking* worse as well, and a perfect threshold would recover roughly five sixths of the
damage but not all of it.

### 4. The linear head does **not** fix this

#2790's conclusion was to ship the linear head, which stops the deep spikes. It does not stop
this:

| head | COCO boolean | VG boolean |
|---|---|---|
| linear | **0.077** | 0.044 |
| reg-mlp | 0.077 | **0.069** |
| anneal-svm | 0.073 | — |
| svm | 0.060 | 0.051 |
| mlp | 0.050 | **0.038** |

On COCO the linear head is the *worst* arm and the MLP the best; on VG the MLP is the best. No
model class is protective. Head-independence agrees: of 123 affected cells, 62% show it in only
one head, and only 29% hit both a linear-family and an MLP-family head. **This is not a model
flexibility problem** — which is exactly what you would expect if the cause is what the loop is
being fed rather than what is fitted to it.

### 5. Region voting is almost clean — and that is partly a real result

| mode | runs | wrong-way |
|---|---|---|
| COCO boolean, SigLIP 2 | 2,100 | **6.1%** |
| VG boolean, SigLIP 2 | 900 | **5.3%** |
| VG boolean, SigLIP 1 | 900 | **4.8%** |
| COCO region voting, DINOv2 | 40 | 2.5% |
| VG region voting, `max_hac` DINOv3-patch | 69 | 1.4% |
| VG boolean `whole_image`, SigLIP | 68 | 1.5% |
| COCO region voting, DINOv3 | 200 | **0.5%** |
| VG region voting, `max_patch` / `max_patch_hac` DINOv3-patch | 138 | **0.0%** |

Read this carefully. Region-voting power is 7–28%, comparable to boolean's 9–11%, so the gap is
not purely a sensitivity artifact — but with only 378 region-voting runs the confidence
interval is wide, and the VG max-patch runs improve by a median **−0.55** in cost over the run
(against −0.10 for VG boolean), which leaves far less room for a wrong-way stretch to stand out.
The defensible statement is: **the phenomenon is concentrated in the boolean/whole-image loop,
and there is no evidence it is a significant problem in region voting.** A dedicated
region-voting sweep with more seeds would be needed to put a real bound on it.

### 6. Which classes

| class | wrong-way |
|---|---|
| COCO `book` | **0.413** |
| COCO `bottle` | **0.400** |
| VG `flower` | **0.300** |
| COCO `carrot` | 0.227 |
| COCO `traffic light` | 0.125 |
| VG `car` | 0.108 |

Small, cluttered, high-count-per-image objects. These are exactly the classes where the `hard`
phase surfaces boundary items that are almost all negatives — the acquisition mix in §2.

### 7. Calibration-rule arms: suggestive, not conclusive

On the six-arm #2790 comparison (50 runs each): `conformal-k8` 0/50, `conformal-k2` 1/50,
`rank-transfer-k2` 2/50, `argmin-k2` 2/50, `conformal-k2-med3` 3/50, `argmin-k8` 4/50. The
direction favours conformal over argmin but 50 runs per arm cannot carry that claim.

---

## What this says to do

The mechanism is a **prolonged run of negative-only votes during the `hard` phase while
`n_good` is still ~3**. Three things follow, in order of how directly they attack it:

1. **Positive-seeking acquisition** — the fix already identified at the end of #2790 and still
   not built. This study is independent evidence for the same thing: the damage tracks the
   *vote mix*, not the model class. If the loop interleaved exploit/top picks so positives keep
   arriving, the 9-bad-votes-to-1-good stretches would not happen.
2. **A no-new-positives guard.** 34% of wrong-way segments received not one positive, and the
   median dry stretch is 7 steps. That is directly observable at runtime — the loop knows
   `n_good` has not moved for k votes — and is a cheap trigger for either switching acquisition
   or declining to move the cut.
3. **Do not expect the linear head to cover this.** It is the right call for #2790's spikes and
   it is not protective here (§4). The two failures need different fixes.

## Limitations

- **Power ~10%.** 5.0% is a floor on the incidence, not an estimate of it.
- Region voting has 378 runs against 4,100 boolean; the near-zero rate is under-powered.
- The VG max-patch source has no `oracle_cost` column, so 2 detections there are reported as
  undecided rather than split into ranking vs calibration.
- Per-item detail (which specific vote rotated the model) needs a targeted trace re-run on the
  identified `(class, seed)` cells; this study works from per-step aggregates only.

## Data-integrity note

The `head` field in every sweep row is the per-step *scoring family* (`cosine` vs `mlp`), not
the `--head-strategy` arm — so it reads `mlp` for every row of the `linear`, `svm` and
`reg-mlp` sweeps alike. The arm survived only in the output directory name, which means it is
unrecoverable from `vg_bool_all.jsonl.gz`. `DATA_AND_SCHEMA.md`'s claim that rows are
self-describing is wrong on this point. `_row_realistic` now records `head_strategy`; the tool
reads the arm from the path for existing data.

## Reproducing

```bash
python scripts/experiments/threshold_stability/sustained_regression.py \
  --jsonl-root "vg-bool=$EXP/vg_bool" \
  --jsonl-root "coco-bool-heads=$EXP/broad_hs2" \
  --jsonl-root "coco-rv-dinov3=$EXP/hac_newgmm" \
  --csv-source "vg-maxpatch=/exp/sgreenberg/max-patch/results/cells.csv.gz" \
  --null-block 5 --per-run-null 99 --top 30
# add --power-rise 0.30 --fp-check 2 to reproduce the power / type-I tables
# add --show "dataset=coco,class=carrot,seed=1" for the per-step deep-dive tables
```

Outputs on the Grid: `/exp/sgreenberg/threshold-stability/sr2825/` — `detections_final.json`,
`logs/final-448402.out` (main scan), `logs/calib4-448367.out` (calibration).
