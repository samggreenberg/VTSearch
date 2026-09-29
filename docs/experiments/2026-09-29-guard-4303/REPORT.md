# Does the shipped opening hurt at a 5% pool?

**No, by the rule fixed on #4303 before the run, but it doesn't help either.** At a
5% pool (COCO Better's haystack thinned to 5% positive), the opening #4288 shipped,
`g3@top,b4@mid,g20+dry1/16@top`, ends the session at the same AP as the old
`g3@top,b4@mid`: −0.0013 ± 0.0007 at vote 150 (1.9 paired SE, inside the 2-SE bound).
No session ends without a detector in either arm.

What the walk costs here is the early dip: −0.035 AP at vote 25. It also costs
slightly at vote 100 (−0.0037 ± 0.0013) and in oracle cost at vote 150
(+0.0033 ± 0.0012, worse). When positives are plentiful, the learned sort finds them
anyway: both arms hold 43 Goods by vote 150.

Weighed against #4222's +0.051 AP at 0.44% and +0.030 at 0.1%, the ship stands.

Pools are scenarios, not claims about users. Binary voting only; region voting is out of
scope for now (owner, 2026-09-29).

## What was run

- COCO Better, SigLIP, binary voting, haystack thinned to 5%
  (`CALIB_HAYSTACK_PREVALENCE=0.05`, as #4220's 5% scenario did). 144 class@band
  cells × 5 seeds, 150 votes each, on dev after #4288.
- **old:** `g3@top,b4@mid`, spelled out, since the default no longer is it.
  **new:** the app default. The #4222 launcher runs these as `h0.05-old` and
  `h0.05-new`.
- Read with `analyze_drystop_4222.py` (old as `g3`, new as `g3b4g20d16`). A session
  with no detector scores AP 0.

## Results

| vote | AP, old | new minus old (± paired SE) |
|---|---|---|
| 25 | 0.44 | −0.035 ± 0.003 |
| 50 | 0.49 | +0.0060 ± 0.0022 |
| 100 | 0.54 | −0.0037 ± 0.0013 |
| 150 | 0.54 | **−0.0013 ± 0.0007** |

- **How the walk ended:** it met 20 Goods in 82% of sessions, ran dry in 18%, and
  never handed over in 0.1%. Median opening: 26 votes (old: 7).
- **Sessions with no detector at vote 150:** 0 in both arms, against 1.1% for both
  at vote 25.
- **Goods found by vote 150:** 43 (new) and 43 (old).

By size at vote 150, every band is within noise of zero:

| band | AP at vote 150, new minus old | at vote 25 |
|---|---|---|
| large | −0.0010 ± 0.0006 | −0.038 ± 0.006 |
| medium | −0.0010 ± 0.0012 | −0.048 ± 0.006 |
| small | −0.0020 ± 0.0016 | −0.017 ± 0.003 |

## What this means

The shipped opening is a low-prevalence win and roughly neutral when positives are
common:

| pool | AP gain at vote 150 | at vote 25 |
|---|---|---|
| 5% | −0.0013 | −0.035 |
| 0.44% | +0.051 | −0.036 |
| 0.1% | +0.030 | −0.067 |

The one cost present everywhere is the early dip. A future refinement could cut the
walk short when the pool turns out to be rich, but at −0.001 there is nothing here to
fix at vote 150.

Files: `opening.csv`, `harvest.csv`, `quality_paired.csv`, `starved.csv`, `verdict.json`.
