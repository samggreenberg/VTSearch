# 2026-08-07 — the fidelity check failed because the incumbent had shipped (#2846)

**What happened.** The #2846 Grid re-measure came back with the check that
licenses the whole study **red**: `pooled_mid`, the variant labelled "this is
what production does", disagreed with the run's own threshold on **84 % of
13 653 steps** (max abs diff 0.24, against 0.0 in #2836 four days earlier). The
obvious readings were "the harness broke" or "the branch under test broke it",
and both would have led to hours of bisecting a diff that was innocent.

**Neither. Production moved.** Splitting the mismatches by the base row's
`threshold_provenance` gave a perfect 1:1 partition:

| production took | steps | `pooled_mid` reproduces it? |
|---|---|---|
| `gmm_blend` | 2 158 | yes — max abs diff **0.0** |
| `fold_anchored[*]` | 11 495 | no |

Between the two runs, `d195b004` shipped the fold-anchored threshold, `196085b5`
moved it to κ=0.3 + midpoint cut, and `b03d54e5` made the fused path
unconditional. `pooled_mid` was still bit-for-bit correct on every step that took
the old path. The study's *baseline definition* was two ship decisions stale.

**Why it is worth an entry rather than a shrug.** The consequence was not a
broken run — every within-rule contrast stayed valid — it was that one *class* of
claim silently expired: "rule X beats the shipped midpoint" no longer meant "rule
X beats what we ship". A re-measure that had not looked at the provenance column
would have republished that claim in good faith. On this cluster a study's
baseline is a moving target, because studies here ship things.

**Now prevented (code), twice over:**

1. `analyze_cut.py`'s `production_blend_sanity` no longer just reports
   `ok: false` — on failure it breaks the mismatch down by
   `threshold_provenance`, so "the harness is broken" and "the incumbent moved"
   are distinguishable at a glance instead of by investigation.
2. The ship decision no longer depends on a reconstruction at all.
   `base_row_contrasts` pairs every rule against **the run's own base row** —
   the threshold production actually used on that step, whatever it was — and
   `decisions.beats_production` / `ship_candidate` are computed from that.
   `beats_midpoint` survives as the historical #2836 contrast and gates nothing.
   A baseline that is read rather than reconstructed cannot go stale, so the
   next incumbent can ship without expiring anyone's conclusions.

**Still advice:** when a study defines *any* arm as "what production does",
re-read that definition against `git log` on the path it names. The base row
covers the threshold; the next study will name something else.
