# 2026-08-12 — memory, not CPU, is what you run out of (#3129)

**What happened.** Every stage of this study asked for 64–96G without measuring
anything. Actual peak RSS was ~1.1 GB for whole-image cells and ~14 GB for patch
cells. The per-user cap is 1.07 TB, so `16 x 64G` claimed 95 % of it and three
separate later submissions queued behind this study's own array.

**Prevented?** *Yes* — `preflight.sh --mem --conc` computes the footprint against
the tightest applicable QOS cap and fails at ≥90 %. Verified against tonight's
exact configuration: `64G x 16` fails at 95 %, `24G x 11` passes at 25 %. Sizing
guidance and measured RSS figures are in GRID-PLAYBOOK.md.
