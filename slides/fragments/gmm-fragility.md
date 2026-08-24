### Iteration 2 — Measured

## Nearly Oracle-Close Early — And the Least Robust Choice

- Closest of any schedule to the oracle cut (gap **0.023** vs production 0.036)
- But it never learns: pure mixture hits **+0.24** cost when false positives cost 4×
- Use it early. Not alone.

<!-- Backup for iteration 2, and be fair about it before condemning it: of every
     schedule measured on region voting, the pure mixture cut sits closest to the
     oracle — mean gap 0.023 against production's 0.036 — and it was among the
     best at the shipped operating point. "Just use the mixture all the time" was
     not an obviously bad idea. -->

<!-- Then the two structural failures. It is an **inconsistent** estimator: its
     high component means "confidently scored", not "true match" — the fitted
     weight on that component was 0.35 against a true prevalence of 0.09 — and no
     number of votes ever corrects it, because it never reads one. -->

<!-- Second, that bias is asymmetric in exactly the wrong direction. Reweight the
     cost so false positives matter four times as much — not a hypothetical, just
     a user who has told the tool to be strict — and the pure mixture cut blows
     up by 0.24 in cost. Use it early, when the labels have nothing to say, and
     never on its own. -->
