### Iteration 1 — Measured

## Right in the Limit, Wild at the Start

- One bug fix took clean-vote regret from **0.010–0.016** to **~0.000**
- But the cut is a **low quantile over tens of positives**, redrawn every vote
- Cold start: "admit nothing" cuts, and budget overshoot below **~20 votes**

<!-- The first calibration study had a happy half and a structural half; give
     them in that order.

     The happy half: there was a runaway-threshold bug — the rule was pinning
     its cut to the single lowest-scoring positive it held, which with four
     positives is a coin flip — and fixing it removed essentially all the
     regret in the clean binary-voting regime. So the rule, when fed, works.
     That is worth stating plainly, because everything after this slide is
     about it not being fed.

     Then the structural half, which no bug fix touches: three deficits that do
     not decay with more of the same data. The cut is a low quantile over tens
     of positives, so its variance is dominated by the handful of rarest
     points — one unusual example moves the line. The halves are redrawn on
     every vote, so the threshold jumps from retrain to retrain even when
     nothing about the user's intent changed. And the scores the two half
     models produce have to transfer onto the scale of the model you actually
     keep. Below roughly twenty votes this shows up as degenerate "admit
     nothing" cuts, and as blowing straight past a requested budget.

     This slide is the setup for the whole talk: iteration 1 is right in the
     limit and unusable at the start, and the previous slide said the start is
     where the user lives. -->
