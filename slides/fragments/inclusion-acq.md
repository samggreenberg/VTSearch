<!-- _class: full -->

![bg fit](figs/calib-acq-flow.png)

## One estimator, two thresholds

<!-- build: figs/calib-acq-flow.build1.png -->

<!-- build: figs/calib-acq-flow.build2.png -->

<!-- build: figs/calib-acq-flow.build3.png -->

<!-- build: figs/calib-acq-flow.build4.png -->

<!-- The last figure of the talk, and it closes the loop back to the fourth
     slide. Remind the room of the promise made there: the threshold decides
     twice — what you keep, and what you are asked about next. Everything since
     has been about the first job. This slide is the second.

     Advance one is the fitted estimator, unchanged, cutting at the reporting
     threshold; advance two is job one, drawn as a bracket over what comes back.

     Advance three is the turn, and it is worth stating precisely, because the
     mechanism is not what people guess. Autopilot's "hard" pick does not read
     the threshold as a decision boundary at all. It ranks the corpus
     descending, finds the first position at or below the cut, and takes the
     unlabeled item whose *index* is closest to it. A rank position, not a
     number — measured in rank space deliberately, so that the pick does not
     bias toward whichever side the scores happen to cluster on. So zoom the
     ranking, because at the scale of the whole corpus the thing about to happen
     is invisible.

     Advance four is the change: take a second cut from the same fitted
     estimator, one inclusion step *below* the reporting one. The direction is
     the opposite of the intuition from the cost weights, so say it slowly — a
     negative offset prices false alarms higher, which *raises* the cut, which
     moves it *up* the ranking, which returns *more* positives to vote on. On
     this corpus one step is about eight items in six thousand, and the figure
     draws that gap at its true size rather than a legible one.

     Then the loop closes: that vote goes back into the labelled set, the model
     retrains, and the threshold that chose the question is re-derived from the
     answer. This is why the number compounds — a better threshold does not only
     fix the final set, it changes which items get shown, which changes which
     votes get cast.

     The measured record, and give it straight because it is not a clean one.
     On COCO with SigLIP2 the interior optimum was minus three: positives found
     per hundred votes went from a median of four to eighteen, final cost fell
     from 0.137 to 0.129, and — the part that settles the obvious objection —
     average precision rose from 0.696 to 0.817. Those were not redundant
     labels; the ranking itself improved, which is a statement about the
     detector rather than about the cut. Then Visual Genome rejected minus three
     against a plus-0.01 tolerance. Only minus one passed in both, so minus one
     is what ships, and the disagreement runs along the *environment*, not the
     voting mode — the largest split is inside binary voting, which no mode gate
     could reach.

     Two things still owed, on the record: the region-voting leg is void and
     awaiting a re-run, because that run scored the acquisition pool by
     whole-image vectors while cutting on region max-pooled scores. And the
     known cost of the conservatism is that a starved environment finds six
     positives per hundred votes at minus one where it would find eighteen at
     minus three — so a supply-dependent offset, aggressive while positives are
     scarce and relaxing as they accumulate, is the open frontier, and it
     subsumes the voting-mode question entirely. -->
