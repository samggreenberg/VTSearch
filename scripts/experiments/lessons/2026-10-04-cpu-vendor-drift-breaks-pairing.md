# 2026-10-04 — Intel and AMD nodes run the same session differently (#4496)

**2026-10-04, #4496.** A State of the App arm was paired against an earlier run of the same code. Sessions the arm never touched, which never met its trigger, should have been byte-identical. 18 of 261 were not: they diverged late (clicks 27 to 144), mostly in the `done` phase.
- Every one of them had run its baseline on an Intel node (rack2) and its arm on the AMD node (rack7n11).
- None of the 252 Intel-to-Intel pairs diverged.

Floating-point drift between the two vendors' kernels flips a late acquisition tie, and the session follows a different path from there.

**Cost:** about 30 min chasing a "bug" in the arm's code, plus an exactness check that could not pass. The drift is mean-zero (paired F at click 150, −0.000 ± 0.000), so it only adds noise to a paired comparison.

**Fix:**
- **Pin the vendor.** Read the baseline's nodes (`sacct -j <redo job> -X -n --format=NodeList`) and pin the arm to the same vendor with `scontrol update JobId=<array> Features=intel`, which reaches the pending tasks.
- **Catch the first wave.** The first wave starts before a post-submit pin can land, so requeue any first-wave task that landed on rack7 (`scontrol requeue <id>_<task>`, then the same `Features` update).
- **Node types:** rack2n10/13/14/15 and rack4n03/04/06 are Intel; rack7n11 is AMD.

**But check capacity first.** The same night, all seven Intel nodes were fully allocated, mostly by other users. A pinned run got 46 concurrent tasks instead of 100, and two requeued arrays sat 45 min on `JobArrayTaskLimit,Priority`: requeued tasks still count against the array's throttle. The pin was lifted (`scontrol update JobId=<array> Features=`), and the drift was accepted as mean-zero noise and reported. Pin only when the baseline's vendor has room. Otherwise, report the drift: the paired mean over untouched sessions shows it.

**Status:** advice only. The launcher has no feature knob, so a pin is a post-submit `scontrol` step.
