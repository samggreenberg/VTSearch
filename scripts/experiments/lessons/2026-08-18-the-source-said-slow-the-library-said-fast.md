# 2026-08-18 — the source said "slow path", the installed library said otherwise (#3146)

**What broke.** Issue #3146 was filed, sized and scheduled on a premise read out
of our own source: no image embedder passes `use_fast`, therefore every one of
them runs the slow PIL/numpy resize, therefore flipping `use_fast=True` is the
win. The source does say exactly that. It has not been true for some time.

`transformers` 5 **removed the `Fast` suffix on image processors.**
`SiglipImageProcessor` is now the torchvision implementation and the PIL one was
renamed `SiglipImageProcessorPil`; passing nothing already selects torchvision,
and `use_fast` is deprecated in favour of `backend=`. Probed against the
installed 5.12.1, all three of `use_fast=None`, `use_fast=True` and
`use_fast=False` returned the *same torchvision class* for `siglip`. The
proposed fix was a no-op, and the issue's whole "is the fast path numerically
safe to adopt" framing was a question about something already adopted.

**Cost.** Small, because the probe ran before any arm was built — about 40
minutes and two login-node jobs. Had it not, the study would have built four
side piles, measured `use_fast=True` against `use_fast` absent, found them
bit-identical, and reported "the fast processor is a free 3x with no drift".
That conclusion is false in both halves: there is no speedup because there is no
change, and the real perturbation is somewhere else entirely.

**The general form, and it is not "read the release notes".** A dependency
range we control spans a version where a *default flipped*:
`requirements/image-embedders.txt` pins only `transformers>=4.49`. So the same
code, same weights and same image produce different pixels depending on what a
host resolved — and neither side of that range announces itself at runtime. The
identifier `SiglipImageProcessor` did not change, gain a warning, or start
failing; it silently began meaning the other implementation.

Two consequences, and the second is worse than the study's own:

1. **A premise read from source is not a premise.** Every arm table already
   asserts its *knob* (#2877, #2897, #2905). This is the same lesson one level
   out: assert what the environment actually did with the knob, not what the
   call site suggests it should. `build_arm.py` here records the processor class
   actually loaded, the device the pixel tensor came back on, and the
   `transformers` version that decided both, and refuses to build a cell when
   any contradicts the arm table.
2. **The shared pile carries an unrecorded backend axis.** Every cell in
   `/expscratch/$USER/vts-cache` was built by whichever backend the installed
   transformers resolved that day, with nothing recording which — the same shape
   as #3160's unrecorded device axis, and *larger*: 7.8e-3 max abs in pixel
   values between backends against 1.5e-4 median 1-cos between V100 parts.

**Why the fallback made it worse.** transformers **warns and continues** when a
backend is unavailable rather than raising: asked for `backend="pil"`, DINOv3
prints "Requested pil backend is not available. Falling back to torchvision
backend" and hands back torchvision. In a sweep that log line scrolls past, and
the result is an arm that is the reference arm wearing a different label —
which reads as "the backend does not matter" rather than "the backend never
changed". Any arm whose treatment is a *request* to a library that can decline
it needs the refusal written into the builder.

**Prevented.** In this study only, so far:

- `build_arm.py` probes and refuses (above), and `check_arms.py` re-checks it
  from the provenance on disk rather than from the launcher's intent.
- `pixel_drift.py` records `backend_honoured` per row, so a fallback appears in
  the CSV as a column rather than as a surprising zero.
- `vtscore/config.py` gains `VTSEARCH_IMAGE_PROCESSOR_BACKEND`, so the backend
  can be *named* rather than resolved, and `embedder_siglip.py` stops naming a
  concrete processor class — a concrete class both pins the code to whatever
  that identifier currently means and cannot honour a backend request at all.

**Still only advice.** `preflight.sh` cannot check this generally: it does not
know which library defaults a given study depends on. The checkable piece worth
promoting later is narrower — *a pile cell should record the library versions
and resolved implementation classes that produced it*, which is what #3160's
provenance sidecar is being extended to do as a result of this incident. Until
existing cells are backfilled, the honest status of every cell in the shared
pile is "built on an unknown device by an unknown transformers".

---

**Postscript (same day): two summary statistics with the same failure shape.**
The study that came out of this incident then hit the identical problem twice
more, one level down, and both would have shipped as findings.

*Max-|Δ| saturates.* Resampling disagreements in an 8-bit pipeline land on whole
levels, so `max|Δpixel|` reads **exactly 7.843e-03 = 2/255** for any pair that
differs at all. The first backend×dispatch matrix was six identical numbers,
which reads as "these are all the same effect" and means "this statistic cannot
see the difference". Fraction-of-elements separated them (53–59% vs 13.7%).

*Jaccard hides nesting.* Replacing max with a set comparison then gave 0.231,
about to be written up as "independent axes". It was not: **99.8% of the pixels
CPU dispatch moves are also moved by the backend** — the index is low only
because one set is 4.3× larger. Containment, not Jaccard, distinguishes nested
from independent, and the two license different claims.

Both statistics were defensible in the abstract and wrong for this data. The
generalisable rule is narrow enough to act on: **before quoting a summary
statistic, ask what value it takes when the effect is maximal and when it is
minimal.** A statistic whose range is one point (the saturated max) or whose
value is dominated by set sizes rather than overlap (Jaccard on very unequal
sets) is not measuring the thing you are about to name it.

**And the figure equivalent, which is worse because it is silent.** The drift
figure could not draw a zero median on a log axis, so it substituted a small
epsilon — rendering the *exactly-zero* reproduction floor as a bar at ~5e-07,
visually comparable to the treatment's real 2.4e-06, inverting the study's main
result. The code comment said a caption would explain the substitution; no
caption can rescue a bar drawn at a value that does not exist. Zero now gets a
marker and a label. Four *other* figure edits in the same session silently
matched nothing because `ruff format` had reflowed the target between commits:
the edit reports success and the PNG regenerates byte-identical. **Only
re-reading the rendered image caught any of it.**
