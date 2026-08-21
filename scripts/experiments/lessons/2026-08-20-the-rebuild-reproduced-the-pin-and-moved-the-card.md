# 2026-08-20 — the rebuild fixed one axis and silently moved another (#3160)

**Cost:** ~20 min and one wasted set of three GPU builds. Nothing shipped wrong,
because the run contained a control that was supposed to be boring.

**What broke.** #3160 shipped a pin (`ATEN_CPU_CAPABILITY=avx2`) that makes a
pile cell reproducible across hosts, and two `vg_box_*` cells were rebuilt under
it. The rebuild requested `--gres=gpu:l40s` because an L40S is 2.3x faster than
the V100 the originals used.

The control — a third band that was *already* built under AVX2 and should
therefore have come back bit-identical — came back at **5.07e-07** median 1-cos.
That is not noise and it is not drift: the same study's census had measured
L40S-vs-V100 at **5.3e-07** on this embedder. The rebuild had fixed the host axis
and moved the device axis, and the two swapped cells would have been *newly*
inconsistent with their sibling by a difference the rebuild itself introduced —
while looking entirely plausible, since nobody expects two cards to matter after
a study concludes the cards do not matter.

Re-run on the original card, the control reproduced **exactly** (100% of rows
bit-identical, eight days and one transformers version later), which is the
result the issue actually wanted.

The near-miss alongside it: the build first failed with `KeyError: 'categories'`
because the archived scan file predates a schema change. Regenerating it is the
obvious fix and the wrong one — the newer scanner filters on a different rule and
would have selected different categories, turning "rebuild the vectors" into
"redefine the dataset" with no error anywhere.

**Prevented?** *Advice only, and the advice is the shape rather than the flag.*
Two habits, both cheap:

- **Put a cell you expect to be unchanged in every rebuild**, and check it first.
  A control that is supposed to be boring is the only thing that catches a
  variable you did not know you were changing. Neither of the two cells under
  test could have revealed this.
- **A rebuild reproduces only what it holds fixed.** Pinning the axis you just
  learned about does not make a build reproducible; enumerate what the artifact
  depends on (host ISA, device, library versions, and any *derived input* like a
  cached scan) and hold or record all of it. The provenance sidecar this issue
  added lists exactly that set — it is the checklist as well as the record.
