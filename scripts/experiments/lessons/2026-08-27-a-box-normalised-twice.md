# 2026-08-27 — a box normalised twice (#3281)

**Study:** #3156 `vg_scale` overview, plus #3115's `vg_scale_any` and the #3276
run in flight.
**Cost:** every per-category and per-band claim touching `backpack@small`,
`bird@small` or `bicycle@small` in three studies, and a "region voting is worse"
reading of the #3276 region arm that was a broken box all along.

**What broke.** `corrections.json` records a reviewer's redrawn box, and that box
arrives from the app's `region_box` — already **normalised** to [0, 1]. VG's and
COCO's boxes arrive in **pixels**. The builder merged all three into one dict
and then normalised the lot on the way into the pickle, so a correction box was
divided by ~500 a second time and landed on the frame origin: 130 boxes, all
sub-pixel, all in the top-left corner.

**Why nothing caught it.** `--verify` already recomputed each box's band and
compared it against the band its cell name claims — the check written after the
last coordinate-space mistake. It passed, because **the band is derived from the
box**. Crush a box to the origin and its area lands in `small`, which is exactly
what `@small` asserts; both sides move together and the cell stays perfectly
self-consistent. A check that compares a value against something computed from
that same value cannot fail. The frame is the only fixed reference, and nothing
was comparing boxes to it.

**The band damage was bigger than the box damage.** The corrupt box does not
merely mis-pool a region vote (which only the region arm reads, which is why
this presented as a voting-mode effect). It also *files the image*: 97 of the
130 do not belong in `@small` at all once repaired — one is a macaw filling 45%
of the frame — so the small band of three classes was partly a bucket of
misfiled large objects, for **every** arm, on the one axis `vg_scale` exists to
measure.

**Prevented?** Code, four ways:

- `corrections.json` rows declare `box_space`, and `build_pile.py` refuses a row
  whose boxes contradict the declaration. The two spaces are indistinguishable
  for a box in the top-left corner of a 1×1 image, which is why inference was
  never going to work — the file has to say.
- The correction box is converted to pixels **once**, against the same `(W, H)`
  the region write divides by, so the stored box is the reviewer's box bit for
  bit rather than approximately.
- `region_geometry_problems` checks boxes against the **frame**: a sub-pixel
  side fails outright (no drawn box is one), and the share crushed into the
  top-left 1% of the frame fails as a *rate* (a real object can sit there — 1.2%
  of healthy boxes do — but not 100% of a class). It runs at build time, before
  the GPU hours, and again in `--verify`.
- `vg_scale_any` is derived from the built `vg_scale` pickle and shares its
  vectors, so it survived a parent rebuild carrying the parent's **previous**
  labels with a healthy media count. It now stamps a digest of the parent's
  labels, `--verify` compares it against the live parent, and a run that
  rebuilds `vg_scale` pulls the derived dataset in with it.

**The general form.** *A consistency check between two values derived from the
same source is not a check.* It is worth asking, of any guard already in place,
what independent thing it is comparing against — and if the answer is "itself, a
step earlier", the guard is decoration.
