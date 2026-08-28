# 2026-08-27 — a repair moves more than the defect (#3281)

**Study:** #3156 `vg_scale` overview.
**Cost:** a wrong number, given to the owner while he was deciding whether to
keep a 5162-cell run: I said 27 of 36 categories were still sound. The measured
answer was **14**.

**What happened.** [The box normalised twice](2026-08-27-a-box-normalised-twice.md)
corrupted 130 boxes across 9 of the 36 `vg_scale` categories, three of them
badly. Asked what an existing run was still good for, I counted the categories
the *defect touched* and answered from that.

Then the dataset was rebuilt and I measured instead of counting:

```
UNCHANGED (14/36) — old runs still describe the live dataset
CHANGED   (22/36) — old runs describe a dataset that no longer exists
```

Six of the changed categories held **no repaired box at all**. `backpack@large`
has 11 different positives; nothing was wrong with any of them.

**Why.** A cell takes exactly 100 positives from a *ranked candidate pool*.
Repairing one band returns images to the pool and removes others from it, so
every band of that class re-selects — the repair propagates along the class, not
along the defect. The shared media pool moved too (7749 → 7747, 96 dropped and
94 added), so even the negative side of an untouched cell is not guaranteed
identical.

**The general form.** *The blast radius of a repair is not the footprint of the
defect.* The right question is not "what did this fix" but "what does this
re-derive" — any selection, ranking, sampling or banding downstream of the
repaired value moves with it, and moves for records the repair never touched.

**Prevented?** Partly, and by measurement rather than by a gate:

- `scripts/experiments/pile/diff_labels.py` reports, per category, whether the
  positive set, the evaluable set and the boxes are identical across a rebuild.
  It takes seconds and replaces the estimate with a list.
- It needs the **pre-rebuild pickle**, which only exists if someone thought to
  keep it. Copy the whole-image cell aside before rebuilding — 26 MB against the
  patch cell's 2.4 GB, and it carries the same labels and boxes.

**Still advice.** Nothing forces that copy. If a rebuild happens without one,
the comparison is unrecoverable and every prior run against that dataset has to
be treated as describing something else — which is a much more expensive answer
than a 26 MB file.
