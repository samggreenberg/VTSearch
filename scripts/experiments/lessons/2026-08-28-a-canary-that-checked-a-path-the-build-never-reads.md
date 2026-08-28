# 2026-08-28 — a canary that checked a path the build never reads (#3299)

**What broke.** `build_pile.py --rebuildable`, added the day before by #3300 to
prove every pile dataset can still be rebuilt from off-scratch sources, ran for
the first time against the real cluster and reported:

```
[pile]   coco_val           BROKEN   coco_val: missing /exp/scale26/datasets/external/COCO/images/val2017
[pile] REBUILD-BROKEN: coco_val: ...
```

Nothing was broken. `_load_coco` reads
`COCO_ROOT/images/**val2017.zip**` — 815 MB, present, staged in July and
untouched. `val2017/` is an extracted directory the staging area has never
held, and the README says so: *"COCO from the staged zip plus flattened
annotations."* The canary checked `pc.COCO_IMAGES`; the builder spelled the zip
path inline. Two names for one source, and the check took the one the build
never opens.

**Cost.** ~10 minutes of investigation, and it very nearly cost more: the issue
that commissioned the run said anything reporting `REBUILD-BROKEN` "is a second
purge-time landmine of the same kind as #3297 and wants its own issue". Filing
that issue would have been the reasonable-looking wrong move. The check that
settled it was two lines — read what `_load_coco` opens, then `ls` it.

**The general form.** A canary is only as good as the identity between what it
checks and what the real path does. Here the constant `COCO_IMAGES` existed and
looked authoritative, so the new code used it without asking whether the builder
did. **The builder had its own copy of the path**, which is what let the two
disagree — and neither was wrong on its own terms.

The consequence is worse than a plain bug. A false alarm costs exactly what a
true alarm costs, and it is spent on nothing; a canary that cries wolf on its
first real run is a canary people learn to skip, which is the one failure mode
that makes it worthless. So a false positive here is not a milder version of a
false negative. It is how you lose the check.

**Prevented.** `pile_config.COCO_VAL_ZIP` now names the zip once; `_load_coco`
and `rebuildable()` both go through it, and `COCO_IMAGES` is documented as the
optional extracted directory that only `box_sheets.py` wants. Three tests in
`tests_lib/core/test_pile_box_scan.py` cover the COCO branch of the canary,
which had none: a staged-zip-only tree passes, a missing zip still fails, and a
source-level assertion fails if the builder ever spells `val2017.zip` inline
again. That last one is the real guard — it pins the *identity*, not the
behaviour.

**Related, and fixed since.** `box_sheets.py` read COCO pixels from the same
non-existent `COCO_IMAGES` directory. It did not error; `path_of` simply
returned `None` for every media and the sheet came out empty. Filed as #3305 and
repaired the same day: it resolves through `pc.COCO_VAL_ZIP` (directories first,
then archives read member-wise) and now *fails* when no image source resolves,
because an empty contact sheet is the one output that looks like an answer and
contains none.
