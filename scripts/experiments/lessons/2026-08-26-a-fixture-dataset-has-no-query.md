# 2026-08-26 — a fixture dataset has no query, so it seeds from somewhere else (#3267)

**Study:** #3267 Good Mining. **Cost:** none — caught at design time, by reading
the seeding path before launching rather than after. It would have cost the
entire run.

[The harness seeded from a crop](2026-08-26-the-harness-seeded-from-a-crop.md)
fixed seeding for `vg_scale` and closed with:

> **Still only advice:** `coco_val` and `vg_box_{small,medium,large}` have no
> `EvalQuery` entries, so they now take the known-good start even on SigLIP,
> where text is available and would be more faithful. Config-only to fix.

The next study to touch those datasets was one whose **every arm names a
position on the seed sort**. `n5@q0.02` means "the 2nd rank percentile of the
seed sort" — and a text sort and a three-random-known-goods sort are not the same
ranking, so that cut is a cut on a different object in half the grid. Both are
real user flows; neither errors; every column is populated either way.

**Why the gap existed at all** is worth keeping. `vtscore.eval.config.EVAL_DATASETS`
is *asserted* to hold only real demo datasets, so a purpose-built fixture like
`coco_val` **cannot** live there — the fixture's query table has to be somewhere
else, and "somewhere else" defaulted to nowhere. The assertion is right; what was
missing was the second table beside it.

**Three controls, smallest first:**

- `EXPERIMENT_QUERIES` gains a COCO-80 table and the `vg_box_*` bands. The texts
  are what a **user would type**, not the raw labels: COCO's strings are terse
  and several are ambiguous alone (`mouse`, `remote`, `orange`, `tv`), where a
  bare noun ranks a different concept and the study measures the query instead
  of the opening.
- `CALIB_REQUIRE_SEED_QUERY=1` filters selection to categories that have a
  query **before** the prevalence spread is drawn, so an ineligible category is
  *replaced* rather than shrinking the grid.
- `preflight.sh --require-text-seed` refuses to launch when any selected cell
  would still take the known-good start. It checks **both halves** — the query
  text *and* whether the embedder has a text tower at all (DINOv3 does not), so
  it cannot be satisfied by adding a query to a cell that could never use one.

`seed_query_text` moved into `experiment_config` as the single implementation all
three read. It had been inlined in `run_cells.py`; two copies of a lookup is how
a preflight gate comes to pass while the run seeds differently.

**Prevented**, for any study that passes the flag. **Still only advice:** a
study that does *not* pass `--require-text-seed` still seeds however the tables
happen to fall. The flag is opt-in because the known-good start is legitimate —
it is what a DINOv3 cell must do — so this cannot become an unconditional check.

**And a smaller one, in the gate itself.** The check first reported `could not
check` on a grid it had in fact verified: loading a SigLIP text tower prints
transformers' `bos_token_id` / `eos_token_id` warnings to **stderr**, `2>&1`
folded them into the captured verdict, and the extra lines matched no `case`
branch. A gate whose own plumbing can turn a pass into a fail teaches people to
reach for `--warn-only`, which is worse than not having the gate. Keep the last
line only.
