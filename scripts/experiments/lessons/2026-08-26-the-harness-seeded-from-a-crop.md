# 2026-08-26 — the harness seeded from a crop where the app types a query (#3156)

**Study:** #3156 `vg_scale` overview, and every calibration study before it.
**Cost:** the stuck-run analysis of a 6480-cell grid, and an hour of adjudicating
two exemplars whose failure turned out to be an artefact of the seeding rather
than a property of either image.

The app's Autopilot starts on a **text sort**: the user types a query, the tool
ranks every item by cosine to it, and the first `GOOD_TARGET` (3) Good votes come
off the top of that ranking. `simulate_voting_iterations` takes that ranking as
`seed_scores`, and three separate places say so in as many words — the
`al_strategies` module docstring ("a per-item similarity to the **typed
query**"), `EVAL.md`'s Autopilot section, and `voting_iterations.py` itself.

`calibration/run_cells.py` passed something else:

```python
exemplar_id, crop_vec = _load_exemplar(ds, emb, cat, seed)   # candidates[seed % len(candidates)]
seed_scores = resolve_style(style).exemplar_sims(medias, crop_vec)
```

— cosine to a **crop of one boxed positive**. `max_patch/run_cells.py` did the
same. No user has ever produced that ranking. A dedicated builder for the real
thing, `vtscore/eval/seed_scores.py` (`build_seed_scores` → `embed_text_query`),
had existed the whole time with exactly one caller, in a different study family.

**What it cost analytically.** The whole "stuck exemplar" finding was really
*"this crop is a bad query vector"*. Two examples, both settled by wiring text
seeding and measuring the same cell:

| exemplar | crop-seeded | text-seeded |
|---|---|---|
| `knife@small` 2322075 (box is wrong — empty pavement) | stuck 13/15, `n_good`=1 every run | the crop was never the query |
| `boat@medium` 2321462 (correct label, atypical) | stuck 18/20 | ranks **4006 of 7749** for its own class |

Under `"a boat on the water"` the first positive is at **rank 1** and five land in
the top 20 — the Good phase clears immediately. The image that looked like the
hardest positive in the dataset would simply never have been shown.

**The shape.** This is not a stale pin ([both sides of the knob check were
stale](2026-08-26-both-sides-of-the-knob-check-were-stale.md)) — no knob was
wrong. It is a **parameter fed something other than what its name and three
docstrings say it holds**. Nothing errors: a crop vector and a query vector are
the same dtype and shape, the sims are real numbers, the ranking is a valid
ranking, and every downstream column is populated. The fidelity is lost in the
argument, where no type check and no knob check can see it.

**Prevented.** Crop seeding is deleted from both `run_cells.py`. Seeding is now:
text sort where the embedder has a text tower; the app's other documented start —
**three random known-good examples** — where it does not (DINOv3 has no
`embed_text`, so `embed_text_query` returns `None` and `build_seed_scores` omits
it, which is already the fallback signal). `vg_scale` gained `EvalQuery` entries:
12 texts over 36 banded cells, because the band is a property of the cell and a
user hunting a small boat still types "boat".

**And recorded.** Every row now carries `seed_mode` (`text` / `known_good`) and
`seed_query`, replacing `exemplar_id`. The root cause was not that the seeding
was wrong but that it was **unnameable after the fact** — the study's own rows
could not say how it had started. A resolved default belongs in the data.

**Still only advice:** `coco_val` and `vg_box_{small,medium,large}` have no
`EvalQuery` entries, so they now take the known-good start even on SigLIP, where
text is available and would be more faithful. Config-only to fix; the mechanism
picks it up with no code change.
