# 2026-08-27 — the region arm could not open the way the app does (#3276)

**Study:** #3156 `vg_scale` overview, on its second rerun.
**Cost:** nothing yet — caught before the array. It would have cost the
headline: the study's whole point is the voting-mode axis, and a second axis was
riding inside it.

The day before, [the harness seeded from a crop](2026-08-26-the-harness-seeded-from-a-crop.md)
was fixed: every cell now opens on a **text sort**, the way Autopilot does.
`_text_seed_scores` returns `None` when the embedder has no text tower, and
`None` is the documented signal to take the app's *other* real start — three
random known-goods. That is a correct fallback and it was written deliberately.

It is also the entire region arm. `dinov3_patch` is the only patch-capable
embedder in the pile, and DINOv3 has no text tower — `embed_text` is the base
class's `return None`. So after the fix:

| arm | opening |
|---|---|
| `siglip` × `whole_image` | text sort |
| `siglip2_l` × `whole_image` | text sort |
| `dinov3_patch` × `max_patch` | **three random known-goods** |

Both whole-image arms opened on a typed query and the region arm did not.

**The shape, and why it is worse than the bug it came from.** The crop-seeding
bug was a *shared baseline shift*: every arm seeded the same wrong way, so the
28 stamped reports could keep their within-study contrasts and only lose claims
about how a run starts. This one is **arm-dependent**. It does not cancel. Any
sentence of the form "region voting costs X less than whole-image" would have
been part voting mode and part opening, in unknown proportion — and #3267 puts
the largest measured effects in the opening.

The general form is one this repo keeps meeting: **a fallback that is correct in
isolation becomes a confound when it fires on one arm of a contrast.** Nothing
errors, nothing is mis-set, and the arm that took the fallback is the one the
study is about. The 2026-08-25 lesson
([a mode contrast that was an embedder contrast](2026-08-25-a-mode-contrast-that-was-an-embedder-contrast.md))
is the same shape with the axes swapped.

**The fix is not to disable the fallback.** A known-good start is a real user
flow and bare `dinov3_patch` should keep it. What was missing is that *the two
things an embedding space is asked for need not be the same space*: Autopilot
wants a **text tower** to open on and a **media space** to learn in, and only
the second one has to be DINOv3. Hence the paired name `siglip+dinov3_patch` —
SigLIP ranks the query, DINOv3 does the vector learning, the region learning and
the learn-sort. `seed_scores` is already just a `{media_id: similarity}` map the
caller builds, so the simulator needed no change at all.

**Controls added, rather than a paragraph:**

- A paired arm that falls back to the known-good start **raises** instead of
  running. Falling back would produce a cell identical to bare `dinov3_patch`
  while labelled as the pair — and a missing cell is visible where a
  mislabelled one is not.
- `seed_embedder` is now a **column** beside `seed_mode` and `seed_query`, for
  the reason those exist: the #3156 root cause was that how a run started was
  unnameable after the fact. A paired arm's opening lives in a different space
  than its `embedder` column implies, so the space has to be recorded.
- preflight check 14 probes the **text half's** tower (probing the arm name
  would ask the registry for an embedder called `siglip+dinov3_patch` and report
  a working pair as broken), and refuses a pair whose text pickle does not cover
  the run's medias.
- preflight check 6 resolves the region-voting premise through
  `cfg.pickle_name`, so it reads the learn half's pickle rather than a filename
  that has never existed.
- `launch_scale.sh` now passes `--require-text-seed` and
  `--require-region-voting`, so both premises are asserted before the array
  instead of read out of the rows afterwards.

**One thing worth keeping beyond this study.** `media["embeddings"]` is already
a dict keyed by embedder name, so one media *can* carry both vectors — that is
production's three-slot design, not a harness trick. The pile happens to store
one embedder per cell pickle, so the harness opens the text half's pickle; but
it checks the media first, and will use an on-media vector the day the pile
writes one. A helper that reaches for the side file *first* would silently keep
ranking against a stale one forever after.
