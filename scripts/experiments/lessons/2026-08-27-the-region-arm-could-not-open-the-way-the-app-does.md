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

**This was not fixed everywhere at first, and deliberately so.** Sixteen other
launchers carried a DINOv3 arm and none passed `--require-text-seed`. Their
existing results are fine — they all ran before #3269, so every arm seeded from a
crop and the shift was shared. It was their *next* run that inherited this.
Sweeping all sixteen in one edit would have replaced one unexamined default with
another, so #3278 went through them one at a time; the answers did differ.
Fourteen pair the region arm, because every one of them reads a patch arm against
a whole-image control, contrasts voting modes, or compares environments — and one
of those (#2905) is a *single-arm* study whose entire purpose is a comparison
against two SigLIP environments, which is the case that shows "no contrast inside
the grid" is not the same as "no contrast". Two are not studies at all
(`launch_errdump.sh`, `launch_horizon.sh`): they re-run named cells of a
completed grid against its own `prepare_info.json`, so their arms are pinned by
that grid and renaming one would drop it from the enumeration and shift every
later index.

**What made the pins worth more than a comment.** Two of the sixteen keep an
opening the rest do not, and a pin nobody can check reads exactly like an
oversight a year later. So the opening became a *declaration* — `run_cells.py`
raises on a cell that opened differently from `CALIB_REQUIRE_OPENING`, preflight
check 14 refuses the array in either direction, and `assert_one_opening` refuses
to pool two openings at analysis time. The third one is not redundant: cells are
skipped when their CSV exists, so a resume across this fix leaves old cells
beside new ones *inside one arm*, where every count still reads N/N.

**And the rename needed a check of its own.** `array_cells` enumerates
`DATASETS x embedders x the categories PREPARE selected`, so an arm prepare never
wrote an entry for contributes **zero cells** — silently, with every later index
meaning a different cell. Most of these launchers reuse a finished study's
`prepare_info.json` to skip a GPU stage, and it is keyed by embedder name, so
`dinov3_patch` -> `siglip+dinov3_patch` is exactly the rename that lands there.
Preflight check 15 compares the configured grid against prepare's own entries;
re-running prepare for the new name reuses the cached pickle and costs no encoder
time, which is the point — the expensive thing was never the fix, it was not
knowing you needed one.

**One thing worth keeping beyond this study.** `media["embeddings"]` is already
a dict keyed by embedder name, so one media *can* carry both vectors — that is
production's three-slot design, not a harness trick. The pile happens to store
one embedder per cell pickle, so the harness opens the text half's pickle; but
it checks the media first, and will use an on-media vector the day the pile
writes one. A helper that reaches for the side file *first* would silently keep
ranking against a stale one forever after.
