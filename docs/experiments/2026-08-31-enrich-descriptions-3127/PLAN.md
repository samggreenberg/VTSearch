# Should `enrich_descriptions` still default to `False`? — plan (#3127)

**Pre-registered 2026-08-31, before any cell of the study was read.** The
decision rule in §5 is fixed here so that "the win does not generalise" is a
reportable finding rather than a rule rewritten around the numbers.

One number was seen first, and it is named here rather than left implicit: the
sizing cell (`esc50_s`, the cheapest cell, run to time the harness before
committing four chunks of GPU) came back at mAP 0.8950 plain — **exactly**
#3077's published `clap_general` figure for that dataset. That is a harness
check, not a result: it says the arm being measured is the arm #3077 measured.

## 1. The question

`enrich_descriptions` defaults to `False` (`vtsearch/settings_models.py`). It is a
single **global** boolean — not per-embedder, not per-media-type — so its correct
value depends on whichever embedders users actually run.

#3077 moved the audio default from `clap` to `clap_general`, and quoted an ESC-50
sweep in which enrichment *helps* `clap_general` (+0.012 to +0.016 text-sort mAP)
and *hurts* `clap` (−0.005 to −0.014). If that holds, the shipped default is now
leaving a win on the table for every audio user who never opens Settings — but a
global flip decided on one media type's numbers is how the other three get worse.

## 2. What the setting actually does

One production consumer, and it is small: `vtsearch/routes/sorting.py` reads
`settings.get_enrich_descriptions()` and passes it to `embed_text_query`, which
replaces the query vector with the L2-normalised **mean over the embedder's
`description_wrappers`** applied to the typed text (`MediaEmbedder.embed_text_enriched`).
Media vectors are untouched. Nothing else in the app reads the setting.

Two consequences the study is built on:

- **The comparison is exactly paired by construction.** One dataset load serves
  both arms, so plain and enriched differ in the query vector and in nothing
  else — same medias, same encoder, same card, same process. The confounds that
  other studies have to pair away (host arithmetic #3160, precision #3143, cache
  state) are shared here rather than controlled.
- **The simulated-user harness does not see this setting.** `run_cells.py` and
  `text_baseline.py` call `embed_text_query` without `enrich`, so every
  calibration study's opening and click-0 anchor is a *plain* query. Flipping the
  default would therefore change the app's opening without changing theirs —
  a fidelity gap to note in the report, not to fix here.

## 3. Scope: which defaults exist, and which can be affected

| media type | default embedder | text tower | wrappers | eval datasets |
|---|---|---|---|---|
| audio | `clap_general` | yes | 5 | `esc50_s/m/l` (50 queries each) |
| image | `siglip` | yes | 5 | `caltech101_s/m`, `caltech256_a`, `enrico_m/a`, `rico_screen2words_m/a`, `rvl_cdip_m/a`, `vggface2_faces_s/m`, `visual_genome_s/m` |
| text | `e5` | yes | 5 | `20newsgroups_s/m/l` (15 each) |
| video | `xclip` | yes | 5 | `ucf101_s/m/l` (10 each) |
| face | `face` | **no** | 0 | — setting is inert (no text sort at all) |
| document | *(no embedders registered)* | — | — | — |

The `vggface2_faces_*` eval datasets are **image** media type and so are scored
with `siglip`, not with the `face` embedder; they are in the image chunk.

The image row is deliberately broad. `siglip`'s wrappers are photographic ("a
photo of {text}", "a picture of {text}") and the image default has to serve
photos, faces, UI screenshots and scanned documents alike — a per-domain split is
exactly where a global setting would be expected to break, so the study is sized
to see it rather than to average over it.

**One control arm.** The audio chunk re-runs its three datasets with
`--embedder clap`. #3077's sign flip is the premise the whole issue rests on, and
a premise is asserted, not assumed (the lesson of #2877).

**Diagnostic arms, near-free.** Each cell also scores every wrapper on its own
(`w0`…`w4`), because "enrichment helps" and "one template helps" are different
findings with different fixes, and the docs note this issue asks for should say
which. The identity wrapper `{text}` is a **planted answer**: its arm must
reproduce the plain arm exactly, and the analyzer checks it.

## 4. Metrics

Primary: **average precision per (dataset, category)** for the typed query,
the same text-sort arm #3077 quoted; mAP is its mean. Secondary: P@10 (what a
user sees on the first screen) and P@5/P@20/R@k, all emitted per cell.

Δ is always **paired**: `AP(enriched) − AP(plain)` on the same dataset, same
category, same medias. There is no seed — the ranking is deterministic given the
media vectors, so the spread being reported is *across categories*, not noise.

## 5. Decision rule (pre-registered)

Let `Δ̄_m ± SE_m` be the mean paired Δ over every (dataset, category) of media
type *m*, with `SE_m` **clustered by (corpus, category)**. The S/M/L datasets are
disjoint slices of *one* corpus with the *same* categories and the *same* typed
queries, so a category's three slices are repeated measures of one question, not
three independent draws; clustering on the category-within-corpus is what stops
three slices being counted as three times the evidence. The naive
`sd/sqrt(n)` is reported beside it in every table, and it is the smaller of the
two wherever the slices agree.

**Flip `enrich_descriptions` to `True`** iff *both*:

1. **No default is harmed**: `Δ̄_m > −2·SE_m` for all four of `clap_general`,
   `siglip`, `e5`, `xclip`; and
2. **Some default is helped and the whole is positive**: at least one
   `Δ̄_m > +2·SE_m`, and the **equal-weighted** mean over the four media types is
   positive by more than 2 SE.

Media types are weighted **equally**, not by category count: ESC-50 contributes
150 paired observations and UCF-101 30, so pooling by category would let audio
decide a global default on its own.

**If any default is significantly harmed** (`Δ̄_m < −2·SE_m`): do not flip. Report
that one global boolean is being asked to do two incompatible jobs, and spin out
"make enrichment per-embedder" as its own issue, seeded with the measured
per-embedder table.

**If nothing resolves** (every `|Δ̄_m| < 2·SE_m`): keep `False`. A default flip
needs evidence; "not resolvable at this sample" is the finding.

**Cost is not expected to gate this.** Enrichment costs 5 text-encoder passes
instead of 1, on the *query* only (~milliseconds, and `embed_text_query` caches
by `(embedder, media_type, enrich, text)`), against an embedding pass over the
whole haystack that has already happened. The per-arm seconds are recorded so
that expectation is checked rather than assumed.

## 6. What this cannot settle

- The eval queries are **fixed phrasings written for the harness**, one per
  category. Enrichment rewrites the *user's* words, so a study whose queries are
  already clean, noun-phrase prompts measures enrichment under favourable
  conditions; nothing here says what it does to a two-word or misspelled query.
- Clustering handles the S/M/L repetition, but categories within one corpus
  still share an annotation scheme and a photographer; the effective sample is
  closer to "seven corpora" than to "335 image observations".
- Text-sort only. The setting has no other consumer, so a learned-sort arm would
  measure nothing that moves.

## 7. Mechanics

- Harness: [`scripts/experiments/enrich/run_enrich.py`](../../../scripts/experiments/enrich/run_enrich.py),
  launcher [`launch_enrich.sh`](../../../scripts/experiments/enrich/launch_enrich.sh).
- Four GPU chunks (one per media type) = the 4gpu_tier per-user cap, so the study
  is one wave.
- Results root: `/expscratch/$USER/enrich-3127` — one CSV per (embedder, dataset)
  cell, long-form, one row per (arm, category).
