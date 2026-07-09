# Visual Genome demo dataset (multi-label + region annotations)

**Status: Phase 1 (multi-label eval + VG ingestion) and Phase 2 (region voting in evals) shipped.** Region-vote eval reporting, richer vocab matching, and attributes/relationships remain open (see Open follow-ups).

VG is the first demo dataset with per-image **multi-label** ground truth (an image is in `man` **and** `apple` at once) and stored **bounding-box region** annotations. Every other demo dataset is single-label and pretend-disjoint (`category == target` is positive, everything else negative). Membership here is closed-world binary: a category is positive if it's in the image's annotated object set, negative otherwise (VG incompleteness → a few accepted false negatives). The vocabulary is a static hardcoded top-100 VG object list (`VISUAL_GENOME_CATEGORIES`), identical across all `visual_genome_{s,m,l,a}` slices.

## Open follow-ups

- **Region-vote eval reporting.** The boxes are consumed now; a dedicated
  side-by-side image-vs-region report/plot (baseline vs `region_voting`) is not
  yet built — callers run the eval twice (flag off/on) and diff the frames.
- **Vocab matching quality.** Object→category matching is a case/plural-folding
  heuristic. VG synonyms/synsets (`names` has multiple aliases; `synsets` exists)
  are only partially exploited; a richer synonym map would recover more positives.
- **Attributes & relationships.** VG also ships `attributes.json` and
  `relationships.json` (e.g. "red apple", "man holding apple"). Out of scope for
  Phase 1; potential future eval axes.
- **Real download verification.** The VG archives are ~15 GB; CI exercises the
  ingestion path against small fixtures only. The hardcoded URLs/sizes should be
  smoke-checked against a real download before relying on them.

## What shipped

Phase 1 — multi-label eval + VG ingestion:
- **Multi-label eval.** `vtscore/eval/labels.py::media_is_positive` — if a media carries a `categories` list, membership is set-based (`category in cats`); else legacy `category ==` compare. Routes the four eval selection sites (text-sort relevant set, learned-sort target/other split, voting-iterations vote sequence + test scoring). Existing datasets byte-for-byte unchanged.
- **Data model (additive).** VG media gain `media["categories"]` (positive categories ⊆ the 100), keep `media["category"]` (first positive, for legacy readers), and store `media["regions"]` (normalized ground-truth boxes). All live only in RAM + the dataset pickle, never in detector JSON/settings.
- **VG ingestion.** `download_visual_genome()` (two image zips + objects.json → `data/visual_genome/`), `_collect_visual_genome_files()` (objects.json → per-image positives + pixel regions, flat fractional slice over sorted image ids for S/M/L/A), `_embed_vg_images()` (embeds + stamps `categories` and pixel-÷-dims-clamped `regions`), and the `visual_genome_{s,m,l,a}` demo datasets.
- **Vocab.** `VISUAL_GENOME_CATEGORIES` (top-100) + `_vg_category_for()` case/plural-folding matcher.
- **Eval registration.** `_VISUAL_GENOME_QUERIES` + `visual_genome_{s,m}` entries in `EVAL_DATASETS`.
- **Tests.** `tests_lib/downloads/test_visual_genome_download.py` (fixture download/collect/load + region normalization) and multi-label eval tests in `tests_lib/detectors/test_eval.py`.

Phase 2 — region voting in evals (stored boxes feed the harness as simulated region votes):
- `vtscore/eval/labels.py::region_box_for_category(media, category)` returns the minimal box covering every annotated instance of the category (`None` → callers fall back to the whole-image vector = an image-level Good vote).
- **Voting-iterations** (`vtscore/eval/voting_iterations.py`): each Good vote region-pools its box via `pool_box_from_media` / `box_to_vote_vector`; Bad votes stay whole-image (matching the live detector). Scoring is independent of the `region_voting` flag — any patch dataset is scored region-aware regardless — so the flag isolates only the Good-vote training vector.
- **Learned-sort** (`vtscore/eval/runner.py::eval_learned_sort`): passes the per-Good `vote_region_boxes` map to `train_and_score`, which already region-pools/region-max-pool-scores.
- **CLI:** `python -m vtscore.eval --embedder dinov3_patch --region-voting`. Region voting needs a patch embedder (DINOv2/v3/EUPE); SigLIP (the default VG embedder) has no `patch_grid`, so `--embedder` re-embeds VG in a patch space. No embedder is hardcoded.
