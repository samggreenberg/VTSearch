# Visual Genome demo dataset (multi-label + region annotations)

**Status:** Region-vote eval reporting, richer vocab matching, and attributes/relationships remain open (see Open follow-ups). The hardcoded download URLs/sizes were smoke-checked against a real download (2026-07-14): all three archives are live and their size estimates now match the servers' Content-Length.

VG is the first demo dataset with per-image **multi-label** ground truth (an image is in `man` **and** `apple` at once) and stored **bounding-box region** annotations. Every other demo dataset is single-label and pretend-disjoint (`category == target` is positive, everything else negative). Membership here is closed-world binary: a category is positive if it's in the image's annotated object set, negative otherwise (VG incompleteness → a few accepted false negatives). The vocabulary is a static hardcoded top-100 VG object list (`VISUAL_GENOME_CATEGORIES`), identical across all `visual_genome_{s,m,l,a}` slices.

## Open follow-ups

- [ ] #2387 — Build a side-by-side image-vs-region eval report (baseline vs `region_voting`)
- **Vocab matching quality.** Object→category matching is a case/plural-folding
  heuristic. VG synonyms/synsets (`names` has multiple aliases; `synsets` exists)
  are only partially exploited; a richer synonym map would recover more positives.
- **Attributes & relationships.** VG also ships `attributes.json` and
  `relationships.json` (e.g. "red apple", "man holding apple"). Out of scope for
  Phase 1; potential future eval axes.
- [x] #2388 — Smoke-check the hardcoded VG download URLs/sizes against a real download (done 2026-07-14; sizes corrected)
