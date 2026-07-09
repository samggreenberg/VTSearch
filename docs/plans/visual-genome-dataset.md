# Visual Genome demo dataset (multi-label + region annotations)

**Status:** Region-vote eval reporting, richer vocab matching, real download verification, and attributes/relationships remain open (see Open follow-ups).

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
