# Visual Genome demo dataset (multi-label + region annotations)

**Status:** Region-vote eval reporting, richer vocab matching, and attributes/relationships remain open (see Open follow-ups).

VG is the first demo dataset with per-image **multi-label** ground truth (an image is in `man` **and** `apple` at once) and stored **bounding-box region** annotations. Every other demo dataset is single-label and pretend-disjoint (`category == target` is positive, everything else negative). Membership here is closed-world binary: a category is positive if it's in the image's annotated object set, negative otherwise (VG incompleteness → a few accepted false negatives). The vocabulary is a static hardcoded top-100 VG object list (`VISUAL_GENOME_CATEGORIES`), identical across all `visual_genome_{s,m,l,a}` slices.

## Open follow-ups

<!-- item-sep -->

- **Vocab matching quality.** Object→category matching is a case/plural-folding
  heuristic, so a category built from one spelling drops every other. **`names`
  is not the way out**: #3618 measured all 2,516,939 VG objects and every one
  carries a `names` list of length **one**, so there is no alias to read. A
  richer map has to come from `synsets`, or be measured the way `vg_scale` does
  it (`scripts/experiments/pile/name_evidence.py` — box agreement and repair
  precision per spelling, rather than string similarity).

<!-- item-sep -->

- **Attributes & relationships.** VG also ships `attributes.json` and
  `relationships.json` (e.g. "red apple", "man holding apple"). Out of scope for
  Phase 1; potential future eval axes.

<!-- item-sep -->
