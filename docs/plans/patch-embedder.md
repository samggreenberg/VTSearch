# Patch-based Image Embedder - Design

**Status:** Only the cross-cutting follow-ups and unvalidated open questions below remain open; the V3 trio and per-detector embedder-type sections below are the living design specs that frame them.

## Open follow-ups & open questions (remaining work)

Cross-cutting, still open:

- **Cross-dataset Find / CLI-chunk scoring stay on the primary vector.**
  `find._score_dataset` / `_score_with_cold_detector` (other datasets'
  `temp_medias`) and `cli._score_medias_with_detectors` (per-chunk subsets)
  never touch the active `DatasetContext`, so they build the matrix from each
  media's primary vector rather than calling `routed_embedder` /
  `keying_embedder_for_snap`. Correct for every single-embedder dataset
  (primary == score embedder); only matters once a real multi-embedder (trio)
  dataset is Find-scored. Wire a binding derived from those medias' own embedder
  names then. (Tracked from 2b.4 and re-surfaced by the per-detector work.)
- **Trio score-precedence (open question #3) not yet validated on real data.**
  The score role resolves structural ▸ patch ▸ text everywhere, but it is
  unproven whether (a) a detector should default to the structural two-stage
  pipeline when both patch and structural are bound, and (b) the diversity tree
  should use a *different* preference (patch ▸ structural) than the detector.
  Now that a patch+structural dataset is creatable, run the spike.
- **`patch_regions` / `patch_grid` / `local_features` stay singular.** The
  binding allows at most one patch and one structural embedder, so these are
  single-valued (owned by that role's embedder) rather than dict-keyed. Only
  revisit if ">1 patch (or structural) embedder per dataset" ever comes off the
  non-goals list.
- **Combine Datasets triple-match guard not added.** `combine_datasets` still
  guards only on media type, not on the `(text_embedder, patch_embedder,
  structural_embedder)` triple (a pre-existing latitude — it never checked the
  single `embedder` either). Harmless until multi-embedder datasets are common;
  add the strict triple-match refusal (v3 open question #2) then.
- **NPZ per-embedder layout (`vectors_<name>`) not added.** The `server_files`
  NPZ importer still carries a single `vectors` array; the per-embedder layout
  lands with the multi-embed path that would populate it.
Per-detector embedder-type follow-ups:

- **In-memory primary drift across A→B→A switches.** `DetectorContext.embedder`
  (the adaptive cache marker) is re-stamped to the dataset's space when the
  active dataset can't supply the detector's type. The persisted
  `embedder_type` is untouched and reloaded on next detector load, but within
  one session a multi-embedder → other-space → back sequence can leave the
  detector scoring via the *adapted* slot until reload. Exotic; revisit if it
  bites.
- **Per-detector diversity tree** and **changing a detector's type after
  creation** remain out of scope (see the per-detector "Out of scope" below).

V3 open questions (design-level, still unresolved):

1. **Where in the dataset header do the three slots live?** Today's single
   `dataset.embedder` field probably can't just be renamed without breaking
   labelset sync. Likely: keep the legacy field as a computed read-only alias to
   the score-role slot for one release, then drop it. Confirm during impl.
2. **Combine Datasets ergonomics.** Strict "embedder triple must match" is the
   v3 rule; if it bites, add a "combine on the text slot only" variant. Punt
   until real demand.
3. **Diversity-tree vs score backbone (patch vs structural).** Structural-over-
   patch for the shared score role is the less obvious call — a structural
   embedder is a deliberate specialist pick, but its Stage-1 VLAD vector may
   cluster *worse* than patch for the diversity tree. Two sub-decisions to
   validate on a real patch+structural dataset: does the detector default to
   structural two-stage when both are bound, and should the diversity tree use a
   different preference than the detector? Until then both use the single score
   precedence.
4. **Patch + structural coexistence at score time.** Storage and routing support
   binding both; the open piece is whether a single detector can ever run *both*
   visual pipelines at once (region max-pool MLP *and* geometric verify) rather
   than choosing one via score precedence. Out of scope for the first trio cut.

---

## V3 design — the text / patch / structural trio (living spec)

V3 lets a dataset bind **up to one embedder per role across three role types** — a
text-capable embedder, a patch-capable embedder, and a structural
(geometric-verification) embedder — instead of exactly one. Text sort runs against
the text embedder; region similarity / voting / region-aware MLP scoring run against
the patch embedder; instance retrieval + geometric re-rank run against the
structural embedder; all bound embedders live side-by-side in the pickle. No dataset
is forced to take more than one; a dataset leaves unused slots empty (the only hard
rule is that **at least one of the three** is set).

The point is to **stop forcing the user to choose** between "good text queries"
(SigLIP/CLIP), "good region voting + visual quality" (DINOv3 patch), and "find this
exact instance" (SIFT/VLAD structural). The three roles are type-distinct because
they ride three independent capability flags — `supports_text`,
`supports_patch_regions`, `supports_geometric_verification` — and each populates its
own per-media storage (full-image vectors / `patch_regions` + `patch_grid` /
`local_features`).

### Schema change

The on-disk per-media fields become dicts keyed by embedder name:

```python
media["embeddings"]     = {"siglip": ndarray, "dinov3_patch": ndarray, "sift_vlad": ndarray}  # fp16, L2-normalised, one entry per bound embedder
media["patch_regions"]  = [RegionVector, ...]   # the patch embedder's HAC tree (single-valued)
media["patch_grid"]     = ndarray               # (H, W, D) fp16, the patch embedder's grid
media["local_features"] = StructuralFeatures(...)  # the structural embedder's keypoints + descriptors (single-valued)
```

`media["embeddings"]` is the genuinely multi-valued field (one vector per bound
embedder, including the structural embedder's Stage-1 VLAD vector).
`patch_regions` / `patch_grid` / `local_features` stay **single-valued**: the
binding allows at most one patch and one structural embedder, so keying them by name
would be a perpetually ≤1-entry dict.

The legacy `media["embedding"]` (singular) is **dropped from the on-disk format** in
v3. Loaders that read an older pickle re-key it on the fly
(`media["embeddings"] = {legacy_embedder_name: media.pop("embedding")}`) — a one-shot
read-time migration, not a runtime compat shim. Per CLAUDE.md ("Backwards
Compatibility") we don't keep a parallel `media["embedding"]` mirror.

### Dataset binding

Three new fields on the dataset header — one per role type:

```python
dataset.text_embedder:       str | None   # e.g. "siglip" / "e5" / None
dataset.patch_embedder:      str | None   # e.g. "dinov3_patch" / None
dataset.structural_embedder: str | None   # e.g. "sift_vlad" / None
```

Constraints:

- **At least one of the three must be set** (else nothing sorts/searches). The
  create flow enforces this; per-slot type checks below don't.
- `text_embedder` must be `supports_text`; `patch_embedder` must be
  `supports_patch_regions`; `structural_embedder` must be
  `supports_geometric_verification`.
- Slots are role-typed; a single-vector embedder (e.g. `dinov2_single`) is eligible
  for **no** slot (it drives cosine sort / the detector MLP via its primary vector,
  read directly rather than through a role slot).
- One embedder may fill more than one slot if it advertises more than one
  capability. Any combination of the three slots may be filled — that's the new
  capability v3 unlocks.

`dataset.supports_text` becomes `text_embedder is not None` (and likewise for
`supports_patch_regions` / `supports_geometric_verification`). The
`MediaEmbedder.supports_*` flags stay — they describe an embedder's *capabilities*;
the dataset slots record which embedder is *bound* to which role.

### Routing rules

The shared **score embedder** (the single vector space the detector MLP, diversity
tree, and example/by-id cosine sort run against) resolves by precedence:
**`structural_embedder` if set, else `patch_embedder`, else `text_embedder`**
(`None` for a slot-less single-vector dataset, where the matrix layer reads each
media's primary vector).

| Operation | Embedder used | Behaviour when slot empty |
|---|---|---|
| Text sort (`POST /api/sort`) | `text_embedder` | HTTP 400 + `supports_text: false` |
| Cosine example sort (`POST /api/example-sort`) | score embedder (structural ▸ patch ▸ text) | HTTP 400 if all three empty |
| Region similarity / region voting / `region_box` | `patch_embedder` | UI hides Shift-drag affordance |
| Geometric (instance) verification + Stage-2 re-rank | `structural_embedder` | re-rank skipped (coarse VLAD only) |
| Diversity tree | score embedder | One tree per dataset; rebuilt when the score embedder changes |
| Detector MLP scoring | score embedder | Region max-pool when score == `patch_embedder`; two-stage verify when score == `structural_embedder` |
| Detector MLP training | same embedder as scoring | - |
| Gallery `best_region` overlay | `patch_embedder` or `structural_embedder` (whichever drove the score) | Outline absent when neither set |

The example-sort fallback chain is **only** for image uploads — text sort never
falls back to the patch/structural embedder (they have no text encoder); the
`supports_text` gate enforces this at request time.

### Detector MLP keying

An MLP is keyed by `(detector_id, dataset_id, embedder_name)`, where `embedder_name`
is the **score** embedder (structural ▸ patch ▸ text) the model was trained against
— **not** the per-media primary mirror. Consequences:

- A detector that ran against `siglip` on a v2-era dataset stays valid
  post-migration (its embedder_name is the pre-migration embedder).
- Switching the bound score embedder doesn't invalidate MLPs keyed to the old one;
  the new embedder-keyed MLP is trained fresh from the existing votes on the next
  Learned sort. Votes are embedder-agnostic (`(media_id, label, region_box?)`), so
  they re-use cleanly.
- A structural-keyed detector carries two learned objects (Stage-1 retrieval MLP on
  VLAD vectors + the match-statistic verification classifier); both re-derive from
  votes and are never persisted.

### Loader / exporter / importer impact

- **Pickle loaders** run every bound embedder during ingest; each writes its own
  vector into `media["embeddings"]`, the patch embedder additionally writes
  `patch_regions` / `patch_grid`, the structural embedder writes `local_features`.
- **`ConcurrencyGate`** (`load_pipeline.py`) gates embed work; a multi-embedder
  dataset takes proportionally longer, gated under the same `_embed_gate` limit.
- **Dataset pickle schema version** bumps; old pickles load via the read-time re-key.
- **NPZ paths-file** grows an optional `vectors_<embedder_name>` layout; the existing
  single-`vectors` layout maps to the score-role slot. (Open follow-up.)
- **Combine Datasets** requires identical
  `(text_embedder, patch_embedder, structural_embedder)` triples. (Guard is an open
  follow-up.)

### Frontend

- **Dataset-create flow**: three independent role pickers (Text / Patch /
  Structural), each defaulted None and filtered by its capability list; submission
  rejected only when all three empty.
- **Sort bar** reads `dataset.supports_text` / `supports_patch_regions`
  (+ `supports_geometric_verification`) — no per-component change.
- **Region-vote UI**: unchanged from v2; Shift-drag works whenever
  `dataset.patch_embedder` **or** `dataset.structural_embedder` is set.

### Migration

Per-dataset, one-time, automatic at first load under v3: read legacy
`dataset.embedder` + `media["embedding"]`; role-type the legacy embedder into its
matching slot(s) by capability, leaving the others `None`; re-key per-media fields;
mark the dataset as v3-schema so the next save writes the new format. There is **no
in-place "add a second embedder to an existing dataset" flow** — same rule as v1:
changing/adding an embedder requires re-import.

### Out of scope for v3

- **>1 embedder of the same role per dataset** (two text/patch/structural). The
  `embeddings` dict is forward-compatible with it, but the `str | None` binding
  rules intentionally aren't. Binding one of *each* role **is** in scope.
- **In-place add-an-embedder on a loaded dataset** (same re-import rule).
- **Cross-embedder MLP transfer** (an MLP trained against `siglip` is not reused
  against `dinov3_patch`; training restarts from the vote pile).
- **Embedding diff / freshness checks** (embedder weights are versioned by HF
  revision; re-import covers newer weights).

---

## Per-detector embedder type (living spec)

Each detector locks one embedder **type** at create time —
`semantic` / `patch_semantic` / `structural` (the buckets of
`vtscore/embedding/binding.py::embedder_type`) — and trains/scores in whichever
*concrete* embedder of that type the active dataset binds, overriding the
dataset-level score precedence (V3 Phases 2b.4/2b.5/2d) for **detector** operations.

### Motivation & model

V3 resolves the score embedder as a **dataset-level** property by precedence, so
every detector on a dataset scores in the *same* space. On a multi-embedder (trio)
dataset that's wrong: the whole point of binding more than one embedder is that
different detectors want different spaces (a "find this exact logo" detector wants
structural; a "sunset mood" detector wants text/patch). So the choice moves from the
dataset to the **detector**: each detector locks one embedder type at creation, and
training + scoring *that detector* run in *that* type's space. Text sort is
unaffected — `POST /api/sort` always runs against the dataset's text slot,
independent of the active detector.

### Design (embedder *type* — authoritative)

- **Taxonomy.** `binding.py::embedder_type(name)` classifies every embedder into
  exactly one of three buckets, precedence `structural ▸ patch_semantic ▸ semantic`:
  `structural` = `supports_geometric_verification` (sift_vlad); `patch_semantic` =
  `supports_patch_regions` (dinov2/3_patch, eupe_patch, face); `semantic` = every
  global single-vector embedder (siglip, clip, clap, e5, dinov2_single, ast, …),
  text-capable or not. The buckets partition the registry (no embedder sets more
  than one flag). `dataset_supplied_types`, `embedder_of_type`, and
  `detector_dataset_compatible` build on it.
- **Persisted form.** The detector JSON carries `embedder_type` (one of the three),
  not a concrete name. `DetectorContext.embedder_type` is the immutable lock;
  `DetectorContext.embedder` stays the adaptive concrete cache marker. Legacy
  detectors with a `primary_embedder` name are migration-read via
  `detector_embedder_type_from_data` (classify the old name → type).
- **Resolution at create.** `resolve_detector_embedder_type` accepts an **explicit**
  type as long as it's one of the three valid types — regardless of whether the
  active dataset binds it (only an unrecognized string is rejected). An **empty**
  request auto-resolves against the *types the active dataset supplies*: one type →
  auto-select; multiple → the client must choose (400 listing options); none / no
  dataset → empty (resolved at first train). The new-detector modal's **Detector
  Embedder Type** dropdown lives in an always-visible, collapsed **Advanced** section
  on both the blank and Trained tabs and offers all three types (a detector can be
  created before any dataset exists). Displayed type defaults to the dataset's
  primary supplied type (else Semantic). Picking a type the active dataset can't
  supply is allowed — the detector just gates incompatible there, with an inline
  heads-up.
- **Compatibility gate (the substantive change).** A detector is compatible with a
  dataset iff same `media_type` **and** the dataset binds an embedder of the
  detector's type. Find-label refuses an incompatible pair (409); Auto-Find silently
  skips incompatible detectors. All detector-scoped training/scoring funnels through
  one resolver, `keying_embedder_for_snap(det_ctx, snap)` (the dataset's concrete
  embedder of the detector's type, else the score precedence), so switching between
  two same-type datasets re-derives the MLP in the new space (SigLIP→CLIP;
  DinoV2→DinoV3, with region boxes re-pooled against the new grid by
  `box_to_vote_vector`). The frontend gates the same pairs via `embedder_types` on
  the dataset registry entry and `embedder_type` on the detector entry
  (`utils/context-compat.ts`). Single-embedder datasets are byte-for-byte unchanged.

**Deliberate deviation from the design.** The design said
`DetectorContext.embedder` *becomes* the authoritative per-detector primary. We
instead kept `DetectorContext.embedder` as the adaptive cache-space marker and added
a separate immutable lock (first `primary_embedder`, now `embedder_type`), so a
detector pointed at a dataset that doesn't bind its type can still re-embed its
labelset against that dataset's space, and the invalidation/re-embed machinery keys
on the *current* cache space, not the preference.

### Out of scope / open questions

- **Per-detector diversity tree.** The diversity tree is currently a dataset-level
  structure (one tree, score precedence). Whether each detector gets its own tree in
  its type's space is left open; decide when a real multi-primary dataset exists.
- **Changing a detector's type after creation.** Out of scope — same spirit as
  "re-import to change a dataset's embedder". A user who wants a different type
  creates a new detector (votes are embedder-agnostic and re-importable).
- **Multi-type single detector** (one detector scoring in two spaces at once). Out of
  scope — the inverse of the trio's "one space per detector" premise (same exclusion
  as V3 open question #4).
