# VTSearch Standard-Workflow Audit — 2026-06-04

A hands-on, end-to-end audit of the core VTSearch workflow, driven through the **live UI** (headed Chrome via the DevTools protocol) on the local dev build. Every step below was actually performed; findings come from observed behaviour, cross-checked against the browser console, the server log, and the source.

## What was exercised

The full "train here, find there" loop on **image** media (SigLIP):

1. **Create dataset A** — imported demo **Caltech-101 (S)** (412 images) → SigLIP embed.
2. **Train a detector** — created `airplane-detector` seeded from the text "airplane", then drove the **Autopilot** wizard (3 good / 4 bad initial, then boundary refinement) to 13 labels (7 good / 6 bad).
3. **Load a different dataset B** — imported demo **Caltech-101 (M)** (838 images) → SigLIP embed.
4. **Find on B with the A-trained detector** — scored all 838 items; 126 hits / 712 non-hits, airplanes ranked at the top.
5. **Export the hits to the clipboard** — Export → Categories=Good → Clipboard → Copy → "Copied 126 rows to clipboard."

**Environment:** Fedora, 3.7 GB RAM (tight); app run with `VTSEARCH_TORCH_THREADS=1` and an empty registry so only SigLIP loaded on demand (no wasteful CLAP preload). Browser console: **zero errors** across the whole session. Server log: **zero tracebacks / 5xx**. The happy path is solid; the findings below are mostly papercuts, inconsistencies, and polish.

---

## Headline

**The core workflow works and is genuinely pleasant** — text-seeded detector creation, the Autopilot labeling wizard (with smart hard-negative selection), cross-dataset detector reuse, and clipboard export all did exactly what they should, with clean console/network/server logs throughout. The improvements worth making are concentrated in: (1) the demo-dataset picker (context-blindness, an inaccessible dropdown, misleading counts), (2) the Browse-vs-projection contradiction, (3) recurring achievement toasts, and (4) export-format polish.

---

## What worked well (keep / build on)

- **Text→media seeding.** Seeding `airplane-detector` with the literal string "airplane" surfaced an actual airplane as the #1 candidate immediately. SigLIP text-image alignment makes the cold-start delightful.
- **Autopilot wizard.** The phased flow (Find Initial Goods → Find Initial Bads → Refine Boundary → Explore Diversity → Done) with live counters is a great on-ramp. Its **hard-negative selection** is smart: during "Find Initial Bads" it surfaced ferries and a helicopter — exactly the near-"airplane" shapes that sharpen the decision boundary — rather than trivially-distant negatives.
- **Cross-dataset reuse.** A detector trained on Caltech-101 (S) found airplanes cleanly in the disjoint Caltech-101 (M), top-ranked. The `origin → file → embedding → MLP` re-derivation (no persisted vectors) held up exactly as the architecture promises.
- **Load progress UX.** The dataset-load row shows count, current filename, a live progress bar, an ETA, and a Cancel button.
- **Readiness + caching.** The picker's READINESS column ("Needs Download" / "Needs Embed") is helpful, and the 2nd Caltech-101 import correctly skipped the 131 MB download (archive already on disk) and went straight to embedding.
- **Auto-persist, origins-only.** Training writes `data/detectors/airplane-detector.json` on each vote with labeled origins and **no** embeddings/MLP — matching the repo's "No Persisted Vectors" rule. No explicit Save step needed.
- **Clean signals.** No console errors, no failed requests, no server tracebacks for the entire multi-step session including two large embeds.

---

## Findings

Severity: **P1** = should fix, user-visible/contradictory · **P2** = papercut/polish · **P3** = nit. Each finding carries a unique `slug` (in its heading) to refer to it by.

### Dataset import / demo picker

**[P1 · `demo-mediatype-default`] Demo MediaType filter is context-blind — always defaults to Audio.**
Opening "Add Dataset → Demo → Downloaded Media" always lands on **Audio** (shows ESC-50/GTZAN/UrbanSound). I hit this twice: even on the *second* import, with an Image dataset **and** an Image detector already active, it still defaulted to Audio and I had to manually switch the MediaType dropdown to Image. It should default to the active context's media type (or the last-used type).
*Pointer:* `frontend/src/app/components/dashboard/dataset-importer-modal/import-config/import-config.component.*`

**[P1 · `mediatype-dropdown-a11y`] MediaType dropdown is inaccessible (and undriveable by a11y tooling).**
The MediaType selector renders its options as `<li class="media-type-option">` with no `role="option"`, no owning `role="listbox"` semantics — the options never appear in the accessibility tree (a screen reader sees an empty popup; I had to click them via JS). Contrast with the **Embedder** select in the same modal's Advanced panel, which *is* a proper accessible combobox with option roles. Two different dropdown patterns in one form; make the MediaType one match.
*Pointer:* `import-config.component.html` (search `media-type-option`).

**[P2 · `media-count-estimate`] Advertised "# MEDIA" is a precise-looking number that's ~37–40% low.**
The picker shows exact integers — Caltech-101 (S) = **300**, (M) = **600** — but the loader actually embedded **412** and **838** respectively. The count is a documented *approximation* (`vtscore/media/base.py:145` — "actual count after loading may differ"), but it's presented as an exact value with no "~", so a user budgeting time/RAM for "300" is surprised by 412 (and a ~17 min vs ~longer embed). Show "~300", a range, or compute the real per-slice count.

**[Not a bug · `importer-no-default-category`] Add-Dataset modal opens with no importer category selected.**
The modal opens to a large empty body ("Select what type of dataset to add.") until you pick Services/Server/Local/Demo.

> **Not a bug.** On the real server the default importer category is a **Service**, and none of the configured Services are available in this audit environment — so the modal correctly falls back to the empty "Select what type of dataset to add." state. The empty body is the expected behaviour when the default Service category has no entries to show, not a missing default.

**[P3 · `ucsf-docs-under-image`] "UCSF Documents" (scanned pages) sits at the top of the *Image* media-type list.**
It's a document dataset surfaced under Image (top row, sorted by item count). Either it's mis-categorized or the doc/image overlap needs a clearer label so users don't think it's photos.

### Browse / projection

**[P1 · `browse-projection-contradiction`] The import offers a Browse projection that the dashboard then refuses to open.**
The dataset-import Advanced panel shows a **"Build 2-D Browse projection now"** checkbox for image datasets (`import-advanced.component.html`), but on the dashboard the loaded image dataset's Browse/eye button is **disabled** with tooltip *"Browsing is only available for audio datasets."* This is confirmed-intentional gating (`dashboard.component.ts:792` — "Browsing currently only supports audio datasets") — so a user can pay to build a UMAP projection for an image dataset they can never browse. Resolve the contradiction: either **(a)** hide/disable the "Build projection" checkbox for media types Browse can't open, or **(b)** lift the audio-only gate (the projection + hex-tile pyramid is embedding-based and media-agnostic; `tests/projection` already covers it). The two surfaces currently disagree.

### Achievements / toasts

**[P2 · `achievement-toast-replay`] Achievement unlock toasts re-fire on every navigation (and after votes). — FIXED**
The three "Bronze: …" toasts ("Detectors Trained", "Days Active", "Media Types Touched") reappeared on dashboard load, again on entering the `/label` view, and again after voting — i.e. 3–4 times in the first minute of the standard flow. While visible they stack top-center and **overlap the right-hand Labels panel**, then auto-dismiss after a few seconds. Root cause: `AchievementsService.refresh()` re-emitted *every* `pending_announcement` on *every* call, and `refresh()` fires after votes, finds, and navigation. The server keeps a milestone in `pending_announcements` until the user opens the panel (which ACKs it), so each refresh re-popped a toast for an already-shown unlock; the toast `dedupKey` only suppresses duplicates while a toast is still on-screen, so anything past the 5 s auto-dismiss re-fired. Fix: `AchievementsService` now tracks emitted milestones in a session-scoped `Set` (`categoryId:tierIdx`) and pushes each to the `unlock$` stream at most once, so the toast fires once per real unlock while the notification dot stays server-driven (cleared when the panel is opened).

### Training / labeling

**[P3 · `bads-phase-focus`] During "Find Initial Bads", keyboard focus sits on the *Good* button.**
The phase's primary action is **Bad**, but focus defaults to **Good** — pressing Enter would mislabel. Default focus should follow the phase's expected action (or be on neither button).

### Find results + export

**[P2 · `goods-overreport`] "Goods (126)" over-reports relative to ground truth.**
Find labeled 126 items "good", but the Caltech-101 (M) airplanes category only holds ~24 true airplanes; the auto threshold leans toward recall, so the down-list goods include near-airplane shapes. Reasonable behaviour, but "Goods (126)" reads as 126 confident hits. Surface the score/threshold (or a confidence band) in the hits header so users calibrate trust.

**[P2 · `duplicate-clipboard-export`] Two separate clipboard-export implementations.**
`export-modal.component` (used here, from the right-panel Export) and `autodetect-results-modal.component` each implement their own clipboard copy with **different** column models and separator vocabularies (`copyColumn`/`copySeparator` = `origin+name`/`newline` in one; column checkboxes + Comma/Tab/Pipe/Semicolon radios in the other). Worth unifying onto one component to avoid drift.

**[P3 · `export-header-casing`] Export column headers mix Title Case with raw field keys.**
Columns are `Label, MD5, Filename, Category, Dimensions, File Size, clipper, name` — the last two leak lowercase internal keys. Re-label "clipper" → "Clipper" and "name" → "Source"/"Origin".

**[P3 · `export-name-ambiguous`] The "name" column is ambiguous next to "Filename".**
The column titled **name** contains the demo origin id (`caltech101_m`), which reads like it should be the item's name/filename — confusing right beside the actual "Filename" column.

**[P3 · `export-filesize-units`] "File Size" exports raw bytes while the UI shows KB.**
Export rows carry `8165`, `10423` (bytes); the focus-view metadata panel shows the same field as `5.7 KB`. Pick one, offer both, or at least title the column "File Size (bytes)".

**[P3 · `export-default-all`] Find-results export defaults Categories to "All".**
Arriving from a Find specifically to grab hits, the export defaults to All (838 rows = hits + misses); the user then switches to Good. Defaulting to Good (or remembering the last choice) for a Find-originated export saves a step.

### Global / cosmetic

**[P3 · `mailto-typo`] Logo "Email us" link has a typo and no recipient.**
`frontend/src/app/app.component.html:97`: the `mailto:` subject hard-codes the word "Issue" misspelled with a tripled "s", and the `mailto:` has an empty to-address, so "Email us" opens a blank-recipient compose window.

**[P3 · `header-data-lag`] Header "Data:" label lags the loaded/active dataset.**
After loading Caltech-101 (S) it stayed "Data: Select a dataset" until I entered a view, even though the New-Detector modal correctly knew the active media type was Image. The dashboard's row-checkbox selection and the header's "active dataset" are two separate notions of "selected", which can read as out-of-sync.

**[P3 · `no-access-log`] No server-side request/activity logging in local mode.**
The dev server log contains only the 6 boot lines — no access logs, no progress lines — for an entire session that downloaded, embedded ~1250 images, ran Find, and exported. Fine if intentional, but it makes "what did the server just do?" debugging harder; a quiet-by-default access log (or a `--verbose`) would help.

---

## Prioritized recommendations

1. **Fix the Browse/projection contradiction** (`browse-projection-contradiction`, P1) — pick (a) hide the checkbox or (b) enable image browse. Users will hit this immediately.
2. **Make the demo picker context-aware** (`demo-mediatype-default`, P1) — default MediaType to the active/last-used type; stop forcing Audio.
3. **Give the MediaType dropdown real listbox/option a11y** (`mediatype-dropdown-a11y`, P1) — align it with the Embedder combobox pattern.
4. **De-dupe achievement unlock toasts** (`achievement-toast-replay`, P2) — fire once per real unlock; don't replay on every navigation; don't occlude the Labels panel.
5. **Signal that "# MEDIA" is an estimate** (`media-count-estimate`, P2) — "~N" or a range; or compute the true per-slice count.
6. **Export polish** (`duplicate-clipboard-export`, `export-header-casing`, `export-name-ambiguous`, `export-filesize-units`, `export-default-all`, P2/P3) — unify the two clipboard exporters, fix column labels ("clipper"/"name"), reconcile File Size units, default Find exports to Good.
7. **Tiny wins** (`mailto-typo`, `bads-phase-focus`, P3) — fix the misspelled mailto subject (tripled "s") + add a recipient; set sensible default focus in the Bads phase. (`importer-no-default-category` was investigated and is **not a bug** — the empty state is the correct fallback when the default Service category has no entries.)

## Reproduction notes

- App: `VTSEARCH_TORCH_THREADS=1 vtsearch-venv/bin/python app.py --local` (empty `data/dataset_registry.json` → no CLAP preload; SigLIP loads on first image import).
- Driven headed via the chrome-devtools MCP on `DISPLAY=:0`.
- Two large CPU embeds (412 and 838 images at one torch thread) dominated wall-clock (~19 min and ~longer); everything else was fast.
- Clipboard write succeeded under the localhost secure context; it could not be byte-verified from the shell (Chrome on X11, only `wl-paste`/Wayland installed) — the app's "Copied 126 rows to clipboard." status is the confirmation, and the in-modal preview confirmed the 126 good/airplane rows.
