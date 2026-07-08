# VTSearch documentation audit — 2026-07-07 (branch `dev`)

Seven reviewers audited 17,483 lines across `README.md`, `CHANGELOG.md`, and `docs/**`, cross-checking each claim against the live code on `origin/dev`. Findings are grouped by severity + audience, most actionable first.

---

## TL;DR

The docs are broadly well-structured — the CLAUDE.md → ARCHITECTURE.md → sub-doc hand-off is real and mostly correct. But there is a **cluster of concrete correctness bugs that will actively mislead new users, plugin authors, and API clients today**, plus a docs-layer bloat problem (CLAUDE.md, DEPLOYMENT.md, plans/ tree, HANDOFF.md) that's growing without pruning.

- **10 correctness bugs** that misinform the reader (a couple make working clients hang, a couple send plugin authors down a dead code path).
- **~1,300 lines of duplication** across README ↔ SETUP ↔ DEPLOYMENT ↔ CLI ↔ HANDOFF and inside `docs/plans/`.
- **`docs/plans/README.md` should be trimmed to policy-only** and stop enumerating the plan files (an index that has to stay in sync with ~40 siblings is more maintenance liability than `ls docs/plans/` is worth). The "5 missing plans" finding is evidence *of* the maintenance cost, not a call to top up the list.
- **`docs/reviews/` is still doing useful work** — every review has open findings. Keep, upgrade README.
- **CLAUDE.md is right-sized for its audience.** Every section is a trip-wire written by Claude after a costly Claude mistake; the cost is per-turn tokens but the benefit is avoided rework. Verdict: **keep as-is**, only apply the small factual fixes elsewhere in this report.

---

## Section 1 — Concrete correctness bugs (fix these first)

Ranked by "how badly does this mislead the reader today?"

### S1.1 API docs describe endpoints that don't behave as documented

1. **`docs/api/medias.md` documents `POST /api/learned-sort` as synchronous — it is async.** Doc lines 229-237 promise `{"results": [...], "threshold": ...}`. Reality (`vtsearch/routes/sorting.py:270-367`) returns `{"job_id": ..., "status": "running"}` by default; the sync body only comes back when the request carries `{"wait": true}` (test-only). Companion `GET /api/learned-sort/result` and `POST /api/learned-sort/cancel/{job_id}` are undocumented. **Any client written from this doc will hang waiting for `results` that never come.**
2. **`docs/api/detectors.md` shows the wrong response shape for import-labels.** Doc line 99 shows `{success, applied, skipped, num_labels, message}`. Actual (`vtsearch/routes/detectors/labels.py:310-319`): `{applied, skipped, resolved, trained, num_labels, message}` — no `success`, plus two undocumented fields (`resolved`, `trained`).
3. **`docs/api/auth.md` quotes wrong 400 message text.** Lines 37 & 48 say `"Login not supported by the active provider"` / `"Logout not supported by the active provider"`. `vtsearch/routes/auth.py:26` actually raises `"Login/logout not supported by the active provider"` for both. Clients matching on the string will not match.
4. **`docs/api/dashboard.md` is misgrouped.** Bundles unrelated surfaces (`/api/dashboard/*` renaming + `/api/find*` multi-dataset Find) and misses `POST /api/find/cancel`, `GET /api/find/stats`, `POST /api/find/corrections-to-detector`, `POST /api/auto-detect`, `GET /api/dashboard/disk-usage`, `GET /api/dashboard/ram-usage` — all present in code.

### S1.2 Extension docs will send plugin authors down a dead code path

5. **`docs/EXTENDING-plugins.md` shows Settings-Source and Labelset-Source authors overriding the wrong methods.** Lines 1279-1298 (settings) and 1392-1404 (labelset) show `def load(...)` and `def save(...)`. `SyncSource` explicitly says *"Subclasses override `_do_load` / `_do_save`, not this method"* (`vtscore/sync/__init__.py:52-69`). A subclass following the doc verbatim will silently never be called. **The same bug is repeated in `docs/EXTENDING.md` lines 166-167 and 178-179 checklists.**
6. **`docs/EXTENDING-plugins.md` FieldType table is incomplete.** Line 46 lists 8 field types. `vtscore/plugins/__init__.py:44-55` defines 10; `"number"` and `"checkbox"` are missing from the docs and are used by real plugins.
7. **`docs/EXTENDING-media.md` documents a phantom property `legacy_bytes_keys`** (line 205). Zero hits in `vtscore/`. Silently removed or never existed. Same false claim propagated into `vtscore/docs/extending/media-types.md:90`.
8. **`docs/EXTENDING-media.md` `load_media_data` signature is out of date.** Says `(Path) -> dict`; actual (`vtscore/media/base.py:424`) is `(self, file_path: Path, media_bytes: bytes | None = None) -> dict`. The kwarg matters — the folder loader passes it to avoid a second disk read.

### S1.3 User- and ops-facing values contradict the code

9. **`docs/ML.md` contradicts itself on hidden-layer sizing.** Line 14 says "8-32 neurons" (correct). Line 27's Training Configuration table says "4-32 neurons | `max(4, min(32, n_train // 3))`" — wrong. `vtscore/config.py:191` sets `MLP_HIDDEN_MIN = 8` and `vtscore/training/mlp.py:33` uses that constant.
10. **`docs/DEPLOYMENT.md` reports wrong default for `calibrate_count`.** Example config line 370 shows `"calibrate_count": 2`. Real default is `1` (`vtscore/config.py:190`, `vtsearch/settings_models.py:199`). CLI.md line 51 agrees on 1. Only the eval CLI defaults to 2 (called out in EVAL.md — that half is fine).
11. **`docs/DEPLOYMENT.md` disagrees with itself on model-directory size.** Line 188 says "**Total: ~3.8 GB** for the five models". Lines 331 and 497 both say `data/models/` is "~3.2 GB". Pick one.
12. **`docs/DEPLOYMENT.md` data-tree omits CLIP.** Lines 329-344 enumerate 4 model dirs; `scripts/download_models.sh:55` fetches a 5th (`openai/clip-vit-base-patch32`). Include or drop the enumeration.
13. **`docs/DEPLOYMENT.md` TOC entry L15 "Requirements file structure" doesn't match the actual heading (L541 "Dependency structure").** Anchor breaks.
14. **`docs/ARCHITECTURE.md` L189 references `vtsearch/settings_factory.py` — does not exist.** Actual module is `vtsearch/settings_store.py` (imported at `vtsearch/settings.py:38`). Same file also omits `threading.py` and `routes/auth_huggingface.py` from the map.
15. **`docs/ARCHITECTURE.md` audio and image embedder inventories are stale.** L71 audio list omits `embedder_paraspeechclap.py`; L75-76 image list mis-splits SigLIP/SigLIP2/CLIP as "single + patch variants" (they aren't) and omits `embedder_face.py` and `embedder_sift_vlad.py`.
16. **`docs/USER_GUIDE.md` L604 says the Settings modal has "seven tabs" — it has eight.** The list at L606-620 omits **HuggingFace**. Confirmed in `frontend/src/app/components/modals/settings-modal/settings-modal.component.html` L63-70.
17. **`docs/SETUP.md` "full test group list" at L612-613 is not full.** Omits `projection` and `frontend` (`gpu` and `vtscore-clean` are separately called out but at least deserve mention).
18. **`docs/HANDOFF.md` L67 links `design/cli-detector-converter.md` — the `docs/design/` directory does not exist.** Dead link.

### S1.4 Minor factual drift (still worth fixing)

19. `docs/EXTENDING-media.md` embedder-inventory table is out of sync with the code tree (missing dinov3, eupe, and the `_single`/`_patch` split).
20. `docs/style-guide.md` §1.8 z-index table omits `--z-offline-banner: 4500` (in `frontend/src/scss/_variables.scss:73`).
21. `docs/demos.md` L122 oversimplifies ("Video demos are downloaded from HuggingFace Datasets") — some come from HuggingFace, others from `serre-lab.clps.brown.edu` and `csc.kth.se`.
22. `docs/CLI.md` L301 anchor `DEPLOYMENT.md#tuning` is fragile (relies on GH heading heuristic).
23. `docs/EXTENDING-plugins.md` L1076 says `POST /api/detectors` for creating a detector in the registry context; the registry endpoint is `POST /api/detectors/registry`.

---

## Section 2 — Global coverage gaps (docs simply don't cover live behavior)

### G2.1 The multi-dataset / multi-detector context model is essentially undocumented in `docs/api/**`
`X-Dataset-Id` and `X-Detector-Id` request headers are load-bearing (`vtsearch/routes/_shared.py:118-163` `require_dataset_header` / `require_detector_header` decorators 400 without them; `vtsearch/shim/__init__.py:39-113` picks the active context from them). Every vote / media / sort / labelset endpoint uses them. The only stray mention is a payload example in `events.md:71-77`. **This is the single biggest hole.** Fix by adding a "Context headers" subsection to `docs/API.md` and referencing it from every per-endpoint file.

### G2.2 Auth is under-documented
`docs/api/auth.md` covers 3 endpoints. It never mentions HuggingFace OAuth (`vtsearch/routes/auth_huggingface.py:96-180`: `/api/auth/huggingface/{status,login,callback,logout}`), never explains `DefaultLoginProvider` vs `TrivialLoginProvider` vs HF, and never says the `login_required` flag in `auth/status` is what the SPA switches on.

### G2.3 Endpoints exist in code but nowhere in docs
- **datasets.md**: 7 undocumented endpoints (`import-local-folder`, `import-local-files`, `promote`, `import/{importer_name}/options`, `detect-media-type`, `registry/{id}/preload-embedder`, `registry/{id}/diversity-tree`).
- **detectors.md**: `portable-bundle`, `labels-detail`, `registry/{id}/readers`, three labels-side preview/thumbnail routes.
- **medias.md**: `server-media-files/from-media-id`, `server-media-files/{filename}/thumbnail`, plus the async `learned-sort/result` and `learned-sort/cancel/{job_id}` mentioned above.

### G2.4 Onboarding gaps
- `README.md` has no 3-line "install → build frontend → run" quick-start; new users have to hop to `SETUP.md` before they can even try.
- `SETUP.md` has no troubleshooting: what happens if `pip install` errors, port 5000 is busy, the frontend build fails, or `nvidia-smi` reports driver mismatch.
- `USER_GUIDE.md` mentions the achievement-phrase mechanic but never says what happens when a phrase unlocks. And documents "two ways" to import pre-trained detectors but never closes the loop on how one gets a detector `.json` into the registry from a file.
- `demos.md` lacks approximate download sizes for most rows (only `visual_genome_a` gets one) and no callout for gated (HuggingFace-token-required) datasets.

---

## Section 3 — Structural / organizational issues

### O3.1 CLAUDE.md — **retracted; no changes recommended**
An earlier pass here proposed slimming ~110 lines by moving Test Isolation, Avoiding Flaky Tests, Nested-modal back buttons, Versioning trip-wire, and CLI-example commands out to linked docs. That framing was wrong. CLAUDE.md is not a reader-facing reference — it's a set of trip-wires written for Claude after concrete costly Claude mistakes, and its every-turn cost is measured in tokens, not scroll fatigue. The right test is "does this rule prevent a mistake whose cost exceeds a few hundred tokens per turn?":

- **Test Isolation** and **Avoiding Flaky Tests** repeatedly *are* the mistake — a Claude that lands per-file autouse fixtures or a `for i in range(100)` cancellable loop wastes an entire test-fix cycle. Keep.
- **Nested-modal back buttons** — Back vs Cancel is the exact bug Claude has to be talked out of; keep.
- **Versioning trip-wire** — a single sentence stopping a Claude from adding a hand-bumped `VERSION` constant, which would immediately cause parallel-branch collisions. Keep.
- **CLI-example commands** — Claude needs to actually run these; kicking them to another file adds a hop, not clarity. Keep.

Verdict: **keep CLAUDE.md as-is.** No sections to move.

### O3.2 `docs/HANDOFF.md` is 70 % duplication
Its "Quick start" (L71-98), "Running the test suite" + "Test markers" + linting (L207-263) all restate CLAUDE.md. Its "Codebase orientation" table (L179-190) and "Known constraints" (L343-370) restate ARCHITECTURE.md. **Verdict: restructure** — trim to a doc-map + "what VTSearch is" preface; delete the dead `design/cli-detector-converter.md` link; let CLAUDE.md and ARCHITECTURE.md own the rest.

### O3.3 `docs/DEPLOYMENT.md` (952 lines) is three docs jammed together
1. Deployment/operations (L1-540) — coherent, ~540 lines.
2. Dependency layout (L541-589) — fine to keep with (1).
3. **GPU install troubleshooting** (L640-898) — ~260 lines of RHEL/EPEL/DKMS/cuML runbooks. Audience is people debugging `scripts/install.sh`, not operators. Split into `docs/GPU-INSTALL.md`.
4. cuML/nvrtc runbook (L899-952) — dev-tier, add to GPU-INSTALL.

### O3.4 README.md and SETUP.md leak dev-only content
- `README.md` L62-84 duplicates ARCHITECTURE.md's project tree; L96-113 duplicates ML.md/EVAL.md; L115-122 is an inline link list to `EXTENDING-*.md`. All of that could live one hop away.
- `SETUP.md` embeds a 190-line SLURM section (L407-595) that dwarfs first-run material. Belongs in DEPLOYMENT or `docs/SLURM.md`. Env-var table (L642-656) duplicates DEPLOYMENT.md's env-var reference that L657 itself points to.
- `CLI.md` "Web server modes" (L251-401) heavily overlaps DEPLOYMENT.md — same VTSEARCH_SERVER_INIT / VTSEARCH_BIND / gunicorn commands. Link, don't restate.

### O3.5 `docs/EXTENDING.md` checklists are pure duplication
L122-279 restate each sub-doc's "What to implement" list adding no information. Collapse to a one-line "see X" table.

### O3.6 MediaConverter section is duplicated across `EXTENDING-plugins.md` L718-798 and `EXTENDING-media.md` L643-757
Two full examples, same interface, both 80-115 lines. Even `EXTENDING.md`'s own checklist (L246) points only to the media.md version. Delete the plugins.md copy.

### O3.7 The parallel plugin doc tree is undocumented
`vtscore/docs/extending/` holds 9 files that mirror `docs/EXTENDING-*.md`. Neither tree cross-links; plugin authors have no idea which is authoritative. Same `legacy_bytes_keys` phantom-property bug lives in both. Either merge the two trees or explicitly scope one as "for external vtscore consumers only."

### O3.8 `docs/vtscore-api.md` is doing two jobs badly
Header markets it as "canonical public-API inventory"; body is a Phase-1/2/3 refactor plan with `[SEAM]` markers and "NOT IMPLEMENTED" callouts. A library consumer can't tell what's callable today. Split: a lean `vtscore-api.md` that documents only what ships now, and move the seams/phases into a `docs/plans/vtscore-carveout.md`.

---

## Section 4 — `docs/plans/` hygiene

### `docs/plans/README.md` should be trimmed to policy-only
Five files (`detector-standalone-export.md`, `docs-audit-2026-06-28.md`, `gpu-acceleration.md`, `huggingface-oauth-gated-datasets.md`, `progress-weight-calibration.md`) exist on disk but aren't in the README's enumeration. That's ~14% drift and it will keep happening — an index that shadows N sibling files needs a manual edit every time a plan is added, and `ls docs/plans/` gives you the same information for free (the filenames are already self-describing: `patch-embedder`, `vtsbrowse`, `zoneless-migration`, …). The right shape is **policy-only**: what belongs in `docs/plans/` vs `docs/reviews/`, how status headers work, when to fold a shipped plan into a `*-shipped-log.md` sibling, and the "Open follow-ups" convention. Drop the enumeration entirely. A per-plan status column would compound the same problem: metadata that has to stay in sync with every file in the directory.

### VTSBrowse cluster fragmentation
Five files totalling ~1,517 lines cover one feature: `vtsbrowse.md` (856) + `vtsbrowse-empirical-tuning.md` (267) + `vtsbrowse-toponymy.md` (260) + `vtsbrowser-hex-circle-radius.md` (136) + `vtsbrowser-qa-followups.md` (68). Naming drift too (`vtsbrowse-*` vs `vtsbrowser-*`). Fold the two small `vtsbrowser-*` follow-up files into a shared "VTSBrowse follow-ups" child of the parent.

### `docs/plans/` vs `docs/reviews/` blurred
Dated single-shot audits sit in `plans/` where they don't belong per the README's own note ("Point-in-time UI review/audit reports live in `../reviews/`"):
- `logical-bug-audit.md` (1,191 lines, fully resolved) — move to `reviews/`.
- `docs-audit-2026-06-28.md` (327 lines, shipped) — move to `reviews/` (and note: **this current audit report is that same shape and should also land there**).
- `code-structure-review.md` (225 lines) — boundary case; part is a review, part is ongoing themes. Split.

### `scalability.md` and `scalability-plan.md` overlap
527 + 723 lines, cross-referencing each other, sharing `S#` IDs. Collapse to one plan with a glossary section; fold `cli-stream-massive-images.md` (68 lines, already flagged as a scalability child) in with a clear cross-link.

### Enormous plans still carrying shipped-work sections
`patch-embedder.md` (1,340), `zoneless-migration.md` (1,337), `logical-bug-audit.md` (1,191), `vtsbrowse.md` (856), `scalability-plan.md` (723), `structural-embedder.md` (650). Most got large because nobody removed shipped sections. Suggested pattern (already used by `logical-bug-audit.md`): collapse shipped items to a struck-through one-liner + PR ref; split archived close-outs into a sibling `*-shipped-log.md` when the living spec is what remains.

---

## Section 5 — `docs/reviews/` (5 dated audits, 1-6 weeks old)

Keep the directory. Every file has at least one open finding that would help a Claude/dev landing in that area:
- **2026-05-27-style-audit.md** — V1/V2/V3 (`.tab-bar`/`.help-tabs`/`.settings-tabs` shims) and V14 (hardcoded 480 px) still live. **Keep.**
- **2026-05-28-edge-states.md** — B1 (autopilot demands 3 goods on 1-item dataset), B2 (200-char rename FS error), V5 (toast leaks absolute paths), U1 ("Select a dataset" literal) all still live. **Keep.**
- **2026-05-28-e2e-flows.md** — O1 stale-request storm, U8/U10/U13 all unresolved. **Keep**; consider promoting U10 into `docs/plans/browser-vision-testing.md` follow-ups.
- **2026-05-28-longops.md** — O11 15-20 s frozen cancel, O8 gate contradiction, O3/O4/O5 progress cosmetics untouched. **Keep.**
- **2026-06-04-standard-workflow.md** — 5 findings fixed in-file; `mailto-typo` half-fixed (typo corrected but recipient still empty); `mediatype-dropdown-a11y` still open. **Keep.**

Directory-level: **leave the README as-is** (a short "what this dir is for" preface is the right shape; do not add an "areas touched" index — it would rot the same way `plans/README.md`'s enumeration does, and `ls` + the self-describing filenames already give you the shortlist). **Migrate two items into `docs/plans/`** (U10 into `browser-vision-testing.md`; O8 into scalability or CLAUDE.md), per the "follow-ups belong in plan file" policy.

---

## Section 6 — Concrete edits queued (proposed batching)

Grouped so a follow-up PR can land coherently.

### Batch A — Correctness bugs (one PR, ~50 line diff)
- `docs/ML.md` L27 4-32 → 8-32.
- `docs/DEPLOYMENT.md` L370 `calibrate_count` 2 → 1; reconcile L188 vs L331/L497 model-dir size; add CLIP to L329-344 tree; fix L15 TOC entry.
- `docs/ARCHITECTURE.md` L189 `settings_factory.py` → `settings_store.py`; add `threading.py` and `auth_huggingface.py` to map; update audio and image embedder inventory.
- `docs/USER_GUIDE.md` L604 "seven" → "eight"; add HuggingFace to L606-620.
- `docs/SETUP.md` L612-613 add `projection` and `frontend` to test-group list.
- `docs/HANDOFF.md` L67 remove or repoint dead `design/cli-detector-converter.md` link.
- `docs/EXTENDING-media.md` remove phantom `legacy_bytes_keys` (L205); fix `load_media_data` signature (L196); refresh embedder table.
- `docs/EXTENDING-plugins.md` FieldType table L46 add `number`, `checkbox`; SettingsSource + LabelsetSource examples override `_do_load` / `_do_save`; fix registry endpoint typo L1076.
- `docs/EXTENDING.md` L166-179 checklists say `_do_load` / `_do_save`.
- `docs/api/auth.md` quote real "Login/logout not supported…" string.
- `docs/api/medias.md` rewrite `POST /api/learned-sort` as async by default; document `/result` and `/cancel/{id}`.
- `docs/api/detectors.md` fix import-labels response shape.
- `docs/style-guide.md` add `--z-offline-banner: 4500` row.
- `docs/demos.md` correct the "HuggingFace" video-source claim.

### Batch B — Global gaps (one PR)
- Add a **"Context headers"** section to `docs/API.md` documenting `X-Dataset-Id` and `X-Detector-Id`; reference from each `api/*.md`.
- Rewrite `docs/api/auth.md` to cover HuggingFace OAuth + login provider variants.
- Add missing endpoints to `datasets.md`, `detectors.md`, `medias.md`, `dashboard.md` (see G2.3).
- Split `dashboard.md` into `dashboard.md` (info/rename/usage) + `find.md` (all `/api/find*` + `/api/auto-detect` + `/api/find-label`).

### Batch C — Trim and re-home (one PR)
- Split `docs/DEPLOYMENT.md` GPU/cuML sections into `docs/GPU-INSTALL.md`.
- Trim `docs/HANDOFF.md` to a doc-map.
- Move SLURM section from `docs/SETUP.md` to `docs/DEPLOYMENT.md` or `docs/SLURM.md`.
- Trim README.md project-structure / ML / Extending blocks to link stubs; add 3-line quick start.
- Trim CLI.md "Web server modes" to a link into DEPLOYMENT.md.
- Delete duplicated MediaConverter section from `EXTENDING-plugins.md`.

### Batch D — Plans / reviews hygiene (one PR)
- Trim `docs/plans/README.md` to policy-only (plans-vs-reviews boundary, status-header convention, shipped-log fold pattern, "Open follow-ups" rule); delete the enumeration and don't replace it with a status column.
- Move `logical-bug-audit.md` and `docs-audit-2026-06-28.md` from `plans/` to `reviews/`.
- Fold `vtsbrowser-hex-circle-radius.md` and `vtsbrowser-qa-followups.md` into `vtsbrowse.md` or a shared VTSBrowse follow-ups doc; unify `vtsbrowse-*` vs `vtsbrowser-*` naming.
- Merge `scalability.md` and `scalability-plan.md`; cross-link (or absorb) `cli-stream-massive-images.md`.
- Leave `docs/reviews/README.md` as-is; migrate U10 and O8 into their respective plan files.

### Batch E — Followup: file this audit into `docs/reviews/`
Land this document as `docs/reviews/2026-07-07-docs-audit.md` and add a row in `docs/reviews/README.md` so future Claudes can find it before touching any of the surfaces above.

---

## Verdict per file (one-line summary)

| File | Verdict |
|---|---|
| `README.md` | edit (quick start + trim dev content) |
| `CHANGELOG.md` | edit (either backfill user-visible entries or mark "no user-visible changes yet") |
| `CLAUDE.md` | **keep** (every section is a trip-wire earned from a real Claude mistake) |
| `docs/API.md` | edit (add events index entry + Context headers) |
| `docs/ARCHITECTURE.md` | edit (correctness fixes) |
| `docs/CLI.md` | edit (dedupe server-mode with DEPLOYMENT) |
| `docs/DEPLOYMENT.md` | restructure (split GPU-INSTALL out; fix correctness bugs) |
| `docs/EVAL.md` | keep |
| `docs/EXTENDING.md` | edit (collapse checklists) |
| `docs/EXTENDING-plugins.md` | edit (multiple correctness bugs) |
| `docs/EXTENDING-media.md` | edit (multiple correctness bugs) |
| `docs/EXTENDING-processors.md` | keep (also receive detectors-sharing subsection) |
| `docs/HANDOFF.md` | restructure (70 % is duplication) |
| `docs/ML.md` | edit (hidden-layer sizing) |
| `docs/SETUP.md` | needs-major-restructure (split SLURM out; fix test-group list; add troubleshooting) |
| `docs/branch-protection.md` | keep |
| `docs/demos.md` | edit (add sizes; note gated ones) |
| `docs/style-guide.md` | edit (small: z-offline-banner row) |
| `docs/vtscore-api.md` | restructure (split "what ships now" from Phase plan) |
| `docs/api/auth.md` | edit (wrong strings; missing HF OAuth) |
| `docs/api/dashboard.md` | restructure (split into dashboard.md + find.md) |
| `docs/api/datasets.md` | edit (7 missing endpoints) |
| `docs/api/detectors.md` | edit (wrong shape + missing endpoints) |
| `docs/api/events.md` | keep (trim stale "replaces" list) |
| `docs/api/file-browser.md` | edit (thin) |
| `docs/api/io.md` | keep |
| `docs/api/labeling.md` | keep |
| `docs/api/medias.md` | edit (async learned-sort bug) |
| `docs/api/settings.md` | keep |
| `docs/user/USER_GUIDE.md` | edit (tab count; trim dev-facing paragraphs; split .npz section) |
| `docs/user/screenshots-reshoot-queue.md` | keep (consider moving out of `docs/user/`) |
| `docs/plans/README.md` | restructure (trim to policy; drop the enumeration; do not add a status column) |
| `docs/plans/**` | see Section 4 |
| `docs/reviews/README.md` | keep (short "what this dir is for" preface is the right shape) |
| `docs/reviews/**` | keep (all 5 reviews still have open findings) |

---

*Audit produced by 7 parallel reviewers on branch `claude/tender-mendel-lz6xnn` (reset to `origin/dev` at session start, HEAD = 6073074a). No files were edited.*
