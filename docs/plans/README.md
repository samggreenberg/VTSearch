# VTSearch Plans

Design docs for features that are proposed, in progress, or recently
landed. Once a plan ships and its design notes are absorbed into
[EXTENDING.md](../EXTENDING.md) / its siblings or
[ARCHITECTURE.md](../ARCHITECTURE.md), the plan file is deleted. Plans
whose work has fully shipped are removed; partially-shipped plans keep
their open items and strike through the completed ones.

Open the file for status; this index is just a grouped filename list so
it can't drift far. (Point-in-time UI review/audit reports live in
[`../reviews/`](../reviews/README.md), not here.)

## VTSBrowse (UMAP hexbin dataset browser)

- [vtsbrowse.md](vtsbrowse.md): core design + what shipped (the feature is live; thin open work — tuning, WebGL escape hatch)
- [vtsbrowse-empirical-tuning.md](vtsbrowse-empirical-tuning.md): UMAP/pyramid/renderer tuning pass (not started — needs a browser)
- [vtsbrowse-toponymy.md](vtsbrowse-toponymy.md): named-region "street signs" (design only)
- [vtsbrowser-hex-circle-radius.md](vtsbrowser-hex-circle-radius.md): singleton-circle radius investigation (needs visual verification)
- [vtsbrowser-qa-followups.md](vtsbrowser-qa-followups.md): QA-drive follow-ups (deferred: startup wedge, tab crash; skipped: toolbar overlay)

## Scalability

- [scalability.md](scalability.md): brainstorm defining the `S#` IDs (reference)
- [scalability-plan.md](scalability-plan.md): phased implementation plan (§3.3, Phase 1.2 GMM subsampling, CLI streaming shipped; rest open)
- [cli-stream-massive-images.md](cli-stream-massive-images.md): CLI streaming for huge media sources (Phase 1 shipped)
- [server-dedup-references.md](server-dedup-references.md): reference (no-copy) server import + lazy clips (Phase 1 + 2a shipped; Phase 2b lazy converter output designed only)

## Detectors / embedders / clippers

- [patch-embedder.md](patch-embedder.md): patch-based image embedder (V1+V2 shipped; V3 in progress)
- [structural-embedder.md](structural-embedder.md): structural (instance-matching) embedder — SIFT/VLAD + RANSAC re-rank (V1 + Stage-2 + Find re-rank shipped; larger VLAD codebook next)
- [clipper-chain.md](clipper-chain.md): ordered converter/clipper chains (Phase 1 shipped)
- [cli-detector-converter.md](cli-detector-converter.md): CLI autodetect with converters + clippers (Phase 1 shipped)

## Find / verification

- [find-verification-workflow.md](find-verification-workflow.md): Find verify loop, frozen scores, Stats (Phases 1–4 shipped; follow-ups open)
- [coverage-atlas.md](coverage-atlas.md): domain-shift + evidence-aware verification for transferred detectors (design/research writeup only)

## Import / plugins

- [server-import-ux.md](server-import-ux.md): Server/Services import UX (Phase 1 shipped; UX follow-ups open)
- [dataset-import-archives.md](dataset-import-archives.md): import media from zip/tar/rar archives via folder + URL importers (shipped; auto-detect / provenance follow-ups open)
- [RCDatasetImporter.md](RCDatasetImporter.md): RCDatasetImporter / Holder / PullWrest extension (scaffolds only; API clients open)

## Progress / UX

- [progress-bar-consolidation.md](progress-bar-consolidation.md): one whole-job progress bar + overall ETA + step count for multi-phase loads (shipped; text-sort unification + a few cleanups open)
- [auto-find-settings-tab.md](auto-find-settings-tab.md): Auto-Find settings tab, per-user detector lists, detector access controls (shipped, incl. Phase 2 rename + hardening)

## Frontend migrations

- [angular-21-upgrade.md](angular-21-upgrade.md): Angular 19→21 bump (clears npm-audit highs) + Vitest spec migration to restore runnable frontend tests (all phases shipped; kept as the migration reference for future bumps)
- [zoneless-migration.md](zoneless-migration.md): drop zone.js for `provideZonelessChangeDetection()` + signalize reactivity (Phases 0–5 complete; production is zoneless; kept as reference)
- [httpresource-migration.md](httpresource-migration.md): move data-layer reads onto Angular `rxResource` primitives (Phases 1–4 shipped; pollers / forkJoin aggregates deferred)

## Audits / tooling / methodology

- [code-structure-review.md](code-structure-review.md): repo-wide structural review of accretion problems (Theme A + mega-file splits + quick wins shipped; Themes B/D/E/F partly open)
- [logical-bug-audit.md](logical-bug-audit.md): codebase logical-bug audit (all findings resolved; kept as the audit record)
- [browser-vision-testing.md](browser-vision-testing.md): browser-vision testing playbook (first round ran; reusable)
- [user-docs-screenshots.md](user-docs-screenshots.md): auto-refreshable screenshots for user docs — manifest + capture harness (shipped 2026-06-10; 32 PNGs embedded; masking polish deferred)
