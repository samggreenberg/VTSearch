# VTSearch Plans

Design docs for **future work**: features that are proposed or in progress,
narrowed to the parts still owed. Plans are **not** an archive of completed
work. When a slice ships, its narrative is deleted from the plan; a plan whose
work has fully shipped is removed entirely (fold any durable design rationale
into [ARCHITECTURE.md](../ARCHITECTURE.md) / [EXTENDING.md](../EXTENDING.md)
first). Keep only a short "Background" note when the remaining work can't be
understood without it. Git history and merged PRs are the record of what
already landed; see [`../../CLAUDE.md`](../../CLAUDE.md) for the full policy.

This index is a grouped filename list; open a file for its remaining scope.
(Point-in-time UI review/audit reports live in
[`../reviews/`](../reviews/README.md), not here.)

## VTSBrowse (UMAP hexbin dataset browser)

- [vtsbrowse.md](vtsbrowse.md): UMAP hexbin browser — design spec; thin open work (thin-pickle save mode, tuning, WebGL escape hatch, compaction fill ceiling)
- [vtsbrowse-empirical-tuning.md](vtsbrowse-empirical-tuning.md): UMAP/pyramid/renderer tuning — quantitative sweep + qualitative browser review remain
- [vtsbrowse-toponymy.md](vtsbrowse-toponymy.md): named-region "street signs" (design only)
- [browse-audio-player.md](browse-audio-player.md): audio-bin waveform tiles + hover player — merge bin-popup preview with now-playing, data-drive from `has_thumbnail`

## Scalability

- [scalability.md](scalability.md): brainstorm defining the `S#` IDs (reference)
- [scalability-plan.md](scalability-plan.md): phased implementation plan — §2.3, §3.1, §3.2, §3.4, and Phase 4 open
- [cli-stream-massive-images.md](cli-stream-massive-images.md): CLI streaming for huge media sources — open follow-ups

## Detectors / embedders / clippers

- [patch-embedder.md](patch-embedder.md): patch-based image embedder — scoring/validation follow-ups + open questions
- [structural-embedder.md](structural-embedder.md): structural (instance-matching) embedder — larger VLAD codebook next
- [cli-detector-converter.md](cli-detector-converter.md): CLI autodetect converter-routing + re-clipping pipeline (open work)

## Find / verification

- [find-verification-workflow.md](find-verification-workflow.md): Find verify loop — visual eyeball pass + follow-ups open
- [coverage-atlas.md](coverage-atlas.md): domain-shift + evidence-aware verification for transferred detectors (design/research writeup only)
- [visual-genome-dataset.md](visual-genome-dataset.md): Visual Genome demo dataset — region-vote reporting, richer vocab, attributes open

## Import / plugins

- [server-import-ux.md](server-import-ux.md): Server/Services import UX — open follow-ups
- [RCDatasetImporter.md](RCDatasetImporter.md): RCDatasetImporter / Holder / PullWrest extension — API client implementation (open)

## Progress / UX

- [progress-bar-consolidation.md](progress-bar-consolidation.md): whole-job progress bar — text-sort unification + cleanups open
- [auto-find-settings-tab.md](auto-find-settings-tab.md): Auto-Find settings tab — UI caller + streaming-export fallback open

## Frontend migrations

- [angular-21-upgrade.md](angular-21-upgrade.md): reusable Angular-bump reference (upgrade path, risks, gotchas) + optional Vitest fakeAsync follow-ups
- [httpresource-migration.md](httpresource-migration.md): move data-layer reads onto Angular `rxResource` primitives (pollers / forkJoin aggregates deferred)

## Audits / tooling / methodology

- [code-structure-review.md](code-structure-review.md): repo-wide structural review — Themes B/C/D/E/F + prioritized backlog open
- [browser-vision-testing.md](browser-vision-testing.md): reusable browser-vision testing playbook (templates for future rounds)
- [user-docs-screenshots.md](user-docs-screenshots.md): screenshot pipeline reference (manifest + capture harness) — masking polish deferred
