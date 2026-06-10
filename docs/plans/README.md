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

- [vtsbrowse.md](vtsbrowse.md): core design + what shipped (the feature is live)
- [vtsbrowse-empirical-tuning.md](vtsbrowse-empirical-tuning.md): UMAP/pyramid/renderer tuning pass (not started — needs a browser)
- [vtsbrowse-toponymy.md](vtsbrowse-toponymy.md): named-region "street signs" (design only)
- [vtsbrowse-audit-fixes.md](vtsbrowse-audit-fixes.md): queued audit fixes (#1 shipped; #2–#3 open)
- [vtsbrowser-hex-circle-radius.md](vtsbrowser-hex-circle-radius.md): singleton-circle radius investigation (needs visual verification)
- [vtsbrowser-qa-followups.md](vtsbrowser-qa-followups.md): QA-drive follow-ups (deferred: startup wedge, tab crash; skipped: toolbar overlay)

## Scalability

- [scalability.md](scalability.md): brainstorm defining the `S#` IDs (reference)
- [scalability-plan.md](scalability-plan.md): phased implementation plan (§3.3 shipped; rest open)
- [cli-stream-massive-images.md](cli-stream-massive-images.md): CLI streaming for huge media sources (Phase 1 shipped)

## Detectors / embedders / clippers

- [patch-embedder.md](patch-embedder.md): patch-based image embedder (V1+V2 shipped; V3 design)
- [clipper-chain.md](clipper-chain.md): ordered converter/clipper chains (Phase 1 shipped)
- [cli-detector-converter.md](cli-detector-converter.md): CLI autodetect with converters + clippers (Phase 1 shipped)

## Find / verification

- [find-verification-workflow.md](find-verification-workflow.md): Find verify loop, frozen scores, Stats (Phases 1–4 shipped; follow-ups open)
- [coverage-atlas.md](coverage-atlas.md): domain-shift + evidence-aware verification for transferred detectors (design/research writeup only)

## Import / plugins

- [server-import-ux.md](server-import-ux.md): Server/Services import UX (Phase 1 shipped; UX follow-ups open)
- [RCDatasetImporter.md](RCDatasetImporter.md): RCDatasetImporter / Holder / PullWrest extension (scaffolds only; API clients open)

## Audits / tooling / methodology

- [logical-bug-audit.md](logical-bug-audit.md): codebase logical-bug audit (C/H shipped; ~12 M/L open)
- [browser-vision-testing.md](browser-vision-testing.md): browser-vision testing playbook (first round ran; reusable)
- [user-docs-screenshots.md](user-docs-screenshots.md): auto-refreshable screenshots for user docs — manifest + capture harness (proposed; no shots yet)
