# `docs/reports/` — index

Standalone HTML study write-ups: self-contained pages (charts and figures
inlined) that explain a measurement to a reader rather than to the harness. Open
them in a browser — GitHub renders raw HTML as source.

**The other archive.** [`docs/experiments/`](../experiments/) holds the per-study
`REPORT.md` files, generated tables and raw CSVs. The two are complementary: an
`experiments/` report is the record of what a run produced; a `reports/` page is
the narrative built on top of one. Some studies have both.

Add a row here whenever you add a report — an unlinked report is an unread one.

| Report | Date | What it answers | Cited from |
|--------|------|-----------------|-----------|
| [Structural search in the wild — VTSearch × OpenLogo](2026-07-11-structural-search-openlogo.html) | 2026-07-11 | Seeded with one real-world logo crop over 27k unstaged photos: how far does SIFT+VLAD get, does 6-DoF beat 4-DoF, and what do labels buy? **Verdict:** Stage-1 VLAD recall is the ceiling (2.5% of true instances reach the top-50); SigLIP cosine is ~10× stronger; 4-DoF wins; on the structural path labels are calibration, not learning. Also caught the deterministic **30th-vote transient** in the shared MLP trainer. | [`structural-embedder.md`](../plans/structural-embedder.md) |
| [Street signs for sound — Toponymy × VTSBrowse audio](2026-07-12-toponymy-audio-signposts.html) | 2026-07-12 | Can Toponymy name the neighbourhoods of an audio browse map, and what is the right audio→text strategy to feed it? 4,445 clips, 4 strategies, 2 namers. | [`vtsbrowse-toponymy.md`](../plans/vtsbrowse-toponymy.md), `vtscore/projection/signpost_texts.py` |
| [Street signs for pictures — Toponymy × VTSBrowse images](2026-07-12-toponymy-image-signposts.html) | 2026-07-12 | The image half: 8 image→text strategies across photos, screenshots and scans, including re-fitting signs over a Find result. | [`vtsbrowse-toponymy.md`](../plans/vtsbrowse-toponymy.md), `vtscore/projection/signpost_texts.py` |
| [Structural search on screenshots & scans](2026-07-13-screenshot-iconography.html) | 2026-07-13 | The OpenLogo follow-up on flat rasters. **Verdict:** SIFT is the bottleneck on line art (5.1% vs SuperPoint+LightGlue's 41% true-pair verify); SP+LG as ranker beats SigLIP on both document corpora — the first time structural search wins on a real corpus. Plus: 3-DoF is a free precision win, LightGlue needs its own inlier floor (~24), and 224 px tiling rescues small targets. | [`structural-embedder.md`](../plans/structural-embedder.md) |
| [MLP vs SVM — should VTSearch swap its ranker?](2026-07-22-mlp-vs-svm-ranker.html) | 2026-07-22 | Would a linear or RBF SVM rank better than the detector MLP? 150 voting trajectories, ≤200 votes each. **Verdict:** keep the MLP. | [`docs/experiments/mlp-vs-svm/`](../experiments/mlp-vs-svm/), `scripts/experiments/mlp_vs_svm/` |
| [Tuning the map — per-embedder UMAP defaults for VTSBrowse](2026-07-22-vtsbrowse-umap-tuning.html) | 2026-07-22 | How should UMAP's dials be set for the Browse projection? ~5,000 scored fits over 23 embed sets. **Verdict:** `n_neighbors` tracks the embedder (10 image / 15 audio), `min_dist` barely matters, and compaction consistently costs layout quality — so it ships off. | [`vtsbrowse-empirical-tuning.md`](../plans/vtsbrowse-empirical-tuning.md), [`vtsbrowse.md`](../plans/vtsbrowse.md) |
