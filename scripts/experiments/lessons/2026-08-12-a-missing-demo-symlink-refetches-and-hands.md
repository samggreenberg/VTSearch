# 2026-08-12 — #3121 a missing demo symlink refetches, and hands you a smaller dataset
**Cost:** ~35 min (one 20-minute GPU cell rebuilt, plus the diagnosis).

**What broke.** Building the shared pile on `/expscratch`, only `embeddings/`
was copied across — not the datadir's `visual_genome -> /exp/scale26/...`
symlink. The demo downloaders treat a **missing** extraction dir as "not
downloaded yet", so the job cheerfully started re-downloading Visual Genome from
the internet, got a partial archive, and embedded **1662 of 4193 medias** into a
cell that then verified as perfectly healthy. Nothing errored. Had it not been
caught, every cross-embedder comparison on VG would have compared a 1662-media
population against 4193-media siblings.

The tell was not an error but an **arithmetic disagreement between cells that
should be identical**: same dataset, same source, different `len(medias)`.

**Now prevented by** `pile_config.require_demo_source` (refuses to build a demo
cell when the source dir is missing — or *empty*, which the downloaders read as
"download complete") and `build_pile.py --verify`, which cross-checks that a
dataset's cells agree on media count and names the odd ones out.
