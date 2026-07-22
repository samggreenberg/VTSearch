#!/usr/bin/env python3
"""Assemble the self-contained VTSBrowse UMAP-tuning report (figures inlined).

Figures come from the sweep harness (``plots.py`` / ``visualize.py``). Point
``UMAP_FIG_DIR`` at the directory holding them (default ``./figures``); the
rendered HTML is written to ``UMAP_REPORT_OUT`` (default ``./report.html``).
"""

import base64
import os
import pathlib

FIG = pathlib.Path(os.environ.get("UMAP_FIG_DIR", "figures"))
OUT = pathlib.Path(os.environ.get("UMAP_REPORT_OUT", "report.html"))


def img(name, alt):
    data = base64.b64encode((FIG / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{alt}" loading="lazy">'


# --- CPU-verify result (filled from cpu_verify.json) ------------------------
CPU_VERIFY = (
    "A CPU-backend re-fit of these exact values on <code>umap-learn</code> reproduced the "
    "ranking on all 23 embedded sets — the tuned settings beat the current default by "
    "+2.6% separability on average (per embedder: CLIP +3.2%, SigLIP +3.0%, SigLIP-L "
    "+2.8%, CLAP +1.2%) — so one defaults table is safe for both the GPU (cuML) and CPU "
    "projection paths."
)

CSS = """
:root{
  --ground:#fbfbfd; --surface:#ffffff; --surface-2:#f3f5f9; --border:#e4e7ee;
  --ink:#191c22; --body:#333844; --muted:#697086; --faint:#8b91a2;
  --accent:#0a6db0; --accent-soft:#e7f1f9; --warn:#c4501a; --warn-soft:#f8ece4;
  --good:#1f7a54;
  --serif:Charter,"Bitstream Charter","Iowan Old Style",Georgia,Cambria,serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono","Cascadia Code",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --ground:#12151b; --surface:#191d25; --surface-2:#20252f; --border:#2c323d;
  --ink:#eceef2; --body:#c7ccd6; --muted:#98a0af; --faint:#767d8c;
  --accent:#54a6e0; --accent-soft:#152b3b; --warn:#e2803f; --warn-soft:#33241a;
  --good:#57b98c;
}}
:root[data-theme="dark"]{
  --ground:#12151b; --surface:#191d25; --surface-2:#20252f; --border:#2c323d;
  --ink:#eceef2; --body:#c7ccd6; --muted:#98a0af; --faint:#767d8c;
  --accent:#54a6e0; --accent-soft:#152b3b; --warn:#e2803f; --warn-soft:#33241a;
  --good:#57b98c;
}
:root[data-theme="light"]{
  --ground:#fbfbfd; --surface:#ffffff; --surface-2:#f3f5f9; --border:#e4e7ee;
  --ink:#191c22; --body:#333844; --muted:#697086; --faint:#8b91a2;
  --accent:#0a6db0; --accent-soft:#e7f1f9; --warn:#c4501a; --warn-soft:#f8ece4;
  --good:#1f7a54;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--ground);color:var(--body);
  font-family:var(--sans);font-size:17px;line-height:1.65;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
.wrap{max-width:760px;margin:0 auto;padding:0 24px 120px}
.measure{max-width:680px}
h1,h2,h3{font-family:var(--serif);color:var(--ink);text-wrap:balance;line-height:1.18;font-weight:600}
h1{font-size:2.55rem;letter-spacing:-.015em;margin:.2em 0 .1em}
h2{font-size:1.7rem;letter-spacing:-.01em;margin:2.8em 0 .1em}
h3{font-size:1.2rem;margin:2em 0 .3em}
p{margin:.9em 0}
a{color:var(--accent);text-underline-offset:2px}
strong{color:var(--ink);font-weight:600}
code{font-family:var(--mono);font-size:.86em;background:var(--surface-2);
  padding:.12em .38em;border-radius:4px;color:var(--ink)}
.eyebrow{font-family:var(--mono);font-size:.72rem;letter-spacing:.13em;
  text-transform:uppercase;color:var(--accent);font-weight:600;margin:0}
.lede{font-size:1.28rem;line-height:1.5;color:var(--body);margin:.6em 0 0;font-family:var(--serif)}
header.top{padding:76px 0 40px;border-bottom:1px solid var(--border);margin-bottom:8px}
.meta-row{display:flex;flex-wrap:wrap;gap:8px 22px;margin-top:26px;
  font-family:var(--mono);font-size:.76rem;color:var(--muted);letter-spacing:.02em}
.meta-row span strong{color:var(--ink)}
/* glance card */
.glance{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:26px 26px 12px;margin:34px 0;box-shadow:0 1px 3px rgba(20,30,50,.04)}
.glance h3{margin:0 0 2px;font-family:var(--sans);font-size:.8rem;letter-spacing:.08em;
  text-transform:uppercase;color:var(--muted);font-weight:600}
table{width:100%;border-collapse:collapse;font-size:.94rem;margin:8px 0 18px}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--border)}
th{font-family:var(--mono);font-size:.72rem;letter-spacing:.04em;text-transform:uppercase;
  color:var(--muted);font-weight:600}
td{font-variant-numeric:tabular-nums;color:var(--body)}
td.emb{font-family:var(--mono);color:var(--ink);font-weight:600}
.chg{font-family:var(--mono);font-size:.88rem}
.chg .old{color:var(--faint);text-decoration:line-through;margin-right:5px}
.chg .new{color:var(--accent);font-weight:600}
.chg .same{color:var(--muted)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:6px 0 20px}
.tile{background:var(--surface-2);border-radius:10px;padding:14px 16px}
.tile .k{font-family:var(--mono);font-size:.7rem;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.tile .v{font-family:var(--serif);font-size:1.7rem;color:var(--warn);font-weight:600;line-height:1.1;margin-top:3px}
.tile .v.pos{color:var(--good)}
.tile .s{font-size:.8rem;color:var(--muted);margin-top:2px}
/* figures */
figure{margin:34px 0;padding:0}
figure img{width:100%;display:block;border:1px solid var(--border);border-radius:10px;background:var(--surface)}
figcaption{font-size:.86rem;line-height:1.5;color:var(--muted);margin-top:11px;padding-left:2px}
figcaption b{color:var(--body);font-weight:600}
.updown{font-family:var(--mono);font-size:.78rem;color:var(--accent)}
/* callout */
.callout{border-left:3px solid var(--accent);background:var(--accent-soft);
  padding:14px 20px;border-radius:0 10px 10px 0;margin:26px 0;font-size:.97rem}
.callout.warn{border-color:var(--warn);background:var(--warn-soft)}
.callout p{margin:.35em 0}
.callout .lab{font-family:var(--mono);font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;
  color:var(--accent);font-weight:600}
.callout.warn .lab{color:var(--warn)}
pre{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:16px 18px;overflow-x:auto;font-family:var(--mono);font-size:.82rem;line-height:1.55;color:var(--ink)}
pre .c{color:var(--muted)}
hr{border:0;border-top:1px solid var(--border);margin:52px 0}
.foot{font-size:.85rem;color:var(--muted);margin-top:40px}
ul.tight li{margin:.3em 0}
.qnum{font-family:var(--mono);color:var(--accent);font-size:.9em}
"""

HTML = """<div class="wrap">
<header class="top">
  <p class="eyebrow">VTSBrowse · projection tuning</p>
  <h1>Tuning the map: per-embedder UMAP defaults for VTSBrowse</h1>
  <p class="lede measure">VTSBrowse turns a whole collection into a single zoomable map by
  squashing each item's embedding down to an (x,&nbsp;y) point with UMAP. This is a from-scratch
  empirical study of how UMAP's dials should be set — and it finds that the best settings depend
  on <em>which embedder</em> produced the numbers, and that one long-standing default is quietly
  costing layout quality.</p>
  <div class="meta-row">
    <span><strong>4</strong> embedders (CLAP · CLIP · SigLIP · SigLIP-L)</span>
    <span><strong>11</strong> datasets · <strong>23</strong> embed sets</span>
    <span><strong>~5,000</strong> scored UMAP fits</span>
    <span>GRID · A100 · cuML</span>
  </div>
</header>

<div class="glance measure">
  <h3>Recommended defaults</h3>
  <table>
    <thead><tr><th>Embedder</th><th>Media</th><th>n_neighbors</th><th>min_dist</th><th>compaction</th></tr></thead>
    <tbody>
      <tr><td class="emb">clap</td><td>audio</td><td class="chg"><span class="same">15 (kept)</span></td><td class="chg"><span class="same">0.10 (kept)</span></td><td class="chg"><span class="old">on</span><span class="new">off</span></td></tr>
      <tr><td class="emb">clip</td><td>image</td><td class="chg"><span class="old">15</span><span class="new">10</span></td><td class="chg"><span class="old">0.10</span><span class="new">0.05</span></td><td class="chg"><span class="old">on</span><span class="new">off</span></td></tr>
      <tr><td class="emb">siglip</td><td>image</td><td class="chg"><span class="old">15</span><span class="new">10</span></td><td class="chg"><span class="old">0.10</span><span class="new">0.05</span></td><td class="chg"><span class="old">on</span><span class="new">off</span></td></tr>
      <tr><td class="emb">siglip_l</td><td>image</td><td class="chg"><span class="old">15</span><span class="new">10</span></td><td class="chg"><span class="old">0.10</span><span class="new">0.05</span></td><td class="chg"><span class="old">on</span><span class="new">off</span></td></tr>
    </tbody>
  </table>
  <h3>What compaction costs (all embedders, every dataset)</h3>
  <div class="tiles">
    <div class="tile"><div class="k">Separability</div><div class="v">&minus;2.0%</div><div class="s">class blobs bleed</div></div>
    <div class="tile"><div class="k">Trustworthiness</div><div class="v">&minus;6.2%</div><div class="s">false neighbors</div></div>
    <div class="tile"><div class="k">Continuity</div><div class="v">&minus;4.7%</div><div class="s">torn neighbors</div></div>
    <div class="tile"><div class="k">Neighbor recall</div><div class="v">&minus;6.0%</div><div class="s">local structure</div></div>
  </div>
</div>

<h2>Why this study</h2>
<p class="measure">VTSBrowse's map is built in two stages. Stage&nbsp;1 runs <strong>UMAP</strong> on each item's
embedding — the long list of numbers a neural network uses to represent its content — and squashes
those hundreds or thousands of numbers down to two, an (x,&nbsp;y) point on screen, placing similar
items near each other. Until now VTSBrowse used <em>one</em> fixed setting of UMAP's dials for
everything, regardless of which model produced the embeddings. This study asks two questions the
codebase had left open: should those dials be set <strong>per embedder</strong>, and should the
cosmetic <strong>compaction</strong> step stay switched on?</p>

<h3>The three dials</h3>
<ul class="tight measure">
  <li><strong>n_neighbors</strong> — how much of the collection UMAP consults when placing each item.
  Small values (5–15) capture tight local lookalikes and make many small islands; large values
  (100–200) weigh the global arrangement into a few broad continents. The most consequential dial.</li>
  <li><strong>min_dist</strong> — how tightly items may pack within a cluster. Changes how the map
  <em>looks</em> but not who counts as a neighbor, so we expect a weaker effect.</li>
  <li><strong>compaction</strong> (on/off) — a post-step that slides UMAP's scattered clusters together
  like puzzle pieces (each moving rigidly, keeping its shape) to close the empty "oceans" UMAP leaves
  between islands, so the canvas isn't mostly dead space after zoom-to-fit.</li>
</ul>
<p class="measure">The four embedders in scope — <strong>CLAP</strong> (audio) and <strong>CLIP</strong>,
<strong>SigLIP</strong>, <strong>SigLIP-L</strong> (images) — output vectors of different sizes
(512, 512, 768, 1152) with different geometry, which is exactly why one-size dials might leave
quality on the table.</p>

<h2>How we scored a map</h2>
<p class="measure">A good map keeps items of the same kind together. To measure that objectively we used
datasets that ship with a <strong>taxonomy</strong> — a labeled tree of categories (ESC-50's
50&nbsp;sounds group into 5&nbsp;families; iNaturalist's species nest into a 6-level biological tree).
For every category we ask: do that category's points sit among their own kind on the map? Formally,
for each point we take its 20 nearest map-neighbors and measure how strongly same-category points are
surrounded by same-category neighbors — summarized as an <strong>AUROC</strong> from 0.5 (no
separation) to 1.0 (a clean boundary), averaged over every category and every level of the tree.</p>
<p class="measure">Two refinements make the number trustworthy. <strong>Ceiling normalization:</strong>
some classes are hard to separate even in the original high-dimensional space, so we compute the same
score there and report the <em>ratio</em> (map ÷ original). A ratio near&nbsp;1.0 means the projection
kept essentially all the separability there was to keep — and that ratio is precisely what the UMAP
dials control. <strong>Guard metrics:</strong> a layout could fake purity by shattering the space, so
label-free checks (trustworthiness, continuity, neighbor-recall) veto any setting that wins on
separability while mangling local structure. Because production runs UMAP <em>unseeded</em>, we also
fit every setting with 3&nbsp;seeds and track run-to-run <strong>stability</strong>. <span class="updown">Up is good on every metric here.</span></p>

<div class="callout"><p class="lab">Setup</p>
<p>Grid: <code>n_neighbors ∈ {5,10,15,30,50,100,200}</code> × <code>min_dist ∈ {0,0.05,0.1,0.25,0.5}</code>
× <code>compact ∈ {off,on}</code> × 3 seeds, over 23 embedded (dataset&nbsp;×&nbsp;embedder) matrices — audio
(ESC-50 S/M/L, GTZAN, FSD50K) and image (Caltech-256 S/M, Places365 S/M/L, iNaturalist). Compaction is a
<em>free</em> axis: both variants are scored from the same fit. Two deep-taxonomy sets — FSD50K (AudioSet
ontology) and an iNaturalist subset (156 species across 3 kingdoms) — were built for this study.</p></div>

<h2><span class="qnum">Q1.</span> The best n_neighbors depends on the embedder</h2>
<p class="measure">Across the whole grid, separability peaks at <strong>small n_neighbors</strong> and falls
steadily as the dial grows — and the peak sits in a slightly different place for audio than for images.
CLAP is happiest around 10–15; all three image embedders peak at <strong>10</strong>, with CLIP the most
sensitive (it drops off fastest). Nobody benefits from large neighborhoods: at n_neighbors&nbsp;=&nbsp;200
every embedder is measurably worse, and images lose the most.</p>
__FIG_HEATMAPS__
__FIG_NNCURVES__

<h2><span class="qnum">Q2.</span> The optimum tracks the embedder, not the dataset size</h2>
<p class="measure">A natural worry: maybe the best n_neighbors just scales with how many items you have.
It doesn't. The Places365 curves above span 5,000 → 21,000 images and lie almost on top of each other —
the peak stays put as N grows. So the deliverable is a per-embedder <em>constant</em>, not a formula in N.
Stability tells the same story from another angle: small-to-mid neighborhoods are not only more separable
but more reproducible run-to-run, while at n_neighbors&nbsp;=&nbsp;200 two unseeded fits share only about
half their neighbors — a real hazard for a production path that never seeds.</p>
__FIG_STABILITY__

<h2><span class="qnum">Q3.</span> Compaction consistently costs layout quality</h2>
<p class="measure">Compaction was added for readability — closing the empty oceans so the canvas fills the
frame. But sliding clusters until they nearly touch pushes neighboring classes into contact, and the metrics
see it everywhere: <strong>every</strong> dataset&nbsp;×&nbsp;embedder loses separability under compaction,
and the structural guards drop 5–6%. The cost grows with the layout's density — small, easy sets barely
notice; the 21k-image Places365 layouts lose the most.</p>
__FIG_COMPDELTA__
__FIG_COMPEYE__
<div class="callout warn"><p class="lab">Verdict</p>
<p>The hit is consistent and material, so the metric-driven call is to <strong>turn compaction off by
default</strong> (it remains a per-media-type server setting, so it's one toggle to restore). The eyeball
above shows the tradeoff plainly: raw UMAP keeps the biological classes as clean, separated islands;
compaction fills the frame but smears the boundaries. The better long-term fix is to <em>rework</em>
<code>compact_layout</code> to keep a minimum margin between islands — closing oceans without the bleed —
which is filed as follow-up.</p></div>

<h3>The layouts, by eye</h3>
<p class="measure">The scores line up with what you see. As n_neighbors grows, a projection moves from a
scatter of tiny fragments (over-local) toward a few coherent continents; the sweet spot keeps genuine
sub-groups distinct without dissolving them into the background.</p>
__FIG_NNGRID__
__FIG_GUARD__

<h2>What shipped</h2>
<p class="measure">The chosen defaults are wired in as a per-embedder map, consulted where the global
<code>PROJECTION_N_NEIGHBORS</code> / <code>PROJECTION_MIN_DIST</code> were read, keyed off the dataset's
primary embedder; the existing <code>ServerSettings</code> still override, and the persisted projection is
already keyed on effective params so new defaults force a clean recompute. The compaction default flips to
off. __CPU_VERIFY__</p>
<pre><span class="c"># vtscore/config.py — consulted by the projection route, per primary embedder</span>
PROJECTION_DEFAULTS_BY_EMBEDDER = {
    <span class="c"># embedder:     (n_neighbors, min_dist)</span>
    "clap":     (15, 0.10),   <span class="c"># audio: flat peak 10–30</span>
    "clip":     (10, 0.05),   <span class="c"># image: most n_neighbors-sensitive</span>
    "siglip":   (10, 0.05),
    "siglip_l": (10, 0.05),
}
<span class="c"># fit_projection(..., compact=False)  — verdict applied; overridable per media type</span></pre>

<h2>Take-aways</h2>
<ul class="tight measure">
  <li><strong>Set n_neighbors per embedder:</strong> 10 for the image embedders, 15 for CLAP. The old
  global 15 was fine for audio but slightly high for images.</li>
  <li><strong>Never go large:</strong> n_neighbors ≥ 100 hurts every embedder on both separability and
  stability — the clearest, most actionable result.</li>
  <li><strong>min_dist barely matters</strong> for grouping; images prefer it a touch lower (0.05), which
  also reads as tighter clusters. It's the one dial safe to decide on looks.</li>
  <li><strong>Compaction trades layout fidelity for screen-fill</strong> — a consistent 2%/6% cost. Off by
  default now; worth reworking to keep the readability win without the bleed.</li>
  <li><strong>The optimum is a per-embedder constant,</strong> not a function of dataset size.</li>
</ul>

<hr>
<h2>Method &amp; reproduction</h2>
<p class="measure">The harness lives in <code>scripts/experiments/umap_params/</code> (dev-only, outside the
shipped package). It embeds each (dataset&nbsp;×&nbsp;embedder) once and caches the matrix, then re-fits
UMAP cheaply over the cache for the whole grid; the metric is pure NumPy/scikit-learn. Winners were
re-fit on the CPU <code>umap-learn</code> backend to confirm the ranking transfers from the GPU cuML path
that production uses. The core of the separability metric:</p>
<pre><span class="c"># metric.py — AUROC that same-class points have more same-class map-neighbors</span>
def node_auroc(knn_idx, mask):
    frac = mask[knn_idx].mean(axis=1)          <span class="c"># in-class neighbor fraction</span>
    return roc_auc_score(mask, frac)           <span class="c"># 1.0 = clean boundary at k</span>
<span class="c"># ratio = (mean node AUROC on 2-D) / (same on the original embedding)</span></pre>
<p class="foot">Ceiling-normalized taxonomy separability with label-free structure guards and multi-seed
stability. Full sweep: ~5,000 scored fits across 23 embedded matrices on an NVIDIA A100. Figures generated
with <code>plots.py</code> / <code>visualize.py</code>; palette validated colorblind-safe (Okabe-Ito).</p>
</div>
"""


def build():
    body = (
        HTML.replace(
            "__FIG_HEATMAPS__",
            fig(
                "fig_heatmaps.png",
                "<b>Separability across the grid.</b> Each panel is one embedder; cells are the "
                "n_neighbors × min_dist grid, shaded by the ceiling-normalized separability ratio "
                "(<span class='updown'>darker = better</span>). Black box marks each embedder's best cell. "
                "CLAP (far left) sits much higher than the image embedders; every panel favors the left "
                "(small n_neighbors) side.",
                "Heatmaps of separability over the parameter grid, per embedder",
            ),
        )
        .replace(
            "__FIG_NNCURVES__",
            fig(
                "fig_nn_curves.png",
                "<b>Where separability peaks.</b> Separability ratio vs n_neighbors (log axis), one line per "
                "dataset. <span class='updown'>Up = better.</span> Image embedders decline steadily past ~10; "
                "the three Places365 sizes (5k/10k/21k) overlap tightly — the optimum doesn't move with N.",
                "Line charts of separability vs n_neighbors per dataset, faceted by embedder",
            ),
        )
        .replace(
            "__FIG_STABILITY__",
            fig(
                "fig_stability.png",
                "<b>Run-to-run stability.</b> How much two unseeded fits agree on each point's neighbors "
                "(<span class='updown'>up = more stable</span>), vs n_neighbors. Small-to-mid neighborhoods are "
                "the most reproducible; at 200 the layout wobbles badly (~0.5 = only half the neighbors shared).",
                "Line chart of inter-seed neighbor agreement vs n_neighbors, per embedder",
            ),
        )
        .replace(
            "__FIG_COMPDELTA__",
            fig(
                "fig_compaction_delta.png",
                "<b>Compaction hurts everywhere.</b> Change in separability (compacted − raw) for each "
                "dataset × embedder. Every bar is negative (<span class='updown'>left = worse</span>); the cost "
                "is largest on the big, dense Places365 layouts and smallest on tiny, easy sets.",
                "Bar chart of separability change under compaction, per dataset and embedder",
            ),
        )
        .replace(
            "__FIG_COMPEYE__",
            fig(
                "compaction_inat_val__siglip.png",
                "<b>The tradeoff, by eye</b> (iNaturalist · SigLIP, colored by biological class). Raw UMAP "
                "(left) keeps classes as clean, separated islands but leaves empty space; compaction (right) "
                "fills the frame but slides the islands into contact so their edges bleed together.",
                "Raw vs compacted UMAP layout of iNaturalist, colored by class",
            ),
        )
        .replace(
            "__FIG_NNGRID__",
            fig(
                "nn_grid_inat_val__siglip.png",
                "<b>n_neighbors, visualized</b> (iNaturalist · SigLIP, colored by biological class). At 5 the "
                "map is over-fragmented; by 50–200 classes merge into broad blobs. The tuned value keeps "
                "distinct groups distinct.",
                "Four UMAP layouts of iNaturalist at increasing n_neighbors",
            ),
        )
        .replace(
            "__FIG_GUARD__",
            fig(
                "fig_guard_scatter.png",
                "<b>Purity isn't gamed.</b> Separability rises with the neighbor-recall guard rather than "
                "against it — high-separability layouts also preserve real neighborhoods, so no setting is "
                "winning by shattering the space.",
                "Scatter of separability vs neighbor-recall guard, per embedder",
            ),
        )
        .replace("__CPU_VERIFY__", CPU_VERIFY)
    )
    # The Artifact host wraps this in its own <!doctype>/<head>/<body>; emit only
    # a <style> block + the page content.
    page = f"<style>{CSS}</style>\n{body}"
    OUT.write_text(page)
    print("wrote", OUT, f"({len(page) // 1024} KB)")


def fig(name, caption, alt):
    return f"<figure>{img(name, alt)}<figcaption>{caption}</figcaption></figure>"


if __name__ == "__main__":
    build()
