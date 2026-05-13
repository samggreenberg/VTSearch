# VTSearch User Guide

A walkthrough for people who want to **use** VTSearch — not install,
extend, or debug it. Open it, load a dataset, label a few things,
export results. For installation see [SETUP.md](SETUP.md). For
CLI workflows (no browser) see [CLI.md](CLI.md).

## Contents

1. [What VTSearch does](#what-vtsearch-does)
2. [Opening the app](#opening-the-app)
3. [Loading a dataset](#loading-a-dataset)
4. [The three-panel layout](#the-three-panel-layout)
5. [Autopilot — the guided workflow](#autopilot--the-guided-workflow) *(start here)*
6. [Manual mode — for power users](#manual-mode--for-power-users)
7. [View options](#view-options)
8. [Dashboard — managing datasets and models](#dashboard--managing-datasets-and-models)
9. [Exporting your work](#exporting-your-work)
10. [Importing pre-trained detectors](#importing-pre-trained-detectors)
11. [Where to go next](#where-to-go-next)

---

## What VTSearch does

VTSearch helps you **find the items you care about** inside a large
collection of audio clips, images, text paragraphs, videos, or
documents. You vote items **good** or **bad**, and a small neural net
learns from your votes to rank the rest of the collection by how likely
each item is to match what you're looking for. You can also search by
typing a natural-language description ("dog barking", "red car in
snow"), and the app uses a pretrained embedding model to rank items
by semantic similarity to your query.

In practice, you usually combine the two: search for a rough starting
point, vote a handful of items, and let the learned model refine the
ranking. VTSearch's **Autopilot** drives that loop for you — so most
users never need to think about sort modes or selection strategies
directly.

---

## Opening the app

Once the server is running (see [SETUP.md](SETUP.md)), visit
`http://localhost:5000` in a browser. The app opens with an empty
workspace and a hamburger menu (☰) in the top-left corner — that's
your entry point for loading data.

The interface is a single page with three vertical panels. They fill
in as you load a dataset and start labeling.

---

## Loading a dataset

Click the hamburger menu (☰) in the top-left. You get two choices:

- **Demo datasets** — VTSearch ships with a catalogue of open
  datasets across all five media types (audio, image, text, video,
  document). See [demos.md](demos.md) for the full list. Each demo
  downloads on first use (~15 MB to ~1.2 GB depending on dataset) and
  is cached, so subsequent loads are instant.
- **Import your own** — Load a folder of files from the server, a
  VTSearch pickle file, a zipped HTTP archive, or combine several
  existing datasets. Each importer asks for the fields it needs
  (path, URL, media type) in a small form.

  Two of the file-list importers — **Server Files** and **Local
  Files** — also accept a `.npz` archive of pre-computed embedding
  vectors so you can import media you have already embedded offline
  without paying for embedding twice. See
  [Pre-computed embeddings (.npz)](#pre-computed-embeddings-npz) below.

Loading a dataset does three things: downloads or reads the media,
generates an embedding for every item using the appropriate model
(CLAP for audio, SigLIP for images, X-CLIP for video, E5 for text),
and builds a **diversity tree** that groups similar items for later
diverse sampling. Progress is shown in a modal while it runs.

If the model for your media type isn't cached yet, the first dataset
of that type triggers a one-time download (e.g. CLAP is ~1.1 GB).
Subsequent datasets of the same type reuse the cached model.

### Pre-computed embeddings (.npz)

If you have already embedded your media offline — for example with
your own script using the same model VTSearch uses — you can skip the
server-side re-embedding step by handing VTSearch a NumPy `.npz`
archive of pre-computed vectors. Two importers accept this:

- **Server Files** — instead of a `.txt`/`.list` paths file, point
  the *Paths File* field at a `.npz`. The archive holds both the
  media-file paths AND their vectors; VTSearch reads the paths from
  disk and reuses the supplied vectors.
- **Local Files** — alongside the media files you upload, attach a
  `.npz` to the optional *Pre-computed embeddings* file picker. Files
  whose name matches an NPZ key reuse the supplied vector; files
  without a matching entry are embedded normally on the server.

VTSearch accepts two NPZ layouts:

1. **`filenames` + `vectors` arrays** — produced by
   `np.savez(path, filenames=names, vectors=vecs)` where `names` is
   a 1-D string array of length *N* and `vecs` is a 2-D float array
   of shape *(N, D)*. The i-th name maps to the i-th row of `vecs`.
2. **Per-key** — produced by
   `np.savez(path, **{name: vec for name, vec in zip(names, vecs)})`.
   Each archive key is a filename; the corresponding value is its
   vector.

The vector dimension and the embedding model must match what
VTSearch would have used (e.g. 512-d CLAP for audio, 768-d SigLIP for
images). Embedding-model selection is **not** persisted inside the
NPZ — the importer's *Embedder* setting still controls which model
is used for any file that doesn't have a pre-computed vector, and
also acts as the model identifier recorded on each media. Pick an
embedder that matches the vectors in your NPZ.

---

## The three-panel layout

Once a dataset is loaded, VTSearch shows three panels left to right:

- **Left panel** — the sort bar, your selection-strategy controls,
  the inclusion slider, and the **media list** (ranked by the current
  sort). This is where you pick what to look at next.
- **Centre panel** — the **media viewer**. The selected item plays
  (audio), displays (image, video, text, document page), and offers
  two big vote buttons: **Good** (green) and **Bad** (red).  On
  image datasets that use a patch-region embedder
  (DINOv2/DINOv3/EUPE `_patch`), the centre panel also supports
  **region voting** — see "Region voting on images" below.
- **Right panel** — your **vote piles**. Everything you've voted good
  or bad is stacked here, most-recent first, so you can scan your
  work, un-vote, or re-vote.

The dividers between panels can be dragged to resize them. The app
remembers your layout per media type.

---

## Autopilot — the guided workflow

**Start here.** Autopilot is the recommended way to use VTSearch.
Most users should never need Manual mode.

Click the **Autopilot** tab in the left panel. Autopilot breaks
labeling into four phases and tells you what to do at each step.
You still click **Good** or **Bad** on each item shown — Autopilot
just picks *which* items to show you and *when* each phase ends.

### The four phases

1. **Good examples** — Vote some **good** items (default: 3). The
   model needs positive examples before it can learn anything.
   Autopilot offers strong candidates first via the same semantic
   ranking the Text sort uses. If you don't see anything good, type
   a text query into the sort bar to jump-start the ranking.
2. **Bad examples** — Vote some **bad** items (default: 4). Now
   the model has both sides of the boundary. Autopilot flips to
   items ranked low, so finding clear bad examples is usually quick.
3. **Boundary refinement (hard)** — Autopilot serves items the
   model is **uncertain about** — the hardest cases near the
   decision boundary. Voting these teaches the model fastest.
   This phase continues until the model's confidence stabilises
   (the "smart" and "stable" indicators in the status bar both
   turn green).
4. **Diversity exploration (new)** — Autopilot serves items from
   parts of the dataset the model hasn't seen yet, using the
   diversity tree. This catches edge cases the boundary phase
   missed. Phase ends when the diversity coverage hits your goal
   (default: 40%).

When all four phases are done, Autopilot says **done**. You can
keep labeling if you want — the model continues to improve — or
move on to exporting results.

### The collapsed bar

You can collapse Autopilot to a thin strip that just shows the
four phase indicators. Click any active phase to re-pick the
current recommendation (useful if you voted the wrong way and
want a fresh suggestion). Collapsed mode is handy once you're
comfortable with the flow and want more vertical room for the
media list.

### Configuring Autopilot

Most people never touch these, but the Settings modal (gear icon)
exposes:

- **Top greens** — how many good votes phase 1 requires (default 3).
- **Hard reds** — how many bad votes phase 2 requires (default 4).
- **Resort interval** — how often the learned model is retrained
  during phases 3 and 4 (default every 10 votes).
- **Goal diversity** — the fraction of the dataset's diversity
  tree that phase 4 must cover before finishing (default 40%).

Raising these numbers trains a more thorough model at the cost of
more labelling effort.

---

## Manual mode — for power users

Manual mode gives you direct control over what the sort bar ranks
by and which unlabeled item is served next. Use it if Autopilot's
defaults don't fit your workflow, you're debugging a weird
ranking, or you want to label under an unusual regime (e.g. pure
diversity sampling with no voting).

The Manual tab shows three control rows above the media list.

### 1. Sort mode

Picks how the left-panel list is ordered.

- **Text** — Type a natural-language query (e.g. "dog barking",
  "aerial photo of farmland"). Items are ranked by semantic
  similarity to your query using the embedding model for this
  media type.
- **Learned** — Trains a small neural net on your current good/bad
  votes and ranks items by its predictions. Needs at least one
  good vote and one bad vote before it works.
- **Load** — Apply a previously saved detector (or one exported
  from another VTSearch instance). Opens a modal to pick a
  detector file or an example media item to sort by.

You can freely switch modes — votes and the model persist across
switches.

### 2. Selection strategy

Picks *which unlabeled item* the app highlights next.

- **Top** — Pick the highest-ranked unlabeled item. Best for
  quickly finding strong matches.
- **Hard** — Pick the item closest to the decision boundary.
  These uncertain cases improve model accuracy fastest.
- **New** — Pick an item from an underexplored region of the
  dataset using diversity sampling. Ensures broad coverage.

Autopilot cycles through these automatically in its four phases,
but in Manual mode you choose directly.

### 3. Inclusion slider

A slider from **-10** (strict) to **+10** (lenient), default 0.

Nudges the classification threshold after the learned model runs.
Negative values mean "only call it good if you're very sure" —
fewer positives, higher precision. Positive values mean "include
borderline items" — more positives, higher recall. The slider
re-ranks instantly; you don't need to re-train.

Leave at 0 unless you have a specific precision/recall trade-off
in mind.

---

## View options

The **View** button above the media list opens the view-settings
modal. All preferences are remembered per media type.

- **List vs. grid** — List shows one item per row with rank, score,
  and a preview. Grid shows a wall of thumbnails. Grid is faster
  for images; list is usually better for audio and text.
- **Grid icon size** — XS, S, M, L, XL. Larger icons = fewer per
  screen but more readable.
- **Focus mode** — Click-focus means you select an item by
  clicking it. Hover-focus means just moving your cursor over
  an item selects it (faster for scanning, more mis-clicks).

There's a separate view modal for the right panel (vote piles),
with the same controls.

---

## Region voting on images

When the dataset's embedder is patch-region-aware (DINOv2, DINOv3,
or EUPE with a `_patch` slug — set when the dataset was created),
you can vote **good** on a *region* of the image instead of the
whole image.  This tells the model "this specific part is what I
like", and the learned sort uses that hint to find similar regions
elsewhere in the dataset.

The binary vote experience is **unchanged**: `→` is good, `←` is
bad.  Region voting is opt-in via a modifier key and never gets in
the way of fast keyboard voting.

### Drawing a region

1. **Hold `Shift`** while the focus pane is showing an image.  The
   cursor flips to a crosshair and the normal pan-on-drag gesture
   is suppressed.
2. **Click-drag-release** to draw a rectangle over the region you
   want to vote good on.  After release the rectangle shows 8
   resize handles plus a draggable body, so you can adjust it.
3. **Press `→`** (or click **Good**) to submit a good vote with
   the region attached.

The rectangle is stored in *normalised image coordinates* — it
stays anchored to the same pixels of the image even if you zoom in,
pan, or rotate before voting.  A `Shift`-click without dragging (a
zero-area "click") restores the previously drawn rectangle rather
than discarding it.  `Esc` clears the rectangle without voting.

### Voting bad while a region is drawn

A `←` press while a region is drawn would normally throw the
rectangle away — and drawing a rectangle is real work, so VTSearch
**asks for confirmation**:

- The rectangle pulses red and a hint banner reads
  *"Press ← again to vote no and discard the box, or Esc to keep
  the box."*
- A second `←` confirms — the no-vote fires and the rectangle is
  discarded.
- `Esc`, clicking on the rectangle, drawing a new one, or
  navigating to the next item all cancel the confirmation and keep
  the rectangle.

There is **no timer** — the confirmation state waits as long as you
need.

### What region voting does to the model

Region-voted good examples train the model on the *region* (pooled
from the patch grid) instead of the full image.  Bad votes are
unaffected — VTSearch already treats every bad vote as "no region
in this image is good" regardless of whether you drew a rectangle.

Region voting is image-only.  Audio, text, video, and document
media types have no region affordance.


---

## Dashboard — managing datasets and models

The Dashboard is your inventory view. Two tables stacked vertically
and a pair of action buttons underneath.

- **Datasets** — every dataset on the server. Each row shows
  media type, item count, duplicate count, creation date, origin,
  clipper, and embedder. The **Loaded** column is a toggle:
  click the **×** to load into memory (a checkmark appears when
  it's in). Per-row icon buttons: **Rename** (pencil), **Stats**
  (pie chart), and **Delete** (trash).
- **Detectors** — every saved detector. Each row shows media type,
  training count, whether it's an **autorun** (scored automatically
  during CLI autodetect), last-trained / created dates, and loaded
  state. Per-row icon buttons: **Rename**, **Add Labels** (import
  labels into this detector), **Export**, and **Delete**.

**Starting a labeling session:** click a dataset row and a model
row to select them, then click the **Train** button in the action
bar below the two tables. That opens the three-panel labeling view
against your selection.

**Scoring a dataset:** select a dataset and a model, then click
**Find** in the action bar. VTSearch scores every item in the
dataset with the model and opens a ranked results modal.

You can keep multiple datasets and multiple models loaded at once.
Loading just pulls them into memory; the Train / Find buttons work
on whichever rows you currently have selected.

---

## Exporting your work

From the Labeling view, the right panel's **Export** button saves
your current labels. Formats:

- **Clipboard** — copies a JSON list of `{id, label, score}` to
  your clipboard.
- **JSON file (server)** — saves to a file on the server via the
  JSON exporter.
- **CSV file (server)** — same but in CSV.
- **Webhook** — POSTs the result to a URL you configure.
- **Email (SMTP)** — emails the result if SMTP is configured.

You can also export **detector weights** from the Models dashboard —
useful for sharing a trained classifier with another VTSearch
instance, or for running it from the CLI. See [CLI.md](CLI.md) for
command-line autodetect.

---

## Importing pre-trained detectors

Two ways to bring in existing work:

- **Labels** — the right panel's **Import Labels** button reads
  a JSON or CSV of `{md5, label}` pairs and populates your vote
  piles from it. Useful for continuing labelling across sessions
  or merging work from multiple labellers.
- **Detectors (models)** — the **Load** sort mode and the Models
  dashboard both have "import detector" options. A detector file
  contains the trained model weights plus the threshold and
  metadata needed to score a new dataset. Once imported, you can
  use it for Load-sort or for autorun scoring.

---

## Where to go next

- Running workflows without the browser — [CLI.md](CLI.md).
- What the ML actually does — [ML.md](ML.md).
- Measuring sort quality on demo datasets — [EVAL.md](EVAL.md).
- How Autopilot and sort modes are implemented — the
  [architecture doc](ARCHITECTURE.md) (developer-oriented).
