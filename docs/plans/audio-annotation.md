# Plan: Audio Event Annotation Tool for ALM Evaluation

> **Status:** Phase 0 in design (corpus exploration + microvent importer). All other phases
> deferred. See *§Open follow-ups* for the full roadmap.

## Goal

Build a tool to efficiently annotate audio events across a large video+audio corpus
(microvent/multivent-raw) for the purpose of constructing an evaluation benchmark for
audio language models (ALMs). "Efficient" means: use embedding-based retrieval and active
learning so annotators spend time labeling events that exist, not scanning silent/irrelevant
chunks.

The annotation unit is a **10-second window** keyed by `(chunk_id, t_offset)` — the natural
granularity of GLAP's pre-computed embeddings. Annotation schema captures:
- **Presence**: yes / no / uncertain
- **Specificity**: exact class / parent class only / off-branch
- **Parent class**: if specificity = parent (free text or class picker)
- **Notes**: free text
- **Modality used**: audio / video / ASR / metadata (checkboxes)
- **Annotator**: identity (username, no auth system)

Multi-annotator support is required: small team of 2–5, with disagreement surfacing for adjudication.

---

## Corpora

| Corpus | Path | Chunks | Hours | Audio embeddings? |
|---|---|---|---|---|
| **microvent** | `/exp/scale26/datasets/microvent/` | 943 | ~26 h | Yes — GLAP (1024-dim) + CLAP (512-dim) pre-computed |
| **multivent-raw** | `/exp/scale26/datasets/multivent-raw/` | 143,288 | ~5,353 h | No — must batch-compute |

Microvent has the full embedding zoo: GLAP, CLAP, SigLIP2, Qwen3-VL, Jina, LCO-Omni-7B,
OmniEmbed, Omni-Nemotron, OCR text, ASR. It is the development set for everything.
WebDataset `.tar` shards throughout; `videos/catalog.csv` is the master chunk index.

GLAP window format: `<chunk_id>.audemb_glap.npz` with `keyframe_ids` = `['t000000', 't000010', ...]`
(one per 10-second window) and `embeddings` shape `(N_windows, 1024)` float32.

---

## Phase 0 — Corpus Exploration (current focus)

**Goal:** Understand what audio events exist before committing to a ~20-class taxonomy.

### 0a. Microvent importer plugin for VTSearch

Add a `MultiventImporter` (or `MicroventImporter`) dataset importer plugin to VTSearch.
VTSearch already has the UMAP hexbin browser (VTSBrowse) and text-query retrieval (GLAP
encoder). Loading microvent into VTSearch gives free access to:
- VTSBrowse — UMAP of ~14,000 audio windows in 2D embedding space, hover-to-preview
- Text queries — "thunder", "crowd cheering", "car engine" → ranked results with audio preview
- Detector training — vote a few examples → MLP reranker → find more

The importer does NOT re-embed. It reads the pre-computed GLAP `.npz` files and injects
embeddings directly into the dataset embedding matrix, bypassing the normal embedder path.

**Implementation steps:**

1. Add `MicroventImporter` in `vtsearch/datasets/importers/` (standard importer plugin).
2. Importer reads `videos/catalog.csv` to enumerate chunks, builds `(chunk_id, t_offset)`
   window list.
3. Extracts `.m4a` audio from `audio/shard_*.tar` members into a temp/cache dir for playback.
4. Loads GLAP `.npz` from `embeddings/audemb_glap/shard_*.tar` → reads `keyframe_ids` and
   `embeddings` arrays.
5. Creates one `Media` object per `(chunk_id, t_offset)` window; origin = tar member path.
6. Injects pre-computed embedding matrix into the dataset context instead of calling the
   GLAP embedder.
7. Optionally also loads the pre-built UMAP projection (if available in a `.npz`) to skip
   the ~10-min VTSBrowse build step.

**Key constraint:** Per CLAUDE.md, embeddings must stay in-memory only — no serialization
to disk outside of the existing dataset `.pkl` format (which is by design a snapshot that
includes embeddings). The importer may produce a standard VTSearch `.pkl` file as its output.

### 0b. Taxonomy discovery workflow

With microvent loaded:
1. Use VTSBrowse to explore the UMAP — identify clusters visually.
2. Use text queries to probe candidate class names.
3. Use fast detector training (vote ~20 items → MLP) to find more examples of promising classes.
4. Document findings; commit to a class list (target: ~20 audio event classes).

**Outcome:** A committed class list with rough prevalence estimates in microvent.

---

## Phase 1 — Annotation Backend

A **standalone Flask app** (`audio_annotator/`) reusing `vtscore.training.mlp` for
active learning. Separate from VTSearch because the annotation schema, query-queue logic,
and multi-annotator model are distinct enough that extension would be more work.

### Key components

**A. Embedding index (startup)**
- Load all GLAP `.npz` files for the target corpus into memory:
  - `embeddings`: `(N_windows, 1024)` float32 numpy array
  - `window_ids`: parallel list of `(chunk_id, t_offset)` tuples
  - `shard_index`: dict `chunk_id → (shard_path, tar_member_name)` for O(1) media extraction
- Microvent: ~14,000 windows, ~56 MB. Multivent-raw: ~2M windows, ~8 GB (SLURM only).

**B. Retrieval engine**
- Text query → GLAP text encoder → cosine similarity over all window embeddings → ranked list.
- After MLP training: replace or blend cosine score with MLP output score.
- Secondary signals (later): SigLIP2 / Qwen3-VL visual embeddings, ASR keyword search,
  metadata filters (duration, language, source type).

**C. Active learning (per class)**
- Per annotation class, maintain in-memory:
  - `labels`: `{window_id: 1|0}` (uncertain excluded from training)
  - `model`: trained MLP or `None`
- After each annotation batch, retrain via `vtscore.training.mlp.build_model` + `train_model`.
  MLP input = 1024-dim GLAP embedding; output = relevance score.
- No weights persisted to disk — relabeled from JSONL labels on restart.
- Queue re-ranked after each retrain.

**D. Annotation store**
- JSONL on `/exp/mfox/`, one record per annotation action:
  ```json
  {
    "chunk_id": "FhjGDGegf-sk6nqT_0000",
    "t_start": 20,
    "t_end": 30,
    "class_id": "thunder",
    "presence": "yes",
    "specificity": "exact",
    "parent_class_id": null,
    "notes": "",
    "annotator": "mfox",
    "query_used": "thunder rumble",
    "score_at_annotation": 0.82,
    "modality_used": ["audio"],
    "timestamp": "2026-06-05T14:32:00Z"
  }
  ```
- On startup: replay JSONL to reconstruct per-class label sets, retrain MLPs.

**E. Video/audio server**
- Microvent prototype: pre-extract all 943 `.mp4` files to `/exp/mfox/` at setup time.
  Serve with Flask `send_file` + `Range` header for seekable playback.
- Show ±15 seconds of context around the target 10-second window.
- Multivent-raw at scale: on-demand tar extraction via shard index.

---

## Phase 2 — Annotation UI

Minimal browser frontend (Flask-served HTML + vanilla JS; no framework needed at this scale).

**Per-window display:**
- Video player with the 10-second window highlighted (time markers)
- GLAP similarity score badge and color heatmap on waveform
- ASR transcript excerpt for surrounding context
- Keyframe thumbnail nearest the window

**Annotation form:**
- Presence: Yes / No / Uncertain (radio, required)
- Specificity (if presence ≠ No): Exact / Parent-only / Off-branch
- Parent class: free text (shown if specificity = Parent-only)
- Notes: free text
- Modality used: checkboxes (Audio / Video / ASR / Metadata)
- Skip: moves to next without labeling

**Session controls:**
- Class selector
- Annotator username (no auth system)
- Queue position and stats (N labeled / N positives / N remaining)
- "Retrain MLP" button (or auto-trigger every N annotations)

**Multi-annotator features:**
- Shared queue (any annotator takes next item) or annotator assignment
- Flag windows where two annotators disagree on presence → show side-by-side for adjudication

---

## Phase 3 — Export

REST endpoints:
- `GET /api/export/<class_id>` → all annotations as JSONL
- `GET /api/export/<class_id>/positives` → confirmed yes + uncertain with `(chunk_id, t_start, t_end)`
- `GET /api/stats` → per-class counts (positives / negatives / uncertain / unlabeled)

Formats: JSONL (primary), CSV (spreadsheet review), classification format for ALM evaluation
ingestion (`chunk_id, class, label`).

---

## Phase 4 — Scale to multivent-raw

Multivent-raw has no pre-computed audio embeddings. Batch SLURM job to compute GLAP:

```bash
#!/bin/bash -l
#SBATCH --partition=gpu --gres=gpu:l40s:1 --cpus-per-task=8 --mem=64G --time=12:00:00
source /exp/mfox/.venv/bin/activate
python compute_glap_embeddings.py \
  --data /exp/scale26/datasets/multivent-raw \
  --output /exp/mfox/embeddings/multivent-raw-glap \
  --batch-size 64
```

Expected time at L40S: 12–24 hours for 143k chunks.

Annotation server as a CPU SLURM job:
```bash
#SBATCH --partition=cpu --cpus-per-task=4 --mem=32G --time=10-00:00:00
python audio_annotator/server.py --port 8765 --data /exp/scale26/datasets/multivent-raw
```

Annotators SSH-tunnel in: `ssh -L 8765:<compute_node>:8765 login1.hltcoe.jhu.edu`

---

## Build order

| Step | What | Estimated time |
|---|---|---|
| **0a** | Microvent importer for VTSearch | 1–2 days |
| **0b** | Taxonomy discovery (corpus exploration with VTSBrowse + text queries) | 1–2 days |
| **1a** | Embedding loader + retrieval engine | 1 day |
| **1b** | Annotation store + queue manager | 1 day |
| **2** | Annotation UI (Flask + HTML/JS) | 2–3 days |
| **1c** | Active learning loop (MLP retraining) | 1 day |
| **2b** | Multi-annotator support + disagreement surfacing | 1 day |
| **3** | Export endpoints | 0.5 days |
| **4** | Batch GLAP embedding compute + multivent-raw scale-up | 1–2 days |

Total: ~10–14 days for a production-ready prototype on microvent; ~2 more days to scale.

---

## Claude model recommendation

Use **`claude-sonnet-4-6`** for LLM-assisted annotation steps (e.g. suggesting which class
a window belongs to given ASR + keyframe context). Analyzing a 10-second ASR + image to
suggest one of ~20 audio event labels is a moderate reasoning + vision task — Sonnet's
3× cost advantage over Opus matters at the throughput of thousands of windows. If
Sonnet misses subtle distinctions (thunder vs. rain on metal roof), upgrade specific
calls to `claude-opus-4-8`.

Use Haiku (`claude-haiku-4-5`) only for a cheap binary pre-screen ("is there likely a
non-speech audio event in this window?") to discard silent/speech-only windows before
routing to Sonnet.

---

## Open follow-ups

- [ ] **0a: Microvent importer** — primary current task; see §Phase 0 above.
- [ ] **Taxonomy**: ~20 classes not yet committed; discovery via Phase 0b.
- [ ] **Temporal sub-bounds**: annotation schema has `t_start`/`t_end` but UI for marking
      sub-bounds within a 10-second window is not designed. Start with 10-second window
      granularity only.
- [ ] **Multi-source retrieval**: metadata filter + ASR keyword + visual embeddings deferred
      to Phase 1 follow-up after GLAP retrieval is working.
- [ ] **Batch GLAP compute for multivent-raw** (Phase 4) — blocked on Phase 2 being stable.
- [ ] **Importer for multivent-raw** — identical to microvent importer once GLAP embeddings
      exist; trivially extended.
- [ ] **LLM-assisted labeling** — using Claude to pre-label windows before human review
      is a natural follow-up to reduce annotation load; design deferred.
