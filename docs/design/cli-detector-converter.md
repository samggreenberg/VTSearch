# Design: CLI Autodetect with Converters and Clippers

> **Status: Design proposal (not yet implemented).** The converter registry
> exists in `vtsearch/converters/` and is fully functional for dataset
> import-time conversion. However, the `input_spec` field on detector JSON
> and the CLI autodetect pipeline changes described here are **not yet
> implemented**. This document captures the design for future work.

## Problem

The CLI autodetect pipeline currently works like this:

1. Load dataset (pickle or importer) → medias with pre-computed embeddings
2. Find detectors matching the dataset's media type
3. Score embeddings against detectors
4. Export results

This breaks down when:

- A detector was trained on **audio extracted from video** (video→audio converter + sound_tiling_2s clipper), but the new dataset is raw video files.
- A detector was trained on **2s audio clips**, but the dataset contains full-length audio files.
- A detector was trained on **document page images**, but the dataset contains PDFs.

The detector expects a specific *input format* (media type + clipper granularity), and the pipeline has no way to reproduce that format at inference time.

## Options Considered

### Option A: Specify converter/clipper as CLI arguments

```
python app.py --autodetect --dataset videos.pkl \
  --converter video2audio --clipper sound_tiling_2s \
  --settings settings.json
```

**Problems:**
- These args apply globally to *all* detectors in the autodetect run. If one detector was trained on 2s clips and another on 5s clips, you can't express that.
- The user has to remember (or look up) what each detector was trained with.
- Gets tangled fast with multiple detectors, each needing different converters/clippers.

### Option B: Per-detector settings in the settings file

```json
{
  "autorun_processors": [
    {
      "processor_name": "bird songs",
      "processor_importer": "server_detector_file",
      "field_values": {"filepath": "data/detectors/birds.json"},
      "converter": "video2audio",
      "clipper": "sound_tiling_2s"
    }
  ]
}
```

**Problems:**
- Duplicates information that should live *with* the detector.
- Settings become a second source of truth that can drift from training reality.
- If you share a detector file, the converter/clipper info doesn't travel with it.

### Option C: Detector stores its input spec (Recommended)

The detector file already stores `media_type`. That field already tells the pipeline what embedding space the detector operates in. The missing piece is the *clipper* — how to split media before embedding.

**Converters are NOT part of the detector's spec.** The detector doesn't care *how* you got to its media type — it just needs the right type of embedding. The converter registry already knows all the routes (video→image, document→image, video→audio, etc.), so the pipeline can figure out conversion automatically.

This matters because **a dataset can contain mixed source types**. An image detector should score:
- Native images → directly
- Videos → via `video2image`
- Documents → via `document2image`

If the detector stored `converter: "video2image"`, it couldn't also handle documents. The detector shouldn't have to enumerate every possible source type — it just says "I need image embeddings" and the pipeline handles the rest.

```json
{
  "name": "page quality",
  "media_type": "image",
  "weights": {"0.weight": [...], ...},
  "threshold": 0.42,
  "input_spec": {
    "clipper": null
  }
}
```

Or for a detector trained on 2s audio clips:

```json
{
  "name": "bird songs",
  "media_type": "audio",
  "weights": {"0.weight": [...], ...},
  "threshold": 0.42,
  "input_spec": {
    "clipper": "sound_tiling_2s"
  }
}
```

At autodetect time, the pipeline reads the detector's `media_type` (for converter lookup) and `input_spec.clipper` (for clipping), then applies the right chain *per media item*. No CLI args needed. No settings duplication.

## Recommended Design: Detector Input Spec

### 1. Detector metadata gains `input_spec`

Add one optional field to the detector data dict:

| Field | Type | Meaning |
|-------|------|---------|
| `input_spec.clipper` | `str \| null` | Clipper name (e.g. `"sound_tiling_2s"`) used to split media into sub-clips before embedding. `null` means the default clipper (whole media). |

The detector's existing `media_type` field already declares the target embedding space. **Converters are not stored on the detector** — the pipeline uses the converter registry (`list_converters_for_target(detector.media_type)`) to automatically find routes from any source type to the detector's target type. This means an image detector automatically handles video→image, document→image, and native images without enumerating converters.

The `clipper` field describes what the detector *expects*, not a user preference. It's captured automatically at training time from whatever clipper the user had active.

### 2. Training captures the input spec

When `export_detector` or `export_detector_server` trains a detector, record:
- The active clipper (if the current dataset was loaded with clipping)

The converter is NOT recorded — it's handled automatically by the registry at inference time. Only the clipper matters because it affects the granularity of the embeddings the detector was trained on.

This is already implicit in the current training flow — the embeddings come from whatever pipeline produced the loaded medias. We just need to make the clipper explicit by storing its name.

### 3. CLI autodetect pipeline applies input specs

The `_run_pipeline` function gains a new step between "load dataset" and "score":

```
Load dataset (raw medias — may contain mixed types: video, document, image, etc.)
  ↓
For each detector:
  ↓
  Group medias by source type
  ↓
  For each source type:
    Same as detector.media_type?
      YES → use directly
      NO  → look up converter via list_converters_for_target(detector.media_type)
            filtered to source_type. If found, apply converter.
            If no converter exists for this source type, skip these medias.
  ↓
  Does detector.input_spec.clipper exist?
    YES → apply clipper to split converted medias into clips, embed each clip
    NO  → use medias as-is (default clipper)
  ↓
  Score clips against detector
  ↓
  Merge clip-level scores back to media-level results
```

Key points:
- Each detector can trigger a *different* clipper. The pipeline handles this per-detector, not globally.
- Conversion is driven by the converter registry, not the detector. A single image detector automatically handles video files (via `video2image`), documents (via `document2image`), and native images — all in the same dataset.
- If a media item's source type has no converter route to the detector's target type, it's simply skipped for that detector (not an error).

### 4. Backwards compatibility

- `input_spec` is optional. Detectors without it behave exactly as today: direct media-type match, no conversion, default clipper.
- Existing detector JSON files continue to work unchanged.
- The in-memory `autorun_detectors` dict gains the same optional `input_spec` key.

### 5. Detector matching becomes richer

Currently `get_autodetect_detectors_by_media("video")` only returns detectors with `media_type == "video"`. With converter-aware matching, a detector with `media_type: "audio"` should also match a video dataset — because `video2audio` exists in the converter registry.

New matching logic:

```python
def get_autodetect_detectors_for_dataset(dataset_media_types: set[str]):
    """Return detectors that can score any of the dataset's media types.

    A detector matches if, for at least one source type in the dataset:
    1. detector.media_type == source_type (direct match), OR
    2. A registered converter exists with
       source_type == source_type AND target_type == detector.media_type
       (converter match)

    This is driven entirely by the converter registry — the detector
    doesn't need to know about converters. It just declares its
    media_type, and the registry provides the routes.
    """
```

For mixed-type datasets (e.g., videos + documents + images), the function accepts a set of source types and returns all detectors that can reach any of them. An image detector matches all three because `video2image` and `document2image` exist in the registry, plus native images match directly.

### 6. Score aggregation for clipped media

When a clipper produces N clips from one media item, we get N scores. We need a single per-media score for the results. Options:

- **Max score** (default): media is positive if *any* clip scores above threshold. Good for "find the needle" use cases (e.g., any 2s segment of this video contains a bird song).
- **Mean score**: average across clips. Good for "overall quality" use cases.
- **Configurable**: store the aggregation method in `input_spec.clip_aggregation` (default `"max"`).

### 7. What the CLI looks like

For the common case, nothing changes:

```bash
# Dataset has videos — the pipeline auto-converts via video2audio, video2image, etc.
# based on each detector's media_type. Clipper comes from detector's input_spec.
python app.py --autodetect \
  --importer http_archive --url https://example.com/videos.zip --media-type video \
  --settings settings.json \
  --exporter server_json_file --filepath results.json
```

The pipeline uses the converter registry to bridge source types to each detector's `media_type`, then applies the detector's `input_spec.clipper`. The user doesn't think about converters or clippers at the CLI level.

For power users who want to override:

```bash
# Force a different clipper than what the detector was trained with
python app.py --autodetect \
  --dataset videos.pkl \
  --override-clipper sound_tiling_5s \
  --settings settings.json
```

This is a niche escape hatch, not the primary interface.

## Summary

| Question | Answer |
|----------|--------|
| Should we specify converters in the CLI? | No — the converter registry handles this automatically. The pipeline looks up all converters that produce the detector's `media_type` and applies the right one per source media type. |
| Should we specify clippers in the CLI? | No — the detector's `input_spec.clipper` drives it automatically. Optional `--override-clipper` flag for power users. |
| Should a detector remember its training converter? | No — converters are a property of the source→target type pair, not the detector. The same image detector handles video→image, document→image, and native images via the converter registry. |
| Should a detector remember its training clipper? | Yes — as `input_spec.clipper`, describing what granularity the detector expects. This *is* the right default for autodetect. |
| Should we have per-detector settings for converter/clipper? | No — clipper lives in the detector itself (travels with the file). Converter is automatic from the registry. |
| Per-detector settings for EACH detector‽ | No. Clipper is per-detector via `input_spec`. Converter is per-source-type via registry. No settings explosion. |
