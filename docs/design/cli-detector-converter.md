# Design: CLI Autodetect with Converters and Clippers

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

The detector file already stores `media_type`. Extend it to also store the converter and clipper it was trained with:

```json
{
  "name": "bird songs",
  "media_type": "audio",
  "weights": {"0.weight": [...], ...},
  "threshold": 0.42,
  "input_spec": {
    "converter": null,
    "clipper": "sound_tiling_2s"
  }
}
```

Or for a detector trained on audio extracted from video:

```json
{
  "input_spec": {
    "converter": "video2audio",
    "clipper": "sound_tiling_2s"
  }
}
```

Or for a detector trained on whole images (no conversion, default clipper):

```json
{
  "input_spec": {
    "converter": null,
    "clipper": null
  }
}
```

At autodetect time, the pipeline reads each detector's `input_spec` and automatically applies the right converter and clipper *per detector*, before scoring. No CLI args needed. No settings duplication.

## Recommended Design: Detector Input Spec

### 1. Detector metadata gains `input_spec`

Add two optional fields to the detector data dict:

| Field | Type | Meaning |
|-------|------|---------|
| `input_spec.converter` | `str \| null` | Converter name (e.g. `"video2audio"`) used to transform source media into the detector's `media_type`. `null` means no conversion — the dataset is already the right type. |
| `input_spec.clipper` | `str \| null` | Clipper name (e.g. `"sound_tiling_2s"`) used to split media into sub-clips before embedding. `null` means the default clipper (whole media). |

These fields describe what the detector *expects*, not a user preference. They're captured automatically at training time from whatever converter/clipper the user had active.

### 2. Training captures the input spec

When `export_detector` or `export_detector_server` trains a detector, record:
- The active converter (if the current dataset was loaded via conversion)
- The active clipper (if the current dataset was loaded with clipping)

This is already implicit in the current training flow — the embeddings come from whatever pipeline produced the loaded medias. We just need to make it explicit by storing the names.

### 3. CLI autodetect pipeline applies input specs

The `_run_pipeline` function gains a new step between "load dataset" and "score":

```
Load dataset (raw medias)
  ↓
For each detector:
  ↓
  Does detector.input_spec.converter match the dataset's media type?
    YES → dataset type == detector.media_type, no conversion needed
    NO  → apply converter to transform medias
  ↓
  Does detector.input_spec.clipper exist?
    YES → apply clipper to split medias into clips, embed each clip
    NO  → use medias as-is (default clipper)
  ↓
  Score clips against detector
  ↓
  Merge clip-level scores back to media-level results
```

Key: each detector can trigger a *different* converter+clipper chain. The pipeline handles this per-detector, not globally.

### 4. Backwards compatibility

- `input_spec` is optional. Detectors without it behave exactly as today: direct media-type match, no conversion, default clipper.
- Existing detector JSON files continue to work unchanged.
- The in-memory `autorun_detectors` dict gains the same optional `input_spec` key.

### 5. Detector matching becomes richer

Currently `get_autodetect_detectors_by_media("video")` only returns detectors with `media_type == "video"`. With input specs, a detector with `media_type: "audio"` and `input_spec.converter: "video2audio"` should *also* match a video dataset.

New matching logic:

```python
def get_autodetect_detectors_for_dataset(dataset_media_type: str):
    """Return detectors that can score this dataset's media type.

    A detector matches if:
    1. detector.media_type == dataset_media_type (direct match), OR
    2. detector.input_spec.converter converts dataset_media_type
       into detector.media_type (converter match)
    """
```

### 6. Score aggregation for clipped media

When a clipper produces N clips from one media item, we get N scores. We need a single per-media score for the results. Options:

- **Max score** (default): media is positive if *any* clip scores above threshold. Good for "find the needle" use cases (e.g., any 2s segment of this video contains a bird song).
- **Mean score**: average across clips. Good for "overall quality" use cases.
- **Configurable**: store the aggregation method in `input_spec.clip_aggregation` (default `"max"`).

### 7. What the CLI looks like

For the common case, nothing changes:

```bash
# Detector knows it needs video2audio + sound_tiling_2s
python app.py --autodetect \
  --importer http_archive --url https://example.com/videos.zip --media-type video \
  --settings settings.json \
  --exporter server_json_file --filepath results.json
```

The pipeline reads each detector's `input_spec`, applies the right converter+clipper, scores, and exports. The user doesn't think about converters or clippers at the CLI level.

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
| Should we specify converters/clippers in the CLI? | No — the detector's `input_spec` drives it automatically. Optional `--override-*` flags for power users. |
| Should a detector remember its training converter/clipper? | Yes — as `input_spec`, describing what input format the detector expects. This *is* the right default for autodetect. |
| Should we have per-detector settings for converter/clipper? | No — the info lives in the detector itself, not in settings. It travels with the detector file. |
| Per-detector settings for EACH detector‽ | No. The detector carries its own input spec. Different detectors in the same autodetect run automatically use different converter/clipper chains. |
