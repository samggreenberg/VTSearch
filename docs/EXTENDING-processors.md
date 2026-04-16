# Extending VTSearch — Processor System

How to add new detectors, localizers, and extractors — the three kinds
of **Processor** in VTSearch. All three subclass `Processor` from
`vtsearch/media/base.py`.

**Related docs:** [EXTENDING.md](EXTENDING.md) (index, checklists, auth,
dependencies) · [EXTENDING-plugins.md](EXTENDING-plugins.md) (importers,
exporters, sources) · [EXTENDING-media.md](EXTENDING-media.md) (media
types, embedders, clippers, converters).

## Contents

- [Processor System](#processor-system) — shared `Processor` base
- [Adding a Detector](#adding-a-detector) — "does this media match?" → bool
- [Adding a Localizer](#adding-a-localizer) — "where inside this media?" → regions
- [Adding an Extractor](#adding-an-extractor) — "what structured data can
  we pull out?" → list of dicts

---

## Processor System

Processors analyze media items. The hierarchy has a common base
(`Processor`) with three concrete subtypes. All are defined in
`vtsearch/media/base.py`.

```
Processor (ABC)
├── Detector      — "does this media match?"      → bool
├── Localizer     — "where in this media?"        → list[dict] (bounding boxes)
└── Extractor     — "what details are inside?"    → list[dict] (structured results)
```

Each processor operates on exactly one media type.

### Adding a Detector

A Detector answers "is this media Good?" with a boolean.

```python
from vtsearch.media.processors import Detector
from typing import Any


class LoudnessDetector(Detector):

    @property
    def name(self) -> str:
        return "loud_audio"

    @property
    def media_type(self) -> str:
        return "audio"

    def load_model(self) -> None:
        """Optional: load heavyweight resources once before first use."""
        pass

    def detect(self, media: dict[str, Any]) -> bool:
        """Return True if the media matches, False otherwise."""
        return media.get("duration", 0) > 5.0
```

### Adding a Localizer

A Localizer returns bounding boxes with confidence scores.

```python
from vtsearch.media.processors import Localizer
from typing import Any


class FaceLocalizer(Localizer):

    @property
    def name(self) -> str:
        return "face_localizer"

    @property
    def media_type(self) -> str:
        return "image"

    def localize(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        """Return bounding boxes. Each dict must include:
        - "confidence": float in [0, 1]
        - "bbox": bounding box (format is media-specific)

        Returns empty list when nothing is found.
        """
        # ... face detection logic ...
        return [
            {"confidence": 0.95, "bbox": [10, 20, 200, 300]},
            {"confidence": 0.73, "bbox": [400, 50, 600, 250]},
        ]
```

### Adding an Extractor

An Extractor returns structured details for each occurrence found.

```python
from vtsearch.media.processors import Extractor
from typing import Any


class ObjectExtractor(Extractor):

    @property
    def name(self) -> str:
        return "object_extractor"

    @property
    def media_type(self) -> str:
        return "image"

    def extract(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        """Return a list of result dicts. Each dict must include
        a "confidence" key (float in [0, 1]).

        Returns empty list when nothing is found.
        """
        # ... object detection logic ...
        return [
            {"confidence": 0.92, "bbox": [10, 20, 200, 300], "label": "car"},
            {"confidence": 0.87, "bbox": [400, 50, 600, 250], "label": "tree"},
        ]
```

### Processor abstract interface reference

**Processor (base class) — required:**

| Member | Type | Description |
|--------|------|-------------|
| `name` | `str` (property) | Unique identifier |
| `media_type` | `str` (property) | Which media type it operates on |
| `process(media)` | `(dict) -> Any` | Run the processor (delegates to subclass) |

**Processor (base class) — optional:**

| Member | Type | Description |
|--------|------|-------------|
| `load_model()` | `() -> None` | One-time model loading (default: no-op) |

**Detector:**

| Member | Signature | Description |
|--------|-----------|-------------|
| `detect(media)` | `(dict) -> bool` | Return True if media matches |

**Localizer:**

| Member | Signature | Description |
|--------|-----------|-------------|
| `localize(media)` | `(dict) -> list[dict]` | Return bounding boxes with `confidence` and `bbox` |

**Extractor:**

| Member | Signature | Description |
|--------|-----------|-------------|
| `extract(media)` | `(dict) -> list[dict]` | Return result dicts with `confidence` key |

---

