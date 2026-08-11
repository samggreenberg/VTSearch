# Extending VTSearch: Processor System

How to add new detectors, localizers, and extractors (the three kinds
of **Processor** in VTSearch). All three subclass `Processor` from
`vtscore/media/processors.py`.

Subclassing is necessary but **not sufficient**: unlike the auto-discovered
plugin families, a processor is only reachable once it is added to a hardcoded
factory dict in the app. Read [Registering a Processor with the
App](#registering-a-processor-with-the-app) alongside the recipes below.

**Related docs:** [EXTENDING.md](EXTENDING.md) (index, checklists, auth,
dependencies) · [EXTENDING-plugins.md](EXTENDING-plugins.md) (importers,
exporters, sources) · [EXTENDING-media.md](EXTENDING-media.md) (media
types, embedders, clippers, converters).

## Contents

- [Processor System](#processor-system): shared `Processor` base
- [Adding a Detector](#adding-a-detector): "does this media match?" → bool
- [Adding a Localizer](#adding-a-localizer): "where inside this media?" → regions
- [Adding an Extractor](#adding-an-extractor): "what structured data can
  we pull out?" → list of dicts
- [Registering a Processor with the App](#registering-a-processor-with-the-app):
  the `from_config` contract and the hardcoded factory dicts — **required**,
  the class alone is not reachable

---

## Processor System

Processors analyze media items. The hierarchy has a common base
(`Processor`) with three concrete subtypes. All are defined in
`vtscore/media/processors.py`.

```
Processor (ABC)
├── Detector      ("does this media match?")      → bool
├── Localizer     ("where in this media?")        → list[dict] (bounding boxes)
└── Extractor     ("what details are inside?")    → list[dict] (structured results)
```

Each processor operates on exactly one media type.

> **Processors are not auto-discovered.** Unlike the nine plugin families in
> [EXTENDING-plugins.md](EXTENDING-plugins.md), there is no registry scan and
> no module-level sentinel here: the app instantiates processors through two
> **hardcoded factory dicts** in `vtsearch/routes/processors/crud.py`. Writing
> the subclass below is only half the job — a class that isn't in a factory
> dict cannot be built by any endpoint. See [Registering a Processor with the
> App](#registering-a-processor-with-the-app) for the other half.

### Adding a Detector

A Detector answers "is this media Good?" with a boolean.

> **Detectors have no app wiring at all.** No concrete `Detector` subclass
> ships in the tree, there is no detector factory dict, and no endpoint builds
> or runs one — the ABC exists so the `Processor` hierarchy is complete, and a
> subclass is usable only from code you call yourself. If you want an ML
> classifier the app can actually run, you want a **detector in the VTSearch
> sense** (a trained ranking head registered via `POST /api/detectors/registry`
> — see [ML.md](ML.md) and [docs/api/detectors.md](api/detectors.md)), which is
> an unrelated concept that happens to share the word. The two registrable processor kinds are `Localizer` and
> `Extractor`.

```python
from vtscore.media.processors import Detector
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
from vtscore.media.processors import Localizer
from typing import Any


class FaceLocalizer(Localizer):

    def __init__(self, name: str, threshold: float = 0.5) -> None:
        self._name = name
        self._threshold = threshold

    @property
    def name(self) -> str:
        return self._name

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

That covers the ABC. To make it reachable from the API it also needs a
`from_config` classmethod and an entry in `_LOCALIZER_FACTORIES` — see
[Registering a Processor with the App](#registering-a-processor-with-the-app).
`vtscore/media/image/face_localizer.py` is the one in-tree example.

### Adding an Extractor

An Extractor returns structured details for each occurrence found.

```python
from vtscore.media.processors import Extractor
from typing import Any


class ObjectExtractor(Extractor):

    def __init__(self, name: str, threshold: float = 0.25) -> None:
        self._name = name
        self._threshold = threshold

    @property
    def name(self) -> str:
        return self._name

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

Note that `name` is an instance attribute, not a hardcoded string: the app
names each processor from the API request, so the constructor must accept a
`name`. The next section explains why, and what else this class still needs
before any endpoint can build it.

### Processor abstract interface reference

**Processor (base class) - required members:**

| Member | Type | Description |
|--------|------|-------------|
| `name` | `str` (property) | Unique identifier |
| `media_type` | `str` (property) | Which media type it operates on |

`process(media)` is declared on the base, but each subtype
(`Detector`/`Localizer`/`Extractor`) already provides a **concrete**
`process()` that delegates to `detect()` / `localize()` / `extract()`.
When you subclass one of those subtypes you implement only the
subtype-specific method below — you do **not** override `process()`.

**Processor (base class) - optional members:**

| Member | Type | Description |
|--------|------|-------------|
| `load_model()` | `() -> None` | One-time model loading (default: no-op) |
| `to_dict()` | `() -> dict` | Metadata for API responses (default: `name` + `media_type`) |

**Not on the ABC, but required by the app** (see
[Registering a Processor with the App](#registering-a-processor-with-the-app)):

| Member | Signature | Description |
|--------|-----------|-------------|
| `from_config(name, config)` | `classmethod (str, dict) -> Self` | Build an instance from an API request body |

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

## Registering a Processor with the App

A `Localizer` or `Extractor` subclass is inert until the app can **build** it.
The app never looks at your class hierarchy: it maps a `localizer_type` /
`extractor_type` **string** from the request body to a factory callable, using
two module-level dicts in `vtsearch/routes/processors/crud.py`. A type string
that isn't a key in the relevant dict raises `ValueError` inside
`_build_extractor` / `_build_localizer` — which surfaces as an HTTP 400
(`Invalid extractor config: Unknown extractor_type: 'object_class'`) from a
route far away from the class you wrote.

So registration is two steps: give the class a `from_config` classmethod, then
add it to the factory dict.

### Step 1: the `from_config` classmethod

`_build_extractor(name, extractor_type, config)` calls the factory as
`factory(name, config)`. The convention every in-tree processor follows is a
`from_config` classmethod paired with a `to_dict` that emits the same shape, so
a processor round-trips through the API:

```python
    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()                 # {"name": ..., "media_type": ...}
        d["extractor_type"] = "object_class"  # the factory-dict key
        d["config"] = {"threshold": self._threshold}
        return d

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]) -> "ObjectExtractor":
        """Reconstruct an ``ObjectExtractor`` from a saved config dict."""
        return cls(name=name, threshold=config.get("threshold", 0.25))
```

Rules that fall out of how it is called:

- **`name` comes from the request, not the class.** Two autorun entries of the
  same type with different configs are distinct processors distinguished only
  by name, so never hardcode `name` in the class.
- **`config` is unvalidated.** `AutorunExtractorCreateRequestSchema` declares it
  as a bare `fields.Dict`, so whatever JSON the client sent arrives verbatim.
  `from_config` *is* the validation layer: use `config.get(key, default)` for
  optional keys and `config[key]` for required ones. A `KeyError` or
  `ValueError` raised here is caught by the route and returned as a 400, which
  is the intended failure mode — don't swallow it and return a half-built
  processor.
- **A localizer uses `localizer_type` in `to_dict`**, not `extractor_type`.
- **`to_dict` is not what registers anything.** It is metadata for API
  responses; the string it reports is only correct because you also added that
  same string to the factory dict in step 2.

Reference implementations: `vtscore/media/image/extractor.py` (`image_class`),
`vtscore/media/image/ocr_extractor.py` (`ocr`),
`vtscore/media/audio/speech_extractor.py` (`speech`),
`vtscore/media/image/face_localizer.py` (`face`).

### Step 2: add it to the factory dict

Edit `vtsearch/routes/processors/crud.py`. Extractors go in
`_ensure_extractor_factories`, localizers in `_ensure_localizer_factories`.
Both populate their dict on first use and import inside the function body —
that laziness is deliberate (it avoids import cycles and keeps heavyweight
optional dependencies out of app startup), so keep your import inside the
function too:

```python
def _ensure_extractor_factories():
    """Populate the factory registry on first use (lazy to avoid import cycles)."""
    if _EXTRACTOR_FACTORIES:
        return
    # ... existing image_class / ocr / speech entries ...

    from vtscore.media.image.object_extractor import ObjectExtractor

    _EXTRACTOR_FACTORIES["object_class"] = ObjectExtractor.from_config
```

The key you choose here *is* the public `extractor_type` / `localizer_type`
value. It is duplicated in your `to_dict`, and nothing checks the two agree —
a mismatch means the processor builds fine but reports a type string that
cannot be rebuilt.

**Yes, this means adding a processor requires editing app code**, and a
third-party package cannot ship one. That is a real divergence from every other
plugin family in this repo (importers, exporters, converters, media sources, …),
which *are* auto-discovered from a module-level sentinel and can be contributed
out-of-tree via entry points — see
[EXTENDING-plugins.md § PluginRegistry](EXTENDING-plugins.md#pluginregistry-auto-discovery).
Processors predate that mechanism and were never moved onto it. Don't assume
the sentinel pattern works here; it does not.

### Step 3 (optional): offer it as a pregen processor

`_PREGEN_PROCESSORS` further down the same file is a hardcoded list of
ready-to-use processor definitions (name, kind, type, media type, default
config). `GET /api/pregen-processors` returns it, and
`POST /api/pregen-processors/add` registers **all** of them as autorun entries
in one call. Add an entry if your processor should be part of that
one-click bundle; the `processor_type` you use must be a factory-dict key, and
`kind` must be `"extractor"` or `"localizer"`.

### Using the registered processor

With steps 1 and 2 done, the endpoints documented in
[docs/api/io.md](api/io.md#pregen-processors) work:

| Endpoint | Effect |
|----------|--------|
| `POST /api/autorun-extractors` | Validates by building the processor once, then stores `{name, extractor_type, media_type, config}` |
| `POST /api/autorun-localizers` | Same, for localizers |
| `POST /api/extract` / `POST /api/localize` | Build one ad-hoc processor from a body and run it over every loaded media |
| `POST /api/auto-extract` / `POST /api/auto-localize` | Build and run every stored processor whose `media_type` matches the loaded medias |

Four behaviours of this layer are worth knowing before you debug it:

- **Registration is validated, autorun is not.** The `POST /api/autorun-*`
  routes build the processor once purely to reject a bad config with a 400, and
  throw the instance away. `/api/auto-extract` and `/api/auto-localize` rebuild
  each stored processor at request time and **silently skip** any that fails to
  build (`_run_single` swallows the exception and returns `None`), so a
  processor removed from a factory dict later just disappears from the results
  rather than erroring.
- **Instances are per-request.** Nothing caches a built processor, so
  `load_model()` runs again on every call. Cache heavy resources on the
  instance (`if self._model is not None: return`, as the in-tree processors do)
  and expect the load cost once per request.
- **The stored `media_type` is what routes it.** `POST /api/autorun-*` takes
  `media_type` from the request body and never cross-checks it against your
  class's `media_type` property; `/api/auto-extract` selects processors by that
  stored string. (`/api/extract` and `/api/localize` *do* check, and 400 on a
  mismatch.) Send the same value your class reports.
- **The autorun store is in-memory and process-global.** It lives in
  `vtsearch/autorun_processors.py` and is not written to `data/settings.json`
  or anywhere else, so registrations are lost on restart. Nothing runs autorun
  processors implicitly on media load either — despite the name, they run only
  when `/api/auto-extract` or `/api/auto-localize` is called.

---

