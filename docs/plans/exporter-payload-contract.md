# Design: the exporter payload contract

Proposed rework of the results-exporter plugin contract so that "which payload
an exporter accepts" is a fact the framework knows, rather than something each
plugin rediscovers at runtime with a dict-key sniff. Motivated by
[#3219](https://github.com/samggreenberg/VTSearch/issues/3219).

Nothing here has shipped; this file is the design under review.

## Background

One ABC, `LabelsetExporter` (`vtscore/exporters/base.py`), currently serves
three structurally different payloads:

| Payload kind | Shape | Produced by | Reaches the plugin via |
|---|---|---|---|
| Find results | `{media_type, detectors_run, results: {det: {hits, negative_hits, threshold, total_hits}}, missing_detectors}` | `POST /api/auto-detect` (`vtsearch/routes/detectors/scoring.py`), the Auto-Find autorun export in the same module, CLI `--autodetect` | `export()` |
| Labelset | `{labels: [LabeledElement], selected_columns: [...]}` | the Export modal (`frontend/src/app/components/modals/export-modal/`) → `POST /api/exporters/export` | `export()` — the same method |
| Trained detectors | `list[descriptor]` | the CLI pipeline only | `export_cli_detectors()`, gated by the `needs_trained_detectors` boolean |

The first two are told apart by a runtime shape sniff **inside each plugin** —
`if "labels" in results` in `server_csv_file`, `server_json_file`, `webhook`,
`open_url`, and `gui`. The third kind already uses the shape this design
generalises (a declared capability plus its own method), and is the reason not
to add a second boolean: the pattern stops paying at three kinds.

### Why the sniff has to go

It is not a tidiness complaint. Three things are wrong today:

- **`email_smtp` is silently broken in the Export modal.** It never sniffs — it
  reads only `results.get("results", {})` — but both pickers list the same
  unfiltered registry, so "Send by Email" is offered for a labelset export. It
  sends `Subject: VTSearch Auto-Detect: 0 hit(s) on unknown dataset` with an
  empty body, returns a success dict, and the modal fires its "Exported N
  labels" toast. No test passes a labels payload to it.
- **`holder` is the mirror bug, pre-armed.** It reads `results.get("labels")`
  only, and is held back solely by `hidden_from_picker = True  # flip to False
  once API clients are implemented`. Flipping that comment ships the same
  failure into the Auto-Find picker.
- **The base class documents one payload of the three, and gets that one
  wrong.** `LabelsetExporter.export()`'s docstring describes the find-results
  shape without `negative_hits` or `missing_detectors`, and does not mention the
  labelset shape at all. The prose docs (`docs/EXTENDING-plugins.md`,
  `vtscore/docs/extending/results-exporters.md`) *do* explain both shapes, which
  is the wrong way round: an extension author reads the ABC.

A documentation fix cannot reach any of this, because every failure is a silent
success. Only a contract can.

### How different the two payloads actually are

Less than the issue assumes, and that is what rules out a hard ABC split.
`build_media_hit` (`vtscore/utils/hits.py`) emits `md5`, `origin`,
`origin_name`, `filename`, `category` plus clip fields; `LabeledElement`
(`vtscore/datasets/labelset.py`) carries the same five. The delta is
`score` / `threshold` / `total_hits` / `id` on the find-results side against
`region_box` / `metadata` / `selected_columns` on the labelset side.

So the payloads differ at their edges — and the five shared keys are precisely
the ones that make an export re-importable. The *delivery* machinery (SMTP
session, atomic write, path templating, browser-URL validation, streaming batch
sizing) is identical across both.

## The design

### One base class, three named payload methods

Keep a single registry entry per destination. Replace `export()` with three
methods, each taking the payload it is named for:

```python
class ResultsExporter(PluginBase):
    def export_find_results(self, results, field_values) -> dict: ...
    def export_labelset(self, labelset, field_values) -> dict: ...
    def export_detector_bundles(self, detectors, field_values) -> dict: ...
```

Base implementations raise `UnsupportedPayloadError` — a `ValueError` subclass,
so `POST /api/exporters/export`'s existing `except ValueError → 400` turns an
unsupported pairing into a clean rejection instead of a 500. The message names
the exporter and the kind.

**A hard ABC split is the wrong axis.** Destination and payload are orthogonal.
Two ABCs mean either two classes per destination — two registry entries, two
picker rows both labelled "Server CSV File", two saved field-value blobs keyed
by exporter name in settings — or a mixin arrangement that recreates the shared
base just deleted. Contrast `SettingsExporter`
(`vtsearch/settings_io/exporters/base.py`), which *is* a separate ABC and
should stay one: it shares no registry, route, modal, or field semantics with
results exporters. Labelset and find-results share all of them.

### Support is derived, never declared

`supported_payloads` is computed from which methods the subclass actually
overrides, not from a class attribute the author sets:

```python
@classmethod
def supported_payloads(cls) -> frozenset[str]: ...  # compares cls.export_X to ResultsExporter.export_X
```

A declared flag is a second place to forget something, and forgetting is the
present bug's whole mechanism. A derived set cannot drift from the
implementation. (A subclass of a concrete exporter inherits its parent's
overrides, which is the correct answer.)

It is serialised on `to_dict()` beside `opens_url`, so `GET /api/exporters`
states each exporter's capability, and `ExporterEntrySchema` grows the field.

### The route names the kind; it never guesses

`RunExportRequestSchema` gains a required `payload_kind`
(`"find_results" | "labelset"`), and the permissive `results` body field is
renamed `payload`. No default value — a default is a sniff with extra steps.
The Export modal sends `labelset`; the Auto-Detect Results modal and the
Auto-Find autorun path send `find_results`. The CLI dispatches on
`supported_payloads()` rather than on `needs_trained_detectors`.

### Both pickers filter

The Export modal lists exporters whose `supported_payloads` contains
`labelset`; the Auto-Find settings picker (and the Auto-Detect Results modal)
lists `find_results`. `holder` then becomes safe to un-hide. A saved Auto-Find
exporter that no longer qualifies is reported through the
`{exporter, success: false, error}` status block `_run_autofind_export` already
returns for an unknown exporter — a scored find is too valuable to sink over a
misconfigured export.

### No framework-level payload adapter

Tempting, and wrong. The one adapter that exists today —
`gui._labelset_to_display` — fabricates `media_type: "labels (3 good, 1 bad)"`
and `detectors_run: 0`, which is display-only fiction that reads fine in a modal
and would read as nonsense in an email subject or a CSV header. A generic
adapter would let an exporter *appear* to support a mode it does not understand:
the empty-email failure with a nicer wrapper. Each exporter implements the modes
it means.

### Streaming stays a find-results mode

`supports_streaming` / `export_cli_streaming` only ever carried
`(detector_name, hit)` records, so it is find-results streaming and should say
so: rename to `export_find_results_streaming`. There is deliberately no labelset
streaming mode — the NDJSON label stream in `vtsearch/routes/labels/vote.py` is a
built-in route, not a plugin path, and inventing a plugin-side twin is scope this
design does not need.

### Naming

`LabelsetExporter` → `ResultsExporter`, matching what every doc already calls the
family ("Results exporters") and retiring a name its own guide apologises for
("named for historical reasons"). `_PLUGIN_NAME_SUFFIXES` in
`vtscore/plugins/__init__.py` swaps `"LabelsetExporter"` for `"ResultsExporter"`,
kept ahead of the bare `"Exporter"` fallback so `HolderResultsExporter` still
derives `holder`. Concrete classes rename to match; every in-tree one declares
`name` explicitly, so no registry key moves.

### Where each in-tree exporter lands

| Exporter | Supports | Change |
|---|---|---|
| `server_csv_file` | find results, labelset | split the existing sniff into the two methods |
| `server_json_file` | find results, labelset | same |
| `webhook` | find results, labelset | same |
| `open_url` | find results, labelset | same |
| `gui` | find results, labelset | same; `_labelset_to_display` becomes the body of `export_labelset` |
| `email_smtp` | find results, labelset | **gains** a labelset mode — the issue's actual use case |
| `holder` | labelset | can be un-hidden once its API clients exist |
| `portable_detector` | detector bundles | `needs_trained_detectors` deleted |

## Open work

<!-- item-sep -->

- **Confirm the naming before any code moves.** `ResultsExporter` is the
  proposal; the issue floats `LabelsetExporter` + `FindResultsExporter` as two
  ABCs, which this design rejects on the orthogonal-axes argument above. The
  rename touches `_PLUGIN_NAME_SUFFIXES`, eight concrete classes, both
  extension guides, and the schema docstrings, so it is cheap to decide now and
  tedious to revisit later.

<!-- item-sep -->

- **Decide what `negative_hits` means to an exporter.** The find-results payload
  has carried `negative_hits` since the route was written, the CLI emits them
  under `keep_negatives`, and **no exporter reads the key** — every export is
  positives-only, and nothing says so. Either document the drop in
  `export_find_results`'s contract, or give exporters a way to opt in (a
  `PluginField`, or a second records stream). This is a real behavioural
  question, not a docstring chore, and it should be settled while the contract
  is open rather than bolted on after.

<!-- item-sep -->

- **Promote the implementation slices to issues once the design is agreed.**
  Per `CLAUDE.md`, concrete shippable work lives in issues, not in a plan body.
  The natural cuts are: the ABC plus derived capability; the route and schema
  change with its OpenAPI snapshot regeneration; the two picker filters; the
  per-exporter method splits; `email_smtp`'s new labelset mode with the
  regression test that is missing today; and the doc pass across the ABC
  docstring, `docs/EXTENDING-plugins.md`, and
  `vtscore/docs/extending/results-exporters.md`. Leave one-line `#N` pointers
  here if the umbrella is worth keeping.

<!-- item-sep -->

- **Per-AutoRun exporter selection.** The second half of
  [#3219](https://github.com/samggreenberg/VTSearch/issues/3219), not designed
  here, but it depends on this one: a per-AutoRun picker is only well-defined
  once "which exporters accept a find-results payload" is something the API
  states.

<!-- item-sep -->

## Notes for whoever implements it

- **Third-party entry-point exporters break.** `vtscore.exporters` is a public
  entry-point group, and an out-of-tree plugin implementing `export()` stops
  working. Per `CLAUDE.md` that is acceptable and gets no shim — but the break
  should be *loud*: with the base `export()` gone, the framework sees an empty
  `supported_payloads()`, so the plugin is filtered out of both pickers and
  rejected by the route with a message naming the missing methods, rather than
  half-working.
- **Gates to re-run beyond the suite:** regenerate the OpenAPI snapshot
  (`cd frontend && npm run regenerate-openapi-snapshot`) for the schema change.
  The plugin-family counts in the generated doc inventories are unchanged (eight
  exporters stay eight). Exporters are not mirrored by
  `scripts/check-eval-app-sync.py`, so that gate is untouched.
- **`vtscore/CHANGELOG.md`** gets an `[Unreleased] › Changed` entry for the
  library-tier breaking change; the `vtscore.__version__` bump waits for a
  release cut, per `CLAUDE.md`.
