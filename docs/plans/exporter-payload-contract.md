# Design: the exporter payload contract

Open follow-ups from the results-exporter payload-kind rework
([#3219](https://github.com/samggreenberg/VTSearch/issues/3219)).

## Background

An exporter is a **destination**; what gets sent there is a separate axis, with
three payload kinds — `find_results` (a scored run), `labelset` (a detector's
labels), and `detector_bundles` (the trained classifiers). `ResultsExporter`
carries a method per kind, and `supported_payloads` is *derived* from which
methods a subclass overrides rather than declared, so each picker offers an
exporter only for the kinds it can actually read. See
`vtscore/exporters/base.py` and
[`vtscore/docs/extending/results-exporters.md`](../../vtscore/docs/extending/results-exporters.md).

Two constraints shaped it, and both still bind anything built on top:

- **The ABC is not split, and shouldn't be.** Destination and payload are
  orthogonal, so two base classes would mean two registry entries and two
  picker rows per destination (or a mixin that recreates the shared base).
  `SettingsExporter` stays a genuinely separate ABC because it shares no
  registry, route, modal, or field semantics; labelset and find-results share
  all of them, plus five of nine payload keys.
- **The legacy `export()` path is permanent.** Out-of-tree plugins implement it
  and sniff the dict shape themselves; the named methods delegate to it, so
  they keep working untouched. Do not delete that delegation to tidy the base
  class up.

## Open work

<!-- item-sep -->

- **Decide what `negative_hits` means to an exporter.** The find-results
  payload has carried `negative_hits` since the route was written, the CLI
  emits them under `keep_negatives`, and **no exporter reads the key** — every
  export is positives-only. That is now *documented* in
  `export_find_results`'s contract, which was the minimum; the behavioural
  question is still open. Either leave it conventional, or give exporters a way
  to opt in (a `PluginField` on the file/webhook exporters, or a second records
  stream for the streaming path). Worth settling deliberately rather than by
  the accident of what the first exporter happened to read.

<!-- item-sep -->

- **Un-hide the `holder` exporter.** It is `hidden_from_picker = True  # flip to
  False once API clients are implemented`, and that comment is now the only
  thing keeping it back: as a labelset-only exporter it is filtered out of the
  find-results pickers by construction, so flipping the flag can no longer
  expose it to a payload it cannot read. Blocked on the Holder API clients, not
  on the contract.

<!-- item-sep -->

- **Per-AutoRun exporter selection.** The second half of
  [#3219](https://github.com/samggreenberg/VTSearch/issues/3219): let a detector
  moved to AutoRun carry its own exporter choice, falling back to the global
  Auto-Find setting when unset. Not designed here. The open design question is
  the one the issue thread raises — whether an exporter preference is really
  per-detector or per-user — and it is worth answering before building the UI.

<!-- item-sep -->

- **No labelset streaming mode.** `export_cli_streaming` is find-results only,
  and deliberately so: the NDJSON label stream in
  `vtsearch/routes/labels/vote.py` is a built-in route rather than a plugin
  path, so a plugin-side twin has no caller today. If a labelset export ever
  needs to outgrow RAM, that is the shape to add — not a widening of the
  existing streaming method, whose records are `(detector_name, hit)` tuples.

<!-- item-sep -->
