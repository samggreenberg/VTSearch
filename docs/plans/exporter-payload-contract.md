# Design: the exporter payload contract

Open follow-ups from the results-exporter payload-kind rework (#3219).

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

- [ ] #5283 — Decide what find-results `negative_hits` means to an exporter

<!-- item-sep -->

<!-- item-sep -->

- [ ] #5284 — Per-detector AutoFind exporter selection

<!-- item-sep -->

- [ ] #5285 — Labelset streaming export mode, if a labelset outgrows RAM

<!-- item-sep -->
