# vtscore

The Flask-free, app-free core of [VTSearch](https://github.com/samggreenberg/vtsearch).
A reusable Python library for trainable media search: dataset origins,
MediaSources, clippers, embedders, detector training and scoring,
and evaluation. The companion `vtsearch` Flask + Angular application wraps
this library with the HTTP / SPA / settings layer; everything described
here works without it.

## Documentation

Developer documentation lives under [`docs/`](docs/README.md); its
[Start here](docs/README.md#start-here) list is the reading order
(quickstart → concepts → tutorials → integration → package reference →
extending). Begin with the **[Quickstart](docs/quickstart.md)**.

## Install

`vtscore` ships from the same repository as `vtsearch`. Install in
editable mode for development:

```bash
git clone https://github.com/samggreenberg/vtsearch
cd vtsearch
bash scripts/install.sh   # auto-detects CPU vs GPU
```

A standalone `vtscore` PyPI distribution is deferred until a real
external consumer asks for it. For now, install the repo and import
`vtscore` directly.

## Plugins

Dataset, datasource and seed importers, results exporters, label
importers, labelset sources, converters and media sources are discovered from module sentinels inside `vtscore` and
from `importlib.metadata` entry points under `vtscore.<family>` groups, so
a separate distribution can add one without touching this repository. The
entry point names the **instantiated** plugin (the sentinel), not its class:

```toml
[project.entry-points."vtscore.importers"]
my_importer = "my_package.importer:IMPORTER"
```

Media types, embedders, clippers and cleaners have no entry-point group.
See [docs/extending/](docs/extending/README.md) for the family table and
per-family guides.

## Conventions

No persisted vectors or model weights, no hardcoded `data/` paths, no Flask
or `vtsearch` imports: the rules every package follows are listed once, in
[docs/README.md § Conventions](docs/README.md#conventions).

## Versioning

Independent semver (`vtscore.__version__`); see
[docs/README.md § Versioning](docs/README.md#versioning) and
[`CHANGELOG.md`](CHANGELOG.md).

## License

Apache License 2.0, same as the parent `vtsearch` project - see the
repository's [`LICENSE`](../LICENSE) file.

Two carve-outs matter if you are embedding `vtscore` in your own product:

- **Model weights carry their own terms.** `vtscore` downloads embedding
  models at runtime rather than vendoring them, and each publisher licenses
  its own weights. Some are noncommercial (EUPE, under the FAIR
  Noncommercial Research License) or gated (DINOv3). An embedder whose
  weights carry a usage restriction says so through its
  `MediaEmbedder.license_notice` property (EUPE sets it; the DINOv3
  embedders currently do not, so check the publisher's terms yourself).
- **Two dependencies are AGPL-3.0, and are skippable**: `ultralytics` (image
  extractor and clipper) and `PyMuPDF` (PDF importer and document
  converters). A default install includes both. The Apache-2.0 grant on
  `vtscore` itself is unaffected, but AGPL terms may attach to a combined
  work you distribute or operate as a service — so if you are shipping a
  closed product on top of `vtscore`, install without them:
  `VTSEARCH_NO_AGPL=1 bash scripts/install.sh`, or
  `pip install -r requirements/base-no-agpl.txt`. Both packages live in the
  `agpl` extra, which every default install path requests; dropping it
  leaves a permissively licensed install in which those four features raise
  an actionable error instead of running (see
  `vtscore/utils/optional_deps.py`).

See [`NOTICE`](../NOTICE) for the full statement.
