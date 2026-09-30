# vtscore

The Flask-free, app-free core of [VTSearch](https://github.com/samggreenberg/vtsearch).
A reusable Python library for trainable media search: dataset origins,
MediaSources, clippers, embedders, detector training and scoring,
and evaluation. The companion `vtsearch` Flask + Angular application wraps
this library with the HTTP / SPA / settings layer; everything described
here works without it.

## Documentation

Developer documentation lives under [`docs/`](docs/README.md), which is the
index. Read it in this order:

1. **[Quickstart](docs/quickstart.md)** - load a folder, train a detector, score new media. Start here.
2. **[Concepts](docs/concepts.md)** - `Media`, `Origin`, `LabelSet`, `Embedding`, `Context`, detector, plugin. The vocabulary every other doc assumes.
3. **[Tutorials](docs/tutorials/README.md)** - longer end-to-end walkthroughs.
4. **[Integration](docs/integration.md)** - the hooks to install when embedding `vtscore` in your own application.
5. **[Package reference](docs/README.md#package-reference)** - one guide per subpackage; the canonical inventory of what each exports.
6. **[Extending vtscore](docs/extending/README.md)** - authoring guides for the plugin families.

[Architecture](docs/architecture.md) (layering, threading, the real import
path of every package) and the [FAQ](docs/faq.md) are reference material to
consult as needed.

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

## Quickstart

There is deliberately no copy-paste snippet here: the shortest version that
actually runs needs a `CoreConfig`, an origin stamp and an explicit embed
step, and [docs/quickstart.md](docs/quickstart.md) walks through exactly
that. Its snippets are executed by
`tests_lib/integration/test_docs_quickstart.py`, so they stay honest.

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

`vtscore.__version__` is independent semver, bumped manually in
`vtscore/__init__.py` on each release (the companion `vtsearch` app uses a
git-derived timestamp instead). See [`CHANGELOG.md`](CHANGELOG.md) for
per-release notes.

## License

Apache License 2.0, same as the parent `vtsearch` project - see the
repository's [`LICENSE`](../LICENSE) file.

Two carve-outs matter if you are embedding `vtscore` in your own product:

- **Model weights carry their own terms.** `vtscore` downloads embedding
  models at runtime rather than vendoring them, and each publisher licenses
  its own weights. Some are noncommercial (EUPE, under the FAIR
  Noncommercial Research License) or gated (DINOv3). Embedders with a
  restriction expose it through their descriptor's `license_notice` field.
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
