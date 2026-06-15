"""Post-import load-pipeline stages.

Each module here holds one cohesive stage of the dataset-load pipeline that
runs *after* an importer has populated a context's ``medias`` dict:

- :mod:`clipper`     — clipper/converter chain + per-clip MD5/embedding fixup
- :mod:`embedding`   — embed media items the importer left unembedded
- :mod:`finalize`    — drop failed embeds, collapse duplicates, diversity tree
- :mod:`projection`  — optional 2-D UMAP projection build + persist
- :mod:`registry`    — save to the dataset registry + context-id migration

The orchestration that strings these together (gate handoff, progress
routing, background threading) stays in
:mod:`vtscore.datasets.load_pipeline`; these modules hold the per-stage work
so the orchestrator reads as a sequence of named steps.
"""
