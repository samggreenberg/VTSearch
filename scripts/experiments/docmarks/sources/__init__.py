"""Per-source adapters for the DocMarks corpus.

Each module exposes the same two-part shape:

* ``fetch(raw_root) -> Path`` — network and filesystem; not unit tested.
* pure parsers over what ``fetch`` left on disk — unit tested against fixtures.

That split is what lets the builder be developed and verified somewhere that
cannot reach Kaggle or hold a 3 GB archive, and run unchanged on the cluster.
"""
