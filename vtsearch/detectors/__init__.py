"""Detector lifecycle: registry, store, training-glue, labels, and the
labeling-session analyzer.

This package owns the resolve→embed→train pipeline that turns origin trails
into a trained MLP plus calibrated thresholds. The neural-net training
primitives themselves still live under ``vtsearch.models`` (until step 2 of
the codebase reorg splits those out too).
"""
