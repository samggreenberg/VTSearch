"""Detector lifecycle: registry, store, training-glue, labels, and the
labeling-session analyzer.

This package owns the resolve→embed→train pipeline that turns origin
trails into a trained MLP plus calibrated thresholds. The generic
neural-net primitives it builds on (MLP/SVM training, threshold
computation) live under :mod:`vtsearch.training`; embedding façades and
torch-runtime helpers live under :mod:`vtsearch.embedding`.
"""
