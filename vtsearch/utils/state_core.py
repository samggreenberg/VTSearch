"""Core state variables and lock shared by all state submodules.

This module defines the mutable global state and the reentrant lock that
protects it.  All other ``state_*.py`` submodules import variables from
here rather than defining their own.
"""

from __future__ import annotations

import threading
from typing import Any

# Reentrant lock protecting all mutable state.
# RLock is used because some public functions call other public functions
# (e.g. clear_all -> clear_medias + clear_votes).
_state_lock = threading.RLock()

# Clips storage: id -> {id, type, duration, file_size, embedding, media_bytes, media_string, ...}
medias: dict[int, dict[str, Any]] = {}

# Optional display-name override for the loaded dataset.  When set, the
# dashboard shows this instead of the name derived from origin info.
_dataset_display_name: str | None = None

# Diversity tree: built from media embeddings after a dataset loads.
# ``None`` until a dataset is loaded and the tree is constructed.
_diversity_tree: Any = None  # DiversityTree | None

# Voting storage (OrderedDict behavior via dict in Python 3.7+)
good_votes: dict[int, None] = {}
bad_votes: dict[int, None] = {}

# Combined label history: [(media_id, label, timestamp), ...]
# Tracks the order of all labels across both categories
label_history: list[tuple[int, str, float]] = []

# Click-time tracking: media_id -> click order (1-indexed).
# Assigned when a vote is cast via the API; labels loaded via import get no entry
# (the frontend treats missing entries as time=-1).
vote_click_times: dict[int, int] = {}
_click_counter: int = 0

# Last learned-sort scores: media_id -> score (float in [0, 1]).
# Updated each time /api/learned-sort completes.
last_learned_scores: dict[int, float] = {}

# Inclusion setting: -10 to +10, default 0.
# ``None`` means "not yet loaded"; on first access the value is read from the
# persisted settings file so that it survives restarts.
inclusion: int | None = None

# Text-sort suggestions: text queries that received a Good vote, most recent last.
textsort_suggestions: list[str] = []

# Autorun detectors: name -> {name, media_type, weights, threshold, created_at}
autorun_detectors: dict[str, dict[str, Any]] = {}

# Autorun extractors: name -> {name, extractor_type, media_type, config, created_at}
autorun_extractors: dict[str, dict[str, Any]] = {}

# Autorun localizers: name -> {name, localizer_type, media_type, config, created_at}
autorun_localizers: dict[str, dict[str, Any]] = {}
