"""Per-media prediction dumps: the evidence behind an aggregate fpr/fnr.

An aggregate error rate says *how often* a configuration is wrong. It cannot say
whether the **model** is wrong or the **label** is, and those two have opposite
remedies: one is a modelling problem, the other means the dataset needs
cleaning and the study needs re-running. Separating them requires looking at the
individual medias, so every study that reports an error rate should be able to
show the errors themselves.

A dump is one row per scored media: its score, the label the dataset carries,
the threshold in force, the source file (so the image can be opened), and every
category the dataset annotates on that media (so "the label is missing" is
checkable by eye). `scripts/experiments/calibration/error_report.py` turns a
dump into the ranked false-positive / false-negative listings a report quotes,
and `label_noise.py` runs the entailment test over it.

Both the clicked-detector path (`voting_iterations`, gated on the
``VTS_DUMP_TEST_SCORES`` environment variable) and the typed-query baseline
(`scripts/experiments/calibration/text_baseline.py --dump-dir`) write through
this one writer, so the two are always the same schema and the same analysis
scripts read both.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from vtscore.io import atomic_write_stream

COLUMNS = (
    "media_id",
    "filename",
    "score",
    "label",
    "threshold",
    "target_category",
    "all_categories",
)


def write_prediction_dump(
    path: Path,
    medias: dict[int, dict[str, Any]],
    ids: list[int],
    scores: Any,
    labels: Any,
    threshold: float,
    target_category: str,
) -> None:
    """Write one row per media in *ids* to *path* (created via a temp rename).

    *scores* and *labels* must be positionally aligned with *ids*; the write goes
    through :func:`~vtscore.io.atomic_write_stream`, so a reader never sees a
    half-written dump from a re-run in flight and two arms dumping the same path
    can't collide on the temp file.
    """
    with atomic_write_stream(path) as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for media_id, score, label in zip(ids, scores, labels, strict=True):
            media = medias[media_id]
            categories = media.get("categories") or []
            writer.writerow(
                [
                    media_id,
                    media.get("filename") or media.get("origin_name") or "",
                    float(score),
                    int(label),
                    float(threshold),
                    target_category,
                    "|".join(str(c) for c in categories),
                ]
            )


def maybe_dump_predictions(
    medias: dict[int, dict[str, Any]],
    ids: list[int],
    scores: Any,
    labels: Any,
    threshold: float,
    target_category: str,
    suffix: str = "",
) -> None:
    """Dump predictions when ``VTS_DUMP_TEST_SCORES`` names an output directory.

    Off unless the environment asks for it: a run that dumps writes one file per
    step (each overwriting the last, so what survives is the final step's
    state), which is evidence for a handful of hand-picked cells, not something
    a 270-cell array should be doing.  ``VTS_DUMP_TAG`` names the file.
    """
    out_dir = os.environ.get("VTS_DUMP_TEST_SCORES")
    if not out_dir:
        return
    tag = os.environ.get("VTS_DUMP_TAG", "cell")
    write_prediction_dump(
        Path(out_dir) / f"{tag}{suffix}.csv",
        medias,
        ids,
        scores,
        labels,
        threshold,
        target_category,
    )
