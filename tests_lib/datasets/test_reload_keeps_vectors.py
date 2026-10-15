"""A reloaded dataset holds the vectors the importing process held, and trains the same line (#4892).

Ingest stores every vector L2-normalized, and the pickle loader normalizes what
it reads again.  Until #4892 the second pass moved the last bit of about a third
of the vectors, so a dataset reloaded after a restart held other bytes than the
process that imported it.  Nothing reads a last bit as a quantity, but the
calibration folds' split sizes were then dithered by an RNG seeded from the
training vectors' bytes.  With ten labels at a 0.3 calibration share each
class trained on 3.5 votes, so the two processes rounded the half differently,
held out different votes, and the labels' separation (d') read 2.69 in one and
0.13 in the other on the same labels.  Since #4864 the split sizes read no
bytes (each class holds out its nearest whole share), but the folds' draws and
the head still do, so the vectors must still come back as they went in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vtscore.datasets.loader import export_dataset_to_file
from vtscore.datasets.loader_pickle import load_dataset_from_pickle
from vtscore.embedding.normalize import l2_normalize
from vtscore.state.core import DetectorContext

#: Width of the fixture's vectors, as SigLIP's.
DIM = 768
#: The labels: five Goods and five Bads of one class, as the issue's, so the split
#: is fractional (5 x 7/10 = 3.5 trained per class), which the dither decided then.
GOOD = range(0, 5)
BAD = range(5, 10)


def _imported_medias() -> dict[int, dict[str, Any]]:
    """Forty medias as an import leaves them in memory: each vector normalized at ingest."""
    rng = np.random.default_rng(4892)
    centre = rng.standard_normal(DIM)
    medias: dict[int, dict[str, Any]] = {}
    for cid in range(40):
        # The first ten are one tight class (the labels), the rest the corpus around them.
        raw = centre + (0.3 if cid < 10 else 1.5) * rng.standard_normal(DIM)
        medias[cid] = {
            "id": cid,
            "media_type": "image",
            "embedder": "siglip",
            "duration": 0,
            "file_size": 1,
            "md5": f"{cid:032x}",
            "embeddings": {"siglip": l2_normalize(raw)},
            "media_bytes": b"x",
            "filename": f"drawing_{cid:04d}.png",
            "category": "test",
            "origin": {"importer": "test", "params": {}},
            "origin_name": f"drawing_{cid:04d}.png",
        }
    return medias


def _reloaded(medias: dict[int, dict[str, Any]], tmp_path: Path) -> dict[int, dict[str, Any]]:
    pkl = tmp_path / "drawings.pkl"
    pkl.write_bytes(export_dataset_to_file(medias, embedder="siglip", media_type="image"))
    loaded: dict[int, dict[str, Any]] = {}
    load_dataset_from_pickle(pkl, loaded)
    return loaded


def _labels_line(medias: dict[int, dict[str, Any]]):
    from vtscore.detectors.training import train_and_score

    ctx = DetectorContext("det-4892", media_type="image")
    train_and_score(
        medias,
        dict.fromkeys(GOOD),
        dict.fromkeys(BAD),
        calibrate_count=2,
        calibration_fraction=0.3,
        det_ctx=ctx,
        beta=1.0,
    )
    return ctx


class TestAReloadedDataset:
    def test_holds_the_importing_process_vectors_bit_for_bit(self, tmp_path: Path):
        imported = _imported_medias()
        reloaded = _reloaded(imported, tmp_path)
        by_name = {m["filename"]: m for m in reloaded.values()}
        for media in imported.values():
            again = by_name[media["filename"]]["embeddings"]["siglip"]
            assert again.tobytes() == media["embeddings"]["siglip"].tobytes(), media["filename"]

    def test_trains_the_same_folds_and_separation(self, tmp_path: Path):
        imported = _imported_medias()
        reloaded = _reloaded(imported, tmp_path)
        assert sorted(reloaded) == sorted(imported), "the fixture's labels are named by id in both"

        first, again = _labels_line(imported), _labels_line(reloaded)

        assert first.labels_line is not None and again.labels_line is not None
        assert first.calibration_cache is not None and again.calibration_cache is not None
        folds_first = first.calibration_cache[1].orderings
        folds_again = again.calibration_cache[1].orderings
        assert [len(s) for s, _ in folds_again] == [len(s) for s, _ in folds_first], "the same votes held out"
        assert folds_again == folds_first
        assert again.labels_line.separation == first.labels_line.separation
