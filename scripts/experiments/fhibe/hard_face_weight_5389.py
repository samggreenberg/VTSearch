"""One session cell with a hard-view starting face down-weighted in the example centroid (#5389).

#5376 found that one hard-view face (Face Quality under #5372's cut) among clear starting faces costs
a four-photo session F 0.10-0.11 at 50 votes at beta 4, and that leaving it out of the session
recovers the loss. This runs one cell with every hard-view example weighing ``--weight`` in the
example sort's centroid (:func:`vtscore.eval.example_opening.example_sort`) while every clear one
weighs 1, as long as at least one example is clear. The hard face still counts as a Good vote.
``--weight 1`` is the control (the plain mean), so arms that differ only in the weight pair cell for
cell. The study found the weight does not help, and the app does not take it.

    python hard_face_weight_5389.py --index N [--weight W]

Run from ``scripts/experiments/calibration`` in a run dir's environment (``sota_face.sh cell`` with
``SOTA_FACE_RUNNER`` pointing here, ``HARD_FACE_WEIGHT`` the weight). The examples' verdicts and
weights are logged as a JSON line on stderr (``hard_face_weight_5389: ...``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CALIB = Path(__file__).resolve().parents[1] / "calibration"


def weights(hard: list, weight: float) -> list[float] | None:
    """Each example's weight: *weight* for a hard view beside a clear one, else 1; ``None`` when all are 1."""
    if weight == 1.0 or not any(h is False for h in hard) or not any(h is True for h in hard):
        return None
    return [weight if h is True else 1.0 for h in hard]


def weighted(original, weight: float, index: int):
    """*original* (``example_sort``) with the centroid weighted by :func:`weights`; the plain call when unweighted."""

    def example_sort(clips_dict, examples):
        import numpy as np  # noqa: PLC0415

        from vtscore.eval.example_opening import _unit  # noqa: PLC0415
        from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415
        from vtscore.media.face.imagekit_bridge import media_hard_view  # noqa: PLC0415

        hard = [media_hard_view(clips_dict[c]) for c in examples]
        w = weights(hard, weight)
        log = {"index": index, "weight": weight, "examples": list(examples), "hard": hard, "weights": w}
        print("hard_face_weight_5389: " + json.dumps(log), file=sys.stderr, flush=True)
        if w is None:
            return original(clips_dict, examples)
        ids = list(clips_dict)
        units = np.stack([_unit(media_embedding(clips_dict[c])) for c in examples])
        query = _unit(np.average(units, axis=0, weights=w))
        matrix = np.stack([_unit(media_embedding(clips_dict[c])) for c in ids])
        cos = matrix @ query
        return {cid: float(cos[k]) for k, cid in enumerate(ids)}

    return example_sort


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--weight", type=float, default=float(os.environ.get("HARD_FACE_WEIGHT", "1")))
    args = ap.parse_args(argv)
    sys.path.insert(0, str(CALIB))
    import run_cells  # noqa: PLC0415

    import vtscore.eval.voting_iterations as vi  # noqa: PLC0415

    vi.example_sort = weighted(vi.example_sort, args.weight, args.index)
    return run_cells.main(["--index", str(args.index)])


if __name__ == "__main__":
    sys.exit(main())
