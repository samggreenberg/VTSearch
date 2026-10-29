"""One session cell with a hard-view starting face down-weighted in the example centroid (#5389).

#5376 found that one hard-view face (Face Quality under #5372's cut) among clear starting faces costs
a four-photo session F 0.10-0.11 at 50 votes at beta 4, and that leaving it out of the session
recovers the loss. The app now weighs such a face :data:`~vtscore.media.face.imagekit_bridge.
HARD_VIEW_EXAMPLE_WEIGHT` in the example centroid (:func:`~vtscore.eval.example_opening.example_sort`
mirrors it) while it still counts as a Good vote. This runs one cell at another weight, so arms that
differ only in the weight pair cell for cell; ``--weight 1`` is the control (the plain mean).

    python hard_face_weight_5389.py --index N [--weight W]

Run from ``scripts/experiments/calibration`` in a run dir's environment (``sota_face.sh cell`` with
``SOTA_FACE_RUNNER`` pointing here). The examples and their weights are logged as a JSON line on
stderr (``hard_face_weight_5389: ...``), so the analysis reads what each session started from.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CALIB = Path(__file__).resolve().parents[1] / "calibration"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--weight", type=float, default=float(os.environ.get("HARD_FACE_WEIGHT", "1")))
    args = ap.parse_args(argv)
    sys.path.insert(0, str(CALIB))
    import run_cells  # noqa: PLC0415

    import vtscore.eval.example_opening as eo  # noqa: PLC0415
    from vtscore.media.face import imagekit_bridge  # noqa: PLC0415

    imagekit_bridge.HARD_VIEW_EXAMPLE_WEIGHT = args.weight
    original = eo.example_weights

    def logged(hard):
        out = original(hard)
        print(
            "hard_face_weight_5389: "
            + json.dumps({"index": args.index, "weight": args.weight, "hard": list(hard), "weights": out}),
            file=sys.stderr,
            flush=True,
        )
        return out

    eo.example_weights = logged
    return run_cells.main(["--index", str(args.index)])


if __name__ == "__main__":
    sys.exit(main())
