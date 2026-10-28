"""One four-photo session with its hard starting face swapped for another photo of the person (#5376).

A K = 4 session whose worst starting face has Face Quality under #5372's cut (14) reaches F 0.80 at 50
votes at beta 4, against 0.93 for the rest. This runs one cell of such a session with every hard face
among its four examples swapped for the next photo of the same person in the example draw's own order
(:func:`vtscore.eval.example_opening.choose_examples`) that is not hard. If the voting half holds no
such photo, the hard face is dropped and the session starts from three. Everything else in the cell
is as the run dir sets it, so a control rerun of the same cell (``--swap off``) pairs with it.

    python hard_face_5376.py --index N [--swap on|off]

Run from ``scripts/experiments/calibration`` in a run dir's environment (``sota_face.sh cell`` with
``SOTA_FACE_RUNNER`` pointing here). The swap is logged as a JSON line on stderr
(``hard_face_5376 swap: ...``), so the analysis reads what each session started from.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

#: #5372's cut on buffalo_s: a face under it is a hard view.
HARD_VIEW_QUALITY = 14.0

CALIB = Path(__file__).resolve().parents[1] / "calibration"


def _quality(dataset: str) -> dict[int, float]:
    pile = os.environ.get("VTS_PILE_CELLS") or str(
        Path(os.environ.get("VTS_FHIBE_ROOT", f"/expscratch/{os.environ['USER']}/fhibe"))
        / f"{os.environ.get('VTS_FHIBE_RELEASE', 'fhibe-full-resolution-674a7dcf')}-derived"
        / "cells"
    )
    with open(Path(pile) / f"{dataset}__face.pkl", "rb") as f:
        medias = pickle.load(f)  # noqa: S301  (our own pile cell)
    return {int(i): float(m["custom_metadata"]["Face Quality"]) for i, m in medias.items()}


def swapped(original, quality: dict[int, float], log: dict):
    """*original* (choose_examples) with every hard example swapped for the next clear one in the draw."""

    def choose(sim_ids, clips_dict, target_category, k, seed):
        from vtscore.eval.labels import media_is_positive  # noqa: PLC0415

        n_pos = sum(1 for cid in sim_ids if media_is_positive(clips_dict[cid], target_category))
        order = original(sim_ids, clips_dict, target_category, n_pos, seed)  # the whole draw, in order
        first = order[:k]
        spare = [c for c in order[k:] if quality[c] >= HARD_VIEW_QUALITY]
        out = []
        for c in first:
            if quality[c] >= HARD_VIEW_QUALITY:
                out.append(c)
            elif spare:
                out.append(spare.pop(0))
        log.update(
            category=target_category,
            original=first,
            original_q=[quality[c] for c in first],
            examples=out,
            examples_q=[quality[c] for c in out],
            voting_positives=n_pos,
        )
        print("hard_face_5376 swap: " + json.dumps(log), file=sys.stderr, flush=True)
        return out

    return choose


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--swap", choices=("on", "off"), default=os.environ.get("HARD_FACE_SWAP", "on"))
    args = ap.parse_args(argv)
    sys.path.insert(0, str(CALIB))
    import common  # noqa: PLC0415
    import experiment_config as cfg  # noqa: PLC0415
    import run_cells  # noqa: PLC0415

    if args.swap == "on":
        import vtscore.eval.voting_iterations as vi  # noqa: PLC0415

        info = json.loads((common.RESULTS / "prepare_info.json").read_text())
        cell = cfg.array_cells(run_cells._categories_by_dataset(info))[args.index]
        vi.choose_examples = swapped(vi.choose_examples, _quality(cell["dataset"]), {"index": args.index})
    return run_cells.main(["--index", str(args.index)])


if __name__ == "__main__":
    sys.exit(main())
