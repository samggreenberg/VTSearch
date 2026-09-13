#!/usr/bin/env python
"""Capture one cell's **real** fit inputs for the #3585 gate.

    python capture_folds_3585.py --index 0 --out <dir>/cell_0.npz

WHY THIS EXISTS.  #3585 replaces the unanchored GMM fit, and the issue's own
gate is "on a corpus of real fold haystacks, report the distribution of
|delta threshold| and the count of steps where the admitted set changes at
all".  Neither number can be read off a synthetic fixture corpus: what decides
whether a re-initialised EM lands somewhere else is the *shape* of the haystack
it is fitted to, and the shapes a trained detector produces mid-run - saturated,
max-pooled, barely bimodal at click 5 - are the thing under test.

So this re-runs a real cell with a recording wrapper around the two production
entry points that fit the mixture, and writes only the arrays those wrappers
saw.  It changes nothing in the eval tier: both wrappers call straight through
and return the real value, so the cell it runs is the cell the arrays came from.

The gate itself is offline (``gate_3585.py``), which is the point of dumping
arrays rather than comparing in place: a corpus captured once can be replayed
against a candidate that did not exist when it was captured.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402

common.setup_env()

#: Capture every Nth threshold fit rather than all of them.  The fitted mixture
#: moves slowly against the vote count, so consecutive steps are near-duplicate
#: fixtures - and a corpus is only worth what its *variety* of shapes is.
DEFAULT_STRIDE = 5


def _cell_embedder(index: int) -> str:
    """The embedder of cell *index*, from ``run_cells``' own cell enumeration."""
    try:
        import experiment_config as cfg  # noqa: PLC0415

        import run_cells  # noqa: PLC0415

        info = json.loads((common.RESULTS / "prepare_info.json").read_text())
        cells = cfg.array_cells(run_cells._categories_by_dataset(info))
        return str(cells[index]["embedder"])
    except Exception as exc:  # pragma: no cover - a missing label must not lose a capture
        print(f"could not resolve the embedder for cell {index}: {exc}", file=sys.stderr)
        return ""


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, required=True, help="cell index, as `launch_gmm_3585.sh list` prints it")
    ap.add_argument("--out", required=True, help="npz to write")
    ap.add_argument("--outdir", default=None, help="cell CSV dir (default: a throwaway beside --out)")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    args = ap.parse_args(list(argv) if argv is not None else None)

    import vtscore.training.thresholds as T
    from vtscore.eval import voting_iterations as vi

    import run_cells  # noqa: PLC0415

    store: dict[str, np.ndarray] = {}
    ctx: dict[str, object] = {"style": "?", "n": 0, "sort_n": 0}
    meta: dict[str, object] = {"index": args.index, "stride": args.stride}

    original_sim = vi.simulate_voting_iterations
    original_cut = T.fit_fold_anchored_cut
    original_sort = T.calculate_gmm_threshold

    def recording_sim(*a, **kw):
        """Tag the captures with the style, and restart the within-style counter.

        The cell's identity is taken from the arguments the harness passes
        rather than from the launcher's own idea of what cell this is: an index
        means whatever the enumeration meant on the day, and a corpus outlives
        that.
        """
        ctx["style"] = str(kw.get("style") or "whole_image")
        ctx["n"] = 0
        ctx["sort_n"] = 0
        meta.setdefault("dataset", kw.get("dataset_name", ""))
        meta.setdefault("category", kw.get("target_category", ""))
        meta.setdefault("seed", kw.get("seed", -1))
        meta.setdefault("region_voting", bool(kw.get("region_voting", False)))
        styles = list(meta.setdefault("styles", []))  # type: ignore[arg-type]
        if ctx["style"] not in styles:
            styles.append(ctx["style"])
            meta["styles"] = styles
        return original_sim(*a, **kw)

    def recording_cut(fold_haystacks, fold_orderings, final_scores, **kw):
        """The shipped fold-anchored fit: record its whole input, then run it."""
        i = ctx["n"]
        ctx["n"] = i + 1
        if i % args.stride == 0:
            key = f"fold|{ctx['style']}|{i:04d}"
            store[f"{key}|final"] = np.asarray(final_scores, dtype=np.float64).ravel()
            for k, hay in enumerate(fold_haystacks):
                store[f"{key}|hay{k}"] = np.asarray(hay, dtype=np.float64).ravel()
            for k, (a_scores, a_labels) in enumerate(fold_orderings):
                store[f"{key}|anchor{k}"] = np.asarray(a_scores, dtype=np.float64).ravel()
                store[f"{key}|label{k}"] = np.asarray(a_labels, dtype=np.float64).ravel()
        return original_cut(fold_haystacks, fold_orderings, final_scores, **kw)

    def recording_sort(scores):
        """The unanchored one-value entry point - the app's cosine/text sort cut."""
        i = ctx["sort_n"]
        ctx["sort_n"] = i + 1
        if i % args.stride == 0:
            store[f"sort|{ctx['style']}|{i:04d}|scores"] = np.asarray(scores, dtype=np.float64).ravel()
        return original_sort(scores)

    vi.simulate_voting_iterations = recording_sim
    T.fit_fold_anchored_cut = recording_cut
    T.calculate_gmm_threshold = recording_sort
    try:
        outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.out)), "_capture_cells")
        os.makedirs(outdir, exist_ok=True)
        rc = run_cells.main(["--index", str(args.index), "--outdir", outdir])
    finally:
        vi.simulate_voting_iterations = original_sim
        T.fit_fold_anchored_cut = original_cut
        T.calculate_gmm_threshold = original_sort
    if rc != 0:
        return rc

    if not store:
        # Not a warning: a cell with no captures contributes nothing to the gate
        # and would be counted as a captured cell by anything that only looks at
        # the file listing.
        print("captured nothing - did the cell reach a safe-threshold step?", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # The one field the harness call does not carry.  Read off the same
    # enumeration ``run_cells`` indexed into, so the corpus records the cell
    # that ran rather than the cell the launcher believed it asked for.
    meta["embedder"] = _cell_embedder(args.index)
    store["_meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(args.out, **store)
    folds = len({k for k in store if k.startswith("fold|")})
    sorts = len([k for k in store if k.startswith("sort|")])
    print(f"wrote {args.out}: {folds} fold arrays, {sorts} sort arrays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
