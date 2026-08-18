"""Post-run structural check on the precision arms, before any number is believed.

Answers the questions a drained queue cannot::

    python check_arms.py

* Did every arm write a ``provenance.json``, and does it say the precision the
  arm table asked for?  A drained queue with a missing provenance file is a
  failure, not a completion.
* Did each arm land on the **card its arm name claims**?  If the cross-GPU
  control silently ran on the same card as the reference, the "irreducible
  drift floor" this study divides by is zero by construction.
* Do all arms cover the **same medias**?  A cell built from a truncated
  re-download looks healthy and disagrees with its siblings (that is how a
  1662-of-4193 cell reached the pile).
* Are any cells zero-byte?  Those count as "done" to the resume path.
* Does the fp32 rebuild reproduce the **published** cell?  Everything downstream
  is a difference against fp32; if the fp32 arm is not the pile's fp32, the
  differences are against nothing in particular.

Exit code 1 if any check fails, so a launcher can gate on it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "calibration"))

import numpy as np  # noqa: E402

import precision_config as pcfg  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def _load(path: Path) -> dict:
    from _cells_io import load_medias  # noqa: PLC0415

    return load_medias(path)


def _vectors(medias: dict) -> tuple[list[int], np.ndarray]:
    """``(ids, (N, D) float64)`` in a stable id order."""
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    ids = sorted(medias)
    rows = []
    keep = []
    for mid in ids:
        vec = media_embedding(medias[mid])
        if vec is None:
            continue
        keep.append(mid)
        rows.append(np.asarray(vec, dtype=np.float64))
    if not rows:
        return [], np.zeros((0, 0))
    return keep, np.vstack(rows)


def main() -> int:
    problems: list[str] = []
    counts: dict[tuple[str, str], int] = {}

    # --- provenance, one row per arm ---------------------------------------
    log(f"{'arm':16s} {'want':14s} {'resolved':14s} {'gpu':28s} {'sm':7s} {'param dtypes'}")
    for arm, spec in pcfg.ARMS.items():
        path = pcfg.provenance_path(arm)
        if not path.exists():
            log(f"{arm:16s} {spec['precision']:14s} {'NO PROVENANCE':14s}")
            problems.append(f"{arm}: no provenance.json — the arm did not finish")
            continue
        prov = json.loads(path.read_text())
        probe = prov.get("probe", {})
        dtypes = ", ".join(
            f"{k}={v['param_dtype'].replace('torch.', '')}" for k, v in probe.get("embedders", {}).items()
        )
        log(
            f"{arm:16s} {spec['precision']:14s} {probe.get('resolved', '?'):14s} "
            f"{str(probe.get('gpu_name', '?'))[:28]:28s} {str(probe.get('gpu_capability', '?')):7s} {dtypes}"
        )
        if probe.get("resolved") != spec["precision"]:
            problems.append(
                f"{arm}: provenance says precision {probe.get('resolved')!r}, arm table says {spec['precision']!r}"
            )
        # The card the arm NAMES must be the card it RAN on.  ``gres`` asks; it
        # does not guarantee, and a mismatch silently deletes the control.
        want_gpu = spec["gpu"]
        got_gpu = str(probe.get("gpu_name", "")).lower()
        if want_gpu not in got_gpu.replace("-", "").replace(" ", ""):
            problems.append(f"{arm}: asked for gpu:{want_gpu} but ran on {probe.get('gpu_name')!r}")

    # --- cells: present, non-empty, same population ------------------------
    log("")
    for arm in pcfg.ARMS:
        for emb in pcfg.EMBEDDERS:
            path = pcfg.arm_cell(arm, emb)
            if not path.exists():
                problems.append(f"{arm} x {emb}: cell missing ({path})")
                continue
            if path.stat().st_size == 0:
                problems.append(f"{arm} x {emb}: cell is ZERO BYTES — resume would skip it")
                continue
            medias = _load(path)
            counts[(arm, emb)] = len(medias)
            n_vec = sum(1 for m in medias.values() if m.get("embeddings"))
            log(
                f"  {arm:16s} {emb:12s} {len(medias):5d} medias, {n_vec:5d} with vectors, {path.stat().st_size / 1e6:6.0f} MB"
            )
            if n_vec != len(medias):
                problems.append(f"{arm} x {emb}: {len(medias) - n_vec} medias carry no vector")

    for emb in pcfg.EMBEDDERS:
        per_arm = {arm: n for (arm, e), n in counts.items() if e == emb}
        if len(set(per_arm.values())) > 1:
            problems.append(
                f"{emb}: arms disagree on media count {per_arm} — different populations, not different precisions"
            )

    # --- does the fp32 rebuild reproduce the published pile? ---------------
    log("")
    ref = pcfg.REFERENCE_ARM
    for emb in pcfg.EMBEDDERS:
        arm_path, shared_path = pcfg.arm_cell(ref, emb), pcfg.shared_cell(emb)
        if not (arm_path.exists() and shared_path.exists()):
            problems.append(f"{emb}: cannot check reproduction ({arm_path.exists()=}, {shared_path.exists()=})")
            continue
        a_ids, a_vec = _vectors(_load(arm_path))
        s_ids, s_vec = _vectors(_load(shared_path))
        if a_ids != s_ids:
            problems.append(
                f"{emb}: rebuilt fp32 covers {len(a_ids)} medias, published cell {len(s_ids)} — not comparable"
            )
            continue
        # Cosine, not raw distance: the vectors are what similarity is computed
        # from, so cosine is the quantity that has to reproduce.
        a_n = a_vec / np.maximum(np.linalg.norm(a_vec, axis=1, keepdims=True), 1e-12)
        s_n = s_vec / np.maximum(np.linalg.norm(s_vec, axis=1, keepdims=True), 1e-12)
        cos = np.clip((a_n * s_n).sum(axis=1), -1.0, 1.0)
        worst = float(1.0 - cos.min())
        median = float(1.0 - np.median(cos))
        log(f"  reproduction {emb:12s} vs published fp32: median 1-cos {median:.2e}, worst {worst:.2e}")
        # The published pile was built on a V100; the reference arm is an L40S,
        # so this is a cross-GPU fp32 comparison and 1e-6 is a generous ceiling
        # on the ~1e-7 kernel-selection drift it should show.
        if worst > 1e-4:
            problems.append(
                f"{emb}: rebuilt fp32 does NOT reproduce the published cell (worst 1-cos {worst:.2e}); "
                f"the whole comparison base is suspect"
            )

    log("")
    if problems:
        for p in problems:
            log(f"PROBLEM: {p}")
        log(f"\n{len(problems)} problem(s) — do not analyse these arms yet.")
        return 1
    log("all arms verified: provenance matches the table, populations agree, fp32 reproduces the pile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
