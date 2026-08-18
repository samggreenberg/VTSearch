"""#3160: diff the per-node mechanism probes and name the divergence.

    python analyze_mechanism.py --mechanism <dir> [--reference rack7n03] [--csv out.csv]

The probe recorded, for a fixed 8-image batch on each node, every vision block's
output under each available SDPA backend, plus bare GEMMs.  Two shapes are
distinguishable in that data and they mean different things:

* **a step** -- the tower is bit-identical up to block *k* and differs from
  block *k* on.  Something the model does at block *k* is implemented
  differently on the two devices.
* **a ramp** -- block 0 already differs, slightly, and the relative difference
  grows with depth.  Nothing is *choosing* differently; the arithmetic itself
  is reordered, and the divergence is accumulation.

And one decisive column: if any forced backend makes the two nodes agree
bit-for-bit, the cause is that backend's selection and a determinism knob
exists.  If none does, the cause is below the backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(msg, flush=True)


def rel(a: list[float], b: list[float]) -> float:
    """Relative L2 distance between two per-channel projections."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(x)
    return float(np.linalg.norm(x - y) / denom) if denom else float("nan")


def sig(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return "0" if x == 0 else f"{x:.2g}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mechanism", required=True)
    ap.add_argument("--reference", default="rack7n03")
    ap.add_argument("--csv", default="")
    args = ap.parse_args(argv)

    root = Path(args.mechanism)
    recs = {
        d.name: json.loads((d / "mechanism.json").read_text())
        for d in sorted(root.iterdir())
        if (d / "mechanism.json").is_file()
    }
    if args.reference not in recs:
        raise SystemExit(f"reference {args.reference} not among {sorted(recs)}")
    ref = recs[args.reference]

    log("=== nodes ===")
    for node, r in sorted(recs.items()):
        log(
            f"  {node:<10} {r['gpu_name']:<26} {r['gpu_capability']}  {r['multi_processor_count']} SMs  torch {r['torch']}"
        )

    log("\n=== input tensors (must be identical, or nothing below means anything) ===")
    for node, r in sorted(recs.items()):
        for key, st in (r.get("input", {}).get("tensors") or {}).items():
            same = st["sha256"] == ref["input"]["tensors"][key]["sha256"]
            log(f"  {node:<10} {key:<20} {st['sha256']}  {'same as reference' if same else 'DIFFERS FROM REFERENCE'}")

    log(f"\n=== bare ops vs {args.reference} ===")
    for shape in sorted(ref.get("gemm", {})):
        line = [f"  {shape:<20}"]
        for node, r in sorted(recs.items()):
            here = r.get("gemm", {}).get(shape, {})
            mark = (
                "identical"
                if here.get("sha256") == ref["gemm"][shape]["sha256"]
                else f"differs ({sig(rel(ref['gemm'][shape]['proj'], here['proj']))})"
            )
            line.append(f"{node}: {mark}")
        log("  ".join(line))

    rows = []
    for node, r in sorted(recs.items()):
        if node == args.reference:
            continue
        log(f"\n=== {node} vs {args.reference} ===")
        for backend in sorted(set(ref.get("backends", {})) & set(r.get("backends", {}))):
            a, b = ref["backends"][backend], r["backends"][backend]
            if "error" in a or "error" in b:
                log(f"  {backend:<10} unavailable on one side ({a.get('error') or b.get('error')})"[:110])
                continue
            layers = sorted(set(a["layers"]) & set(b["layers"]), key=int)
            first = next((i for i in layers if a["layers"][i]["sha256"] != b["layers"][i]["sha256"]), None)
            feats_same = a["image_features"]["sha256"] == b["image_features"]["sha256"]
            profile = [(int(i), rel(a["layers"][i]["proj"], b["layers"][i]["proj"])) for i in layers]
            rows.extend({"node": node, "backend": backend, "layer": i, "rel_l2": v} for i, v in profile)
            shape = "identical" if first is None else f"first differing block = {first} of {len(layers)}"
            log(
                f"  {backend:<10} {shape:<34} image_features {'IDENTICAL' if feats_same else 'differ'} "
                f"(rel {sig(rel(a['image_features']['proj'], b['image_features']['proj']))})"
            )
            steps = " ".join(f"{i}:{sig(v)}" for i, v in profile[: min(len(profile), 8)])
            log(f"             blocks 0-7 rel L2: {steps}")
            log(f"             last block rel L2: {sig(profile[-1][1])}")

    if args.csv and rows:
        import csv

        with Path(args.csv).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        log(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
