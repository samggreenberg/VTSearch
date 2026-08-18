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

    log("\n=== preprocessing (does the same JPEG become the same tensor?) ===")
    ref_px = ref["input"]["own_cpu"]["pixel_values"]
    for node, r in sorted(recs.items()):
        host = r.get("host", {})
        px = r["input"]["own_cpu"]["pixel_values"]
        same = px["sha256"] == ref_px["sha256"]
        log(
            f"  {node:<10} {'IDENTICAL' if same else 'DIFFERS'}  rel {sig(rel(ref_px['proj'], px['proj'])):<10} "
            f"absmax {px['absmax']:.6g}  {r.get('preprocessing', {}).get('image_processor_class')}  "
            f"transformers {host.get('transformers')}  threads {host.get('torch_num_threads')}"
        )
        log(f"             cpu: {host.get('cpu')}")

    rows = []
    for node, r in sorted(recs.items()):
        if node == args.reference:
            continue
        log(f"\n=== {node} vs {args.reference} ===")
        for label in ("own", "reference_pixels"):
            if label not in r:
                continue
            # The reference node ran only its own pixels, and its own pixels ARE
            # the reference tensor -- so both of this node's runs are compared
            # against the same reference forward.
            a_all, b_all = ref.get("own", {}), r[label]
            note = "this node's own pixels" if label == "own" else "the REFERENCE node's pixels"
            log(f"  -- {label}: {note}")
            for backend in sorted(set(a_all) & set(b_all)):
                a, b = a_all[backend], b_all[backend]
                if "error" in a or "error" in b:
                    log(f"     {backend:<10} unavailable ({(a.get('error') or b.get('error'))[:70]})")
                    continue
                layers = sorted(set(a["layers"]) & set(b["layers"]), key=int)
                first = next((i for i in layers if a["layers"][i]["sha256"] != b["layers"][i]["sha256"]), None)
                feats_same = a["image_features"]["sha256"] == b["image_features"]["sha256"]
                profile = [(int(i), rel(a["layers"][i]["proj"], b["layers"][i]["proj"])) for i in layers]
                rows.extend(
                    {"node": node, "pixels": label, "backend": backend, "layer": i, "rel_l2": v} for i, v in profile
                )
                shape = "every block identical" if first is None else f"first differing block {first}/{len(layers)}"
                log(
                    f"     {backend:<10} {shape:<28} features {'IDENTICAL' if feats_same else 'differ'} "
                    f"(rel {sig(rel(a['image_features']['proj'], b['image_features']['proj']))})"
                )
                head = " ".join(f"{i}:{sig(v)}" for i, v in profile[:6])
                log(f"                blocks 0-5 rel L2: {head}   last: {sig(profile[-1][1])}")

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
