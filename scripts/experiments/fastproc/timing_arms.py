"""End-to-end embed cost per processor arm, interleaved and repeated (#3146).

    python timing_arms.py --reps 5 --n 1500

The side-pile builds gave one wall-clock number per arm, and that number cannot
carry a speedup claim.  ``tv_cpu_rep`` is the *same code as the reference on the
same node*, and it came out 1.08x faster on ``siglip`` — so run-to-run noise on
a shared cluster is 8%, which is the size of the effect being claimed.  Quoting
1.20x from one run against one run would be quoting noise with a decimal point.

Two things fix that, and both matter:

* **Repeats**, so the spread is measured rather than assumed.
* **Interleaving.** The arms run A,B,C,A,B,C,... inside ONE process on ONE GPU
  rather than as separate jobs, because the confound is node load drifting over
  minutes.  Three consecutive runs of arm A followed by three of arm B would
  attribute any drift between those two windows to the treatment.  Interleaved,
  a drift hits every arm roughly equally and shows up as spread, which is
  honest, instead of as a difference, which is not.

The path timed is ``bulk_embed_image_files`` — the real one, the same function
the pile builder calls — because that is where #3151's decode/forward overlap
lives.  A processor that is 4x faster *in isolation* buys only what the overlap
does not already hide, and the whole reason the issue's 68% projection needs
re-measuring is that it was computed on a stage in isolation.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fastproc_config as fcfg  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def sig2(x: float) -> str:
    if x == 0 or not (x == x) or x in (float("inf"), float("-inf")):
        return f"{x:.0f}"
    if abs(x) >= 0.01:
        return f"{x:.2g}"
    return f"{x:.1e}"


def corpus_medias(n: int) -> list[dict]:
    """*n* corpus images as the media dicts the bulk path expects."""
    roots = [
        fcfg.SHARED_PILE / "datadir" / "visual_genome" / "VG_100K",
        fcfg.SHARED_PILE / "datadir" / "visual_genome" / "VG_100K_2",
    ]
    paths: list[Path] = []
    for root in roots:
        if root.is_dir():
            paths.extend(sorted(root.glob("*.jpg")))
        if len(paths) >= n:
            break
    paths = paths[:n]
    if not paths:
        raise SystemExit(f"no images under {roots}")
    # ``media_path`` is the key ``_pil_source_for`` reads.  A wrong key here does
    # not raise -- every decode returns None and the timed loop measures an empty
    # pipeline, which looks like a spectacular speedup on every arm at once.
    return [{"id": i, "media_path": str(p), "filename": p.name} for i, p in enumerate(paths)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n", type=int, default=1500, help="medias per timed run")
    ap.add_argument("--embedders", default=",".join(fcfg.EMBEDDERS))
    ap.add_argument("--out", default=str(fcfg.results_dir()))
    args = ap.parse_args(argv)

    os.environ.setdefault("VTSEARCH_MODELS_DIR", str(fcfg.SHARED_MODELS))
    os.environ.setdefault("HF_HOME", str(fcfg.SHARED_MODELS))

    import torch

    from vtscore.embedding import initialize_models
    from vtscore.media import get_embedder
    from vtscore.media.image._image_bulk import bulk_embed_image_files

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    node = os.uname().nodename
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "?")
    threads = os.environ.get("VTSEARCH_TORCH_THREADS", "?")
    log(f"node={node} gpu={gpu} cpus={cpus} torch_threads={threads}")
    log(f"{args.reps} interleaved reps x {args.n} medias\n")

    medias = corpus_medias(args.n)
    initialize_models()

    # (backend, device) per arm, from the arm table -- deduplicated, since the
    # repeat arm is the same configuration and is what these reps replace.
    variants: list[tuple[str, str, str]] = []
    for arm, spec in fcfg.ARMS.items():
        key = (spec["backend"], spec["device"])
        if any((b, d) == key for _, b, d in variants):
            continue
        variants.append((arm, spec["backend"], spec["device"]))

    samples: dict[tuple[str, str], list[float]] = {}
    for emb_name in [e for e in args.embedders.split(",") if e]:
        for rep in range(args.reps):
            for arm, backend, device in variants:
                # The knobs are read at processor-construction time, so the
                # embedder is rebuilt per arm rather than mutated in place.
                os.environ["VTSEARCH_IMAGE_PROCESSOR_BACKEND"] = backend
                os.environ["VTSEARCH_IMAGE_PROCESSOR_DEVICE"] = device
                import vtscore.config as cfg

                cfg.IMAGE_PROCESSOR_BACKEND = backend
                cfg.IMAGE_PROCESSOR_DEVICE = device

                emb = get_embedder(emb_name)
                emb._model = None
                emb._processor = None
                emb.load_models()
                proc = getattr(emb._processor, "image_processor", emb._processor)
                # Assert the premise every rep: a fallback would otherwise time
                # the reference arm three times and call one of them a treatment.
                cls = type(proc).__name__
                want_pil = backend == "pil"
                got_pil = cls.endswith("Pil")
                if want_pil != got_pil:
                    raise SystemExit(f"{arm}: asked for {backend}, loaded {cls} — timing would be mislabelled")

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                vecs = bulk_embed_image_files(
                    medias,
                    forward_pil_batch=emb._forward_pil_batch,
                    batch_size=emb.embed_batch_size,
                    on_progress=lambda *a, **k: None,
                    label=emb_name,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                n_ok = sum(1 for v in vecs if v is not None)
                if n_ok < len(medias):
                    raise SystemExit(
                        f"{arm}: only {n_ok}/{len(medias)} medias embedded — a timed run that "
                        f"decoded nothing is not a fast run, it is an empty one"
                    )
                samples.setdefault((emb_name, arm), []).append(dt)
                log(
                    f"  rep {rep} {emb_name:10s} {arm:11s} {backend}/{device:4s} {dt:7.2f}s  ({args.n / dt:5.1f} medias/s)"
                )

    log("\n=== end-to-end cost per arm (interleaved reps; median, and mean ± SE) ===")
    log(f"{'embedder':12s} {'arm':12s} {'backend/dev':18s} {'median s':>10s} {'mean ± SE':>20s} {'medias/s':>10s}")
    for (emb_name, arm), ts in samples.items():
        spec = fcfg.ARMS[arm]
        se = statistics.stdev(ts) / (len(ts) ** 0.5) if len(ts) > 1 else 0.0
        log(
            f"{emb_name:12s} {arm:12s} {spec['backend'] + '/' + spec['device']:18s} "
            f"{statistics.median(ts):10.2f} {statistics.mean(ts):11.2f} ± {se:<6.2f} "
            f"{args.n / statistics.median(ts):10.1f}"
        )

    log("\n=== speedup vs the shipped path, with the noise that bounds it ===")
    log("A speedup is only real if it clears the spread of the SAME code run twice.")
    for emb_name in {e for e, _ in samples}:
        base = samples.get((emb_name, fcfg.REFERENCE_ARM))
        if not base:
            continue
        b_med = statistics.median(base)
        b_se = statistics.stdev(base) / (len(base) ** 0.5) if len(base) > 1 else 0.0
        log(f"  {emb_name}: reference {b_med:.2f}s ± {b_se:.2f}")
        for (e2, arm), ts in samples.items():
            if e2 != emb_name or arm == fcfg.REFERENCE_ARM:
                continue
            med = statistics.median(ts)
            se = statistics.stdev(ts) / (len(ts) ** 0.5) if len(ts) > 1 else 0.0
            # Speedup uncertainty from both arms' SEs, first order.
            ratio = b_med / med
            rel = ((b_se / b_med) ** 2 + (se / med) ** 2) ** 0.5
            log(f"    {arm:12s} {ratio:5.2f}x ± {ratio * rel:4.2f}   ({med:.2f}s ± {se:.2f})")

    payload = {
        "node": node,
        "gpu": gpu,
        "cpus": cpus,
        "torch_threads": threads,
        "n_medias": args.n,
        "reps": args.reps,
        "samples": {f"{e}|{a}": ts for (e, a), ts in samples.items()},
    }
    (outdir / "timing_arms.json").write_text(json.dumps(payload, indent=2) + "\n")
    log(f"\nwrote {outdir}/timing_arms.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
