"""Is the backend difference separable from PyTorch's CPU dispatch? (#3146 x #3160)

    python dispatch_matrix.py --emit --tag avx512      # under default dispatch
    ATEN_CPU_CAPABILITY=avx2 python dispatch_matrix.py --emit --tag avx2
    python dispatch_matrix.py --compare

#3160 established that PyTorch's CPU kernel dispatch changes the *preprocessed
pixels*: an AVX-512 host and an AVX2 host disagree on 12.3% of elements, and the
dominant magnitude is 7.843e-03 = 2/255 — exactly one 8-bit level.

That is the same magnitude as this study's PIL-vs-torchvision difference
(7.8e-03 max abs), which makes the two axes **indistinguishable at the pixel
level** on a fleet that does not pin dispatch. Two consequences, and the second
is the one that would quietly invalidate a headline:

1. This study's arms all ran on one node, so dispatch is constant across them
   and the backend contrast is internally clean. That was luck of the host, not
   design.
2. But the *reference* is dispatch-specific, so every drift is quoted against an
   AVX-512 baseline. If ``pil`` happens to sit closer to torchvision-under-AVX2
   than to torchvision-under-AVX-512, then part of what this study calls a
   backend difference is a dispatch difference wearing a different label.

Only a full pairwise matrix separates them, which is what this builds. The
prediction under #3160's mechanism is specific and falsifiable: torchvision is
an ATen kernel and should move with dispatch, PIL is not and should not move at
all — and ``siglip`` at 224px should be dispatch-invariant while ``siglip2_l``
at 384px is not.

Two processes are needed because ``ATEN_CPU_CAPABILITY`` is read when torch
loads its kernels, so it cannot be changed in-process.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import fastproc_config as fcfg  # noqa: E402
from pixel_drift import backend_of, build_processor, corpus_images  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def sig2(x: float) -> str:
    if x == 0 or not np.isfinite(x):
        return f"{x:.0f}"
    return f"{x:.2g}" if abs(x) >= 0.01 else f"{x:.1e}"


def emit(tag: str, n: int, batch: int, embedders: list[str], outdir: Path) -> int:
    import torch

    from vtscore.config import MODELS_CACHE_DIR
    from vtscore.media import get_embedder

    cache_dir = str(MODELS_CACHE_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    _, images = corpus_images(n)

    # Record what the process actually resolved, not what was requested: the
    # env var is advisory and an unsupported value is ignored silently.
    cpu_model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    flags = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                have = set(line.split())
                flags = ",".join(sorted(f for f in ("avx2", "avx512f") if f in have))
                break
    except OSError:
        pass
    log(f"tag={tag} node={os.uname().nodename}")
    log(f"  ATEN_CPU_CAPABILITY={os.environ.get('ATEN_CPU_CAPABILITY', '(unset)')}")
    log(f"  cpu={cpu_model}  isa={flags}")

    have_cuda = torch.cuda.is_available()
    variants = [("torchvision", "cpu"), ("pil", "cpu")]
    if have_cuda:
        variants.append(("torchvision", "cuda"))

    for emb_name in embedders:
        model_id = get_embedder(emb_name).model_id
        for backend, device in variants:
            proc = build_processor(model_id, backend, cache_dir)
            cls = type(proc).__name__
            if backend_of(cls) != backend:
                log(f"  SKIP {emb_name} {backend}/{device}: got {cls}, backend not honoured")
                continue
            call_kw: dict[str, object] = {} if device == "cpu" else {"device": device}
            chunks = []
            for i in range(0, len(images), batch):
                out = proc(images=images[i : i + batch], return_tensors="pt", **call_kw)
                chunks.append(out["pixel_values"].float().cpu().numpy())
            pv = np.concatenate(chunks, axis=0)
            path = outdir / f"{emb_name}__{backend}_{device}__{tag}.npy"
            np.save(path, pv)
            log(f"  wrote {path.name}  shape={pv.shape}")
    (outdir / f"env__{tag}.txt").write_text(
        f"node={os.uname().nodename}\nATEN_CPU_CAPABILITY={os.environ.get('ATEN_CPU_CAPABILITY', '')}\n"
        f"cpu={cpu_model}\nisa={flags}\ntorch={torch.__version__}\n"
    )
    return 0


def compare(outdir: Path) -> int:
    files = sorted(outdir.glob("*.npy"))
    if not files:
        raise SystemExit(f"no emitted tensors under {outdir}; run --emit first")
    by_emb: dict[str, dict[str, Path]] = {}
    for f in files:
        emb, variant, tag = f.stem.split("__")
        by_emb.setdefault(emb, {})[f"{variant}@{tag}"] = f

    for env in sorted(outdir.glob("env__*.txt")):
        log(f"--- {env.name}")
        log("    " + env.read_text().replace("\n", "\n    ").rstrip())

    for emb, variants in sorted(by_emb.items()):
        names = sorted(variants)
        mats = {k: np.load(v) for k, v in variants.items()}
        log(f"\n=== {emb}: pairwise pixel disagreement ===")
        log("One 8-bit level is 2/255 = 7.8e-03; that is the magnitude BOTH the backend")
        log("change and the AVX-512/AVX2 dispatch change produce, which is why they can")
        log("only be told apart in a matrix like this one.")
        # MAX ABS SATURATES and is the wrong headline here.  Resampling
        # disagreements are quantised to whole 8-bit levels, so on a 384px model
        # *every* pair -- backend, dispatch, device -- comes out at exactly
        # 7.8e-03, and a matrix of identical numbers reads as "these are all the
        # same effect" when it in fact means "this statistic cannot see the
        # difference".  The separating quantity is how MANY elements differ, so
        # that is the matrix, with max kept underneath for scale.
        w = max(len(n) for n in names) + 1
        log("  % of elements that differ (max |delta| beneath):")
        log(" " * w + " ".join(f"{n:>22s}" for n in names))
        for a in names:
            cells, maxes = [], []
            for b in names:
                if mats[a].shape != mats[b].shape:
                    cells.append(f"{'shape!':>22s}")
                    maxes.append(f"{'':>22s}")
                    continue
                d = np.abs(mats[a] - mats[b])
                cells.append(f"{float((d > 0).mean()) * 100:21.2f}%")
                maxes.append(f"{sig2(float(d.max())):>22s}")
            log(f"{a:{w}s}" + " ".join(cells))
            log(" " * w + " ".join(maxes))

        # The question the matrix exists to answer, stated as a comparison
        # rather than left for the reader to compute.
        tv512 = mats.get("torchvision_cpu@avx512")
        tv2 = mats.get("torchvision_cpu@avx2")
        pil512 = mats.get("pil_cpu@avx512")
        pil2 = mats.get("pil_cpu@avx2")

        def frac(a, b):
            return float((np.abs(a - b) > 0).mean())

        if tv512 is not None and tv2 is not None:
            f = frac(tv512, tv2)
            log(
                f"\n  torchvision moves with dispatch: {f * 100:.2f}% of elements"
                + ("  (dispatch-invariant at this resolution)" if f == 0 else "")
            )
        if pil512 is not None and pil2 is not None:
            f = frac(pil512, pil2)
            log(
                f"  PIL moves with dispatch:         {f * 100:.2f}% of elements"
                + ("  (as predicted - PIL is not an ATen kernel)" if f == 0 else "  ** UNEXPECTED **")
            )
        if tv512 is not None and tv2 is not None and pil512 is not None:
            near_512, near_2 = frac(pil512, tv512), frac(pil512, tv2)
            log(f"  PIL vs torchvision@avx512: {near_512 * 100:.2f}% of elements")
            log(f"  PIL vs torchvision@avx2:   {near_2 * 100:.2f}% of elements")
            if near_512 == 0 or near_2 == 0:
                log("  => one of these is EXACT: the backend and dispatch axes are the same")
                log("     axis at this resolution. Re-quote the backend numbers against a")
                log("     dispatch-pinned reference.")
            elif near_2 < near_512 * 0.5:
                log("  => PIL sits CLOSER to torchvision under AVX2: part of what this study")
                log("     calls a backend difference is a dispatch difference. Re-quote the")
                log("     backend numbers against a dispatch-pinned reference.")
            else:
                log("  => the backend difference does NOT collapse under either dispatch:")
                log("     it is a separate axis and this study's numbers stand as backend numbers.")

            # Fractions alone cannot prove two effects are different populations
            # -- they could differ in size and still be nested.  Both effects are
            # quantisation in an 8-bit pipeline, so ONE LEVEL is the only
            # magnitude either can produce and the shared step size is forced,
            # carrying no information about shared cause.  What is not forced is
            # WHICH pixels move, so compare the index sets directly: a Jaccard
            # near 1 would mean the backend change and the dispatch change touch
            # the same pixels and are one axis; near 0 means they are independent
            # axes that merely share a quantum.
            back = np.abs(pil512 - tv512) > 0
            disp = np.abs(tv512 - tv2) > 0
            inter = float((back & disp).sum())
            union = float((back | disp).sum())
            if union > 0:
                log(
                    f"  differing-pixel sets, backend vs dispatch: Jaccard {inter / union:.3f}"
                    f"  (backend {int(back.sum())} px, dispatch {int(disp.sum())} px, shared {int(inter)})"
                )
                if disp.sum() == 0:
                    log("  dispatch moves no pixels at this resolution, so there is no set to")
                    log("  compare: the backend is the only axis here and the number above is")
                    log("  host-independent.")
                else:
                    contained = inter / float(disp.sum())
                    log(f"  of the pixels dispatch moves, {contained * 100:.1f}% are also moved by the backend")
                    # Jaccard alone would call this "independent" because the
                    # backend set is several times larger, and that reading is
                    # wrong.  Containment is the statistic that distinguishes
                    # NESTED from INDEPENDENT, and the two license different
                    # claims: nested means both changes are flipping the same
                    # population of rounding-boundary pixels, one more
                    # aggressively than the other, so the axes are not the same
                    # axis but they are not unrelated either.
                    if contained > 0.9:
                        log("  => NESTED, not independent: dispatch flips a SUBSET of the pixels the")
                        log("     backend flips. Both are rounding at the same boundaries, so the")
                        log("     backend difference does not reduce to dispatch (it is several times")
                        log("     larger and survives both settings) but its exact size is")
                        log("     host-dependent. Quote it with the host's CPU capability attached.")
                    elif contained < 0.3:
                        log("  => INDEPENDENT: the two changes move largely different pixels and")
                        log("     share only a step size, which the 8-bit pipeline forces on both.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--tag", default="default", help="label for this process's dispatch setting")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--embedders", default=",".join(fcfg.EMBEDDERS))
    ap.add_argument("--outdir", default=str(fcfg.results_dir() / "dispatch"))
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    embedders = [e for e in args.embedders.split(",") if e]
    if args.emit:
        return emit(args.tag, args.n, args.batch, embedders, outdir)
    if args.compare:
        return compare(outdir)
    ap.error("pass --emit or --compare")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
