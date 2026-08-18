"""#3160 figures: the ISA predicts the drift, the card does not.

    python make_figures.py --study /expscratch/$USER/gpu-node-3160 --out <figures dir>

Three figures, each answering one of the study's questions:

1. ``census_by_isa`` -- every node's `siglip2_l` drift against the reference,
   coloured by the host's CPU ISA and marked by GPU part. The claim it has to
   carry is that the two groupings disagree: the ISA separates the drift cleanly
   and the card does not.
2. ``layer_divergence`` -- relative L2 per vision block, own pixels versus the
   reference node's pixels. The flat zero line is the whole result.
3. ``pixel_diff`` -- the distribution of |Δ| over the 3.5M preprocessed pixels,
   which is a single spike at one 8-bit level, plus where in the image they land.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AVX2_CPUS = ("E5-2698",)  # the Broadwell part on the DGX-1 boxes; no AVX-512


def log(msg: str) -> None:
    print(msg, flush=True)


def cpu_map(study: Path) -> dict[str, tuple[str, str]]:
    """node -> (cpu model, ISA), from the cheap per-node cpuinfo jobs."""
    out = {}
    for f in sorted((study / "cpuinfo").glob("*.txt")):
        text = f.read_text()
        model = next((ln.split(":", 1)[1].strip() for ln in text.splitlines() if "model name" in ln), "unknown")
        isa = "AVX-512" if "avx512f" in text else "AVX2 only"
        out[f.stem] = (model, isa)
    return out


def fig_census(study: Path, out: Path, reference: str = "rack7n03") -> None:
    census = study / "census"
    cpus = cpu_map(study)
    ref = np.load(census / reference / "vectors_siglip2_l.npy").astype(np.float64)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)

    rows = []
    for d in sorted(census.iterdir()):
        vec = d / "vectors_siglip2_l.npy"
        info = d / "device.json"
        if not vec.is_file() or not info.is_file() or d.name.endswith("-avx2"):
            continue
        v = np.load(vec).astype(np.float64)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        drift = float(np.median(1.0 - (ref * v).sum(1)))
        gpu = json.loads(info.read_text())["device"].get("gpu_name", "?")
        model, isa = cpus.get(d.name, ("unknown", "unknown"))
        rows.append((d.name, drift, gpu, model, isa))

    rows.sort(key=lambda r: (r[4], -r[1], r[0]))
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    colors = {"AVX-512": "#3b7dd8", "AVX2 only": "#d1495b", "unknown": "#888888"}
    markers = {"Tesla V100-SXM2-32GB-LS": "o", "Tesla V100S-PCIE-32GB": "s", "NVIDIA L40S": "^"}
    floor = 1e-13
    for i, (node, drift, gpu, _model, isa) in enumerate(rows):
        ax.scatter(
            max(drift, floor),
            i,
            s=70,
            color=colors.get(isa, "#888"),
            marker=markers.get(gpu, "D"),
            zorder=3,
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [f"{n}  ·  {g.replace('Tesla ', '').replace('NVIDIA ', '')}" for n, _, g, _, _ in rows], fontsize=8
    )
    ax.set_xscale("log")
    ax.set_xlabel(f"median 1 − cos vs {reference}   (siglip2_l, 256 VG images, fp32)")
    ax.axvline(0.005, color="#444", ls="--", lw=1)
    ax.text(0.005, len(rows) - 0.5, " 0.005 decision margin", fontsize=8, va="top", color="#444")
    ax.axvline(2.9e-6, color="#777", ls=":", lw=1)
    ax.text(2.9e-6, len(rows) - 0.5, " fp16 costs 2.9e-6", fontsize=8, va="top", color="#777")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in colors.items() if k != "unknown"]
    handles += [
        plt.Line2D([], [], marker=m, ls="", color="#333", label=k.replace("Tesla ", "").replace("NVIDIA ", ""))
        for k, m in markers.items()
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.95)
    ax.set_title(
        "The host's CPU ISA predicts the drift; the GPU part does not\n"
        f"(points at the left edge are exactly 0, drawn at {floor:g})",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "census_by_isa.png", dpi=150)
    log(f"wrote {out / 'census_by_isa.png'}  ({len(rows)} nodes)")


def fig_layers(study: Path, out: Path, reference: str = "rack7n03") -> None:
    mech = study / "mechanism"
    ref = json.loads((mech / reference / "mechanism.json").read_text())

    def rel(a, b):
        x, y = np.asarray(a, float), np.asarray(b, float)
        n = np.linalg.norm(x)
        return float(np.linalg.norm(x - y) / n) if n else np.nan

    fig, ax = plt.subplots(figsize=(8.5, 5))
    styles = {
        ("rack5n03", "own"): ("#d1495b", "-", "rack5n03 (AVX2 host), its own pixels"),
        ("rack5n03", "reference_pixels"): ("#d1495b", "--", "rack5n03, fed rack7n03's pixels"),
        ("rack4n02", "own"): ("#3b7dd8", "-", "rack4n02 (L40S, AVX-512), its own pixels"),
        ("rack4n02", "reference_pixels"): ("#3b7dd8", "--", "rack4n02, fed rack7n03's pixels"),
    }
    floor = 1e-17
    for (node, label), (color, ls, name) in styles.items():
        f = mech / node / "mechanism.json"
        if not f.is_file():
            continue
        rec = json.loads(f.read_text())
        if label not in rec or "default" not in rec[label] or "layers" not in rec[label]["default"]:
            continue
        a, b = ref["own"]["default"]["layers"], rec[label]["default"]["layers"]
        idx = sorted(set(a) & set(b), key=int)
        vals = [max(rel(a[i]["proj"], b[i]["proj"]), floor) for i in idx]
        ax.plot([int(i) for i in idx], vals, color=color, ls=ls, marker="o", ms=3, label=name)
    ax.set_yscale("log")
    ax.set_ylim(floor / 2, 1)
    ax.set_xlabel("vision transformer block")
    ax.set_ylabel("relative L2 vs rack7n03")
    ax.set_title(
        "Same pixels, same answer: the divergence is upstream of the GPU\n"
        "(dashed lines sit on the floor — every block bit-identical)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "layer_divergence.png", dpi=150)
    log(f"wrote {out / 'layer_divergence.png'}")


def fig_pixels(study: Path, out: Path, reference: str = "rack7n03", other: str = "rack5n03") -> None:
    mech = study / "mechanism"
    a = np.load(mech / reference / "pixels.npy").astype(np.float64)
    b = np.load(mech / other / "pixels.npy").astype(np.float64)
    d = np.abs(a - b)
    ne = d > 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    vals = d[ne]
    ax1.hist(vals * 255 / 2, bins=np.arange(0.0, 2.05, 0.05), color="#d1495b")
    ax1.set_xlabel("|Δ| in 8-bit levels")
    ax1.set_ylabel("pixels")
    ax1.set_title(
        f"{ne.mean():.1%} of {d.size:,} pixels differ,\nall by one 8-bit level (2/255 = {d.max():.3e})", fontsize=10
    )
    ax1.grid(alpha=0.25)

    m = ne[0].any(0)
    ax2.imshow(m, cmap="Reds", interpolation="nearest")
    ax2.set_title(f"where they land (image 0): {m.mean():.1%} of its 384x384 grid", fontsize=10)
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.suptitle(f"Preprocessed pixels: {other} (AVX2) vs {reference} (AVX-512), same JPEG, same code", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "pixel_diff.png", dpi=150)
    log(f"wrote {out / 'pixel_diff.png'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    study, out = Path(args.study), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_census(study, out)
    fig_layers(study, out)
    fig_pixels(study, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
