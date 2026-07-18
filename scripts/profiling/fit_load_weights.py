#!/usr/bin/env python3
"""Fit the affine per-phase load-cost model from calibration JSONL.

Reads the per-phase rows emitted by ``calibrate_load_weights.py`` (via the
``VTSEARCH_PROFILE_LOAD`` recorder) and, per ``(device, media_type, embedder)``,
fits:

    T_load     ≈ a_model + b_load · n         (warm model load + per-item decode, least-squares vs n)
    T_embed    ≈ a_embed + b_embed · n        (least-squares vs n)
    T_finalize ≈ a_fin   + b_fin   · n        (least-squares vs n)
    bandwidth  ≈ download_size_mb / seconds   (cold-download rows, device-pooled)
    extract    ≈ download_size_mb / seconds   (extract rows, pooled)

and prints (a) a checked-in ``_load_cost_model.py`` body and (b) a human-readable
summary table for docs/plans/progress-weight-calibration.md. See that plan.

Usage:
    python scripts/profiling/fit_load_weights.py calib.gpu.jsonl calib.cpu.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict

# Warm model-load floor (seconds). Warm loads skip the model-load phase (the
# encoder is already resident, so no "loading model" status fires), so there is
# usually no warm row to fit; a small floor keeps the model slice present but
# tiny — matching the "model kept deliberately small" pacing design. The cold
# first-load model cost is recorded separately as a note, not paced against.
_WARM_MODEL_FLOOR_S = 0.5


def _load_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _affine_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Ordinary least squares y ≈ a + b·x. Returns (a, b, r2).

    Falls back to (mean, 0, 0) when x has no spread (can't estimate a slope).
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-9 or n < 2:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 1.0
    return a, b, r2


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: fit_load_weights.py <jsonl> [<jsonl> ...]", file=sys.stderr)
        return 2
    rows = _load_rows(sys.argv[1:])

    # Group phase-seconds by (device, media, embedder). ``device`` is collapsed
    # to the coarse cost-model key: "cuda:0" → "cuda", and a CUDA row measured
    # with cuML active becomes the "cuda+cuml" variant (cuML moves the
    # coverage-atlas k-means / UMAP to the GPU, changing the finalize cost —
    # the profiler stamps each row with the active cuML state).
    groups: dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    dl_by_device: dict[str, list[tuple[float, float]]] = defaultdict(list)  # (size_mb, seconds)
    extract_pts: list[tuple[float, float]] = []  # (size_mb, seconds), device-pooled
    for r in rows:
        device = str(r["device"])
        if device.startswith("cuda"):
            device = "cuda+cuml" if r.get("cuml") else "cuda"
        key = (device, r["media_type"], r["embedder"])
        phase = r["phase"]
        n = float(r.get("n") or 0)
        secs = float(r.get("seconds") or 0)
        if phase == "download":
            if r.get("cold_download") and r.get("download_size_mb"):
                dl_by_device[r["device"]].append((float(r["download_size_mb"]), secs))
            continue
        if phase == "extract":
            # Only a cold acquire actually unpacks; a cached extract dir skips
            # the phase entirely, so any recorded row is a real extraction.
            if r.get("download_size_mb") and secs > 0.1:
                extract_pts.append((float(r["download_size_mb"]), secs))
            continue
        if phase.startswith("finalize:"):
            continue  # sub-slots: deferred (see plan follow-ups)
        if phase == "model_load":
            # warm rows only; fall back to all if none warm
            groups[key]["model_warm" if not r.get("cold_model") else "model_cold"].append((n, secs))
        elif phase in ("embed", "finalize"):
            groups[key][phase].append((n, secs))

    # Bandwidth (MB/s), device-pooled then collapsed if similar.
    bw: dict[str, float] = {}
    for dev, pts in dl_by_device.items():
        # seconds ≈ size / bw  ->  bw = median(size/seconds)
        rates = [size / s for size, s in pts if s > 0]
        if rates:
            bw[dev] = statistics.median(rates)

    model = {}
    summary_lines = []
    for key in sorted(groups):
        dev, media, emb = key
        g = groups[key]
        # Warm loads skip the model phase, so a warm row is rare; floor when
        # absent. Never fall back to the (large) cold value for pacing.
        warm = g.get("model_warm")
        cold_rows = g.get("model_cold")
        # The "model_load" phase is warm model load PLUS per-item source
        # decode, so it scales with n: fit an affine a_model + b_load·n from
        # the warm rows (cold rows fold in the one-time encoder download, so
        # they only feed the intercept fallback / note).
        if warm:
            m_xs = [n for n, _ in warm]
            m_ys = [s for _, s in warm]
            a_model, b_load, r2_m = _affine_fit(m_xs, m_ys)
            a_model = max(_WARM_MODEL_FLOOR_S if b_load <= 0 else 0.0, a_model)
        else:
            a_model, b_load, r2_m = _WARM_MODEL_FLOOR_S, 0.0, 0.0
        cold_model = statistics.median([s for _, s in cold_rows]) if cold_rows else a_model
        e_xs = [n for n, _ in g.get("embed", [])]
        e_ys = [s for _, s in g.get("embed", [])]
        a_e, b_e, r2_e = _affine_fit(e_xs, e_ys)
        f_xs = [n for n, _ in g.get("finalize", [])]
        f_ys = [s for _, s in g.get("finalize", [])]
        a_f, b_f, r2_f = _affine_fit(f_xs, f_ys)
        model[key] = {
            "a_model": round(max(0.0, a_model), 4),
            "b_load": round(max(0.0, b_load), 6),
            "a_embed": round(max(0.0, a_e), 4),
            "b_embed": round(max(0.0, b_e), 6),
            "a_fin": round(max(0.0, a_f), 4),
            "b_fin": round(max(0.0, b_f), 6),
        }
        summary_lines.append(
            f"| {dev} | {media} | {emb} | {a_model:.2f}+{b_load * 1000:.2f}m·n (cold {cold_model:.1f}) | {r2_m:.2f} | "
            f"{a_e:.2f}+{b_e * 1000:.2f}m·n | {r2_e:.2f} | {a_f:.2f}+{b_f * 1000:.2f}m·n | {r2_f:.2f} | "
            f"{len(e_xs)} |"
        )

    # ---- emit the checked-in module body ----
    print("# ===== _load_cost_model.py body =====")
    print("LOAD_COST_MODEL = {")
    for key in sorted(model):
        print(f"    {key!r}: {model[key]!r},")
    print("}")
    bw_val = statistics.median(list(bw.values())) if bw else 0.0
    print(f"DOWNLOAD_MB_PER_S = {round(bw_val, 3)}  # per-device: {bw}")
    ex_rates = [size / s for size, s in extract_pts if s > 0]
    ex_val = statistics.median(ex_rates) if ex_rates else 0.0
    print(f"EXTRACT_MB_PER_S = {round(ex_val, 3)}  # {len(extract_pts)} extractions")
    print()
    # ---- human-readable summary (paste under Results) ----
    print("# ===== summary =====")
    print("| device | media | embedder | load a+b·n s (cold) | R² | embed a+b·n | R² | finalize a+b·n | R² | pts |")
    print("|--------|-------|----------|---------------------|----|-------------|----|----------------|----|----|")
    for line in summary_lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
