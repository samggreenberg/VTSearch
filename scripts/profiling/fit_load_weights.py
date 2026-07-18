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

It also aggregates the ``finalize:<slot>`` sub-slot rows (recorded by
``FinalizeProgress.begin`` via the profiler) into per-``(device, media)``
sub-slot **shares** — the measured counterpart to the static
``FinalizeProgress._SLOTS`` ballpark — and emits a ``FINALIZE_SLOT_SHARES``
table body for ``_load_cost_model.py``.

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

# Canonical finalize sub-slot execution order (matches FinalizeProgress._SLOTS).
# The emitted shares list preserves this order; any slot seen in the data but
# not listed here (e.g. a renamed/added sub-stage) is appended after it.
_FIN_SLOT_ORDER = ("cleanup", "dedup", "coverage", "signpost_texts", "registry", "projection")


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

    # Group phase-seconds by (device, media, embedder).
    groups: dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    dl_by_device: dict[str, list[tuple[float, float]]] = defaultdict(list)  # (size_mb, seconds)
    extract_pts: list[tuple[float, float]] = []  # (size_mb, seconds), device-pooled
    # finalize sub-slot seconds by (device, media) -> slot -> [seconds].
    fin_slots: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["device"], r["media_type"], r["embedder"])
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
            # Sub-slot durations partition the finalize phase; a slot that
            # actually ran (>0s) contributes to its (device, media) share. A
            # skipped slot (0s) leaves no row and simply isn't paced separately.
            slot = phase.split(":", 1)[1]
            if secs > 0:
                fin_slots[(r["device"], r["media_type"])][slot].append(secs)
            continue
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

    # ---- finalize sub-slot shares (per device, media) ----
    fin_shares, fin_summary = _fit_finalize_slots(fin_slots)
    print("# ===== FINALIZE_SLOT_SHARES body =====")
    print("FINALIZE_SLOT_SHARES = {")
    for key in sorted(fin_shares):
        print(f"    {key!r}: {fin_shares[key]!r},")
    print("}")
    print()

    # ---- human-readable summary (paste under Results) ----
    print("# ===== summary =====")
    print("| device | media | embedder | load a+b·n s (cold) | R² | embed a+b·n | R² | finalize a+b·n | R² | pts |")
    print("|--------|-------|----------|---------------------|----|-------------|----|----------------|----|----|")
    for line in summary_lines:
        print(line)
    print()
    print("# ===== finalize sub-slot shares =====")
    print("| device | media | slot shares (fraction of finalize) | loads |")
    print("|--------|-------|-------------------------------------|-------|")
    for line in fin_summary:
        print(line)
    return 0


def _fit_finalize_slots(
    fin_slots: dict[tuple[str, str], dict[str, list[float]]],
) -> tuple[dict[tuple[str, str], tuple[tuple[str, float], ...]], list[str]]:
    """Aggregate ``finalize:<slot>`` rows into per-(device, media) normalized
    sub-slot shares. Per slot: median seconds across loads (robust to outliers),
    then normalize across the cell's slots. Slots are emitted in canonical
    execution order, with any unrecognized slot appended after.
    """
    shares: dict[tuple[str, str], tuple[tuple[str, float], ...]] = {}
    summary: list[str] = []
    for key in sorted(fin_slots):
        dev, media = key
        # Median seconds per slot; drop slots that never took measurable time
        # (a ~0s sub-stage earns no slice — it keeps its static ballpark share
        # when merged in _finalize_slots, or simply isn't paced separately).
        slot_med = {}
        for slot, v in fin_slots[key].items():
            if v and (m := statistics.median(v)) > 0:
                slot_med[slot] = m
        total = sum(slot_med.values())
        if total <= 0:
            continue
        known = [s for s in _FIN_SLOT_ORDER if s in slot_med]
        extra = [s for s in slot_med if s not in _FIN_SLOT_ORDER]
        ordered = tuple((s, round(slot_med[s] / total, 4)) for s in known + extra)
        shares[key] = ordered
        n_loads = max((len(v) for v in fin_slots[key].values()), default=0)
        cells = ", ".join(f"{s} {frac:.2f}" for s, frac in ordered)
        summary.append(f"| {dev} | {media} | {cells} | {n_loads} |")
    return shares, summary


if __name__ == "__main__":
    raise SystemExit(main())
