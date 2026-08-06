"""Emit the #2861 report figures as standalone inline SVG (no JS, no libs).

Reads the analyzer's aggregates and writes two <svg> snippets that are pasted
straight into the report artifact, so every number in a figure comes from the
same CSV the tables come from.

  fig_kappa_curve.svg   - the pooled deep-regime kappa curve, four series
                          (fold/label x rate/mid), with the tied plateau shaded
  fig_kappa_envs.svg    - small multiples: the fold-anchored `mid` curve (the
                          recommended rule) in each environment, shared y scale

Usage: python make_rate_figs.py [outdir]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import common

common.setup_env()

import pandas as pd  # noqa: E402

AGG = common.RESULTS / "agg"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else common.RESULTS / "figs"

SERIES = [
    ("fold_anchored", "rate", "var(--s1)", "fold-anchored · rate"),
    ("fold_anchored", "mid", "var(--s1)", "fold-anchored · mid"),
    ("label_anchored", "rate", "var(--s2)", "label-anchored · rate"),
    ("label_anchored", "mid", "var(--s2)", "label-anchored · mid"),
]
DASH = {"rate": "", "mid": "5 4"}


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def kappa_curve_svg(curve: pd.DataFrame, plateau: pd.DataFrame) -> str:
    W, H = 760, 400
    L, R, T, B = 62, 250, 26, 56
    xs = sorted(curve["kappa"].unique())
    lx = [math.log10(k) for k in xs]
    x0, x1 = min(lx), max(lx)
    ys = curve["d_regret"]
    y0, y1 = float(ys.min()), float(ys.max())
    pad = 0.06 * (y1 - y0 or 1)
    y0, y1 = y0 - pad, y1 + pad

    def px(k: float) -> float:
        return L + (math.log10(k) - x0) / (x1 - x0) * (W - L - R)

    def py(v: float) -> float:
        return T + (v - y0) / (y1 - y0) * (H - T - B)

    p: list[str] = []
    # Plateau band for the RECOMMENDED series (fold-anchored, midpoint cut) -
    # shading a different arm's plateau would quietly mislabel the figure.
    row = plateau[
        (plateau["scope"] == "pooled_deep") & (plateau["family"] == "fold_anchored") & (plateau["rule"] == "mid")
    ]
    if not row.empty:
        lo, hi = float(row["plateau_lo"].iloc[0]), float(row["plateau_hi"].iloc[0])
        p.append(
            f'<rect x="{px(lo):.1f}" y="{T}" width="{px(hi) - px(lo):.1f}" height="{H - T - B:.1f}" '
            f'fill="var(--s1)" opacity="0.07"/>'
        )
        p.append(
            f'<text x="{(px(lo) + px(hi)) / 2:.1f}" y="{T + 14}" font-size="11" fill="var(--s1)" '
            f'text-anchor="middle" font-weight="600">tied with the best (fold · mid)</text>'
        )
    # zero line (= pure cross-calibration)
    if y0 <= 0 <= y1:
        p.append(
            f'<line x1="{L}" y1="{py(0):.1f}" x2="{W - R}" y2="{py(0):.1f}" stroke="var(--axis)" '
            f'stroke-width="1" stroke-dasharray="4 3"/>'
        )
        p.append(
            f'<text x="{W - R - 4}" y="{py(0) - 6:.1f}" font-size="11" fill="var(--ink-muted)" '
            f'text-anchor="end">pure cross-calibration</text>'
        )
    # axes
    p.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}" stroke="var(--axis)" stroke-width="1"/>')
    for k in xs:
        p.append(
            f'<text x="{px(k):.1f}" y="{H - B + 18}" font-size="11" fill="var(--ink-sec)" '
            f'text-anchor="middle">{k:g}</text>'
        )
        p.append(
            f'<line x1="{px(k):.1f}" y1="{H - B}" x2="{px(k):.1f}" y2="{H - B + 5}" '
            f'stroke="var(--axis)" stroke-width="1"/>'
        )
    p.append(
        f'<text x="{(L + W - R) / 2:.1f}" y="{H - 14}" font-size="11.5" fill="var(--ink-sec)" '
        f'text-anchor="middle">anchor mass κ — how many haystack points one vote is worth (log scale)</text>'
    )
    step = _nice_step(y1 - y0)
    g = math.ceil(y0 / step) * step
    while g <= y1 + 1e-9:
        p.append(
            f'<line x1="{L}" y1="{py(g):.1f}" x2="{W - R}" y2="{py(g):.1f}" stroke="var(--grid)" stroke-width="1"/>'
        )
        p.append(
            f'<text x="{L - 8}" y="{py(g) + 4:.1f}" font-size="11" fill="var(--ink-sec)" '
            f'text-anchor="end">{g:+.02f}</text>'
        )
        g += step
    p.append(
        f'<text x="14" y="{T + 4}" font-size="11.5" fill="var(--ink-sec)" text-anchor="start">Δ regret vs x-cal</text>'
    )
    p.append(
        f'<text x="14" y="{T + 20}" font-size="10.5" fill="var(--ink-muted)" text-anchor="start">'
        f"(lower is better)</text>"
    )
    # series
    legend_y = T + 46
    for fam, rule, colour, label in SERIES:
        s = curve[(curve["family"] == fam) & (curve["rule"] == rule)].sort_values("kappa")
        if s.empty:
            continue
        pts = " ".join(f"{px(r.kappa):.1f},{py(r.d_regret):.1f}" for r in s.itertuples())
        dash = f' stroke-dasharray="{DASH[rule]}"' if DASH[rule] else ""
        p.append(
            f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2.2" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash}/>'
        )
        best = s.loc[s["d_regret"].idxmin()]
        p.append(f'<circle cx="{px(best.kappa):.1f}" cy="{py(best.d_regret):.1f}" r="4" fill="{colour}"/>')
        for r in s.itertuples():
            p.append(f'<circle cx="{px(r.kappa):.1f}" cy="{py(r.d_regret):.1f}" r="2" fill="{colour}" opacity="0.55"/>')
        p.append(
            f'<line x1="{W - R + 16}" y1="{legend_y - 4}" x2="{W - R + 44}" y2="{legend_y - 4}" '
            f'stroke="{colour}" stroke-width="2.2"{dash}/>'
        )
        p.append(f'<text x="{W - R + 50}" y="{legend_y}" font-size="12" fill="var(--ink-sec)">{esc(label)}</text>')
        p.append(
            f'<text x="{W - R + 50}" y="{legend_y + 15}" font-size="11" fill="var(--ink-muted)">'
            f"best κ={best.kappa:g} · {best.d_regret:+.4f}</text>"
        )
        legend_y += 44
    return (
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Paired regret against pure cross-calibration '
        f"as a function of anchor mass kappa, for the fold-anchored and label-anchored families under the "
        f'mid and rate cut rules.">' + "".join(p) + "</svg>"
    )


def _nice_step(span: float) -> float:
    raw = span / 5
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def env_small_multiples_svg(curve: pd.DataFrame, envs: pd.DataFrame) -> str:
    s = curve[(curve["family"] == "fold_anchored") & (curve["rule"] == "mid")]
    names = list(envs.sort_values("n_fit")["env"])
    names = [n for n in names if n in set(s["env"])]
    if not names:
        return "<svg viewBox='0 0 10 10'></svg>"
    cols = 3
    rows = math.ceil(len(names) / cols)
    PW, PH = 246, 168
    W, H = cols * PW + 16, rows * PH + 34
    y0 = float(s["d_regret"].min())
    y1 = float(s["d_regret"].max())
    pad = 0.08 * (y1 - y0 or 1)
    y0, y1 = y0 - pad, y1 + pad
    xs = sorted(s["kappa"].unique())
    x0, x1 = math.log10(min(xs)), math.log10(max(xs))
    p: list[str] = []
    n_fit = dict(zip(envs["env"], envs["n_fit"], strict=False))
    for i, env in enumerate(names):
        ox = 8 + (i % cols) * PW
        oy = 26 + (i // cols) * PH
        iw, ih = PW - 54, PH - 62

        def px(k: float, ox=ox, iw=iw) -> float:
            return ox + 42 + (math.log10(k) - x0) / (x1 - x0) * iw

        def py(v: float, oy=oy, ih=ih) -> float:
            return oy + 18 + (v - y0) / (y1 - y0) * ih

        e = s[s["env"] == env].sort_values("kappa")
        p.append(
            f'<line x1="{ox + 42}" y1="{oy + 18}" x2="{ox + 42}" y2="{oy + 18 + ih}" '
            f'stroke="var(--axis)" stroke-width="1"/>'
        )
        if y0 <= 0 <= y1:
            p.append(
                f'<line x1="{ox + 42}" y1="{py(0):.1f}" x2="{ox + 42 + iw}" y2="{py(0):.1f}" '
                f'stroke="var(--axis)" stroke-width="1" stroke-dasharray="4 3"/>'
            )
        pts = " ".join(f"{px(r.kappa):.1f},{py(r.d_regret):.1f}" for r in e.itertuples())
        p.append(f'<polyline points="{pts}" fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round"/>')
        best = e.loc[e["d_regret"].idxmin()]
        p.append(f'<circle cx="{px(best.kappa):.1f}" cy="{py(best.d_regret):.1f}" r="4" fill="var(--s1)"/>')
        ds, emb, style = env.split("/")
        p.append(
            f'<text x="{ox + 42}" y="{oy + 8}" font-size="11.5" fill="var(--ink)" font-weight="600">'
            f"{esc(ds)} · {esc(emb)}</text>"
        )
        p.append(
            f'<text x="{ox + 42}" y="{oy + 18 + ih + 16}" font-size="10.5" fill="var(--ink-muted)">'
            f"{esc(style)} · N={int(n_fit.get(env, 0))} · best κ={best.kappa:g}</text>"
        )
        for k in (min(xs), 1.0, max(xs)):
            if k in xs:
                p.append(
                    f'<text x="{px(k):.1f}" y="{oy + 18 + ih + 30}" font-size="10" fill="var(--ink-sec)" '
                    f'text-anchor="middle">{k:g}</text>'
                )
        p.append(
            f'<text x="{ox + 38}" y="{py(y1 - pad) + 4:.1f}" font-size="10" fill="var(--ink-sec)" '
            f'text-anchor="end">{y1 - pad:+.02f}</text>'
        )
        p.append(
            f'<text x="{ox + 38}" y="{py(y0 + pad) + 4:.1f}" font-size="10" fill="var(--ink-sec)" '
            f'text-anchor="end">{y0 + pad:+.02f}</text>'
        )
    p.append(
        '<text x="8" y="14" font-size="11" fill="var(--ink-muted)">fold-anchored · mid — Δ regret vs x-cal, '
        "deep regime; panels ordered by fit population N</text>"
    )
    return (
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Small multiples of the fold-anchored midpoint '
        f'kappa curve, one panel per environment, ordered by fit population size.">' + "".join(p) + "</svg>"
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pooled = pd.read_csv(AGG / "rate_curve_pooled_deep.csv")
    plateau = pd.read_csv(AGG / "rate_plateau.csv")
    curve = pd.read_csv(AGG / "rate_curve.csv")
    envs = pd.read_csv(AGG / "rate_environments.csv")
    deep = curve[curve["window"].str.replace("le_", "", regex=False).astype(int) >= 100]
    per_env = deep.groupby(["env", "family", "rule", "kappa"], observed=True)["d_regret"].mean().reset_index()
    (OUT / "fig_kappa_curve.svg").write_text(kappa_curve_svg(pooled, plateau))
    (OUT / "fig_kappa_envs.svg").write_text(env_small_multiples_svg(per_env, envs))
    common.log(f"wrote {OUT}/fig_kappa_curve.svg and {OUT}/fig_kappa_envs.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
