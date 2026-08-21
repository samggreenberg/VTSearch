#!/usr/bin/env python
"""Mechanism figures for the calibration decks.

Uses matplotlib and vtscore, both already project dependencies, so a normal
checkout install is enough. Run from the repo root:

    python slides/figs/src/make-calib-figs.py

The two mixture figures run the *real* vtscore estimators (`fit_score_gmm`,
`fit_anchored_score_gmm`) on labelled synthetic data — schematic inputs, real
code. The schedule figure plots the shipped `blend_schedules` registry verbatim.
The decomposition figure re-plots published numbers from
`docs/experiments/gmm-cut/REPORT-2881.md` (the #2879 re-measure).
"""

import sys
from pathlib import Path

# Ensure the repo root (where app.py lives) is importable no matter the cwd:
# ``python slides/figs/src/x.py`` only puts the script's own dir on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import save  # noqa: E402

from vtscore.training.blend_schedules import BlendContext, get_schedule
from vtscore.training.thresholds import fit_anchored_score_gmm, fit_score_gmm

OUT = Path(__file__).resolve().parent.parent

INK = "#14181f"
SOFT = "#5b6472"
RULE = "#d8dee6"
NEUTRAL_FILL = "#e8ebef"
BLUE = "#0b5fa5"  # production / the shipped thing
RUST = "#b45309"  # the Bad component / cross-calibration
GREEN = "#0d8a5f"  # the Good component

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        "font.size": 15,
        "text.color": INK,
        "axes.edgecolor": SOFT,
        "axes.labelcolor": INK,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
    }
)


def gaussian(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


def xcal_flow_fig() -> None:
    """Schematic of cross-calibration (issue #3207), drawn to match the code.

    Deliberate divergences from the hand mockup the issue attached, because the
    mockup drew a different algorithm than `calculate_cross_calibration_threshold`:
    the rounds are independent stratified *re-draws* of the whole labelset (not
    a partition into halves), and there are no per-fold thresholds to average —
    the held-out scores are pooled and one conformal quantile is cut
    (`threshold_from_fold_orderings`). Green = Good, red = Bad, everything else
    ink-on-white per the issue.
    """
    from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_axis_off()

    # One square per labelled vote; True = Good (green), False = Bad (rust).
    votes = [True, False, False, True, False, True, False, False, True, False]
    sq, gap = 0.34, 0.075

    def vote_row(x0: float, y0: float, labels: list[bool]) -> float:
        for i, good in enumerate(labels):
            ax.add_patch(
                Rectangle(
                    (x0 + i * (sq + gap), y0),
                    sq,
                    sq,
                    facecolor=GREEN if good else RUST,
                    edgecolor=INK,
                    linewidth=1.0,
                    zorder=3,
                )
            )
        return x0 + len(labels) * (sq + gap) - gap

    def arrow(xy_from: tuple[float, float], xy_to: tuple[float, float], rad: float = 0.0) -> None:
        ax.add_patch(
            FancyArrowPatch(
                xy_from,
                xy_to,
                arrowstyle="-|>",
                mutation_scale=14,
                color=INK,
                linewidth=1.6,
                connectionstyle=f"arc3,rad={rad}",
                zorder=2,
            )
        )

    def model_box(cx: float, cy: float, label: str) -> None:
        w, h = 0.85, 0.62
        ax.add_patch(
            Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor="white", edgecolor=INK, linewidth=1.6, zorder=3)
        )
        ax.text(cx, cy, label, ha="center", va="center", fontsize=16, color=INK, zorder=4)

    # ── the model you keep: train M0 on every vote ────────────────────────────
    ax.text(0.5, 9.6, "D₀ — the labelled votes", ha="left", va="bottom", fontsize=15, color=INK)
    xa_end = vote_row(0.5, 9.0, votes)
    arrow((xa_end + 0.15, 9.17), (5.55, 9.17))
    ax.text(4.95, 8.85, "train on all of D₀", ha="center", va="top", fontsize=15, color=INK)
    model_box(6.05, 9.17, "M₀")
    ax.text(6.6, 9.17, "the model you keep", ha="left", va="center", fontsize=15, color=SOFT)

    # ── K independent stratified re-splits ────────────────────────────────────
    ax.text(
        0.5, 8.1, "K independent stratified re-splits  (shipped: K = 2)", fontsize=15.5, color=INK, fontweight="bold"
    )
    ax.text(0.5, 7.73, "re-drawn each round, not a partition —", fontsize=15, color=SOFT)
    ax.text(0.5, 7.38, "a vote can be held out twice, or never", fontsize=15, color=SOFT)

    # Round 1 holds out votes {0, 2, 4, 6, 8}; round 2 re-draws {0, 2, 3, 7, 9}.
    # Both splits are stratified: 2 Good + 3 Bad on each side, every round.
    holdouts = [[0, 2, 4, 6, 8], [0, 2, 3, 7, 9]]
    for row, (y0, held) in enumerate(zip((6.6, 5.3), holdouts)):
        train = [votes[i] for i in range(len(votes)) if i not in held]
        cal = [votes[i] for i in held]
        yc = y0 + sq / 2
        xt_end = vote_row(0.5, y0, train)
        arrow((xt_end + 0.15, yc), (3.35, yc))
        ax.text((xt_end + 3.4) / 2, y0 - 0.14, "train", ha="center", va="top", fontsize=15, color=INK)
        model_box(3.8, yc, f"M{chr(0x2081 + row)}")
        arrow((4.25, yc), (5.1, yc))
        ax.text(4.68, y0 - 0.14, "score", ha="center", va="top", fontsize=15, color=INK)
        xc_end = vote_row(5.25, y0, cal)
        ax.text(xc_end + 0.2, yc, "held out — never\nseen in training", ha="left", va="center", fontsize=15, color=INK)
        # Held-out scores drop into the pooled line below; the x positions sit
        # in the gaps between round 2's squares so the arrows cross nothing.
        arrow((6.04 + 0.42 * row, y0 - 0.08), (6.04 + 0.42 * row, 3.55))

    # ── pool the held-out scores; cut one quantile ────────────────────────────
    ax.text(0.5, 4.45, "pool the held-out scores", fontsize=15.5, color=INK, fontweight="bold")
    ax.text(0.5, 4.1, "no per-round θ to average", fontsize=15, color=SOFT)
    line_y = 3.0
    ax.plot([0.9, 9.1], [line_y, line_y], color=INK, linewidth=1.8, zorder=2)
    for x in (1.5, 2.2, 2.9, 3.5, 4.05, 4.5):
        ax.text(x, line_y - 0.14, "✗", ha="center", va="top", fontsize=16, color=RUST, fontweight="bold")
    for x in (5.4, 6.3, 7.2, 8.1):
        ax.text(x, line_y + 0.1, "✓", ha="center", va="bottom", fontsize=16, color=GREEN, fontweight="bold")
    cut_x = 4.95
    ax.plot([cut_x, cut_x], [line_y - 0.55, line_y + 0.62], color=INK, linewidth=2.6, zorder=4)
    ax.text(cut_x, 2.2, "θ₀ — one conformal quantile of the pool", ha="center", va="top", fontsize=15.5, color=INK)
    ax.text(
        cut_x,
        1.83,
        "the Inclusion knob slides the quantile; training never sees it",
        ha="center",
        va="top",
        fontsize=15,
        color=SOFT,
    )

    ax.text(5.0, 0.9, "return (M₀, θ₀)", ha="center", va="center", fontsize=18, color=INK, fontweight="bold")

    save(fig, OUT, "calib-xcal-flow.png")


def blend_schedule_fig() -> None:
    n = np.arange(0, 121)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for name, color, style, label, xy, ha, va in (
        ("prod", SOFT, (0, (4, 3)), "historical ramp:\npure x-cal by 20 votes", (40, 0.96), "left", "top"),
        ("cap50", BLUE, (0, (1, 1.6)), "cap50 — binary voting", (23, 0.42), "left", "top"),
        ("slow_cap50", BLUE, "solid", "slow_cap50 — region voting", (44, 0.55), "left", "bottom"),
    ):
        w = [
            get_schedule(name).weight(BlendContext(n_labels=int(k), n_good=int(k) // 2, n_bad=int(k) - int(k) // 2))
            for k in n
        ]
        ax.plot(n, w, color=color, linestyle=style, linewidth=2.4)
        ax.annotate(label, xy=xy, ha=ha, va=va, fontsize=15, color=color)
    ax.set_xlim(0, 122)
    ax.set_ylim(-0.02, 1.08)
    ax.set_yticks([0, 0.5, 1.0], ["pure\nGMM", "0.5", "pure\nx-cal"])
    ax.set_xlabel("votes")
    ax.set_ylabel("weight on the cross-calibration cut")
    ax.set_title("Measured schedules never hand over", loc="left", pad=14, fontsize=16)
    ax.grid(axis="y", color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT, "calib-blend-schedule.png")


def gmm_cut_fig() -> None:
    rng = np.random.default_rng(42)
    neg = rng.normal(0.22, 0.09, 5700)
    pos = rng.normal(0.72, 0.09, 300)
    scores = np.clip(np.concatenate([neg, pos]), 0.0, 1.0)
    fit = fit_score_gmm(scores)
    assert fit is not None
    mid = 0.5 * (fit.mu_lo + fit.mu_hi)

    x = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.hist(scores, bins=48, density=True, color=NEUTRAL_FILL, zorder=1)
    ax.plot(x, fit.w_lo * gaussian(x, fit.mu_lo, fit.var_lo), color=RUST, linewidth=2.4, zorder=3)
    ax.plot(x, fit.w_hi * gaussian(x, fit.mu_hi, fit.var_hi), color=GREEN, linewidth=2.4, zorder=3)
    ax.axvline(mid, color=BLUE, linewidth=2.4, zorder=4)
    ymax = ax.get_ylim()[1]
    ax.annotate(
        "low (Bad) mode",
        xy=(fit.mu_lo, fit.w_lo * gaussian(fit.mu_lo, fit.mu_lo, fit.var_lo)),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        color=RUST,
        fontsize=15,
    )
    ax.annotate(
        "high (Good)\nmode",
        xy=(0.99, fit.w_hi * gaussian(fit.mu_hi, fit.mu_hi, fit.var_hi) * 0.9),
        ha="right",
        color=GREEN,
        fontsize=15,
    )
    ax.annotate(
        "cut: the midpoint\nbetween the modes",
        xy=(mid, ymax * 0.82),
        xytext=(10, 0),
        textcoords="offset points",
        ha="left",
        color=BLUE,
        fontsize=15,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("detector score — every item in the haystack, no labels")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title("A threshold with no labels at all", loc="left", pad=14)
    fig.tight_layout()
    save(fig, OUT, "calib-gmm-cut.png")


def anchored_fig() -> None:
    # Haystack whose negatives are themselves bimodal: unanchored EM splits the
    # negative bulk and parks the cut inside it. A handful of votes re-identify
    # the components.
    rng = np.random.default_rng(7)
    neg = np.concatenate([rng.normal(0.18, 0.06, 1300), rng.normal(0.42, 0.07, 650)])
    pos = rng.normal(0.74, 0.08, 55)
    scores = np.clip(np.concatenate([neg, pos]), 0.0, 1.0)

    good_anchors = np.clip(rng.normal(0.72, 0.07, 6), 0, 1)
    bad_anchors = np.clip(np.concatenate([rng.normal(0.2, 0.05, 9), rng.normal(0.44, 0.05, 6)]), 0, 1)
    a_scores = np.concatenate([good_anchors, bad_anchors])
    a_labels = np.concatenate([np.ones_like(good_anchors), np.zeros_like(bad_anchors)])

    plain = fit_score_gmm(scores)
    anchored, prov = fit_anchored_score_gmm(scores, a_scores, a_labels, anchor_weight=30.0)
    assert plain is not None and anchored is not None and prov == "anchored"
    plain_mid = 0.5 * (plain.mu_lo + plain.mu_hi)
    anch_mid = 0.5 * (anchored.mu_lo + anchored.mu_hi)
    print(f"unanchored cut {plain_mid:.3f}  anchored cut {anch_mid:.3f}")

    x = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.hist(scores, bins=48, density=True, color=NEUTRAL_FILL, zorder=1)
    ax.plot(
        x,
        plain.w_lo * gaussian(x, plain.mu_lo, plain.var_lo) + plain.w_hi * gaussian(x, plain.mu_hi, plain.var_hi),
        color=SOFT,
        linewidth=1.8,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    ax.plot(x, anchored.w_lo * gaussian(x, anchored.mu_lo, anchored.var_lo), color=RUST, linewidth=2.4, zorder=3)
    ax.plot(x, anchored.w_hi * gaussian(x, anchored.mu_hi, anchored.var_hi), color=GREEN, linewidth=2.4, zorder=3)
    ax.axvline(plain_mid, color=SOFT, linewidth=2.0, linestyle=(0, (4, 3)), zorder=4)
    ax.axvline(anch_mid, color=BLUE, linewidth=2.4, zorder=4)

    ymax = ax.get_ylim()[1]
    for s in bad_anchors:
        ax.plot([s, s], [-0.05 * ymax, 0.035 * ymax], color=RUST, linewidth=2.2, zorder=5, clip_on=False)
    for s in good_anchors:
        ax.plot([s, s], [-0.05 * ymax, 0.035 * ymax], color=GREEN, linewidth=2.2, zorder=5, clip_on=False)
    ax.annotate(
        "unanchored fit\ncuts inside the\nnegative bulk",
        xy=(plain_mid, ymax * 0.8),
        xytext=(-10, 0),
        textcoords="offset points",
        ha="right",
        color=SOFT,
        fontsize=15,
    )
    ax.annotate(
        "anchored cut",
        xy=(anch_mid, ymax * 0.8),
        xytext=(10, 0),
        textcoords="offset points",
        ha="left",
        color=BLUE,
        fontsize=15,
    )
    ax.annotate(
        "votes, clamped one-hot at mass κ",
        xy=(0.44, -0.175 * ymax),
        ha="center",
        va="top",
        color=INK,
        fontsize=15,
        annotation_clip=False,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("detector score", labelpad=44)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title("Votes re-identify the components", loc="left", pad=14)
    fig.tight_layout()
    save(fig, OUT, "calib-anchored-em.png")


def decomposition_fig() -> None:
    # docs/experiments/gmm-cut/REPORT-2881.md — the #2879 re-measure of #2836's
    # decomposition (region arm, ramp 6-20): total excess cost 0.0686.
    terms = [
        ("identification", 0.0057),
        ("prior / loss", 0.0111),
        ("Gaussian\nmisspecification", 0.0129),
        ("sim → test\ntransfer", 0.0389),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ys = np.arange(len(terms))
    for y, (name, v) in zip(ys, terms):
        emphasized = name.startswith("sim")
        ax.barh(y, v, height=0.55, color=BLUE if emphasized else "#9aa4b0", zorder=2)
        ax.annotate(
            f"{v:.4f}",
            xy=(v, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=15,
            color=BLUE if emphasized else SOFT,
            fontweight="bold" if emphasized else "normal",
        )
    ax.set_yticks(ys, [t[0] for t in terms])
    ax.set_xlim(0, 0.047)
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(left=False)
    ax.set_title("Where the threshold error lives", loc="left", pad=34, fontsize=16)
    ax.annotate(
        "excess cost vs the test oracle, region arm",
        xy=(0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 8),
        textcoords="offset points",
        fontsize=15,
        color=SOFT,
    )
    fig.tight_layout()
    save(fig, OUT, "calib-error-decomposition.png")


if __name__ == "__main__":
    xcal_flow_fig()
    blend_schedule_fig()
    gmm_cut_fig()
    anchored_fig()
    decomposition_fig()
    print("wrote figures to", OUT)
