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
    """Schematic of the original cross-calibration idea (issue #3207).

    Follows the issue's hand mockup: this is the *iteration-1* slide, so it
    draws the simple initial version — partition the labelled data in half,
    train a model per half, score the half each model never saw, find a cut
    per half, and average the two cuts. The refinements the shipped code adds
    on top (pooled folds, the conformal quantile, re-drawn splits) are later
    iterations of the deck's story and deliberately absent here.

    Layout rules carried over from the mockup on purpose: the split is drawn,
    not written — D₀ is one block with a centre divider whose halves are
    labelled D₁ and D₂; the train arrows D₁ → M₁ and D₂ → M₂ are vertical, so
    the only diagonals are the two scoring paths D₂ → M₁ → M₁(D₂) and
    D₁ → M₂ → M₂(D₁), which cross into the X that names cross-calibration;
    and one held-out ordering is imperfect — a Bad lands above θ₂ — because a
    fold model's ranking of votes it never saw is not trivially clean.
    Green = Good media, red = Bad media (amorphous regions); everything else
    ink on white.
    """
    from matplotlib.patches import FancyArrow, FancyArrowPatch, Rectangle  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_axis_off()

    def data_block(x0: float, y0: float, w: float, h: float, split: bool = False) -> None:
        """A block of labelled media: an amorphous green (Good) region over a red (Bad) one."""
        good_h = 0.42 * h
        ax.add_patch(
            Rectangle(
                (x0, y0 + h - good_h),
                w,
                good_h,
                facecolor="white",
                edgecolor=GREEN,
                hatch="//////",
                linewidth=0,
                zorder=2,
            )
        )
        ax.add_patch(
            Rectangle((x0, y0), w, h - good_h, facecolor="white", edgecolor=RUST, hatch="\\\\\\", linewidth=0, zorder=2)
        )
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=3))
        ax.plot([x0, x0 + w], [y0 + h - good_h] * 2, color=INK, linewidth=1.0, zorder=3)
        if split:
            ax.plot([x0 + w / 2] * 2, [y0, y0 + h], color=INK, linewidth=1.6, zorder=3)

    def arrow(xy_from: tuple[float, float], xy_to: tuple[float, float]) -> None:
        ax.add_patch(
            FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=14, color=INK, linewidth=1.6, zorder=2)
        )

    def labeled_arrow(xy_from: tuple[float, float], xy_to: tuple[float, float], label: str, z: float = 2.0) -> None:
        """An outlined block arrow with an open core, *label* written along the shaft.

        The label sits at the centre of the shaft, both length-wise and
        width-wise, rotated to the arrow's own angle. 14.5pt is the smallest
        size the 20px type floor admits in the 56% slot — small enough to sit
        fully inside the 0.5-unit shaft with clear margins. Where two of these
        arrows cross, the higher-*z* one's open core simply covers the other's
        label near the crossing; that is deliberate (issue #3207 review).
        """
        (x0, y0), (x1, y1) = xy_from, xy_to
        dx, dy = x1 - x0, y1 - y0
        length = float(np.hypot(dx, dy))
        head_len = 0.32
        ax.add_patch(
            FancyArrow(
                x0,
                y0,
                dx,
                dy,
                width=0.5,
                head_width=0.66,
                head_length=head_len,
                length_includes_head=True,
                facecolor="white",
                edgecolor=INK,
                linewidth=1.4,
                zorder=z,
            )
        )
        frac = 0.5 * (length - head_len) / length
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if angle < -90 or angle > 90:  # keep the label reading left-to-right
            angle += 180
        ax.text(
            x0 + dx * frac,
            y0 + dy * frac,
            label,
            rotation=angle,
            ha="center",
            va="center",
            fontsize=14.5,
            color=INK,
            zorder=z + 0.05,
        )

    def model_box(cx: float, cy: float, label: str) -> None:
        w, h = 0.85, 0.62
        ax.add_patch(
            Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor="white", edgecolor=INK, linewidth=1.6, zorder=3)
        )
        # center_baseline + a hair of lift: optically centers the cap-height "M"
        # in the box despite the subscript digit hanging below the baseline.
        ax.text(cx, cy + 0.025, label, ha="center", va="center_baseline", fontsize=16, color=INK, zorder=4)

    def score_line(cx: float, label: str, bad: list[float], good: list[float], theta_x: float, theta: str) -> None:
        """Held-out scores on a number line: Bad low, Good high, a cut between."""
        y = 3.65
        ax.plot([cx - 1.7, cx + 1.7], [y, y], color=INK, linewidth=1.8, zorder=2)
        ax.text(cx, 4.15, label, ha="center", va="bottom", fontsize=16, color=INK)
        for x in bad:
            ax.text(cx + x, y - 0.12, "✗", ha="center", va="top", fontsize=16, color=RUST, fontweight="bold")
        for x in good:
            ax.text(cx + x, y + 0.08, "✓", ha="center", va="bottom", fontsize=16, color=GREEN, fontweight="bold")
        ax.plot([cx + theta_x] * 2, [y - 0.32, y], color=INK, linewidth=2.2, zorder=3)
        ax.text(cx + theta_x, y - 0.44, theta, ha="center", va="top", fontsize=16, color=INK)

    # ── D0, already drawn split in half, and the model trained on all of it ───
    ax.text(5.0, 9.8, "D₀", ha="center", va="bottom", fontsize=16, color=INK)
    data_block(3.2, 8.7, 3.6, 1.0, split=True)
    ax.text(3.05, 9.45, "Good", ha="right", va="center", fontsize=15, color=GREEN)
    ax.text(3.05, 8.95, "Bad", ha="right", va="center", fontsize=15, color=RUST)
    ax.text(4.1, 8.55, "D₁", ha="center", va="top", fontsize=16, color=INK)
    ax.text(5.9, 8.55, "D₂", ha="center", va="top", fontsize=16, color=INK)
    labeled_arrow((6.9, 9.2), (8.32, 9.2), "train")
    model_box(8.8, 9.2, "M₀")

    # ── train a model per half; score the other half ──────────────────────────
    # The train arrows are vertical; the only diagonals are the scoring paths
    # D₂ → M₁ → M₁(D₂) and D₁ → M₂ → M₂(D₁), whose crossing draws the X that
    # names cross-calibration.
    model_box(4.1, 6.3, "M₁")
    model_box(5.9, 6.3, "M₂")
    labeled_arrow((4.1, 8.1), (4.1, 6.68), "train")
    labeled_arrow((5.9, 8.1), (5.9, 6.68), "train")
    # Each scoring path is one geometric line (slope Δx/Δy = 0.789) that runs
    # from under the opposite half, through the M box, down to the score line —
    # the entry and exit arrows are collinear, so the eye reads one straight
    # stroke and the two strokes cross in a symmetric X.
    labeled_arrow((5.52, 8.1), (4.42, 6.71), "score", z=2.1)
    labeled_arrow((4.48, 8.1), (5.58, 6.71), "score", z=2.2)
    arrow((3.82, 5.94), (2.88, 4.75))
    arrow((6.18, 5.94), (7.12, 4.75))

    # ── per-half held-out scores; cut each, average ───────────────────────────
    score_line(2.3, "M₁(D₂)", bad=[-1.4, -0.95, -0.5, -0.05], good=[0.5, 0.95, 1.4], theta_x=0.22, theta="θ₁")
    score_line(7.7, "M₂(D₁)", bad=[-1.4, -1.0, -0.6, 0.3], good=[0.0, 0.7, 1.15], theta_x=-0.3, theta="θ₂")

    ax.text(5.0, 2.5, "θ₀ = avg(θ₁, θ₂)", ha="center", va="center", fontsize=16.5, color=INK)
    ax.text(5.0, 1.6, "return (M₀, θ₀)", ha="center", va="center", fontsize=18, color=INK, fontweight="bold")

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
