#!/usr/bin/env python
"""Figures for the Test slides: what the Test button measures, and how (#4594).

Run from the repo root:

    python slides/figs/src/make-test-figs.py

The Test button scores a corpus the detector never trained on and opens the
Test autopilot, which asks one question of the detector's line there: *of what
it keeps, how much is right, and of all the matches, how much does it keep?*
The answer is `vtscore/training/thresholds/line_test.py` (#4527): uniform
picks within rank bands on both sides of the line, Beta posteriors drawn
jointly, a prior borrowed from neighbouring bands above the line, the model's
own count corrected by the picks below it, an allocation rule, and a stop.

Schematic inputs, real code. Every figure but the last runs the shipped
`LineTest` on a planted corpus whose truth is known, answering each pick from
that truth, so the numbers on the slides are the ones the app would show for
those picks and the truth they are judged against is the planted one:

* the running example (`_run`): 2,048 photos, a line at the middle radio's
  expected peak that keeps 32, and a model that reads the matches a little
  short of the truth, as a session's labels line does. Every Test mechanism
  fires in it: the first pass up from the line, one round placed by the
  allocation rule, the walk below the line to its budget, and a deepest band
  the walk never reaches.
* Good Neighbours (`_sparse_run`): a big, sparse line, 2,048 kept of 16,384 at
  the recall end, where independent Jeffreys priors over-read the empty bands
  (#4523) and the neighbours' prior is the fix (#4539, #4560).

The last figure re-plots the shipped estimator's coverage from
`docs/experiments/2026-10-06-pooled-taper-4560/summary_shipped.csv`, the
#4523 replay of 3,669 saved sessions at the defaults the app ships.

Every canvas is the whole 1280x720 slide at 100 px to the inch, saved at its
declared bounds, so every stage of a build shares one framing by construction
and `save()` checks the title notch against exactly what the slide shows.
"""

from __future__ import annotations

import csv
import functools
import math
import sys
from pathlib import Path

# Ensure the repo root is importable no matter the cwd: ``python
# slides/figs/src/x.py`` only puts the script's own dir on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import FULL_BLEED, INK, SOFT, save  # noqa: E402

from vtscore.training.thresholds.line_test import JEFFREYS, TEST_ALPHA, LineTest  # noqa: E402

OUT = Path(__file__).resolve().parent.parent
SUMMARY = _REPO_ROOT / "docs" / "experiments" / "2026-10-06-pooled-taper-4560" / "summary_shipped.csv"

# The calibration figures' palette (`make-calib-figs.py`, `themes/vtsearch.css`):
# blue is the line, red the losing arm and a Bad, green a Good and the arm that
# wins against a red one (a red mark is the wrong one throughout the deck, so
# its alternative is the positive side's colour, not the line's blue: #4563).
# Everything else is ink and greys.
BLUE = "#0b5fa5"
RED = "#b91c1c"
GREEN = "#0d8a5f"
RULE = "#d8dee6"
NEUTRAL_FILL = "#e8ebef"
UNLABELED_FILL = "#dae0e8"
MODEL_GREY = "#9aa3ae"
#: The grid a chart's values are read against: the shade #4563 settled on for
#: the deck's precision-recall panel, which at the old lighter shade vanished on
#: a projector. Still lighter than the axes, so it stays behind the data.
GRID_COLOUR, GRID_LW = "#bcc4ce", 1.3
#: A chart slide's own figures run up the slide right of the title notch, as
#: Follow Suit's do (#4563), and the numbers that read it stack under the notch
#: in this column: `SIDE_X` is its left edge, `SIDE_TOP` its first line.
CHART_X0 = 0.37
SIDE_X, SIDE_TOP = 0.04, 0.70

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 17,
        "text.color": INK,
        "axes.edgecolor": SOFT,
        "axes.labelcolor": INK,
        "axes.labelsize": 17,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
    }
)

#: The canvas: the whole slide, at 100 slide pixels to the inch, in drawing
#: units of one inch. 15pt is the smallest type here: 20.8px on the slide.
W, H = 12.8, 7.2
#: Where a statement over the drawing starts: right of the title notch
#: (`slide_figure.TITLE_NOTCH_PX`, 60-360 px across), with a margin.
STATEMENT_X = 4.0

# ── the running example ──────────────────────────────────────────────────────

#: The planted corpus: how many photos Test scored, and the seeds of its truth
#: and of the Test's own picks.
CORPUS_N = 2048
CORPUS_SEED = 12
TEST_SEED = 12
#: Each photo's chance of being a match, by rank: rich at the top, falling
#: through the line, a thin tail all the way down.
CHANCE_TOP, CHANCE_SCALE, CHANCE_SLOPE, CHANCE_FLOOR = 0.95, 19.0, 1.7, 0.003
#: The model reads every chance this much short of the truth: the labels line
#: reads low early in a session, because a session's Goods are its easiest.
MODEL_SHORT = 0.85
#: The balance the Test runs at: the middle radio, every detector's default.
LINE_BETA = 1.0
#: The three radios, left to right as the Threshold shows them.
PRESETS = (4.0, 1.0, 0.25)


def _chance(n: int, scale: float, floor: float) -> np.ndarray:
    ranks = np.arange(n)
    return CHANCE_TOP / (1 + (ranks / scale) ** CHANCE_SLOPE) + floor


@functools.lru_cache(maxsize=None)
def _corpus() -> tuple[np.ndarray, np.ndarray]:
    """The planted truth, in rank order, and the model's chance for each photo."""
    chance = _chance(CORPUS_N, CHANCE_SCALE, CHANCE_FLOOR)
    truth = np.random.default_rng(CORPUS_SEED).random(CORPUS_N) < chance
    return truth, np.clip(MODEL_SHORT * chance, 0.0, 1.0)


def _preset_count(beta: float) -> int:
    """Where a radio's line falls: the expected F-beta's peak down the ranking, from the model's chances.

    The labels line's rule (`labels_line.py`): *kept* counted, *hits* the sum of
    the kept photos' chances, *matches* the sum over the whole corpus.
    """
    _, post = _corpus()
    hits = np.cumsum(post)
    kept = np.arange(1, len(post) + 1)
    fbeta = (1 + beta * beta) * hits / (beta * beta * post.sum() + kept)
    return int(np.argmax(fbeta)) + 1


@functools.lru_cache(maxsize=None)
def _run() -> tuple[LineTest, tuple[tuple[int, tuple[tuple[int, bool], ...]], ...]]:
    """The running example to Done: the test, and its rounds as ``(band, ((id, match), ...))``."""
    truth, post = _corpus()
    test = LineTest.start(range(CORPUS_N), _preset_count(LINE_BETA), LINE_BETA, posteriors=post, seed=TEST_SEED)
    rounds = []
    while True:
        picks = test.draw()
        if not picks:
            break
        band = test.band
        labels = tuple((int(i), bool(truth[i])) for i in picks)
        test.record(dict(labels))
        rounds.append((band, labels))
    # The shape the slides narrate: a line of 32, three bands above it audited
    # from the line up, one round the allocation rule placed, then the walk one
    # band deeper a round to its 40 picks, short of the deepest band.
    assert test.line_count == 32, test.line_count
    assert tuple(b for b, _ in rounds) == (2, 1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10), [b for b, _ in rounds]
    report = test.phase()
    assert (report.phase, report.matches_stop, report.misses_stop) == ("done", "budget", "budget"), report
    return test, tuple(rounds)


def _after(rounds: int) -> LineTest:
    """The running example as it stood after its first *rounds* rounds (the same picks, restored)."""
    test, history = _run()
    _, post = _corpus()
    labels = {cid: match for _, picks in history[:rounds] for cid, match in picks}
    return LineTest.start(test.ranking_ids, test.line_count, test.beta, posteriors=post, seed=TEST_SEED, labels=labels)


def _truth_at(count: int) -> tuple[float, float, float]:
    """The planted precision, recall and F-beta (at the line's beta) of keeping the top *count*."""
    truth, _ = _corpus()
    hits, matches = float(truth[:count].sum()), float(truth.sum())
    b2 = LINE_BETA * LINE_BETA
    return hits / count, hits / matches, (1 + b2) * hits / (b2 * matches + count)


def _right(picks: tuple[tuple[int, bool], ...]) -> int:
    return sum(match for _, match in picks)


# ── shared drawing ───────────────────────────────────────────────────────────


def _side_block(fig: Figure, lines: list[tuple[str, str, float]], top: float = SIDE_TOP) -> None:
    """A column of ``(words, style, gap before)`` lines under the notch; *style* is ``head``, ``body`` or a colour name.

    ``head`` is a bold heading in ink, ``body`` plain ink indented under it, and
    any other style a bold line in that colour: the arm it names, as the chart
    draws it.
    """
    y = top
    for words, style, gap in lines:
        y -= gap
        if style == "body":
            fig.text(SIDE_X + 0.015, y, words, ha="left", va="center", fontsize=16)
        else:
            colour = INK if style == "head" else style
            fig.text(SIDE_X, y, words, ha="left", va="center", fontsize=17, fontweight="bold", color=colour)


def _canvas() -> tuple[Figure, plt.Axes]:
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    return fig, ax


def _save(stages: int, draw, name: str) -> None:
    """Every stage of a build at the canvas's declared bounds, the last as the slide's own figure."""
    for stage in range(1, stages):
        save(draw(stage), OUT, f"{name}.build{stage}.png", column=FULL_BLEED, tight=False)
    save(draw(stages), OUT, f"{name}.png", column=FULL_BLEED, tight=False)


def _pct(share: float) -> str:
    return f"{100 * share:.0f}%"


def _span(lo: float, hi: float, fine: bool = False) -> str:
    """A range of shares as the result pane says it: ``44–66%``; *fine* gives a share under 10% a decimal."""
    return f"{_fine(lo) if fine else f'{100 * lo:.0f}'}–{_fine(hi) if fine else f'{100 * hi:.0f}'}%"


def _fine(share: float) -> str:
    return f"{100 * share:.1f}" if share < 0.1 else f"{100 * share:.0f}"


def _mark(ax: plt.Axes, x: float, y: float, match: bool, size: float = 20, va: str = "bottom") -> None:
    ax.text(
        x,
        y,
        "✓" if match else "✗",
        ha="center",
        va=va,
        fontsize=size,
        fontweight="bold",
        color=GREEN if match else RED,
    )


def _bracket(ax: plt.Axes, x0: float, x1: float, y: float, down: bool = False, lw: float = 1.6) -> None:
    """A bracket from *x0* to *x1*, its feet pointing at what it spans (down onto it unless *down*)."""
    foot = 0.12 if down else -0.12
    ax.plot([x0, x0, x1, x1], [y - foot, y, y, y - foot], color=INK, linewidth=lw, zorder=5)


def _line(ax: plt.Axes, x: float, y0: float, y1: float, label: bool = True, label_y: float | None = None) -> None:
    """The line, in the palette's blue: the one shipped decision on every drawing."""
    ax.plot([x, x], [y0, y1], color=BLUE, linewidth=2.8, zorder=6, solid_capstyle="butt")
    if label:
        ax.text(x - 0.12, y0 if label_y is None else label_y, "the line", ha="right", va="bottom", fontsize=16)


#: The band strip the test figures share: one equal-width cell per band, best on
#: the right as every ranking in the deck is drawn, each cell twice the size of
#: the one to its right. A doubling scale rather than a linear one, because on a
#: linear one the 32 the line keeps would be a sliver of 2,048.
BAND_X0, BAND_X1 = 0.9, 12.1


def _band_cell(test: LineTest, band: int) -> tuple[float, float]:
    """``(left, width)`` of *band*'s cell on the strip: band 0, the top of the ranking, is rightmost."""
    width = (BAND_X1 - BAND_X0) / len(test.bands)
    return BAND_X1 - (band + 1) * width, width


def _line_x(test: LineTest) -> float:
    return _band_cell(test, len(test.above) - 1)[0]


def _band_strip(ax: plt.Axes, test: LineTest, y0: float, h: float, sizes: bool = True) -> None:
    """The bands as cells, each with its size, and the line between the two sides."""
    for band in test.bands:
        x, w = _band_cell(test, band.index)
        ax.add_patch(Rectangle((x, y0), w, h, facecolor=UNLABELED_FILL, edgecolor="white", linewidth=2.5, zorder=2))
        if sizes:
            ax.text(x + w / 2, y0 + h / 2, f"{band.size:,}", ha="center", va="center", fontsize=16, zorder=3)


# ── Final Exam: the question ─────────────────────────────────────────────────

QUESTION_STAGES = 4


def question_fig() -> None:
    """What Test asks of a line, and the two rules that make the answer mean anything.

    Four stages: the corpus Test scored, ranked, and the line through it; the
    first half of the question, how much of what the line keeps is right
    (*Right*); the second, how many of all the matches it keeps (*Found*),
    which needs the matches the line left behind counted too; and the two
    rules — every number from picks drawn at random, and a test vote never
    trains the detector.
    """
    _save(QUESTION_STAGES, _question_stage, "test-question")


def _question_stage(stage: int) -> Figure:
    fig, ax = _canvas()
    test, _ = _run()
    y0, h = 3.0, 0.62
    line_x = BAND_X1 - 2.8
    kept_mid, rest_mid = (line_x + BAND_X1) / 2, (BAND_X0 + line_x) / 2
    # ── stage 1: the corpus, ranked, and the line ────────────────────────────
    ax.text(
        STATEMENT_X,
        6.2,
        f"Test scored {test.size:,} photos the detector never saw",
        ha="left",
        va="center",
        fontsize=19,
        fontweight="bold",
    )
    for x0, x1 in ((BAND_X0, line_x), (line_x, BAND_X1)):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, h, facecolor=UNLABELED_FILL, edgecolor="white", linewidth=2.5))
    ax.text(BAND_X0, y0 - 0.12, "ranked, best on the right", ha="left", va="top", fontsize=15, color=SOFT)
    brace_y = y0 - 0.62
    _bracket(ax, line_x + 0.06, BAND_X1, brace_y, down=True)
    ax.text(kept_mid, brace_y - 0.1, f"kept: the top {test.line_count}", ha="center", va="top", fontsize=16)
    _bracket(ax, BAND_X0, line_x - 0.06, brace_y, down=True)
    ax.text(
        rest_mid,
        brace_y - 0.1,
        f"left behind: the other {test.size - test.line_count:,}",
        ha="center",
        va="top",
        fontsize=16,
    )
    _line(ax, line_x, y0 - 0.25, y0 + h + 0.3, label=False)
    # ── stage 2: Right ───────────────────────────────────────────────────────
    q_y = y0 + h + 0.55
    if stage >= 2:
        ax.text(kept_mid, q_y + 0.95, "Right", ha="center", va="bottom", fontsize=20, fontweight="bold")
        for row, words in enumerate((f"of these {test.line_count},", "how many are matches?")):
            ax.text(kept_mid, q_y + 0.55 - row * 0.4, words, ha="center", va="bottom", fontsize=17)
    # ── stage 3: Found ───────────────────────────────────────────────────────
    if stage >= 3:
        ax.text(rest_mid, q_y + 0.95, "Found", ha="center", va="bottom", fontsize=20, fontweight="bold")
        for row, words in enumerate(
            ("of all the matches, how many did the line keep?", "the ones it left behind have to be counted too")
        ):
            ax.text(rest_mid, q_y + 0.55 - row * 0.4, words, ha="center", va="bottom", fontsize=17)
    # ── stage 4: the rules ───────────────────────────────────────────────────
    if stage >= 4:
        for row, words in enumerate(
            (
                "Every number comes from photos picked at random, five at a time,",
                "and a test vote never trains the detector it is testing.",
            )
        ):
            ax.text(W / 2, 1.05 - row * 0.5, words, ha="center", va="center", fontsize=18)
    return fig


# ── Band Practice: stratified sampling ───────────────────────────────────────

BANDS_STAGES = 4
#: The kept set as a strip of cells, best on the right.
KEPT_X0, KEPT_W, KEPT_Y0, KEPT_H = 1.0, 11.0, 3.0, 0.62


def bands_fig() -> None:
    """Stratified sampling on the kept set: bands, uniform picks within each, and weighting by band size.

    Four stages: the kept 32 cut into bands from the top (8, 8, 16); five
    picks drawn at random from each, with their votes (the running example's
    first pass); each band's share right times its size, summed — and why
    the picks cannot simply be counted together, since equal picks per band
    over-represent the small top bands; and the same picks read at every
    depth, which is what the result's chart draws.
    """
    _save(BANDS_STAGES, _bands_stage, "test-bands")


def _kept_cell_x(rank: int, k: int) -> float:
    """The left edge of the cell for *rank* (0 is the top) in a strip of the kept *k*, best on the right."""
    return KEPT_X0 + (k - 1 - rank) * KEPT_W / k


def _bands_stage(stage: int) -> Figure:
    fig, ax = _canvas()
    test, history = _run()
    k = test.line_count
    cell_w = KEPT_W / k
    first_pass = {band: picks for band, picks in history[: len(test.above)]}
    names = ("top 8", "next 8", "next 16")
    assert tuple(b.size for b in test.above) == (8, 8, 16)

    # ── stage 1: the kept set, cut into bands ────────────────────────────────
    ax.text(
        STATEMENT_X, 6.2, f"The {k} the line keeps, cut into bands from the top", ha="left", va="center", fontsize=19
    )
    for rank in range(k):
        ax.add_patch(
            Rectangle(
                (_kept_cell_x(rank, k), KEPT_Y0),
                cell_w,
                KEPT_H,
                facecolor=UNLABELED_FILL,
                edgecolor="white",
                linewidth=1.5,
            )
        )
    ax.text(KEPT_X0, KEPT_Y0 - 0.12, "ranked, best on the right", ha="left", va="top", fontsize=15, color=SOFT)
    bracket_y = KEPT_Y0 + KEPT_H + 0.75
    spans = {}
    for band in test.above:
        x0 = _kept_cell_x(band.hi - 1, k) + 0.05
        x1 = _kept_cell_x(band.lo, k) + cell_w - 0.05
        spans[band.index] = (x0, x1)
        _bracket(ax, x0, x1, bracket_y)
        ax.text((x0 + x1) / 2, bracket_y + 0.1, names[band.index], ha="center", va="bottom", fontsize=17)

    # ── stage 2: five picks at random in each band, and their votes ──────────
    if stage >= 2:
        for band, picks in first_pass.items():
            for cid, match in picks:
                x = _kept_cell_x(test.ranking_ids.index(cid), k)
                ax.add_patch(
                    Rectangle(
                        (x + 0.03, KEPT_Y0 + 0.03),
                        cell_w - 0.06,
                        KEPT_H - 0.06,
                        facecolor="none",
                        edgecolor=INK,
                        linewidth=2.6,
                        zorder=4,
                    )
                )
                _mark(ax, x + cell_w / 2, KEPT_Y0 + KEPT_H + 0.06, match)

    # ── stage 3: weight each band by its size ────────────────────────────────
    if stage >= 3:
        _bands_weights(ax, test, first_pass, spans)
    # ── stage 4: the same picks read every depth ─────────────────────────────
    if stage >= 4:
        _bands_depths(ax, test, first_pass)
    return fig


def _bands_weights(ax: plt.Axes, test: LineTest, first_pass: dict, spans: dict) -> None:
    """Each band's share right times its size, their sum, and what counting the picks as one pile says instead."""
    k = test.line_count
    parts = []
    for band in test.above:
        picks = first_pass[band.index]
        right = _right(picks)
        share = band.size * right / len(picks)
        parts.append(share)
        x0, x1 = spans[band.index]
        ax.text(
            (x0 + x1) / 2,
            KEPT_Y0 - 0.75,
            f"{band.size} × {right}/{len(picks)} = {share:.1f}",
            ha="center",
            va="center",
            fontsize=17,
        )
    estimate = sum(parts)
    sum_words = " + ".join(f"{p:.1f}" for p in reversed(parts))
    ax.text(
        W / 2,
        1.5,
        f"{sum_words} = {estimate:.1f} of {k}: {_pct(estimate / k)} right",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
    )
    picks_total = sum(len(p) for p in first_pass.values())
    right_total = sum(_right(p) for p in first_pass.values())
    top_half = sum(len(first_pass[b]) for b in (0, 1))
    ax.text(
        W / 2,
        0.85,
        f"Counted as one pile, {right_total} of {picks_total} is {_pct(right_total / picks_total)}: "
        f"the top half of the set drew {top_half} of the {picks_total} picks",
        ha="center",
        va="center",
        fontsize=17,
    )


def _bands_depths(ax: plt.Axes, test: LineTest, first_pass: dict) -> None:
    """The same picks read at every band edge: the top 8, the top 16, the top 32."""
    depth, running = [], 0.0
    for band in test.above:
        picks = first_pass[band.index]
        running += band.size * _right(picks) / len(picks)
        depth.append(f"top {band.hi} {_pct(running / band.hi)}")
    ax.text(
        STATEMENT_X,
        5.55,
        "Read at every depth: " + " · ".join(depth),
        ha="left",
        va="center",
        fontsize=17,
    )


# ── Posterior Motive: one band's Beta posterior ──────────────────────────────

POSTERIOR_STAGES = 4
#: The band the slide reads, and its two rounds: the running example's band of
#: 16 just above the line, 2 of 5 right in the first pass and 1 of 5 in the
#: round the allocation rule sent back to it.
POSTERIOR_BAND = 2


def posterior_fig() -> None:
    """One band, the textbook way: a Beta posterior on its share right, then its count of matches.

    Four stages: Jeffreys' prior, Beta(1/2, 1/2), before any pick; the
    posterior after 2 of 5 right, with its central 95% range; the band's count
    of matches, the picks' own plus the unseen photos drawn binomially at a
    share drawn from that posterior (a beta-binomial); and the same after a
    second round, narrower. A band no bigger than a round is counted outright,
    and then the count is exact.
    """
    _save(POSTERIOR_STAGES, _posterior_stage, "test-posterior")


def _beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    log_b = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return np.exp((a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - log_b)


def _beta_range(a: float, b: float, alpha: float = TEST_ALPHA) -> tuple[float, float]:
    """The central ``1 - alpha`` range of Beta(*a*, *b*), off a fine grid of its density."""
    x = np.linspace(1e-6, 1 - 1e-6, 200_001)
    cdf = np.cumsum(_beta_pdf(x, a, b))
    cdf /= cdf[-1]
    return float(np.interp(alpha / 2, cdf, x)), float(np.interp(1 - alpha / 2, cdf, x))


def _beta_binomial(unseen: int, a: float, b: float) -> np.ndarray:
    """P(*j* of *unseen* are matches), *j* = 0 … *unseen*, at a share drawn from Beta(*a*, *b*)."""
    log_b = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    out = []
    for j in range(unseen + 1):
        log_bj = math.lgamma(j + a) + math.lgamma(unseen - j + b) - math.lgamma(unseen + a + b)
        out.append(math.comb(unseen, j) * math.exp(log_bj - log_b))
    return np.array(out)


def _count_range(probs: np.ndarray, offset: int, alpha: float = TEST_ALPHA) -> tuple[int, int]:
    cdf = np.cumsum(probs)
    lo = int(np.searchsorted(cdf, alpha / 2)) + offset
    hi = int(np.searchsorted(cdf, 1 - alpha / 2)) + offset
    return lo, hi


def _posterior_rounds() -> tuple[int, tuple[tuple[int, int], tuple[int, int]]]:
    """The band's size and its ``(picks, right)`` after one round and after two."""
    test, history = _run()
    rounds = [picks for band, picks in history if band == POSTERIOR_BAND]
    assert len(rounds) == 2, rounds
    one = (len(rounds[0]), _right(rounds[0]))
    two = (one[0] + len(rounds[1]), one[1] + _right(rounds[1]))
    return test.bands[POSTERIOR_BAND].size, (one, two)


def _posterior_stage(stage: int) -> Figure:
    fig = plt.figure(figsize=(W, H))
    size, ((n1, r1), (n2, r2)) = _posterior_rounds()
    a1, b1 = JEFFREYS + r1, JEFFREYS + n1 - r1

    # ── left: the share ──────────────────────────────────────────────────────
    left = fig.add_axes((0.08, 0.15, 0.40, 0.53))
    left.set_xlim(0, 1)
    left.set_ylim(0, 3.0)
    left.set_yticks([])
    left.spines["left"].set_visible(False)
    left.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    left.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    left.set_xlabel("the band's share right")
    fig.text(0.28, 0.715, "One band's share right", ha="center", va="bottom", fontsize=19, fontweight="bold")
    x = np.linspace(0.002, 0.998, 1200)
    # ── stage 1: Jeffreys' prior ─────────────────────────────────────────────
    left.plot(x, _beta_pdf(x, JEFFREYS, JEFFREYS), color=SOFT, linewidth=2.4, linestyle=(0, (5, 3)))
    for row, words in enumerate(("before a pick:", "Beta(½, ½)")):
        left.text(0.94, 2.75 - row * 0.3, words, ha="right", va="center", fontsize=16, color=SOFT)
    # ── stage 2: the posterior after one round ───────────────────────────────
    if stage >= 2:
        lo, hi = _beta_range(a1, b1)
        inside = (x >= lo) & (x <= hi)
        left.fill_between(x[inside], _beta_pdf(x[inside], a1, b1), color=NEUTRAL_FILL, zorder=1)
        left.plot(x, _beta_pdf(x, a1, b1), color=INK, linewidth=3.0, zorder=3)
        mode = (a1 - 1) / (a1 + b1 - 2)
        peak = float(_beta_pdf(np.array([mode]), a1, b1)[0])
        for row, words in enumerate((f"after {r1} of {n1} right:", f"Beta({_half(a1)}, {_half(b1)})")):
            left.text(mode, peak + 0.45 - row * 0.3, words, ha="center", va="bottom", fontsize=16)
        left.text(
            (lo + hi) / 2, 0.25, f"likely {_span(lo, hi)}", ha="center", va="bottom", fontsize=16, fontweight="bold"
        )

    # ── right: the band's count, after one round and after two ───────────────
    if stage >= 3:
        fig.text(0.76, 0.715, f"That band's count of matches, of {size}", ha="center", va="bottom", fontsize=19,
                 fontweight="bold")  # fmt: skip
    for row, ((n, r), show) in enumerate((((n1, r1), stage >= 3), ((n2, r2), stage >= 4))):
        if not show:
            continue
        ax = fig.add_axes((0.57, 0.45 - row * 0.3, 0.38, 0.17))
        a, b = JEFFREYS + r, JEFFREYS + n - r
        probs = _beta_binomial(size - n, a, b)
        counts = np.arange(r, r + len(probs))
        lo, hi = _count_range(probs, r)
        colours = [INK if lo <= c <= hi else MODEL_GREY for c in counts]
        ax.bar(counts, probs, width=0.75, color=colours)
        ax.set_xlim(-0.6, size + 0.6)
        ax.set_ylim(0, probs.max() * 1.08)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xticks(range(0, size + 1, 4))
        if row == 1:
            ax.set_xlabel("matches in the band")
        ax.text(
            0.0,
            1.12,
            f"{n} of {size} seen, {r} right: likely {lo}–{hi}",
            ha="left",
            va="bottom",
            fontsize=16,
            transform=ax.transAxes,
        )
    return fig


def _half(value: float) -> str:
    """A Beta parameter as the slide writes it: ``2½`` rather than ``2.5``."""
    whole = int(math.floor(value))
    return f"{whole}½" if value - whole > 0.25 else f"{whole}"


# ── Luck of the Draw: joint Monte Carlo ──────────────────────────────────────

DRAWS_STAGES = 3
#: The draws the table prints before eliding the rest.
DRAWS_SHOWN = 5
#: The table: each column's x centre, the two lines of its header, and the rows.
DRAW_COLS = (
    (0.95, ("", "draw")),
    (1.85, ("top", "8")),
    (2.75, ("next", "8")),
    (3.7, ("next", "16")),
    (4.85, ("below", "the line")),
    (6.15, ("", "Right")),
    (7.1, ("", "Found")),
    (8.0, ("", "F1")),
)
#: Where the table's two halves end: the band counts, then the line's three numbers.
DRAW_RULE_X = (0.5, 5.55, 8.45)
DRAW_HEAD_Y, DRAW_ROW0_Y, DRAW_ROW_DY = 4.62, 4.05, 0.48
#: The histograms' column, right of the table, in figure fractions.
DRAW_HIST_X0, DRAW_HIST_W = 0.70, 0.27


def draws_fig() -> None:
    """Every number from one set of joint draws: per-band counts, then the line's three ranges.

    Three stages: draws of each band's count of matches, as the app takes
    them at the end of the running example's test (one column per band above
    the line, and the bands below summed, the deepest one the model's point in
    every draw); the line's *Right*, *Found* and F-beta in each draw, computed
    from that draw's counts; and the 4,000 draws as distributions, each range
    their central 95%. The ranges are the test's own (`LineTest.estimates`).
    """
    _save(DRAWS_STAGES, _draws_stage, "test-draws")


def _draws_stage(stage: int) -> Figure:
    fig, ax = _canvas()
    test, _ = _run()
    counts = test._base_draws()
    above = [b.index for b in test.above]
    below = [b.index for b in test.below]
    k = float(test.line_count)
    tp = counts[above].sum(axis=0)
    total = counts.sum(axis=0)
    right = tp / k
    found = tp / total
    f1 = 2 * tp / (total + k)
    n = counts.shape[1]
    estimates = test.estimates()
    for got, want in ((right, estimates.precision), (found, estimates.recall), (f1, estimates.fbeta)):
        assert abs(float(np.mean(got)) - want.point) < 1e-9
    shown = list(range(DRAWS_SHOWN)) + [n - 1]
    columns = 5 if stage < 2 else len(DRAW_COLS)

    # ── stage 1: each band's count, draw by draw ─────────────────────────────
    ax.text(STATEMENT_X, 6.2, f"Draw every band's count at once, {n:,} times", ha="left", va="center", fontsize=19)
    for col, (x, (top, bottom)) in enumerate(DRAW_COLS[:columns]):
        colour = SOFT if col == 0 else INK
        weight = "normal" if col == 0 else "bold"
        ax.text(x, DRAW_HEAD_Y, bottom, ha="center", va="bottom", fontsize=16, fontweight=weight, color=colour)
        if top:
            ax.text(x, DRAW_HEAD_Y + 0.36, top, ha="center", va="bottom", fontsize=16, fontweight=weight)
    rule_end = DRAW_RULE_X[1] if stage < 2 else DRAW_RULE_X[2]
    ax.plot([DRAW_RULE_X[0], rule_end], [DRAW_HEAD_Y - 0.1] * 2, color=SOFT, linewidth=1.2)
    for row, j in enumerate(shown):
        y = DRAW_ROW0_Y - row * DRAW_ROW_DY - (0.45 if j == n - 1 else 0.0)
        cells = [f"{j + 1:,}"] + [f"{counts[b, j]:.0f}" for b in above] + [f"{counts[below, j].sum():.1f}"]
        if stage >= 2:
            cells += [_pct(right[j]), _pct(found[j]), f"{f1[j]:.2f}"]
        for col, value in enumerate(cells):
            ax.text(DRAW_COLS[col][0], y, value, ha="center", va="center", fontsize=17,
                    color=SOFT if col == 0 else INK)  # fmt: skip
    ax.text(DRAW_COLS[0][0], DRAW_ROW0_Y - DRAWS_SHOWN * DRAW_ROW_DY + 0.08, "⋮", ha="center", va="center",
            fontsize=18, color=SOFT)  # fmt: skip
    # ── stage 3: all the draws, as distributions ─────────────────────────────
    if stage >= 3:
        for i, (values, est, name, unit) in enumerate(
            (
                (right, estimates.precision, "Right", "pct"),
                (found, estimates.recall, "Found", "pct"),
                (f1, estimates.fbeta, "F1", "f"),
            )
        ):
            hist = fig.add_axes((DRAW_HIST_X0, 0.535 - i * 0.19, DRAW_HIST_W, 0.085))
            # Right moves in steps of one match in 32, so its bins are those steps.
            bins = (np.arange(test.line_count + 2) - 0.5) / k if values is right else np.linspace(0, 1, 41)
            heights, edges = np.histogram(values, bins=bins)
            mids = (edges[:-1] + edges[1:]) / 2
            colours = [INK if est.lo <= m <= est.hi else MODEL_GREY for m in mids]
            hist.bar(mids, heights, width=edges[1] - edges[0], color=colours)
            hist.set_xlim(0, 1)
            hist.set_ylim(0, heights.max() * 1.05)
            hist.set_yticks([])
            hist.spines["left"].set_visible(False)
            hist.set_xticks([0, 0.5, 1])
            hist.set_xticklabels(["0%", "50%", "100%"] if unit == "pct" else ["0", "0.5", "1"])
            words = _span(est.lo, est.hi) if unit == "pct" else f"{est.lo:.2f}–{est.hi:.2f}"
            hist.text(0.0, 1.1, f"{name}: likely {words}", ha="left", va="bottom", fontsize=16, fontweight="bold",
                      transform=hist.transAxes)  # fmt: skip
    return fig


# ── Good Neighbours: the prior above the line ────────────────────────────────

POOL_STAGES = 3
#: The big sparse line: a recall-end line on a corpus eight times the running
#: example's, with a thinner tail. Nine bands above the line.
SPARSE_N, SPARSE_K, SPARSE_BETA = 16_384, 2_048, 4.0
SPARSE_SCALE, SPARSE_FLOOR = 12.0, 0.004
SPARSE_SLOPE = 1.6
SPARSE_SEED = 6
#: The share axis's floor, log-scaled: a band with nothing planted is drawn on it.
POOL_FLOOR = 0.0007


@functools.lru_cache(maxsize=None)
def _sparse_run() -> tuple[LineTest, np.ndarray]:
    """The big sparse line after its first pass above the line: one round in every band, as the app takes it."""
    ranks = np.arange(SPARSE_N)
    chance = CHANCE_TOP / (1 + (ranks / SPARSE_SCALE) ** SPARSE_SLOPE) + SPARSE_FLOOR
    truth = np.random.default_rng(SPARSE_SEED).random(SPARSE_N) < chance
    test = LineTest.start(range(SPARSE_N), SPARSE_K, SPARSE_BETA, seed=SPARSE_SEED)
    while test.phase().phase == "matches":
        picks = test.draw()
        test.record({i: bool(truth[i]) for i in picks})
    assert len(test.above) == 9 and test.picks_on("above") == 45, (len(test.above), test.picks_on("above"))
    return test, truth


def _jeffreys_alone(test: LineTest, rng: np.random.Generator, n: int) -> np.ndarray:
    """``(bands above, n)`` draws of each band's matches with its own Jeffreys prior: #4523's estimator."""
    out = []
    for band in test.above:
        size, labelled, right = test.band_counts(band.index)
        share = rng.beta(JEFFREYS + right, JEFFREYS + labelled - right, size=n)
        out.append(right + rng.binomial(size - labelled, share))
    return np.array(out, dtype=float)


def pool_fig() -> None:
    """Why a band above the line borrows its prior from its neighbours (#4523, #4539, #4560).

    Three stages on a big sparse line (2,048 kept, nine bands above it, one
    round each): the planted share right in each band and what its five picks
    found; each band read on its own Jeffreys prior, under which every band
    that found nothing reads about 8% right, whatever its size, so the
    deep bands invent matches by the dozen and the line's range sits above the
    truth; and each band read on a prior pooled from the picks in the bands
    beside it, at five picks' weight, as the app does: the empty deep bands
    borrow their emptiness, and the range holds.
    """
    _save(POOL_STAGES, _pool_stage, "test-pool")


def _pool_stage(stage: int) -> Figure:
    fig = plt.figure(figsize=(W, H))
    test, truth = _sparse_run()
    above = test.above
    order = list(reversed(above))  # left to right: the line's band first, the top last
    xs = np.arange(len(order))
    ax = fig.add_axes((CHART_X0, 0.15, 0.60, 0.72))
    ax.set_yscale("log")
    ax.set_ylim(POOL_FLOOR, 1.3)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_yticks([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    ax.set_yticklabels(["0.1%", "0.3%", "1%", "3%", "10%", "30%", "100%"])
    ax.yaxis.grid(True, color=GRID_COLOUR, linewidth=GRID_LW)
    ax.set_axisbelow(True)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b.size:,}" for b in order])
    ax.set_xlabel("each band above the line, its size; the top on the right")
    ax.set_ylabel("share right")
    ax.minorticks_off()

    planted = np.array([truth[b.lo : b.hi].mean() for b in order])
    picks = [test.band_counts(b.index) for b in order]
    k = float(test.line_count)
    true_tp = float(truth[: test.line_count].sum())
    lines = [
        ("Matches above the line", "head", 0.0),
        (f"planted: {true_tp:.0f}, {_fine(true_tp / k)}% right", "body", 0.055),
    ]
    # ── stage 1: the planted shares, and what each band's round found ────────
    # A band with nothing planted in it sits on the axis's floor, and says so.
    ax.plot(xs, np.maximum(planted, POOL_FLOOR), color=INK, linewidth=0, marker="o", markersize=11,
            markerfacecolor="white", markeredgewidth=2.4, zorder=5, clip_on=False)  # fmt: skip
    for x in xs[planted == 0]:
        ax.text(x + 0.14, POOL_FLOOR * 1.15, "none", ha="left", va="bottom", fontsize=15)
    for x, (_, labelled, right) in zip(xs, picks, strict=True):
        ax.text(x, 1.6, f"{right} of {labelled}", ha="center", va="bottom", fontsize=16)
    ax.text(-0.75, 1.6, "picks:", ha="right", va="bottom", fontsize=16, color=SOFT)
    ax.text(5.15, planted[5], "planted", ha="left", va="center", fontsize=16)

    # ── stage 2: every band on its own Jeffreys prior ────────────────────────
    if stage >= 2:
        alone = _jeffreys_alone(test, np.random.default_rng(1), test.budgets.draws)
        alone_share = np.array([alone[b.index].mean() / b.size for b in order])
        ax.plot(xs, alone_share, color=RED, linewidth=2.6, marker="s", markersize=10, zorder=4)
        lo, hi = np.quantile(alone.sum(axis=0) / k, [TEST_ALPHA / 2, 1 - TEST_ALPHA / 2])
        assert lo > true_tp / k, (lo, true_tp / k)
        ax.text(1.0, alone_share[1] * 1.3, "each band on its own", ha="left", va="bottom", fontsize=16, color=RED)
        lines += [
            (f"on its own: {alone.sum(axis=0).mean():.0f}", RED, 0.095),
            (f"likely {_span(lo, hi, fine=True)} right", "body", 0.05),
            ("misses the truth", "body", 0.045),
        ]
    # ── stage 3: the neighbours' prior, as the app reads it ──────────────────
    if stage >= 3:
        counts = test._base_draws()
        pooled_share = np.array([counts[b.index].mean() / b.size for b in order])
        ax.plot(xs, pooled_share, color=GREEN, linewidth=2.6, marker="D", markersize=9, zorder=4)
        est = test.estimates()
        assert est.precision.holds(true_tp / k), (est.precision, true_tp / k)
        ax.text(1.55, pooled_share[2] / 1.35, "with its neighbours", ha="left", va="top", fontsize=16, color=GREEN)
        lines += [
            (f"with neighbours: {est.positives_above.point:.0f}", GREEN, 0.095),
            (f"likely {_span(est.precision.lo, est.precision.hi, fine=True)} right", "body", 0.05),
            ("holds the truth", "body", 0.045),
        ]
    _side_block(fig, lines)
    return fig


# ── Second Opinion: model-assisted below the line ────────────────────────────

MODEL_STAGES = 4


def model_fig() -> None:
    """Below the line, the model's own count, corrected band by band by the picks.

    Four stages on the running example's bands below the line: the planted
    matches in each band and what its round found; every band read on
    Jeffreys alone, where five picks that find nothing in a band of 512 read
    as 42 matches; the model's own count for each band, the sum of its
    chances; and that count corrected by the band's picks, as the app takes
    it (the model's share as the prior, at one round's weight), with the band
    the walk never reached taken from the model alone.
    """
    _save(MODEL_STAGES, _model_stage, "test-model")


def _model_stage(stage: int) -> Figure:
    fig = plt.figure(figsize=(W, H))
    test, _ = _run()
    truth, _ = _corpus()
    below = test.below
    order = list(reversed(below))  # left to right: deepest first, the line's neighbour last
    xs = np.arange(len(order))
    ax = fig.add_axes((CHART_X0, 0.15, 0.60, 0.72))
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, 46)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.yaxis.grid(True, color=GRID_COLOUR, linewidth=GRID_LW)
    ax.set_axisbelow(True)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b.size:,}" for b in order])
    ax.set_xlabel("each band below the line, its size; the line on the right")
    ax.set_ylabel("matches in the band")
    width = 0.2
    #: Each band's four bars, left to right: planted, the picks alone, the model, the model corrected.
    slot = {name: (i - 1.5) * width for i, name in enumerate(("planted", "alone", "model", "corrected"))}
    planted = np.array([truth[b.lo : b.hi].sum() for b in order], dtype=float)
    picks = [test.band_counts(b.index) for b in order]
    reached = np.array([labelled > 0 for _, labelled, _ in picks])
    true_below = float(planted.sum())
    hits = float(truth[: test.line_count].sum())
    est = test.estimates()
    lines = [
        ("Matches below the line", "head", 0.0),
        (f"planted: {true_below:.0f}", "body", 0.055),
        (f"Found {_pct(hits / (hits + true_below))}", "body", 0.045),
    ]
    # ── stage 1: what is planted, and what the walk's picks found ────────────
    ax.bar(xs + slot["planted"], planted, width=width, color="white", edgecolor=INK, linewidth=2.0, zorder=3)
    for x, (_size, labelled, right) in zip(xs, picks, strict=True):
        words = f"{right} of {labelled}" if labelled else "no picks"
        ax.text(x, 47.5, words, ha="center", va="bottom", fontsize=15, color=INK if labelled else SOFT)
    ax.text(-0.75, 47.5, "picks:", ha="right", va="bottom", fontsize=15, color=SOFT)
    # ── stage 2: Jeffreys alone, on the bands the walk reached ───────────────
    alone = np.array(
        [right + (size - labelled) * (right + JEFFREYS) / (labelled + 2 * JEFFREYS) if labelled else 0.0
         for size, labelled, right in picks]
    )  # fmt: skip
    if stage >= 2:
        ax.bar(xs[reached] + slot["alone"], alone[reached], width=width, color=RED, zorder=3)
        deep = int(np.argmax(alone))
        ax.text(xs[deep] + slot["alone"], alone[deep] + 0.6, f"{alone[deep]:.0f}", ha="center", va="bottom",
                fontsize=16, color=RED)  # fmt: skip
        kept = est.positives_above.point
        lines += [
            (f"picks alone: {alone.sum():.0f}", RED, 0.095),
            (f"Found {_pct(kept / (kept + alone.sum()))}", "body", 0.05),
        ]
    # ── stage 3: the model's own count ───────────────────────────────────────
    mass = np.array([test.band_mass(b.index) for b in order])
    if stage >= 3:
        ax.bar(xs + slot["model"], mass, width=width, color=MODEL_GREY, zorder=3)
        lines += [(f"the model's count: {mass.sum():.0f}", SOFT, 0.095)]
    # ── stage 4: the model, corrected by the picks ───────────────────────────
    if stage >= 4:
        counts = test._base_draws()
        mean = np.array([counts[b.index].mean() for b in order])
        ax.bar(xs[reached] + slot["corrected"], mean[reached], width=width, color=GREEN, zorder=4)
        tail = ~reached
        ax.bar(xs[tail] + slot["corrected"], mean[tail], width=width, color="white", edgecolor=GREEN, linewidth=2.0,
               hatch="//", zorder=4)  # fmt: skip
        ax.text(xs[tail][0], mean[tail][0] + 1.2, "the model\nalone", ha="center", va="bottom", fontsize=15,
                color=GREEN)  # fmt: skip
        assert est.tail_from_model and est.recall.holds(hits / (hits + true_below)), est.recall
        lines += [
            (f"corrected: {est.positives_below.point:.0f}", GREEN, 0.095),
            (f"Found likely {_span(est.recall.lo, est.recall.hi)}", "body", 0.05),
        ]
    _side_block(fig, lines)
    return fig


# ── Pencils Down: the rounds, the phases and the stop ────────────────────────

ROUNDS_STAGES = 5
ROUNDS_Y0, ROUNDS_H = 2.8, 0.62
ROUND_R = 0.22


def rounds_fig() -> None:
    """The running example's test, round by round, as the Test autopilot deals them.

    Five stages: the bands on both sides of the line; the matches phase's first
    pass, one round in each band from the line upward; the round the
    allocation rule placed, in the band whose next round would narrow the
    F-beta range most in expectation; the misses phase, one band deeper a
    round, to its 40 picks; and Done, with the band the walk never reached
    taken from the model.
    """
    _save(ROUNDS_STAGES, _rounds_stage, "test-rounds")


def _round_disc(ax: plt.Axes, x: float, y: float, number: int, right: int, picks: int) -> None:
    ax.add_patch(Circle((x, y), ROUND_R, facecolor=INK, edgecolor="none", zorder=6))
    ax.text(x, y, f"{number}", ha="center", va="center", fontsize=16, color="white", fontweight="bold", zorder=7)
    ax.text(x, y + ROUND_R + 0.06, f"{right} of {picks}", ha="center", va="bottom", fontsize=15, zorder=7)


def _shrinks() -> dict[int, float]:
    """How much narrower each band above the line would make the F-beta range, after the first pass."""
    state = _after(3)
    return {band.index: state.expected_shrink(band.index) for band in state.above}


def _rounds_stage(stage: int) -> Figure:
    fig, ax = _canvas()
    test, history = _run()
    n_above = len(test.above)
    line_x = _line_x(test)
    # ── stage 1: the bands, both sides of the line ───────────────────────────
    ax.text(STATEMENT_X, 6.2, "The Test autopilot deals five picks at a time", ha="left", va="center", fontsize=19)
    _band_strip(ax, test, ROUNDS_Y0, ROUNDS_H)
    ax.text(BAND_X0, ROUNDS_Y0 - 0.12, "bands, best on the right; each twice the next", ha="left", va="top",
            fontsize=15, color=SOFT)  # fmt: skip
    _line(ax, line_x, ROUNDS_Y0 - 0.3, ROUNDS_Y0 + ROUNDS_H + 0.3, label=False)
    seen: dict[int, int] = {}

    def disc(number: int) -> None:
        band, picks = history[number - 1]
        x, w = _band_cell(test, band)
        level = seen.get(band, 0)
        seen[band] = level + 1
        _round_disc(ax, x + w / 2, ROUNDS_Y0 + ROUNDS_H + 0.45 + level * 0.95, number, _right(picks), len(picks))

    # ── stage 2: the first pass, from the line upward ────────────────────────
    if stage >= 2:
        for number in range(1, n_above + 1):
            disc(number)
    # ── stage 3: the round the allocation rule placed ────────────────────────
    if stage >= 3:
        shrinks = _shrinks()
        chosen = history[n_above][0]
        assert chosen == max(shrinks, key=shrinks.get), shrinks
        disc(n_above + 1)
        x, w = _band_cell(test, chosen)
        ax.text(x - 0.12, ROUNDS_Y0 + ROUNDS_H + 1.58, "where a round narrows", ha="right", va="center",
                fontsize=15)  # fmt: skip
        ax.text(x - 0.12, ROUNDS_Y0 + ROUNDS_H + 1.22, "the F1 range most", ha="right",
                va="center", fontsize=15)  # fmt: skip
    # ── stage 4: the walk below the line ─────────────────────────────────────
    if stage >= 4:
        for number in range(n_above + 2, len(history) + 1):
            disc(number)
    # ── stage 5: Done ────────────────────────────────────────────────────────
    _rounds_phases(ax, test, stage)
    if stage >= 5:
        x, w = _band_cell(test, len(test.bands) - 1)
        for row, words in enumerate(("never", "reached")):
            ax.text(x + w / 2, ROUNDS_Y0 + ROUNDS_H + 0.83 - row * 0.38, words, ha="center", va="center",
                    fontsize=15, color=SOFT)  # fmt: skip
        ax.text(W / 2, 0.75, "Done!", ha="center", va="center", fontsize=22, fontweight="bold")
    return fig


def _rounds_phases(ax: plt.Axes, test: LineTest, stage: int) -> None:
    """The two phases under the strip, each bracketing its side, and at Done why each ended."""
    brace_y = ROUNDS_Y0 - 0.75
    report = test.phase()
    line_x = _line_x(test)
    deepest = _band_cell(test, len(test.bands) - 1)
    phases = (
        (2, line_x + 0.06, BAND_X1 - 0.06, "Check the matches", report.picks_above),
        (4, deepest[0] + deepest[1] + 0.06, line_x - 0.06, "Check the misses", report.picks_below),
    )
    for from_stage, x0, x1, name, picks in phases:
        if stage < from_stage:
            continue
        _bracket(ax, x0, x1, brace_y, down=True)
        ax.text((x0 + x1) / 2, brace_y - 0.12, name, ha="center", va="top", fontsize=17, fontweight="bold")
        if stage >= 5:
            ax.text((x0 + x1) / 2, brace_y - 0.55, f"{picks} picks: its budget", ha="center", va="top", fontsize=16)


# ── Report Card: what the result pane says ───────────────────────────────────

RESULT_STAGES = 3


def result_fig() -> None:
    """The result pane at Done: the headline ranges, Precision by Number Returned, and Lean the Threshold.

    Three stages: the balance, Right and Found, with the chart of how right
    the top *N* likely are at every band edge (the count axis logarithmic, as
    the app draws it) and the line marked; the planted truth over it, which
    only a planted corpus can show, inside the ranges; and Lean the
    Threshold, the same draws read at the line each radio would draw on this
    corpus.
    """
    _save(RESULT_STAGES, _result_stage, "test-result")


def _result_stage(stage: int) -> Figure:
    fig = plt.figure(figsize=(W, H))
    test, _ = _run()
    est = test.estimates()
    edges = est.at_edges
    counts = np.array([e.count for e in edges], dtype=float)
    ax = fig.add_axes((0.40, 0.14, 0.57, 0.70))
    ax.set_xscale("log", base=2)
    ax.set_xlim(6, 2600)
    ax.set_ylim(0, 1.04)
    ax.set_xticks([8, 32, 128, 512, 2048])
    ax.set_xticklabels(["8", "32", "128", "512", "2,048"])
    ax.minorticks_off()
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=GRID_COLOUR, linewidth=GRID_LW)
    ax.set_axisbelow(True)
    ax.set_xlabel("items returned (log scale)")
    fig.text(0.685, 0.87, "Precision by Number Returned", ha="center", va="bottom", fontsize=18, fontweight="bold")
    lo = np.array([e.precision.lo for e in edges])
    hi = np.array([e.precision.hi for e in edges])
    point = np.array([e.precision.point for e in edges])
    # ── stage 1: the headline, and the ranges at every band edge ─────────────
    ax.fill_between(counts, lo, hi, color=NEUTRAL_FILL, zorder=1, linewidth=0)
    ax.plot(counts, point, color=INK, linewidth=2.4, marker="o", markersize=7, zorder=4)
    ax.axvline(test.line_count, color=BLUE, linewidth=2.6, zorder=3)
    ax.text(test.line_count * 1.08, 0.04, "the line", ha="left", va="bottom", fontsize=16)
    _side_block(
        fig,
        [
            ("Balance (Fβ)", "head", 0.0),
            (f"{est.fbeta.point:.2f}, likely {est.fbeta.lo:.2f}–{est.fbeta.hi:.2f}", "body", 0.045),
            ("Right", "head", 0.07),
            (f"likely {_span(est.precision.lo, est.precision.hi)}", "body", 0.045),
            ("Found", "head", 0.07),
            (f"{est.found},", "body", 0.045),
            (_span(est.recall.lo, est.recall.hi), "body", 0.045),
        ],
    )
    # ── stage 2: the planted truth ───────────────────────────────────────────
    if stage >= 2:
        k = np.arange(6, CORPUS_N + 1)
        truth, _ = _corpus()
        true_p = np.cumsum(truth)[k - 1] / k
        ax.plot(k, true_p, color=INK, linewidth=1.8, linestyle=(0, (2, 2)), zorder=3)
        held = sum(e.precision.holds(_truth_at(e.count)[0]) for e in edges)
        assert held == len(edges), held
        ax.text(1100, true_p[1100 - 6] + 0.06, "planted", ha="center", va="bottom", fontsize=16)
    # ── stage 3: Lean the Threshold, as the pane's table lays it out ─────────
    if stage >= 3:
        top = 0.305
        fig.text(SIDE_X, top, "Lean the Threshold", ha="left", va="center", fontsize=17, fontweight="bold")
        fig.text(0.155, top - 0.05, "keeps", ha="right", va="center", fontsize=15, color=SOFT)
        fig.text(0.175, top - 0.05, "right", ha="left", va="center", fontsize=15, color=SOFT)
        for i, beta in enumerate(PRESETS):
            count = _preset_count(beta)
            lean = test.estimate_at(count)
            name = {4.0: "4", 1.0: "1", 0.25: "¼"}[beta]
            y = top - 0.1 - i * 0.05
            fig.text(SIDE_X + 0.015, y, f"β {name}", ha="left", va="center", fontsize=16)
            fig.text(0.155, y, f"{count}", ha="right", va="center", fontsize=16)
            fig.text(0.175, y, _span(lean.precision.lo, lean.precision.hi), ha="left", va="center", fontsize=16)
            if count != test.line_count:
                ax.plot([count], [lean.precision.point], marker="D", markersize=10, color=INK, zorder=6)
                ax.text(count, lean.precision.point - 0.06, f"β {name}", ha="center", va="top", fontsize=15, zorder=6)
    return fig


# ── Grading the Grader: does the range hold? ─────────────────────────────────

COVERAGE_STAGES = 3
#: The rows, top to bottom: (world, its label, beta, the radio's label).
COVERAGE_WORLDS = (("0.1%", "0.1% matches"), ("0.44%", "0.44% matches"), ("5%", "5% matches"))
COVERAGE_BETAS = ((0.25, "β ¼"), (1.0, "β 1"), (4.0, "β 4"))


def coverage_fig() -> None:
    """How often the shipped Test's ranges held the truth, and what a Test cost (#4523, #4540, #4560).

    Three stages over the nine (world, radio) cells of the #4523 replay at the
    shipped defaults: the share of Tests whose Right range held the truth,
    against the 95% the range claims; the same for Found; and the picks a
    Test took to reach Done.
    """
    rows = _coverage_rows()
    _save(COVERAGE_STAGES, lambda stage: _coverage_stage(stage, rows), "test-coverage")


def _coverage_rows() -> dict[tuple[str, float], dict[str, float]]:
    if not SUMMARY.exists():
        raise SystemExit(f"{SUMMARY.relative_to(_REPO_ROOT)} is missing: it is the #4560 study's shipped summary")
    out = {}
    with SUMMARY.open() as handle:
        for row in csv.DictReader(handle):
            out[(row["world"], float(row["beta"]))] = {
                key: float(row[key]) for key in ("precision_cov", "recall_cov", "picks_mean", "picks_p90")
            }
    return out


def _coverage_cells() -> list[tuple[str, float, float, str]]:
    """``(world, beta, y, the radio's label)`` per row, top to bottom, a gap between worlds."""
    cells, y = [], 0.0
    for world, _label in COVERAGE_WORLDS:
        for beta, beta_label in COVERAGE_BETAS:
            cells.append((world, beta, y, beta_label))
            y -= 1.0
        y -= 0.6
    return cells


def _coverage_stage(stage: int, rows: dict) -> Figure:
    fig = plt.figure(figsize=(W, H))
    held = fig.add_axes((0.40, 0.14, 0.34, 0.71))
    picks = fig.add_axes((0.79, 0.14, 0.18, 0.71))
    cells = _coverage_cells()
    ys = [y for _w, _b, y, _l in cells]
    for axis in (held, picks):
        axis.set_ylim(min(ys) - 0.7, 0.7)
        axis.set_yticks([])
        axis.spines["left"].set_visible(False)
    # Row labels: the radio beside each row, the world over each group.
    trans = held.get_yaxis_transform()
    for world, world_label in COVERAGE_WORLDS:
        top = max(y for w, _b, y, _l in cells if w == world)
        held.text(-0.30, top + 0.55, world_label, transform=trans, ha="left", va="bottom", fontsize=16,
                  fontweight="bold")  # fmt: skip
    for _w, _b, y, label in cells:
        held.text(-0.03, y, label, transform=trans, ha="right", va="center", fontsize=16)
    held.set_xlim(0.7, 1.005)
    held.set_xticks([0.7, 0.8, 0.9, 0.95, 1.0])
    held.set_xticklabels(["70%", "80%", "90%", "95%", "100%"])
    held.set_xlabel("Tests whose range held the truth")
    held.axvline(0.95, color=SOFT, linewidth=1.6, linestyle=(0, (4, 3)), zorder=1)
    held.text(0.95, 0.75, "claimed", ha="center", va="bottom", fontsize=15, color=SOFT)
    # ── stage 1: Right; stage 2: Found ───────────────────────────────────────
    # The key stands under the notch, where the chart leaves room: each mark
    # with the range it scores, then the cost's two marks with the cost.
    marks = (
        ("precision_cov", "● Right (precision)", "o", INK, 1),
        ("recall_cov", "■ Found (recall)", "s", SOFT, 2),
    )
    fig.text(SIDE_X, SIDE_TOP, "Range held the truth", ha="left", va="center", fontsize=17, fontweight="bold")
    for row, (key, words, marker, colour, from_stage) in enumerate(marks):
        if stage < from_stage:
            continue
        xs = [rows[(w, b)][key] for w, b, _y, _l in cells]
        held.plot(xs, ys, linewidth=0, marker=marker, markersize=12, color=colour, zorder=4)
        fig.text(SIDE_X + 0.015, SIDE_TOP - 0.055 - row * 0.05, words, ha="left", va="center", fontsize=16,
                 color=colour)  # fmt: skip
    # ── stage 3: what a Test cost ────────────────────────────────────────────
    picks.set_xlim(0, 85)
    picks.set_xticks([0, 40, 80])
    picks.set_xlabel("picks to Done")
    picks.set_visible(stage >= 3)
    if stage >= 3:
        for w, b, y, _l in cells:
            mean, p90 = rows[(w, b)]["picks_mean"], rows[(w, b)]["picks_p90"]
            picks.barh(y, mean, height=0.6, color=MODEL_GREY)
            picks.plot([p90, p90], [y - 0.3, y + 0.3], color=INK, linewidth=2.2)
        top = SIDE_TOP - 0.22
        fig.text(SIDE_X, top, "Picks to Done", ha="left", va="center", fontsize=17, fontweight="bold")
        fig.text(SIDE_X + 0.015, top - 0.055, "bar: the mean", ha="left", va="center", fontsize=16)
        fig.text(SIDE_X + 0.015, top - 0.105, "tick: 9 Tests in 10 done", ha="left", va="center", fontsize=16)
    return fig


if __name__ == "__main__":
    question_fig()
    bands_fig()
    posterior_fig()
    draws_fig()
    pool_fig()
    model_fig()
    rounds_fig()
    result_fig()
    coverage_fig()
    print("wrote figures to", OUT)
