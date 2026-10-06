"""State of the App: Document Logo, the analysis (#4392).

Reads a ``sota_documents.py`` run (``steps.csv``, ``clicks.csv``) and writes,
beside it:

* ``summary.md``: the headline at click 0 / 10 / 25 / final (AP, Goods found,
  the gate's precision / recall / F1, the best cut's F1, the floor-style
  oracle cuts at P), then the per-class table at the final click;
* ``figures/``: ``ap_found.png``, ``f1_over_clicks.png``,
  ``line_at_floors.png``, ``per_class.png`` and, when the run scored every
  balance (#4457), ``returned_at_beta.png`` and ``retrain.png``;
* ``images/``: thumbnails of the most helpful and most harmful clicks (credit =
  that click's change in AP, one observation each), the class's box drawn on
  positives, so a reader can check them against the labels;
* ``misses/``: for the weakest classes, the test positives the final ranking left
  beyond the verified shortlist (from ``positives_final.csv``).

    python sota_documents_analyze.py --run <run dir>/round1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DPI = 130
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"
FLOORS = (10, 50, 90)
#: The balance's presets (#4413) and their column suffixes in ``steps.csv``.
BETAS = (("1/4", "025"), ("1", "1"), ("4", "4"))


def _num(x: Any) -> float:
    return float("nan") if x in ("", "nan", None) else float(x)


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [
            {k: (v if k in ("class_id", "source", "page_id", "label") else _num(v)) for k, v in r.items()}
            for r in csv.DictReader(fh)
        ]


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _at(steps: list[dict[str, Any]], v: Optional[int]) -> list[dict[str, Any]]:
    """Each class's row at click *v*, or its last click when *v* is None or past its end."""
    by: dict[str, list[dict[str, Any]]] = {}
    for r in steps:
        by.setdefault(r["class_id"], []).append(r)
    out = []
    for rows in by.values():
        rows.sort(key=lambda r: r["v"])
        hit = [r for r in rows if v is not None and int(r["v"]) == v]
        out.append(hit[0] if hit else rows[-1])
    return out


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    xs = [r[key] for r in rows if not np.isnan(r[key])]
    return float(np.mean(xs)) if xs else float("nan")


def summary(steps: list[dict[str, Any]]) -> str:
    points = [("click 0 (example sort)", 0), ("10 clicks", 10), ("25 clicks", 25), ("final", None)]
    out = [f"{len({r['class_id'] for r in steps})} classes.", "", "### Headline (mean over classes)", ""]
    head = "| | AP | Goods found | gate precision | gate recall | gate F1 | best-cut F1 |"
    out += [head, "|---|---:|---:|---:|---:|---:|---:|"]
    for label, v in points:
        rows = _at(steps, v)
        out.append(
            f"| {label} | {_mean(rows, 'ap'):.2f} | {_mean(rows, 'found'):.1f} | {_mean(rows, 'gate_precision'):.2f} | "
            f"{_mean(rows, 'gate_recall'):.2f} | {_mean(rows, 'gate_f1'):.2f} | {_mean(rows, 'best_f1'):.2f} |"
        )
    out += [
        "",
        "### Floor-style cuts at P (oracle: the deepest cut of the ranking with precision >= P)",
        "",
        "| | " + " | ".join(f"recall at P={p}%" for p in FLOORS) + " | gate recall |",
        "|---|" + "---:|" * (len(FLOORS) + 1),
    ]
    for label, v in points:
        rows = _at(steps, v)
        cells = [f"{_mean(rows, f'recall_at_p{p}'):.2f}" for p in FLOORS]
        out.append(f"| {label} | " + " | ".join(cells) + f" | {_mean(rows, 'gate_recall'):.2f} |")
    out += [
        "",
        "### Per class, final click",
        "",
        "| class | positives | pool | click-0 AP | final AP | found | gate P / R / F1 | recall at P=50% | retrain median |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    finals = {r["class_id"]: r for r in _at(steps, None)}
    zero = {r["class_id"]: r for r in _at(steps, 0)}
    for cid in sorted(finals, key=lambda c: finals[c]["ap"]):
        r = finals[cid]
        rt = np.median([s["retrain_s"] for s in steps if s["class_id"] == cid])
        out.append(
            f"| `{cid}` | {int(r['n_positive'])} | {int(r['n_pool'])} | {zero[cid]['ap']:.2f} | {r['ap']:.2f} | "
            f"{int(r['found'])} | {r['gate_precision']:.2f} / {r['gate_recall']:.2f} / {r['gate_f1']:.2f} | "
            f"{r['recall_at_p50']:.2f} | {rt:.1f} s |"
        )
    return "\n".join(out) + "\n"


def _share(r: dict[str, Any], tag: str) -> float:
    """The returned set's F-beta as a share of the best cut's, for one step (nan without positives)."""
    best = r.get(f"best_fb{tag}", float("nan"))
    return r[f"gate_fb{tag}"] / best if best and not np.isnan(best) and best > 0 else float("nan")


def balance(steps: list[dict[str, Any]]) -> str:
    """The returned set at each balance: F-beta as a share of the best cut's, with P and R beside it.

    The structural line ignores beta, so every beta reads the same sessions and the same set.
    """
    if not steps or "gate_fb025" not in steps[0]:
        return ""
    points = [("click 0 (example sort)", 0), ("10 clicks", 10), ("25 clicks", 25), ("final", None)]
    out = [
        "### The returned set at each balance (F-beta as a share of the best cut's; one set serves every beta)",
        "",
        "| | " + " | ".join(f"beta {b}: F-beta / best = share" for b, _ in BETAS) + " | precision | recall |",
        "|---|" + "---|" * len(BETAS) + "---:|---:|",
    ]
    for label, v in points:
        rows = _at(steps, v)
        cells = []
        for _b, tag in BETAS:
            shares = [x for r in rows if not np.isnan(x := _share(r, tag))]
            cells.append(
                f"{_mean(rows, f'gate_fb{tag}'):.2f} / {_mean(rows, f'best_fb{tag}'):.2f} = "
                f"{float(np.mean(shares)) if shares else float('nan'):.2f}"
            )
        out.append(
            f"| {label} | " + " | ".join(cells) + f" | {_mean(rows, 'gate_precision'):.2f} | "
            f"{_mean(rows, 'gate_recall'):.2f} |"
        )
    times = np.array([r["retrain_s"] for r in steps if not np.isnan(r["retrain_s"])])
    later = np.array([r["retrain_s"] for r in steps if r["v"] > 0 and not np.isnan(r["retrain_s"])])
    out += [
        "",
        f"Retrain wall clock: median {np.median(times):.1f} s, p90 {np.percentile(times, 90):.1f} s, "
        f"{(times > 5).mean():.0%} of steps over 5 s (after click 0: p90 {np.percentile(later, 90):.1f} s).",
        "",
    ]
    return "\n".join(out)


def _curve(ax, steps: list[dict[str, Any]], key: str, color: str, label: str) -> None:
    by: dict[str, list[dict[str, Any]]] = {}
    for r in steps:
        by.setdefault(r["class_id"], []).append(r)
    vmax = int(max(r["v"] for r in steps))
    for rows in by.values():
        rows.sort(key=lambda r: r["v"])
        ax.plot([r["v"] for r in rows], [r[key] for r in rows], color=color, alpha=0.18, linewidth=1)
    xs = list(range(vmax + 1))
    ys = [_mean([r for r in steps if int(r["v"]) == v], key) for v in xs]
    ax.plot(xs, ys, color=color, linewidth=2.4, label=label)


def figures(steps: list[dict[str, Any]], out: Path) -> None:
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), dpi=DPI)
    for ax, (key, ylabel) in zip(axes, (("ap", "AP on the held-out test half"), ("found", "Goods found"))):
        _style(ax)
        _curve(ax, steps, key, BLUE, "mean (thin: each class)")
        ax.set_xlabel("clicks", color=INK, fontsize=9)
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
        ax.set_title(ylabel, loc="left", color=INK, fontsize=10)
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "ap_found.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=DPI)
    _style(ax)
    _curve(ax, steps, "gate_f1", ORANGE, "the returned set (the inlier gate)")
    _curve(ax, steps, "best_f1", GREEN, "the best cut of the same ranking")
    ax.set_ylim(0, 1)
    ax.set_xlabel("clicks", color=INK, fontsize=9)
    ax.set_ylabel("F1 on the held-out test half", color=INK, fontsize=9)
    ax.set_title("F1 over clicks", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "f1_over_clicks.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=DPI)
    _style(ax)
    for p, color in zip(FLOORS, (PINK, GREEN, BLUE)):
        _curve(ax, steps, f"recall_at_p{p}", color, f"recall at P = {p}% (oracle cut)")
    _curve(ax, steps, "gate_recall", ORANGE, "the gate's recall")
    ax.set_ylim(0, 1)
    ax.set_xlabel("clicks", color=INK, fontsize=9)
    ax.set_ylabel("recall on the held-out test half", color=INK, fontsize=9)
    ax.set_title("The line at each floor P, and the gate", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "line_at_floors.png")
    plt.close(fig)

    if "gate_fb025" in steps[0]:
        for r in steps:
            for _b, tag in BETAS:
                r[f"share_{tag}"] = _share(r, tag)
        fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=DPI)
        _style(ax)
        vmax = int(max(r["v"] for r in steps))
        for (b, tag), color in zip(BETAS, (BLUE, ORANGE, GREEN)):
            xs = list(range(vmax + 1))
            ax.plot(
                xs,
                [_mean([r for r in steps if int(r["v"]) == v], f"share_{tag}") for v in xs],
                color=color,
                linewidth=2.4,
                label=f"beta {b}",
            )
        ax.set_ylim(0, 1)
        ax.set_xlabel("clicks", color=INK, fontsize=9)
        ax.set_ylabel("returned set's F-beta / best cut's", color=INK, fontsize=9)
        ax.set_title("The returned set at each balance (one line for every beta)", loc="left", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
        fig.tight_layout()
        fig.savefig(out / "returned_at_beta.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=DPI)
        _style(ax)
        xs = list(range(vmax + 1))
        for q, color in ((50, BLUE), (90, ORANGE)):
            ax.plot(
                xs,
                [float(np.percentile([r["retrain_s"] for r in steps if int(r["v"]) == v], q)) for v in xs],
                color=color,
                linewidth=2.2,
                label=f"p{q} over classes",
            )
        ax.axhline(5, color=MUTED, linestyle="--", linewidth=1, label="5 s budget")
        ax.set_xlabel("clicks", color=INK, fontsize=9)
        ax.set_ylabel("retrain wall clock (s)", color=INK, fontsize=9)
        ax.set_title("Retrain time per click", loc="left", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
        fig.tight_layout()
        fig.savefig(out / "retrain.png")
        plt.close(fig)

    finals = sorted(_at(steps, None), key=lambda r: r["ap"])
    zero = {r["class_id"]: r for r in _at(steps, 0)}
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(finals) + 1.2), dpi=DPI)
    _style(ax)
    ys = np.arange(len(finals))
    ax.barh(ys, [r["ap"] for r in finals], color=BLUE, height=0.6, label="final AP")
    ax.scatter([zero[r["class_id"]]["ap"] for r in finals], ys, color=INK, zorder=3, s=18, label="click-0 AP")
    ax.set_yticks(ys, [r["class_id"] for r in finals], fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AP on the held-out test half", color=INK, fontsize=9)
    ax.set_title("Per class: click 0 and final", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "per_class.png")
    plt.close(fig)


def thumbnails(clicks: list[dict[str, Any]], out: Path, n: int = 6) -> list[dict[str, Any]]:
    """The *n* most helpful and most harmful clicks, rendered with the class box on positives."""
    import embed_corpus  # noqa: PLC0415
    import fullmarks_config as cfg  # noqa: PLC0415
    import template_matrix as tm  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    scored = [c for c in clicks if not np.isnan(c["credit"])]
    scored.sort(key=lambda c: c["credit"])
    picks = [("harmful", c) for c in scored[:n]] + [("helpful", c) for c in reversed(scored[-n:])]
    pages = {p.page_id: p for p in embed_corpus.pages_for_tier(cfg.OUT, "m")}
    out.mkdir(exist_ok=True)
    rows = []
    for kind, c in picks:
        page = pages.get(c["page_id"])
        if page is None:
            continue
        with Image.open(page.path) as im:
            img = im.convert("RGB")
        box = tm.largest_box(page, c["class_id"])
        if box is not None:
            w, h = img.size
            ImageDraw.Draw(img).rectangle(
                [box[0] * w, box[1] * h, box[2] * w, box[3] * h], outline=(235, 104, 52), width=max(3, w // 200)
            )
        img.thumbnail((360, 360))
        name = f"{kind}-{c['class_id'].replace('/', '__')}-{c['page_id'].replace('/', '__').replace('#', '_')}.png"
        img.save(out / name)
        rows.append({**c, "kind": kind, "file": name})
    return rows


def misses(run: Path, steps: list[dict[str, Any]], weakest: int = 4) -> str:
    """Where each class's test positives sit at the final click, and thumbnails of the weakest classes' misses.

    A positive the final ranking places beyond the verified shortlist was never
    checked by Stage 2 (a Stage-1 miss); one inside it with a gate score below
    0.5 failed verification.
    """
    import embed_corpus  # noqa: PLC0415
    import fullmarks_config as cfg  # noqa: PLC0415
    import template_matrix as tm  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    with (run / "positives_final.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    finals = {r["class_id"]: r for r in _at(steps, None)}
    by: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by.setdefault(r["class_id"], []).append(r)
    out = [
        "### Where the test positives sit at the final click",
        "",
        "| class | final AP | test positives | beyond the shortlist | inside, failed the gate |",
        "|---|---:|---:|---:|---:|",
    ]
    order = sorted(by, key=lambda c: finals[c]["ap"])
    for cid in order:
        rs = by[cid]
        beyond = sum(r["in_shortlist"] == "False" for r in rs)
        failed = sum(r["in_shortlist"] == "True" and float(r["gate_score"]) < 0.5 for r in rs)
        if beyond or failed:
            out.append(f"| `{cid}` | {finals[cid]['ap']:.2f} | {len(rs)} | {beyond} | {failed} |")
    pages = {p.page_id: p for p in embed_corpus.pages_for_tier(cfg.OUT, "m")}
    (run / "misses").mkdir(exist_ok=True)
    out += ["", "| class | page | rank in pool | box (fraction of page) | thumbnail |", "|---|---|---:|---|---|"]
    for cid in [c for c in order if any(r["in_shortlist"] == "False" for r in by[c])][:weakest]:
        beyond = sorted((r for r in by[cid] if r["in_shortlist"] == "False"), key=lambda r: int(r["rank_in_pool"]))
        for r in beyond[:1] + (beyond[-1:] if len(beyond) > 1 else []):
            page = pages[r["page_id"]]
            with Image.open(page.path) as im:
                img = im.convert("RGB")
            box = tm.largest_box(page, cid)
            w, h = img.size
            size = ""
            if box is not None:
                ImageDraw.Draw(img).rectangle(
                    [box[0] * w, box[1] * h, box[2] * w, box[3] * h], outline=(235, 104, 52), width=max(3, w // 200)
                )
                size = f"{box[2] - box[0]:.2f} x {box[3] - box[1]:.2f}"
            img.thumbnail((420, 420))
            name = f"{cid.replace('/', '__')}-{r['page_id'].replace('/', '__').replace('#', '_')}.png"
            img.save(run / "misses" / name)
            out.append(f"| `{cid}` | `{r['page_id']}` | {r['rank_in_pool']} | {size} | `misses/{name}` |")
    return "\n".join(out) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--no-thumbnails", action="store_true")
    args = ap.parse_args(argv)
    steps = _read(args.run / "steps.csv")
    clicks = _read(args.run / "clicks.csv")
    text = summary(steps) + "\n" + balance(steps)
    figures(steps, args.run / "figures")
    if not args.no_thumbnails:
        rows = thumbnails(clicks, args.run / "images")
        text += (
            "\n### Most helpful and most harmful clicks (credit = that click's change in AP; one observation each)\n\n"
        )
        text += "| | class | page | label | click | credit |\n|---|---|---|---|---:|---:|\n"
        for r in rows:
            text += f"| {r['kind']} | `{r['class_id']}` | `{r['page_id']}` | {r['label']} | {int(r['click'])} | {r['credit']:+.3f} |\n"
        (args.run / "images.json").write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")
    if (args.run / "positives_final.csv").exists() and not args.no_thumbnails:
        text += "\n" + misses(args.run, steps)
    (args.run / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
