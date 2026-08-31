#!/usr/bin/env python
"""Description-enrichment on/off, per eval dataset (issue #3127).

``enrich_descriptions`` is a single **global** setting whose only production
consumer is the Text Sort query embedding (``vtsearch/routes/sorting.py`` ->
``embed_text_query(..., enrich=...)``).  Enrichment replaces the query vector
with the L2-normalised mean of a set of ``description_wrappers`` applied to the
typed text; the *media* vectors are untouched.

Since #3341 the wrapper set is **per-embedder and measured**: this study's own
result is why ``siglip``, ``e5``, ``bge`` and ``clap`` now ship an empty list.
So the harness cannot simply read ``embedder.description_wrappers`` any more -
on those four it would find nothing, emit an ``enriched`` arm identical to
``plain``, and report a flat zero while looking perfectly healthy.  It resolves
through :data:`CANDIDATE_WRAPPERS` instead (the embedder's own list when it has
one, else the pre-#3341 templates for its media type) and overrides the
embedder for the cell, so a re-run reproduces this report and a new checkpoint
can be measured against a wrapper set it does not yet ship.

That is what makes this study cheap and exactly paired: one dataset load serves
every arm, so the two arms differ in the query vector and in nothing else -
same medias, same encoder, same GPU, same process.  Anything that would
otherwise need pairing away (host arithmetic #3160, precision #3143, cache
state) is shared by construction.

One cell = one (embedder, dataset).  Each cell writes a long-form CSV of
per-query metrics for every arm:

    plain      the shipped default: embed the typed text as-is
    enriched   the setting turned on: mean over all wrappers
    w<i>       one arm per individual wrapper (diagnostic, --wrappers)

The ``{text}`` wrapper is an identity, so its ``w<i>`` arm must reproduce
``plain`` exactly.  That is a planted answer, checked by the analyzer: if the
two disagree, the harness - not the setting - is what moved.

Usage::

    python run_enrich.py --exp /expscratch/$USER/enrich-3127 \\
        --datasets esc50_s esc50_m esc50_l [--embedder clap] [--wrappers]
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import socket
import sys
import time
from pathlib import Path

CSV_FIELDS = [
    "dataset",
    "media_type",
    "embedder",
    "n_media",
    "arm",
    "wrapper",
    "category",
    "query",
    "ap",
    "p5",
    "p10",
    "p20",
    "r5",
    "r10",
    "r20",
    "n_relevant",
    "n_pool",
    "arm_seconds",
]

K_VALUES = [5, 10, 20]


def _git(repo: Path, *args: str) -> str:
    """Run a fixed git argv in *repo*; empty string when git cannot answer."""
    import subprocess  # noqa: PLC0415, S404 -- fixed argv, no shell

    try:
        out = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo), *args],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


#: Wrapper templates per media type, as they stood before #3341 emptied the
#: list on ``siglip``, ``e5``, ``bge`` and ``clap``.
#:
#: Production reads the embedder; this study has to be able to measure a
#: *candidate* set the model does not ship - both to reproduce its own report
#: (whose whole point was that these templates lose) and to put a wrapper set
#: in front of a new checkpoint before deciding whether to ship it.  Keeping
#: the deleted templates only in git history would make the report
#: unreproducible and quietly turn the ``w<i>`` arms - and with them the
#: ``{text}`` planted answer - into an empty set.
CANDIDATE_WRAPPERS: dict[str, list[str]] = {
    "audio": [
        "the sound of {text}",
        "a recording of {text}",
        "{text}",
        "audio of {text}",
        "the noise of {text}",
    ],
    "image": [
        "a photo of {text}",
        "a photograph of {text}",
        "an image of {text}",
        "{text}",
        "a picture of {text}",
    ],
    "text": [
        "a document about {text}",
        "an article discussing {text}",
        "{text}",
        "a text passage about {text}",
        "writing on the topic of {text}",
    ],
    "video": [
        "a video of {text}",
        "a media showing {text}",
        "{text}",
        "footage of {text}",
        "a video media of {text}",
    ],
}


def _wrappers_for(embedder) -> tuple[list[str], str]:
    """The wrapper set this cell measures, and where it came from.

    An embedder that ships wrappers is measured on its own (that is the
    production path).  One that ships none - the four #3341 emptied - falls
    back to the pre-#3341 templates for its media type, so the arms stay
    comparable to the rows already in this study's tables.
    """
    own = list(embedder.description_wrappers)
    if own:
        return own, "embedder"
    return list(CANDIDATE_WRAPPERS.get(embedder.media_type_id, [])), "candidate"


@contextlib.contextmanager
def _wrappers_override(embedder, wrappers: list[str]):
    """Make *embedder* report *wrappers* for the duration of the block.

    ``eval_text_sort(enrich=True)`` reaches ``embed_text_enriched``, which
    reads the property off the class - so measuring a candidate set means
    patching the class, not passing a list.  Scoped to one cell; a no-op when
    the embedder already reports exactly this set.
    """
    cls = type(embedder)
    if list(embedder.description_wrappers) == wrappers:
        yield
        return
    original = cls.description_wrappers
    cls.description_wrappers = property(lambda _self: list(wrappers))
    try:
        yield
    finally:
        cls.description_wrappers = original


def _arms(wrappers: list[str], want_wrappers: bool) -> list[tuple[str, str]]:
    """(arm name, wrapper template) pairs.  Empty template = no rewrite."""
    arms: list[tuple[str, str]] = [("plain", ""), ("enriched", "")]
    if want_wrappers:
        for i, wrapper in enumerate(wrappers):
            arms.append((f"w{i}", wrapper))
    return arms


def run_cell(
    ds_id: str,
    embedder_flag: str,
    want_wrappers: bool,
    results_dir: Path,
    force: bool,
) -> str:
    """Run every arm for one (embedder, dataset) cell.  Returns a status word."""
    from vtscore.datasets.loader import load_demo_dataset
    from vtscore.embedding.media_vectors import media_embedder_names
    from vtscore.eval.config import EVAL_DATASETS, EvalQuery
    from vtscore.eval.runner import eval_text_sort
    from vtscore.media import all_demo_datasets, get_embedder

    label = embedder_flag or "default"
    out = results_dir / f"{label}__{ds_id}.csv"
    if out.exists() and out.stat().st_size > 0 and not force:
        print(f"[{ds_id}] SKIP (cell exists: {out.name})", flush=True)
        return "skipped"

    cfg = EVAL_DATASETS[ds_id]
    demo_id = cfg["demo_dataset"]
    media_type = all_demo_datasets()[demo_id].get("media_type", "")

    t0 = time.monotonic()
    medias: dict[int, dict] = {}
    load_demo_dataset(demo_id, medias, embedder_name=embedder_flag)
    load_s = time.monotonic() - t0
    if not medias:
        print(f"[{ds_id}] FAILED: dataset loaded zero medias", file=sys.stderr, flush=True)
        return "failed"

    first = next(iter(medias.values()), {})
    names = media_embedder_names(first)
    loaded = names[0] if names else ""
    if embedder_flag and loaded != embedder_flag:
        # The query has to land in the space the medias were built in; a
        # mismatch here is the #3076 bug that made every non-default arm
        # report near-chance mAP.
        print(
            f"[{ds_id}] FAILED: asked for {embedder_flag!r} but medias carry {loaded!r}",
            file=sys.stderr,
            flush=True,
        )
        return "failed"

    embedder = get_embedder(loaded)
    wrappers, source = _wrappers_for(embedder)
    if not wrappers:
        print(
            f"[{ds_id}] FAILED: no wrappers for {loaded!r} (media type {embedder.media_type_id!r}); "
            "add a CANDIDATE_WRAPPERS entry -- an enrichment study with nothing to "
            "enrich reports a flat zero rather than an error",
            file=sys.stderr,
            flush=True,
        )
        return "failed"
    queries = cfg["queries"]
    print(
        f"[{ds_id}] {len(medias)} medias, {len(queries)} queries, embedder={loaded}, "
        f"wrappers={len(wrappers)} ({source}), load={load_s:.0f}s",
        flush=True,
    )

    rows: list[dict] = []
    for arm, wrapper in _arms(wrappers, want_wrappers):
        armed = [
            EvalQuery(
                text=wrapper.format(text=q.text) if wrapper else q.text,
                target_category=q.target_category,
            )
            for q in queries
        ]
        t_arm = time.monotonic()
        with _wrappers_override(embedder, wrappers):
            metrics = eval_text_sort(
                medias,
                armed,
                media_type,
                K_VALUES,
                enrich=(arm == "enriched"),
                embedder_name=loaded,
            )
        arm_s = time.monotonic() - t_arm
        m_ap = sum(m.average_precision for m in metrics) / len(metrics)
        print(f"    {arm:10s} mAP={m_ap:.4f}  ({arm_s:.1f}s)", flush=True)
        for m, q in zip(metrics, queries):
            rows.append(
                {
                    "dataset": ds_id,
                    "media_type": media_type,
                    "embedder": loaded,
                    "n_media": len(medias),
                    "arm": arm,
                    "wrapper": wrapper,
                    "category": m.target_category,
                    # The *typed* query, identical across arms, so the CSV pairs
                    # on it; the wrapper column says what was actually embedded.
                    "query": q.text,
                    "ap": f"{m.average_precision:.6f}",
                    "p5": f"{m.precision_at_k.get(5, 0.0):.6f}",
                    "p10": f"{m.precision_at_k.get(10, 0.0):.6f}",
                    "p20": f"{m.precision_at_k.get(20, 0.0):.6f}",
                    "r5": f"{m.recall_at_k.get(5, 0.0):.6f}",
                    "r10": f"{m.recall_at_k.get(10, 0.0):.6f}",
                    "r20": f"{m.recall_at_k.get(20, 0.0):.6f}",
                    "n_relevant": m.num_relevant,
                    "n_pool": m.num_total,
                    "arm_seconds": f"{arm_s:.3f}",
                }
            )

    tmp = out.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out)
    print(f"[{ds_id}] wrote {out.name} ({len(rows)} rows)", flush=True)
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help="study root; cells land in <exp>/results")
    ap.add_argument("--datasets", nargs="+", required=True, help="eval dataset ids")
    ap.add_argument(
        "--embedder",
        default="",
        help="embedder override (default: media type's default)",
    )
    ap.add_argument(
        "--wrappers",
        action="store_true",
        help="also run one arm per individual wrapper",
    )
    ap.add_argument("--force", action="store_true", help="rebuild cells that already exist")
    args = ap.parse_args()

    from vtscore.embedding import initialize_models

    initialize_models()

    exp = Path(args.exp)
    results = exp / "results"
    results.mkdir(parents=True, exist_ok=True)

    repo = Path(os.environ.get("VTS_REPO") or Path(__file__).resolve().parents[3])
    provenance = {
        "commit": _git(repo, "rev-parse", "HEAD"),
        # A dirty worktree means the cells do not correspond to any commit; it
        # is recorded per cell rather than blocked, because a mid-run diagnostic
        # is legitimate and a silently unattributable cell is not.
        "dirty": bool(_git(repo, "status", "--porcelain")),
        "repo": str(repo),
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
        "gpu": os.environ.get("SLURM_JOB_GRES", ""),
        "data_dir": os.environ.get("VTSEARCH_DATA_DIR", ""),
        "embedder_flag": args.embedder,
        "datasets": args.datasets,
        "wrappers": args.wrappers,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(json.dumps(provenance, indent=2), flush=True)

    status: dict[str, str] = {}
    for ds_id in args.datasets:
        try:
            status[ds_id] = run_cell(ds_id, args.embedder, args.wrappers, results, args.force)
        except Exception as exc:  # one bad dataset must not sink the chunk
            import traceback

            traceback.print_exc()
            print(f"[{ds_id}] FAILED: {exc}", file=sys.stderr, flush=True)
            status[ds_id] = "failed"

    label = args.embedder or "default"
    prov_path = exp / "logs" / f"provenance-{label}-{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    provenance["status"] = status
    provenance["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    prov_path.write_text(json.dumps(provenance, indent=2))

    failed = [d for d, s in status.items() if s == "failed"]
    print("\n=== chunk summary ===", flush=True)
    for ds_id, s in status.items():
        print(f"  {ds_id:22s} {s}", flush=True)
    if failed:
        print(f"FAILED cells: {', '.join(failed)}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
