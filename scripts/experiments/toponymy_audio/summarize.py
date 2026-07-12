"""Aggregate all eval/run JSONs into one machine-readable summary.

Runs anywhere (no GPU, no vtscore): reads ``RESULTS/*/eval_topo_*.json`` +
``topo_*.json`` + ``texts_*_info.json`` and writes ``RESULTS/summary.json``
with, per run: layer metrics, timings, llm call counts, example topics
(largest / best / worst by name-label sim), and text-variant stats.

Usage::  python summarize.py [--results DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    args = ap.parse_args()

    if args.results:
        results = Path(args.results)
    else:
        import common

        results = common.RESULTS

    summary: dict = {}
    for ds_dir in sorted(p for p in results.iterdir() if p.is_dir()):
        ds: dict = {"texts": {}, "runs": {}}
        for info in sorted(ds_dir.glob("texts_*_info.json")):
            d = json.loads(info.read_text())
            ds["texts"][d["variant"] + d.get("params", {}).get("out_suffix", "")] = {
                "n": d["n"],
                "empty_frac": d["empty_frac"],
                "time_s": list(d["timings_s"].values())[0] if d["timings_s"] else None,
            }
        prep = ds_dir / "prepare_info.json"
        if prep.exists():
            ds["prepare"] = json.loads(prep.read_text())
        for run_path in sorted(ds_dir.glob("topo_*.json")):
            run = json.loads(run_path.read_text())
            entry: dict = {
                "timings_s": run.get("timings_s"),
                "llm_calls": run.get("llm_calls"),
            }
            if "error" in run:
                entry["error"] = run["error"]
                ds["runs"][run_path.stem] = entry
                continue
            ev_path = ds_dir / f"eval_{run_path.stem}.json"
            if ev_path.exists():
                ev = json.loads(ev_path.read_text())
                entry["layers"] = []
                for pl in ev["per_layer"]:
                    row = {k: v for k, v in pl.items() if k != "topics"}
                    topics = [t for t in pl["topics"] if t["sim"] is not None]
                    topics_by_sim = sorted(topics, key=lambda t: t["sim"])
                    row["examples"] = {
                        "largest": sorted(topics, key=lambda t: -t["size"])[:3],
                        "best_named": topics_by_sim[-3:][::-1],
                        "worst_named": topics_by_sim[:3],
                    }
                    entry["layers"].append(row)
            else:
                entry["layers_raw"] = [
                    {
                        "layer": layer["layer"],
                        "n_topics": layer["n_topics"],
                        "noise_frac": layer["noise_frac"],
                        "topic_names": layer["topic_names"],
                    }
                    for layer in run["layers"]
                ]
            ds["runs"][run_path.stem] = entry
        summary[ds_dir.name] = ds

    out = results / "summary.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
