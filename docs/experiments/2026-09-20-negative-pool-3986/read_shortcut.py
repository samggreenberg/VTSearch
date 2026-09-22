"""Print the trained-head summary from a saved measurement JSON.

The run that produced these wrote its JSON but lost its stdout, so the summary is
recomputed here rather than re-run. Reads the same fields the script prints.
"""

import json
import statistics
import sys


def summarise(path: str) -> None:
    d = json.load(open(path))
    rows = d["cells"]
    print(
        "=== %s  (embedder=%s, train-pool=%s, %d cells)"
        % (path.split("/")[-1], d["embedder"], d.get("train_pool", "?"), len(rows))
    )
    print("  n_neg per cell (median):        {:,}".format(int(statistics.median(r["n_neg"] for r in rows))))
    print(
        "  co-occurrence: baseline %.0f%%, representative %.0f%%"
        % (
            100 * statistics.mean(r.get("cooccur_rate_baseline", 0.0) for r in rows),
            100 * statistics.mean(r["cooccur_rate"] for r in rows),
        )
    )
    print("  FPR ratio (representative / baseline):  %.2f +- %.2f" % (d["ratio_perclass_mean"], d["ratio_perclass_se"]))
    print("  FPR ratio (co-occurring  / baseline):   %.2f +- %.2f" % (d["ratio_cooccur_mean"], d["ratio_cooccur_se"]))
    print("  paired dAUC:                            %+.3f" % d["d_auc_mean"])
    print("  paired dAP:                             %+.3f" % d["d_ap_mean"])
    print(
        "  mean AP: %.3f -> %.3f"
        % (statistics.mean(r["ap_shared"] for r in rows), statistics.mean(r["ap_perclass"] for r in rows))
    )
    print("  by band (co-occurring ratio / representative ratio):")
    for b, v in d.get("by_band", {}).items():
        print("    @%-8s %5.2f  %5.2f" % (b, v["ratio_cooccur"], v["ratio_perclass"]))
    worst = sorted(rows, key=lambda r: -r["ratio_cooccur"])[:4]
    print(
        "  worst cells by co-occurring ratio: " + ", ".join("%s %.2f" % (r["cell"], r["ratio_cooccur"]) for r in worst)
    )


for p in sys.argv[1:]:
    summarise(p)
    print()
