"""Recover the 100%-vs-0% co-occurrence contrast, to compare with #3667's 1.88.

The shipped baseline is a MIXTURE: a fraction w of it co-occurs with another
class in C (#3667's negatives) and 1-w is barren. So the baseline FPR is

    fpr_base = w * fpr_cooccur + (1 - w) * fpr_barren

and the barren-subset FPR follows exactly, with no extra run. That matters
because #3667's 1.88 is co-occurring against a 0%-co-occurring pool, and the
shipped ratio here is against a 40% one -- the two are not the same contrast,
and reading them side by side without this step understates what remains.
"""

import json
import statistics
import sys

d = json.load(open(sys.argv[1]))
rows = d["cells"]


def contrast(r: dict) -> float | None:
    """co-occurring / barren FPR, whatever the baseline's mixture.

    When w is 0 the baseline already IS the barren subset, so the ratio the
    script printed is the contrast; skipping those rows made this unable to read
    the `--train-pool barren` arm at all.
    """
    w = r["cooccur_rate_baseline"]
    if w <= 0:
        return r["ratio_cooccur"]
    if w >= 1:
        return None
    fpr_barren = (r["fpr_shared"] - w * r["fpr_cooccur"]) / (1 - w)
    return r["fpr_cooccur"] / fpr_barren if fpr_barren > 0 else None


recovered, mixed = [], []
for r in rows:
    if (v := contrast(r)) is not None:
        recovered.append(v)
        mixed.append(r["ratio_cooccur"])


def pm(a: list[float]) -> str:
    return "%.2f +- %.2f" % (statistics.mean(a), statistics.stdev(a) / len(a) ** 0.5)


print("train-pool=%s, %d cells usable" % (d.get("train_pool", "?"), len(recovered)))
print(
    "  baseline co-occurrence w:                      %.0f%%"
    % (100 * statistics.mean(r["cooccur_rate_baseline"] for r in rows))
)
print("  FPR ratio co-occurring / MIXED baseline:       %s   (as printed)" % pm(mixed))
print("  FPR ratio co-occurring / BARREN subset:        %s   <- the #3667 contrast (its 1.88)" % pm(recovered))
for b in ("small", "medium", "large"):
    sub = [v for r in rows if r["cell"].endswith("@" + b) if (v := contrast(r)) is not None]
    if sub:
        print("    @%-7s %s" % (b, pm(sub)))
