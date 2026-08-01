"""What is the model actually doing at the first Bad vote vs later? (#2790 mechanism check)"""

import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
traces = sorted(root.rglob("trace.json"))

# 1) example: full step table for a few traces
print("===== EXAMPLE TRACES (t | phase | select | calib_mode | head | n_good | n_bad | threshold | cost | Δcost) =====")
for tj in traces[:3]:
    t = sorted(json.loads(tj.read_text()), key=lambda e: e["t"])
    print(f"\n--- {tj.parent.parent.name} / {tj.parent.name} ---")
    prev = None
    for e in t[:10]:
        dc = "" if prev is None else f"{e.get('cost', 0) - prev.get('cost', 0):+.3f}"
        mark = "  <-- FIRST BAD" if (prev and prev.get("n_bad") == 0 and e.get("gt_label") == "bad") else ""
        print(
            f"  t{e.get('t'):>2} {str(e.get('phase')):<5} {str(e.get('select_mode')):<5} "
            f"{str(e.get('calib_mode')):<16} {str(e.get('head')):<8} g{e.get('n_good')} b{e.get('n_bad')} "
            f"thr={e.get('threshold')} cost={e.get('cost')} d={dc}{mark}"
        )

# 2) aggregate mechanism stats
cost_flat_before_first_bad = 0
n_traces_with_bad = 0
first_bad_calib = Counter()
first_bad_head = Counter()
first_bad_costjump = []  # Δcost at the first bad
head_by_nbad0 = Counter()  # head during all-good phase
recurring_spike_calib = Counter()  # calib_mode at spikes that are NOT the first bad
recurring_spike_head = Counter()
n_first_bad_spikes = 0
n_recurring_spikes = 0

for tj in traces:
    t = sorted(json.loads(tj.read_text()), key=lambda e: e["t"])
    # cost during n_bad==0 steps
    good_costs = [e.get("cost") for e in t if (e.get("n_bad") or 0) == 0 and e.get("cost") is not None]
    if len(good_costs) >= 2 and max(good_costs) - min(good_costs) < 1e-6:
        cost_flat_before_first_bad += 1
    for e in t:
        if (e.get("n_bad") or 0) == 0:
            head_by_nbad0[str(e.get("head"))] += 1
    for i in range(1, len(t)):
        prev, cur = t[i - 1], t[i]
        if cur.get("gt_label") != "bad":
            continue
        dcost = (cur.get("cost") or 0) - (prev.get("cost") or 0)
        is_first_bad = (prev.get("n_bad") or 0) == 0
        is_spike = dcost > 0.1
        if is_first_bad:
            n_traces_with_bad += 1
            first_bad_calib[str(cur.get("calib_mode"))] += 1
            first_bad_head[str(cur.get("head"))] += 1
            first_bad_costjump.append(dcost)
            if is_spike:
                n_first_bad_spikes += 1
        elif is_spike:
            n_recurring_spikes += 1
            recurring_spike_calib[str(cur.get("calib_mode"))] += 1
            recurring_spike_head[str(cur.get("head"))] += 1

print("\n\n===== MECHANISM AGGREGATE =====")
print(f"traces: {len(traces)}")
print(f"cost is FLAT across all n_bad==0 steps in: {cost_flat_before_first_bad}/{len(traces)} traces")
print(f"head during all-good (n_bad==0) steps: {dict(head_by_nbad0.most_common())}")
print(f"\nAt the FIRST bad vote (n={n_traces_with_bad}):")
print(f"  calib_mode: {dict(first_bad_calib.most_common())}")
print(f"  head:       {dict(first_bad_head.most_common())}")
if first_bad_costjump:
    js = sorted(first_bad_costjump)
    print(
        f"  Δcost at first bad: min={js[0]:+.3f} median={js[len(js) // 2]:+.3f} max={js[-1]:+.3f} "
        f"mean={sum(js) / len(js):+.3f}; fraction Δcost>0 (cost got WORSE): {sum(1 for x in js if x > 0) / len(js):.2f}"
    )
print(f"\nSpikes (Δcost>0.1): first-bad={n_first_bad_spikes}, recurring(n_bad>=1)={n_recurring_spikes}")
print(f"  recurring-spike calib_mode: {dict(recurring_spike_calib.most_common())}")
print(f"  recurring-spike head:       {dict(recurring_spike_head.most_common())}")
