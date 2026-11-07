"""#5373: read size_*.csv -> per-face keep-highest-quality outcome, by retry cut."""

import glob
import sys
import numpy as np
import pandas as pd

pat = sys.argv[1]
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pat))], ignore_index=True)
df = df[df.n_own > 0]
m = df.dropna(subset=["q"])
base = m[m.rot == 0].set_index("idx")
best = m.loc[m.groupby("idx").q.idxmax()].set_index("idx")
f = (
    base[["q_pile", "cos_own", "ap"]]
    .rename(columns={"cos_own": "cos0", "ap": "ap0"})
    .join(best[["rot", "q", "cos_own", "ap"]].rename(columns={"q": "qb", "cos_own": "cosb", "ap": "apb"}))
)
f["sw"] = f.rot != 0
f["dcos"] = f.cosb - f.cos0
f["dap"] = f.apb - f.ap0
print(f"faces {len(f)}  (0-deg reproduces pile: max |q-q_pile| = {(base.q - base.q_pile).abs().max():.3f})")
print(f"pile median quality {f.q_pile.median():.1f}")
rows = []
for cut in [12, 13, 14, 15, 16, 17, 18, 20, 22, 25, 999]:
    s = f[f.q_pile < cut]
    w = s[s.sw]
    rows.append(
        dict(
            cut=cut,
            retried=len(s),
            pct=100 * len(s) / len(f),
            switched=len(w),
            better_cos=int((w.dcos > 0).sum()),
            worse_cos=int((w.dcos < 0).sum()),
            better_ap=int((w.dap > 0.01).sum()),
            worse_ap=int((w.dap < -0.01).sum()),
            mean_dap_sw=w.dap.mean(),
            sum_dap=w.dap.sum(),
            mean_dcos_sw=w.dcos.mean(),
        )
    )
print(pd.DataFrame(rows).round(3).to_string(index=False))
# does a gain in quality predict a gain in match? every matched non-zero rotation of every face
nz = m[m.rot != 0].join(
    base[["q", "cos_own", "ap"]].rename(columns={"q": "q0r", "cos_own": "c0", "ap": "a0"}), on="idx"
)
nz["dq"] = nz.q - nz.q0r
nz["dc"] = nz.cos_own - nz.c0
print(f"\nrotations matched: {len(nz)}; corr(dq, dcos) = {np.corrcoef(nz.dq, nz.dc)[0, 1]:.2f}")
for lo, hi in [(-99, 0), (0, 2), (2, 5), (5, 99)]:
    b = nz[(nz.dq > lo) & (nz.dq <= hi)]
    print(f"  dq in ({lo},{hi}]: n={len(b)}  dcos>0 {100 * (b.dc > 0).mean():.0f}%  mean dcos {b.dc.mean():+.3f}")
w = f[f.sw]
print("\nswitched faces by start quality:")
print(
    w.assign(b=pd.cut(w.q_pile, [0, 14, 16, 18, 22, 99]))
    .groupby("b", observed=True)
    .agg(
        n=("dap", "size"),
        dq=("qb", lambda x: (x - w.loc[x.index, "q_pile"]).mean()),
        dcos=("dcos", "mean"),
        dap=("dap", "mean"),
        worse=("dcos", lambda x: int((x < 0).sum())),
    )
    .round(3)
    .to_string()
)
f.to_csv(pat.replace("*", "faces").replace("size_", "faces_"))
