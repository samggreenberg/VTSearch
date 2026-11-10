"""#5373: read size_*.csv -> switches, worse matches and summed AP gain per (retry cut, minimum quality gain)."""

import glob
import sys
import pandas as pd

for pat in sys.argv[1:]:
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pat))], ignore_index=True)
    m = df[df.n_own > 0].dropna(subset=["q"])
    base = m[m.rot == 0].set_index("idx")
    print(pat)
    rows = []
    for cut in (14, 16, 18, 20):
        for mg in (0, 1, 2, 3, 5):
            nz = m[(m.rot != 0)].copy()
            nz["q0"] = nz.idx.map(base.q)
            nz = nz[(nz.q0 < cut) & (nz.q > nz.q0 + mg)]
            b = nz.loc[nz.groupby("idx").q.idxmax()].set_index("idx")
            d = b.ap - base.loc[b.index, "ap"]
            dc = b.cos_own - base.loc[b.index, "cos_own"]
            rows.append(
                dict(
                    cut=cut,
                    margin=mg,
                    retried=int((base.q < cut).sum()),
                    switched=len(b),
                    worse_cos=int((dc < 0).sum()),
                    worse_ap=int((d < -0.01).sum()),
                    sum_dap=round(d.sum(), 1),
                    lost_ap=round(d[d < 0].sum(), 2),
                )
            )
    print(pd.DataFrame(rows).to_string(index=False))
