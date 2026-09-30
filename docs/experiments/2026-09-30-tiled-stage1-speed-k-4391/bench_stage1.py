import time
import numpy as np
import torch

rng = np.random.default_rng(0)
pages = 47000
counts = rng.integers(40, 75, pages)
T = int(counts.sum())
tiles = rng.standard_normal((T, 512), dtype=np.float32).astype(np.float16)
starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
for Q in (1, 10, 60):
    q = rng.standard_normal((Q, 512), dtype=np.float32)
    # numpy path as shipped
    t = time.perf_counter()
    best = np.empty(T, np.float32)
    for lo in range(0, T, 262144):
        best[lo : lo + 262144] = (tiles[lo : lo + 262144].astype(np.float32) @ q.T).max(1)
    s1 = np.maximum.reduceat(best, starts)
    tn = time.perf_counter() - t
    # torch GPU
    dev = "cuda"
    M = torch.from_numpy(tiles).to(dev)
    seg = torch.from_numpy(np.repeat(np.arange(pages), counts)).to(dev)
    torch.cuda.synchronize()
    t = time.perf_counter()
    qq = torch.from_numpy(q).to(dev).half()
    b = (M @ qq.T).max(1).values.float()
    out = torch.full((pages,), -1e9, device=dev).scatter_reduce(0, seg, b, "amax", include_self=True)
    s2 = out.cpu().numpy()
    torch.cuda.synchronize()
    tg = time.perf_counter() - t
    print(f"T={T} Q={Q}: numpy {tn:.2f}s  torch-gpu {tg * 1000:.0f}ms  maxdiff {np.abs(s1 - s2).max():.3f}")
t = time.perf_counter()
order = np.argsort(-s1, kind="stable")
out = [{"id": i, "score": round(float(s1[i]), 4)} for i in order]
print("sort+dicts", round(time.perf_counter() - t, 2))
