"""#4110: the cell's live text-query embedding (SigLIP's text tower on the CPU), per node and pin."""

import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.environ["VTS_REPO"], "scripts/experiments/calibration"))
import common

common.setup_env()
import numpy as np
import experiment_config as cfg
import torch
from vtscore.embedding.helpers import embed_text_query

variant = sys.argv[1]
qs = ["a bus", "a cat", "a refrigerator", "a sports ball", "a person riding a horse"]
r = {}
t0 = time.perf_counter()
for q in qs:
    v = np.asarray(embed_text_query(q, "image", enrich=cfg.SEED_ENRICH, embedder_name="siglip"), dtype=np.float32)
    r[q] = hashlib.sha1(v.tobytes()).hexdigest()[:12]
first = time.perf_counter() - t0
t0 = time.perf_counter()
for q in qs:
    embed_text_query(q + " again", "image", enrich=cfg.SEED_ENRICH, embedder_name="siglip")
warm = (time.perf_counter() - t0) / len(qs)
out = {
    "node": os.uname().nodename.split(".")[0],
    "variant": variant,
    "cap": torch.backends.cpu.get_cpu_capability(),
    "first_s": round(first, 2),
    "warm_query_ms": round(warm * 1000, 1),
    "r": r,
    "enrich": str(cfg.SEED_ENRICH),
}
json.dump(out, open(f"{os.environ['PROBE_OUT']}/{out['node']}__{variant}.json", "w"), indent=1)
print(json.dumps(out))
