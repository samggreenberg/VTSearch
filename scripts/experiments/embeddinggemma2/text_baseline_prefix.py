"""#4837: text_baseline.py with EmbeddingGemma 2's query prefix replaced by $EG2_PREFIX (default: none).

Run from scripts/experiments/calibration with the same env and args as text_baseline.py, e.g.
``EG2_PREFIX="" python ../embeddinggemma2/text_baseline_prefix.py --results <grid>/results --out <csv>``;
only the EmbeddingGemma 2 rows differ from a plain run.
"""

import os
import runpy
import sys

sys.path.insert(0, os.getcwd())
import vtscore.media._embeddinggemma2 as eg2  # noqa: E402

eg2.EMBEDDINGGEMMA2_QUERY_PREFIX = os.environ.get("EG2_PREFIX", "")
print(f"EmbeddingGemma 2 query prefix: {eg2.EMBEDDINGGEMMA2_QUERY_PREFIX!r}", flush=True)
sys.argv = ["text_baseline.py", *sys.argv[1:]]
runpy.run_path("text_baseline.py", run_name="__main__")
