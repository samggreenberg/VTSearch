# 2026-08-12 — #3121 prefetching weights where the loader does not look
**Cost:** ~10 min and 7.6 G of wasted download.

**What broke.** Weights were prefetched with a bare `snapshot_download(...)`,
which writes to `HF_HOME/hub`. The embedders load with
`cache_dir=<VTSEARCH_MODELS_DIR>`, which puts `models--*` at the top of that
dir. The two never met: the jobs saw no cached weights, and three parallel GPU
jobs would each have re-downloaded into the same directory, racing.

**Now prevented by** passing `cache_dir=` explicitly in `prefetch_models.py`.
**Still advice:** a prefetch stage is only worth having if it writes where the
consumer reads — verify by listing the directory the consumer will actually
open, not by trusting that the download reported success.
